from collections.abc import Iterator
from copy import deepcopy
from dataclasses import fields
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch

from swarmbots.learn.algos.ppo.ppo import PPO
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisodeSegment
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamples, PPOSamplerConfig
from swarmbots.learn.algos.sac.sac import SAC
from swarmbots.learn.metrics_list import MetricsLists, materialize_scalar_metrics
from swarmbots.learn.optimizers import make_adam
from swarmbots.learn.rollout_utils import append_episode_infos
from swarmbots.learn.summary_statistics import (
    NO_DATA, TensorSummaryStatistics, compute_summary_statistics, compute_tensor_summary_statistics,
)
from tests.test_ppo_masking import _make_samples, _ppo_with_reduction
from tests.test_ppo_policy_dropout import _make_env, _make_policy
from tests.test_sac_algorithm import _make_env as _make_sac_env, _make_policy as _make_sac_policy


CUDA = pytest.param("cuda", marks=(
    pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
))


@pytest.fixture(scope="module", autouse=True)
def single_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("device", ["cpu", CUDA])
def test_episode_statistics_preserve_types_vectors_masks_and_lane_order(device: str) -> None:
    infos = {
        "episode": {
            "r": torch.tensor([1.5, 2.5, 3.5], dtype=torch.float64, device=device),
            "l": torch.tensor([3, 7, 11], device=device),
            "success": torch.tensor([True, False, False], device=device),
            "_success": torch.tensor([True, True, False], device=device),
            "vector": torch.arange(6, device=device).reshape(3, 2),
            "_private": object(),
        },
        "_episode": torch.tensor([True, False, True], device=device),
        "scenario_id": torch.tensor([4, 5, 6], device=device),
        "scenario_name": np.array(["wall", "payload", "bridge"], dtype=object),
    }
    entries = []
    append_episode_infos(episode_infos=entries, infos=infos, dones=infos["_episode"])
    assert len(entries) == 2
    for entry, index in zip(entries, (0, 2), strict=True):
        assert entry["r"] == 1.5 + index and isinstance(entry["r"], float)
        assert entry["l"] == [3, 7, 11][index] and isinstance(entry["l"], int)
        np.testing.assert_array_equal(entry.pop("vector"), np.arange(6).reshape(3, 2)[index])
        assert entry["scenario_id"] == 4 + index and isinstance(entry["scenario_id"], int)
        assert entry["scenario_name"] == ["wall", "payload", "bridge"][index]
    assert entries[0]["success"] is True
    assert "success" not in entries[1]


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_episode_statistics_transfer_count_does_not_grow_with_completed_episodes() -> None:
    transfer_counts = []
    original_cpu, original_item = torch.Tensor.cpu, torch.Tensor.item

    def reject_device_scalar(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        assert tensor.device.type == "cpu", "episode fields must transfer in batches"
        return original_item(tensor, *args, **kwargs)

    for count in (4, 64):
        transfers = []

        def record_transfer(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            transfers.append(tensor.numel())
            return original_cpu(tensor, *args, **kwargs)

        dones = torch.ones(count, dtype=torch.bool, device="cuda")
        infos = {"episode": {
            "r": torch.arange(count, dtype=torch.float64, device="cuda"),
            "progress_reward": torch.ones(count, dtype=torch.float64, device="cuda"),
            "l": torch.arange(count, device="cuda"),
            "success": torch.ones(count, dtype=torch.bool, device="cuda"),
        }}
        entries = []
        with patch.object(torch.Tensor, "cpu", new=record_transfer), patch.object(
            torch.Tensor, "item", new=reject_device_scalar,
        ):
            append_episode_infos(episode_infos=entries, infos=infos, dones=dones)
        assert len(entries) == count
        transfer_counts.append(len(transfers))
    assert transfer_counts[0] == transfer_counts[1] <= 4


@pytest.mark.parametrize("device", ["cpu", CUDA])
@pytest.mark.parametrize("mask_values", [None, [True, False, True], [False, False, False], [False, True, False]])
def test_tensor_statistics_match_selected_values_without_materializing_scalars(
    device: str, mask_values: list[bool] | None,
) -> None:
    values = torch.tensor([1., 4., 9.], device=device)
    mask = None if mask_values is None else torch.tensor(mask_values, device=device)
    with (
        patch.object(torch.Tensor, "item", side_effect=AssertionError("statistics must stay on device")),
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("statistics must stay on device")),
    ):
        deferred = compute_tensor_summary_statistics(values, mask)
    actual = materialize_scalar_metrics({"values": deferred})["values"]
    expected = compute_summary_statistics(values if mask is None else values[mask], find_min=True, find_max=True)
    assert actual.n == expected.n
    for name in ("mean", "std", "min_value", "max_value"):
        if getattr(expected, name) is NO_DATA:
            assert getattr(actual, name) is NO_DATA
        else:
            assert getattr(actual, name) == pytest.approx(getattr(expected, name))


@pytest.mark.parametrize("device", ["cpu", CUDA])
@pytest.mark.parametrize("reduction", [None, "sum", "mean"])
def test_deferred_ppo_metrics_preserve_masked_statistics_losses_and_gradients(device: str, reduction: str | None) -> None:
    batch = _make_samples()
    for field in fields(PPOSamples):
        value = getattr(batch, field.name)
        if isinstance(value, torch.Tensor):
            setattr(batch, field.name, value.to(device))
    batch.time_mask = torch.tensor([[True, False, True], [False, True, True]], device=device)
    ppo = _ppo_with_reduction(reduction)
    ppo.value_loss_fn.to(device)
    ppo.normalize_advantage = True
    ppo.clip_range = .2
    ppo.clip_range_vf = None
    ppo.use_popart = False
    ppo.vf_coef = .5
    ppo.mc_ent_coef = .01
    log_probs = torch.linspace(-.2, .3, batch.log_probs.numel(), device=device).reshape_as(batch.log_probs)
    log_probs.requires_grad_()
    values = torch.randn_like(batch.values, requires_grad=True)
    expected_loss, expected_kl, expected_metrics = ppo.compute_ppo_loss(batch, log_probs, values)
    with (
        patch.object(torch.Tensor, "item", side_effect=AssertionError("diagnostics must stay on device")),
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("diagnostics must stay on device")),
    ):
        loss, kl, metrics = ppo.compute_ppo_loss(batch, log_probs, values, materialize_metrics=False)
        collected = MetricsLists()
        collected.add(metrics)
    torch.testing.assert_close(loss, expected_loss)
    torch.testing.assert_close(torch.autograd.grad(loss, (log_probs, values)),
                               torch.autograd.grad(expected_loss, (log_probs, values)))
    assert kl.item() == expected_kl
    for value in metrics.values():
        tensor = value.moments if isinstance(value, TensorSummaryStatistics) else value
        assert not tensor.requires_grad and tensor.grad_fn is None
    metrics["ratio"].moments.zero_()
    actual = collected.compute_summary_statistics()
    for name, expected in expected_metrics.items():
        if name == "ratio":
            assert actual[name].n == expected.n
            assert actual[name].mean == pytest.approx(expected.mean)
            assert actual[name].std == pytest.approx(expected.std)
            assert actual[name].min_value == pytest.approx(expected.min_value)
            assert actual[name].max_value == pytest.approx(expected.max_value)
        else:
            assert actual[name].n == 1
            assert actual[name].mean == pytest.approx(expected)


@pytest.mark.parametrize("device", ["cpu", CUDA])
def test_metrics_aggregation_keeps_sample_weighted_ratio_statistics(device: str) -> None:
    collected = MetricsLists()
    samples = [torch.tensor([1., 2., 3.], device=device), torch.tensor([5.], device=device)]
    for values in samples:
        collected.add({"ratio": compute_tensor_summary_statistics(values)})
    actual = collected.compute_summary_statistics()["ratio"]
    expected = compute_summary_statistics(torch.cat(samples), find_min=True, find_max=True)
    assert actual.n == 4 and actual.mean == expected.mean
    assert actual.std == pytest.approx(expected.std)
    assert actual.min_value == expected.min_value and actual.max_value == expected.max_value


@pytest.mark.parametrize("dtype,count", [(torch.bfloat16, 257), (torch.float16, 2049)])
def test_tensor_statistics_keep_exact_counts_with_low_precision_values(dtype: torch.dtype, count: int) -> None:
    values = torch.ones(count, dtype=dtype)
    actual = materialize_scalar_metrics({"ratio": compute_tensor_summary_statistics(values)})["ratio"]
    assert actual.n == count
    assert actual.mean == 1. and actual.std == 0.


def test_tensor_statistics_preserve_small_float16_ratio_variation() -> None:
    values = torch.ones(1024, dtype=torch.float16)
    values[-1] = torch.nextafter(values[-1], values.new_tensor(2.))
    expected = compute_summary_statistics(values, find_min=True, find_max=True)
    actual = materialize_scalar_metrics({"ratio": compute_tensor_summary_statistics(values)})["ratio"]
    assert expected.std > 0.
    assert actual.std == expected.std
    assert actual.mean == expected.mean
    assert actual.min_value == expected.min_value and actual.max_value == expected.max_value


@pytest.mark.parametrize("device", ["cpu", CUDA])
def test_metric_lists_preserve_precision_across_different_metric_dtypes(device: str) -> None:
    integer_value = 2**24 + 1
    collected = MetricsLists()
    collected.add({
        "integer": torch.tensor(integer_value, dtype=torch.int64, device=device),
        "ordinary": torch.tensor(1.5, dtype=torch.float32, device=device),
    })
    actual = collected.compute_summary_statistics()
    assert actual["integer"].mean == integer_value
    assert actual["ordinary"].mean == 1.5


@pytest.mark.parametrize("device", ["cpu", CUDA])
def test_metric_lists_keep_vector_samples_and_snapshot_their_storage(device: str) -> None:
    collected = MetricsLists()
    samples = torch.tensor([1., 2., 3.], device=device, requires_grad=True)
    collected.add({"samples": samples})
    with torch.no_grad():
        samples.fill_(10.)
    actual = collected.compute_summary_statistics(find_min=True, find_max=True)["samples"]
    assert actual.n == 3 and actual.mean == 2.
    assert actual.std == pytest.approx(np.std([1., 2., 3.]))
    assert actual.min_value == 1. and actual.max_value == 3.


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_metrics_aggregation_supports_metrics_on_different_devices() -> None:
    collected = MetricsLists()
    for value in (1., 3.):
        collected.add({"cpu": torch.tensor(value), "cuda": torch.tensor(value, device="cuda")})
    actual = collected.compute_summary_statistics()
    assert actual["cpu"].mean == actual["cuda"].mean == 2.
    assert actual["cpu"].std == actual["cuda"].std == 1.


@pytest.mark.parametrize("device", ["cpu", CUDA])
def test_adam_matches_reference_across_learning_rate_changes(device: str) -> None:
    parameter = torch.nn.Parameter(torch.tensor([1., -2.], device=device))
    reference = torch.nn.Parameter(parameter.detach().clone())
    optimizer = make_adam([parameter], lr=.01, device=torch.device(device))
    reference_optimizer = torch.optim.Adam([reference], lr=.01, fused=False)
    for lr in (.01, .002, 0.):
        optimizer.param_groups[0]["lr"] = reference_optimizer.param_groups[0]["lr"] = lr
        for value, current_optimizer in ((parameter, optimizer), (reference, reference_optimizer)):
            current_optimizer.zero_grad()
            (value - .5).square().sum().backward()
            current_optimizer.step()
        torch.testing.assert_close(parameter, reference)
        for name in optimizer.state[parameter]:
            torch.testing.assert_close(optimizer.state[parameter][name].cpu(),
                                       reference_optimizer.state[reference][name].cpu())


@pytest.mark.parametrize("device", ["cpu", CUDA])
@pytest.mark.parametrize("early_stop", [False, True])
def test_ppo_training_preserves_updates_and_kl_early_stopping(device: str, early_stop: bool) -> None:
    env = _make_env()
    algorithm = PPO(_make_policy(env), env, train_device=device, n_epochs=2,
                    sampler_config=PPOSamplerConfig(batch_size=2), target_kl=.001 if early_stop else None)
    algorithm.n_total_iterations = 1
    algorithm.max_grad_norm = .01
    algorithm.policy.to(algorithm.train_device).eval()
    observations = dict(
        local_obs=torch.randn(4, 2, 3, device=device), global_obs=torch.randn(4, 1, device=device),
        hidden_local_vars=torch.empty(4, 2, 0, device=device),
        hidden_global_vars=torch.empty(4, 0, device=device), agent_mask=None,
    )
    with torch.no_grad():
        actions, log_probs, values = algorithm.policy(**observations)
    segment = PPOEpisodeSegment(
        **observations, actions=actions, rewards=torch.zeros(4, device=device),
        log_probs=log_probs - .5 if early_stop else log_probs, values=values,
        final_local_obs=None, final_global_obs=None, final_hidden_local_vars=None,
        final_hidden_global_vars=None, final_agent_mask=None, final_value=None,
        returns=values + torch.arange(4, device=device), advantages=torch.arange(4, device=device).float(),
    )
    initial_parameters = [value.detach().clone() for value in algorithm.policy.parameters()]
    expected_norms = []
    clip_grad_norm = torch.nn.utils.clip_grad_norm_

    def record_gradient_norm(*args: Any, **kwargs: Any) -> torch.Tensor:
        norm = clip_grad_norm(*args, **kwargs)
        expected_norms.append(float(norm.detach().cpu()))
        return norm

    with patch.object(torch.nn.utils, "clip_grad_norm_", side_effect=record_gradient_norm):
        metrics = algorithm.train([segment])
    assert metrics["updates"] == algorithm.n_total_updates == (0 if early_stop else 4)
    assert metrics["grad_norm"].n == metrics["updates"]
    assert metrics["ratio"].n == (4 if early_stop else 16)
    assert 0 <= metrics["grad_clip_frac"] <= 1
    if expected_norms:
        assert metrics["grad_norm"].mean == pytest.approx(np.mean(expected_norms))
        assert metrics["grad_norm"].std == pytest.approx(np.std(expected_norms))
        assert metrics["grad_norm"].min_value == min(expected_norms)
        assert metrics["grad_norm"].max_value == max(expected_norms)
        expected_clip_fraction = np.mean(np.asarray(expected_norms) > algorithm.max_grad_norm)
        assert expected_clip_fraction > 0
        assert metrics["grad_clip_frac"] == expected_clip_fraction
    changed = any(not torch.equal(before, after) for before, after in zip(
        initial_parameters, algorithm.policy.parameters(), strict=True,
    ))
    assert changed is not early_stop


@pytest.mark.parametrize("source_device,target_device", [
    ("cpu", "cpu"),
    pytest.param("cpu", "cuda", marks=(pytest.mark.cuda, pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires CUDA"))),
    pytest.param("cuda", "cpu", marks=(pytest.mark.cuda, pytest.mark.skipif(
        not torch.cuda.is_available(), reason="requires CUDA"))),
])
def test_ppo_optimizer_checkpoint_continues_on_the_selected_device(source_device: str, target_device: str) -> None:
    env = _make_env()
    source = PPO(_make_policy(env), env, train_device=source_device)
    target = PPO(_make_policy(env), env, train_device=target_device)
    source.policy.to(source.train_device)
    target.policy.to(target.train_device)

    def step(algorithm: PPO) -> None:
        algorithm.optimizer.zero_grad()
        sum(parameter.square().mean() for parameter in algorithm.policy.parameters()).backward()
        algorithm.optimizer.step()

    step(source)
    target.policy.load_state_dict(source.policy.state_dict())
    target._apply_optimizer_state_dict(deepcopy(source._get_optimizer_state_dict()), [], [])
    assert all(group["fused"] == (target_device == "cuda") for group in target.optimizer.param_groups)
    step(source)
    step(target)
    for source_parameter, target_parameter in zip(source.policy.parameters(), target.policy.parameters(), strict=True):
        torch.testing.assert_close(source_parameter.cpu(), target_parameter.cpu())


def test_cpu_resume_clears_cuda_optimizer_capture_settings() -> None:
    env = _make_env()
    source = PPO(_make_policy(env), env, train_device="cpu")
    target = PPO(_make_policy(env), env, train_device="cpu")
    for parameter in source.policy.parameters():
        parameter.grad = torch.ones_like(parameter)
    source.optimizer.step()
    target.policy.load_state_dict(source.policy.state_dict())
    state = deepcopy(source._get_optimizer_state_dict())
    for group in state["param_groups"]:
        group.update(capturable=True, fused=True)
    target._apply_optimizer_state_dict(state, [], [])
    for group in target.optimizer.param_groups:
        assert not group["capturable"] and not group["fused"]
        for parameter in group["params"]:
            parameter.grad = torch.ones_like(parameter)
    target.optimizer.step()


@pytest.mark.parametrize("device", ["cpu", CUDA])
def test_sac_selects_adam_for_actor_critic_and_temperature(device: str) -> None:
    env = _make_sac_env()
    try:
        def make_algorithm() -> SAC:
            return SAC(_make_sac_policy(env), env, train_device=device, buffer_capacity_per_env=16,
                       batch_size=2, learning_starts=0, sac_compile_tensor_operations=False,
                       sac_compile_optimizer_steps=False)

        def optimizers(algorithm: SAC) -> tuple[torch.optim.Adam, ...]:
            assert algorithm.ent_coef_optimizer is not None
            return algorithm.actor_optimizer, algorithm.critic_optimizer, algorithm.ent_coef_optimizer

        def step(algorithm: SAC) -> None:
            for optimizer in optimizers(algorithm):
                assert all(group["fused"] == (device == "cuda") for group in optimizer.param_groups)
                for group in optimizer.param_groups:
                    for parameter in group["params"]:
                        parameter.grad = torch.ones_like(parameter)
                optimizer.step()

        source, restored = make_algorithm(), make_algorithm()
        step(source)
        restored.policy.load_state_dict(source.policy.state_dict())
        state = deepcopy(source._get_optimizer_state_dict())
        for name in ("actor_optimizer", "critic_optimizer", "ent_coef_optimizer"):
            for group in state[name]["param_groups"]:
                group.update(fused=True, capturable=True)
        restored._apply_optimizer_state_dict(state, [], [])
        step(source)
        step(restored)
        for expected_optimizer, actual_optimizer in zip(optimizers(source), optimizers(restored), strict=True):
            expected_parameters = [value for group in expected_optimizer.param_groups for value in group["params"]]
            actual_parameters = [value for group in actual_optimizer.param_groups for value in group["params"]]
            for expected, actual in zip(expected_parameters, actual_parameters, strict=True):
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(actual_optimizer.state[actual], expected_optimizer.state[expected])
    finally:
        env.close()
