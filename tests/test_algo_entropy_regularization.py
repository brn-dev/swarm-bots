from collections.abc import Iterator
from contextlib import nullcontext
import csv
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from gymnasium import spaces

from swarmbots.learn.action_dists.predicted_std_gaussian_action_dist import PredictedStdGaussianConfig
from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.mat_qcx.mat_qcx_decoder import MATQCXDecoderConfig
from swarmbots.learn.algos.off_policy.actor_heads import TMASACActorHeadConfig, TMASACActorHeadKind
from swarmbots.learn.algos.off_policy.nop import SACNOPConfig
from swarmbots.learn.algos.off_policy.replay_buffer import OffPolicyReplayBatch
from swarmbots.learn.algos.sac.recurrent_sac import RecurrentSAC
from swarmbots.learn.algos.sac.recurrent_tmasac_policy import RecurrentTMASACPolicy
from swarmbots.learn.algos.sac.sac import SAC
from swarmbots.learn.algos.sac.segment_tmasac_policy import SegmentTMASACPolicy
from swarmbots.learn.algos.sac.tmasac_policy import TMASACPolicy, TMASACPolicyConfig
from swarmbots.learn.algos.xlstm.slstm import SLSTMTemporalSequenceModel, SLSTMTemporalSequenceModelConfig
from swarmbots.learn.hybrid_action_space import VectorHybridActionSpace
from swarmbots.learn.exponential_moving_average import ExponentialMovingAverage
from swarmbots.learn.metrics_logger import MetricsLogger
from tests.test_sac_algorithm import _make_env as _make_rollout_env, _make_policy as _make_rollout_policy
from tests.test_recurrent_tmasac import (
    _DummyContinuousEnv,
    _encoder_config,
    _make_recording_eager_compile,
    _make_segment_batch,
    _policy_config,
    _segment_policy_config,
    _shared_recurrent_policy_config,
)


@pytest.fixture(scope="module", autouse=True)
def single_torch_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def isolated_compiler_cache() -> Iterator[None]:
    """Policy variants share Dynamo's limited recompilation cache."""
    torch.compiler.reset()
    yield
    torch.compiler.reset()


def _make_env() -> SimpleNamespace:
    return SimpleNamespace(
        n_agents=3, local_obs_dim=3, global_obs_dim=1, hidden_local_vars_dim=0, hidden_global_vars_dim=0,
        action_space=VectorHybridActionSpace({"cont": spaces.Box(-1.0, 1.0, (2, 3, 1))}),
        observation_space=spaces.Dict({
            "local_obs": spaces.Box(-1.0, 1.0, (2, 3, 3)),
            "global_obs": spaces.Box(-1.0, 1.0, (2, 1)),
            "hidden_local_vars": spaces.Box(-1.0, 1.0, (2, 3, 0)),
            "hidden_global_vars": spaces.Box(-1.0, 1.0, (2, 0)),
            "agent_mask": spaces.MultiBinary((2, 3)),
        }),
    )


def _make_policy(env: SimpleNamespace, kind: TMASACActorHeadKind = TMASACActorHeadKind.INDEPENDENT) -> TMASACPolicy:
    encoder = MATEncoderConfig(d_model=8, nhead=2, num_layers=1, dim_feedforward=16)
    return TMASACPolicy(env, TMASACPolicyConfig(
        actor_encoder_config=encoder, critic_encoder_config=encoder,
        continuous_config=PredictedStdGaussianConfig(base_std=0.5),
        actor_head_config=TMASACActorHeadConfig(
            kind=kind,
            qcx_decoder_config=MATQCXDecoderConfig(
                d_model=8, nhead=2, num_layers=1, dim_feedforward=16,
                assume_agent_mask_is_active_prefix=False,
            ),
        ),
    ))


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("temporal", [False, True])
@pytest.mark.parametrize("num_action_samples", [1, 3])
@pytest.mark.parametrize("kind", [TMASACActorHeadKind.INDEPENDENT, TMASACActorHeadKind.QCX])
def test_sac_diagnostics_use_parameter_sample_masks_without_changing_checkpoint_structure(
        compiled: bool,
        temporal: bool,
        num_action_samples: int,
        kind: TMASACActorHeadKind,
) -> None:
    policy = _make_policy(_make_env(), kind)
    checkpoint_shapes = {name: value.shape for name, value in policy.state_dict().items()}
    if compiled:
        policy._compiled_actor_actions_and_log_probs = torch.compile(
            policy._actor_actions_and_log_probs_impl, backend="eager", fullgraph=True,
        )
        policy._compiled_actor_action_samples = torch.compile(
            policy._actor_action_samples_impl, backend="eager", fullgraph=True,
        )
    agent_mask = torch.tensor([[True, True, False], [True, False, True]])
    if temporal:
        agent_mask = torch.stack((agent_mask, agent_mask.roll(1, dims=1)), dim=1)
    actor_latents = torch.randn(*agent_mask.shape, 8)
    actions, _ = policy.action_log_prob_from_latents(
        actor_latents=actor_latents, agent_mask=agent_mask, previous_actions=None,
        num_action_samples=num_action_samples,
    )
    expected_mask = (
        agent_mask if num_action_samples == 1
        else agent_mask.unsqueeze(0).expand(num_action_samples, *agent_mask.shape)
    )
    torch.testing.assert_close(policy.action_dist_metrics_agent_mask, expected_mask)

    algorithm = SAC.__new__(SAC)
    algorithm.policy = policy
    algorithm.metrics_action_splitters = None
    metrics = algorithm._compute_action_metrics(actions[expected_mask], prefix="replay")
    expected_entropy = policy.action_dist.distributions[0].distribution.entropy().squeeze(-1)[expected_mask].mean()
    assert metrics["replay_act0_ent_normal"] == pytest.approx(expected_entropy.item(), abs=1e-6)
    assert {name: value.shape for name, value in policy.state_dict().items()} == checkpoint_shapes


@pytest.mark.parametrize("actor_action_samples,target_action_samples", [(1, 1), (2, 3)])
def test_rollout_diagnostics_describe_the_collected_policy_before_replay_training(
        actor_action_samples: int,
        target_action_samples: int,
) -> None:
    torch.manual_seed(42)
    env = _make_rollout_env()
    try:
        policy = _make_rollout_policy(env)
        algorithm = SAC(
            policy, env, buffer_capacity_per_env=16, learning_starts=0, batch_size=4,
            rollout_steps_per_iteration=4, gradient_steps=1, learning_rate_warmup_updates=0,
            actor_action_samples=actor_action_samples, target_action_samples=target_action_samples,
            train_device="cpu", rollout_device="cpu", replay_storage_device="cpu",
        )
        expected_entropy: list[float] = []
        expected_std: list[float] = []
        train = algorithm.train

        def capture_rollout_distribution(*, gradient_steps: int) -> dict[str, Any]:
            for distribution in policy.action_dist.distributions:
                expected_entropy.append(distribution.distribution.entropy().mean().item())
                expected_std.append(distribution.distribution.stddev.mean().item())
            return train(gradient_steps=gradient_steps)

        with patch.object(algorithm, "train", side_effect=capture_rollout_distribution):
            metrics, _ = algorithm.perform_iteration(
                ExponentialMovingAverage(0.1), ExponentialMovingAverage(0.1), update_ema=False,
            )

        assert metrics["updates"] == 1
        for index, (entropy, std) in enumerate(zip(expected_entropy, expected_std, strict=True)):
            assert metrics[f"rollout_act{index}_ent_normal"] == pytest.approx(entropy)
            assert metrics[f"rollout_std{index}"].mean == pytest.approx(std)
    finally:
        env.close()


def test_sac_keeps_its_entropy_objective_and_automatic_temperature_updates() -> None:
    env = _make_env()
    policy = _make_policy(env)
    algorithm = SAC(
        policy, env, learning_rate_warmup_updates=0, buffer_capacity_per_env=4, learning_starts=0,
        batch_size=2, ent_coef="auto_0.2", target_entropy=-0.1,
        actor_action_samples=2, target_action_samples=3, train_device="cpu",
        rollout_device="cpu", replay_storage_device="cpu",
        replay_compile_tensor_operations=False, sac_compile_tensor_operations=False,
        sac_compile_optimizer_steps=False,
    )
    batch = OffPolicyReplayBatch(
        local_obs=torch.randn(2, 3, 3), global_obs=torch.randn(2, 1),
        hidden_local_vars=torch.empty(2, 3, 0), hidden_global_vars=torch.empty(2, 0),
        agent_mask=torch.tensor([[True, True, False], [True, False, False]]),
        actions=torch.zeros(2, 3, 1), rewards=torch.zeros(2),
        terminations=torch.zeros(2, dtype=torch.bool), truncations=torch.zeros(2, dtype=torch.bool),
        previous_actions=None, next_local_obs=torch.randn(2, 3, 3), next_global_obs=torch.randn(2, 1),
        next_hidden_local_vars=torch.empty(2, 3, 0), next_hidden_global_vars=torch.empty(2, 0),
        next_agent_mask=torch.tensor([[True, False, False], [True, True, False]]),
    )
    original_temperature = algorithm.log_ent_coef.detach().clone()
    metrics, _, _ = algorithm._train_step(batch, global_update_idx=0)

    assert metrics["actor_loss"] == pytest.approx(metrics["ent_coef"] * metrics["log_prob"] - metrics["q_pi"])
    assert metrics["actor_total_loss"] == pytest.approx(metrics["actor_loss"])
    assert not torch.equal(algorithm.log_ent_coef.detach(), original_temperature)
    assert "ent_coef_loss" in metrics
    torch.testing.assert_close(
        policy.action_dist_metrics_agent_mask, batch.next_agent_mask.unsqueeze(0).expand(3, 2, 3),
    )


@pytest.mark.parametrize("sampling_path", ["recurrent", "compiled", "shared", "segment"])
@pytest.mark.parametrize("has_truncation", [False, True])
@pytest.mark.parametrize("has_agent_mask", [False, True])
@pytest.mark.usefixtures("isolated_compiler_cache")
def test_recurrent_replay_diagnostics_exclude_unused_successor_rows(
        sampling_path: str,
        has_truncation: bool,
        has_agent_mask: bool,
) -> None:
    torch.manual_seed(42)
    env = _make_rollout_env()
    try:
        nop_config = SACNOPConfig(enabled=False)
        if sampling_path == "segment":
            policy_class = SegmentTMASACPolicy
            config = _segment_policy_config(nop_config=nop_config)
        elif sampling_path == "shared":
            policy_class = RecurrentTMASACPolicy
            config = _shared_recurrent_policy_config(nop_config=nop_config)
        else:
            policy_class = RecurrentTMASACPolicy
            config = _policy_config(_encoder_config(
                SLSTMTemporalSequenceModel, SLSTMTemporalSequenceModelConfig(num_heads=2),
            ), nop_config=nop_config)

        compile_context = (
            patch("swarmbots.learn.algos.sac.tmasac_policy.torch.compile",
                  side_effect=_make_recording_eager_compile([]))
            if sampling_path == "compiled" else nullcontext()
        )
        with compile_context:
            policy = policy_class(env, replace(config, compile_modules=sampling_path == "compiled"))
            algorithm = RecurrentSAC(
                policy, env, burn_in_steps=0, learning_steps=3, temporal_state_store_interval=1,
                buffer_capacity_per_env=8, learning_starts=0, batch_size=2, learning_rate_warmup_updates=0,
                train_device="cpu", rollout_device="cpu", replay_storage_device="cpu",
            )
        batch = _make_segment_batch(
            batch_size=2, sequence_length=3, n_agents=env.n_agents, local_obs_dim=env.local_obs_dim,
            global_obs_dim=env.global_obs_dim, hidden_local_vars_dim=env.hidden_local_vars_dim,
            hidden_global_vars_dim=env.hidden_global_vars_dim, action_dim=env.action_space.total_agent_action_dim,
        )
        batch.train_mask[0, -1] = False
        batch.truncations[0, 1] = has_truncation
        if has_agent_mask:
            batch.agent_mask[..., -1] = False
            batch.next_agent_mask[..., -1] = False
        else:
            batch = replace(batch, agent_mask=None, next_agent_mask=None)
        valid_targets = torch.tensor([False, True, has_truncation, False]).unsqueeze(-1).expand(4, env.n_agents).clone()
        if has_agent_mask:
            valid_targets[..., -1] = False

        with patch.object(algorithm, "_sample_training_batches", return_value=(batch, None, False)):
            metrics = algorithm.train(gradient_steps=1)

        assert metrics["updates"] == 1
        for index, distribution in enumerate(policy.action_dist.distributions):
            expected_entropy = distribution.distribution.entropy()[valid_targets].mean().item()
            expected_stds = distribution.distribution.stddev[valid_targets]
            assert metrics[f"replay_act{index}_ent_normal"] == pytest.approx(expected_entropy)
            std_metrics = metrics[f"replay_std{index}"]
            assert std_metrics.n == expected_stds.numel()
            assert std_metrics.mean == pytest.approx(expected_stds.mean().item())
    finally:
        env.close()


@pytest.mark.parametrize("num_action_samples", [1, 3])
def test_sac_diagnostic_masks_preserve_sampled_morphology_when_environment_buffers_are_reused(
        num_action_samples: int,
) -> None:
    policy = _make_policy(_make_env())
    agent_mask = torch.tensor([[True, True, False], [True, False, True]])
    sampled_mask = agent_mask.clone()
    actions, _ = policy.action_log_prob_from_latents(
        actor_latents=torch.randn(2, 3, 8), agent_mask=agent_mask, previous_actions=None,
        num_action_samples=num_action_samples,
    )
    expected_mask = (
        sampled_mask if num_action_samples == 1
        else sampled_mask.unsqueeze(0).expand(num_action_samples, *sampled_mask.shape)
    )
    sampled_actions = actions[expected_mask]
    algorithm = SAC.__new__(SAC)
    algorithm.policy = policy
    algorithm.metrics_action_splitters = None
    expected = algorithm._compute_action_metrics(sampled_actions, prefix="rollout")
    agent_mask.logical_not_()
    actual = algorithm._compute_action_metrics(sampled_actions, prefix="rollout")
    assert actual["rollout_act0_ent_normal"] == pytest.approx(expected["rollout_act0_ent_normal"])
    assert actual["rollout_std0"] == expected["rollout_std0"]


@pytest.mark.parametrize("compiled", [False, True])
def test_actor_gradients_survive_later_target_parameter_refresh(compiled: bool) -> None:
    env = _make_env()
    policy = _make_policy(env)
    reference = _make_policy(env)
    reference.load_state_dict(policy.state_dict())
    if compiled:
        policy._compiled_actor_action_samples = torch.compile(
            policy._actor_action_samples_impl, backend="eager", fullgraph=True,
        )
    latents = torch.randn(2, 3, 8)
    mask = torch.tensor([[True, True, False], [True, False, True]])
    torch.manual_seed(431)
    expected_actions, expected_log_probs = reference.action_log_prob_from_latents(
        actor_latents=latents, agent_mask=mask, previous_actions=None, num_action_samples=4,
    )
    expected_loss = expected_log_probs.mean() + expected_actions.square().mean()
    expected_gradients = torch.autograd.grad(expected_loss, tuple(reference.action_dist.parameters()))

    torch.manual_seed(431)
    actions, log_probs = policy.action_log_prob_from_latents(
        actor_latents=latents, agent_mask=mask, previous_actions=None, num_action_samples=4,
    )
    with torch.no_grad():
        policy.action_log_prob_from_latents(
            actor_latents=latents + 1.0, agent_mask=mask.roll(1, dims=1),
            previous_actions=None, num_action_samples=4, use_rsample=False,
        )
    torch.testing.assert_close(actions, expected_actions)
    torch.testing.assert_close(log_probs, expected_log_probs)
    gradients = torch.autograd.grad(log_probs.mean() + actions.square().mean(), tuple(policy.action_dist.parameters()))
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected)


def test_distribution_entropy_diagnostics_are_numeric_in_actual_csv_logs(tmp_path: Path) -> None:
    policy = _make_policy(_make_env())
    agent_mask = torch.tensor([[True, True, False], [True, False, True]])
    actions, _ = policy.action_log_prob_from_latents(
        actor_latents=torch.randn(2, 3, 8), agent_mask=agent_mask, previous_actions=None,
    )
    algorithm = SAC.__new__(SAC)
    algorithm.policy = policy
    algorithm.metrics_action_splitters = None
    metrics = algorithm._compute_action_metrics(actions[agent_mask], prefix="replay")
    assert isinstance(metrics["replay_act0_ent_normal"], float)
    metrics_logger = MetricsLogger(log_dir=tmp_path, console_keys=[])
    try:
        metrics_logger.log(metrics)
    finally:
        metrics_logger.close()

    with (tmp_path / "log.csv").open(newline="", encoding="utf-8") as log_file:
        row = next(csv.DictReader(log_file, delimiter=";"))
    assert float(row["replay_act0_ent_normal"]) == pytest.approx(metrics["replay_act0_ent_normal"])


@pytest.mark.parametrize("sampling_path", [
    "recurrent", "recurrent_selected", "compiled", "compiled_selected", "shared", "segment",
])
@pytest.mark.parametrize("kind", [TMASACActorHeadKind.INDEPENDENT, TMASACActorHeadKind.QCX])
@pytest.mark.parametrize("num_action_samples", [1, 3])
@pytest.mark.parametrize("has_agent_mask", [False, True])
@pytest.mark.usefixtures("isolated_compiler_cache")
def test_recurrent_sac_diagnostics_exclude_padding_and_preserve_mask_snapshots(
        sampling_path: str,
        kind: TMASACActorHeadKind,
        num_action_samples: int,
        has_agent_mask: bool,
) -> None:
    torch.manual_seed(10)
    env = _DummyContinuousEnv()
    if sampling_path == "segment":
        policy = SegmentTMASACPolicy(env, _segment_policy_config(actor_head_kind=kind))
    else:
        config = (
            _shared_recurrent_policy_config()
            if sampling_path == "shared"
            else _policy_config(_encoder_config(
                SLSTMTemporalSequenceModel, SLSTMTemporalSequenceModelConfig(num_heads=2),
            ), actor_head_kind=kind)
        )
        config = replace(config, actor_head_config=replace(config.actor_head_config, kind=kind))
        if sampling_path.startswith("compiled"):
            compiled_graphs: list[torch.fx.GraphModule] = []
            with patch(
                "swarmbots.learn.algos.sac.tmasac_policy.torch.compile",
                side_effect=_make_recording_eager_compile(compiled_graphs),
            ):
                policy = RecurrentTMASACPolicy(env, replace(config, compile_modules=True))
                policy.configure_actor_compilation(
                    encoder_only_sequence_lengths=(), action_sequence_lengths=(3,),
                    action_sequence_with_selected_states_lengths=(3,),
                )
        else:
            policy = RecurrentTMASACPolicy(env, config)
    policy.eval()
    with torch.no_grad():
        policy.action_dist.distributions[0].log_std_net.weight.fill_(0.25)
    time_mask = torch.tensor([[True, False, False], [True, True, False]])
    agent_mask = torch.tensor([
        [[True, False, True], [True, True, True], [True, True, True]],
        [[False, True, True], [True, True, False], [True, True, True]],
    ]) if has_agent_mask else None
    valid_agents = time_mask.unsqueeze(-1).expand(2, 3, env.n_agents).clone()
    if agent_mask is not None:
        valid_agents &= agent_mask
    expected_mask = (
        valid_agents if num_action_samples == 1
        else valid_agents.unsqueeze(0).expand(num_action_samples, *valid_agents.shape)
    )
    sampling_kwargs = dict(
        local_obs=torch.randn(2, 3, env.n_agents, env.local_obs_dim),
        global_obs=torch.randn(2, 3, env.global_obs_dim), agent_mask=agent_mask,
        previous_actions=None, deterministic=True, use_rsample=False, initial_state=None,
        time_mask=time_mask, num_action_samples=num_action_samples,
    )
    sample_actions = policy.action_log_prob_sequence
    if sampling_path.endswith("selected") or sampling_path == "shared":
        sampling_kwargs["state_output_indices"] = torch.tensor([[0, 0], [1, 1]])
        sample_actions = (
            policy.action_log_prob_sequence_with_selected_states_and_shared_latents
            if sampling_path == "shared"
            else policy.action_log_prob_sequence_with_selected_states
        )
    with torch.no_grad():
        actions, *_ = sample_actions(**sampling_kwargs)
    distribution = policy.action_dist.distributions[0].distribution
    expected_entropy = distribution.entropy()[expected_mask].mean().item()
    expected_stds = distribution.stddev[expected_mask]
    time_mask.logical_not_()
    if agent_mask is not None:
        agent_mask.logical_not_()
    algorithm = SAC.__new__(SAC)
    algorithm.policy = policy
    algorithm.metrics_action_splitters = None
    metrics = algorithm._compute_action_metrics(actions[expected_mask], prefix="replay")

    assert metrics["replay_act0_ent_normal"] == pytest.approx(expected_entropy, abs=1e-6)
    std_metrics = metrics["replay_std0"]
    assert std_metrics.n == expected_stds.numel()
    assert std_metrics.mean == pytest.approx(expected_stds.mean().item(), abs=1e-6)
    assert std_metrics.min_value == pytest.approx(expected_stds.min().item(), abs=1e-6)
    assert std_metrics.max_value == pytest.approx(expected_stds.max().item(), abs=1e-6)


@pytest.mark.parametrize("num_action_samples", [1, 3])
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=[
        pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    ]),
])
def test_temporal_diagnostic_masks_follow_the_sampled_agent_count(num_action_samples: int, device: str) -> None:
    policy = _make_policy(_make_env()).to(device)
    time_mask = torch.tensor([[True, False, False], [True, True, False]], device=device)
    actor_latents = torch.randn(2, 3, policy.n_agents - 1, 8, device=device)
    valid_agents = time_mask.unsqueeze(-1).expand(actor_latents.shape[:-1])
    sampled_mask = (
        valid_agents if num_action_samples == 1
        else valid_agents.unsqueeze(0).expand(num_action_samples, *valid_agents.shape)
    )
    actions, _ = policy.action_log_prob_from_latents(
        actor_latents=actor_latents, agent_mask=None, previous_actions=None,
        time_mask=time_mask, num_action_samples=num_action_samples,
    )
    algorithm = SAC.__new__(SAC)
    algorithm.policy = policy
    algorithm.metrics_action_splitters = None
    metrics = algorithm._compute_action_metrics(actions[sampled_mask], prefix="replay")
    assert metrics["replay_std0"].n == int(sampled_mask.sum())
