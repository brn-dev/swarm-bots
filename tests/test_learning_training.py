from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from gymnasium import spaces
from torch import nn

from swarmbots import ALL_BENCHMARK_IDS, evaluate_policy
from swarmbots.learn import as_benchmark_policy, list_variants, make_training, train
from swarmbots.learn.action_dists.gumbel_softmax_sign_magnitude_action_dist import (
    GumbelSoftmaxSignMagnitudeBetaActionDist,
)
from swarmbots.learn.action_dists.predicted_std_gaussian_action_dist import PredictedStdGaussianActionDist
from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import SignMagnitudeBetaActionDist
from swarmbots.learn.algos.mappo.mappo_policy import MAPPOPolicy
from swarmbots.learn.algos.mat.mat_ind_policy import MATIndPolicy
from swarmbots.learn.algos.ppo.ppo import PPO
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamples
from swarmbots.learn.algos.sac.recurrent_sac import RecurrentSAC
from swarmbots.learn.algos.sac.sac import SAC
from swarmbots.learn.algos.sac.tmasac_policy import TMASACPolicy
from swarmbots.learn.checkpointing import capture_env_state
from swarmbots.learn.exponential_moving_average import ExponentialMovingAverage
from swarmbots.learn.hybrid_action_space import HybridActionSpace
from swarmbots.learn.presets.policy_factory import _is_sac_policy_variant, _make_base_policy
from swarmbots.learn.presets.transformer import MATInitGains, MATNormalizationConfig
from swarmbots.learn.training import _variant_options
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWPreConnectedUnitLocationsConfig


@pytest.fixture(scope="module", autouse=True)
def single_torch_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _scenario_kwargs() -> dict[str, Any]:
    return {
        "unit_start_locations": MJWPreConnectedUnitLocationsConfig(
            num_units=2,
            pool_seeds=(42000,),
            max_radius=1.5,
            z_pos=0.5,
        ),
        "compile_reward_kernel": False,
        "reset_settle_time": 0,
    }


def _training_kwargs() -> dict[str, Any]:
    return {
        "num_envs": 2,
        "device": "cpu",
        "episode_length": 4,
        "scenario_kwargs": _scenario_kwargs(),
        "env_kwargs": {"compile_tensor_operations": False},
        "policy_kwargs": {"enc_d_model": 32, "dec_d_model": 16},
    }


def test_variants_expose_only_canonical_architectures_and_ablations() -> None:
    assert set(list_variants()) == {
        "ppo",
        "ppo_small",
        "mappo",
        "mappo_small",
        "mat_orig",
        "mat_ind",
        "mat_ind_no_attention",
        "mat_qcx",
        "mat_dec",
        "mat_ind_lstm",
        "mat_qcx_lstm",
        "tmasac",
        "tmasac_dec",
        "tmasac_slstm",
        "tmasac_lstm",
        "tmasac_lstm_no_actor_state",
        "tmasac_segment",
        "tmasac_shared_encoder",
        "tmasac_slstm_shared_encoder",
        "mat_qcx_no_nop",
        "tmasac_no_nop",
        "tmasac_slstm_no_residual",
        "tmasac_slstm_no_nop",
        "tmasac_swiglu",
        "tmasac_slstm_swiglu",
    }
    assert len(list_variants()) == len(set(list_variants()))


@pytest.mark.parametrize("variant", [
    "tmasac_slstm",
    "tmasac_slstm_no_nop",
    "tmasac_slstm_shared_encoder",
    "tmasac_slstm_swiglu",
])
def test_slstm_presets_include_the_temporal_residual_connection(variant: str) -> None:
    options, _ = _variant_options(variant)
    assert options["rmat_temporal_residual"] is True


def test_slstm_no_residual_ablation_changes_only_the_residual_connection() -> None:
    main_options, main_algorithm = _variant_options("tmasac_slstm")
    ablation_options, ablation_algorithm = _variant_options("tmasac_slstm_no_residual")
    assert ablation_options["rmat_temporal_residual"] is False
    assert main_algorithm == ablation_algorithm
    assert main_options == {**ablation_options, "rmat_temporal_residual": True}


def _small_sac_policy(variant: str, n_agents: int = 3) -> TMASACPolicy:
    env = SimpleNamespace(
        n_agents=n_agents,
        local_obs_dim=5,
        global_obs_dim=2,
        hidden_local_vars_dim=0,
        hidden_global_vars_dim=0,
        action_space=HybridActionSpace({
            "actuators": spaces.Box(-1.0, 1.0, shape=(n_agents, 2)),
            "connectors": spaces.Box(-1.0, 1.0, shape=(n_agents, 1)),
        }),
    )
    options, _ = _variant_options(variant)
    options["use_nop"] = False
    return _make_base_policy(
        env=env,
        enc_d_model=32,
        enc_nhead=4,
        dec_d_model=16,
        dec_nhead=2,
        use_popart=False,
        popart_beta=5e-4,
        popart_init_sigma=0.65,
        compile_policy_modules=False,
        policy_compile_mode="default",
        gsde_init_stds=[0.25, 0.30],
        mat_add_agent_embeddings=True,
        act_fn_cls=nn.GELU,
        mat_init_gains=MATInitGains(),
        mat_normalization=MATNormalizationConfig(),
        **options,
    )


@pytest.mark.parametrize("variant", ["tmasac", "tmasac_dec", "tmasac_lstm_no_actor_state", "tmasac_segment"])
def test_sac_presets_produce_actions_for_swarms_larger_than_twenty_agents(variant: str) -> None:
    n_agents = 21
    policy = _small_sac_policy(variant, n_agents)
    with torch.no_grad():
        actions = policy.act(
            local_obs=torch.zeros(1, n_agents, policy.local_obs_dim),
            global_obs=torch.zeros(1, policy.global_obs_dim),
            agent_mask=torch.ones(1, n_agents, dtype=torch.bool),
            deterministic=True,
        )
    assert actions.shape == (1, n_agents, 3)
    assert torch.isfinite(actions).all()
    assert (actions >= -1).all() and (actions <= 1).all()


def test_tmasac_dec_actor_is_independent_and_critic_mixes_agents() -> None:
    torch.manual_seed(0)
    policy = _small_sac_policy("tmasac_dec")
    baseline = _small_sac_policy("tmasac")
    assert not any(isinstance(module, nn.MultiheadAttention) for module in policy.actor_encoder.modules())
    assert any(isinstance(module, nn.MultiheadAttention) for module in policy.critic.modules())
    assert policy.critic_encoder_config == baseline.critic_encoder_config
    assert policy.config.critic_config == baseline.config.critic_config
    assert _variant_options("tmasac_dec")[1] == _variant_options("tmasac")[1]

    local_obs = torch.randn(2, 3, policy.local_obs_dim)
    global_obs = torch.randn(2, policy.global_obs_dim)
    agent_mask = torch.tensor([[True, True, False], [True, True, True]])
    changed_local_obs = local_obs.clone()
    changed_local_obs[:, 1] += torch.tensor([10.0, -20.0, 30.0, -40.0, 50.0])
    with torch.no_grad():
        actions = policy.act(local_obs, global_obs, agent_mask=agent_mask, deterministic=True)
        changed_actions = policy.act(changed_local_obs, global_obs, agent_mask=agent_mask, deterministic=True)
        torch.testing.assert_close(actions[:, 0], changed_actions[:, 0], rtol=0, atol=0)
        assert not torch.allclose(actions[:, 1], changed_actions[:, 1])
        assert (actions[~agent_mask] == 0).all()

        critic_inputs = dict(
            global_inputs=global_obs, actions=actions, hidden_local_vars=None,
            hidden_global_vars=None, agent_mask=agent_mask,
        )
        latents = policy.critic.encode(local_inputs=local_obs, **critic_inputs)
        changed_latents = policy.critic.encode(local_inputs=changed_local_obs, **critic_inputs)
        assert not torch.allclose(latents[:, 0], changed_latents[:, 0])

    sampled_actions, log_probs = policy.action_log_prob(
        local_obs=local_obs, global_obs=global_obs, agent_mask=agent_mask,
    )
    q1, q2 = policy.q_values(
        local_obs=local_obs, global_obs=global_obs, actions=sampled_actions, agent_mask=agent_mask,
    )
    (log_probs.mean() - torch.minimum(q1, q2).mean()).backward()
    for parameters in (policy.actor_parameters(), policy.critic_parameters()):
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) for gradient in gradients)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("variant", "expected_ratios"),
    [
        ("ppo", (1.04**2, 1.04)),
        ("ppo_small", (1.04**2, 1.04)),
        ("mappo", (1.04, 1.04, 1.04)),
        ("mappo_small", (1.04, 1.04, 1.04)),
        ("mat_ind_no_attention", (1.04, 1.04, 1.04)),
    ],
)
def test_ppo_presets_clip_and_measure_kl_at_the_intended_action_scope(
    variant: str, expected_ratios: tuple[float, ...]
) -> None:
    algorithm = make_training(
        "SwarmBots-WallEasy-v0",
        variant,
        algorithm_kwargs={"use_popart": False},
        **_training_kwargs(),
    )
    try:
        batch = PPOSamples(
            local_obs=torch.empty(2, 2, 0),
            global_obs=torch.empty(2, 0),
            hidden_local_vars=torch.empty(2, 2, 0),
            hidden_global_vars=torch.empty(2, 0),
            agent_mask=torch.tensor([[True, True], [True, False]]),
            previous_actions=None,
            actions=torch.zeros(2, 2, 1),
            log_probs=torch.zeros(2, 2),
            values=torch.zeros(2),
            returns=torch.zeros(2),
            advantages=torch.ones(2),
        )
        ratios = torch.tensor([[1.04, 1.04], [1.04, 10.0]])
        loss, approximate_kl, metrics = algorithm.compute_ppo_loss(
            batch, ratios.log(), torch.zeros(2), normalize_advantage=False
        )
        expected = torch.tensor(expected_ratios)
        torch.testing.assert_close(loss, -expected.clamp(max=1.05).mean())
        assert approximate_kl == pytest.approx((expected - 1 - expected.log()).mean().item(), abs=1e-7)
        assert metrics["clip_frac"] == pytest.approx((expected > 1.05).float().mean().item())
    finally:
        algorithm.env.close()


@pytest.mark.integration
@pytest.mark.parametrize("variant", list_variants())
def test_each_variant_produces_finite_benchmark_actions(variant: str) -> None:
    algorithm = make_training("SwarmBots-WallEasy-v0", variant, **_training_kwargs())
    try:
        expected_distribution = (
            PredictedStdGaussianActionDist if isinstance(algorithm, SAC) else SignMagnitudeBetaActionDist
        )
        for distribution in algorithm.policy.action_dist.distributions:
            assert isinstance(distribution, expected_distribution)
            if isinstance(distribution, PredictedStdGaussianActionDist):
                assert distribution.squash_output
        observations, _ = algorithm.env.unwrapped.reset(seed=17)
        policy = as_benchmark_policy(algorithm)
        actions = policy(observations, torch.ones(2, dtype=torch.bool))
        assert actions.keys() == {"actuators", "connectors"}
        for key, space in algorithm.env.unwrapped.action_space.items():
            assert actions[key].shape == space.shape
            assert torch.isfinite(actions[key]).all()
            assert (actions[key] >= -1).all() and (actions[key] <= 1).all()
    finally:
        algorithm.env.close()


@pytest.mark.integration
@pytest.mark.parametrize("variant", ["tmasac", "tmasac_dec", "tmasac_slstm"])
def test_sac_action_distribution_can_be_overridden_without_changing_the_architecture(variant: str) -> None:
    algorithm = make_training(
        "SwarmBots-WallEasy-v0", variant,
        continuous_action_dist="gumbel_softmax_sign_magnitude_beta",
        **_training_kwargs(),
    )
    try:
        assert all(
            isinstance(distribution, GumbelSoftmaxSignMagnitudeBetaActionDist)
            for distribution in algorithm.policy.action_dist.distributions
        )
        observations, _ = algorithm.env.unwrapped.reset(seed=17)
        actions = as_benchmark_policy(algorithm)(observations, torch.ones(2, dtype=torch.bool))
        assert all(torch.isfinite(action).all() for action in actions.values())
    finally:
        algorithm.env.close()


@pytest.mark.integration
@pytest.mark.parametrize("benchmark_id", ALL_BENCHMARK_IDS)
def test_nop_can_be_constructed_for_every_registered_scenario(benchmark_id: str) -> None:
    algorithm = make_training(benchmark_id, "mat_qcx", **_training_kwargs())
    try:
        observations, _ = algorithm.env.reset(seed=17)
        with torch.no_grad():
            actions = algorithm.policy.act(
                local_obs=observations["local_obs"],
                global_obs=observations["global_obs"],
                agent_mask=observations["agent_mask"],
            )
        assert torch.isfinite(actions).all()
    finally:
        algorithm.env.close()


@pytest.mark.integration
@pytest.mark.parametrize("variant", list_variants())
def test_learning_updates_parameters_on_actual_benchmark_transitions(variant: str) -> None:
    algorithm_options: dict[str, Any]
    policy_options, _ = _variant_options(variant)
    if _is_sac_policy_variant(policy_options["policy_variant"]):
        algorithm_options = {
            "learning_starts": 0,
            "batch_size": 2,
            "buffer_capacity_per_env": 32,
            "gradient_steps": 1,
            "learning_rate": 1e-3,
            "learning_rate_warmup_updates": 0,
        }
        if policy_options["policy_variant"] in {"tmasac_recurrent", "tmasac_segment"}:
            algorithm_options.update(burn_in_steps=1, learning_steps=3, temporal_state_store_interval=1)
    else:
        algorithm_options = {"learning_rate": 1e-3, "n_epochs": 1, "target_kl": None}
    algorithm = make_training(
        "SwarmBots-WallEasy-v0",
        variant,
        rollout_steps_per_env=4,
        algorithm_kwargs=algorithm_options,
        **_training_kwargs(),
    )
    try:
        parameters_before = [parameter.detach().clone() for parameter in algorithm.policy.parameters()]
        metrics, transitions = algorithm.perform_iteration(
            ExponentialMovingAverage(0.1),
            ExponentialMovingAverage(0.1),
            update_ema=True,
        )
        assert transitions == 8
        assert metrics
        assert algorithm.n_total_updates > 0
        assert any(
            not torch.equal(before, after)
            for before, after in zip(parameters_before, algorithm.policy.parameters(), strict=True)
        )
        assert all(torch.isfinite(parameter).all() for parameter in algorithm.policy.parameters())
    finally:
        algorithm.env.close()


@pytest.mark.integration
def test_training_saves_and_resumes_then_evaluates_without_updating_normalization(tmp_path: Path) -> None:
    kwargs = _training_kwargs()
    algorithm = train(
        "SwarmBots-WallEasy-v0",
        "mat_ind",
        total_timesteps=8,
        run_dir=tmp_path / "first",
        learn_kwargs={"extra_run_metadata": {"experiment_label": "metadata-forwarding"}},
        algorithm_kwargs={"n_epochs": 1},
        **kwargs,
    )
    checkpoint = next((tmp_path / "first/models").glob("*_final.pt"))
    metadata_path = next((tmp_path / "first").glob("run_metadata*.json"))
    metadata = json.loads(metadata_path.read_text())
    assert metadata["experiment_label"] == "metadata-forwarding"
    assert metadata["benchmark_id"] == "SwarmBots-WallEasy-v0"
    assert metadata["learning_variant"] == "mat_ind"
    resumed = train(
        "SwarmBots-WallEasy-v0",
        "mat_ind",
        total_timesteps=16,
        run_dir=tmp_path / "resumed",
        load_path=checkpoint,
        algorithm_kwargs={"n_epochs": 1},
        **kwargs,
    )
    assert algorithm.n_total_timesteps == 8
    assert resumed.n_total_timesteps == 16
    before = capture_env_state(resumed.env)
    actor = as_benchmark_policy(resumed)
    result = evaluate_policy(
        actor,
        "SwarmBots-WallEasy-v0",
        num_envs=2,
        num_episodes=2,
        device="cpu",
        episode_length=2,
        scenario_kwargs=kwargs["scenario_kwargs"],
        env_kwargs=kwargs["env_kwargs"],
        action_mode="deterministic",
    )
    assert len(result.episode_returns) == 2
    after = capture_env_state(resumed.env)
    for old_wrapper, new_wrapper in zip(before, after, strict=True):
        for key in old_wrapper:
            if key.endswith("_rms"):
                for field in ("mean", "var", "count"):
                    torch.testing.assert_close(getattr(old_wrapper[key], field), getattr(new_wrapper[key], field))


@pytest.mark.integration
def test_presets_select_the_intended_algorithm_and_actor() -> None:
    for variant, expected_class in (
        ("mappo", PPO), ("mat_ind_no_attention", PPO), ("tmasac", SAC),
        ("tmasac_dec", SAC), ("tmasac_slstm", RecurrentSAC),
        ("tmasac_slstm_no_residual", RecurrentSAC),
    ):
        algorithm = make_training("SwarmBots-WallEasy-v0", variant, **_training_kwargs())
        try:
            assert isinstance(algorithm, expected_class)
            if variant == "mappo":
                assert isinstance(algorithm.policy.policy, MAPPOPolicy)
            elif variant == "mat_ind_no_attention":
                assert isinstance(algorithm.policy.policy, MATIndPolicy)
                assert algorithm.policy.policy.config.encoder_config.use_agent_attention is False
            elif variant in {"tmasac_slstm", "tmasac_slstm_no_residual"}:
                assert algorithm.policy.actor_encoder.config.temporal_residual is (variant == "tmasac_slstm")
        finally:
            algorithm.env.close()


@pytest.mark.integration
@pytest.mark.parametrize("phase", ["warmup_stopped", "growth_pending", "disabled"])
def test_ppo_checkpoint_continues_the_same_learning_rate_schedule(tmp_path: Path, phase: str) -> None:
    original = make_training("SwarmBots-WallEasy-v0", "mat_ind", **_training_kwargs())
    try:
        original.n_total_iterations = 100
        original._maybe_update_automatic_lr(0.004, 0, {})
        if phase == "growth_pending":
            original._maybe_update_automatic_lr(None, None, {})
        elif phase == "disabled":
            assert original.execute_command("disable_auto_lr", "", None)
        checkpoint = tmp_path / "model.pt"
        original.save(checkpoint, optimizer_state_dict=original._get_optimizer_state_dict())

        resumed = make_training("SwarmBots-WallEasy-v0", "mat_ind", **_training_kwargs())
        try:
            resumed.load(checkpoint)
            assert resumed.learning_rate == pytest.approx(original.learning_rate)
            for _ in range(3):
                original.n_total_iterations += 1
                resumed.n_total_iterations += 1
                expected = original._maybe_update_automatic_lr(None, None, {})
                actual = resumed._maybe_update_automatic_lr(None, None, {})
                assert actual == expected
                assert [group["lr"] for group in resumed.optimizer.param_groups] == pytest.approx(
                    [group["lr"] for group in original.optimizer.param_groups]
                )
        finally:
            resumed.env.close()
    finally:
        original.env.close()


@pytest.mark.integration
@pytest.mark.parametrize("target_kl", [0.0002, 0.002, 0.02])
def test_ppo_learning_rate_controller_uses_the_selected_kl_target(target_kl: float) -> None:
    algorithm = make_training(
        "SwarmBots-WallEasy-v0",
        "mat_ind",
        algorithm_kwargs={"target_kl": target_kl},
        **_training_kwargs(),
    )
    try:
        algorithm.n_total_iterations = 300
        initial_lr = algorithm.learning_rate
        algorithm._maybe_update_automatic_lr(target_kl * 1.51, 7, {})
        assert algorithm.learning_rate == pytest.approx(initial_lr)
        algorithm._maybe_update_automatic_lr(target_kl * 1.6, 7, {})
        assert algorithm.learning_rate < initial_lr
    finally:
        algorithm.env.close()


@pytest.mark.integration
@pytest.mark.parametrize("variant", ["mat_ind_lstm", "tmasac_slstm", "tmasac_lstm"])
def test_benchmark_adapter_resets_memory_for_the_selected_lanes(variant: str) -> None:
    algorithm = make_training("SwarmBots-WallEasy-v0", variant, **_training_kwargs())
    try:
        observations, _ = algorithm.env.unwrapped.reset(seed=17)
        adapter = as_benchmark_policy(algorithm)
        first = adapter(observations, torch.ones(2, dtype=torch.bool))
        adapter(observations, torch.zeros(2, dtype=torch.bool))
        after_reset = adapter(observations, torch.tensor([True, False]))
        for key in first:
            torch.testing.assert_close(after_reset[key][0], first[key][0])
    finally:
        algorithm.env.close()


@pytest.mark.integration
def test_stochastic_gsde_evaluation_preserves_training_noise_state() -> None:
    algorithm = make_training(
        "SwarmBots-WallEasy-v0", "mat_qcx", continuous_action_dist="gsde", **_training_kwargs(),
    )
    try:
        observations, _ = algorithm.env.unwrapped.reset(seed=17)
        action_dist = algorithm.policy.action_dist
        action_dist.reset_temporal_correlations_on_step(batch_shape=(2, 2))
        before = action_dist.get_temporal_correlation_state()
        adapter = as_benchmark_policy(algorithm, deterministic=False)
        for episode_starts in (torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)):
            actions = adapter(observations, episode_starts)
            assert all(torch.isfinite(values).all() for values in actions.values())
            torch.testing.assert_close(action_dist.get_temporal_correlation_state(), before)
    finally:
        algorithm.env.close()


@pytest.mark.integration
def test_training_accepts_independent_nop_and_value_normalization_options() -> None:
    kwargs = _training_kwargs()
    kwargs["policy_kwargs"].update(world_model_loss_coef=0.0, transition_model_d_model=16)
    algorithm = make_training(
        "SwarmBots-WallEasy-v0",
        "mat_qcx",
        algorithm_kwargs={"use_popart": False, "n_epochs": 1},
        **kwargs,
    )
    try:
        assert algorithm.policy.world_model_loss_coef == 0.0
        assert algorithm.use_popart is False
        algorithm.learn(max_total_timesteps=8, enable_command_prompt=False)
        assert algorithm.n_total_updates > 0
        assert all(torch.isfinite(parameter).all() for parameter in algorithm.policy.parameters())
    finally:
        algorithm.env.close()


@pytest.mark.integration
def test_benchmark_evaluation_preserves_mixed_module_training_modes() -> None:
    algorithm = make_training("SwarmBots-WallEasy-v0", "mat_ind", **_training_kwargs())
    try:
        algorithm.policy.train()
        algorithm.policy.action_dist.eval()
        modes_before = {module: module.training for module in algorithm.policy.modules()}
        observations, _ = algorithm.env.unwrapped.reset(seed=17)
        as_benchmark_policy(algorithm)(observations, torch.ones(2, dtype=torch.bool))
        assert {module: module.training for module in algorithm.policy.modules()} == modes_before
    finally:
        algorithm.env.close()


def test_conflicting_value_normalization_options_fail_before_creating_an_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_environment(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Should not create an environment for inconsistent PopArt options")

    monkeypatch.setattr("swarmbots.learn.training.make_env", unexpected_environment)
    with pytest.raises(ValueError, match="use_popart must agree"):
        make_training(
            "SwarmBots-WallEasy-v0",
            "mat_ind",
            policy_kwargs={"use_popart": True},
            algorithm_kwargs={"use_popart": False},
        )


def test_unknown_variant_fails_before_constructing_an_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_environment(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Should not create an environment for an unknown variant")

    monkeypatch.setattr("swarmbots.learn.training.make_env", unexpected_environment)
    with pytest.raises(ValueError, match="Unknown learning variant"):
        make_training("SwarmBots-WallEasy-v0", "typo")
