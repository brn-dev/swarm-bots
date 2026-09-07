from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from swarmbots.learn.action_dists.action_sampling import sample_base_uniform
from swarmbots.learn.action_dists.beta_action_dist import BetaConfig
from swarmbots.learn.action_dists.bernstein_quantile_action_dist import (
    BernsteinQuantileActionDist,
    BernsteinQuantileConfig,
)
from swarmbots.learn.action_dists.gumbel_softmax_sign_magnitude_action_dist import (
    GumbelSoftmaxSignMagnitudeBetaConfig,
)
from swarmbots.learn.action_dists.hybrid_action_dist import ContinuousActionDistConfig
from swarmbots.learn.action_dists.predicted_std_action_dist import PredictedStdConfig
from swarmbots.learn.action_dists.rational_quadratic_spline_quantile_action_dist import (
    RationalQuadraticSplineQuantileActionDist,
    RationalQuadraticSplineQuantileConfig,
)
from swarmbots.learn.algos.sac import (
    SAC,
    ActorStateCriticInputConfig,
    RecurrentSAC,
    RecurrentTMASACPolicy,
    SegmentTMASACPolicy,
    TMASACActorHeadKind,
    TMASACPolicy,
)
from swarmbots.learn.algos.sac.sac_tensor_ops import (
    _actor_loss,
    _entropy_coefficient_loss,
)
from swarmbots.learn.algos.r_mat.temporal_sequence_model import (
    LSTMTemporalSequenceModel,
    LSTMTemporalSequenceModelConfig,
)
from swarmbots.learn.algos.xlstm.slstm import (
    SLSTMTemporalSequenceModel,
    SLSTMTemporalSequenceModelConfig,
)
from swarmbots.learn.action_dists.reparameterized_squashed_gaussian_mixture_action_dist import (
    ReparameterizedSquashedGaussianMixtureConfig,
)
from swarmbots.learn.action_dists.squashed_diag_gaussian_action_dist import SquashedDiagGaussianConfig
from swarmbots.learn.checkpointing import align_torch_compile_state_dict_keys
from swarmbots.learn.temporal_state import clone_temporal_state
from tests.test_recurrent_tmasac import (
    _actor_head_config,
    _encoder_config,
    _make_recording_eager_compile,
    _make_recurrent_critic_algorithm,
    _make_recurrent_critic_batch,
    _make_segment_batch,
    _policy_config,
    _shared_recurrent_policy_config,
    _small_nop_config,
)
from tests.test_sac_algorithm import (
    _make_bootstrap_batch,
    _make_env,
    _make_policy,
    _move_replay_batch,
)


@pytest.mark.parametrize("count", [2, 4, 8])
@pytest.mark.parametrize("shape,axis", [((4, 7, 3), 0), ((2, 4, 3, 5, 2), 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_latin_hypercube_base_noise(
    count: int, shape: tuple[int, ...], axis: int, dtype: torch.dtype
) -> None:
    shape = (*shape[:axis], count, *shape[axis + 1 :])
    torch.manual_seed(19)
    noise = sample_base_uniform(shape, device=torch.device("cpu"), dtype=dtype, stratified_sample_dim=axis)
    assert noise.dtype == dtype and noise.device.type == "cpu"
    assert ((noise >= 0) & (noise < 1)).all()
    strata = (noise * count).floor().movedim(axis, 0).reshape(count, -1)
    torch.testing.assert_close(
        strata.sort(dim=0).values, torch.arange(count, dtype=dtype)[:, None].expand_as(strata)
    )
    assert (strata != strata[:, :1]).any()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_stratified_base_noise_stays_inside_lower_boundary_after_rounding(dtype: torch.dtype) -> None:
    count = 22
    shape = (count, 1)
    zero_jitter = torch.zeros(shape, dtype=dtype)
    ordered_permutation_scores = torch.arange(count, dtype=torch.float64).unsqueeze(-1)
    with patch(
        "swarmbots.learn.action_dists.action_sampling.torch.rand",
        side_effect=(zero_jitter, ordered_permutation_scores),
    ):
        samples = sample_base_uniform(
            shape,
            device=torch.device("cpu"),
            dtype=dtype,
            stratified_sample_dim=0,
        )

    actual_strata = (samples * count).floor().to(dtype=torch.int64).squeeze(-1)
    torch.testing.assert_close(actual_strata, torch.arange(count))


@pytest.mark.parametrize("dist_cls", [BernsteinQuantileActionDist, RationalQuadraticSplineQuantileActionDist])
def test_stratified_quantile_samples_have_pathwise_gradients(dist_cls: type) -> None:
    dist = dist_cls(latent_dim=5, action_dim=2)
    latent = torch.randn(4, 3, 2, 5)
    actions, log_probs = dist.get_actions_with_log_probs(latent, use_rsample=True, stratified_sample_dim=0)
    assert actions.shape == (4, 3, 2, 2) and log_probs.shape == (4, 3, 2)
    strata = ((actions + 1) * 2).floor()
    torch.testing.assert_close(
        strata.sort(dim=0).values, torch.arange(4)[:, None, None, None].expand_as(strata).float()
    )
    for index in range(4):
        gradient = torch.autograd.grad(
            actions[index].square().sum(), dist.action_net.weight, retain_graph=True
        )[0]
        assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    (actions.square().mean() + log_probs.mean()).backward()
    assert torch.isfinite(dist.action_net.weight.grad).all()


def _algorithm(policy: TMASACPolicy, env: object, **kwargs: object) -> SAC:
    return SAC(
        policy=policy,
        env=env,
        buffer_capacity_per_env=8,
        learning_starts=0,
        batch_size=2,
        train_device="cpu",
        replay_storage_device="cpu",
        **kwargs,
    )


@pytest.mark.parametrize(
    "config",
    [
        BernsteinQuantileConfig(ent_loss_coef=0.1),
        RationalQuadraticSplineQuantileConfig(ent_loss_coef=0.1),
        GumbelSoftmaxSignMagnitudeBetaConfig(ent_loss_coef=0.1),
    ],
)
@pytest.mark.parametrize("kind", list(TMASACActorHeadKind))
@pytest.mark.parametrize("counts", [(4, 4), (4, 1), (1, 4)])
def test_feedforward_multisample_training(
    config: object, kind: TMASACActorHeadKind, counts: tuple[int, int]
) -> None:
    env = _make_env()
    try:
        base = _make_policy(env, continuous_config=config)
        policy = TMASACPolicy(
            env=env, config=replace(base.config, actor_head_config=_actor_head_config(kind))
        )
        algo = _algorithm(
            policy,
            env,
            actor_action_samples=counts[0],
            target_action_samples=counts[1],
            action_sample_strategy="stratified",
        )
        batch = _make_bootstrap_batch(env)
        with patch.object(
            policy, "q_values_with_nop_latents", wraps=policy.q_values_with_nop_latents
        ) as replay_critic:
            metrics, actor_grad, critic_grad = algo._train_step(batch, global_update_idx=1)
        assert replay_critic.call_args.kwargs["actions"] is batch.actions
        assert all(
            torch.isfinite(torch.tensor(value)) for value in (*metrics.values(), actor_grad, critic_grad)
        )
        for params in (policy.actor_parameters(), policy.critic_parameters()):
            gradients = [p.grad for p in params if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
        hp = algo.get_hyper_parameters()
        assert (hp["actor_action_samples"], hp["target_action_samples"]) == counts
        assert hp["action_sample_strategy"] == "stratified"
    finally:
        env.close()


@pytest.mark.parametrize("count", [1, 4])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("compile_modules", [False, True])
def test_recurrent_public_sampler_matches_reset_state_sequence(
        count: int, shared: bool, compile_modules: bool,
) -> None:
    env = _make_env()
    try:
        config = (
            _shared_recurrent_policy_config()
            if shared
            else _policy_config(
                _encoder_config(SLSTMTemporalSequenceModel, SLSTMTemporalSequenceModelConfig(num_heads=2)),
            )
        )
        with patch("torch.compile", side_effect=_make_recording_eager_compile([], use_aot_autograd=True)):
            policy = RecurrentTMASACPolicy(
                env=env,
                config=replace(config, continuous_config=BernsteinQuantileConfig(), compile_modules=compile_modules),
            )
        batch = _make_bootstrap_batch(env)
        agent_mask = torch.ones(batch.actions.shape[:-1], dtype=torch.bool, device=batch.actions.device)
        agent_mask[0, -1] = False
        initial_state = policy.initial_temporal_state(
            batch_size=batch.actions.shape[0], n_agents=env.n_agents,
            device=batch.actions.device, dtype=batch.actions.dtype,
        )
        sampling_kwargs = {
            "local_obs": batch.local_obs,
            "global_obs": batch.global_obs,
            "agent_mask": agent_mask,
            "previous_actions": batch.previous_actions,
            "num_action_samples": count,
            "action_sample_strategy": "stratified",
            "deterministic": False,
            "use_rsample": True,
        }
        torch.manual_seed(642)
        expected_actions, expected_logs, _latents, _state = policy.action_log_prob_sequence(
            **sampling_kwargs, initial_state=initial_state,
        )
        expected_gradients = torch.autograd.grad(
            expected_actions.square().mean() + expected_logs.mean(), policy.actor_parameters(),
        )
        torch.manual_seed(642)
        actions, logs = policy.action_log_prob(**sampling_kwargs)
        actual_gradients = torch.autograd.grad(
            actions.square().mean() + logs.mean(), policy.actor_parameters(),
        )
        expected_shape = tuple(batch.actions.shape) if count == 1 else (count, *batch.actions.shape)
        assert actions.shape == expected_shape
        assert logs.shape == expected_shape[:-1]
        torch.testing.assert_close(actions, expected_actions)
        torch.testing.assert_close(logs, expected_logs)
        for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
            torch.testing.assert_close(actual, expected)
        assert sum(gradient.abs().sum() for gradient in actual_gradients) > 0
        active_mask = agent_mask if count == 1 else agent_mask.expand(count, -1, -1)
        assert not actions[~active_mask].any()
        assert not logs[~active_mask].any()
    finally:
        if compile_modules:
            torch._dynamo.reset()
        env.close()


@pytest.mark.parametrize(
    "config",
    [
        BetaConfig(ent_loss_coef=0.1),
        SquashedDiagGaussianConfig(std=0.5, std_learnable=True, ent_loss_coef=0.1),
        PredictedStdConfig(base_std=0.5, ent_loss_coef=0.1),
        ReparameterizedSquashedGaussianMixtureConfig(ent_loss_coef=0.1),
    ],
)
@pytest.mark.parametrize("counts", [(1, 1), (1, 4), (4, 4)])
@pytest.mark.parametrize("sequence_length", [1, 3])
def test_segment_training_preserves_entropy_loss_masks(
        config: ContinuousActionDistConfig, counts: tuple[int, int], sequence_length: int,
) -> None:
    env = _make_env()
    try:
        policy = SegmentTMASACPolicy(env=env, config=_make_policy(env, continuous_config=config).config)
        algo = RecurrentSAC(
            policy=policy, env=env, burn_in_steps=0, learning_steps=sequence_length,
            temporal_state_store_interval=1, buffer_capacity_per_env=8, learning_starts=0,
            batch_size=2, train_device="cpu", replay_storage_device="cpu",
            actor_action_samples=counts[0], target_action_samples=counts[1],
        )
        batch = _make_segment_batch(
            batch_size=2, sequence_length=sequence_length, n_agents=env.n_agents,
            local_obs_dim=env.local_obs_dim, global_obs_dim=env.global_obs_dim,
            hidden_local_vars_dim=env.hidden_local_vars_dim,
            hidden_global_vars_dim=env.hidden_global_vars_dim,
            action_dim=env.action_space.total_agent_action_dim,
        )
        batch.actions.clamp_(-0.9, 0.9)
        batch.agent_mask[0, :, -1] = False
        batch.next_agent_mask[0, :, -1] = False
        metrics, actor_grad, critic_grad = algo._train_step(batch, global_update_idx=1)
        assert all(torch.isfinite(torch.tensor(value)) for value in (*metrics.values(), actor_grad, critic_grad))
        assert actor_grad > 0 and critic_grad > 0
        entropy_losses = [
            value for name, value in metrics.items()
            if name.startswith("actor_action_dist_") and name.endswith("entropy_loss_scaled")
        ]
        assert entropy_losses and any(value != 0 for value in entropy_losses)
    finally:
        env.close()


def test_segment_length_one_preserves_actor_and_critic_sequence_axes() -> None:
    env = _make_env()
    try:
        policy = SegmentTMASACPolicy(env=env, config=_make_policy(env).config)
        batch = _make_bootstrap_batch(env)
        observations = {
            "local_obs": batch.local_obs,
            "global_obs": batch.global_obs,
            "agent_mask": torch.tensor([[True, False], [True, True]]),
        }
        step_outputs = policy.action_log_prob_sequence(
            **observations, previous_actions=None, deterministic=True, use_rsample=False, initial_state=None,
        )
        sequence_outputs = policy.action_log_prob_sequence(
            **{name: tensor.unsqueeze(1) for name, tensor in observations.items()},
            previous_actions=None, deterministic=True, use_rsample=False, initial_state=None,
        )
        for step, sequence in zip(step_outputs[:3], sequence_outputs[:3], strict=True):
            torch.testing.assert_close(sequence, step.unsqueeze(1))

        critic_inputs = {
            **observations,
            "actions": batch.actions,
            "hidden_local_vars": batch.hidden_local_vars,
            "hidden_global_vars": batch.hidden_global_vars,
        }
        for target in (False, True):
            step_values = policy.q_values_sequence(**critic_inputs, target=target)
            sequence_values = policy.q_values_sequence(
                **{name: tensor.unsqueeze(1) for name, tensor in critic_inputs.items()}, target=target,
            )
            for step, sequence in zip(step_values[:2], sequence_values[:2], strict=True):
                torch.testing.assert_close(sequence, step.unsqueeze(1))
    finally:
        env.close()


def test_clipped_double_q_and_temperature_average_over_candidates() -> None:
    env = _make_env()
    try:
        policy = _make_policy(env, continuous_config=BernsteinQuantileConfig())
        algo = _algorithm(
            policy, env, actor_action_samples=2, target_action_samples=2, gamma=0.5, ent_coef=0.2
        )
        batch = _make_bootstrap_batch(env)
        q1 = torch.tensor([[1.0, 8.0], [7.0, 2.0]])
        q2 = torch.tensor([[9.0, 3.0], [2.0, 6.0]])
        logs = torch.tensor([[-2.0, -1.0], [-4.0, -3.0]])
        expected_soft_value = torch.tensor([1.5, 2.5]) - 0.2 * logs.mean(0)
        assert not torch.allclose(torch.minimum(q1.mean(0), q2.mean(0)), torch.minimum(q1, q2).mean(0))
        torch.testing.assert_close(_actor_loss(q1, q2, logs, torch.tensor(0.2)), -expected_soft_value.mean())
        actions = torch.zeros(2, *batch.actions.shape)
        agent_logs = logs[..., None].expand(2, 2, env.n_agents)
        with (
            patch.object(policy, "action_log_prob", return_value=(actions, agent_logs)),
            patch.object(policy, "target_q_values", return_value=(q1.flatten(), q2.flatten())),
        ):
            target = algo._target_forward_phase(
                batch,
                batch.next_local_obs,
                batch.next_global_obs,
                batch.next_hidden_local_vars,
                batch.next_hidden_global_vars,
                batch.next_agent_mask,
                torch.tensor(0.2),
            )
        expected = torch.where(batch.terminal_mask, batch.rewards, batch.rewards + 0.5 * expected_soft_value)
        torch.testing.assert_close(target, expected)
    finally:
        env.close()


@pytest.mark.parametrize(
    "config",
    [
        BernsteinQuantileConfig(ent_loss_coef=0.1),
        RationalQuadraticSplineQuantileConfig(ent_loss_coef=0.1),
        GumbelSoftmaxSignMagnitudeBetaConfig(ent_loss_coef=0.1),
    ],
)
@pytest.mark.parametrize("variant", ["independent", "qcx", "recurrent_critic", "shared", "actor_state"])
@pytest.mark.parametrize("counts", [(4, 4), (4, 1), (1, 4)])
def test_recurrent_multisample_training(config: object, variant: str, counts: tuple[int, int]) -> None:
    env = _make_env()
    try:
        encoder = _encoder_config(SLSTMTemporalSequenceModel, SLSTMTemporalSequenceModelConfig(num_heads=2))
        policy_config = (
            replace(_shared_recurrent_policy_config(), continuous_config=config)
            if variant == "shared"
            else _policy_config(
                encoder,
                continuous_config=config,
                recurrent_critic=variant == "recurrent_critic",
                actor_head_kind=TMASACActorHeadKind.QCX
                if variant == "qcx"
                else TMASACActorHeadKind.INDEPENDENT,
                nop_config=_small_nop_config(),
            )
        )
        if variant == "actor_state":
            policy_config = replace(
                policy_config, actor_state_critic_input_config=ActorStateCriticInputConfig(projection_dim=4)
            )
        policy = RecurrentTMASACPolicy(env=env, config=policy_config)
        algo = RecurrentSAC(
            policy=policy,
            env=env,
            burn_in_steps=1,
            learning_steps=3,
            temporal_state_store_interval=1,
            buffer_capacity_per_env=8,
            learning_starts=0,
            batch_size=2,
            train_device="cpu",
            replay_storage_device="cpu",
            actor_action_samples=counts[0],
            target_action_samples=counts[1],
            action_sample_strategy="stratified",
        )
        batch = _make_segment_batch(
            batch_size=2,
            sequence_length=4,
            n_agents=env.n_agents,
            local_obs_dim=env.local_obs_dim,
            global_obs_dim=env.global_obs_dim,
            hidden_local_vars_dim=env.hidden_local_vars_dim,
            hidden_global_vars_dim=env.hidden_global_vars_dim,
            action_dim=env.action_space.total_agent_action_dim,
        )
        batch.actions.mul_(0.1)
        batch.truncations[0, 2] = True
        batch.episode_start_mask[0, 3] = True
        batch.next_local_obs[0, 2].fill_(7)
        batch.terminations[1, -1] = True
        batch.agent_mask[0, :, -1] = False
        batch.next_agent_mask[0, :, -1] = False
        metrics, actor_grad, critic_grad = algo._train_step(batch, global_update_idx=1)
        assert all(
            torch.isfinite(torch.tensor(value)) for value in (*metrics.values(), actor_grad, critic_grad)
        )
        for params in (policy.actor_parameters(), policy.critic_parameters()):
            gradients = [p.grad for p in params if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
    finally:
        env.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"actor_action_samples": 0},
        {"target_action_samples": -1},
        {"actor_action_samples": 1.5},
        {"action_sample_strategy": "sobol"},
    ],
)
def test_invalid_multisample_configuration(kwargs: dict[str, object]) -> None:
    env = _make_env()
    try:
        with pytest.raises(ValueError):
            _algorithm(_make_policy(env), env, **kwargs)
    finally:
        env.close()


@pytest.mark.parametrize("target", [False, True])
def test_recurrent_candidates_branch_from_executed_action_history(target: bool) -> None:
    env = _make_env()
    try:
        policy, algo = _make_recurrent_critic_algorithm(env)
        batch = _make_recurrent_critic_batch(env)
        batch.episode_start_mask.zero_()
        candidates = torch.stack([batch.actions + offset for offset in (10, 20, 30, 40)])
        initial_state = torch.full((batch.actions.shape[0],), 3.0)

        def accumulate_actions(
            *, actions: torch.Tensor, initial_state: torch.Tensor, **kwargs: object
        ) -> tuple:
            next_state = initial_state + actions.sum((-1, -2))
            return next_state, next_state + 1, None, next_state

        with patch.object(policy, "q_values_sequence", side_effect=accumulate_actions) as critic:
            if target:
                q1, q2 = algo._target_next_q_values(
                    batch=batch, next_actions=candidates, initial_state=initial_state, actor_state=None
                )
            else:
                q1, q2 = algo._actor_q_values(
                    batch=batch, actions_pi=candidates, initial_state=initial_state, actor_state=None
                )
        for time_idx in range(batch.sequence_length):
            assert sum(
                torch.equal(call.kwargs["actions"], batch.actions[:, time_idx])
                for call in critic.call_args_list
            ) == 1
        executed = batch.actions.sum((-1, -2))
        replay_history = executed.cumsum(1)
        if not target:
            replay_history = replay_history - executed
        expected = initial_state[:, None] + replay_history + candidates.sum((-1, -2))
        torch.testing.assert_close(q1, expected)
        torch.testing.assert_close(q2, expected + 1)
        torch.testing.assert_close(initial_state, torch.full_like(initial_state, 3.0))
    finally:
        env.close()


@pytest.mark.parametrize("count", [1, 4])
def test_auxiliary_loss_and_temperature_are_sample_count_invariant(count: int) -> None:
    env = _make_env()
    try:
        policy = _make_policy(env, continuous_config=BernsteinQuantileConfig(ent_loss_coef=0.1))
        algo = _algorithm(policy, env, actor_action_samples=count)
        batch = _make_bootstrap_batch(env)
        batch = replace(batch, agent_mask=torch.tensor([[True, False], [True, True]]))
        per_agent_loss = torch.tensor([[2.0, 100.0], [4.0, 6.0]])
        losses = per_agent_loss if count == 1 else per_agent_loss.expand(count, -1, -1)
        with patch.object(
            policy.action_dist, "compute_extra_losses_without_metrics", return_value={"entropy": losses}
        ):
            extra, _ = algo._compute_actor_action_dist_extra_losses(agent_mask=batch.agent_mask)
        reduced = algo._reduce_actor_action_dist_extra_losses(batch=batch, extra_losses=extra)
        torch.testing.assert_close(reduced["entropy"], torch.tensor(6.0))
        logs = torch.tensor([-2.0, -4.0])
        samples = logs if count == 1 else logs.expand(count, -1)
        alpha_loss = _entropy_coefficient_loss(
            torch.tensor(0.5), algo._mean_action_samples(samples, count), torch.tensor(-1.0)
        )
        torch.testing.assert_close(alpha_loss, torch.tensor(2.0))
    finally:
        env.close()


@pytest.mark.parametrize("kind", list(TMASACActorHeadKind))
def test_actor_encodes_once_and_preserves_sample_axis(kind: TMASACActorHeadKind) -> None:
    env = _make_env()
    try:
        config = replace(
            _make_policy(env).config,
            continuous_config=RationalQuadraticSplineQuantileConfig(),
            actor_head_config=_actor_head_config(kind),
        )
        policy = TMASACPolicy(env=env, config=config)
        batch = _make_bootstrap_batch(env)
        with patch.object(policy, "encode_actor", wraps=policy.encode_actor) as encoder:
            actions, logs = policy.action_log_prob(
                local_obs=batch.local_obs,
                global_obs=batch.global_obs,
                agent_mask=batch.agent_mask,
                num_action_samples=4,
                action_sample_strategy="stratified",
            )
        assert encoder.call_count == 1
        assert actions.shape == (4, *batch.actions.shape)
        assert logs.shape == (4, *batch.actions.shape[:-1])
        # Uniform initialization makes every scalar action's base stratum observable.
        strata = ((actions + 1) * 2).floor().sort(0).values
        torch.testing.assert_close(strata, torch.arange(4)[:, None, None, None].expand_as(strata).float())
    finally:
        env.close()


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("counts", [(4, 4), (4, 1), (1, 4)])
def test_multisample_recurrent_training_with_compiled_encoders(shared: bool, counts: tuple[int, int]) -> None:
    env = _make_env()
    try:
        config = (
            _shared_recurrent_policy_config()
            if shared
            else _policy_config(
                _encoder_config(SLSTMTemporalSequenceModel, SLSTMTemporalSequenceModelConfig(num_heads=2)),
            )
        )
        with patch("torch.compile", side_effect=_make_recording_eager_compile([], use_aot_autograd=True)):
            policy = RecurrentTMASACPolicy(
                env=env,
                config=replace(
                    config, continuous_config=RationalQuadraticSplineQuantileConfig(), compile_modules=True
                ),
            )
            algo = RecurrentSAC(
                policy=policy,
                env=env,
                burn_in_steps=1,
                learning_steps=3,
                temporal_state_store_interval=1,
                buffer_capacity_per_env=8,
                learning_starts=0,
                batch_size=2,
                train_device="cpu",
                replay_storage_device="cpu",
                actor_action_samples=counts[0],
                target_action_samples=counts[1],
                action_sample_strategy="stratified",
            )
        batch = _make_segment_batch(
            batch_size=2,
            sequence_length=4,
            n_agents=env.n_agents,
            local_obs_dim=env.local_obs_dim,
            global_obs_dim=env.global_obs_dim,
            hidden_local_vars_dim=env.hidden_local_vars_dim,
            hidden_global_vars_dim=env.hidden_global_vars_dim,
            action_dim=env.action_space.total_agent_action_dim,
        )
        metrics, actor_grad, critic_grad = algo._train_step(batch, global_update_idx=1)
        assert all(torch.isfinite(torch.tensor(v)) for v in (*metrics.values(), actor_grad, critic_grad))
    finally:
        torch._dynamo.reset()
        env.close()


@pytest.mark.parametrize("config", [BernsteinQuantileConfig(), RationalQuadraticSplineQuantileConfig()])
@pytest.mark.parametrize("kind", [TMASACActorHeadKind.INDEPENDENT, TMASACActorHeadKind.QCX])
def test_multisample_actor_compiles_sampling_and_matches_eager(config: object, kind: TMASACActorHeadKind) -> None:
    env = _make_env()
    try:
        policy_config = replace(_make_policy(env).config, continuous_config=config, actor_head_config=_actor_head_config(kind))
        eager = TMASACPolicy(env=env, config=policy_config)
        with torch.no_grad():
            for dist in eager.action_dist.distributions:
                dist.action_net.weight.normal_(std=0.05)
        graphs: list[torch.fx.GraphModule] = []
        with patch("torch.compile", side_effect=_make_recording_eager_compile(graphs)):
            compiled = TMASACPolicy(env=env, config=replace(policy_config, compile_modules=True))
        compiled.load_state_dict(align_torch_compile_state_dict_keys(
            eager.state_dict(), target_keys=compiled.state_dict(),
        ))
        batch = _make_bootstrap_batch(env)
        outputs = []
        for policy in (eager, compiled):
            torch.manual_seed(617)
            actions, logs = policy.action_log_prob(
                local_obs=batch.local_obs,
                global_obs=batch.global_obs,
                agent_mask=batch.agent_mask,
                num_action_samples=4,
                action_sample_strategy="stratified",
            )
            (actions.square().mean() + logs.mean()).backward()
            outputs.append((actions, logs))
        for actual, expected in zip(outputs[1], outputs[0], strict=True):
            torch.testing.assert_close(actual, expected)
        for actual, expected in zip(compiled.actor_parameters(), eager.actor_parameters(), strict=True):
            torch.testing.assert_close(actual.grad, expected.grad)
        assert any("argsort" in str(node.target) for graph in graphs for node in graph.graph.nodes)
    finally:
        torch._dynamo.reset()
        env.close()


@pytest.mark.parametrize(
    "variant", ["segment", "stateless", "actor_state", "shared", "recurrent", "recurrent_lstm", "recurrent_twin"],
)
@pytest.mark.parametrize("target", [False, True])
@pytest.mark.parametrize("sequence_api", [False, True])
def test_candidate_critics_match_separate_values_and_gradients(
        variant: str, target: bool, sequence_api: bool,
) -> None:
    env = _make_env()
    try:
        encoder = (
            _encoder_config(LSTMTemporalSequenceModel, LSTMTemporalSequenceModelConfig())
            if variant == "recurrent_lstm"
            else _encoder_config(SLSTMTemporalSequenceModel, SLSTMTemporalSequenceModelConfig(num_heads=2))
        )
        if variant == "segment":
            policy = SegmentTMASACPolicy(env=env, config=_make_policy(env).config)
        else:
            config = (
                _shared_recurrent_policy_config()
                if variant == "shared"
                else _policy_config(encoder, recurrent_critic=variant.startswith("recurrent"))
            )
            if variant == "actor_state":
                config = replace(config, actor_state_critic_input_config=ActorStateCriticInputConfig(projection_dim=4))
            elif variant == "recurrent_twin":
                config = replace(config, critic_config=replace(config.critic_config, independent_encoders=True))
            policy = RecurrentTMASACPolicy(env=env, config=config)
        algo = RecurrentSAC(
            policy=policy, env=env, burn_in_steps=0, learning_steps=3,
            temporal_state_store_interval=1, buffer_capacity_per_env=8,
            learning_starts=0, batch_size=2, train_device="cpu", replay_storage_device="cpu",
        )
        batch = _make_recurrent_critic_batch(env)
        batch.episode_start_mask[0, 1] = True
        batch.agent_mask[0, :, -1] = False
        batch.next_agent_mask[0, :, -1] = False
        batch.train_mask[1, -1] = False
        candidates = torch.randn(3, *batch.actions.shape, requires_grad=True)
        critic_inputs = {
            name: getattr(batch, name) for name in (
                "local_obs", "global_obs", "hidden_local_vars", "hidden_global_vars", "agent_mask", "scenario_ids",
            )
        }
        critic_state = None
        if algo._critic_uses_temporal_state:
            with torch.no_grad():
                _q1, _q2, _latents, critic_state = policy.q_values_sequence(
                    **critic_inputs, actions=batch.actions, target=target,
                )
        original_critic_state = clone_temporal_state(critic_state)
        actor_state = None
        if variant == "actor_state":
            state = policy.initial_temporal_state(
                batch_size=2, n_agents=env.n_agents, device=torch.device("cpu"), dtype=torch.float32,
            )
            actor_state = policy.actor_state_critic_input(state).unsqueeze(1).expand(-1, 3, -1, -1)

        def evaluate(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if sequence_api:
                count = actions.shape[0] if actions.ndim == batch.actions.ndim + 1 else 1
                q1, q2, latents, next_state = policy.q_values_sequence(
                    **critic_inputs,
                    actions=actions,
                    initial_state=critic_state,
                    time_mask=batch.train_mask,
                    reset_mask=batch.episode_start_mask,
                    target=target,
                    num_action_samples=count,
                    **({} if actor_state is None else {"actor_state": actor_state}),
                )
                if count > 1:
                    assert latents is None
                    if variant != "shared":
                        assert next_state is None
                return q1, q2
            if target:
                return algo._target_next_q_values(
                    batch=batch, next_actions=actions, initial_state=critic_state, actor_state=actor_state,
                )
            return algo._actor_q_values(
                batch=batch, actions_pi=actions, initial_state=critic_state, actor_state=actor_state,
            )

        critic_module = policy.critic_target if target else policy.critic
        with patch.object(critic_module, "forward", wraps=critic_module.forward) as critic:
            actual = evaluate(candidates)
        if sequence_api:
            assert critic.call_count == 1
        elif variant == "shared":
            assert critic.call_count == batch.sequence_length * (2 if target else 1)
        elif variant.startswith("recurrent"):
            assert critic.call_count == batch.sequence_length * 2
        else:
            assert critic.call_count == 1
        separate = [evaluate(candidate) for candidate in candidates.unbind(0)]
        expected = tuple(torch.stack(values) for values in zip(*separate, strict=True))
        for actual_q, expected_q in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_q, expected_q)
        actual_grad = torch.autograd.grad(sum(q.sum() for q in actual), candidates)[0]
        expected_grad = torch.autograd.grad(sum(q.sum() for q in expected), candidates)[0]
        torch.testing.assert_close(actual_grad, expected_grad)
        torch.testing.assert_close(critic_state, original_critic_state)
    finally:
        env.close()


@pytest.mark.parametrize("config", [BernsteinQuantileConfig(), RationalQuadraticSplineQuantileConfig()])
def test_single_sample_strategy_preserves_seeded_baseline(config: object) -> None:
    env = _make_env()
    try:
        policy = _make_policy(env, continuous_config=config)
        reference = _make_policy(env, continuous_config=config)
        reference.load_state_dict(policy.state_dict())
        baseline = _algorithm(reference, env)
        explicit = _algorithm(
            policy, env, actor_action_samples=1, target_action_samples=1, action_sample_strategy="stratified"
        )
        batch = _make_bootstrap_batch(env)
        torch.manual_seed(812)
        expected = baseline._train_step(batch, global_update_idx=1)
        torch.manual_seed(812)
        actual = explicit._train_step(batch, global_update_idx=1)
        assert actual == expected
        for actual_parameter, expected_parameter in zip(
            policy.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(actual_parameter, expected_parameter, rtol=0, atol=0)
    finally:
        env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("config", [BernsteinQuantileConfig(), RationalQuadraticSplineQuantileConfig()])
def test_compiled_cuda_multisample_training(config: object) -> None:
    env = _make_env()
    try:
        policy = _make_policy(env, continuous_config=config, compile_modules=True)
        algo = SAC(
            policy=policy,
            env=env,
            buffer_capacity_per_env=8,
            learning_starts=0,
            batch_size=2,
            train_device="cuda",
            replay_storage_device="cpu",
            actor_action_samples=4,
            target_action_samples=4,
            action_sample_strategy="stratified",
        )
        batch = _move_replay_batch(_make_bootstrap_batch(env), "cuda")
        metrics, actor_grad, critic_grad = algo._train_step(batch, global_update_idx=1)
        assert all(
            torch.isfinite(torch.tensor(value)) for value in (*metrics.values(), actor_grad, critic_grad)
        )
    finally:
        torch._dynamo.reset()
        env.close()


def test_recurrent_target_stratifies_final_and_truncation_successors() -> None:
    env = _make_env()
    try:
        encoder = _encoder_config(SLSTMTemporalSequenceModel, SLSTMTemporalSequenceModelConfig(num_heads=2))
        policy = RecurrentTMASACPolicy(
            env=env, config=_policy_config(encoder, continuous_config=BernsteinQuantileConfig())
        )
        algo = RecurrentSAC(
            policy=policy,
            env=env,
            burn_in_steps=0,
            learning_steps=3,
            temporal_state_store_interval=1,
            buffer_capacity_per_env=8,
            learning_starts=0,
            batch_size=2,
            train_device="cpu",
            replay_storage_device="cpu",
            target_action_samples=4,
            action_sample_strategy="stratified",
        )
        batch = _make_recurrent_critic_batch(env)
        latents = torch.randn(2, 3, env.n_agents, 8)
        special_latents = torch.randn(4, env.n_agents, 8)
        indices = torch.tensor([[0, 1], [1, 0]])
        mask = torch.tensor([True, False])
        expected_latents = torch.stack((latents[:, 1], latents[:, 2], special_latents[:2]), dim=1)
        expected_latents[0, 1] = special_latents[2]
        with (
            patch.object(
                policy, "encode_actor_sequence", return_value=(special_latents, None)
            ) as encoder_call,
            patch.object(
                policy,
                "action_log_prob_from_latents",
                wraps=policy.action_log_prob_from_latents,
            ) as sampler,
        ):
            actions, logs = algo._next_policy_actions(
                batch=batch,
                actions_pi=batch.actions,
                log_prob_pi=batch.actions[..., 0],
                actor_latents=latents,
                next_actor_state=torch.zeros(2, 1),
                truncation_actor_states=torch.zeros(2, 1),
                truncation_indices=indices,
                truncation_mask=mask,
            )
        assert encoder_call.call_count == 1
        assert sampler.call_args.kwargs["use_rsample"] is False
        torch.testing.assert_close(sampler.call_args.kwargs["actor_latents"], expected_latents)
        assert actions.shape == (4, *batch.actions.shape)
        assert logs.shape == actions.shape[:-1]
        strata = ((actions + 1) * 2).floor().sort(0).values
        torch.testing.assert_close(
            strata, torch.arange(4)[:, None, None, None, None].expand_as(strata).float()
        )
    finally:
        env.close()
