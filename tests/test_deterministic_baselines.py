from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from torch import nn

from swarmbots.learn.algos.off_policy import collect_off_policy_steps
from swarmbots.learn.algos.off_policy.joint_critic import JointCriticConfig, JointQNetwork
from swarmbots.learn.algos.sac import SAC, MASACPolicy
from swarmbots.learn.algos.sac.sac_nop import SACNOPConfig
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig
from swarmbots.learn.algos.sac.tmasac_actor_heads import TMASACActorHeadConfig, TMASACActorHeadKind
from swarmbots.learn.algos.td3 import DDPG, TD3, TD3Policy
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.hybrid_action_space import HybridActionSpace, VectorHybridActionSpace
from swarmbots.learn.nn_components.feed_forward import MLPConfig
from swarmbots.learn.obs_indices import ObsIndices
from swarmbots.learn.presets.policy_factory import _make_base_policy
from swarmbots.learn.presets.transformer import MATInitGains, MATNormalizationConfig
from swarmbots.learn.testing_env import TestingSwarmBotsEnv as SwarmTestEnv
from swarmbots.learn.training import _variant_options


VARIANTS = (
    "maddpg_mlp",
    "maddpg_deepset",
    "matd3_mlp",
    "matd3_deepset",
    "masac_mlp",
    "masac_deepset",
    "tmatd3",
    "tmatd3_dec",
)

DETERMINISTIC_VARIANTS = tuple(variant for variant in VARIANTS if not variant.startswith("masac"))


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def env():
    env = SwarmBotsLearnEnvWrapper(
        SyncVectorEnv(
            [
                lambda: SwarmTestEnv(
                    n_agents=3,
                    n_local_obs=4,
                    n_global_obs=2,
                    actuators_dim=1,
                    connectors_dim=1,
                    n_hidden_local_vars=1,
                    n_hidden_global_vars=1,
                    max_steps=3,
                    continuous_connector_actions=True,
                ),
            ],
            autoreset_mode=AutoresetMode.SAME_STEP,
        )
    )
    yield env
    env.close()


def make_policy(env, variant, *, small=True, compile_modules=False, **overrides):
    options, _ = _variant_options(variant)
    options["use_nop"] = False
    if small:
        options.update(
            mat_encoder_transformer_ff_config=MLPConfig(hidden_dims=[32, 32]),
            baseline_critic_hidden_dims=(32, 32),
            baseline_critic_element_hidden_dims=(32, 32),
        )
    options.update(overrides)
    return _make_base_policy(
        env=env,
        enc_d_model=16 if small else 256,
        enc_nhead=4,
        dec_d_model=16 if small else 128,
        dec_nhead=2,
        use_popart=False,
        popart_beta=5e-4,
        popart_init_sigma=0.65,
        compile_policy_modules=compile_modules,
        policy_compile_mode="default",
        gsde_init_stds=[0.25],
        mat_add_agent_embeddings=False,
        act_fn_cls=nn.GELU,
        mat_init_gains=MATInitGains(),
        mat_normalization=MATNormalizationConfig(),
        **options,
    )


def make_algorithm(env, variant, *, device="cpu", compile_modules=False, policy_options=None, **overrides):
    policy = make_policy(env, variant, compile_modules=compile_modules, **(policy_options or {}))
    cls = DDPG if variant.startswith("maddpg") else SAC if variant.startswith("masac") else TD3
    options = dict(
        learning_starts=0,
        buffer_capacity_per_env=16,
        batch_size=4,
        rollout_steps_per_iteration=4,
        learning_rate_warmup_updates=0,
        train_device=device,
        rollout_device="cpu",
        replay_storage_device="cpu",
        sac_compile_tensor_operations=None if device == "cuda" else False,
        sac_compile_optimizer_steps=None if device == "cuda" else False,
    )
    options.update(overrides)
    return cls(policy, env, **options)


def fill_replay(algorithm):
    collect_off_policy_steps(algorithm.env, algorithm.replay_buffer, n_steps=8, random_actions=True)
    return algorithm.replay_buffer.sample(4)


NOP_VARIANTS = ("maddpg_deepset", "matd3_deepset", "masac_deepset", "tmatd3", "tmatd3_dec")


def nop_options(source="critic", steps=3, **overrides):
    indices = ObsIndices(
        local_scalar_indices=[0], local_angle_indices=[], local_rot6d_indices=[], local_binary_indices=[],
        local_quaternion_indices=[], global_scalar_indices=[0], global_rot6d_indices=[],
        global_quaternion_indices=[], hidden_local_vars_scalar_indices=[0], hidden_local_vars_quaternion_indices=[],
        hidden_global_vars_scalar_indices=[0], hidden_global_vars_quaternion_indices=[],
    )
    return dict(
        use_nop=True, obs_indices=indices, tmasac_nop_latent_source=source, world_model_num_next_steps=steps,
        transition_model_d_model=8, transition_model_nhead=2, **overrides,
    )


def observations(env, batch=2):
    return dict(
        local_obs=torch.randn(batch, env.n_agents, env.local_obs_dim),
        global_obs=torch.randn(batch, env.global_obs_dim),
        hidden_local_vars=torch.randn(batch, env.n_agents, env.hidden_local_vars_dim),
        hidden_global_vars=torch.randn(batch, env.hidden_global_vars_dim),
        agent_mask=torch.tensor([[True, False, True]]).expand(batch, -1),
    )


def assert_optimizer_states_equal(actual, expected):
    actual_state, expected_state = actual._get_optimizer_state_dict(), expected._get_optimizer_state_dict()
    assert actual_state.keys() == expected_state.keys()
    for key, value in expected_state.items():
        if isinstance(value, str):
            assert actual_state[key] == value
        else:
            torch.testing.assert_close(actual_state[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("variant", VARIANTS)
def test_baseline_rollout_update_and_privileged_boundary(env, variant):
    torch.manual_seed(42)
    algorithm = make_algorithm(env, variant)
    policy = algorithm.policy
    obs = observations(env)
    actions = policy.act(**obs, deterministic=True)
    assert torch.isfinite(actions).all() and (actions.abs() <= 1).all()
    assert (actions[:, 1] == 0).all()
    privileged_changed = {
        **obs,
        "hidden_local_vars": obs["hidden_local_vars"] + 100,
        "hidden_global_vars": obs["hidden_global_vars"] - 100,
    }
    torch.testing.assert_close(actions, policy.act(**privileged_changed, deterministic=True), rtol=0, atol=0)
    if variant != "tmatd3":
        changed = {**obs, "local_obs": obs["local_obs"].clone()}
        changed["local_obs"][:, 2] += 100
        torch.testing.assert_close(actions[:, 0], policy.act(**changed, deterministic=True)[:, 0], rtol=0, atol=0)
    masked_changed = {**obs}
    for key in ("local_obs", "hidden_local_vars"):
        masked_changed[key] = obs[key].clone()
        masked_changed[key][:, 1] = 10000
    masked_actions = actions.clone()
    masked_actions[:, 1] = -9999
    q_before = policy.q_values(actions=actions, **obs)
    q_after = policy.q_values(actions=masked_actions, **masked_changed)
    for before, after in zip(q_before, q_after, strict=True):
        torch.testing.assert_close(before, after)
    fill_replay(algorithm)
    before = [p.detach().clone() for p in policy.actor_parameters()]
    metrics = algorithm.train(gradient_steps=2)
    assert metrics["updates"] == 2
    assert any(not torch.equal(a, b) for a, b in zip(before, policy.actor_parameters(), strict=True))
    assert all(torch.isfinite(p).all() for p in policy.parameters())
    assert all(p.grad is None and not p.requires_grad for p in policy.critic_target.parameters())
    if isinstance(algorithm, TD3):
        assert all(p.grad is None and not p.requires_grad for p in policy.actor_target.parameters())
        assert algorithm.ent_coef_optimizer is None and algorithm.ent_coef_tensor is None
        assert not hasattr(policy, "action_dist")


@pytest.mark.parametrize("kind", ["mlp", "deepset"])
def test_joint_critics_mask_nan_padding_and_include_presence_bits(kind):
    critic = JointQNetwork(
        n_agents=3,
        local_obs_dim=2,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
        action_dim=1,
        config=JointCriticConfig(kind, (16, 16), (16, 16)),
    )
    obs = dict(
        local_obs=torch.zeros(2, 3, 2),
        global_obs=torch.zeros(2, 1),
        hidden_local_vars=torch.zeros(2, 3, 1),
        hidden_global_vars=torch.zeros(2, 1),
        actions=torch.zeros(2, 3, 1),
        agent_mask=torch.tensor([[True, False, True], [False, False, False]]),
    )
    baseline = critic(**obs)
    for key in ("local_obs", "hidden_local_vars", "actions"):
        obs[key][~obs["agent_mask"]] = float("nan")
    torch.testing.assert_close(critic(**obs), baseline)
    assert torch.isfinite(baseline).all()
    obs["agent_mask"] = torch.zeros_like(obs["agent_mask"])
    q_empty = critic(**obs)
    assert not torch.equal(q_empty[0], baseline[0])


@pytest.mark.parametrize("variant", VARIANTS)
def test_critic_gradients_ignore_nan_padding_and_reach_all_active_actions(env, variant):
    torch.manual_seed(42)
    policy = make_policy(env, variant)
    obs = observations(env)
    obs["actions"] = torch.randn(2, env.n_agents, env.action_space.total_agent_action_dim)
    mask = obs["agent_mask"]
    for key in ("local_obs", "hidden_local_vars", "actions"):
        obs[key][~mask] = float("nan")
        obs[key].requires_grad_()

    sum(q.sum() for q in policy.q_values(**obs)).backward()

    for key in ("local_obs", "hidden_local_vars", "actions"):
        gradient = obs[key].grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert (gradient[~mask] == 0).all()
        assert (gradient[mask].abs().sum(-1) > 0).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.critic_parameters())
    assert all(p.grad is None for p in policy.actor_parameters())
    assert all(p.grad is None for p in policy.critic_target.parameters())


@pytest.mark.parametrize("variant", ["maddpg_deepset", "matd3_deepset", "masac_deepset", "tmatd3"])
def test_set_and_transformer_critics_are_permutation_invariant(env, variant):
    policy = make_policy(env, variant).eval()
    obs = observations(env)
    actions = policy.act(**obs, deterministic=True)
    expected = policy.q_values(**obs, actions=actions)
    perm = torch.tensor([2, 0, 1])
    shuffled = {k: v[:, perm] if k in {"local_obs", "hidden_local_vars", "agent_mask"} else v for k, v in obs.items()}
    actual = policy.q_values(**shuffled, actions=actions[:, perm])
    for q1, q2 in zip(expected, actual, strict=True):
        torch.testing.assert_close(q1, q2, atol=1e-6, rtol=1e-5)


def test_deepset_joint_critic_retains_cardinality_and_privileged_context():
    critic = JointQNetwork(
        n_agents=3, local_obs_dim=1, global_obs_dim=1, hidden_local_vars_dim=1,
        hidden_global_vars_dim=1, action_dim=1, config=JointCriticConfig("deepset", (4,), (1,)),
    )
    encoder = nn.Linear(6, 1, bias=False)
    decoder = nn.Linear(4, 1, bias=False)
    with torch.no_grad():
        encoder.weight.copy_(torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]))
        decoder.weight.copy_(torch.tensor([[1.0, 1.0, 1.0, 100.0]]))
    critic.network.deepset.element_encoder = encoder
    critic.network.deepset.set_decoder = decoder
    mask = torch.tensor([[True, False, False], [True, True, False], [False, False, False]])
    obs = dict(
        local_obs=torch.full((3, 3, 1), 2.0), actions=torch.full((3, 3, 1), 3.0),
        hidden_local_vars=torch.full((3, 3, 1), 5.0), global_obs=torch.full((3, 1), 7.0),
        hidden_global_vars=torch.full((3, 1), 11.0), agent_mask=mask,
    )
    for key in ("local_obs", "actions", "hidden_local_vars"):
        obs[key][~mask] = float("nan")

    values, latents = critic.forward_with_latents(**obs)

    torch.testing.assert_close(values, torch.tensor([128.0, 228.0, 18.0]))
    torch.testing.assert_close(
        latents.squeeze(-1), torch.tensor([[10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 0.0, 0.0]]),
    )


@pytest.mark.parametrize("variant", ["maddpg_deepset", "matd3_deepset", "masac_deepset"])
@pytest.mark.parametrize("context_in_elements", [False, True])
def test_deepset_nop_latents_condition_on_globals_and_match_encoding_paths(env, variant, context_in_elements):
    torch.manual_seed(42)
    policy = make_policy(
        env, variant, **nop_options(), baseline_critic_context_in_elements=context_in_elements,
    )
    config_key = "td3_policy_config" if isinstance(policy, TD3Policy) else "masac_policy_config"
    assert policy.get_hyper_parameters()[config_key]["joint_critic_config"]["context_in_elements"] is context_in_elements
    obs = observations(env)
    obs["agent_mask"] = obs["agent_mask"].clone()
    obs["agent_mask"][1] = False
    for key in ("local_obs", "hidden_local_vars"):
        obs[key][~obs["agent_mask"]] = float("nan")
    obs["actions"] = torch.randn(2, 3, env.action_space.total_agent_action_dim)
    obs["actions"][~obs["agent_mask"]] = float("nan")
    obs["global_obs"].requires_grad_()
    obs["hidden_global_vars"].requires_grad_()
    result = policy.q_values_with_nop_latents(**obs)
    latents = result[1] if isinstance(policy, TD3Policy) else result[2]
    networks = list(policy.critic) if isinstance(policy, TD3Policy) else [policy.critic.q1, policy.critic.q2]
    torch.testing.assert_close(torch.cat([network.encode(**obs) for network in networks], dim=-1), latents)
    assert torch.isfinite(latents).all() and (latents[~obs["agent_mask"]] == 0).all()
    assert all(network.network.context_after_pool for network in networks)

    for field in ("global_obs", "hidden_global_vars"):
        changed = {**obs, field: obs[field] + 10}
        changed_result = policy.q_values_with_nop_latents(**changed)
        changed_latents = changed_result[1] if isinstance(policy, TD3Policy) else changed_result[2]
        if context_in_elements:
            assert not torch.allclose(changed_latents[obs["agent_mask"]], latents[obs["agent_mask"]])
        else:
            torch.testing.assert_close(changed_latents, latents, rtol=0, atol=0)
    gradients = torch.autograd.grad(
        latents.square().sum(), (obs["global_obs"], obs["hidden_global_vars"]), allow_unused=True,
    )
    for gradient in gradients:
        if context_in_elements:
            assert gradient is not None and torch.isfinite(gradient).all()
            assert gradient[0].abs().sum() > 0 and (gradient[1] == 0).all()
        else:
            assert gradient is None


@pytest.mark.parametrize("variant", ["maddpg_deepset", "matd3_deepset", "masac_deepset"])
@pytest.mark.parametrize("independent", [False, True])
def test_deepset_nop_reused_and_reencoded_global_context_produce_the_same_loss(env, variant, independent):
    torch.manual_seed(42)
    algorithm = make_algorithm(env, variant, policy_options=nop_options(), independent_nop_sampling=independent)
    fill_replay(algorithm)
    _, nop_batch, reuse = algorithm._sample_training_batches()
    assert reuse is not independent
    origin = nop_batch.origin_batch
    obs = dict(
        local_obs=origin.local_obs, global_obs=origin.global_obs, actions=origin.actions,
        hidden_local_vars=origin.hidden_local_vars, hidden_global_vars=origin.hidden_global_vars,
        agent_mask=origin.agent_mask,
    )
    result = algorithm.policy.q_values_with_nop_latents(**obs)
    latents = result[1] if isinstance(algorithm, TD3) else result[2]
    reused_loss, _ = algorithm.policy.compute_critic_nop_loss(nop_batch, source_latents=latents)
    reencoded_loss, _ = algorithm.policy.compute_critic_nop_loss(nop_batch)
    torch.testing.assert_close(reencoded_loss, reused_loss)
    parameters = algorithm.policy.critic_parameters()
    expected_gradients = torch.autograd.grad(reused_loss, parameters, allow_unused=True)
    actual_gradients = torch.autograd.grad(reencoded_loss, parameters, allow_unused=True)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("variant", ["maddpg_deepset", "matd3_deepset", "masac_deepset"])
def test_legacy_deepset_checkpoint_loads_with_late_context_option(env, variant, tmp_path):
    algorithm = make_algorithm(env, variant, policy_options={"baseline_critic_context_in_elements": False})
    fill_replay(algorithm)
    algorithm.train(gradient_steps=2)
    path = tmp_path / "late-context.pt"
    algorithm.save(path, optimizer_state_dict=algorithm._get_optimizer_state_dict())
    checkpoint = torch.load(path, weights_only=False)
    config_key = "td3_policy_config" if isinstance(algorithm, TD3) else "masac_policy_config"
    critic_config = checkpoint["policy_hyper_parameters"][config_key]["joint_critic_config"]
    assert critic_config.pop("context_in_elements") is False
    torch.save(checkpoint, path)
    restored = make_algorithm(env, variant, policy_options={"baseline_critic_context_in_elements": False})
    restored.load(path, restore_env_state=False)
    torch.testing.assert_close(restored.policy.state_dict(), algorithm.policy.state_dict(), rtol=0, atol=0)
    assert_optimizer_states_equal(restored, algorithm)


@pytest.mark.parametrize("variant", VARIANTS)
def test_optimizer_parameter_ownership_excludes_targets_and_includes_nop(env, variant):
    options = nop_options("both") if variant in NOP_VARIANTS else {}
    algorithm = make_algorithm(env, variant, policy_options=options)
    policy = algorithm.policy
    actor_ids = {id(parameter) for parameter in policy.actor_parameters()}
    critic_ids = {id(parameter) for parameter in policy.critic_parameters()}
    assert actor_ids and critic_ids and actor_ids.isdisjoint(critic_ids)
    assert actor_ids | critic_ids == {id(parameter) for parameter in policy.parameters() if parameter.requires_grad}
    for optimizer, expected in ((algorithm.actor_optimizer, actor_ids), (algorithm.critic_optimizer, critic_ids)):
        parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        assert len(parameters) == len(expected)
        assert {id(parameter) for parameter in parameters} == expected
    for name, owner in (("actor_nop", actor_ids), ("critic_nop", critic_ids)):
        module = getattr(policy, name)
        if module is not None:
            assert {id(parameter) for parameter in module.parameters()} <= owner


@pytest.mark.parametrize("variant", DETERMINISTIC_VARIANTS)
def test_actor_nan_padding_and_empty_swarms_have_zero_input_gradients(env, variant):
    torch.manual_seed(42)
    policy = make_policy(env, variant)
    obs = observations(env)
    obs["agent_mask"] = obs["agent_mask"].clone()
    obs["agent_mask"][1] = False
    for key in ("local_obs", "hidden_local_vars"):
        obs[key][~obs["agent_mask"]] = float("nan")
        obs[key].requires_grad_()
    obs["global_obs"].requires_grad_()
    obs["hidden_global_vars"].requires_grad_()
    clean = {**obs, "local_obs": obs["local_obs"].masked_fill(~obs["agent_mask"].unsqueeze(-1), 0)}

    actions = policy.act(**obs, deterministic=True)

    torch.testing.assert_close(actions, policy.act(**clean, deterministic=True), rtol=0, atol=0)
    assert torch.isfinite(actions).all() and (actions[~obs["agent_mask"]] == 0).all()
    actions.sum().backward()
    assert torch.isfinite(obs["local_obs"].grad).all()
    assert (obs["local_obs"].grad[~obs["agent_mask"]] == 0).all()
    assert obs["local_obs"].grad[obs["agent_mask"]].abs().sum() > 0
    assert torch.isfinite(obs["global_obs"].grad).all() and (obs["global_obs"].grad[1] == 0).all()
    assert obs["hidden_local_vars"].grad is None and obs["hidden_global_vars"].grad is None
    assert all(parameter.grad is None for parameter in policy.critic_parameters())


@pytest.mark.parametrize("variant", NOP_VARIANTS)
@pytest.mark.parametrize("source", ["critic", "actor", "both"])
def test_nop_sources_update_online_encoders_and_predictors(env, variant, source):
    torch.manual_seed(42)
    algorithm = make_algorithm(env, variant, policy_options=nop_options(source))
    policy = algorithm.policy
    assert policy.has_nop_loss() and policy.get_nop_num_next_steps() == 3
    assert (policy.actor_nop is not None) == (source in {"actor", "both"})
    assert (policy.critic_nop is not None) == (source in {"critic", "both"})
    fill_replay(algorithm)
    batch, nop_batch, reuse = algorithm._sample_training_batches()
    assert nop_batch is not None and nop_batch.sequence_length == 3
    for name in ("actor", "critic"):
        module = getattr(policy, f"{name}_nop")
        if module is None:
            continue
        policy.zero_grad(set_to_none=True)
        loss, _ = getattr(policy, f"compute_{name}_nop_loss")(nop_batch)
        assert torch.isfinite(loss) and loss.requires_grad
        loss.backward()
        if name == "actor":
            encoder = policy.actor.encoder if isinstance(algorithm, TD3) else policy.actor_encoder
            encoders = [encoder]
        elif policy.config.joint_critic_config.kind == "deepset" and not variant.startswith("tmatd3"):
            networks = policy.critic if isinstance(algorithm, TD3) else (policy.critic.q1, policy.critic.q2)
            encoders = [network.network.deepset.element_encoder for network in networks]
        else:
            encoders = [encoder for encoder in (policy.critic.encoder, policy.critic.encoder2) if encoder is not None]
        for encoder in encoders:
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
        assert all(p.grad is None for p in policy.critic_target.parameters())
        if isinstance(algorithm, TD3):
            assert all(p.grad is None for p in policy.actor_target.parameters())
    before = {name: p.clone() for name, p in policy.named_parameters() if "_nop." in name}
    metrics = algorithm.train(gradient_steps=2)
    for name in ("actor", "critic"):
        if getattr(policy, f"{name}_nop") is not None:
            assert metrics[f"{name}_nop_loss_scaled"].mean > 0
            assert any(not torch.equal(p, before[key]) for key, p in policy.named_parameters()
                       if key.startswith(f"{name}_nop."))
    assert all(torch.isfinite(p).all() for p in policy.parameters())


@pytest.mark.parametrize("variant", ["maddpg_mlp", "matd3_mlp", "masac_mlp"])
@pytest.mark.parametrize("source", ["critic", "actor", "both"])
def test_mlp_presets_reject_nop_for_every_source(env, variant, source):
    with pytest.raises(ValueError, match="MLP critics do not support NOP"):
        make_policy(env, variant, **nop_options(source))


@pytest.mark.parametrize("variant", NOP_VARIANTS)
def test_nop_critic_latents_ignore_padding_and_follow_agent_permutations(env, variant):
    policy = make_policy(env, variant, **nop_options()).eval()
    obs = observations(env)
    obs["actions"] = torch.randn(2, 3, env.action_space.total_agent_action_dim)
    for key in ("local_obs", "actions", "hidden_local_vars"):
        obs[key][~obs["agent_mask"]] = float("nan")
    if isinstance(policy, TD3Policy):
        values, latents = policy.q_values_with_nop_latents(**obs)
    else:
        q1, q2, latents = policy.q_values_with_nop_latents(**obs)
        values = (q1, q2)
    assert latents.shape == (2, 3, policy.critic_nop.source_latent_dim)
    assert torch.isfinite(latents).all()
    assert (latents[~obs["agent_mask"]] == 0).all()
    perm = torch.tensor([2, 0, 1])
    shuffled = {key: value[:, perm] if key in {"local_obs", "actions", "hidden_local_vars", "agent_mask"}
                else value for key, value in obs.items()}
    actual = policy.q_values_with_nop_latents(**shuffled)
    shuffled_values, shuffled_latents = actual if isinstance(policy, TD3Policy) else (actual[:2], actual[2])
    torch.testing.assert_close(shuffled_latents, latents[:, perm], atol=1e-6, rtol=1e-5)
    for expected, value in zip(values, shuffled_values, strict=True):
        torch.testing.assert_close(expected, value, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("variant", NOP_VARIANTS)
@pytest.mark.parametrize("independent", [False, True])
def test_nop_checkpoint_continuation_and_independent_sampling(env, variant, independent, tmp_path):
    torch.manual_seed(42)
    options = dict(policy_options=nop_options("both"), independent_nop_sampling=independent)
    if independent:
        options["nop_batch_size"] = 2
    algorithm = make_algorithm(env, variant, **options)
    fill_replay(algorithm)
    algorithm.train(gradient_steps=1)
    path = tmp_path / "nop.pt"
    algorithm.save(path, optimizer_state_dict=algorithm._get_optimizer_state_dict())
    restored = make_algorithm(env, variant, **options)
    restored.load(path, restore_env_state=False)
    assert restored.policy.has_nop_loss()
    assert restored.n_total_updates == 1
    assert_optimizer_states_equal(restored, algorithm)
    for name, value in algorithm.policy.state_dict().items():
        torch.testing.assert_close(restored.policy.state_dict()[name], value, rtol=0, atol=0)
    # Both learners see exactly the same replay and random samples on the next update.
    restored.replay_buffer = algorithm.replay_buffer
    rng_state = torch.get_rng_state()
    expected = algorithm.train(gradient_steps=1)
    torch.set_rng_state(rng_state)
    actual = restored.train(gradient_steps=1)
    assert actual.keys() == expected.keys()
    for name, value in algorithm.policy.state_dict().items():
        torch.testing.assert_close(restored.policy.state_dict()[name], value, rtol=0, atol=0)
    assert_optimizer_states_equal(restored, algorithm)


@pytest.mark.parametrize("variant", ["matd3_deepset", "tmatd3", "tmatd3_dec"])
def test_td3_delays_actor_nop_but_trains_critic_nop_every_step(env, variant):
    algorithm = make_algorithm(env, variant, policy_options=nop_options("both"))
    fill_replay(algorithm)
    policy = algorithm.policy
    before = {name: p.clone() for name, p in policy.named_parameters()}
    metrics = algorithm.train(gradient_steps=1)
    assert metrics["actor_updated"].mean == 0
    assert "actor_nop_loss" not in metrics
    for name, parameter in policy.named_parameters():
        if name.startswith(("actor.", "actor_target.", "critic_target.", "actor_nop.")):
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    assert any(not torch.equal(p, before[name]) for name, p in policy.named_parameters()
               if name.startswith("critic_nop."))
    metrics = algorithm.train(gradient_steps=1)
    assert metrics["actor_updated"].mean == 1 and metrics["actor_nop_loss_scaled"].mean > 0


def test_low_level_mlp_policies_reject_nop(env):
    from swarmbots.learn.algos.sac.masac_policy import MASACPolicyConfig
    from swarmbots.learn.algos.td3.td3_policy import TD3PolicyConfig

    config = SACNOPConfig(enabled=True, next_obs_pred_config=NextObsPredConfig(local_scalar_target_indices=[0]))
    with pytest.raises(ValueError, match="MLP critics do not support NOP"):
        TD3Policy(env, TD3PolicyConfig(nop_config=config))
    with pytest.raises(ValueError, match="MLP critics do not support NOP"):
        MASACPolicy(env, MASACPolicyConfig(nop_config=config))


@pytest.mark.parametrize("variant", NOP_VARIANTS)
def test_nop_independent_sampling_skips_unavailable_segments(env, variant):
    algorithm = make_algorithm(
        env, variant, policy_options=nop_options("both", steps=4), independent_nop_sampling=True,
    )
    fill_replay(algorithm)
    before = {name: p.clone() for name, p in algorithm.policy.named_parameters() if "_nop." in name}
    metrics = algorithm.train(gradient_steps=2)
    assert metrics["nop_loss_skipped"].mean == 1
    for name, parameter in algorithm.policy.named_parameters():
        if name in before:
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("variant", NOP_VARIANTS)
def test_single_step_nop_can_train_transition_and_change_loss_weight(env, variant):
    algorithm = make_algorithm(env, variant, policy_options=nop_options(
        "both", steps=1, nop_skip_first_transition_for_critic=False,
    ))
    batch = fill_replay(algorithm)
    policy = algorithm.policy
    loss, _ = policy.compute_critic_nop_loss(batch)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in policy.critic_nop.transition_model.parameters())
    policy.update_loss_weights(nop_loss_coef=0.0)
    for name in ("actor", "critic"):
        loss, metrics = getattr(policy, f"compute_{name}_nop_loss")(batch)
        assert loss.item() == 0.0 and metrics[f"{name}_nop_loss_scaled"] == 0.0
    with pytest.raises(ValueError, match="nonnegative|>= 0"):
        policy.update_loss_weights(nop_loss_coef=-0.1)


@pytest.mark.parametrize("variant", NOP_VARIANTS)
def test_compiled_nop_update_matches_eager(env, variant):
    from swarmbots.learn.checkpointing import align_torch_compile_state_dict_keys

    real_compile = torch.compile
    torch.compiler.reset()
    options = nop_options("both", compile_world_model_modules=True)
    with patch("torch.compile", side_effect=lambda fn, **kwargs: real_compile(
        fn, backend="aot_eager", fullgraph=kwargs.get("fullgraph", False), dynamic=False,
    )):
        compiled = make_algorithm(env, variant, compile_modules=True, policy_options=options)
        eager = make_algorithm(env, variant, policy_options=options)
        eager.policy.load_state_dict(align_torch_compile_state_dict_keys(
            compiled.policy.state_dict(), target_keys=eager.policy.state_dict().keys(),
        ))
        fill_replay(compiled)
        eager.replay_buffer = compiled.replay_buffer
        rng_state = torch.get_rng_state()
        compiled.train(gradient_steps=2)
        torch.set_rng_state(rng_state)
        eager.train(gradient_steps=2)
        compiled_state = align_torch_compile_state_dict_keys(
            compiled.policy.state_dict(), target_keys=eager.policy.state_dict().keys(),
        )
        for name, parameter in compiled_state.items():
            torch.testing.assert_close(parameter, eager.policy.state_dict()[name], atol=1e-6, rtol=1e-5)
    torch.compiler.reset()


@pytest.mark.cuda
@pytest.mark.parametrize("variant", NOP_VARIANTS)
def test_cuda_compiled_nop_update(env, variant):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.compiler.reset()
    algorithm = make_algorithm(
        env, variant, device="cuda", compile_modules=True,
        policy_options=nop_options("both", compile_world_model_modules=True),
    )
    try:
        fill_replay(algorithm)
        before = {name: p.clone() for name, p in algorithm.policy.named_parameters() if "_nop." in name}
        metrics = algorithm.train(gradient_steps=2)
        assert metrics["critic_nop_loss_scaled"].mean > 0 and metrics["actor_nop_loss_scaled"].mean > 0
        for prefix in ("actor_nop.", "critic_nop."):
            assert any(not torch.equal(p, before[name]) for name, p in algorithm.policy.named_parameters()
                       if name.startswith(prefix))
        assert all(torch.isfinite(p).all() for p in algorithm.policy.parameters())
        assert all(p.grad is None and not p.requires_grad for p in algorithm.policy.critic_target.parameters())
        if isinstance(algorithm, TD3):
            assert all(p.grad is None and not p.requires_grad for p in algorithm.policy.actor_target.parameters())
    finally:
        torch.compiler.reset()


def test_td3_delays_actor_and_targets_and_resumes_delay_across_train_calls(env, tmp_path):
    algorithm = make_algorithm(env, "matd3_mlp")
    fill_replay(algorithm)
    policy = algorithm.policy
    snapshot = {name: value.clone() for name, value in policy.state_dict().items()}
    algorithm.train(gradient_steps=1)
    for name, value in policy.state_dict().items():
        if not name.startswith("critic."):
            torch.testing.assert_close(value, snapshot[name], rtol=0, atol=0)
    assert any(
        not torch.equal(value, snapshot[name])
        for name, value in policy.state_dict().items()
        if name.startswith("critic.")
    )
    path = tmp_path / "td3.pt"
    algorithm.save(path, optimizer_state_dict=algorithm._get_optimizer_state_dict())
    restored = make_algorithm(env, "matd3_mlp")
    restored.load(path, restore_env_state=False)
    assert restored.n_total_updates == 1
    assert restored.actor_optimizer.state_dict() == algorithm.actor_optimizer.state_dict()
    assert restored.critic_optimizer.state_dict()["state"]
    fill_replay(restored)
    restored.train(gradient_steps=1)
    assert restored.n_total_updates == 2
    for prefix in ("actor.", "actor_target.", "critic_target."):
        assert any(
            not torch.equal(value, snapshot[name])
            for name, value in restored.policy.state_dict().items()
            if name.startswith(prefix)
        )


@pytest.mark.parametrize("variant", ["matd3_mlp", "matd3_deepset", "tmatd3", "tmatd3_dec"])
def test_custom_td3_delay_and_polyak_values_across_uneven_train_calls(env, variant):
    torch.manual_seed(42)
    algorithm = make_algorithm(env, variant, policy_delay=3, tau=0.2)
    fill_replay(algorithm)
    policy = algorithm.policy
    for source, target in ((policy.actor, policy.actor_target), (policy.critic, policy.critic_target)):
        for parameter, target_parameter in zip(source.parameters(), target.parameters(), strict=True):
            assert parameter.data_ptr() != target_parameter.data_ptr()

    for steps, actor_updates in ((2, 0), (1, 1), (1, 0), (2, 1), (2, 0), (1, 1)):
        actor_before = {name: p.detach().clone() for name, p in policy.actor.named_parameters()}
        targets_before = {
            name: p.detach().clone()
            for name, p in policy.named_parameters()
            if name.startswith(("actor_target.", "critic_target."))
        }
        with patch.object(algorithm.actor_optimizer, "step", wraps=algorithm.actor_optimizer.step) as actor_step:
            algorithm.train(gradient_steps=steps)
        assert actor_step.call_count == actor_updates
        if actor_updates == 0:
            torch.testing.assert_close(dict(policy.actor.named_parameters()), actor_before, rtol=0, atol=0)
        else:
            assert any(not torch.equal(p, actor_before[name]) for name, p in policy.actor.named_parameters())
        for prefix, source, target in (
            ("actor_target", policy.actor, policy.actor_target),
            ("critic_target", policy.critic, policy.critic_target),
        ):
            source_parameters = dict(source.named_parameters())
            for name, parameter in target.named_parameters():
                before = targets_before[f"{prefix}.{name}"]
                expected = (1 - algorithm.tau) * before + algorithm.tau * source_parameters[name]
                if not actor_updates:
                    expected = before
                torch.testing.assert_close(parameter, expected, rtol=1e-6, atol=1e-7)
                assert not parameter.requires_grad and parameter.grad is None
            assert not target.training
    assert algorithm.n_total_updates == 9


@pytest.mark.parametrize("variant", VARIANTS)
def test_checkpoint_continuation_matches_uninterrupted_updates(env, variant, tmp_path):
    torch.manual_seed(42)
    algorithm = make_algorithm(env, variant, learning_rate_warmup_updates=5)
    batch = fill_replay(algorithm)
    algorithm.train(gradient_steps=3)
    path = tmp_path / f"{variant}.pt"
    algorithm.save(path, optimizer_state_dict=algorithm._get_optimizer_state_dict())

    restored = make_algorithm(env, variant, learning_rate_warmup_updates=5)
    fill_replay(restored)
    restored.load(path, restore_env_state=False)
    assert len(restored.replay_buffer) == 0
    assert not restored._should_train()
    assert restored.n_total_updates == algorithm.n_total_updates == 3
    assert restored.actor_optimizer.state_dict()["state"]
    assert_optimizer_states_equal(restored, algorithm)

    # Isolate resume semantics from the intentionally discarded replay and RNG state.
    # Both learners receive the same transitions and random action/noise samples.
    with (
        patch.object(algorithm, "_sample_training_batches", return_value=(batch, None, False)),
        patch.object(restored, "_sample_training_batches", return_value=(batch, None, False)),
    ):
        for steps in (1, 2, 1):
            rng_state = torch.get_rng_state()
            algorithm.train(gradient_steps=steps)
            torch.set_rng_state(rng_state)
            restored.train(gradient_steps=steps)
            assert restored.n_total_updates == algorithm.n_total_updates
            torch.testing.assert_close(restored.policy.state_dict(), algorithm.policy.state_dict(), rtol=0, atol=0)
            assert_optimizer_states_equal(restored, algorithm)


@pytest.mark.parametrize("variant", ["maddpg_mlp", "matd3_mlp"])
def test_bellman_uses_target_actor_minimum_and_only_terminations_stop_bootstrap(env, variant):
    algorithm = make_algorithm(env, variant)
    batch = fill_replay(algorithm)
    batch = replace(
        batch,
        rewards=torch.ones(4),
        terminations=torch.tensor([True, False, False, False]),
        truncations=torch.tensor([False, True, False, False]),
    )
    values = (torch.full((4,), 4.0),) if isinstance(algorithm, DDPG) else (torch.full((4,), 4.0), torch.full((4,), 2.0))
    with (
        patch.object(algorithm.policy, "actor_actions", wraps=algorithm.policy.actor_actions) as actor,
        patch.object(algorithm.policy, "target_q_values", return_value=values),
    ):
        target = algorithm._bellman_target(batch)
    assert actor.call_args.kwargs["target"] is True
    expected = 4.0 if isinstance(algorithm, DDPG) else 2.0
    torch.testing.assert_close(target, torch.tensor([1.0] + [1.0 + algorithm.gamma * expected] * 3))


@pytest.mark.parametrize("variant", ["maddpg_mlp", "matd3_deepset", "tmatd3"])
def test_terminal_nan_observations_cannot_poison_bellman_targets(env, variant):
    torch.manual_seed(42)
    algorithm = make_algorithm(env, variant)
    batch = replace(
        fill_replay(algorithm),
        terminations=torch.tensor([True, False, True, False]),
        truncations=torch.tensor([False, True, True, False]),
        next_agent_mask=torch.ones(4, env.n_agents, dtype=torch.bool),
    )
    poisoned_fields = {}
    for name in ("next_local_obs", "next_global_obs", "next_hidden_local_vars", "next_hidden_global_vars"):
        poisoned_fields[name] = getattr(batch, name).clone()
        poisoned_fields[name][batch.terminations] = float("nan")
    poisoned_mask = batch.next_agent_mask.clone()
    poisoned_mask[batch.terminations] = False
    poisoned = replace(batch, **poisoned_fields, next_agent_mask=poisoned_mask)
    rng_state = torch.get_rng_state()
    expected = algorithm._bellman_target(batch)
    torch.set_rng_state(rng_state)
    actual = algorithm._bellman_target(poisoned)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[batch.terminations], batch.rewards[batch.terminations], rtol=0, atol=0)
    assert torch.isfinite(actual).all() and not actual.requires_grad
    algorithm._train_step(poisoned, global_update_idx=1)
    assert all(torch.isfinite(p).all() for p in algorithm.policy.parameters())


@pytest.mark.parametrize("variant", ["masac_mlp", "masac_deepset"])
@pytest.mark.parametrize("distribution", ["predicted_std_gaussian", "gumbel_softmax_sign_magnitude_beta"])
def test_masac_multiple_action_samples_match_individual_critic_evaluations(env, variant, distribution):
    torch.manual_seed(42)
    algorithm = make_algorithm(
        env, variant,
        policy_options={"continuous_action_dist": distribution},
        actor_action_samples=3, target_action_samples=2,
    )
    policy = algorithm.policy
    obs = observations(env)
    actions, log_probs = policy.action_log_prob(**obs, num_action_samples=3)
    assert actions.shape == (3, 2, env.n_agents, env.action_space.total_agent_action_dim)
    assert log_probs.shape == (3, 2, env.n_agents)
    assert torch.isfinite(actions).all() and torch.isfinite(log_probs).all()
    assert (actions[:, :, 1] == 0).all() and (log_probs[:, :, 1] == 0).all()
    for target in (False, True):
        batched = policy.q_values_samples(actions=actions, num_action_samples=3, target=target, **obs)
        evaluate = policy.target_q_values if target else policy.q_values
        individual = [evaluate(actions=sample, **obs) for sample in actions]
        for critic_idx, q_values in enumerate(batched):
            torch.testing.assert_close(q_values, torch.stack([values[critic_idx] for values in individual]))
    fill_replay(algorithm)
    temperature_before = algorithm.log_ent_coef.detach().clone()
    actor_before = [p.detach().clone() for p in policy.actor_parameters()]
    assert algorithm.train(gradient_steps=3)["updates"] == 3
    assert not torch.equal(algorithm.log_ent_coef, temperature_before)
    assert any(not torch.equal(a, b) for a, b in zip(actor_before, policy.actor_parameters(), strict=True))
    assert all(torch.isfinite(p).all() for p in policy.parameters())


def test_noise_respects_asymmetric_physical_bounds_and_masks():
    env = SimpleNamespace(
        n_agents=3,
        local_obs_dim=2,
        global_obs_dim=0,
        hidden_local_vars_dim=0,
        hidden_global_vars_dim=0,
        action_space=HybridActionSpace({"control": spaces.Box(2.0, 6.0, (3, 2))}),
    )
    policy = make_policy(env, "matd3_mlp")
    actions = torch.full((2, 3, 2), 4.0)
    mask = torch.tensor([[True, False, True], [True, True, True]])
    smoothed = policy.add_action_noise(actions, std=100.0, clip=0.25, agent_mask=mask)
    assert ((smoothed[mask] >= 3.5) & (smoothed[mask] <= 4.5)).all()
    assert (smoothed[~mask] == 0).all()
    noisy = policy.add_action_noise(actions, std=100.0, agent_mask=mask)
    assert ((noisy[mask] >= 2) & (noisy[mask] <= 6)).all()


def heterogeneous_bounds_env(vectorized):
    low = torch.tensor([[-2.0, 2.0], [-4.0, 3.0], [1.0, -1.0]])
    high = torch.tensor([[6.0, 6.0], [2.0, 9.0], [9.0, 3.0]])
    if vectorized:
        low, high = low.repeat(2, 1, 1), high.repeat(2, 1, 1)
    action_space_class = VectorHybridActionSpace if vectorized else HybridActionSpace
    return SimpleNamespace(
        n_agents=3, local_obs_dim=2, global_obs_dim=0, hidden_local_vars_dim=0, hidden_global_vars_dim=0,
        action_space=action_space_class([
            ("control", spaces.Box(low.numpy(), high.numpy())),
            ("connectors", spaces.Box(0.5, 1.5, (*low.shape[:-1], 1))),
        ]),
    )


@pytest.mark.parametrize("vectorized", [False, True])
def test_actor_maps_normalized_actions_to_each_hybrid_component_bound(vectorized):
    policy = make_policy(heterogeneous_bounds_env(vectorized), "matd3_mlp")
    with torch.no_grad():
        for actor in (policy.actor, policy.actor_target):
            actor.action_net.weight.zero_()
            actor.action_net.bias.copy_(torch.atanh(torch.tensor([-0.5, 0.0, 0.5])))
    obs = dict(
        local_obs=torch.zeros(2, 3, 2), global_obs=torch.empty(2, 0),
        agent_mask=torch.tensor([[True, False, True], [False, True, True]]),
    )
    expected = torch.tensor([[0.0, 4.0, 1.25], [-2.5, 6.0, 1.25], [3.0, 1.0, 1.25]])
    expected = expected.expand(2, -1, -1).masked_fill(~obs["agent_mask"].unsqueeze(-1), 0)
    for target in (False, True):
        torch.testing.assert_close(policy.actor_actions(**obs, target=target), expected)


@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("clip", [None, 0.25])
def test_noise_is_scaled_per_component_clipped_before_scaling_and_bounded(vectorized, clip):
    policy = make_policy(heterogeneous_bounds_env(vectorized), "matd3_mlp")
    actions = torch.tensor([
        [[-2.0, 2.0, 0.5], [-4.0, 3.0, 0.5], [1.0, -1.0, 0.5]],
        [[6.0, 6.0, 1.5], [2.0, 9.0, 1.5], [9.0, 3.0, 1.5]],
    ])
    actions_before = actions.clone()
    mask = torch.tensor([[True, False, True], [True, True, True]])
    draws = torch.tensor([-4.0, 0.5, 4.0]).expand_as(actions)
    expected = torch.tensor([
        [[-2.0, 2.2, 0.625], [0.0, 0.0, 0.0], [1.0, -0.8, 0.625]],
        [[5.0, 6.0, 1.5], [1.25, 9.0, 1.5], [8.0, 3.0, 1.5]],
    ]) if clip is not None else torch.tensor([
        [[-2.0, 2.2, 0.9], [0.0, 0.0, 0.0], [1.0, -0.8, 0.9]],
        [[2.8, 6.0, 1.5], [-0.4, 9.0, 1.5], [5.8, 3.0, 1.5]],
    ])
    with patch("torch.randn_like", return_value=draws) as noise:
        actual = policy.add_action_noise(actions, std=0.2, clip=clip, agent_mask=mask)
    noise.assert_called_once()
    torch.testing.assert_close(actual, expected)
    # Noise generation must not overwrite replay or actor output tensors.
    torch.testing.assert_close(actions, actions_before, rtol=0, atol=0)


@pytest.mark.parametrize("noise_std", [0.0, 0.4])
def test_bellman_evaluates_clipped_target_noise_and_ignores_exploration_noise(env, noise_std):
    algorithm = make_algorithm(
        env, "matd3_mlp", target_policy_noise=noise_std, target_noise_clip=0.2, exploration_noise=100.0, gamma=0.5,
    )
    batch = replace(
        fill_replay(algorithm), rewards=torch.arange(4, dtype=torch.float32),
        terminations=torch.zeros(4, dtype=torch.bool), truncations=torch.zeros(4, dtype=torch.bool),
        next_agent_mask=torch.tensor([[True, False, True]]).expand(4, -1),
    )
    target_actions = torch.full_like(batch.actions, 0.9).masked_fill(~batch.next_agent_mask.unsqueeze(-1), 0)
    expected_actions = torch.tensor([[1.0, 0.7], [0.0, 0.0], [0.7, 1.0]]) if noise_std else target_actions[0]
    expected_actions = expected_actions.expand_as(target_actions)

    def target_values(**observations):
        torch.testing.assert_close(observations["actions"], expected_actions)
        q1 = observations["actions"].sum(dim=(-2, -1))
        return q1, q1 - 2

    draws = torch.tensor([[1.0, -1.0], [100.0, -100.0], [-1.0, 1.0]]).expand_as(target_actions)
    with (
        patch.object(algorithm.policy, "actor_actions", return_value=target_actions) as actor,
        patch.object(algorithm.policy, "target_q_values", side_effect=target_values),
        patch("torch.randn_like", return_value=draws) as noise,
    ):
        actual = algorithm._bellman_target(batch)
    assert actor.call_args.kwargs["target"] is True
    assert noise.call_count == int(noise_std > 0)
    torch.testing.assert_close(actual, batch.rewards + (0.7 if noise_std else 0.8))
    assert not actual.requires_grad


@pytest.mark.parametrize("failure_point", ["actor_actions", "optimizer_step"])
def test_actor_update_failure_restores_critic_gradients_and_leaves_targets_unchanged(env, failure_point):
    algorithm = make_algorithm(env, "matd3_mlp")
    batch = fill_replay(algorithm)
    policy = algorithm.policy
    targets_before = {name: value.clone() for name, value in policy.state_dict().items()
                      if name.startswith(("actor_target.", "critic_target."))}
    original_actions = policy.actor_actions

    def fail_actor(**observations):
        if not observations.get("target", False):
            assert all(not parameter.requires_grad for parameter in policy.critic_parameters())
            raise RuntimeError("actor update failed")
        return original_actions(**observations)

    failure = patch.object(policy, "actor_actions", side_effect=fail_actor) if failure_point == "actor_actions" else (
        patch.object(algorithm.actor_optimizer, "step", side_effect=RuntimeError("actor update failed"))
    )
    with failure, pytest.raises(RuntimeError, match="actor update failed"):
        algorithm._train_step(batch, global_update_idx=1)
    assert all(parameter.requires_grad for parameter in policy.critic_parameters())
    for name, expected in targets_before.items():
        torch.testing.assert_close(policy.state_dict()[name], expected, rtol=0, atol=0)
    # A failed actor update must not leave the learner unable to train the critic.
    metrics, _, _ = algorithm._train_step(batch, global_update_idx=1)
    assert metrics["actor_updated"] == 1
    assert all(torch.isfinite(parameter).all() for parameter in policy.parameters())


@pytest.mark.parametrize("invalid_space,message", [
    ("discrete", "continuous Box"),
    ("unbounded", "finite action bounds"),
    ("constant", "upper bounds"),
    ("different_lanes", "identical action bounds"),
])
def test_deterministic_actors_reject_unsupported_action_spaces(invalid_space, message):
    env = heterogeneous_bounds_env(invalid_space == "different_lanes")
    if invalid_space == "different_lanes":
        env.action_space["control"].high[1, 0, 0] += 1
    else:
        subspace = {
            "discrete": spaces.MultiBinary((3, 2)),
            "unbounded": spaces.Box(-float("inf"), float("inf"), (3, 2)),
            "constant": spaces.Box(1.0, 1.0, (3, 2)),
        }[invalid_space]
        env.action_space = HybridActionSpace({"control": subspace})
    with pytest.raises(ValueError, match=message):
        make_policy(env, "matd3_mlp")


@pytest.mark.parametrize("field", ["hidden_dims", "element_hidden_dims"])
@pytest.mark.parametrize("dimensions", [(), (0,), (16, -1)])
def test_joint_critic_config_rejects_empty_and_nonpositive_widths(field, dimensions):
    with pytest.raises(ValueError, match="nonempty and positive"):
        JointCriticConfig(**{field: dimensions})


@pytest.mark.parametrize("kind", ["mlp", "deepset"])
@pytest.mark.parametrize("missing", ["hidden_local_vars", "hidden_global_vars"])
def test_joint_critics_require_configured_privileged_fields(kind, missing):
    critic = JointQNetwork(
        n_agents=3, local_obs_dim=2, global_obs_dim=1, hidden_local_vars_dim=1,
        hidden_global_vars_dim=1, action_dim=1, config=JointCriticConfig(kind, (4,), (4,)),
    )
    obs = dict(
        local_obs=torch.zeros(2, 3, 2), global_obs=torch.zeros(2, 1), actions=torch.zeros(2, 3, 1),
        hidden_local_vars=torch.zeros(2, 3, 1), hidden_global_vars=torch.zeros(2, 1),
    )
    del obs[missing]
    with pytest.raises(ValueError, match=f"{missing} are required"):
        critic(**obs)


@pytest.mark.parametrize("variant", ["maddpg_mlp", "matd3_deepset", "tmatd3"])
@pytest.mark.parametrize("exploration_noise", [0.0, 0.4])
def test_deterministic_evaluation_bypasses_configured_exploration(env, variant, exploration_noise):
    algorithm = make_algorithm(env, variant, exploration_noise=exploration_noise)
    policy = algorithm.policy
    obs = observations(env)
    shape = (2, env.n_agents, env.action_space.total_agent_action_dim)
    with patch("torch.randn_like", return_value=torch.ones(shape)) as noise:
        clean = policy.act(**obs, deterministic=True)
        noise.assert_not_called()
        exploratory = policy.act(**obs, deterministic=False)
        assert noise.call_count == int(exploration_noise > 0)
    expected = (clean + exploration_noise).clamp(-1, 1).masked_fill(~obs["agent_mask"].unsqueeze(-1), 0)
    torch.testing.assert_close(exploratory, expected, rtol=0, atol=0)
    torch.testing.assert_close(policy.act(**obs, deterministic=True), clean, rtol=0, atol=0)


def test_td3_actor_optimizes_q1_even_when_q2_is_smaller(env):
    algorithm = make_algorithm(env, "matd3_mlp")
    batch = fill_replay(algorithm)
    original_q_values = algorithm.policy.q_values

    def controlled_values(**observations):
        if observations["actions"] is batch.actions:
            return original_q_values(**observations)
        # Actor updates must minimize -Q1; Q2 has the opposite action gradient.
        q1 = observations["actions"].sum(dim=(-2, -1))
        return q1, -100 - q1

    with patch.object(algorithm.policy, "q_values", side_effect=controlled_values):
        with torch.no_grad():
            before = algorithm.policy.actor_actions(
                local_obs=batch.local_obs,
                global_obs=batch.global_obs,
                agent_mask=batch.agent_mask,
            ).sum()
        algorithm._train_step(batch, global_update_idx=1)
        with torch.no_grad():
            after = algorithm.policy.actor_actions(
                local_obs=batch.local_obs,
                global_obs=batch.global_obs,
                agent_mask=batch.agent_mask,
            ).sum()
    assert after > before


def test_direct_masac_construction_defaults_to_a_decentralized_actor(env):
    policy = MASACPolicy(env)
    assert not any(isinstance(module, nn.MultiheadAttention) for module in policy.actor_encoder.modules())
    assert not any(isinstance(module, nn.MultiheadAttention) for module in policy.critic.modules())


def test_low_level_td3_configuration_is_resolved_and_reported_consistently(env):
    template = make_policy(env, "tmatd3").config
    config = replace(
        template,
        critic_kind="deepset",
        joint_critic_config=JointCriticConfig("mlp", (16,), (16,)),
        actor_head_config=TMASACActorHeadConfig(kind=TMASACActorHeadKind.DECENTRALIZED),
    )
    policy = TD3Policy(env, config)
    assert not any(isinstance(module, nn.MultiheadAttention) for module in policy.actor.modules())
    assert policy.get_hyper_parameters()["td3_policy_config"]["joint_critic_config"]["kind"] == "deepset"
    with pytest.raises(ValueError, match="QCX"):
        TD3Policy(env, replace(config, actor_head_config=TMASACActorHeadConfig(kind=TMASACActorHeadKind.QCX)))


def test_td3_does_not_accept_entropy_commands(env):
    algorithm = make_algorithm(env, "matd3_mlp")
    with pytest.raises(ValueError, match="entropy coefficient"):
        algorithm._execute_command("set_ent_coef", "0.5", None)
    assert algorithm.ent_coef_tensor is None
    assert algorithm.ent_coef_optimizer is None


@pytest.mark.parametrize(
    "option,value",
    [
        ("policy_delay", 0),
        ("policy_delay", 1.5),
        ("policy_delay", True),
        ("exploration_noise", -0.1),
        ("target_policy_noise", float("nan")),
        ("target_noise_clip", float("inf")),
        ("ent_coef", 0.0),
        ("ent_coef_learning_rate", 1e-3),
        ("target_entropy", "auto"),
        ("target_update_interval", 1),
        ("actor_action_samples", 2),
        ("target_action_samples", 2),
        ("nop_batch_size", 4),
        ("independent_nop_sampling", True),
    ],
)
def test_invalid_td3_settings_are_rejected_before_allocating_replay(env, option, value):
    policy = make_policy(env, "matd3_mlp")
    with patch.object(TD3, "_build_replay_buffer") as allocate_replay:
        with pytest.raises(ValueError):
            TD3(policy, env, **{option: value})
        allocate_replay.assert_not_called()


@pytest.mark.parametrize("option,value", [("policy_delay", 2), ("target_policy_noise", 0.2), ("target_noise_clip", 0.5)])
def test_ddpg_rejects_td3_delay_and_smoothing_before_allocating_replay(env, option, value):
    policy = make_policy(env, "maddpg_mlp")
    with patch.object(DDPG, "_build_replay_buffer") as allocate_replay:
        with pytest.raises(ValueError, match=f"DDPG requires {option}"):
            DDPG(policy, env, **{option: value})
        allocate_replay.assert_not_called()


def test_ddpg_updates_actor_and_targets_on_the_first_step(env):
    algorithm = make_algorithm(env, "maddpg_mlp")
    batch = fill_replay(algorithm)
    before = {name: parameter.clone() for name, parameter in algorithm.policy.state_dict().items()}
    algorithm._train_step(batch, global_update_idx=0)
    for prefix in ("actor.", "actor_target.", "critic.", "critic_target."):
        assert any(
            not torch.equal(parameter, before[name])
            for name, parameter in algorithm.policy.state_dict().items()
            if name.startswith(prefix)
        )
    assert len(algorithm.policy.critic) == 1
    assert algorithm.target_policy_noise == 0
    assert algorithm.target_noise_clip == 0


@pytest.mark.parametrize("variant", VARIANTS)
def test_compiled_policy_preserves_state_dict_and_gradients(env, variant):
    real_compile = torch.compile
    with patch(
        "torch.compile", side_effect=lambda function, **kwargs: real_compile(function, backend="eager", **kwargs)
    ):
        policy = make_policy(env, variant, compile_modules=True)
        eager = make_policy(env, variant)
        # SAC wraps modules; the existing checkpoint loader aligns these names.
        from swarmbots.learn.checkpointing import align_torch_compile_state_dict_keys

        eager.load_state_dict(
            align_torch_compile_state_dict_keys(policy.state_dict(), target_keys=eager.state_dict().keys())
        )
        obs = observations(env)
        actions = policy.act(**obs, deterministic=True)
        torch.testing.assert_close(actions, eager.act(**obs, deterministic=True))
        q = policy.q_values(**obs, actions=actions)[0]
        (-q.mean()).backward()
        assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in policy.actor_parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.parameters())
    torch.compiler.reset()


@pytest.mark.cuda
@pytest.mark.parametrize("variant", VARIANTS)
def test_cuda_compiled_baseline_update(env, variant):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.compiler.reset()
    algorithm = make_algorithm(env, variant, device="cuda", compile_modules=True)
    try:
        fill_replay(algorithm)
        initial_actor = [p.detach().clone() for p in algorithm.policy.actor_parameters()]
        for steps in (3, 2, 3):
            metrics = algorithm.train(gradient_steps=steps)
            assert metrics["updates"] == steps
            assert all(torch.isfinite(p).all() for p in algorithm.policy.parameters())
            assert all(p.grad is None and not p.requires_grad for p in algorithm.policy.critic_target.parameters())
        assert algorithm.n_total_updates == 8
        assert any(
            not torch.equal(before, after)
            for before, after in zip(initial_actor, algorithm.policy.actor_parameters(), strict=True)
        )
    finally:
        del algorithm
        torch.compiler.reset()


@pytest.mark.cuda
@pytest.mark.parametrize("variant", ["tmatd3", "tmatd3_dec"])
def test_cuda_compiled_joint_embedding_trains_actor_and_critic_nop(env, variant):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.compiler.reset()
    try:
        algorithm = make_algorithm(
            env, variant, device="cuda", compile_modules=True,
            policy_options=nop_options("both", mat_joint_obs_embedding=True),
        )
        fill_replay(algorithm)
        before = [parameter.detach().clone() for parameter in algorithm.policy.actor_parameters()]
        for steps in (3, 2):
            metrics = algorithm.train(gradient_steps=steps)
            assert metrics["updates"] == steps
            assert metrics["critic_nop_loss_scaled"].mean > 0 and metrics["actor_nop_loss_scaled"].mean > 0
            assert all(torch.isfinite(parameter).all() for parameter in algorithm.policy.parameters())
        assert algorithm.n_total_updates == 5
        assert any(not torch.equal(parameter, old)
                   for parameter, old in zip(algorithm.policy.actor_parameters(), before, strict=True))
        for target in (algorithm.policy.actor_target, algorithm.policy.critic_target):
            assert all(not parameter.requires_grad and parameter.grad is None for parameter in target.parameters())
    finally:
        torch.compiler.reset()


@pytest.mark.parametrize("variant", VARIANTS)
def test_baseline_default_capacity_is_substantial(env, variant):
    policy = make_policy(env, variant, small=False)
    assert sum(p.numel() for p in policy.actor_parameters()) > 500_000
    assert sum(p.numel() for p in policy.critic_parameters()) > 500_000


@pytest.mark.parametrize("variant", VARIANTS)
def test_one_agent_limit(env, variant):
    single = SimpleNamespace(
        n_agents=1,
        local_obs_dim=4,
        global_obs_dim=2,
        hidden_local_vars_dim=0,
        hidden_global_vars_dim=0,
        action_space=HybridActionSpace({"control": spaces.Box(-1.0, 1.0, (1, 2))}),
    )
    policy = make_policy(single, variant)
    inputs = dict(local_obs=torch.randn(2, 1, 4), global_obs=torch.randn(2, 2))
    actions = policy.act(**inputs, deterministic=True)
    assert actions.shape == (2, 1, 2)
    assert all(torch.isfinite(q).all() for q in policy.q_values(**inputs, actions=actions))
