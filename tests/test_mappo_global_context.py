from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
from gymnasium import spaces

from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import SignMagnitudeBetaConfig
from swarmbots.learn.algos.mappo.mappo_actor import MAPPOActorConfig
from swarmbots.learn.algos.mappo.mappo_policy import MAPPOPolicy, MAPPOPolicyConfig
from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.mat_qc_base_policy import MATCriticConfig
from swarmbots.learn.checkpointing import align_torch_compile_state_dict_keys
from swarmbots.learn.hybrid_action_space import HybridActionSpace
from swarmbots.learn.nn_components.feed_forward import MLPConfig


@pytest.fixture(scope="module", autouse=True)
def single_torch_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_env(global_obs_dim: int = 3, hidden_global_vars_dim: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        n_agents=4, local_obs_dim=6, global_obs_dim=global_obs_dim, hidden_local_vars_dim=2,
        hidden_global_vars_dim=hidden_global_vars_dim,
        action_space=HybridActionSpace({"control": spaces.Box(-1.0, 1.0, (4, 2))}),
    )


def make_policy(
    env: SimpleNamespace, *, joint_obs_embedding: bool = False, use_popart: bool = False,
    compile_modules: bool = False,
) -> MAPPOPolicy:
    return MAPPOPolicy(env, MAPPOPolicyConfig(
        actor_config=MAPPOActorConfig(
            hidden_dims=[8], latent_dim=6, actor_head_hidden_dims=[8], latent_pi_dim=4,
        ),
        critic_encoder_config=MATEncoderConfig(
            d_model=16, nhead=4, num_layers=2, dim_feedforward=24, joint_obs_embedding=joint_obs_embedding,
            local_obs_encoder_config=MLPConfig(hidden_dims=[16]),
            global_obs_encoder_config=MLPConfig(hidden_dims=[16]),
        ),
        critic_config=MATCriticConfig(
            local_projection_hidden_dims=[8], value_regressor_hidden_dims=[8], use_popart=use_popart,
        ),
        continuous_config=SignMagnitudeBetaConfig(), compile_modules=compile_modules,
    ))


def observations(env: SimpleNamespace, *, device: str = "cpu") -> dict[str, torch.Tensor]:
    return dict(
        local_obs=torch.randn(2, env.n_agents, env.local_obs_dim, device=device, requires_grad=True),
        global_obs=torch.randn(2, env.global_obs_dim, device=device, requires_grad=True),
        hidden_local_vars=torch.randn(2, env.n_agents, env.hidden_local_vars_dim, device=device, requires_grad=True),
        hidden_global_vars=torch.randn(2, env.hidden_global_vars_dim, device=device, requires_grad=True),
        agent_mask=torch.tensor([[True, False, True, False], [False, False, False, False]], device=device),
    )


@pytest.mark.parametrize("joint_obs_embedding", [False, True])
@pytest.mark.parametrize("use_popart", [False, True])
def test_mappo_deepset_critic_preserves_masking_permutation_invariance_and_value_gradients(
    joint_obs_embedding: bool, use_popart: bool,
) -> None:
    torch.manual_seed(42)
    env = make_env()
    policy = make_policy(env, joint_obs_embedding=joint_obs_embedding, use_popart=use_popart)
    obs = observations(env)
    actions = policy.act(**obs, deterministic=True)
    values = policy.predict_values(**obs)
    privileged_changed = {**obs, "hidden_global_vars": obs["hidden_global_vars"] + 10}
    torch.testing.assert_close(policy.act(**privileged_changed, deterministic=True), actions, rtol=0, atol=0)
    assert not torch.allclose(policy.predict_values(**privileged_changed)[0], values[0])
    assert policy.has_popart is use_popart
    assert torch.isfinite(values).all()

    padded = {**obs}
    for field in ("local_obs", "hidden_local_vars"):
        padded[field] = obs[field].masked_fill(~obs["agent_mask"].unsqueeze(-1), torch.nan)
    torch.testing.assert_close(policy.predict_values(**padded), values, rtol=0, atol=0)
    permutation = torch.tensor([2, 0, 3, 1])
    shuffled = {
        field: tensor[:, permutation] if field in {"local_obs", "hidden_local_vars", "agent_mask"} else tensor
        for field, tensor in obs.items()
    }
    torch.testing.assert_close(policy.predict_values(**shuffled), values)

    values.sum().backward()
    for field in ("local_obs", "global_obs", "hidden_local_vars", "hidden_global_vars"):
        assert obs[field].grad is not None and torch.isfinite(obs[field].grad).all()
        assert obs[field].grad[0].abs().sum() > 0
        assert obs[field].grad[1].abs().sum() == 0
    for field in ("local_obs", "hidden_local_vars"):
        assert (obs[field].grad[~obs["agent_mask"]] == 0).all()
    for module in (policy.actor_encoder, policy.actor, policy.action_dist):
        assert all(parameter.grad is None for parameter in module.parameters())
    for module in (policy.critic_encoder, policy.critic):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        assert any(gradient.abs().sum() > 0 for gradient in gradients)


@pytest.mark.parametrize("joint_obs_embedding", [False, True])
@pytest.mark.parametrize("global_obs_dim,hidden_global_vars_dim", [(3, 5), (3, 0), (0, 5), (0, 0)])
def test_mappo_deepset_critic_supports_public_and_privileged_global_input_combinations(
    joint_obs_embedding: bool, global_obs_dim: int, hidden_global_vars_dim: int,
) -> None:
    env = make_env(global_obs_dim, hidden_global_vars_dim)
    policy = make_policy(env, joint_obs_embedding=joint_obs_embedding)
    obs = observations(env)
    if not hidden_global_vars_dim:
        obs["hidden_global_vars"] = None
    actions, _, rollout_values = policy(**obs, deterministic=True)
    torch.testing.assert_close(rollout_values, policy.predict_values(**obs))
    torch.testing.assert_close(actions, policy.act(**obs, deterministic=True))
    metadata = policy.get_hyper_parameters()["mappo_policy_config"]
    assert not metadata["critic_encoder_config"]["use_agent_attention"]
    assert metadata["critic_encoder_config"]["joint_obs_embedding"] is joint_obs_embedding
    assert "critic_interaction_config" in metadata


def test_mappo_actor_remains_local_while_critic_communicates_between_agents() -> None:
    env = make_env()
    policy = make_policy(env)
    obs = observations(env)
    changed = {**obs, "local_obs": obs["local_obs"].detach().clone()}
    changed["local_obs"][:, 2] += 5
    original_values, original_features = policy._evaluate_critic(**obs)
    changed_values, changed_features = policy._evaluate_critic(**changed)
    torch.testing.assert_close(
        policy.act(**obs, deterministic=True)[:, 0], policy.act(**changed, deterministic=True)[:, 0],
    )
    assert not torch.allclose(original_features[0, 0], changed_features[0, 0])
    assert not torch.allclose(original_values[0], changed_values[0])


@pytest.mark.cuda
@pytest.mark.parametrize("use_popart", [False, True])
@pytest.mark.parametrize("joint_obs_embedding", [False, True])
def test_mappo_compiled_deepset_critic_matches_eager_values_and_gradients(
    use_popart: bool, joint_obs_embedding: bool,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.compiler.reset()
    try:
        torch.manual_seed(42)
        env = make_env()
        options = dict(use_popart=use_popart, joint_obs_embedding=joint_obs_embedding)
        eager = make_policy(env, **options).to("cuda")
        compiled = make_policy(env, **options, compile_modules=True).to("cuda")
        compiled.load_state_dict(align_torch_compile_state_dict_keys(
            eager.state_dict(), target_keys=compiled.state_dict().keys(),
        ))
        obs = observations(env, device="cuda")
        expected = eager.predict_values(**obs)
        actual = compiled.predict_values(**obs)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)
        expected.sum().backward()
        actual.sum().backward()
        for actual_parameter, expected_parameter in zip(compiled.parameters(), eager.parameters(), strict=True):
            if expected_parameter.grad is None:
                assert actual_parameter.grad is None
            else:
                torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=1e-4, atol=1e-6)
    finally:
        torch.compiler.reset()
