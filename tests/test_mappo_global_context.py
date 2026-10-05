from types import SimpleNamespace

import pytest
import torch
from gymnasium import spaces
from torch import nn

from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import SignMagnitudeBetaConfig
from swarmbots.learn.algos.mappo.mappo_actor import MAPPOActorConfig
from swarmbots.learn.algos.mappo.mappo_policy import MAPPOCriticConfig, MAPPOPolicy, MAPPOPolicyConfig
from swarmbots.learn.checkpointing import align_torch_compile_state_dict_keys
from swarmbots.learn.hybrid_action_space import HybridActionSpace
from swarmbots.learn.nn_components.deep_set import DeepSetCriticConfig


@pytest.fixture(scope="module", autouse=True)
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_env(hidden_global_vars_dim=5):
    return SimpleNamespace(
        n_agents=4, local_obs_dim=6, global_obs_dim=3, hidden_local_vars_dim=2,
        hidden_global_vars_dim=hidden_global_vars_dim,
        action_space=HybridActionSpace({"control": spaces.Box(-1.0, 1.0, (4, 2))}),
    )


def make_policy(env, *, context_in_elements=True, use_popart=False, deepset=True, compile_modules=False):
    return MAPPOPolicy(env, MAPPOPolicyConfig(
        actor_config=MAPPOActorConfig(
            hidden_dims=[8], shared_encoder_latent_dim=6, actor_head_hidden_dims=[8], latent_pi_dim=4,
        ),
        critic_config=MAPPOCriticConfig(
            deep_set_config=DeepSetCriticConfig([8], [8]) if deepset else None,
            mlp_hidden_dims=[8], context_in_elements=context_in_elements, use_popart=use_popart,
        ),
        continuous_config=SignMagnitudeBetaConfig(), compile_modules=compile_modules,
    ))


def observations(env, *, device="cpu"):
    return dict(
        local_obs=torch.randn(2, env.n_agents, env.local_obs_dim, device=device, requires_grad=True),
        global_obs=torch.randn(2, env.global_obs_dim, device=device, requires_grad=True),
        hidden_local_vars=torch.randn(2, env.n_agents, env.hidden_local_vars_dim, device=device, requires_grad=True),
        hidden_global_vars=torch.randn(2, env.hidden_global_vars_dim, device=device, requires_grad=True),
        agent_mask=torch.tensor([[True, False, True, False], [False, False, False, False]], device=device),
    )


@pytest.mark.parametrize("context_in_elements", [False, True])
def test_mappo_critic_conditions_elements_and_keeps_direct_context_for_empty_swarms(context_in_elements):
    env = make_env()
    policy = make_policy(env, context_in_elements=context_in_elements)
    local_width = policy.local_latent_dim + env.hidden_local_vars_dim
    encoder = nn.Linear(local_width + (env.hidden_global_vars_dim if context_in_elements else 0), 1, bias=False)
    decoder = nn.Linear(1 + env.hidden_global_vars_dim, 1, bias=False)
    with torch.no_grad():
        encoder.weight.zero_()
        if context_in_elements:
            encoder.weight[0, local_width] = 1
        decoder.weight.copy_(torch.tensor([[2.0, 1.0, 0.0, 0.0, 0.0, 0.0]]))
    policy.critic.deepset.element_encoder = encoder
    policy.critic.deepset.set_decoder = decoder
    local = torch.zeros(2, env.n_agents, local_width)
    global_context = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    mask = torch.tensor([[True, False, True, False], [False, False, False, False]])

    values, latents = policy.critic.forward_with_latents(local, global_context, mask)

    expected_latents = global_context[:, :1, None].expand(-1, env.n_agents, -1) if context_in_elements else (
        torch.zeros(2, env.n_agents, 1)
    )
    torch.testing.assert_close(latents, expected_latents)
    torch.testing.assert_close(policy.critic.encode_elements(local, global_context), expected_latents)
    torch.testing.assert_close(values, torch.tensor([3.0 if context_in_elements else 1.0, 2.0]))
    assert policy.critic.context_after_pool
    values.sum().backward()
    torch.testing.assert_close(global_context.grad, torch.tensor([
        [3.0 if context_in_elements else 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0],
    ]))
    metadata = policy.get_hyper_parameters()["mappo_policy_config"]["critic_config"]
    assert metadata["context_in_elements"] is context_in_elements


@pytest.mark.parametrize("context_in_elements", [False, True])
@pytest.mark.parametrize("use_popart", [False, True])
@pytest.mark.parametrize("deepset", [False, True])
def test_mappo_global_context_preserves_actor_boundary_masking_and_value_gradients(
    context_in_elements, use_popart, deepset,
):
    torch.manual_seed(42)
    env = make_env()
    policy = make_policy(env, context_in_elements=context_in_elements, use_popart=use_popart, deepset=deepset)
    obs = observations(env)
    actions = policy.act(**obs, deterministic=True)
    values = policy.predict_values(**obs)
    privileged_changed = {**obs, "hidden_global_vars": obs["hidden_global_vars"] + 10}
    torch.testing.assert_close(policy.act(**privileged_changed, deterministic=True), actions, rtol=0, atol=0)
    assert not torch.allclose(policy.predict_values(**privileged_changed), values)
    assert policy.has_popart is use_popart

    padded = {**obs}
    for field in ("local_obs", "hidden_local_vars"):
        padded[field] = obs[field].clone()
        padded[field][~obs["agent_mask"]] = 10000
    torch.testing.assert_close(policy.predict_values(**padded), values, rtol=0, atol=0)
    if deepset:
        permutation = torch.tensor([2, 0, 3, 1])
        shuffled = {field: tensor[:, permutation] if field in {"local_obs", "hidden_local_vars", "agent_mask"} else tensor
                    for field, tensor in obs.items()}
        torch.testing.assert_close(policy.predict_values(**shuffled), values)

    values.sum().backward()

    for field in ("local_obs", "global_obs", "hidden_local_vars", "hidden_global_vars"):
        assert obs[field].grad is not None and torch.isfinite(obs[field].grad).all()
        assert obs[field].grad[0].abs().sum() > 0
    for field in ("local_obs", "hidden_local_vars"):
        assert (obs[field].grad[~obs["agent_mask"]] == 0).all()
    assert (obs["global_obs"].grad[1] == 0).all()
    assert obs["hidden_global_vars"].grad[1].abs().sum() > 0
    assert all(parameter.grad is None for parameter in policy.actor.parameters())
    assert all(parameter.grad is None for parameter in policy.action_dist.parameters())
    for module in (policy.shared_encoder, policy.critic):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        assert any(gradient.abs().sum() > 0 for gradient in gradients)


@pytest.mark.parametrize("context_in_elements", [False, True])
def test_mappo_without_privileged_globals_keeps_the_previous_critic_layout(context_in_elements):
    env = make_env(hidden_global_vars_dim=0)
    policy = make_policy(env, context_in_elements=context_in_elements)
    reference = make_policy(env, context_in_elements=False)
    reference.load_state_dict(policy.state_dict())
    obs = observations(env)
    obs["hidden_global_vars"] = None
    assert not policy.critic.context_in_elements and policy.critic.context_after_pool
    torch.testing.assert_close(policy.predict_values(**obs), reference.predict_values(**obs), rtol=0, atol=0)


def test_mappo_flattened_critic_ignores_element_context_setting():
    env = make_env()
    policy = make_policy(env, context_in_elements=True, deepset=False)
    reference = make_policy(env, context_in_elements=False, deepset=False)
    reference.load_state_dict(policy.state_dict())
    obs = observations(env)
    torch.testing.assert_close(policy.predict_values(**obs), reference.predict_values(**obs), rtol=0, atol=0)


def test_mappo_mlp_flattening_masks_agent_slots_and_is_order_sensitive():
    env = make_env()
    policy = make_policy(env, deepset=False)
    local_width = policy.local_latent_dim + env.hidden_local_vars_dim
    policy.critic.value_features = nn.Identity()
    head = nn.Linear(env.n_agents * local_width + env.hidden_global_vars_dim, 1, bias=False)
    with torch.no_grad():
        head.weight.zero_()
        head.weight[0, ::local_width][:env.n_agents] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        head.weight[0, env.n_agents * local_width] = 5
    policy.critic.value_head = head
    local = torch.zeros(2, env.n_agents, local_width)
    local[:, :, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    local.requires_grad_()
    global_context = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    mask = torch.tensor([[True, False, True, False], [False, False, False, False]])
    values = policy.critic(local, global_context, agent_mask=mask)
    torch.testing.assert_close(values, torch.tensor([20.0, 15.0]))
    permutation = torch.tensor([2, 1, 0, 3])
    shuffled = policy.critic(local[:, permutation], global_context, agent_mask=mask[:, permutation])
    torch.testing.assert_close(shuffled, torch.tensor([16.0, 15.0]))
    values.sum().backward()
    assert (local.grad[~mask] == 0).all()
    torch.testing.assert_close(global_context.grad[:, 0], torch.full((2,), 5.0))


@pytest.mark.cuda
@pytest.mark.parametrize("use_popart", [False, True])
@pytest.mark.parametrize("deepset", [False, True], ids=["mlp", "deepset"])
def test_mappo_compiled_global_context_matches_eager_values_and_gradients(use_popart, deepset):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.compiler.reset()
    try:
        torch.manual_seed(42)
        env = make_env()
        eager = make_policy(env, use_popart=use_popart, deepset=deepset).to("cuda")
        compiled = make_policy(env, use_popart=use_popart, deepset=deepset, compile_modules=True).to("cuda")
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
