import pytest
import torch
from torch import nn

from swarmbots.learn.algos.mat.mat_encoder import MATEncoder, MATEncoderConfig
from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoder, RMATEncoderConfig
from swarmbots.learn.algos.sac.recurrent_tmasac_policy import RecurrentTMASACTwinCritic
from swarmbots.learn.algos.sac.tmasac_policy import TMASACActionConditionedEncoder, TMASACCriticConfig
from swarmbots.learn.nn_components.feed_forward import MLPConfig, SwiGLUConfig
from swarmbots.learn.serialization_utils import serialize_dataclass


@pytest.fixture(scope="module", autouse=True)
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_encoder(kind, *, joint=False, global_dim=3, normalize=False, feedforward=None, layers=1):
    config_cls = RMATEncoderConfig if kind == "rmat" else MATEncoderConfig
    config = config_cls(
        d_model=8, nhead=2, num_layers=layers, dim_feedforward=16,
        local_obs_encoder_config=feedforward or MLPConfig([8, 8]),
        global_obs_encoder_config=MLPConfig([8]),
        joint_obs_embedding=joint, normalize_obs_inputs=normalize,
    )
    if kind == "critic":
        return TMASACActionConditionedEncoder(
            max_agents=3, local_input_dim=4, global_input_dim=global_dim, hidden_local_vars_dim=0,
            hidden_global_vars_dim=0, action_dim=2, encoder_config=config,
            critic_config=TMASACCriticConfig(), act_fn_cls=nn.GELU,
        )
    cls = RMATEncoder if kind == "rmat" else MATEncoder
    return cls(config, max_agents=3, local_obs_dim=4, global_obs_dim=global_dim)


def encoder_inputs(kind, *, global_dim=3, device="cpu"):
    inputs = dict(
        local_obs=torch.randn(2, 3, 4, device=device, requires_grad=True),
        global_obs=torch.randn(2, global_dim, device=device, requires_grad=True),
        agent_mask=torch.tensor([[True, False, True], [True, True, True]], device=device),
    )
    if kind == "critic":
        inputs = dict(
            local_inputs=inputs["local_obs"], global_inputs=inputs["global_obs"], agent_mask=inputs["agent_mask"],
            actions=torch.randn(2, 3, 2, device=device, requires_grad=True),
            hidden_local_vars=None, hidden_global_vars=None,
        )
    return inputs


def run_encoder(kind, encoder, inputs):
    if kind == "critic":
        return encoder(**inputs)
    result = encoder(inputs["local_obs"], inputs["global_obs"], agent_mask=inputs["agent_mask"])
    return result[0] if kind == "rmat" else result


@pytest.mark.parametrize("kind", ["mat", "rmat", "critic"])
def test_default_embedding_matches_explicit_additive_state_and_forward(kind):
    torch.manual_seed(42)
    default = make_encoder(kind)
    torch.manual_seed(42)
    explicit = make_encoder(kind, joint=False)
    assert not default.joint_obs_embedding
    torch.testing.assert_close(default.state_dict(), explicit.state_dict(), rtol=0, atol=0)
    assert any("global_obs_encoder" in key or "global_encoder" in key for key in default.state_dict())
    local_encoder = default.local_action_encoder if kind == "critic" else default.local_obs_encoder
    first_linear = next(module for module in local_encoder.modules() if isinstance(module, nn.Linear))
    assert first_linear.in_features == (6 if kind == "critic" else 4)
    inputs = encoder_inputs(kind)
    torch.testing.assert_close(run_encoder(kind, default, inputs), run_encoder(kind, explicit, inputs), rtol=0, atol=0)
    assert serialize_dataclass(MATEncoderConfig())["joint_obs_embedding"] is False


@pytest.mark.parametrize("kind", ["mat", "rmat"])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("feedforward", [MLPConfig([8, 8]), SwiGLUConfig(hidden_dim=12)])
def test_joint_observation_input_keeps_separate_normalization_and_reaches_both_fields(kind, normalize, feedforward):
    torch.manual_seed(42)
    encoder = make_encoder(kind, joint=True, normalize=normalize, feedforward=feedforward)
    inputs = encoder_inputs(kind)
    captured = []
    hook = encoder.local_obs_encoder.register_forward_pre_hook(lambda _module, args: captured.append(args[0]))
    try:
        output = run_encoder(kind, encoder, inputs)
    finally:
        hook.remove()
    normalized_local = encoder.local_obs_input_norm(inputs["local_obs"])
    normalized_global = encoder.global_obs_input_norm(inputs["global_obs"])
    expected_input = torch.cat((normalized_local, normalized_global.unsqueeze(1).expand(-1, 3, -1)), dim=-1)
    if kind == "rmat":
        expected_input = expected_input.unsqueeze(1)
    torch.testing.assert_close(captured[0], expected_input)
    assert encoder.global_obs_encoder is None and encoder.joint_obs_embedding
    assert output.shape == (2, 3, 8) and torch.isfinite(output).all()
    loss = (output * torch.arange(1, 9)).masked_select(inputs["agent_mask"].unsqueeze(-1)).sum()
    loss.backward()
    for field in ("local_obs", "global_obs"):
        assert torch.isfinite(inputs[field].grad).all() and inputs[field].grad.abs().sum() > 0
    assert (inputs["local_obs"].grad[~inputs["agent_mask"]] == 0).all()
    permutation = torch.tensor([2, 0, 1])
    shuffled = {**inputs, "local_obs": inputs["local_obs"][:, permutation],
                "agent_mask": inputs["agent_mask"][:, permutation]}
    torch.testing.assert_close(run_encoder(kind, encoder, shuffled), output[:, permutation], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("kind", ["mat", "rmat"])
def test_joint_nonlinear_embedding_can_gate_local_features_using_global_context(kind):
    config_cls = RMATEncoderConfig if kind == "rmat" else MATEncoderConfig
    cls = RMATEncoder if kind == "rmat" else MATEncoder
    encoder = cls(config_cls(d_model=4, nhead=2, num_layers=0, joint_obs_embedding=True),
                  max_agents=3, local_obs_dim=3, global_obs_dim=2)
    projection = nn.Linear(5, 4, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.tensor([
            [1.0, 0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0],
        ]))
    encoder.local_obs_encoder = nn.Sequential(projection, nn.ReLU())
    encoder.norm = nn.Identity()
    local = torch.tensor([[[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]).expand(2, -1, -1)
    global_obs = torch.tensor([[-1.0, 3.0], [1.0, 4.0]])
    expected = torch.tensor([
        [[0.0, 0.0, 0.0, 3.0], [0.0, 1.0, 0.0, 3.0], [1.0, 3.0, 0.0, 3.0]],
        [[0.0, 0.0, 0.0, 4.0], [1.0, 0.0, 0.0, 4.0], [3.0, 1.0, 0.0, 4.0]],
    ])
    result = encoder(local, global_obs)
    torch.testing.assert_close(result[0] if kind == "rmat" else result, expected)


@pytest.mark.parametrize("kind", ["mat", "rmat", "critic"])
def test_joint_option_without_globals_preserves_additive_checkpoint_shapes(kind):
    joint = make_encoder(kind, joint=True, global_dim=0)
    additive = make_encoder(kind, global_dim=0)
    joint.load_state_dict(additive.state_dict(), strict=True)
    assert not joint.joint_obs_embedding
    inputs = encoder_inputs(kind, global_dim=0)
    torch.testing.assert_close(run_encoder(kind, joint, inputs), run_encoder(kind, additive, inputs), rtol=0, atol=0)


@pytest.mark.parametrize("joint", [False, True])
def test_joint_recurrent_embedding_matches_stepwise_resets_and_masked_gradients(joint):
    torch.manual_seed(42)
    encoder = make_encoder("rmat", joint=joint)
    local = torch.randn(2, 4, 3, 4, requires_grad=True)
    global_obs = torch.randn(2, 4, 3, requires_grad=True)
    mask = torch.tensor([[[True, False, True]]]).expand(2, 4, -1)
    time_mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    resets = torch.tensor([[True, False, True, False], [True, False, False, False]])
    output, state = encoder(local, global_obs, agent_mask=mask, time_mask=time_mask, reset_mask=resets)
    step_state = None
    steps = []
    for time in range(4):
        step, step_state = encoder(local[:, time], global_obs[:, time], agent_mask=mask[:, time],
                                   time_mask=time_mask[:, time], reset_mask=resets[:, time], initial_state=step_state)
        steps.append(step)
    torch.testing.assert_close(output, torch.stack(steps, dim=1))
    torch.testing.assert_close(state, step_state)
    (output * torch.arange(1, 9)).sum().backward()
    valid = mask & time_mask.unsqueeze(-1)
    assert torch.isfinite(local.grad).all() and (local.grad[~valid] == 0).all()
    assert torch.isfinite(global_obs.grad).all() and (global_obs.grad[~time_mask] == 0).all()
    assert global_obs.grad[time_mask].abs().sum() > 0


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("separate_actions", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("global_dims", [(3, 2), (0, 2), (3, 0), (0, 0)])
def test_action_conditioned_joint_embedding_includes_privileged_globals_and_supports_action_preencoders(
    joint, separate_actions, normalize, global_dims,
):
    torch.manual_seed(42)
    public_dim, hidden_dim = global_dims
    encoder = TMASACActionConditionedEncoder(
        max_agents=3, local_input_dim=4, global_input_dim=public_dim, hidden_local_vars_dim=2,
        hidden_global_vars_dim=hidden_dim, action_dim=2,
        encoder_config=MATEncoderConfig(d_model=8, nhead=2, num_layers=1, joint_obs_embedding=joint,
                                       normalize_obs_inputs=normalize),
        critic_config=TMASACCriticConfig(separate_observation_action_encoders=separate_actions), act_fn_cls=nn.GELU,
    )
    inputs = dict(
        local_inputs=torch.randn(2, 3, 4, requires_grad=True),
        global_inputs=torch.randn(2, public_dim, requires_grad=True),
        hidden_local_vars=torch.randn(2, 3, 2, requires_grad=True),
        hidden_global_vars=torch.randn(2, hidden_dim, requires_grad=True) if hidden_dim else None,
        actions=torch.randn(2, 3, 2, requires_grad=True),
        agent_mask=torch.tensor([[True, False, True], [True, True, True]]),
    )
    captured = []
    hook = encoder.local_action_encoder.register_forward_pre_hook(lambda _module, args: captured.append(args[0]))
    try:
        output = encoder(**inputs)
    finally:
        hook.remove()
    observation_inputs = torch.cat((inputs["local_inputs"], inputs["hidden_local_vars"]), dim=-1)
    local_inputs = encoder.observation_action_encoder(observation_inputs, inputs["actions"]) if separate_actions else (
        torch.cat((observation_inputs, inputs["actions"]), dim=-1)
    )
    expected_input = encoder.local_action_input_norm(local_inputs)
    effective_joint = joint and public_dim + hidden_dim > 0
    if effective_joint:
        parts = [inputs["global_inputs"]] if public_dim else []
        if hidden_dim:
            parts.append(inputs["hidden_global_vars"])
        global_inputs = encoder.global_input_norm(torch.cat(parts, dim=-1))
        expected_input = torch.cat((expected_input, global_inputs.unsqueeze(1).expand(-1, 3, -1)), dim=-1)
    torch.testing.assert_close(captured[0], expected_input)
    assert encoder.joint_obs_embedding is effective_joint
    if effective_joint:
        assert encoder.global_encoder is None
    (output * torch.arange(1, 9)).masked_select(inputs["agent_mask"].unsqueeze(-1)).sum().backward()
    for field in ("local_inputs", "hidden_local_vars", "actions", "global_inputs", "hidden_global_vars"):
        tensor = inputs[field]
        if tensor is not None and tensor.numel():
            assert tensor.grad is not None and torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0
    for field in ("local_inputs", "hidden_local_vars", "actions"):
        assert (inputs[field].grad[~inputs["agent_mask"]] == 0).all()


@pytest.mark.parametrize("separate_actions", [False, True])
def test_recurrent_twin_critics_use_joint_context_for_both_nop_sources(separate_actions):
    torch.manual_seed(42)
    encoder_config = RMATEncoderConfig(d_model=8, nhead=2, num_layers=1, joint_obs_embedding=True)
    critic = RecurrentTMASACTwinCritic(
        n_agents=3, max_agents=3, local_input_dim=4, global_input_dim=3, hidden_local_vars_dim=2,
        hidden_global_vars_dim=2, action_dim=2, encoder_config=encoder_config,
        critic_config=TMASACCriticConfig(independent_encoders=True,
                                       separate_observation_action_encoders=separate_actions), dropout=0.0,
    )
    obs = dict(
        local_obs=torch.randn(2, 3, 4), global_obs=torch.randn(2, 3), actions=torch.randn(2, 3, 2),
        hidden_local_vars=torch.randn(2, 3, 2), hidden_global_vars=torch.randn(2, 2, requires_grad=True),
        agent_mask=torch.ones(2, 3, dtype=torch.bool),
    )
    q1, q2, latents, state = critic(**obs)
    encode_obs = {**obs, "local_inputs": obs["local_obs"], "global_inputs": obs["global_obs"]}
    del encode_obs["local_obs"], encode_obs["global_obs"]
    encoded, encoded_state = critic.encode(**encode_obs)
    torch.testing.assert_close(encoded, latents)
    torch.testing.assert_close(encoded_state, state)
    assert latents.shape == (2, 3, 16)
    assert critic.encoder.joint_obs_embedding and critic.encoder2.joint_obs_embedding
    assert critic.encoder.global_obs_encoder is None and critic.encoder2.global_obs_encoder is None
    (q1.sum() + q2.sum() + (latents * torch.arange(1, 17)).sum()).backward()
    assert torch.isfinite(obs["hidden_global_vars"].grad).all() and obs["hidden_global_vars"].grad.abs().sum() > 0


@pytest.mark.cuda
@pytest.mark.parametrize("kind", ["mat", "rmat", "critic"])
def test_cuda_compiled_joint_embedding_matches_eager_forward_and_gradients(kind):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.compiler.reset()
    try:
        torch.manual_seed(42)
        eager = make_encoder(kind, joint=True).to("cuda")
        compiled_module = make_encoder(kind, joint=True).to("cuda")
        compiled_module.load_state_dict(eager.state_dict())
        compiled = torch.compile(compiled_module)
        inputs = encoder_inputs(kind, device="cuda")
        expected = run_encoder(kind, eager, inputs)
        actual = run_encoder(kind, compiled, inputs)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        (expected * torch.arange(1, 9, device="cuda")).sum().backward()
        (actual * torch.arange(1, 9, device="cuda")).sum().backward()
        # Inductor's LSTM reductions differ from cuDNN's float32 reduction order.
        grad_rtol, grad_atol = (1e-3, 5e-4) if kind == "rmat" else (1e-4, 1e-5)
        for actual_parameter, expected_parameter in zip(compiled_module.parameters(), eager.parameters(), strict=True):
            if expected_parameter.grad is None:
                assert actual_parameter.grad is None
            else:
                torch.testing.assert_close(
                    actual_parameter.grad, expected_parameter.grad, rtol=grad_rtol, atol=grad_atol,
                )
    finally:
        torch.compiler.reset()
