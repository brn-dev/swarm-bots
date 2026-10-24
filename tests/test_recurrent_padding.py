from copy import deepcopy
from collections.abc import Iterator
from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.utils._pytree import tree_flatten

from swarmbots.learn.algos.off_policy.recurrent_transformer_critic import RecurrentTransformerTwinCritic
from swarmbots.learn.algos.off_policy.transformer_critic import TransformerCriticConfig
from swarmbots.learn.algos.off_policy.actor_state_critic import ActorStateCriticInputConfig
from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoder, RMATEncoderConfig
from swarmbots.learn.algos.r_mat.temporal_sequence_model import (
    LSTMTemporalSequenceModel, LSTMTemporalSequenceModelConfig, TemporalSequenceModel,
)
from swarmbots.learn.algos.xlstm.mlstm import MLSTMTemporalSequenceModel, MLSTMTemporalSequenceModelConfig
from swarmbots.learn.algos.xlstm.slstm import SLSTMTemporalSequenceModel, SLSTMTemporalSequenceModelConfig
from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.sac.recurrent_tmasac_policy import RecurrentTMASACPolicy
from swarmbots.learn.algos.td3.recurrent_td3_policy import RecurrentTD3Policy, RecurrentTD3PolicyConfig
from tests.test_recurrent_tmasac import (
    _DummyContinuousEnv, _actor_state_critic_inputs, _encoder_config, _policy_config, _shared_recurrent_policy_config,
)


@pytest.fixture(scope="module", autouse=True)
def single_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _temporal_model(kind: str) -> TemporalSequenceModel:
    if kind == "lstm":
        return LSTMTemporalSequenceModel(8)
    if kind == "slstm":
        return SLSTMTemporalSequenceModel(8, SLSTMTemporalSequenceModelConfig(num_heads=2))
    return MLSTMTemporalSequenceModel(8, MLSTMTemporalSequenceModelConfig(
        num_heads=2, use_parallel_sequence=kind == "mlstm_parallel",
    ))


@pytest.mark.parametrize("kind", ["lstm", "slstm", "mlstm_parallel", "mlstm_recurrent"])
def test_temporal_inference_does_not_copy_inputs_for_training_only_sanitization(kind: str) -> None:
    model = _temporal_model(kind)
    projection = model.lstm if kind == "lstm" else model.input_projection if kind == "slstm" else model.qkv_projection
    pointers: list[int] = []
    hook = projection.register_forward_pre_hook(lambda _module, args: pointers.append(args[0].data_ptr()))
    inputs = torch.randn(2, 3, 8)
    mask = torch.tensor([[True, False, True], [True, True, False]])
    try:
        with torch.no_grad():
            model(inputs, valid_mask=mask)
        assert pointers[0] == inputs.data_ptr()
    finally:
        hook.remove()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=[
    pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
])])
@pytest.mark.parametrize("kind", ["lstm", "slstm", "mlstm_parallel", "mlstm_recurrent"])
def test_temporal_models_ignore_nonfinite_padding_in_outputs_states_and_gradients(device: str, kind: str) -> None:
    torch.manual_seed(8)
    model = _temporal_model(kind).to(device)
    reference = deepcopy(model)
    mask = torch.tensor([[True, False, True, False], [False, True, True, False]], device=device)
    reset = torch.tensor([[True, False, False, False], [False, True, False, False]], device=device)
    inputs = torch.randn(2, 4, 8, device=device)
    clean = inputs.masked_fill(~mask.unsqueeze(-1), 0).requires_grad_()
    padded = inputs.masked_fill(~mask.unsqueeze(-1), torch.nan).requires_grad_()
    kwargs = dict(valid_mask=mask, reset_mask=reset, state_output_indices=torch.tensor([[0, 2], [1, 3]], device=device))
    expected = reference(clean, **kwargs)
    actual = model(padded, **kwargs)
    for tensor, expected_tensor in zip(tree_flatten(actual)[0], tree_flatten(expected)[0], strict=True):
        torch.testing.assert_close(tensor, expected_tensor)
    expected[0].square().sum().backward()
    actual[0].square().sum().backward()
    torch.testing.assert_close(padded.grad, clean.grad)
    assert torch.count_nonzero(padded.grad[~mask]) == 0
    assert padded.grad[mask].abs().sum() > 0
    for (name, parameter), (_, reference_parameter) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True,
    ):
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, msg=name)


def _compare_padding_gradients(
    model: nn.Module,
    clean_inputs: dict[str, torch.Tensor | None],
    padded_inputs: dict[str, torch.Tensor | None],
    *,
    encode_only: bool = False,
) -> None:
    reference = deepcopy(model)
    expected_outputs = reference.encode(**clean_inputs) if encode_only else reference(**clean_inputs)
    actual_outputs = model.encode(**padded_inputs) if encode_only else model(**padded_inputs)
    # Temporal histories need not contribute to the training objective.
    expected = expected_outputs[:-1]
    actual = actual_outputs[:-1]
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor)
    sum(tensor.square().sum() for tensor in expected).backward()
    sum(tensor.square().sum() for tensor in actual).backward()
    for (name, parameter), (_, expected_parameter) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True,
    ):
        if expected_parameter.grad is None:
            assert parameter.grad is None, name
        else:
            assert torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, expected_parameter.grad, msg=name)


@pytest.mark.parametrize("joint_obs_embedding", [False, True])
def test_rmat_padding_does_not_change_outputs_or_parameter_gradients(joint_obs_embedding: bool) -> None:
    torch.manual_seed(5)
    model = RMATEncoder(
        RMATEncoderConfig(
            d_model=8, nhead=2, num_layers=1, dim_feedforward=16,
            joint_obs_embedding=joint_obs_embedding,
        ),
        max_agents=3, local_obs_dim=3, global_obs_dim=2,
    )
    agent_mask = torch.tensor([[[True, True, False]] * 3] * 2)
    time_mask = torch.tensor([[True, True, True], [True, True, False]])
    valid = agent_mask & time_mask.unsqueeze(-1)
    local = torch.randn(2, 3, 3, 3)
    global_obs = torch.randn(2, 3, 2)
    clean_inputs = {
        "local_obs": local.masked_fill(~valid.unsqueeze(-1), 0),
        "global_obs": global_obs.masked_fill(~time_mask.unsqueeze(-1), 0),
        "agent_mask": agent_mask,
        "time_mask": time_mask,
    }
    padded_inputs = {
        **clean_inputs,
        "local_obs": local.masked_fill(~valid.unsqueeze(-1), torch.nan),
        "global_obs": global_obs.masked_fill(~time_mask.unsqueeze(-1), torch.nan),
    }
    _compare_padding_gradients(model, clean_inputs, padded_inputs)


@pytest.mark.parametrize("separate_encoders", [False, True])
@pytest.mark.parametrize("independent_encoders", [False, True])
@pytest.mark.parametrize("sequence", [False, True])
@pytest.mark.parametrize("encode_only", [False, True])
def test_recurrent_critic_padding_keeps_gradients_finite(
    separate_encoders: bool, independent_encoders: bool, sequence: bool, encode_only: bool,
) -> None:
    torch.manual_seed(6)
    critic = RecurrentTransformerTwinCritic(
        n_agents=3, max_agents=3, local_input_dim=3, global_input_dim=2,
        hidden_local_vars_dim=1, hidden_global_vars_dim=1, action_dim=2,
        encoder_config=RMATEncoderConfig(d_model=8, nhead=2, num_layers=1, dim_feedforward=16),
        critic_config=TransformerCriticConfig(
            separate_observation_action_encoders=separate_encoders,
            independent_encoders=independent_encoders,
            action_coembed_hidden_dims=[8], local_projection_hidden_dims=[8],
            value_regressor_hidden_dims=[8],
        ),
        dropout=0,
    )
    prefix = (2, 3) if sequence else (2,)
    agent_mask = torch.ones((*prefix, 3), dtype=torch.bool)
    agent_mask[..., -1] = False
    time_mask = torch.ones(prefix, dtype=torch.bool)
    time_mask[-1] = False
    valid = agent_mask & time_mask.unsqueeze(-1)
    inputs = {
        "local_inputs" if encode_only else "local_obs": torch.randn(*prefix, 3, 3),
        "global_inputs" if encode_only else "global_obs": torch.randn(*prefix, 2),
        "actions": torch.randn(*prefix, 3, 2),
        "hidden_local_vars": torch.randn(*prefix, 3, 1),
        "hidden_global_vars": torch.randn(*prefix, 1),
    }
    clean_inputs = {
        key: value.masked_fill(~(valid if value.ndim == len(prefix) + 2 else time_mask).unsqueeze(-1), 0)
        for key, value in inputs.items()
    }
    padded_inputs = {
        key: value.masked_fill(~(valid if value.ndim == len(prefix) + 2 else time_mask).unsqueeze(-1), torch.nan)
        for key, value in inputs.items()
    }
    clean_inputs.update(agent_mask=agent_mask, time_mask=time_mask)
    padded_inputs.update(agent_mask=agent_mask, time_mask=time_mask)
    _compare_padding_gradients(critic, clean_inputs, padded_inputs, encode_only=encode_only)


def _recurrent_policy(
    family: str, actor_state_input: bool, device: str,
) -> RecurrentTMASACPolicy | RecurrentTD3Policy:
    encoder = _encoder_config(LSTMTemporalSequenceModel, LSTMTemporalSequenceModelConfig())
    state_config = ActorStateCriticInputConfig() if actor_state_input else None
    if family == "sac_shared":
        policy = RecurrentTMASACPolicy(_DummyContinuousEnv(), _shared_recurrent_policy_config())
    elif family == "sac":
        policy = RecurrentTMASACPolicy(_DummyContinuousEnv(), replace(
            _policy_config(encoder), actor_state_critic_input_config=state_config,
        ))
    else:
        policy = RecurrentTD3Policy(_DummyContinuousEnv(), RecurrentTD3PolicyConfig(
            actor_encoder_config=encoder,
            critic_encoder_config=MATEncoderConfig(d_model=8, nhead=2, num_layers=1, dim_feedforward=16),
            actor_state_critic_input_config=state_config,
            critic_kind=family.removeprefix("td3_"),
        ))
    return policy.to(device)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=[
    pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
])])
@pytest.mark.parametrize("family,actor_state_input,precomputed,num_action_samples", [
    (family, actor_state, False, 1)
    for family in ("sac", "td3_transformer", "td3_mlp", "td3_deepset")
    for actor_state in (False, True)
] + [
    ("sac_shared", False, precomputed, samples)
    for precomputed in (False, True) for samples in (1, 3)
] + [
    ("sac", actor_state, False, 3) for actor_state in (False, True)
])
@pytest.mark.parametrize("sequence", ["sequence", "single", "singleton_time_mask"])
@pytest.mark.parametrize("mask_mode", ["both", "agent_only", "time_only", "neither"])
def test_feedforward_critics_ignore_padding_in_recurrent_policy_apis(
    device: str, family: str, actor_state_input: bool, precomputed: bool,
    num_action_samples: int, sequence: str, mask_mode: str,
) -> None:
    torch.manual_seed(7)
    policy = _recurrent_policy(family, actor_state_input, device)
    reference = deepcopy(policy)
    inputs = _actor_state_critic_inputs(sequence_length=3 if sequence == "sequence" else None)
    inputs = {name: value.to(device) for name, value in inputs.items()}
    inputs["agent_mask"][..., -1] = False
    inputs["agent_mask"][1] = False
    time_mask = (
        torch.tensor([[True, False, True], [True, True, False]], device=device)
        if sequence == "sequence"
        else torch.tensor([True, False], device=device)
    )
    if mask_mode in ("time_only", "neither"):
        inputs["agent_mask"] = None
    agent_valid = torch.ones(inputs["local_obs"].shape[:-1], dtype=torch.bool, device=device)
    if inputs["agent_mask"] is not None:
        agent_valid &= inputs["agent_mask"]
    if mask_mode in ("agent_only", "neither"):
        time_mask = torch.ones_like(time_mask)
    agent_valid &= time_mask.unsqueeze(-1)
    if actor_state_input:
        state = policy.initial_temporal_state(2, 3, device=torch.device(device), dtype=torch.float32)
        if family != "sac":
            state = state["actor"]
        actor_state = policy.actor_state_critic_input(state)
        inputs["actor_state"] = (
            actor_state[:, None].expand(-1, 3, -1, -1) if sequence == "sequence" else actor_state
        )
    if num_action_samples > 1:
        inputs["actions"] = torch.stack([inputs["actions"] + offset * 0.1 for offset in range(num_action_samples)])
    clean = {}
    padded = {}
    for name, value in inputs.items():
        if name == "agent_mask":
            clean[name] = padded[name] = value
            continue
        valid = agent_valid if name in ("local_obs", "hidden_local_vars", "actions", "actor_state") else time_mask
        clean[name] = value.masked_fill(~valid.unsqueeze(-1), 0.0).requires_grad_()
        padded[name] = value.masked_fill(~valid.unsqueeze(-1), torch.nan).requires_grad_()
    mask_argument = time_mask[:, None] if sequence == "singleton_time_mask" else time_mask
    if mask_mode in ("agent_only", "neither"):
        mask_argument = None
    if precomputed:
        for model, observations in ((reference, clean), (policy, padded)):
            shared_latents, shared_state = model.shared_observation_encoder(
                observations["local_obs"], observations["global_obs"],
                agent_mask=observations["agent_mask"], time_mask=mask_argument,
            )
            observations.update(precomputed_shared_latents=shared_latents, precomputed_shared_state=shared_state)
    expected = reference.q_values_sequence(
        **clean, target=False, time_mask=mask_argument, num_action_samples=num_action_samples,
    )
    actual = policy.q_values_sequence(
        **padded, target=False, time_mask=mask_argument, num_action_samples=num_action_samples,
    )
    output_mask = time_mask if num_action_samples == 1 else time_mask.expand(num_action_samples, *time_mask.shape)
    for result, expected_result in zip(actual[:2], expected[:2], strict=True):
        torch.testing.assert_close(result[output_mask], expected_result[output_mask])
    sum(value[output_mask].square().sum() for value in expected[:2]).backward()
    sum(value[output_mask].square().sum() for value in actual[:2]).backward()
    for (name, parameter), (_, reference_parameter) in zip(
        policy.named_parameters(), reference.named_parameters(), strict=True,
    ):
        if reference_parameter.grad is None:
            assert parameter.grad is None, name
        else:
            assert torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, reference_parameter.grad, msg=name)
    for name in ("local_obs", "global_obs", "actions", "hidden_local_vars", "hidden_global_vars"):
        torch.testing.assert_close(padded[name].grad, clean[name].grad)
        assert torch.isfinite(padded[name].grad).all(), name
        valid = agent_valid if name in ("local_obs", "hidden_local_vars", "actions") else time_mask
        assert padded[name].grad.masked_select(valid.unsqueeze(-1)).abs().sum() > 0, name


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=[
    pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
])])
@pytest.mark.parametrize("family,entry_point", [
    ("sac", "forward"), ("sac", "encode"),
    ("td3_mlp", "forward"), ("td3_mlp", "forward_with_latents"),
    ("td3_deepset", "forward"), ("td3_deepset", "encode"),
])
@pytest.mark.parametrize("compiled", [False, True])
def test_actor_state_critic_entry_points_mask_padding_before_projection(
    device: str, family: str, entry_point: str, compiled: bool,
) -> None:
    torch.manual_seed(11)
    policy = _recurrent_policy(family, True, device)
    critic = policy.critic[0] if isinstance(policy.critic, nn.ModuleList) else policy.critic
    reference = deepcopy(critic)
    inputs = {name: value.to(device) for name, value in _actor_state_critic_inputs().items()}
    inputs["agent_mask"][..., -1] = False
    inputs["agent_mask"][1] = False
    state = policy.initial_temporal_state(2, 3, device=torch.device(device), dtype=torch.float32)
    if family != "sac":
        state = state["actor"]
    inputs["actor_state"] = torch.randn_like(policy.actor_state_critic_input(state))
    clean, padded = {}, {}
    for name, value in inputs.items():
        if name == "agent_mask":
            clean[name] = padded[name] = value
            continue
        if name in ("local_obs", "hidden_local_vars", "actions", "actor_state"):
            mask = ~inputs["agent_mask"].unsqueeze(-1)
            clean[name] = value.masked_fill(mask, 0.0).requires_grad_()
            padded[name] = value.masked_fill(mask, torch.nan).requires_grad_()
        else:
            clean[name] = value.clone().requires_grad_()
            padded[name] = value.clone().requires_grad_()
    if entry_point == "encode":
        for observations in (clean, padded):
            observations["local_inputs"] = observations.pop("local_obs")
            observations["global_inputs"] = observations.pop("global_obs")
    forward = getattr(critic, entry_point)
    if compiled:
        torch.compiler.reset()
        forward = torch.compile(forward, backend="aot_eager", fullgraph=True)
    try:
        expected = [value for value in tree_flatten(getattr(reference, entry_point)(**clean))[0] if torch.is_tensor(value)]
        actual = [value for value in tree_flatten(forward(**padded))[0] if torch.is_tensor(value)]
        for result, expected_result in zip(actual, expected, strict=True):
            torch.testing.assert_close(result, expected_result)
        sum(value.square().sum() for value in expected).backward()
        sum(value.square().sum() for value in actual).backward()
        for (name, parameter), (_, reference_parameter) in zip(
            critic.named_parameters(), reference.named_parameters(), strict=True,
        ):
            if reference_parameter.grad is None:
                assert parameter.grad is None, name
            else:
                assert torch.isfinite(parameter.grad).all(), name
                torch.testing.assert_close(parameter.grad, reference_parameter.grad, msg=name)
        assert clean["actor_state"].grad is None
        assert padded["actor_state"].grad is None
        for name, value in clean.items():
            if name not in ("agent_mask", "actor_state"):
                torch.testing.assert_close(padded[name].grad, value.grad)
    finally:
        if compiled:
            torch.compiler.reset()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=[
    pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
])])
@pytest.mark.parametrize("family", ["sac", "td3_mlp", "td3_deepset"])
def test_actor_state_critic_inference_retains_input_storage(device: str, family: str) -> None:
    policy = _recurrent_policy(family, True, device)
    critic = policy.critic[0] if isinstance(policy.critic, nn.ModuleList) else policy.critic
    inputs = {name: value.to(device) for name, value in _actor_state_critic_inputs().items()}
    state = policy.initial_temporal_state(2, 3, device=torch.device(device), dtype=torch.float32)
    if family != "sac":
        state = state["actor"]
    inputs["actor_state"] = policy.actor_state_critic_input(state)
    inputs["agent_mask"][..., -1] = False
    pointers = []
    hook = critic.actor_state_encoder.register_forward_pre_hook(lambda _module, args: pointers.append(args[0].data_ptr()))
    try:
        with torch.no_grad():
            critic(**inputs)
        assert pointers == [inputs["actor_state"].data_ptr()]
    finally:
        hook.remove()
