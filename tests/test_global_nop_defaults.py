from collections.abc import Iterator
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch
from gymnasium import spaces
from torch import nn

from swarmbots.benchmark.registry import ALL_BENCHMARK_IDS, make_scenario
from swarmbots.learn.algos.off_policy.nop import SACNOPConfig, SACNOPModule
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig
from swarmbots.learn.algos.world_modeling.next_obs_pred_ppo_wrapper import NOPWorldModelConfig, NextObsPredWrapper
from swarmbots.learn.hybrid_action_space import HybridActionSpace
from swarmbots.learn.obs_indices import ObsIndices
from swarmbots.learn.presets.nop import make_next_obs_pred_config
from swarmbots.learn.presets.policy_factory import _is_off_policy_variant, _make_base_policy
from swarmbots.learn.presets.transformer import MATInitGains, MATNormalizationConfig
from swarmbots.learn.swarmbots_obs_indices import build_obs_indices
from swarmbots.learn.training import _add_ppo_nop, _variant_options, list_variants


@pytest.fixture(scope="module", autouse=True)
def single_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(params=[False, True])
def compile_nop(request: pytest.FixtureRequest) -> Iterator[bool]:
    if request.param:
        torch.compiler.reset()
    yield request.param
    if request.param:
        torch.compiler.reset()


def _indices(**overrides: list[int]) -> ObsIndices:
    return ObsIndices(**{**{field.name: [] for field in fields(ObsIndices)}, **overrides})


@pytest.mark.parametrize("benchmark_id", ALL_BENCHMARK_IDS)
def test_registered_tasks_predict_their_public_global_fields(benchmark_id: str) -> None:
    scenario = make_scenario(benchmark_id, compile_reward_kernel=False)
    observations = scenario.get_single_observation_space()
    indices = build_obs_indices(
        env_settings={"scenario": scenario.get_settings()},
        local_obs_dim=observations["local_obs"].shape[-1],
        global_obs_dim=observations["global_obs"].shape[-1],
        hidden_local_vars_dim=observations["hidden_local_vars"].shape[-1],
        hidden_global_vars_dim=observations["hidden_global_vars"].shape[-1] + 1,
    )
    targets = make_next_obs_pred_config(indices)
    width = observations["global_obs"].shape[-1]
    if "MultiPayloadGoal" in benchmark_id:
        offsets = list(range(0, width, 12))
        assert targets.global_scalar_target_indices == [
            offset + feature for offset in offsets for feature in (1, 2, 3, 10, 11)
        ]
        assert targets.global_rot6d_target_indices == [offset + 4 for offset in offsets]
        assert targets.global_scalar_active_mask_indices == [offset for offset in offsets for _ in range(5)]
        assert targets.global_rot6d_active_mask_indices == offsets
    elif "Payload" in benchmark_id:
        assert targets.global_scalar_target_indices == [offset + feature for offset in range(0, width, 9)
                                                        for feature in range(3)]
        assert targets.global_rot6d_target_indices == list(range(3, width, 9))
    else:
        assert targets.global_scalar_target_indices == list(range(width))
        assert targets.global_rot6d_target_indices == []
    assert targets.local_scalar_target_indices
    assert targets.local_angle_target_indices
    assert targets.local_rot6d_target_indices
    assert targets.local_binary_target_indices


@pytest.mark.parametrize("variant", list_variants(include_hidden=True))
def test_registered_policy_builders_enable_public_global_nop_targets(variant: str) -> None:
    indices = _indices(local_scalar_indices=[0], global_scalar_indices=[1, 13], global_rot6d_indices=[4, 16],
                       global_scalar_active_mask_indices=[0, 12], global_rot6d_active_mask_indices=[0, 12])
    env = SimpleNamespace(
        n_agents=2, local_obs_dim=5, global_obs_dim=24, hidden_local_vars_dim=2, hidden_global_vars_dim=3,
        action_space=HybridActionSpace({"actuators": spaces.Box(-1.0, 1.0, shape=(2, 2)),
                                        "connectors": spaces.Box(-1.0, 1.0, shape=(2, 1))}),
    )
    options, _ = _variant_options(variant, model_scale="2.5M")
    options.update(
        enc_d_model=8, enc_nhead=2, dec_d_model=8, dec_nhead=2,
        use_popart=not _is_off_policy_variant(options["policy_variant"]),
        popart_beta=5e-4, popart_init_sigma=0.65, compile_policy_modules=False,
        compile_world_model_modules=False, policy_compile_mode="default", gsde_init_stds=[0.25],
        mat_add_agent_embeddings=False, act_fn_cls=nn.GELU,
        mat_init_gains=MATInitGains(), mat_normalization=MATNormalizationConfig(),
    )
    with torch.device("meta"):
        policy = _make_base_policy(env=env, obs_indices=indices, **options)
        if _is_off_policy_variant(options["policy_variant"]):
            nop = policy.critic_nop
        elif options["use_nop"]:
            nop = _add_ppo_nop(policy, env, indices, options)
        else:
            nop = None
    if not options["use_nop"]:
        assert nop is None
        return
    assert nop.global_scalar_target_indices == [1, 13]
    assert nop.global_rot6d_target_indices == [4, 16]
    assert nop.global_scalar_active_mask_indices == [0, 12]
    assert nop.global_rot6d_active_mask_indices == [0, 12]
    assert nop.global_scalars_predictor is not None
    assert nop.global_rot6ds_predictor is not None


def _module(kind: str) -> NextObsPredWrapper | SACNOPModule:
    targets = make_next_obs_pred_config(_indices(
        local_scalar_indices=[0], global_scalar_indices=[1, 13], global_rot6d_indices=[4, 16],
        global_scalar_active_mask_indices=[0, 12], global_rot6d_active_mask_indices=[0, 12],
    ))
    if kind == "ppo":
        return NextObsPredWrapper(object(), world_model_config=NOPWorldModelConfig(
            n_agents=2, local_latent_dim=8, action_dim=1, d_model_transition_model=8,
            nhead_transition_model=2, num_layers_transition_model=1, dim_feedforward_transition_model=16,
            scalar_loss_fn="smooth_l1", next_obs_pred_config=targets,
        ))
    return SACNOPModule(n_agents=2, source_latent_dim=8, action_dim=1, name="critic", config=SACNOPConfig(
        enabled=True, transition_model_d_model=8, transition_model_nhead=2, transition_model_num_layers=1,
        transition_model_dim_feedforward=16, scalar_loss_fn="smooth_l1", next_obs_pred_config=targets,
    ))


@pytest.mark.parametrize("kind", ["ppo", "off_policy"])
@pytest.mark.parametrize("active", [False, True])
def test_global_payload_prediction_ignores_inactive_slots_before_losses_and_backward(
    kind: str, active: bool, compile_nop: bool,
) -> None:
    torch.manual_seed(91)
    module = _module(kind)
    if compile_nop:
        module._compute_next_obs_pred_loss_fn = torch.compile(
            module._compute_next_obs_pred_loss_impl, backend="aot_eager", fullgraph=True,
        )
    globals_ = torch.full((2, 24), torch.nan)
    globals_[:, (0, 12)] = 0.0
    if active:
        globals_[:, 0] = 1.0
        globals_[:, 1:12] = 0.0
        globals_[:, 4:10] = torch.tensor([1., 0., 0., 0., 1., 0.])
    targets = globals_[:, None].expand(-1, 3, -1).clone()
    if active:
        targets[:, :, 1] = torch.tensor([1., 2., 3.])
        targets[:, :, 4:10] = torch.tensor([0., 1., 0., -1., 0., 0.])
    globals_.requires_grad_()
    targets.requires_grad_()
    loss, metrics = module.compute_next_obs_pred_loss(
        local_latents=torch.randn(2, 2, 8), local_obs=torch.zeros(2, 2, 1),
        next_local_obs=torch.ones(2, 3, 2, 1), actions=torch.randn(2, 3, 2, 1),
        global_obs=globals_, next_global_obs=targets,
        agent_mask=torch.ones(2, 3, 2, dtype=torch.bool), time_mask=torch.ones(2, 3, dtype=torch.bool),
    )
    assert torch.isfinite(loss)
    loss.backward()
    for tensor in (globals_, targets):
        if tensor.grad is not None:
            assert torch.isfinite(tensor.grad).all()
            assert torch.count_nonzero(tensor.grad[..., 12:]) == 0
    scalar_head = next(layer for layer in module.global_scalars_predictor.modules() if isinstance(layer, nn.Linear))
    rotation_head = next(layer for layer in module.global_rot6ds_predictor.modules() if isinstance(layer, nn.Linear))
    assert torch.count_nonzero(scalar_head.weight.grad[1:]) == 0
    assert torch.count_nonzero(rotation_head.weight.grad[3:]) == 0
    if active:
        assert metrics["global_scalar_loss"] > 0
        assert metrics["global_rot6d_loss"] > 0
        assert scalar_head.weight.grad[0].abs().sum() > 0
        assert rotation_head.weight.grad[:3].abs().sum() > 0
    else:
        assert metrics["global_scalar_loss"] == metrics["global_rot6d_loss"] == 0
        assert torch.count_nonzero(scalar_head.weight.grad) == 0
        assert torch.count_nonzero(rotation_head.weight.grad) == 0


def test_zero_valued_goal_coordinates_remain_valid_global_targets() -> None:
    module = SACNOPModule(n_agents=1, source_latent_dim=2, action_dim=1, name="critic", config=SACNOPConfig(
        enabled=True, transition_model_d_model=2, transition_model_nhead=1, transition_model_num_layers=1,
        scalar_loss_fn="smooth_l1", next_obs_pred_config=NextObsPredConfig(
            global_scalar_target_indices=[0, 1, 2],
        ),
    ))
    module.global_scalars_predictor = nn.Linear(2, 3)
    nn.init.zeros_(module.global_scalars_predictor.weight)
    nn.init.ones_(module.global_scalars_predictor.bias)
    loss = module.compute_global_scalar_loss(torch.zeros(1, 2), torch.zeros(1, 3), None, torch.zeros(1, 3))
    assert loss > 0
    loss.backward()
    assert torch.count_nonzero(module.global_scalars_predictor.bias.grad) == 3
