from copy import deepcopy
from pathlib import Path

import pytest
import torch

from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import SignMagnitudeBetaConfig
from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.mat.mat_ind_policy import MATIndPolicy, MATIndPolicyConfig
from swarmbots.learn.algos.ppo.ppo import PPO, StepsRolloutMode
from swarmbots.learn.checkpointing import align_torch_compile_state_dict_keys
from swarmbots.learn.nn_components.feed_forward import MLPConfig
from tests.test_ppo_parameter_lr_multipliers import (
    _assert_optimizer_state_equal,
    _make_ppo,
    _step_with_distinct_gradients,
)
from tests.test_ppo_policy_dropout import _make_env


def _make_algorithm(rules: dict[str, float], wrapper: str) -> PPO:
    env = _make_env()
    policy = MATIndPolicy(env, MATIndPolicyConfig(
        encoder_config=MATEncoderConfig(
            d_model=8, nhead=2, num_layers=2, transformer_ff_config=MLPConfig(hidden_dims=[16, 16]),
        ),
        continuous_config=SignMagnitudeBetaConfig(),
    ))
    if wrapper == "whole":
        policy = torch.compile(policy, backend="eager")
    elif wrapper == "encoder":
        policy.actor_encoder = torch.compile(policy.actor_encoder, backend="eager")
    return PPO(
        policy, env, learning_rate=0.001, rollout_mode=StepsRolloutMode(4),
        train_device="cpu", rollout_device="cpu", parameter_lr_multipliers=rules,
    )


WRAPPER_PAIRS = [("none", "none"), ("whole", "none"), ("none", "whole"), ("encoder", "none"), ("none", "encoder")]


@pytest.mark.parametrize("source_wrapper,target_wrapper", WRAPPER_PAIRS)
@pytest.mark.parametrize("strict", [False, True])
def test_equal_rule_mappings_with_different_ownership_reject_before_mutation(
    tmp_path: Path, source_wrapper: str, target_wrapper: str, strict: bool,
) -> None:
    rules = {"0.feedforward": 0.5, "actor_encoder": 0.25, "1.feedforward": 0.5}
    source = _make_algorithm(rules, source_wrapper)
    target = _make_algorithm(dict(reversed(rules.items())), target_wrapper)
    assert source.parameter_lr_multipliers == target.parameter_lr_multipliers
    _step_with_distinct_gradients(source)
    _step_with_distinct_gradients(target)
    checkpoint = tmp_path / "overlapping_groups.pt"
    source.save(checkpoint, optimizer_state_dict=source._get_optimizer_state_dict())
    weights = deepcopy(target.policy.state_dict())
    optimizer_state = deepcopy(target._get_optimizer_state_dict())
    rollout_state = object()
    target._rollout_state = rollout_state

    with pytest.raises(ValueError, match="parameter learning-rate groups"):
        target.load(checkpoint, strict_load_state_dict=strict, restore_env_state=False)

    torch.testing.assert_close(target.policy.state_dict(), weights, rtol=0, atol=0)
    _assert_optimizer_state_equal(target, optimizer_state)
    assert target._rollout_state is rollout_state
    assert target.n_total_updates == target.n_total_timesteps == 0


@pytest.mark.parametrize("source_wrapper,target_wrapper", WRAPPER_PAIRS)
@pytest.mark.parametrize("rules", [{}, {"actor_encoder": 0.5, "critic_encoder": 0.25}])
def test_matching_ownership_restores_exact_moments_and_next_update_across_wrappers(
    tmp_path: Path, source_wrapper: str, target_wrapper: str, rules: dict[str, float],
) -> None:
    source = _make_algorithm(rules, source_wrapper)
    target = _make_algorithm(dict(reversed(rules.items())), target_wrapper)
    _step_with_distinct_gradients(source)
    checkpoint = tmp_path / "matching_groups.pt"
    source.save(checkpoint, optimizer_state_dict=source._get_optimizer_state_dict())

    target.load(checkpoint, restore_env_state=False)

    expected_moments = align_torch_compile_state_dict_keys(
        {name: source.optimizer.state[parameter] for name, parameter in source.policy.named_parameters()},
        target_keys=dict(target.policy.named_parameters()),
    )
    for name, parameter in target.policy.named_parameters():
        torch.testing.assert_close(target.optimizer.state[parameter], expected_moments[name], rtol=0, atol=0)
    _step_with_distinct_gradients(source)
    _step_with_distinct_gradients(target)
    expected_weights = align_torch_compile_state_dict_keys(
        source.policy.state_dict(), target_keys=target.policy.state_dict(),
    )
    torch.testing.assert_close(target.policy.state_dict(), expected_weights, rtol=0, atol=0)


def test_tied_parameter_aliases_are_restored_once(tmp_path: Path) -> None:
    source, _ = _make_ppo({"shared": 0.5})
    target, _ = _make_ppo({"shared": 0.5})
    source.policy.shared_alias = source.policy.shared
    target.policy.shared_alias = target.policy.shared
    _step_with_distinct_gradients(source)
    checkpoint = tmp_path / "tied_parameters.pt"
    source.save(checkpoint, optimizer_state_dict=source._get_optimizer_state_dict())

    target.load(checkpoint, restore_env_state=False)

    _step_with_distinct_gradients(source)
    _step_with_distinct_gradients(target)
    torch.testing.assert_close(target.policy.state_dict(), source.policy.state_dict(), rtol=0, atol=0)


@pytest.mark.parametrize("wrapper,source_prefix", [
    ("whole", "_orig_mod.actor_encoder"), ("encoder", "actor_encoder._orig_mod"),
])
def test_original_compiler_names_resolve_saved_rules_before_alignment(
    tmp_path: Path, wrapper: str, source_prefix: str,
) -> None:
    source = _make_algorithm({source_prefix: 0.5}, wrapper)
    target = _make_algorithm({"actor_encoder": 0.5}, "none")
    _step_with_distinct_gradients(source)
    checkpoint = tmp_path / "compiler_prefix.pt"
    source.save(checkpoint, optimizer_state_dict=source._get_optimizer_state_dict())

    target.load(checkpoint, restore_env_state=False)

    _step_with_distinct_gradients(source)
    _step_with_distinct_gradients(target)
    expected_weights = align_torch_compile_state_dict_keys(
        source.policy.state_dict(), target_keys=target.policy.state_dict(),
    )
    torch.testing.assert_close(target.policy.state_dict(), expected_weights, rtol=0, atol=0)
