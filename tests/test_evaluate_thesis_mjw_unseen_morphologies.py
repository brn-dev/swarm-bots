from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.evaluate_thesis_mjw_unseen_morphologies import (
    EvaluationConfig,
    _align_torch_compile_state_dict_keys,
    _serialize_config,
    discover_final_checkpoints,
    evaluate_policy,
    make_pool_seeds,
    resolve_num_envs,
    summarize_episode_metrics,
)


def test_align_torch_compile_state_dict_keys_loads_legacy_compiled_modules_strictly() -> None:
    target = torch.nn.ModuleDict(
        {
            "actor": torch.nn.Linear(3, 2),
            "critic_nop": torch.nn.ModuleDict({"transition_model": torch.nn.Linear(2, 1)}),
        }
    )
    expected = {key: value.detach().clone() for key, value in target.state_dict().items()}
    legacy_compiled = {
        key.replace("actor.", "actor._orig_mod.").replace(
            "critic_nop.transition_model.",
            "critic_nop.transition_model._orig_mod.",
        ): value
        for key, value in expected.items()
    }

    aligned = _align_torch_compile_state_dict_keys(
        legacy_compiled,
        target_keys=target.state_dict().keys(),
    )

    assert aligned.keys() == expected.keys()
    target.load_state_dict(aligned, strict=True)


def test_align_torch_compile_state_dict_keys_rejects_ambiguous_checkpoint_keys() -> None:
    with pytest.raises(ValueError, match="ambiguous torch.compile state-dict keys"):
        _align_torch_compile_state_dict_keys(
            {
                "actor.weight": torch.zeros(1),
                "actor._orig_mod.weight": torch.zeros(1),
            },
            target_keys=("actor.weight",),
        )


def _fake_obs() -> dict[str, torch.Tensor]:
    return {
        "local_obs": torch.zeros((2, 3, 4)),
        "global_obs": torch.zeros((2, 0)),
        "hidden_local_vars": torch.zeros((2, 3, 0)),
        "hidden_global_vars": torch.zeros((2, 1)),
        "agent_mask": torch.ones((2, 3), dtype=torch.bool),
    }


class _FakePolicy:
    def __init__(self) -> None:
        self.episode_start_masks: list[torch.Tensor] = []

    def eval(self) -> None:
        return None

    def requires_previous_actions(self) -> bool:
        return False

    def initial_temporal_state(self, **kwargs: object) -> torch.Tensor:
        _ = kwargs
        return torch.zeros(())

    def act_with_temporal_state(
        self,
        *,
        episode_start_mask: torch.Tensor,
        temporal_state: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = kwargs
        self.episode_start_masks.append(episode_start_mask.clone())
        return torch.zeros((2, 3, 2)), temporal_state + 1


class _FakeEnv:
    num_envs = 2
    n_agents = 3
    action_space = SimpleNamespace(total_agent_action_dim=2)

    def __init__(self) -> None:
        self.step_index = 0

    def reset(self, *, seed: int) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
        assert seed == 123
        self.step_index = 0
        return _fake_obs(), {}

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
        assert actions.shape == (2, 3, 2)
        done_env_idx = self.step_index
        dones = torch.zeros((2,), dtype=torch.bool)
        dones[done_env_idx] = True
        stats = {
            "r": torch.tensor([1.0, 3.0]),
            "l": torch.tensor([10, 20]),
            "progress_reward": torch.tensor([2.0, 4.0]),
            "guidance_reward": torch.tensor([-1.0, 1.0]),
            "success": torch.tensor([0.0, 1.0]),
        }
        self.step_index += 1
        return _fake_obs(), torch.zeros((2,)), dones, torch.zeros_like(dones), {
            "episode": stats,
            "_episode": dones,
        }


def test_serialized_config_uses_json_stable_unit_count_list() -> None:
    config = EvaluationConfig(
        unit_counts=(2, 3),
        pool_size=50,
        episodes=500,
        max_parallel_envs=250,
        pool_seed_base=1_000_000,
        rollout_seed=2_000_000,
        deterministic=True,
        episode_length=512,
    )

    assert _serialize_config(config)["unit_counts"] == [2, 3]


def test_evaluate_policy_uses_same_step_episode_stats_and_resets_recurrent_rows() -> None:
    env = _FakeEnv()
    policy = _FakePolicy()

    result = evaluate_policy(
        env=env,
        policy=policy,
        episode_count=2,
        deterministic=True,
        rollout_seed=123,
        show_progress=False,
    )

    assert result["success_rate_percent"] == 50.0
    assert result["episode_return"] == {
        "mean": 2.0,
        "std": 1.0,
        "min": 1.0,
        "max": 3.0,
    }
    assert torch.equal(policy.episode_start_masks[0], torch.tensor([True, True]))
    assert torch.equal(policy.episode_start_masks[1], torch.tensor([True, False]))


def test_discover_final_checkpoints_selects_latest_final_per_run(tmp_path: Path) -> None:
    group_dir = tmp_path / "group"
    first_models = group_dir / "run-a" / "models"
    second_models = group_dir / "run-b" / "models"
    first_models.mkdir(parents=True)
    second_models.mkdir(parents=True)
    for name in (
        "model_100_steps_final.pt",
        "model_200_steps_final.pt",
        "model_300_steps_stopped.pt",
    ):
        first_models.joinpath(name).touch()
    second_models.joinpath("model_150_steps_final.pt").touch()
    second_models.joinpath("unrelated_final.pt").touch()

    checkpoints = discover_final_checkpoints([group_dir])

    assert checkpoints == [
        (first_models / "model_200_steps_final.pt").resolve(),
        (second_models / "model_150_steps_final.pt").resolve(),
    ]


def test_make_pool_seeds_is_repeatable_and_disjoint_between_unit_counts() -> None:
    four_units = make_pool_seeds(pool_seed_base=1_000_000, unit_count=4, pool_size=50)
    five_units = make_pool_seeds(pool_seed_base=1_000_000, unit_count=5, pool_size=50)
    four_unit_prefix = make_pool_seeds(pool_seed_base=1_000_000, unit_count=4, pool_size=10)

    assert len(four_units) == 50
    assert len(set(four_units)) == 50
    assert four_unit_prefix == four_units[:10]
    assert set(four_units).isdisjoint(five_units)
    assert max(four_units) < min(five_units)


@pytest.mark.parametrize(
    ("episodes", "max_parallel_envs", "expected"),
    [
        (500, 256, 256),
        (100, 32, 32),
        (17, 256, 17),
        (500, 1, 1),
    ],
)
def test_resolve_num_envs_uses_available_parallelism_without_exceeding_episode_count(
    episodes: int,
    max_parallel_envs: int,
    expected: int,
) -> None:
    assert resolve_num_envs(episodes=episodes, max_parallel_envs=max_parallel_envs) == expected


def test_summarize_episode_metrics_reports_population_statistics() -> None:
    result = summarize_episode_metrics(
        {
            "episode_return": [1.0, 3.0],
            "episode_length": [10.0, 20.0],
            "progress_reward": [2.0, 4.0],
            "guidance_reward": [-1.0, 1.0],
            "success": [0.0, 1.0],
        }
    )

    assert result["episode_count"] == 2
    assert result["success_count"] == 1
    assert result["success_rate_percent"] == 50.0
    assert result["episode_return"] == {
        "mean": 2.0,
        "std": 1.0,
        "min": 1.0,
        "max": 3.0,
    }
