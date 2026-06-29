from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

import swarmbots.learn.recording as recording


def _make_obs(num_envs: int = 2) -> dict[str, torch.Tensor]:
    return {
        "local_obs": torch.zeros(num_envs, 1, 1),
        "global_obs": torch.zeros(num_envs, 1),
        "hidden_local_vars": torch.empty(num_envs, 1, 0),
        "hidden_global_vars": torch.empty(num_envs, 0),
    }


class _RecordingActionDist:
    def __init__(self) -> None:
        self.ep_start_masks: list[torch.Tensor] = []

    def reset_temporal_correlations_on_ep_start(self, mask: torch.Tensor) -> None:
        self.ep_start_masks.append(mask.detach().cpu().clone())


class _RecordingPolicy:
    gsde_enabled = False

    def __init__(self) -> None:
        self.action_dist = _RecordingActionDist()
        self.episode_start_masks: list[torch.Tensor] = []
        self.previous_actions: list[torch.Tensor | None] = []

    def eval(self) -> None:
        pass

    def to(self, device: torch.device) -> None:
        _ = device

    def requires_previous_actions(self) -> bool:
        return True

    def initial_temporal_state(
            self,
            batch_size: int,
            n_agents: int,
            *,
            device: torch.device,
            dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, n_agents, 1, device=device, dtype=dtype)

    def act_with_temporal_state(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor,
            hidden_global_vars: torch.Tensor,
            agent_mask: torch.Tensor | None,
            previous_actions: torch.Tensor | None,
            deterministic: bool,
            temporal_state: torch.Tensor,
            episode_start_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = global_obs, hidden_local_vars, hidden_global_vars, agent_mask, deterministic
        self.episode_start_masks.append(episode_start_mask.detach().cpu().clone())
        self.previous_actions.append(None if previous_actions is None else previous_actions.detach().cpu().clone())
        action_value = float(len(self.episode_start_masks))
        actions = torch.full((local_obs.shape[0], local_obs.shape[1], 1), action_value, device=local_obs.device)
        return actions, temporal_state + 1.0


class _RecordingEnv:
    num_envs = 2
    n_agents = 1
    action_space = SimpleNamespace(total_agent_action_dim=1)

    def __init__(self) -> None:
        self.step_count = 0

    def reset(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        self.step_count = 0
        return _make_obs(), {}

    def step(
            self,
            actions: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        _ = actions
        self.step_count += 1
        terminations = torch.tensor([self.step_count == 3, False])
        truncations = torch.tensor([False, self.step_count == 2])
        return _make_obs(), torch.zeros(2), terminations, truncations, {}

    def render(self) -> np.ndarray:
        return np.zeros((8, 8, 3), dtype=np.uint8)


class _DummyClip:
    def __init__(self, frames: list[np.ndarray], fps: int) -> None:
        self.frames = frames
        self.fps = fps

    def write_videofile(self, video_path: str, logger: Any = None) -> None:
        _ = video_path, logger


def test_record_policy_updates_recurrent_episode_start_mask_for_vector_resets(
        tmp_path: Path,
        monkeypatch: Any,
) -> None:
    monkeypatch.setattr(recording.moviepy.video.io.ImageSequenceClip, "ImageSequenceClip", _DummyClip)
    policy = _RecordingPolicy()

    recording.record_policy(
        env=_RecordingEnv(),
        policy=policy,
        video_folder=str(tmp_path),
        video_name_prefix="episode",
        num_episodes=1,
        deterministic=True,
    )

    expected_masks = [
        torch.tensor([True, True]),
        torch.tensor([False, False]),
        torch.tensor([False, True]),
    ]
    assert len(policy.episode_start_masks) == len(expected_masks)
    assert len(policy.action_dist.ep_start_masks) == len(expected_masks)
    for actual, expected in zip(policy.episode_start_masks, expected_masks, strict=True):
        assert torch.equal(actual, expected)
    for actual, expected in zip(policy.action_dist.ep_start_masks, expected_masks, strict=True):
        assert torch.equal(actual, expected)

    assert policy.previous_actions[0] is not None
    assert torch.equal(policy.previous_actions[0], torch.zeros(2, 1, 1))
    assert policy.previous_actions[2] is not None
    assert torch.equal(policy.previous_actions[2], torch.tensor([[[2.0]], [[0.0]]]))
