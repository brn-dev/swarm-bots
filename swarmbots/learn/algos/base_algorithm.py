import abc
import json
from pathlib import Path
from typing import Any, Optional

import torch

from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.checkpointing import load_checkpoint, extract_policy_state_dict, extract_optimizer_state_dict, \
    apply_env_state, extract_env_state
from swarmbots.learn.env_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper


class BaseAlgorithm(abc.ABC):

    def __init__(
            self,
            policy: BasePolicy,
            env: BaseLearnEnvWrapper,
    ):
        self.policy = policy
        self.env = env

        self.n_total_iterations = 0
        self.n_total_updates = 0
        self.n_total_timesteps = 0

    @abc.abstractmethod
    def get_hyper_parameters(self) -> dict[str, Any]:
        raise NotImplementedError()

    @abc.abstractmethod
    def _apply_optimizer_state_dict(self, state_dict: dict[str, Any]) -> None:
        raise NotImplementedError()

    def _write_run_metadata(self, run_dir: Path, extra_run_metadata: dict[str, Any] | None) -> None:
        metadata: dict[str, Any] = {
            "algorithm": type(self).__name__,
            "hyper_parameters": self.get_hyper_parameters(),
            "policy_hyper_parameters": self.policy.get_hyper_parameters(),
            "policy_repr": str(self.policy),
            "env_repr": str(self.env),
        }
        if extra_run_metadata:
            metadata.update(extra_run_metadata)
        metadata["timesteps"] = int(self.n_total_timesteps)
        metadata_json: dict[str, Any] = json.loads(json.dumps(metadata, default=str))

        existing_metadata_files = list(run_dir.glob("run_metadata*.json"))
        if existing_metadata_files:
            def _sort_key(path: Path) -> tuple[int, float, str]:
                step = _parse_step_from_metadata_filename(path)
                step_key = step if step is not None else -1
                return step_key, path.stat().st_mtime, path.name

            latest_path = max(existing_metadata_files, key=_sort_key)
            latest_metadata = _read_metadata_json(latest_path)
            if latest_metadata == metadata_json:
                return

        step = int(self.n_total_timesteps)
        base_path = run_dir / f"run_metadata_{step}.json"
        if base_path.exists():
            existing = _read_metadata_json(base_path)
            if existing == metadata_json:
                return

            suffix_idx = 1
            while (run_dir / f"run_metadata_{step}_{suffix_idx}.json").exists():
                suffix_idx += 1
            metadata_path = run_dir / f"run_metadata_{step}_{suffix_idx}.json"
        else:
            metadata_path = base_path

        metadata_path.write_text(json.dumps(metadata_json, indent=2, sort_keys=True), encoding="utf-8")

    def save(
            self,
            path: str | Path,
            optimizer_state_dict: Optional[dict[str, Any]] = None,
            return_ema: Optional[float] = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        env_state = []
        current_env = self.env
        while hasattr(current_env, 'env'):
            wrapper_state = {}
            if hasattr(current_env, 'local_obs_rms'):
                wrapper_state['local_obs_rms'] = current_env.local_obs_rms
            if hasattr(current_env, 'global_obs_rms'):
                wrapper_state['global_obs_rms'] = current_env.global_obs_rms
            if hasattr(current_env, 'return_rms'):
                wrapper_state['return_rms'] = current_env.return_rms

            if wrapper_state:
                wrapper_state['wrapper_class'] = type(current_env).__name__
                env_state.append(wrapper_state)

            current_env = current_env.env

        save_dict = {
            'policy_state_dict': self.policy.state_dict(),
            'env_state': env_state,
            'n_total_iterations': self.n_total_iterations,
            'n_total_updates': self.n_total_updates,
            'n_total_timesteps': self.n_total_timesteps,
            'return_ema': return_ema,
        }

        if optimizer_state_dict is not None:
            save_dict['optimizer_state_dict'] = optimizer_state_dict

        torch.save(save_dict, path)

        metadata = {
            'n_total_iterations': self.n_total_iterations,
            'n_total_updates': self.n_total_updates,
            'n_total_timesteps': self.n_total_timesteps,
            'return_ema': return_ema,
        }
        metadata_path = path.with_name(f"{path.name}.json")
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        checkpoint = load_checkpoint(path)
        self.policy.load_state_dict(extract_policy_state_dict(checkpoint))

        optimizer_state_dict = extract_optimizer_state_dict(checkpoint)
        if optimizer_state_dict is not None:
            self._apply_optimizer_state_dict(optimizer_state_dict)

        if isinstance(checkpoint, dict):
            self.n_total_iterations = checkpoint.get("n_total_iterations", 0)
            self.n_total_updates = checkpoint.get("n_total_updates", 0)
            self.n_total_timesteps = checkpoint.get("n_total_timesteps", 0)

        apply_env_state(self.env, extract_env_state(checkpoint))


def _parse_step_from_metadata_filename(path: Path) -> int | None:
    if path.name == "run_metadata.json":
        return None
    if not (path.name.startswith("run_metadata_") and path.name.endswith(".json")):
        return None

    suffix = path.name.removeprefix("run_metadata_").removesuffix(".json")
    step_str = suffix.split("_", 1)[0]
    if not step_str.isdigit():
        return None
    return int(step_str)

def _read_metadata_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
