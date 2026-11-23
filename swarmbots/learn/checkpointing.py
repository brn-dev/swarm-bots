import hashlib
import io
import json
import pathlib
import zipfile
from collections.abc import Callable, Collection, Iterable, Mapping
from types import SimpleNamespace
from typing import Any, BinaryIO

import torch
from filelock import FileLock
from torch.nn.parameter import is_lazy

from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_wrapper import TorchFeatureWiseObsNormWrapper
from swarmbots.learn.torch_device import as_device
from swarmbots.utils.atomic_file import atomic_output, publish_staged_output, staged_output


_BEST_ARCHIVE_COMMENT = b"swarmbots:pending-best:sha256:"
_SHA256_HEX_SIZE = 2 * hashlib.sha256().digest_size


def load_checkpoint(path: str | pathlib.Path, map_location: Any | None = "cpu") -> dict[str, Any]:
    return torch.load(pathlib.Path(path), map_location=map_location, weights_only=False)


def _checkpoint_archive_digest(stream: BinaryIO) -> bytes:
    end = stream.seek(-_SHA256_HEX_SIZE, io.SEEK_END)
    stream.seek(0)
    digest = hashlib.sha256()
    chunk_size = 1024 * 1024
    for offset in range(0, end, chunk_size):
        digest.update(stream.read(min(chunk_size, end - offset)))
    return digest.hexdigest().encode("ascii")


def _seal_best_checkpoint(path: pathlib.Path) -> None:
    # A standard ZIP comment keeps the proof in the exact PyTorch artifact, independent of pickle order.
    with zipfile.ZipFile(path, "a") as archive:
        archive.comment = _BEST_ARCHIVE_COMMENT + b"0" * _SHA256_HEX_SIZE
    with path.open("r+b") as stream:
        digest = _checkpoint_archive_digest(stream)
        stream.seek(-_SHA256_HEX_SIZE, io.SEEK_END)
        stream.write(digest)


def _has_valid_best_checkpoint_seal(stream: BinaryIO) -> bool:
    try:
        stream.seek(-len(_BEST_ARCHIVE_COMMENT) - _SHA256_HEX_SIZE, io.SEEK_END)
        if stream.read(len(_BEST_ARCHIVE_COMMENT)) != _BEST_ARCHIVE_COMMENT:
            return False
        expected = stream.read(_SHA256_HEX_SIZE)
        return expected == _checkpoint_archive_digest(stream)
    except OSError:
        return False


def load_training_checkpoint(
    path: str | pathlib.Path, map_location: Any | None = "cpu",
) -> tuple[dict[str, Any], bool]:
    # Verify the same open file that supplied the payload, even if another writer replaces its name.
    with pathlib.Path(path).open("rb") as stream:
        checkpoint = torch.load(stream, map_location=map_location, weights_only=False)
        verified = checkpoint.get("pending_best_archive_checksum") is True and _has_valid_best_checkpoint_seal(stream)
        return checkpoint, verified


def write_checkpoint(path: pathlib.Path, payload: dict[str, Any], metadata: dict[str, Any]) -> None:
    metadata_path = path.with_name(f"{path.name}.json")
    # Match the sidecar's length bound and let the filesystem resolve aliases of the same name.
    lock_path = path.with_name(f".{path.name}.lck")
    seal = payload.get("pending_best_extra_state") is not None
    if seal:
        payload = {**payload, "pending_best_archive_checksum": True}
    with staged_output(path) as staged_model:
        torch.save(payload, staged_model)
        if seal:
            _seal_best_checkpoint(staged_model)
        # Two writers can serialize in parallel. Publish the sidecar and model as one writer at a time.
        with FileLock(lock_path):
            previous_metadata = metadata_path.read_bytes() if metadata_path.exists() else None
            with atomic_output(metadata_path) as staged_metadata:
                staged_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
            try:
                publish_staged_output(staged_model, path)
            except OSError:
                if previous_metadata is None:
                    metadata_path.unlink()
                else:
                    with atomic_output(metadata_path) as rollback:
                        rollback.write_bytes(previous_metadata)
                raise


def _canonical_policy_key(key: str) -> str:
    return ".".join(part for part in key.split(".") if part != "_orig_mod")


def _is_module_extra_state(key: str) -> bool:
    return key.rsplit(".", 1)[-1] == "_extra_state"


def align_torch_compile_state_dict_keys(
    state_dict: Mapping[str, Any],
    *,
    target_keys: Iterable[str],
) -> dict[str, Any]:
    """Align state-dict keys across compiled and uncompiled module wrappers."""

    target_by_canonical_key: dict[str, str] = {}
    for target_key in target_keys:
        canonical = _canonical_policy_key(target_key)
        previous = target_by_canonical_key.setdefault(canonical, target_key)
        if previous != target_key:
            raise ValueError(
                "Target policy has ambiguous torch.compile state-dict keys: "
                f"{previous!r} and {target_key!r}"
            )

    aligned: dict[str, Any] = {}
    source_by_target_key: dict[str, str] = {}
    for source_key, value in state_dict.items():
        target_key = target_by_canonical_key.get(_canonical_policy_key(source_key), source_key)
        previous_source = source_by_target_key.setdefault(target_key, source_key)
        if previous_source != source_key:
            raise ValueError(
                "Checkpoint has ambiguous torch.compile state-dict keys: "
                f"{previous_source!r} and {source_key!r} both map to {target_key!r}"
            )
        aligned[target_key] = value
    return aligned


def validate_policy_state(
    source: Mapping[str, Any], target: Mapping[str, Any], *, strict: bool,
    parameter_names: Collection[str] = (),
) -> dict[int, torch.Size]:
    # load_state_dict copies matching parameters before it reports errors on later entries.
    if strict and (missing := target.keys() - source.keys()):
        raise RuntimeError(f"Missing key(s) in state_dict: {sorted(missing)}")
    if strict and (unexpected := source.keys() - target.keys()):
        raise RuntimeError(f"Unexpected key(s) in state_dict: {sorted(unexpected)}")
    lazy_shapes: dict[int, torch.Size | None] = {}
    for key in target.keys() & source.keys():
        if _is_module_extra_state(key):
            # PyTorch delegates the shape and representation of this entry to the module.
            continue
        saved, current = source[key], target[key]
        if (not isinstance(saved, torch.Tensor) or saved.layout != current.layout
                or saved.is_meta or saved.is_quantized != current.is_quantized):
            raise ValueError(f"Invalid checkpoint tensor for {key}")
        if is_lazy(current):
            # Lazy modules keep the live parameter or buffer in their state dict and materialize it on load.
            shape = None if is_lazy(saved) else saved.shape
            if lazy_shapes.setdefault(id(current), shape) != shape:
                raise RuntimeError(f"size mismatch for tied lazy checkpoint tensor {key}")
        elif is_lazy(saved):
            raise ValueError(f"Uninitialized checkpoint tensor cannot restore materialized {key}")
        elif saved.shape != current.shape:
            raise RuntimeError(f"size mismatch for {key}: checkpoint {saved.shape}, expected {current.shape}")
        if (
            key in parameter_names and not is_lazy(saved) and saved.layout == torch.strided
            and (current.dtype.is_floating_point or current.dtype.is_complex)
        ):
            restored = saved.detach().to(dtype=current.dtype)
            # FP8 finiteness kernels may be unavailable. Widen only after destination rounding.
            if restored.is_floating_point() and restored.element_size() < 2:
                restored = restored.float()
            if not torch.isfinite(restored).all():
                raise ValueError(f"Non-finite checkpoint parameter {key} in destination dtype {current.dtype}")
    return {identity: shape for identity, shape in lazy_shapes.items() if shape is not None}


def checkpoint_lazy_parameter_shapes(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]], source: Mapping[str, Any],
) -> dict[int, torch.Size]:
    lazy_parameters = {name: parameter for name, parameter in named_parameters if is_lazy(parameter)}
    if not lazy_parameters:
        return {}
    aligned = align_torch_compile_state_dict_keys(source, target_keys=lazy_parameters)
    return validate_policy_state(aligned, lazy_parameters, strict=False)


def snapshot_policy_for_extra_state(
    policy: torch.nn.Module, source: Mapping[str, Any], current: Mapping[str, Any],
) -> Callable[[], None] | None:
    if not any(_is_module_extra_state(key) and key in source for key in current):
        return None
    # A setter can reject its input after PyTorch has installed earlier parameters or mutated nested metadata.
    snapshot = io.BytesIO()
    torch.save(current, snapshot)
    lazy_tensors = [(value, type(value), value.data) for value in current.values() if is_lazy(value)]

    def restore() -> None:
        # Undo LazyModuleMixin's in-place materialization while retaining references held by the optimizer.
        for tensor, lazy_type, empty_data in lazy_tensors:
            if not is_lazy(tensor):
                tensor.data = empty_data
                tensor.__class__ = lazy_type
        snapshot.seek(0)
        policy.load_state_dict(torch.load(snapshot, weights_only=False))

    return restore


def clone_running_mean_std(src: Any) -> Any:
    return SimpleNamespace(
        mean=src.mean.detach().clone(),
        var=src.var.detach().clone(),
        count=src.count.detach().clone(),
    )


def iter_running_mean_std(env: Any) -> list[tuple[str, Any]]:
    rms_entries: list[tuple[str, Any]] = []
    for attr_name in dir(env):
        if not attr_name.endswith("_rms"):
            continue

        value = getattr(env, attr_name, None)
        if value is None:
            continue
        if all(hasattr(value, field) for field in ("mean", "var", "count")):
            rms_entries.append((attr_name, value))
    return rms_entries


def capture_env_state(env: Any) -> list[dict[str, Any]]:
    env_state: list[dict[str, Any]] = []
    current_env = env
    while hasattr(current_env, "env"):
        wrapper_state: dict[str, Any] = {}

        if isinstance(current_env, TorchFeatureWiseObsNormWrapper):
            wrapper_state['obs_key'] = current_env.obs_key

        for attr_name, value in iter_running_mean_std(current_env):
            wrapper_state[attr_name] = clone_running_mean_std(value)

        if wrapper_state:
            wrapper_state["wrapper_class"] = type(current_env).__name__
            env_state.append(wrapper_state)

        current_env = current_env.env
    return env_state


def snapshot_best_candidate_extra_state(policy_state: Mapping[str, Any]) -> bytes | None:
    extra_state = {
        _canonical_policy_key(key): policy_state[key]
        for key in sorted(policy_state, key=_canonical_policy_key) if _is_module_extra_state(key)
    }
    if not extra_state:
        return None
    data = io.BytesIO()
    torch.save(extra_state, data)
    return data.getvalue()


def best_candidate_digest(
    policy_state: Mapping[str, Any], env_state: list[dict[str, Any]],
    counters: tuple[int, int, int], return_ema: float | None, *, extra_state: bytes | None = None,
) -> bytes:
    """Identify the scored training state when retrying a failed best save."""
    digest = hashlib.sha256(json.dumps((counters, return_ema)).encode())

    def add_tensor(name: tuple[Any, ...], tensor: torch.Tensor) -> None:
        if is_lazy(tensor):
            digest.update(json.dumps((name, str(tensor.dtype), "uninitialized")).encode())
            return
        digest.update(json.dumps((name, str(tensor.dtype), tuple(tensor.shape))).encode())
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(raw.numpy())

    for key in sorted(policy_state, key=_canonical_policy_key):
        if not _is_module_extra_state(key):
            add_tensor(("policy", _canonical_policy_key(key)), policy_state[key])
    for index, wrapper in enumerate(env_state):
        for key, rms in wrapper.items():
            if key.endswith("_rms"):
                for field in ("mean", "var", "count"):
                    name = (index, wrapper["wrapper_class"], wrapper.get("obs_key"), key, field)
                    add_tensor(name, getattr(rms, field))
    if extra_state is None:
        extra_state = snapshot_best_candidate_extra_state(policy_state)
    if extra_state is not None:
        digest.update(hashlib.sha256(extra_state).digest())
    return digest.digest()


def restore_best_candidate_digest(
    digest: bytes, archived_extra_state: bytes | None,
    policy_state: Mapping[str, Any], env_state: list[dict[str, Any]],
    counters: tuple[int, int, int], return_ema: float | None, *, verified_archive: bool,
) -> bytes | None:
    if archived_extra_state is None:
        return None if any(map(_is_module_extra_state, policy_state)) else digest
    if not verified_archive:
        return None
    if best_candidate_digest(policy_state, env_state, counters, return_ema, extra_state=archived_extra_state) != digest:
        return None
    # Bind eligibility to the live representation after the module restored the verified saved state.
    return best_candidate_digest(policy_state, env_state, counters, return_ema)


def _prepare_running_stats(saved: Any, current: Any, label: str) -> dict[str, torch.Tensor]:
    fields = {}
    for name in ("mean", "var", "count"):
        src, dst = getattr(saved, name), getattr(current, name)
        if src.shape != dst.shape:
            raise ValueError(f"Invalid {label}.{name} shape {src.shape}, expected {dst.shape}")
        field = src.detach().to(dst, copy=True)
        valid = torch.isfinite(field)
        if name != "mean":
            valid &= field >= 0
        if not valid.all():
            raise ValueError(f"Invalid {label}.{name} values")
        fields[name] = field
    return fields


def prepare_env_state(env: Any, env_state: list[dict[str, Any]]) -> list[tuple[Any, dict[str, torch.Tensor]]]:
    updates = []
    current_env = env
    search_start_idx = 0

    while hasattr(current_env, "env"):
        rms_entries = dict(iter_running_mean_std(current_env))
        has_rms = bool(rms_entries)
        if has_rms:
            current_class = type(current_env).__name__
            match_idx = None
            candidates = [
                i
                for i in range(search_start_idx, len(env_state))
                if env_state[i]["wrapper_class"] == current_class
                and any(key.endswith("_rms") for key in env_state[i])
            ]
            if candidates:
                if isinstance(current_env, TorchFeatureWiseObsNormWrapper):
                    for i in candidates:
                        if env_state[i]["obs_key"] == current_env.obs_key:
                            match_idx = i
                            break
                    if match_idx is None:
                        raise ValueError(
                            f"Obs keys not equal: {current_env.obs_key} vs "
                            f"{[env_state[i]['obs_key'] for i in candidates]}"
                        )
                else:
                    match_idx = candidates[0]

            if match_idx is None:
                raise ValueError(
                    f"No saved env_state entry for wrapper {current_class}; normalization stats not restored for it."
                )
            saved_state = env_state[match_idx]
            for attr_name, value in rms_entries.items():
                fields = _prepare_running_stats(saved_state[attr_name], value, f"{current_class}.{attr_name}")
                updates.append((value, fields))
            search_start_idx = match_idx + 1

        current_env = current_env.env

    return updates


def install_env_state(updates: list[tuple[Any, dict[str, torch.Tensor]]]) -> None:
    for destination, fields in updates:
        for name, value in fields.items():
            setattr(destination, name, value)


def apply_env_state(env: Any, env_state: list[dict[str, Any]]) -> None:
    install_env_state(prepare_env_state(env, env_state))


def freeze_env_normalization(env: Any) -> None:
    current_env = env
    while hasattr(current_env, "env"):
        if hasattr(current_env, "update_running_mean"):
            current_env.update_running_mean = False
        current_env = current_env.env


def move_env_to_device(env: Any, device: torch.device | str) -> None:
    target_device = as_device(device)
    current_env = env
    while True:
        if hasattr(current_env, "set_device"):
            current_env.set_device(target_device)
        elif hasattr(current_env, "device"):
            current_env.device = target_device

        if not hasattr(current_env, "env"):
            break
        current_env = current_env.env
