"""Count policy parameters once, with an additive breakdown by their role."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from torch import nn

from swarmbots.learn.algos.mat_orig.mat_orig_encoder import MATOrigSelfAttention
from swarmbots.learn.algos.r_mat.temporal_sequence_model import TemporalSequenceModel
from swarmbots.learn.algos.xlstm.head_utils import HeadwiseLinearProjection, MultiHeadLayerNorm
from swarmbots.learn.nn_components.feed_forward import GLU, MLP, StackedGLU
from swarmbots.learn.nn_components.popart import PopArtLinear


PARAMETER_ROLES = ("actor", "critic", "shared_encoder", "next_obs_prediction", "targets", "other")
PROCESSING_ROLES = ("actor", "critic", "shared_encoder", "other")
PROCESSING_KINDS = ("mlp", "linear_projection", "attention", "recurrent", "normalization", "embedding", "other")

_WORLD_MODEL_MODULES = {
    "pre_transition_transform", "transition_model", "pre_predictors_transform",
    "local_scalars_predictor", "local_angles_predictor", "local_rot6ds_predictor",
    "local_binaries_predictor", "global_pool_encoder", "global_scalars_predictor",
    "global_rot6ds_predictor",
}
_ACTOR_MODULES = {
    "actor", "actor_encoder", "_actor_encoder", "actor_scenario_encoder", "actor_head",
    "actor_head_input_norm", "action_dist", "decoder", "encoder_decoder_projection",
    "agent_embeddings_decoder", "query_input_norm", "query_encoder", "query_token_norm",
    "context_input_norm", "context_encoder", "context_token_norm", "memory_input_norm",
    "memory_encoder", "memory_token_norm", "action_input_norm", "action_encoder", "action_token_norm",
}


def _parameter_role(path: str, base_policy: nn.Module) -> str:
    root = path.split(".")[0]
    if root.endswith("_target") or root.startswith("target_"):
        return "targets"
    if root in _WORLD_MODEL_MODULES or root in {"actor_nop", "critic_nop"}:
        return "next_obs_prediction"
    if root in {"shared_encoder", "shared_observation_encoder"}:
        return "shared_encoder"
    if root == "encoder":
        # MATDec has a separate actor encoder; its inherited encoder feeds the
        # critic and NOP only. Other on-policy MAT encoders feed actor and critic.
        return "critic" if hasattr(base_policy, "actor_encoder") else "shared_encoder"
    if root == "critic" or root.startswith("critic_"):
        return "critic"
    if root in _ACTOR_MODULES:
        return "actor"
    return "other"


def _component_path(path: str) -> str:
    parts = path.split(".")
    root = parts[0]
    if root in {"actor", "critic", "actor_target", "critic_target", "actor_nop", "critic_nop"}:
        # Preserve critic encoder vs Q heads and NOP transition vs predictors.
        root = ".".join(parts[:2]) if len(parts) > 2 else root
    if "temporal_model" in parts:
        return root + ".temporal_model"
    if parts[0] in {"encoder", "actor_encoder", "_actor_encoder", "shared_observation_encoder"}:
        return root + ".backbone"
    return root


def _empty_counts() -> dict[str, int]:
    return {"total": 0, "trainable": 0, "frozen": 0, "parameter_bytes": 0}


def _normalized_path(path: str) -> str:
    parts = [part for part in path.split(".") if part != "_orig_mod"]
    while parts and parts[0] == "policy":
        parts.pop(0)
    return ".".join(parts)


def _parameter_kind(name: str, modules: dict[str, nn.Module]) -> tuple[str, str]:
    """Classify by owning modules, giving attention/recurrence priority over Linear.

    The second value identifies the module whose classified parameters are
    grouped together. Transformer linear1/linear2 are grouped at their layer.
    """
    owner_path = name.rpartition(".")[0]
    parts = owner_path.split(".") if owner_path else []
    ancestors = [(".".join(parts[:depth]), modules[".".join(parts[:depth])]) for depth in range(len(parts) + 1)]
    for path, module in ancestors:
        if isinstance(module, (TemporalSequenceModel, nn.RNNBase, nn.RNNCellBase)) or path.split(".")[-1] == "temporal_model":
            return "recurrent", _normalized_path(path)
    for path, module in ancestors:
        if isinstance(module, (nn.MultiheadAttention, MATOrigSelfAttention)):
            return "attention", _normalized_path(path)

    owner = modules[owner_path]
    normalized = _normalized_path(owner_path)
    if isinstance(owner, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, MultiHeadLayerNorm)):
        return "normalization", normalized
    if isinstance(owner, (nn.Embedding, nn.EmbeddingBag)) or name.split(".")[-1].startswith("agent_embeddings"):
        return "embedding", normalized
    if isinstance(owner, (nn.Linear, PopArtLinear, HeadwiseLinearProjection)):
        for path, module in ancestors:
            # A single-layer repository MLP still belongs to the MLP budget.
            # Raw Sequential MLPs (e.g. MATOrig) need at least two affine layers.
            if isinstance(module, (MLP, GLU, StackedGLU)) or (
                isinstance(module, nn.Sequential)
                and sum(isinstance(child, (nn.Linear, PopArtLinear)) for child in module.children()) >= 2
            ):
                return "mlp", _normalized_path(path)
        if len(parts) >= 1 and parts[-1] in {"linear1", "linear2"}:
            parent = modules[".".join(parts[:-1])]
            if isinstance(getattr(parent, "linear1", None), nn.Linear) and isinstance(getattr(parent, "linear2", None), nn.Linear):
                return "mlp", _normalized_path(".".join(parts[:-1]))
        if "feedforward" in parts or "inter_module_feedforward" in parts:
            return "mlp", normalized
        return "linear_projection", normalized
    return "other", normalized


def _empty_processing_counts() -> dict[str, Any]:
    return {
        **_empty_counts(),
        "mlp_and_linear": _empty_counts(),
        "kinds": {kind: _empty_counts() for kind in PROCESSING_KINDS},
    }


def _add_counts(destination: dict[str, int], counts: dict[str, int]) -> None:
    for key, value in counts.items():
        destination[key] += value


def count_policy_parameters(policy: nn.Module) -> dict[str, Any]:
    """Return unique parameter counts, role totals, and component totals.

    Actor counts include distribution parameters. Shared actor/critic encoders
    are reported separately, and NOP counts cover its extra modules only.
    Frozen target copies are included in ``total`` but excluded from
    ``trainable``. Buffers, optimizer state, replay, and activations are excluded.
    Parameter identity deduplicates aliases, including tied critic encoders.
    ``total_minus_nop`` retains frozen target copies while removing NOP modules.
    ``processing`` excludes NOP and targets and separates affine MLP/projection
    weights from attention, complete recurrent modules, norms, and embeddings.
    This is a parameter budget, not a FLOP or effective-capacity measurement.
    """
    base_policy = policy
    while isinstance(getattr(base_policy, "policy", None), nn.Module):
        base_policy = base_policy.policy

    aliases: dict[int, list[tuple[str, nn.Parameter]]] = defaultdict(list)
    for name, parameter in policy.named_parameters(remove_duplicate=False):
        aliases[id(parameter)].append((name, parameter))
    modules = dict(policy.named_modules(remove_duplicate=False))

    totals = _empty_counts()
    roles = {role: _empty_counts() for role in PARAMETER_ROLES}
    components: dict[tuple[str, str], dict[str, int]] = {}
    processing = _empty_processing_counts()
    processing["roles"] = {role: _empty_processing_counts() for role in PROCESSING_ROLES}
    processing_components: dict[tuple[str, str, str], dict[str, int]] = {}
    shared_parameter_aliases = []
    for entries in aliases.values():
        name, parameter = entries[0]
        path = _normalized_path(name)
        role = _parameter_role(path, base_policy)
        alias_roles = set()
        for alias, _ in entries:
            alias_roles.add(_parameter_role(_normalized_path(alias), base_policy))
        if {"actor", "critic"} <= alias_roles <= {"actor", "critic", "shared_encoder"}:
            role = "shared_encoder"
        elif len(alias_roles) > 1:
            raise ValueError(f"Parameter has aliases with incompatible roles: {[entry[0] for entry in entries]}")
        component = components.setdefault((role, _component_path(path)), _empty_counts())
        count = parameter.numel()
        counts = {
            "total": count,
            "trainable": count if parameter.requires_grad else 0,
            "frozen": 0 if parameter.requires_grad else count,
            "parameter_bytes": count * parameter.element_size(),
        }
        for destination in (totals, roles[role], component):
            _add_counts(destination, counts)
        if role in PROCESSING_ROLES:
            kind, module_path = _parameter_kind(name, modules)
            alias_kinds = {_parameter_kind(alias, modules)[0] for alias, _ in entries}
            if len(alias_kinds) > 1:
                raise ValueError(f"Parameter has aliases with incompatible processing kinds: {[entry[0] for entry in entries]}")
            mechanism_component = processing_components.setdefault((role, kind, module_path), _empty_counts())
            role_processing = processing["roles"][role]
            for destination in (
                processing, role_processing, processing["kinds"][kind], role_processing["kinds"][kind],
                mechanism_component,
            ):
                _add_counts(destination, counts)
            if kind in {"mlp", "linear_projection"}:
                _add_counts(processing["mlp_and_linear"], counts)
                _add_counts(role_processing["mlp_and_linear"], counts)
        if len(entries) > 1:
            shared_parameter_aliases.append({"names": [entry[0] for entry in entries], "parameters": count})

    for key in totals:
        assert totals[key] == sum(counts[key] for counts in roles.values())
        assert totals[key] == sum(counts[key] for counts in components.values())
        assert processing[key] == sum(roles[role][key] for role in PROCESSING_ROLES)
        assert processing[key] == sum(counts[key] for counts in processing["kinds"].values())
        assert processing[key] == sum(counts[key] for counts in processing_components.values())
        assert processing["mlp_and_linear"][key] == sum(processing["kinds"][kind][key] for kind in ("mlp", "linear_projection"))
        for role in PROCESSING_ROLES:
            assert processing["roles"][role][key] == roles[role][key]
    assert totals["total"] == sum(parameter.numel() for parameter in policy.parameters())
    processing["components"] = [
        {"role": role, "kind": kind, "module": name, **counts}
        for (role, kind, name), counts in sorted(processing_components.items())
    ]
    return {
        **totals,
        "total_minus_nop": totals["total"] - roles["next_obs_prediction"]["total"],
        "roles": roles,
        "components": [
            {"role": role, "module": name, **counts}
            for (role, name), counts in sorted(components.items())
        ],
        "shared_parameter_aliases": shared_parameter_aliases,
        "processing": processing,
    }
