"""Instantiate registered presets and report their parameter counts without training.

Run from the repository root:
    python examples/inspect_policy_parameters.py --output-dir docs/policy_parameters
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, get_args

import torch

from swarmbots import ALL_BENCHMARK_IDS
from swarmbots.learn import list_variants
from swarmbots.learn.parameter_counts import (
    PARAMETER_ROLES,
    PROCESSING_KINDS,
    PROCESSING_ROLES,
    count_policy_parameters,
)
from swarmbots.learn.presets.policy_factory import ContinuousActionDistVariant, _is_deterministic_policy_variant
from swarmbots.learn.training import _make_policy_env, _variant_options


def _shape_metadata(env: Any) -> dict[str, Any]:
    return {
        "n_agents": env.n_agents,
        "local_obs_dim": env.local_obs_dim,
        "global_obs_dim": env.global_obs_dim,
        "hidden_local_vars_dim": env.hidden_local_vars_dim,
        "hidden_global_vars_dim": env.hidden_global_vars_dim,
        "actuators_per_agent": env.actuators_dim,
        "connectors_per_agent": env.connectors_dim,
        "actions_per_agent": env.action_space.total_agent_action_dim,
        "continuous_connector_actions": env.continuous_connector_actions,
    }


def inspect_variants(
    benchmark_id: str,
    variants: tuple[str, ...],
    *,
    device: str = "cpu",
    seed: int = 42,
    use_nop: bool | None = None,
    continuous_action_dist: str | None = None,
    policy_kwargs: dict[str, Any] | None = None,
    scenario_kwargs: dict[str, Any] | None = None,
    env_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = []
    errors = []
    for variant in variants:
        print(f"Counting {variant}...", file=sys.stderr, flush=True)
        options, _ = _variant_options(variant)
        options.update(policy_kwargs or {})
        if use_nop is not None:
            options["use_nop"] = use_nop
        # A mixed sweep retains deterministic presets' continuous action heads.
        if continuous_action_dist is not None and not _is_deterministic_policy_variant(options["policy_variant"]):
            options["continuous_action_dist"] = continuous_action_dist
        env = None
        try:
            env, policy = _make_policy_env(
                benchmark_id,
                num_envs=1,
                device=device,
                seed=seed,
                episode_length=None,
                scenario_kwargs={"compile_reward_kernel": False, **(scenario_kwargs or {})},
                env_kwargs={"compile_tensor_operations": False, **(env_kwargs or {})},
                policy_options=options,
                compile_modules=False,
            )
            rows.append({
                "variant": variant,
                "policy_class": type(policy).__name__,
                "base_policy_class": type(getattr(policy, "policy", policy)).__name__,
                "shape": _shape_metadata(env),
                "use_nop": bool(options["use_nop"]),
                "continuous_action_dist": options["continuous_action_dist"],
                "hyper_parameters": policy.get_hyper_parameters(),
                **count_policy_parameters(policy),
            })
            del policy
        except Exception as error:
            errors.append({"variant": variant, "error": f"{type(error).__name__}: {error}"})
            print(f"Failed {variant}: {error}", file=sys.stderr, flush=True)
        finally:
            if env is not None:
                env.close()
    return {
        "benchmark_id": benchmark_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "device": device,
        "seed": seed,
        "requested_variants": list(variants),
        "overrides": {
            "use_nop": use_nop,
            "continuous_action_dist": continuous_action_dist,
            "policy_kwargs": policy_kwargs or {},
            "scenario_kwargs": scenario_kwargs or {},
            "env_kwargs": env_kwargs or {},
        },
        "policies": rows,
        "errors": errors,
    }


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Policy parameter counts", "",
        f"Task: `{report['benchmark_id']}`. Instantiated {len(report['policies'])} of "
        f"{len(report['requested_variants'])} requested registered variants, with one environment on "
        f"`{report['device']}` and compilation disabled. Generated {report['generated_at'][:10]}.", "",
        "Counts are actual unique PyTorch parameters, including weights and biases. "
        "Actor includes action-distribution heads and learned distribution parameters. "
        "Shared encoder is counted separately from both actor and critic. "
        "NOP (next-observation prediction) counts its additional projections, transition model, "
        "and prediction heads; its source encoder remains in actor, critic, or shared encoder. "
        "Targets are the frozen actor/critic/shared-encoder copies used by off-policy learning.", "",
        "Actor + critic + shared encoder + NOP + targets + other = total. "
        "**Trainable − NOP** is the trainable total minus trainable NOP parameters. "
        "It excludes all frozen parameters, including target networks. "
        "Trainable excludes frozen parameters. Actor execution needs actor + shared encoder. "
        "Counts describe the entire policy, not a separate network per agent. "
        "Buffers (including normalization/PopArt state), optimizer state, replay, and activations are excluded. "
        "Parameter MiB below uses the tensors' actual dtypes and is not runtime or checkpoint memory.", "",
        "Sizes depend on task shapes and architecture settings. These are the registered preset defaults "
        "unless overrides below are nonempty. A residual-connection ablation can have the same count.", "",
        f"Overrides: `{json.dumps(report['overrides'], sort_keys=True)}`.", "",
        "| Variant | Actor | Critic | Shared encoder | NOP | Targets | Other | Total | Trainable | Trainable − NOP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["policies"]:
        counts = [row["roles"][role]["total"] for role in PARAMETER_ROLES]
        lines.append(f"| {row['variant']} | " + " | ".join(f"{count:,}" for count in [*counts, row["total"], row["trainable"], row["trainable_minus_nop"]]) + " |")
    lines.extend([
        "", "## Main-policy MLP, projection, attention, and recurrent parameters", "",
        "This comparison includes only online actor, critic, shared encoder, and any other main-policy parameters. "
        "It excludes all NOP modules and frozen target copies. Shared parameters count once. "
        "The **MLP + linear** column is the affine parameter budget to compare when matching "
        "feed-forward processing across variants. It includes biases and action/value output heads.", "",
        "Classification:", "",
        "- **MLP:** affine weights/biases in MLPs, SwiGLU/GLU gate/value/output projections, "
        "and transformer feed-forward blocks. Transformer MLPs remain here even when their layer also has attention.",
        "- **Linear projection:** remaining standalone affine layers, including observation/token/latent "
        "projections, action and value heads, and PopArt affine weights/biases. A projection "
        "outside a recurrent module, such as `temporal_output_projection` or critic actor-state encoding, "
        "remains in MLP + linear.",
        "- **Attention:** complete attention modules, including Q/K/V and output projections; "
        "these affine parameters are excluded from MLP + linear.",
        "- **Recurrent:** complete LSTM/sLSTM/other temporal modules, including input/gate "
        "projections, recurrent kernels, biases, and any internal normalization. These are excluded from MLP + linear.",
        "- **Normalization / embedding / other:** parameters outside the above categories; "
        "these are also excluded from MLP + linear. They are separated in the role and component tables below.", "",
        "MLP + linear + attention + recurrent + remainder = online main-policy total. "
        "Remainder is normalization + embedding + other. Parameter counts measure stored affine capacity; "
        "they do not measure FLOPs, parameter reuse per timestep, or effective model capacity. "
        "Attention's matrix operations can add substantial compute without additional parameters. "
        "Matching MLP + linear totals alone can also hide different actor/critic allocations or layer shapes; "
        "use the role and module breakdowns when choosing an experimental control.", "",
        "| Variant | Actor MLP + linear | Critic MLP + linear | Shared MLP + linear | Total MLP + linear | Attention | Recurrent | Remainder | Online total | Trainable − NOP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in report["policies"]:
        processing = row["processing"]
        remainder = sum(processing["kinds"][kind]["total"] for kind in ("normalization", "embedding", "other"))
        values = [
            *(processing["roles"][role]["mlp_and_linear"]["total"] for role in PROCESSING_ROLES[:3]),
            processing["mlp_and_linear"]["total"], processing["kinds"]["attention"]["total"],
            processing["kinds"]["recurrent"]["total"], remainder, processing["total"], row["trainable_minus_nop"],
        ]
        lines.append(f"| {row['variant']} | " + " | ".join(f"{value:,}" for value in values) + " |")
    lines.extend([
        "", "## Reproduce", "",
        "Run from the repository root with the installed package:", "", "```bash",
        f"python examples/inspect_policy_parameters.py {report['benchmark_id']} --output-dir docs/policy_parameters",
        "```", "",
        "This command counts all current registered defaults. Use `--variants` to select a subset, "
        "`--no-nop` to disable NOP, or `--policy-kwargs`, `--scenario-kwargs`, and `--env-kwargs` "
        "with JSON objects to match custom architectures/tasks. For example, "
        "`--policy-kwargs '{\"enc_d_model\": 128, \"dec_d_model\": 64}'`. "
        "Reapply the overrides recorded above to reproduce a customized report. "
        "The losslessly compressed `counts.json.gz` output records resolved policy hyperparameters; "
        "CSVs contain overall counts, "
        "components, processing by role, and processing components. "
        "Failed variants are listed explicitly and cause a nonzero exit code.",
        "", "Read the compressed JSON in Python with:", "", "```python",
        "import gzip, json",
        'with gzip.open("docs/policy_parameters/counts.json.gz", "rt", encoding="utf-8") as stream:',
        "    counts = json.load(stream)", "```",
        "", "## Task shapes", "",
    ])
    seen_shapes = []
    for row in report["policies"]:
        if row["shape"] not in seen_shapes:
            seen_shapes.append(row["shape"])
    for shape in seen_shapes:
        lines.append(f"- `{json.dumps(shape, sort_keys=True)}`")
    lines.extend(["", "## Component breakdowns", "",
                  "Each component table is additive. Recurrent temporal-model weights are split from "
                  "the encoder backbone; critic encoders, Q heads, and NOP heads are separated.", ""])
    for row in report["policies"]:
        lines.extend([
            "<details>", f"<summary>{row['variant']} — {row['total']:,} parameters</summary>", "",
            f"Policy: `{row['base_policy_class']}`. Parameter storage: "
            f"{row['parameter_bytes'] / 1024**2:.3f} MiB.", "",
            "| Role | Module | Parameters | Trainable | Frozen |",
            "| --- | --- | ---: | ---: | ---: |",
        ])
        for component in row["components"]:
            lines.append(
                f"| {component['role']} | `{component['module']}` | {component['total']:,} | "
                f"{component['trainable']:,} | {component['frozen']:,} |"
            )
        lines.extend([
            "", "Online main-policy processing by role (NOP/targets excluded):", "",
            "| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for role in PROCESSING_ROLES:
            processing = row["processing"]["roles"][role]
            if not processing["total"]:
                continue
            values = [
                processing["kinds"]["mlp"]["total"], processing["kinds"]["linear_projection"]["total"],
                processing["mlp_and_linear"]["total"],
                *(processing["kinds"][kind]["total"] for kind in PROCESSING_KINDS[2:]),
                processing["total"],
            ]
            lines.append(f"| {role} | " + " | ".join(f"{value:,}" for value in values) + " |")
        lines.extend([
            "", "Online processing components (all seven kinds are additive):", "",
            "| Role | Kind | Module | Parameters |",
            "| --- | --- | --- | ---: |",
        ])
        for component in row["processing"]["components"]:
            lines.append(
                f"| {component['role']} | {component['kind']} | `{component['module']}` | {component['total']:,} |"
            )
        if row["shared_parameter_aliases"]:
            lines.extend(["", "Tied parameter aliases are counted once; see `counts.json.gz` for their names."])
        lines.extend(["", "</details>", ""])
    if report["errors"]:
        lines.extend(["## Failed variants", ""])
        lines.extend(f"- `{error['variant']}`: {error['error']}" for error in report["errors"])
    return "\n".join(lines)


def write_compressed_json(value: Any, path: Path) -> None:
    """Preserve all JSON data in a compact, reproducible gzip export."""
    payload = (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    path.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))


def save_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.md").write_text(render_report(report), encoding="utf-8")
    write_compressed_json(report, output_dir / "counts.json.gz")
    (output_dir / "counts.json").unlink(missing_ok=True)
    processing_columns = ["main_policy_total", "mlp_and_linear", *(f"processing_{kind}" for kind in PROCESSING_KINDS)]
    role_columns = [f"{role}_mlp_and_linear" for role in PROCESSING_ROLES]
    columns = ["variant", *PARAMETER_ROLES, "total", "trainable", "trainable_minus_nop", "frozen", "parameter_bytes", *processing_columns, *role_columns]
    with (output_dir / "counts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in report["policies"]:
            processing = row["processing"]
            writer.writerow({
                **{key: row[key] for key in ("variant", "total", "trainable", "trainable_minus_nop", "frozen", "parameter_bytes")},
                **{role: row["roles"][role]["total"] for role in PARAMETER_ROLES},
                "main_policy_total": processing["total"],
                "mlp_and_linear": processing["mlp_and_linear"]["total"],
                **{f"processing_{kind}": processing["kinds"][kind]["total"] for kind in PROCESSING_KINDS},
                **{f"{role}_mlp_and_linear": processing["roles"][role]["mlp_and_linear"]["total"] for role in PROCESSING_ROLES},
            })
    with (output_dir / "components.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["variant", "role", "module", "total", "trainable", "frozen", "parameter_bytes"])
        writer.writeheader()
        for row in report["policies"]:
            writer.writerows({"variant": row["variant"], **component} for component in row["components"])
    with (output_dir / "processing.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["variant", "role", "total", "trainable", "frozen", "mlp_and_linear", *PROCESSING_KINDS])
        writer.writeheader()
        for row in report["policies"]:
            for role in (*PROCESSING_ROLES, "all"):
                counts = row["processing"] if role == "all" else row["processing"]["roles"][role]
                writer.writerow({
                    "variant": row["variant"], "role": role,
                    **{key: counts[key] for key in ("total", "trainable", "frozen")},
                    "mlp_and_linear": counts["mlp_and_linear"]["total"],
                    **{kind: counts["kinds"][kind]["total"] for kind in PROCESSING_KINDS},
                })
    with (output_dir / "processing_components.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["variant", "role", "kind", "module", "total", "trainable", "frozen", "parameter_bytes"])
        writer.writeheader()
        for row in report["policies"]:
            writer.writerows({"variant": row["variant"], **component} for component in row["processing"]["components"])


def _json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_id", nargs="?", default="SwarmBots-WallEasy-v0", choices=ALL_BENCHMARK_IDS)
    parser.add_argument("--variants", nargs="+", choices=list_variants(), help="default: all registered variants")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nop", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--continuous-action-dist", choices=get_args(ContinuousActionDistVariant))
    parser.add_argument("--policy-kwargs", type=_json_object, default={}, help="JSON architecture overrides")
    parser.add_argument("--scenario-kwargs", type=_json_object, default={}, help="JSON task overrides")
    parser.add_argument("--env-kwargs", type=_json_object, default={}, help="JSON environment overrides")
    parser.add_argument("--output-dir", type=Path, help="save Markdown, compressed JSON (.json.gz), overall/component CSVs, and processing CSVs")
    args = parser.parse_args()
    # Small CPU initialization jobs are much faster without oversized BLAS pools.
    torch.set_num_threads(1)
    report = inspect_variants(
        args.benchmark_id,
        tuple(args.variants) if args.variants else list_variants(),
        device=args.device,
        seed=args.seed,
        use_nop=args.nop,
        continuous_action_dist=args.continuous_action_dist,
        policy_kwargs=args.policy_kwargs,
        scenario_kwargs=args.scenario_kwargs,
        env_kwargs=args.env_kwargs,
    )
    print(render_report(report))
    if args.output_dir is not None:
        save_report(report, args.output_dir)
        print(f"Saved reports to {args.output_dir}", file=sys.stderr)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
