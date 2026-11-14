# Agent notes

Start here before changing several files. Keep durable navigation and costly gotchas in these notes; use source and Git history for defaults and implementation history.

## Documentation boundaries

The public audience is researchers considering or using the SwarmBots benchmark. Lead with research goals, capabilities, tasks, and practical use: installation, custom-learner integration, baseline training, evaluation, recording, and analysis. The benchmark is the primary product; the RL suite is an optional starting point and comparison tool.

Keep scientific details that affect experimental interpretation public: observation/action contracts, information boundaries, reward and success definitions, evaluation populations, method adaptations, and comparison limits. Focused guides in `docs/` hold the detailed API and model references.

Put repository navigation, implementation invariants, compiler workarounds, test hazards, release procedures, and instructions for future agents here or in the linked `.agent-notes/` references. Do not link these notes from public guides. Public docs should not instruct researchers to maintain graph builders, synchronize generated exports, or run release checks.

`CHANGELOG.md` records concise user-visible changes and compatibility consequences. Group unreleased changes by topic and describe the final behavior. Keep each item to one or two short sentences about the capability, workflow, or compatibility impact; link to a focused guide for task constants, formulas, configuration examples, or API details. Do not append review transcripts, test inventories, internal refactors, serialization edge-case inventories, or successive revisions of the same design. Preserve important maintenance findings in the relevant agent reference instead; Git retains implementation history.

Prefer reader-neutral, subject-focused prose and active voice. Avoid unnecessary imperatives or framing every capability around what the reader can or should do; do not substitute passive voice mechanically. Reader-directed language fits tutorials and procedures. Keep exact dimensions/defaults in their focused references or source rather than duplicating them in overviews.

## Navigation

Learning paths below are relative to `swarmbots/learn`.

| Area | Entry points |
| --- | --- |
| Benchmark API, task registry, evaluation, recording, CLI | `swarmbots/benchmark/`; public exports in `swarmbots/__init__.py` |
| Physics and task definitions | `swarmbots/mjw_env/`, `swarmbots/scenario_presets/`; shared spaces in `mjw_env/scenarios/scenario_spaces.py`, reward compilation in `reward_compilation.py` |
| Learning API and construction | Lazy `swarmbots.learn` exports; `training.py`; `presets/variants.py` registers presets, `policy_variants.py` classifies resolved policies; `policy_factory.py` resolves options, `policy_builders.py` constructs families |
| Policy layouts and off-policy defaults | `presets/model_scale.py`, `off_policy.py`, `sac.py`, `tmasac.py`; shared lifecycle in `algos/off_policy/` |
| Shared critic interactions | `nn_components/deep_set.py`, `DeepSetInteractionConfig`, `algos/mat/deep_set_encoder.py` |
| Tensor conversion | `swarmbots/utils/torch_utils.py`; learning-specific conversion and reset masks in `tensor_conversion.py` |
| Programmatic loss-weight updates | `loss_weights.py`, `algos/off_policy/nop.py`; runtime commands are read-only |
| Training provenance | `run_metadata.py`, `training.py`, `BaseAlgorithm._write_run_metadata` |
| Inspection and plotting | `swarmbots/tools/`, `swarmbots/plotting/`; researcher guides in `docs/policy_parameters/` |
| Runnable workflows and verification | `examples/`, `tests/`, `.github/workflows/` |

## Read before working in these areas

- [Policies and checkpoints](.agent-notes/policies-and-checkpoints.md): model budgets, actor/critic isolation, NOP masks, QCX caches, padding, checkpoint reconstruction, and optimizer ownership.
- [Training and evaluation](.agent-notes/training-and-evaluation.md): accepted design decisions, rollout/replay semantics, normalization, CUDA graph ownership, metrics, and evaluation adapters.
- [Simulation, tools, and releases](.agent-notes/simulation-and-tools.md): reset sampling/settling, autoreset, geometry, diagrams, plotting, sandbox/test hazards, and publication.

Read the relevant reference before editing its subsystem. Update only durable facts when behavior or structure changes; avoid duplicating these details in the changelog. New maintenance notes belong in the relevant reference so this entry point stays short.
