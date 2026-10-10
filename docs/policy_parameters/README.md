# Policy parameter sizing

See the [fixed layouts and allocation rules](model_scale.md), [cross-scale comparison](scale_comparison.md), and [diagram reading guide](diagrams/README.md). The commands below are [installed tools](../tools.md); run them from the repository root to reuse the checked-in snapshots.

The repository keeps the guides, compact comparison, and three compressed audit snapshots (`counts.json.gz`, `2.5M/counts.json.gz`, and `10M/counts.json.gz`). Together the snapshots are about 100 KiB and provide the dimensions, parameter counts, and configurations needed by the diagram generator and its tests. The small historical preset audit is retained separately.

Detailed Markdown reports and CSV exports live in `generated/<scale>/`; rendered diagrams, their export formats, and the offline gallery live in `diagrams/`. These generated outputs stay local and are excluded from Git and source packages. Existing local outputs remain usable.

Generate the gallery from the compact snapshots:

```bash
swarmbots-diagram-policies
```

Then open `docs/policy_parameters/diagrams/index.html`. Add `--refresh` to rebuild the policies, update the compact snapshots, and write fresh detailed reports under `generated/`.

To inspect a specific task or scale and save local reports:

```bash
swarmbots-inspect-policies --model-scale 5M --output-dir docs/policy_parameters/generated/5M
swarmbots-inspect-policies --model-scale 2.5M --output-dir docs/policy_parameters/generated/2.5M
swarmbots-inspect-policies --model-scale 10M --output-dir docs/policy_parameters/generated/10M
```

The inspector's exports include actual module/layer dimensions, role and processing breakdowns, CSV tables, a Markdown report, and a compressed JSON audit. Custom inspections stay in their chosen output directory; the diagram generator's `--refresh` updates the three checked-in WallEasy snapshots.

## Inspect policy parameters

```bash
swarmbots-inspect-policies SwarmBots-WallEasy-v0 --output-dir docs/policy_parameters/generated/5M
swarmbots-inspect-policies SwarmBots-POWallMedium-v0 --variants tmasac tmasac_slstm tmasac_lstm
```

The tool instantiates every core preset by default, using the same policy factory as training, without allocating a trainer, replay buffer, or optimizer. It reports actor, critic, shared encoder, next-observation prediction (NOP), frozen target copies, and other parameters. Detailed component tables separate recurrent temporal modules, critic encoders/Q heads, and NOP projections/transition/prediction heads. Shared parameters count once; buffers are excluded. CPU execution and eager modules are the defaults.

The main-policy processing comparison excludes NOP and targets. It splits MLPs, standalone linear projections, attention, recurrent modules, normalization, embeddings, and other parameters, by actor, critic, and shared encoder. **MLP + linear** includes transformer feed-forward blocks, SwiGLU gates, observation/latent projections, and action/value heads. Attention's Q/K/V/output projections and complete recurrent modules (including input/gate projections) count in their respective categories. This affine budget helps compare processing capacity across variants; it does not measure FLOPs or effective capacity.

The **Trainable − NOP** column counts the trainable total minus trainable NOP parameters. It excludes all frozen parameters, including target networks. The **Online total** in the processing comparison excludes NOP and targets but can include other frozen main-policy parameters.

`--include-hidden` adds optional MLP baselines and ablation controls to the default core sweep. `--output-dir` saves `layer_layout.csv` with actual module dimensions and parameter counts, a Markdown report, summary and component CSVs, `processing.csv` (processing counts by role), `processing_components.csv` (classified modules), and losslessly compressed `counts.json.gz` with resolved hyperparameters and task shapes. Read the JSON using `json.load(gzip.open(path, "rt", encoding="utf-8"))`. Use `--model-scale "5M NOP1M"` to select the fixed layout, `--model-scale legacy` for explicit widths, `--no-nop`, or JSON `--policy-kwargs`, `--scenario-kwargs`, and `--env-kwargs` to match custom settings. Parameter counts depend on the task's observation/action shapes. See the [18-variant WallEasy comparison](scale_comparison.md) for the measured sizes and the [storage guide](README.md) for local report generation.
