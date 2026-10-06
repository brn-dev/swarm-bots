# Policy parameter sizing

See the [fixed layouts and allocation rules](model_scale.md), [cross-scale comparison](scale_comparison.md), and [diagram reading guide](diagrams/README.md).

The repository keeps the guides, compact comparison, and three compressed audit snapshots (`counts.json.gz`, `2.5M/counts.json.gz`, and `10M/counts.json.gz`). Together the snapshots are about 100 KiB and provide the dimensions, parameter counts, and configurations needed by the diagram generator and its tests. The small historical preset audit is retained separately.

Detailed Markdown reports and CSV exports live in `generated/<scale>/`; rendered diagrams, their export formats, and the offline gallery live in `diagrams/`. These generated outputs stay local and are excluded from Git and source packages. Existing local outputs remain usable.

Generate the gallery from the compact snapshots:

```bash
python examples/diagram_policy_architectures.py
```

Then open `docs/policy_parameters/diagrams/index.html`. Add `--refresh` to rebuild the policies, update the compact snapshots, and write fresh detailed reports under `generated/`.

To inspect a specific task or scale and save local reports:

```bash
python examples/inspect_policy_parameters.py --model-scale 5M --output-dir docs/policy_parameters/generated/5M
python examples/inspect_policy_parameters.py --model-scale 2.5M --output-dir docs/policy_parameters/generated/2.5M
python examples/inspect_policy_parameters.py --model-scale 10M --output-dir docs/policy_parameters/generated/10M
```

The inspector's exports include actual module/layer dimensions, role and processing breakdowns, CSV tables, a Markdown report, and a compressed JSON audit. Custom inspections stay in their chosen output directory; the diagram generator's `--refresh` updates the three checked-in WallEasy snapshots.
