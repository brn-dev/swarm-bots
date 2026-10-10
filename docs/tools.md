# Command-line tools

These utilities are included in the installed `swarmbots` package. Use them to inspect policy sizes, draw architectures, and analyze training logs.

| Command | Python module | Purpose |
| --- | --- | --- |
| `swarmbots-inspect-policies` | `swarmbots.tools.inspect_policy_parameters` | Report policy parameter counts and layer dimensions |
| `swarmbots-diagram-policies` | `swarmbots.tools.diagram_policy_architectures` | Draw policy architectures and generate an offline gallery |
| `swarmbots-plot-logs` | `swarmbots.tools.plot_logs` | Plot scalar metrics from training logs |
| `swarmbots-plot-experiments` | `swarmbots.tools.plot_experiment_results` | Compare experiment groups and individual seeds |

Each command supports `--help`. You can also run a tool with the environment's Python, for example:

```bash
python -m swarmbots.tools.inspect_policy_parameters --help
python -m swarmbots.tools.diagram_policy_architectures --variants tmasac --scales 5M
python -m swarmbots.tools.plot_logs runs/my-run --output plots/training.png
python -m swarmbots.tools.plot_experiment_results runs/my-experiment --output-dir plots/comparison
```

On Windows, use `.venv\Scripts\python.exe` or the corresponding command executable in `.venv\Scripts` when the environment is not activated.

Paths are relative to your working directory unless you pass an absolute path. The diagram tool reads or creates audits under `docs/policy_parameters/` and defaults to writing its gallery under `docs/policy_parameters/diagrams/`. Running it from the repository root uses the checked-in snapshots; elsewhere it reconstructs the requested scales from the installed presets. `--output-dir` selects a different gallery directory. A partial diagram selection retains the complete audit when rebuilding snapshots.

Install `swarmbots[plot]` for plotting. Log column inspection with `swarmbots-plot-logs --list-columns` does not require Matplotlib.

See [policy parameter sizing](policy_parameters/README.md), [the diagram reading guide](policy_parameters/diagrams/README.md), and [log plotting](log_plotting.md) for options and output formats.
