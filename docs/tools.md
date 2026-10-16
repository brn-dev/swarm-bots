# Command-line tools

The current source checkout installs commands for task discovery, smoke testing, recording, policy-size inspection, architecture diagrams, and training-log analysis. The published `0.1.0a3` package provides the main `swarmbots` command; the inspection, diagram, and plotting utilities below require the source checkout.

| Command | Python module | Purpose |
| --- | --- | --- |
| `swarmbots` | `swarmbots.benchmark.cli` | `list` tasks, `describe` a task, `smoke` test simulation, and `record` policy videos |
| `swarmbots-inspect-policies` | `swarmbots.tools.inspect_policy_parameters` | Report policy parameter counts and layer dimensions |
| `swarmbots-diagram-policies` | `swarmbots.tools.diagram_policy_architectures` | Draw policy architectures and generate an offline gallery |
| `swarmbots-plot-logs` | `swarmbots.tools.plot_logs` | Plot scalar metrics from training logs |
| `swarmbots-plot-experiments` | `swarmbots.tools.plot_experiment_results` | Compare experiment groups and individual seeds |

Each command supports `--help`. Tools can also run with the environment's Python, for example:

```bash
python -m swarmbots.tools.inspect_policy_parameters --help
python -m swarmbots.tools.diagram_policy_architectures --variants tmasac --scales 5M-2L
python -m swarmbots.tools.plot_logs runs/my-run --output plots/training.png
python -m swarmbots.tools.plot_experiment_results runs/my-experiment --output-dir plots/comparison
```

On Windows, use `.venv\Scripts\python.exe` or the corresponding command executable in `.venv\Scripts` when the environment is not activated.

Paths are relative to the working directory unless an absolute path is supplied. The diagram tool builds the selected policies and writes fresh parameter reports under `docs/policy_parameters/generated/<scale>/`, with a gallery under `docs/policy_parameters/diagrams/`. Open the gallery's `index.html` after generation. Use `--variants` and `--scales` to limit the selection, or `--output-dir` to choose a different gallery directory.

In the checkout, `uv sync --extra plot` installs plotting dependencies. Log column inspection with `swarmbots-plot-logs --list-columns` does not require Matplotlib.

The [policy parameter sizing](policy_parameters/README.md), [diagram](policy_parameters/diagrams/README.md), and [log plotting](log_plotting.md) guides describe options and output formats.
