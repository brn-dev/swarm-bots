# Plot training logs

`swarmbots.plotting` reads the semicolon-delimited CSV logs written by the included learners and produces Matplotlib figures. The plotting implementation lives in `swarmbots/plotting/plot_logs.py`. Install the optional plotting dependency:

```bash
pip install 'swarmbots[plot]'
```

In a source checkout, use `uv sync --extra dev --extra plot`. Plotting needs no simulator instance or GPU. The helpers accept a CSV file, a compressed CSV (`.gz`, `.bz2`, `.xz`, or `.zip`), or a run directory containing `log.csv` or its compressed counterpart. A directory prefers `log.csv` when both versions exist. ZIP archives must contain one CSV, or exactly one file named `log.csv`.

## Plot one or more runs

```python
from swarmbots.plotting import plot_logs

figure = plot_logs(
    "runs/SwarmBots-WallEasy-v0/mappo",
    output_path="plots/mappo.png",
)
```

The default panels show the available `ep_rew_ema` and `ep_success_rate_ema` columns. Success EMA values are already percentages in the training log. These charts describe training behavior; use [benchmark evaluation](benchmark_protocol.md) to report policy scores.

Pass a list of paths to overlay runs, and select any scalar columns explicitly:

```python
import matplotlib.pyplot as plt
from swarmbots.plotting import log_columns, plot_logs

print(log_columns("runs/seed-42"))
figure = plot_logs(
    ["runs/seed-42", "runs/seed-43/log.csv.gz"],
    columns=["ep_rew_ema", "ep_len__mean"],
    labels=["Seed 42", "Seed 43"],
    std_columns={"ep_len__mean": "ep_len__std"},
    max_steps=100_000_000,
    title="Wall Easy",
    output_path="plots/seeds.pdf",
)
plt.close(figure)
```

`std_columns` maps each metric to its logged standard-deviation column. Blank metric cells stay missing. Explicitly requested columns must exist in every input log; default selection permits a missing success column. List-valued histogram columns are not scalar metrics.

`smooth=0.05` optionally applies another EMA to finite metric values, retaining gaps at missing cells. The default adds no smoothing, including to the learner's existing EMA columns. `max_steps` cuts numeric X values before plotting or aggregation and has no default limit. `x_column="timestamp"` supports ISO timestamps; `max_steps` applies only to numeric axes. Use `delimiter=","` for comma-delimited external logs. `timesteps` axes display millions; other numeric axes retain their original units.

## Compare methods across seeds

Explicit groups avoid depending on a particular folder layout. Mapping keys are the legend labels, and insertion order determines plotting order:

```python
from swarmbots.plotting import plot_experiment_results

result = plot_experiment_results(
    {
        "MAPPO": ["runs/mappo/seed-42", "runs/mappo/seed-43"],
        "TMASAC": ["runs/tmasac/seed-42", "runs/tmasac/seed-43"],
    },
    "plots/wall-easy",
    max_steps=100_000_000,
    colors={"MAPPO": "#0072B2", "TMASAC": "#D55E00"},
    linestyles={"MAPPO": "--"},
    reference_values={"ep_success_rate_ema": 100},
    formats=("png", "pdf"),
)
print(result.output_paths)
```

Each group can also be a single file, run directory, or directory containing seed runs. Repeated paths within a group count only once. Alternatively, pass an experiment root with either of these layouts:

```text
runs/wall-easy/
    mappo/
        seed-42/log.csv
        seed-43/log.csv.gz
    tmasac/
        seed-42/log.csv
        seed-43/log.csv

runs/single-seed/
    mappo/log.csv
    tmasac/log.csv
```

```python
result = plot_experiment_results("runs/wall-easy", "plots/wall-easy")
```

Discovery sorts groups and runs alphabetically and skips unrelated directories. Explicit paths that do not exist raise an error.

For each metric with finite values, the helper saves `<metric>_grouped.png` and `<metric>_individual_runs.png`. `formats` accepts `png`, `pdf`, and `svg`; `dpi` defaults to 200. Saved figures are closed automatically. `result.groups` contains the loaded runs, and `result.output_paths` lists the generated files. A missing success metric produces only return plots.

Grouped curves use linear interpolation on the union of the recorded X values. Each seed contributes equally within its finite metric range, with no endpoint extrapolation. Missing metric cells are ignored during interpolation, so grouped curves can bridge internal gaps. At duplicate X values, the last finite sample is used. Shading shows ± one population standard deviation across contributing runs and is drawn only where at least two runs contribute. A single-point run contributes at its recorded X value. Legend counts report runs with finite values for that metric; the contributing count can vary along the curve.

## Customize figures and inspect statistics

```python
import matplotlib.pyplot as plt
from swarmbots.plotting import group_statistics, load_groups, plot_groups

groups = load_groups({"MAPPO": "runs/mappo", "TMASAC": "runs/tmasac"})
statistics = group_statistics(groups["MAPPO"], "ep_rew_ema")
print(statistics.x_values, statistics.mean, statistics.std, statistics.counts)

figure = plot_groups(groups, columns=["ep_rew_ema"], font_size=14)
figure.axes[0].set_ylim(bottom=0)
figure.savefig("plots/custom.svg")
plt.close(figure)
```

`load_log(path, columns=[...])` reads an individual `RunLog` with `path`, `label`, `x_column`, `x_values`, `series`, and `x_is_datetime`. `load_groups` returns a mapping of names to lists of these runs. Both readers support `x_column`, `delimiter`, and `max_steps`.

`plot_groups(..., individual_runs=True)` draws individual seed curves with shared group colors. Both figure helpers accept `smooth`, `title`, `ylabel_overrides`, `reference_values`, `font_size`, and `figsize`. `plot_groups` additionally accepts `colors` and `linestyles`; `plot_logs` accepts `labels`, `std_columns`, `output_path`, and `dpi`.

The library keeps the caller's Matplotlib backend. For a headless job, set `MPLBACKEND=Agg` or call `matplotlib.use("Agg")` before importing plotting functions. Use `plt.show()` for an interactive Matplotlib window, and close returned figures when finished.

## Command-line tools

```bash
swarmbots-plot-logs runs/SwarmBots-WallEasy-v0/mappo --output plots/mappo.png
swarmbots-plot-logs runs/my-run --list-columns
swarmbots-plot-logs runs/my-run --columns grad_norm_actor__mean grad_norm_critic__mean --x-column total_updates --output plots/gradients.png
swarmbots-plot-experiments runs/wall-easy --output-dir plots/wall-easy --formats png pdf
swarmbots-plot-experiments --group MAPPO runs/mappo/seed-42 runs/mappo/seed-43 --group TMASAC runs/tmasac/seed-42 runs/tmasac/seed-43
```

Use gradient column names printed by `--list-columns`; their names depend on the learner. Both tools support `--help`, custom columns, an X column, smoothing, and a cutoff. `swarmbots-plot-logs --show` opens the saved figure. The tools use the headless Agg backend by default. See [command-line tools](tools.md) for equivalent Python module commands.
