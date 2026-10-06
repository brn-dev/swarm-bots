"""Save mean/std and individual-seed plots for an experiment."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", nargs="?", type=Path, help="root containing group/run directories")
    parser.add_argument(
        "--group",
        action="append",
        nargs="+",
        metavar="NAME_OR_PATH",
        help="repeat: --group NAME FILE_OR_DIR [FILE_OR_DIR ...]; preserves group order",
    )
    parser.add_argument("--columns", nargs="+", help="defaults to available return/success EMAs")
    parser.add_argument("--x-column", default="timesteps")
    parser.add_argument("--delimiter", default=";")
    parser.add_argument("--max-steps", type=float)
    parser.add_argument("--smooth", type=float, help="optional EMA alpha in (0, 1]")
    parser.add_argument("--title")
    parser.add_argument("--output-dir", type=Path, default=Path("plots/comparison"))
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=["png"])
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    if (args.experiment_dir is None) == (args.group is None):
        parser.error("provide an experiment directory or --group entries")
    sources = args.experiment_dir
    if args.group is not None:
        sources = {}
        for entry in args.group:
            if len(entry) < 2:
                parser.error("each --group needs a name and at least one path")
            name, *paths = entry
            if name in sources:
                parser.error(f"duplicate group name: {name}")
            sources[name] = paths

    import matplotlib

    matplotlib.use("Agg")
    from swarmbots.plotting import plot_experiment_results

    result = plot_experiment_results(
        sources,
        args.output_dir,
        columns=args.columns,
        x_column=args.x_column,
        delimiter=args.delimiter,
        max_steps=args.max_steps,
        smooth=args.smooth,
        title=args.title,
        formats=args.formats,
        dpi=args.dpi,
    )
    for path in result.output_paths:
        print(path)


if __name__ == "__main__":
    main()
