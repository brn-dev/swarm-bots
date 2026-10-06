"""Plot scalar training metrics from one or more files or run directories."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path, help="CSV/archive files or run directories")
    parser.add_argument("--columns", nargs="+", help="defaults to available return/success EMAs")
    parser.add_argument("--labels", nargs="+", help="one label per resolved log")
    parser.add_argument("--x-column", default="timesteps")
    parser.add_argument("--delimiter", default=";")
    parser.add_argument("--smooth", type=float, help="optional EMA alpha in (0, 1]")
    parser.add_argument("--max-steps", type=float, help="cut numeric X values at this limit")
    parser.add_argument("--title")
    parser.add_argument("--output", type=Path, default=Path("plots/training.png"))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--show", action="store_true", help="open a Matplotlib window")
    parser.add_argument("--list-columns", action="store_true", help="print the first log's columns and exit")
    args = parser.parse_args()

    from swarmbots.plotting import log_columns

    if args.list_columns:
        print("\n".join(log_columns(args.sources[0], delimiter=args.delimiter)))
        return
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from swarmbots.plotting import plot_logs

    figure = plot_logs(
        args.sources,
        columns=args.columns,
        labels=args.labels,
        x_column=args.x_column,
        delimiter=args.delimiter,
        smooth=args.smooth,
        max_steps=args.max_steps,
        title=args.title,
        output_path=args.output,
        dpi=args.dpi,
    )
    print(f"Saved {args.output.resolve()}")
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
