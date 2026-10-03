"""Record checkpoints or custom policies; pass --help for scenario/render options."""

import sys

from swarmbots.benchmark.cli import main


if __name__ == "__main__":
    main(["record", *sys.argv[1:]])
