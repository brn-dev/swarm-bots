# Contributing

Bug reports, new scenarios, documentation fixes, and performance improvements are welcome.

## Development setup

Install the project and its locked dependencies:

```bash
uv sync --extra dev
.venv/bin/python -m pytest
```

On Windows, use `.venv\Scripts\python.exe` instead of `.venv/bin/python`.

For a different accelerator build, configure uv's PyTorch index before syncing. Verify CUDA availability in the resulting environment. The regular test suite runs the CPU integration matrix; CUDA-marked tests additionally run on machines with a CUDA device. See [release validation](docs/publishing.md) for the explicit GPU check.

Before opening a pull request, run:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check swarmbots tests examples
uv build --no-sources
```

New scenarios should expose Gymnasium spaces, deterministic seeded resets, reward terms in `info["reward_terms"]`, and a success signal when the task has a terminal objective. Add the scenario to the registry only after its semantics and default parameters are documented.

Do not add training algorithms or experiment outputs here. This repository intentionally contains only the benchmark; research code belongs in a separate project.
