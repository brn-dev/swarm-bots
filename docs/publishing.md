# Publishing releases

Releases are built and uploaded with uv through `.github/workflows/release.yml`. The publish job uses PyPI Trusted Publishing, so the repository does not need a long-lived PyPI token.

## One-time PyPI setup

Create a `pypi` environment under the GitHub repository's **Settings → Environments** page. Configure environment protection rules if desired.

On PyPI, add a Trusted Publisher with these exact values:

- PyPI project name: `swarmbots`
- GitHub owner: `brn-dev`
- GitHub repository: `swarm-bots`
- Workflow filename: `release.yml`
- Environment name: `pypi`

For the first upload, configure a pending Trusted Publisher if the PyPI project does not exist yet. PyPI will create the project when the workflow publishes the first distribution.

## Release process

Run the checks from a clean checkout:

```bash
uv sync --locked --extra dev
.venv/bin/python -m ruff check swarmbots tests examples
.venv/bin/python -m pytest
uv build --no-sources
uvx --from twine twine check dist/*
```

On Windows, use `.venv\Scripts\python.exe` instead of `.venv/bin/python`.

Set the next version with uv. Keep `pyproject.toml`, `uv.lock`, `CITATION.cff`, and `CHANGELOG.md` on the same version:

```bash
uv version 0.1.0a1 --no-sync
```

Commit the release metadata, then create and push an annotated tag matching that version:

```bash
git tag -a v0.1.0a1 -m "SwarmBots 0.1.0a1"
git push origin v0.1.0a1
```

The workflow rejects a tag that does not match the version in `pyproject.toml`. It builds and checks the wheel and source distribution in an unprivileged job, then passes only those artifacts to the OIDC-enabled publish job.

The build job depends on the reusable CI workflow: Python 3.11/3.13 tests and lint must pass first. After testing, CI replaces the editable installation with the wheel, verifies that imports resolve inside the virtual environment, and runs the CLI outside the source checkout. Its dependency versions come from the tested environment; this is not a minimum-dependency-version compatibility test.

Before tagging, validate the compiled GPU path on a CUDA machine with the appropriate compiler toolchain:

```bash
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA validation requires a GPU'"
.venv/bin/swarmbots smoke SwarmBots-WallEasy-v0 --device cuda --num-envs 2 --compiled
.venv/bin/python -m pytest -m cuda
```

See [GPU setup](gpu_setup.md) for accelerator selection and compiler prerequisites. Ordinary CI runs on CPU and skips CUDA tests. Record the GPU, driver, dependency versions, and test output in the release notes. The integration matrix covers all registered scenarios with default settling, finite outputs, SAME_STEP reset, and repeatable seeded resets. Additional lifecycle tests cover partial resets, asynchronous settled-reset buffers, and staggered completion across three episodes per lane, with 500-step episode limits on CUDA. These checks do not replace extended stability or throughput measurements.

Push the documentation assets to `main` before publishing: the README uses absolute image URLs so the demonstrations also work on PyPI.

PyPI does not allow replacing a published file. Increment the version before retrying a release whose artifacts need to change.
