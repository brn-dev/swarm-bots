# GPU setup

SwarmBots uses NVIDIA CUDA through PyTorch and MuJoCo Warp. Start with a recent NVIDIA driver and verify that `nvidia-smi` sees the intended GPU. The regular installation can select a CPU-only PyTorch build, particularly on Windows.

## Install CUDA PyTorch in the existing environment

After installing the project dependencies, replace CPU-only PyTorch with the CUDA build used for this alpha's validation:

```bash
uv pip install --python .venv/Scripts/python.exe --index https://download.pytorch.org/whl/cu132 --reinstall-package torch "torch==2.12.1+cu132"
uv pip install --python .venv/Scripts/python.exe "triton-windows==3.7.1.post27"
```

These commands target Windows. On Linux, use `.venv/bin/python` and omit the `triton-windows` command; CUDA PyTorch installs its matching Triton dependency.

This replaces packages in the environment without changing the project configuration. A later `uv sync` can restore the CPU build or remove Windows Triton. To retain the CUDA selection across syncs, configure its source as described below.

## Retain a CUDA build with uv

For a source checkout, add the following to `pyproject.toml`. Merge these entries into any existing uv tables:

```toml
[tool.uv.sources]
torch = [
    { index = "pytorch-cu132", marker = "sys_platform == 'win32' or sys_platform == 'linux'" },
]

[[tool.uv.index]]
name = "pytorch-cu132"
url = "https://download.pytorch.org/whl/cu132"
explicit = true
```

Then select the PyTorch version used for this alpha's Windows CUDA validation:

```bash
uv add "torch==2.12.1"
uv sync --extra dev
```

This changes your local project configuration and lockfile so future syncs retain the selected accelerator source. The explicit index applies only to PyTorch; other packages continue to come from PyPI. See [uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) for other CUDA builds and driver compatibility choices.

The tested Windows combination is Python 3.13, PyTorch `2.12.1+cu132`, MuJoCo `3.10.0`, MuJoCo Warp `3.10.0.1`, Warp `1.14.0`, and `triton-windows==3.7.1.post27`, on an RTX 5070 Ti. This is a known working combination, not a claim that every version allowed by the package metadata has been tested. The Linux commands below are provided for setup; this local validation was on Windows.

## Linux

Install a C++ compiler such as GCC using your distribution's package manager. The CUDA PyTorch distribution supplies its matching Triton dependency. After syncing, check the selected build and run the compiled path:

```bash
.venv/bin/python -c "import torch; print(torch.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
.venv/bin/swarmbots smoke SwarmBots-WallEasy-v0 --device cuda --num-envs 2 --compiled
```

## Windows

Install Visual Studio Build Tools with **Desktop development with C++** and a Windows SDK. Run these commands from its **x64 Native Tools Command Prompt**, where `cl.exe` and the SDK are available:

```powershell
uv add "triton-windows==3.7.1.post27; sys_platform == 'win32'"
uv sync --extra dev
.venv\Scripts\python.exe -c "import torch; print(torch.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
.venv\Scripts\swarmbots.exe smoke SwarmBots-WallEasy-v0 --device cuda --num-envs 2 --compiled
```

Triton must match the PyTorch minor version: the tested PyTorch 2.12 build uses Triton 3.7. Consult the [Triton Windows compatibility table](https://github.com/triton-lang/triton-windows#3-pytorch) before changing either version.

## What the smoke test checks

Without `--compiled`, the smoke command exercises simulation and settled resets with eager PyTorch operations. With `--compiled`, it explicitly compiles both environment tensor operations and scenario rewards; compiler failures propagate and the JSON summary reports `"compiled": true` only after stepping succeeds. The first invocation can take substantially longer while kernels compile.

For the release integration checks, run:

```bash
.venv/bin/python -m pytest -m cuda
```

On Windows, use `.venv\Scripts\python.exe`. These tests require CUDA; verify availability with the command above so an unavailable GPU does not silently turn the run into skipped tests. See [publishing](publishing.md) for the complete release process.
