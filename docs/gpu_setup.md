# GPU setup

SwarmBots uses NVIDIA CUDA through PyTorch and MuJoCo Warp. Start with a recent NVIDIA driver and verify that `nvidia-smi` sees the intended GPU.

## Source checkout

The checkout includes a CUDA PyTorch source and a default `cuda` dependency group. Install the locked environment:

```bash
uv sync --locked
```

On Windows and Linux, this selects PyTorch `2.12.1+cu132`. Windows also installs `triton-windows==3.7.1.post27`; Linux receives its matching Triton dependency from PyTorch. Other platforms use PyPI's PyTorch build.

Ordinary `uv sync` also includes the default CUDA group, so subsequent syncs retain the CUDA build and Windows Triton. No local configuration edits or manual PyTorch reinstall are needed. CUDA PyTorch can also run the benchmark with `--device cpu`. Add `--extra wandb` or `--extra prompt` if you want optional logging or interactive training prompts.

The tested Windows combination is Python 3.13, PyTorch `2.12.1+cu132`, MuJoCo `3.10.0`, MuJoCo Warp `3.10.0.1`, Warp `1.14.0`, and `triton-windows==3.7.1.post27`, on an RTX 5070 Ti. This is a known working combination, not a claim that every version allowed by the package metadata has been tested. The Linux commands below are provided for setup; this local validation was on Windows.

## Using the published package

The PyPI package declares a standard PyTorch dependency. Configure accelerator selection in the application project that installs SwarmBots. For the validated CUDA build, add these entries to that project's `pyproject.toml`, merging any existing uv tables:

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

Declare the matching PyTorch and Windows compiler dependencies in that application project:

```bash
uv add "torch==2.12.1"
uv add "triton-windows==3.7.1.post27; sys_platform == 'win32'"
uv sync
```

The explicit index applies only to PyTorch; other packages continue to come from PyPI. This selection is retained by future syncs. See [uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/) for other CUDA builds and driver compatibility choices.

## Linux

Install a C++ compiler such as GCC using your distribution's package manager. The CUDA PyTorch distribution supplies its matching Triton dependency. After syncing, check the selected build and run the compiled path:

```bash
.venv/bin/python -c "import torch; print(torch.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
.venv/bin/swarmbots smoke SwarmBots-WallEasy-v0 --device cuda --num-envs 2 --compiled
```

## Windows

Install Visual Studio Build Tools with **Desktop development with C++** and a Windows SDK. Run these commands from its **x64 Native Tools Command Prompt**, where `cl.exe` and the SDK are available:

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
.venv\Scripts\swarmbots.exe smoke SwarmBots-WallEasy-v0 --device cuda --num-envs 2 --compiled
```

Triton must match the PyTorch minor version: the tested PyTorch 2.12 build uses Triton 3.7. Consult the [Triton Windows compatibility table](https://github.com/triton-lang/triton-windows#3-pytorch) before changing either version.

## What the smoke test checks

Without `--compiled`, the smoke command exercises simulation and settled resets with eager PyTorch operations. With `--compiled`, it explicitly compiles both environment tensor operations and scenario rewards; compiler failures propagate and the JSON summary reports `"compiled": true` only after stepping succeeds. The first invocation can take substantially longer while kernels compile.
