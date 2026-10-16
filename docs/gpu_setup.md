# GPU setup

SwarmBots uses NVIDIA CUDA through PyTorch and MuJoCo Warp. Start with a recent NVIDIA driver and verify that `nvidia-smi` sees the intended GPU.

Use [source checkout](#source-checkout) for a cloned repository or [published-package setup](#using-the-published-package) for an application project. Then run the verification commands for [Linux](#linux) or [Windows](#windows). A successful check identifies the GPU and completes a compiled smoke test.

Normal CUDA training, evaluation, and recording compile environment tensor operations by default. The training example's `--compile` flag separately enables policy and auxiliary-model compilation. The ordinary CLI smoke test deliberately uses eager environment operations, so it can succeed before compiler setup is complete.

## Source checkout

The checkout includes a CUDA PyTorch source and a default `cuda` dependency group. Install the locked environment:

```bash
uv sync --locked
```

On Windows and Linux, this selects PyTorch `2.12.1+cu132`. Windows also installs `triton-windows==3.7.1.post27`; Linux receives its matching Triton dependency from PyTorch. Other platforms use PyPI's PyTorch build.

Ordinary `uv sync` also includes the default CUDA group, so subsequent syncs retain the CUDA build and Windows Triton. No local configuration edits or manual PyTorch reinstall are needed. CUDA PyTorch can also run the benchmark with `--device cpu`. Add `--extra wandb` or `--extra prompt` for optional logging or interactive training prompts.

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

Install a C++ compiler such as GCC using the distribution's package manager. The CUDA PyTorch distribution supplies its matching Triton dependency. After syncing, check the selected build and run the compiled path:

```bash
.venv/bin/python -c "import torch; print(torch.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
.venv/bin/swarmbots smoke SwarmBots-WallEasy-v0 --device cuda --num-envs 2 --compiled
```

### Headless video recording

Video recording also requires EGL/OpenGL system libraries. This matters for the first training example, which records videos after evaluation: a simulation-only smoke test does not check rendering.

On Debian/Ubuntu, the repository's CI uses:

```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends libegl1 libegl-mesa0 libgl1-mesa-dri
```

SwarmBots selects `MUJOCO_GL=egl` automatically on Linux when no display or backend override is present. Other distributions require equivalent EGL/OpenGL packages and a working driver rendering backend.

## Windows

Install Visual Studio Build Tools with **Desktop development with C++** and a Windows SDK. Run these commands from its **x64 Native Tools Command Prompt**, where `cl.exe` and the SDK are available:

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
.venv\Scripts\swarmbots.exe smoke SwarmBots-WallEasy-v0 --device cuda --num-envs 2 --compiled
```

Triton must match the PyTorch minor version: the tested PyTorch 2.12 build uses Triton 3.7. Consult the [Triton Windows compatibility table](https://github.com/triton-lang/triton-windows#3-pytorch) before changing either version.

## What the smoke test checks

Without `--compiled`, the smoke command exercises simulation and settled resets with eager PyTorch operations. With `--compiled`, it explicitly compiles both environment tensor operations and scenario rewards; compiler failures propagate and the JSON summary reports `"compiled": true` only after stepping succeeds. The first invocation can take substantially longer while kernels compile.

## Troubleshooting

| Symptom | Next step |
| --- | --- |
| `swarmbots` is not found | Use `.venv/bin/swarmbots` on Linux or `.\.venv\Scripts\swarmbots.exe` on Windows, from the project root. |
| PyTorch's CUDA check fails | Check `nvidia-smi`, then verify that CUDA PyTorch is installed in the same environment. Windows application projects need the [published-package configuration](#using-the-published-package). |
| Simulation works but `--compiled` fails | Check the C++ compiler and matching Triton setup in the platform instructions above. On Windows, use the x64 Native Tools Command Prompt. |
| The first run takes a long time | Kernel compilation can take longer on the first invocation. Start with two worlds and check the final summary. |
| A run exceeds GPU memory | Reduce parallel worlds with `--num-envs`; the training example also supports `--eval-envs` and `--model-scale 2.5M`. |
| Simulation succeeds but headless recording fails | Check the [EGL/OpenGL prerequisites](#headless-video-recording) and driver rendering backend. The smoke test does not render video. |

For a small CPU installation check, run `swarmbots smoke SwarmBots-WallEasy-v0 --device cpu --num-envs 2`. This checks simulation; it does not verify CUDA or compilation. If setup still fails, include the full command, error, Python/PyTorch versions, and GPU/driver information in a [bug report](https://github.com/brn-dev/swarm-bots/issues).
