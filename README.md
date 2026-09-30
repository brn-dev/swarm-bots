# SwarmBots

SwarmBots is a GPU-vectorized multi-agent reinforcement-learning benchmark for self-assembling modular robots. Each policy controls identical articulated units that can move independently and create or release load-bearing connections during an episode. The collective's physical graph therefore changes as part of the control problem.

The benchmark is built on [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) and exposes vector environments with Gymnasium spaces and reset/step conventions, using PyTorch tensors on the simulation device. Gymnasium wrappers and training libraries that expect NumPy arrays require adaptation. It includes obstacle traversal, partial-observability, climbing, navigation, and payload-transport tasks.

This repository deliberately does **not** contain a training framework or paper-specific experiment configurations. Bring your own MARL implementation and use the stable registry and evaluation API here.

| Wall traversal | Finding a hidden opening |
| --- | --- |
| ![SwarmBots wall traversal](https://raw.githubusercontent.com/brn-dev/swarm-bots/main/docs/assets/wall.gif) | ![SwarmBots finding an opening](https://raw.githubusercontent.com/brn-dev/swarm-bots/main/docs/assets/find-opening.gif) |

These rollouts illustrate the tasks; they are not reference scores for protocol 0.1. See the [scenario catalog](https://github.com/brn-dev/swarm-bots/blob/main/docs/scenarios.md) for task definitions.

## Install

Python 3.11 or newer is required. CUDA is strongly recommended; CPU execution exists for development and tests but is not the benchmark's performance target. SwarmBots is currently versioned as an alpha. Once `0.1.0a1` is published to PyPI, install it explicitly with:

```bash
uv add "swarm-bots==0.1.0a1"
```

Install from source:

```bash
git clone https://github.com/brn-dev/swarm-bots.git
cd swarm-bots
uv sync
```

The lockfile selects the dependency versions. Follow [GPU setup](https://github.com/brn-dev/swarm-bots/blob/main/docs/gpu_setup.md) to select a CUDA PyTorch build, configure the compiler, and verify the compiled environment on Linux or Windows. An exact sync may replace a manually installed accelerator build, so keep the index selection in your project configuration.

For contributors:

```bash
uv sync --extra dev
```

## Quick start

List the registered tasks and run a short smoke test:

```bash
.venv/bin/swarmbots list
.venv/bin/swarmbots smoke SwarmBots-WallEasy-v0 --device cuda --num-envs 64
```

On Windows, use `.venv\Scripts\swarmbots.exe`. No environment activation is required. For a CPU smoke test, use `--device cpu --num-envs 2`; CPU simulation is much slower than the intended CUDA path.

Add `--compiled` to verify PyTorch compilation as well as simulation. The ordinary smoke test uses eager operations; it does not check compiler setup.

Create an environment directly:

```python
import torch
from swarmbots import make_env

env = make_env("SwarmBots-WallEasy-v0", num_envs=256, device="cuda", seed=42)
observations, info = env.reset(seed=42)

actions = {
    "actuators": torch.zeros(env.action_space["actuators"].shape, device=env.device),
    "connectors": torch.zeros(env.action_space["connectors"].shape, device=env.device),
}
observations, rewards, terminations, truncations, info = env.step(actions)
env.close()
```

The environment uses Gymnasium `SAME_STEP` autoreset. When a lane ends, the returned observation is already its next reset observation; the terminal observation is in `info["final_obs"]` and selected by `info["_final_obs"]`.

Explicit resets and autoresets both settle physics before returning observations. The settling interval is outside the episode's control-step budget, so evaluation and training start from the same reset distribution.

## Multi-agent interface

Observations are dictionaries of batched tensors:

- `local_obs`: per-agent observations, shaped `(worlds, agents, features)`.
- `global_obs`: task information visible to all agents.
- `agent_mask`: which padded agent slots are active.
- `hidden_local_vars` and `hidden_global_vars`: privileged state for centralized training or diagnostics. Do not pass these to an evaluated actor.

Actions contain per-agent `actuators` and `connectors` tensors. Rewards and done flags are team-level tensors with one value per simulated world.

The built-in `evaluate_policy` function passes only the non-privileged observation keys to the policy and reports episode returns, lengths, and success rate where defined. Centralized, partially centralized, and decentralized actors are all allowed; actors may combine the permitted observations across agents within each world.

It collects only the first episode from each world. `num_episodes` must be no greater than `num_envs`; use additional seeds for more samples. The [evaluation example](https://github.com/brn-dev/swarm-bots/blob/main/docs/api.md#five-seed-report) writes raw episodes, settings, runtime versions, and statistics across five seeds to JSON.

## Documentation

- [Scenario catalog](https://github.com/brn-dev/swarm-bots/blob/main/docs/scenarios.md)
- [GPU setup and compiled smoke test](https://github.com/brn-dev/swarm-bots/blob/main/docs/gpu_setup.md)
- [Benchmark and reporting protocol](https://github.com/brn-dev/swarm-bots/blob/main/docs/benchmark_protocol.md)
- [Python API, compatibility, and custom policies](https://github.com/brn-dev/swarm-bots/blob/main/docs/api.md)
- [Contributing](https://github.com/brn-dev/swarm-bots/blob/main/CONTRIBUTING.md)
- [Publishing releases](https://github.com/brn-dev/swarm-bots/blob/main/docs/publishing.md)

The API is alpha. Benchmark IDs and protocol versions are explicit so semantic changes can be introduced without silently invalidating results.

## Citation

The benchmark and its original evaluation are described in:

> Dominik Baron. *SwarmBots: A GPU-Accelerated Multi-Agent Continuous Control Benchmark with Transformer Baselines*. Master's thesis, Johannes Kepler University Linz, 2026.

The complete [thesis and reproducibility artifact](https://github.com/brn-dev/msc-thesis-swarmbots-qcx-tmasac-nop-smb) contains the thesis PDF, training algorithms, experiment configurations, and analysis code. Machine-readable citation metadata is available in [CITATION.cff](https://github.com/brn-dev/swarm-bots/blob/main/CITATION.cff).

## License

SwarmBots is released under the [Apache License 2.0](https://github.com/brn-dev/swarm-bots/blob/main/LICENSE).
