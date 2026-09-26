# SwarmBots

SwarmBots is a GPU-vectorized multi-agent reinforcement-learning benchmark for self-assembling modular robots. Each policy controls identical articulated units that can move independently and create or release load-bearing connections during an episode. The collective's physical graph therefore changes as part of the control problem.

The benchmark is built on [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) and exposes Gymnasium-compatible vector environments backed by PyTorch tensors. It includes obstacle traversal, partial-observability, climbing, navigation, and payload-transport tasks.

This repository deliberately does **not** contain a training framework or paper-specific experiment configurations. Bring your own MARL implementation and use the stable registry and evaluation API here.

## Install

Python 3.11 or newer is required. CUDA is strongly recommended; CPU execution exists for development and tests but is not the benchmark's performance target. SwarmBots is currently versioned as an alpha. Once `0.1.0a1` is published to PyPI, install it explicitly with:

```bash
uv add "swarm-bots==0.1.0a1"
```

Install the PyTorch build appropriate for your system first, then install from source:

```bash
git clone https://github.com/brn-dev/swarm-bots.git
cd swarm-bots
uv sync
```

For contributors:

```bash
uv sync --extra dev
```

## Quick start

List the registered tasks and run a short smoke test:

```bash
swarmbots list
swarmbots smoke SwarmBots-WallEasy-v0 --device cuda --num-envs 64
```

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

## Multi-agent interface

Observations are dictionaries of batched tensors:

- `local_obs`: per-agent observations, shaped `(worlds, agents, features)`.
- `global_obs`: task information visible to all agents.
- `agent_mask`: which padded agent slots are active.
- `hidden_local_vars` and `hidden_global_vars`: privileged state for centralized training or diagnostics. Do not pass these to an evaluated actor.

Actions contain per-agent `actuators` and `connectors` tensors. Rewards and done flags are team-level tensors with one value per simulated world.

The built-in `evaluate_policy` function passes only the non-privileged observation keys to the policy and reports episode returns, lengths, and success rate where defined.

## Documentation

- [Scenario catalog](docs/scenarios.md)
- [Benchmark and reporting protocol](docs/benchmark_protocol.md)
- [Python API and custom policies](docs/api.md)
- [Contributing](CONTRIBUTING.md)
- [Publishing releases](docs/publishing.md)

The API is alpha. Benchmark IDs and protocol versions are explicit so semantic changes can be introduced without silently invalidating results.

## Citation

The benchmark and its original evaluation are described in:

> Dominik Baron. *SwarmBots: A GPU-Accelerated Multi-Agent Continuous Control Benchmark with Transformer Baselines*. Master's thesis, Johannes Kepler University Linz, 2026.

The complete [thesis and reproducibility artifact](https://github.com/brn-dev/msc-thesis-swarmbots-qcx-tmasac-nop-smb) contains the thesis PDF, training algorithms, experiment configurations, and analysis code. Machine-readable citation metadata is available in [CITATION.cff](CITATION.cff).

## License

SwarmBots is released under the [Apache License 2.0](LICENSE).
