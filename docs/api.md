# Benchmark Python API

`swarmbots` exposes task discovery, parallel environments, custom-policy evaluation, and video recording. These APIs work independently of the included learning baselines. The [scenario catalog](scenarios.md) defines the tasks; the [benchmark protocol](benchmark_protocol.md) specifies comparable evaluation.

## Registry

```python
from swarmbots import CORE_BENCHMARK_IDS, get_benchmark_spec, list_benchmarks

for spec in list_benchmarks():
    print(spec.id, spec.maturity, spec.description)

print(CORE_BENCHMARK_IDS)
spec = get_benchmark_spec("SwarmBots-Climb-v0")
print(spec.maturity, spec.episode_length, spec.supports_success_metric)
```

`CORE_BENCHMARK_IDS` contains the eight core tasks; `ALL_BENCHMARK_IDS` contains all 15 registered tasks. Benchmark IDs belong to SwarmBots' own registry; `swarmbots.make_env` constructs their environments.

`BenchmarkSpec.maturity` and `spec.to_dict()["maturity"]` expose each task's alpha or beta status. The benchmark as a whole remains alpha; core-suite membership does not imply a validated task. [Scenario maturity and versioning](scenarios.md#scenario-maturity-and-versioning) defines the testing criteria and version identifiers.

The CLI exposes the same task descriptions and maturity labels:

```bash
swarmbots list
swarmbots list --json
swarmbots describe SwarmBots-Climb-v0
```

## Create and step an environment

```python
import torch
from swarmbots import make_env

env = make_env("SwarmBots-Climb-v0", num_envs=256, device="cuda", seed=42)
try:
    observations, info = env.reset(seed=42)
    actions = {
        key: torch.zeros(space.shape, device=env.device)
        for key, space in env.action_space.items()
    }
    observations, rewards, terminations, truncations, info = env.step(actions)
finally:
    env.close()
```

This example applies zero actions to the registered task. Use `num_envs=2, device="cpu"` for a small development run. CPU simulation is much slower than the intended CUDA path; see [GPU setup](gpu_setup.md) before running larger batches.

`env.single_observation_space` and `env.single_action_space` describe one world. The spaces without `single_` include the leading world dimension.

| Field | Meaning | Typical dimensions |
| --- | --- | --- |
| `local_obs` | Each robot's public observations | `(worlds, agents, features)` |
| `global_obs` | Public task information | `(worlds, features)` |
| `agent_mask` | Active robot slots; exclude inactive padding | `(worlds, agents)` |
| `hidden_local_vars`, `hidden_global_vars` | Privileged training/diagnostic information; exclude from actors | Per-agent or per-world |
| `actuators`, `connectors` | Per-agent action tensors | `(worlds, agents, actions)` |
| Rewards, terminations, truncations | Team outcome for each world | `(worlds,)` |

The selected task's spaces define exact dimensions and action bounds. The [observation contract](scenarios.md#observation-contract) and [action contract](scenarios.md#action-contract) describe field meanings.

`make_env` forwards `scenario_kwargs` to the registered scenario factory and remaining keyword arguments to `MJWSwarmBotsVectorEnv`. `make_scenario(benchmark_id, seed=..., **scenario_kwargs)` constructs a scenario without a vector environment. Overrides create custom task variants and must be reported with their results.

## Gymnasium compatibility

The environment subclasses Gymnasium's `VectorEnv` and uses Gymnasium spaces and the `reset()` and `step()` return conventions. Autoreset defaults to `SAME_STEP`; `NEXT_STEP` is available for external learners. Observations, rewards, termination flags, and truncation flags are PyTorch tensors on `env.device`. Actions are dictionaries of batched tensors; `info` contains tensors and nested dictionaries.

Compatibility with Gymnasium's interface does not imply compatibility with every Gymnasium wrapper or training library. Wrappers that assume NumPy arrays, including `gymnasium.wrappers.vector.RecordEpisodeStatistics`, cannot be applied directly. Integration requires tensor-aware code or an adapter that handles device transfer, nested values, and terminal observations. Converting CUDA tensors to CPU NumPy arrays adds transfers and synchronization to the rollout loop.

The built-in evaluator consumes the tensor interface directly. CUDA environments compile tensor operations by default; the [GPU setup guide](gpu_setup.md) covers compiler prerequisites.

## Integrate a custom learner

The [standalone PyTorch PPO example](../examples/README.md#train-without-the-included-learning-framework) shows a minimal training loop without `swarmbots.learn`: environment construction, squashed Gaussian actions, active-agent masking, rollout storage, advantage estimation, and optimizer updates. Both actor and critic use only public observations; the returned model can be evaluated directly.

The example handles reusable observation buffers, time-limit bootstrapping from terminal observations, and temporal traces across autoreset boundaries. Libraries expecting NumPy arrays need an adapter as described above.

## Episode lifecycle

The default `SAME_STEP` autoreset returns the reset episode's observation on a done step. The terminal observation is in `info["final_obs"]`, selected by `info["_final_obs"]`. A truncated episode bootstraps from that terminal observation; a terminated episode does not bootstrap.

External learners expecting [Gymnasium's `NEXT_STEP` lifecycle](https://farama.org/Vector-Autoreset-Mode) can select it at construction:

```python
from gymnasium.vector import AutoresetMode
from swarmbots import make_env

env = make_env(
    "SwarmBots-Climb-v0", num_envs=256, device="cuda",
    autoreset_mode=AutoresetMode.NEXT_STEP,
)
```

In `NEXT_STEP`, a done step returns the terminal observation directly, without `final_obs` or `_final_obs`. On the following `step()` call, only the finished worlds reset: their supplied actions are ignored, rewards are zero, and termination/truncation flags are false. Other worlds advance normally. External learners must exclude reset-only entries from transition storage. The returned reset observations have not advanced through a control step. Explicit partial resets cancel pending autoresets only for the selected worlds.

`env.autoreset_mode`, `env.metadata["autoreset_mode"]`, and `env.get_settings()["autoreset_mode"]` identify the selected lifecycle; settings store its string value (`"SameStep"` or `"NextStep"`). The matching enum or string is accepted at construction. `DISABLED` autoreset is unsupported. The included learning stack continues to require `SAME_STEP` and rejects `NEXT_STEP` environments.

Observation tensors can share environment buffers. Clone tensors that must be retained across calls to `step()` or `reset()`.

Every reset applies the scenario's physics settling before returning observations. Settling is outside the episode's control-step budget and return. Use `reset(seed=...)` for reproducible initial sampling with the same configuration and backend; numerical trajectories can still differ between devices.

For a partial reset, pass `options={"reset_mask": mask}` to preserve unselected worlds. Disabling settling with `scenario_kwargs={"reset_settle_time": 0.0}` creates a custom task variant.

## Policy evaluation

A policy is any callable with this signature. Replace the action variables below with the model's bounded outputs, matching the task's batched action spaces:

```python
from collections.abc import Mapping
import torch

def policy(
    observations: Mapping[str, torch.Tensor],
    episode_starts: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "actuators": actuator_actions,
        "connectors": connector_actions,
    }
```

The tensors stay on the environment device. `episode_starts` is true on the initial call and on the call immediately following an autoreset, so recurrent policies can reset state per lane.

```python
from swarmbots import evaluate_policy

result = evaluate_policy(
    policy,
    "SwarmBots-POWallMedium-v0",
    num_envs=256,
    num_episodes=256,
    seed=1000,
    action_mode="deterministic",
    policy_metadata={"name": "my-policy", "checkpoint": "checkpoints/final.pt"},
)
print(result.to_dict())
```

Only `local_obs`, `global_obs`, and `agent_mask` are passed to the policy. The policy receives these for all agents in each world and may combine them in a centralized or partially centralized actor, or process them separately in a decentralized actor. The benchmark does not require decentralized execution. Privileged observation keys remain excluded for every actor architecture.

Use `policy.eval()` and frozen normalization for a trained neural network. The evaluator uses `torch.inference_mode()` but cannot configure an arbitrary callable's action distribution. `action_mode` records the configured action mode; it does not change policy behavior. The evaluator raises an error if a success-enabled task fails to provide `info["success"]`.

Only the first episode per lane contributes. Results are ordered by lane, not completion time. `num_episodes <= num_envs` is required; requesting fewer episodes constructs only that many worlds. Reset recurrent state using `episode_starts`, including for lanes whose first episode has already completed.

This call evaluates one seed. The five-seed workflow below produces a protocol report.

## Five-seed report

Run the included uniform-random policy as a stochastic sanity baseline:

```bash
.venv/bin/python examples/evaluate_random_policy.py SwarmBots-WallEasy-v0 --device cuda --output runs/wall-random.json
```

On Windows, replace `.venv/bin/python` with `.venv\Scripts\python.exe`. The default run uses 256 worlds for each seed `1000` through `1004`. For a smaller development run, pass `--num-envs 2 --seeds 1000 --device cpu`. CPU episodes can still be slow; use the CLI smoke test for an installation check.

The JSON contains every raw episode, per-seed metadata, the mean of seed-level mean returns, and the population standard deviation across those seed means. Success rates are aggregated the same way for tasks that define success. This random policy is labeled stochastic, so it is not a canonical deterministic-policy result.

Adapt the example to load a trained policy, set `action_mode="deterministic"`, and supply checkpoint metadata. Keep the five-seed loop and aggregation. Reinitialize policy state for each seed; use `--source-revision` to record the benchmark commit. See the [protocol](benchmark_protocol.md) for reporting requirements.

The [saved-checkpoint report recipe](learning.md#five-seed-report-from-a-saved-checkpoint) provides a complete five-seed workflow for an included trained policy, including loading, aggregation, and JSON export.

For sweeps across many task shapes or world counts, use one process per configuration to avoid exhausting PyTorch's compilation cache. Close environments when finished.

## Deferred stepping

`step(actions)` provides the ordinary stepping interface. The CUDA `begin_step(actions)` / `end_step()` pair supports overlap between host work and device execution. Environment operations follow the active PyTorch stream.

## Recording

`swarmbots.record_policy(policy, benchmark_id, video_folder=...)` records any benchmark policy callable; `swarmbots.learn.record_checkpoint(...)` loads and records an included learning preset. The `swarmbots record` command and `examples/record_policy.py` expose these helpers. The [recording guide](recording.md) covers checkpoint/custom-policy examples and camera, episode, and rendering options.

For recording during a custom rollout, call `env.start_video_recording(...)` before stepping. Recording is intended for qualitative inspection and adds rendering and encoding work.

## Reference learning baselines

The [learning guide](learning.md) covers included presets, training, and checkpoint continuation. `as_benchmark_policy(trainer)` adapts those actors to the evaluator's policy interface and freezes observation normalization.
