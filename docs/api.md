# Benchmark Python API

Use `swarmbots` to discover tasks, create parallel environments, evaluate your own policies, and record their behavior. These APIs work independently of the included learning baselines. See the [scenario catalog](scenarios.md) for task definitions and the [benchmark protocol](benchmark_protocol.md) for comparable evaluation.

## Registry

```python
from swarmbots import CORE_BENCHMARK_IDS, get_benchmark_spec, list_benchmarks

for spec in list_benchmarks():
    print(spec.id, spec.maturity, spec.description)

print(CORE_BENCHMARK_IDS)
spec = get_benchmark_spec("SwarmBots-Climb-v0")
print(spec.maturity, spec.episode_length, spec.supports_success_metric)
```

`CORE_BENCHMARK_IDS` contains the eight core tasks; `ALL_BENCHMARK_IDS` contains all 15 registered tasks. Benchmark IDs belong to SwarmBots' own registry, so construct them with `swarmbots.make_env`.

`BenchmarkSpec.maturity` and `spec.to_dict()["maturity"]` expose each task's alpha or beta status. The benchmark as a whole remains alpha; core-suite membership does not imply a validated task. See [scenario maturity and versioning](scenarios.md#scenario-maturity-and-versioning) for the testing criteria and version identifiers.

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

`env.single_observation_space` and `env.single_action_space` describe one world. `env.observation_space` and `env.action_space` include the leading world dimension. Observations contain per-agent `local_obs`, shared `global_obs`, `agent_mask`, and privileged `hidden_local_vars` / `hidden_global_vars`. Actions contain per-agent `actuators` and `connectors`; rewards, terminations, and truncations have one value per world. See the [observation contract](scenarios.md#observation-contract) and [action contract](scenarios.md#action-contract) for field meanings and bounds.

`make_env` forwards `scenario_kwargs` to the registered scenario factory and remaining keyword arguments to `MJWSwarmBotsVectorEnv`. `make_scenario(benchmark_id, seed=..., **scenario_kwargs)` constructs a scenario without a vector environment. Overrides create custom task variants and must be reported with their results.

## Gymnasium compatibility

The environment subclasses Gymnasium's `VectorEnv` and uses Gymnasium spaces, the `reset()` and `step()` return conventions, and `SAME_STEP` autoreset. Observations, rewards, termination flags, and truncation flags are PyTorch tensors on `env.device`. Actions are dictionaries of batched tensors; `info` contains tensors and nested dictionaries.

Compatibility with Gymnasium's interface does not imply compatibility with every Gymnasium wrapper or training library. Wrappers that assume NumPy arrays, including `gymnasium.wrappers.vector.RecordEpisodeStatistics`, cannot be applied directly. Use tensor-aware integration code or an adapter that handles device transfer, nested values, and terminal observations. Converting CUDA tensors to CPU NumPy arrays adds transfers and synchronization to the rollout loop.

The built-in evaluator consumes the tensor interface directly.

## Episode lifecycle

The environment uses `SAME_STEP` autoreset. On a done step, the returned observation belongs to the reset episode. The terminal observation is in `info["final_obs"]`, selected by `info["_final_obs"]`. Bootstrap a truncated episode from that terminal observation; terminated episodes should not bootstrap.

Observation tensors can share environment buffers. Clone tensors you need to retain across calls to `step()` or `reset()`.

Every reset, including the first reset and `options={"reset_mask": mask}` partial resets, applies the scenario's physics settling. Settling is outside the episode's control-step budget and return. Repeating `reset(seed=...)` with the same seed and configuration reproduces the initial observations on the same device/backend. Reseeding also clears prefetched reset snapshots. Partial resets preserve the unselected lanes' physical state.

The old `settle_initial_reset` constructor argument and `force_settled` reset option are no longer needed. Remove them from callers. To disable settling for a custom variant, set `scenario_kwargs={"reset_settle_time": 0.0}`; this affects both explicit resets and autoresets and is not a canonical benchmark configuration.

## Policy evaluation

A policy is any callable with this signature:

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

Use `policy.eval()` and frozen normalization for a trained neural network. The evaluator uses `torch.inference_mode()` but cannot configure an arbitrary callable's action distribution. `action_mode` records your choice; it does not change policy behavior. The evaluator raises an error if a success-enabled task fails to provide `info["success"]`.

Only the first episode per lane contributes. Results are ordered by lane, not completion time. `num_episodes <= num_envs` is required; requesting fewer episodes constructs only that many worlds. Reset recurrent state using `episode_starts`, including for lanes whose first episode has already completed.

This call evaluates one seed. Use the five-seed reporting workflow below for a protocol report.

## Five-seed report

Run the included uniform-random policy as a stochastic sanity baseline:

```bash
.venv/bin/python examples/random_policy.py SwarmBots-WallEasy-v0 --device cuda --output runs/wall-random.json
```

On Windows, replace `.venv/bin/python` with `.venv\Scripts\python.exe`. The default run uses 256 worlds for each seed `1000` through `1004`. For a smaller development run, pass `--num-envs 2 --seeds 1000 --device cpu`. CPU episodes can still be slow; use the CLI smoke test for an installation check.

The JSON contains every raw episode, per-seed metadata, the mean of seed-level mean returns, and the population standard deviation across those seed means. Success rates are aggregated the same way for tasks that define success. This random policy is labeled stochastic, so it is not a canonical deterministic-policy result.

Adapt the example to load your trained policy, set `action_mode="deterministic"`, and supply checkpoint metadata. Keep the five-seed loop and aggregation. Reinitialize policy state for each seed; use `--source-revision` to record the benchmark commit. See the [protocol](benchmark_protocol.md) for reporting requirements.

For a sweep across many different tasks or world counts, use one process per task/configuration. PyTorch caches compiled variants by function and can exhaust its specialization limit when many observation shapes share one process. If running tasks sequentially in one process, close the previous environment and call `torch.compiler.reset()` before constructing the next; this also clears compiled policy caches. The GPU integration matrix isolates compiler state between cases for this reason.

## Deferred stepping

CUDA environments support `begin_step(actions)` followed by `end_step()`. This lets a caller overlap host work with asynchronous device execution. Ordinary callers should use `step(actions)`.

## Recording

Use `swarmbots.record_policy(policy, benchmark_id, video_folder=...)` for any benchmark policy callable, or `swarmbots.learn.record_checkpoint(...)` to load a learning preset directly. The `swarmbots record` command and `examples/record_policy.py` expose these helpers. See [recording](recording.md) for checkpoint/custom-policy examples and camera, episode, and rendering options.

Call `env.start_video_recording(...)` before stepping. Recording renders selected worlds with MuJoCo and writes MP4 files asynchronously. It is intended for qualitative inspection, not for high-throughput evaluation.

## Reference learning baselines

If you want an included training implementation, `swarmbots.learn` provides PPO/MAPPO, MADDPG, MATD3, MASAC, MAT, TMASAC, TMATD3, and recurrent presets. The MADDPG/MATD3/MASAC baselines offer flattened MLP or Deep Set critics; TMATD3 offers transformer and decentralized actors with transformer critics. See [reference learning baselines](learning.md) for training, checkpoint continuation, and `as_benchmark_policy`, which adapts their actors to this policy interface.
