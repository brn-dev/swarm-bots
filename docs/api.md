# Python API

Built-in PPO, MAT, TMASAC, recurrent policies, and configurable learning presets are available through `swarmbots.learn`. See [learning algorithms](learning.md) for `make_training`, `train`, and the benchmark policy adapter.

## Registry

```python
from swarmbots import list_benchmarks, make_env, make_scenario

for spec in list_benchmarks():
    print(spec.id, spec.description)

scenario = make_scenario("SwarmBots-Climb-v0", seed=42)
env = make_env("SwarmBots-Climb-v0", num_envs=256, device="cuda", seed=42)
```

`make_env` forwards `scenario_kwargs` to the registered scenario factory and remaining keyword arguments to `MJWSwarmBotsVectorEnv`.

## Gymnasium compatibility

The environment subclasses Gymnasium's `VectorEnv` and uses Gymnasium spaces, the `reset()` and `step()` return conventions, and `SAME_STEP` autoreset. Observations, rewards, termination flags, and truncation flags are PyTorch tensors on `env.device`. Actions are dictionaries of batched tensors; `info` contains tensors and nested dictionaries.

Compatibility with Gymnasium's interface does not imply compatibility with every Gymnasium wrapper or training library. Wrappers that assume NumPy arrays, including `gymnasium.wrappers.vector.RecordEpisodeStatistics`, cannot be applied directly. Use tensor-aware integration code or an adapter that handles device transfer, nested values, and terminal observations. Converting CUDA tensors to CPU NumPy arrays adds transfers and synchronization to the rollout loop.

Construct registered tasks with `swarmbots.make_env`; benchmark IDs are held in SwarmBots' own registry. The built-in evaluator consumes the tensor interface directly.

## Policy evaluation

A policy is any callable with this signature:

```python
def policy(observations, episode_starts):
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

Observation tensors can share environment buffers. Clone tensors you need to retain across calls to `step()` or `reset()`. On SAME_STEP autoreset, bootstrap a truncated episode from `info["final_obs"]`, selected by `info["_final_obs"]`, rather than from the returned reset observation. Terminated episodes should not bootstrap.

Every reset, including the first reset and `options={"reset_mask": mask}` partial resets, applies the scenario's physics settling. Repeating `reset(seed=...)` with the same seed and configuration reproduces the initial observations on the same device/backend. Reseeding also clears prefetched reset snapshots. Partial resets preserve the unselected lanes' physical state.

The old `settle_initial_reset` constructor argument and `force_settled` reset option are no longer needed. Remove them from callers. To disable settling for a custom variant, set `scenario_kwargs={"reset_settle_time": 0.0}`; this affects both explicit resets and autoresets and is not a canonical benchmark configuration.

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
