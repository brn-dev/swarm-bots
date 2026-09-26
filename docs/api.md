# Python API

## Registry

```python
from swarmbots import list_benchmarks, make_env, make_scenario

for spec in list_benchmarks():
    print(spec.id, spec.description)

scenario = make_scenario("SwarmBots-Climb-v0", seed=42)
env = make_env("SwarmBots-Climb-v0", num_envs=256, device="cuda", seed=42)
```

`make_env` forwards `scenario_kwargs` to the registered scenario factory and remaining keyword arguments to `MJWSwarmBotsVectorEnv`.

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
)
print(result.to_dict())
```

Only `local_obs`, `global_obs`, and `agent_mask` are passed to the policy.

## Deferred stepping

CUDA environments support `begin_step(actions)` followed by `end_step()`. This lets a caller overlap host work with asynchronous device execution. Ordinary callers should use `step(actions)`.

## Recording

Call `env.start_video_recording(...)` before stepping. Recording renders selected worlds with MuJoCo and writes MP4 files asynchronously. It is intended for qualitative inspection, not for high-throughput evaluation.
