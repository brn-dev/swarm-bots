# Scenario catalog

All registered scenarios use five padded agent slots by default, while reset pools may activate four or five units. Inactive slots are identified by `agent_mask`. Scenario constructors accept keyword overrides through `make_scenario(..., **scenario_kwargs)` or `make_env(..., scenario_kwargs={...})`.

| Benchmark ID | Category | Terminal success metric |
| --- | --- | --- |
| `SwarmBots-WallEasy-v0` | fixed 0.2 m wall | yes |
| `SwarmBots-WallMedium-v0` | fixed 0.3 m wall | yes |
| `SwarmBots-WallHard-v0` | fixed 0.4 m wall | yes |
| `SwarmBots-POWallEasy-v0` | randomized hidden 0.25 m wall | yes |
| `SwarmBots-POWallMedium-v0` | randomized hidden 0.3 m wall | yes |
| `SwarmBots-Bridge-v0` | narrow bridge traversal | yes |
| `SwarmBots-FindOpening-v0` | hidden-opening exploration | yes |
| `SwarmBots-Climb-v0` | platform climbing | yes |
| `SwarmBots-VerticalReach-v0` | elevated goal reaching | yes |
| `SwarmBots-PayloadPlane-v0` | single-payload transport | no; report return |
| `SwarmBots-PayloadStep-v0` | payload transport over a step | yes |
| `SwarmBots-DualPayloadPlane-v0` | two-payload transport | no; report return |
| `SwarmBots-MultiPayloadGoal-v0` | variable payload-to-goal assignment | yes |
| `SwarmBots-MoveTo-v0` | sampled-goal navigation | no; report return |

## Observation contract

`local_obs` contains proprioception, connection state, and optional connector positions for every agent. `global_obs` contains shared task features that the actor is allowed to observe. `agent_mask` distinguishes active units from padding.

`hidden_local_vars` and `hidden_global_vars` are explicitly privileged. They support centralized critics, auxiliary losses, and diagnostics, but using them as actor inputs invalidates a benchmark result. The official evaluator removes both keys before invoking a policy.

## Action contract

`actuators` are continuous values in `[-1, 1]`. `connectors` are continuous by default: positive values accumulate connection intent and negative values accumulate disconnection intent. Scenarios can still be constructed with `continuous_connector_actions=False` for ablations, but those results are not directly comparable to the registered defaults.

## Scenario customization

The registry fixes defaults, not the implementation. Custom variants remain easy to construct:

```python
from swarmbots import make_env

env = make_env(
    "SwarmBots-POWallMedium-v0",
    num_envs=128,
    device="cuda",
    scenario_kwargs={
        "wall_height": 0.35,
        "continuous_connector_actions": False,
    },
)
```

Treat any override as a new task. Report every override and do not label the result with the unchanged canonical benchmark ID.
