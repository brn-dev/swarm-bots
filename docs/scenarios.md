# Scenario catalog

SwarmBots has 15 registered tasks covering coordinated locomotion, physical reconfiguration, partial observability, and payload transport. Eight tasks form the core evaluation suite. Every task uses the same articulated modular units and actuator/connector interface, so policies must control both the units' movement and their connections.

| Wall traversal | Finding an opening |
| --- | --- |
| ![Wall traversal rollout](assets/wall.gif) | ![Opening exploration rollout](assets/find-opening.gif) |

These demonstrations illustrate behavior, not canonical evaluation results.

## Scenario maturity and versioning

SwarmBots as a whole is an **alpha benchmark**. Each registered scenario also has a maturity label describing how thoroughly its task design has been tested:

**All fixed-wall and PO-wall variants (easy, medium, and hard), plus FindOpening, are beta; every other registered scenario is alpha.**

| Maturity | Meaning |
| --- | --- |
| **Alpha** | Limited task-level testing. Difficulty may be too high or too low, reward functions may encourage unintended behavior, and success conditions or other task details may need revision. |
| **Beta** | Extensive internal testing, but no feedback from other researchers yet. Task design may still have gaps and is not considered final. |
| **Stable** | Reserved for a future stage after broader validation and independent research feedback. No scenario currently has this status. |

These labels assess testing confidence, not difficulty, policy performance, or core-suite membership. They apply to the registered defaults; custom variants need their own validation. The overall benchmark remains alpha even when individual scenarios are beta. Report problems and suggestions through [GitHub issues](https://github.com/brn-dev/swarm-bots/issues) or by emailing Dominik Baron at [dominik.b4ron@gmail.com](mailto:dominik.b4ron@gmail.com); useful feedback includes the task ID, package version or Git commit, configuration, seed, and observed behavior.

Maturity and version identifiers serve different purposes:

- **Scenario maturity** (`alpha`, `beta`, or eventually `stable`) records the current testing assessment. It is available as `BenchmarkSpec.maturity`, in `swarmbots list` / `describe`, and as `metadata.benchmark_maturity` in evaluation reports.
- **Task version** (the `-v0` suffix in a benchmark ID) identifies the registered task definition. A maturity promotion alone does not change the ID. Changes to rewards, observations, success or termination conditions, morphology distributions, or physics that make scores incomparable should use a new task version.
- **Package version** (such as `0.1.0a2`) identifies the implementation release. Record it and the Git commit, especially for development checkouts; the maturity label and task ID alone do not identify an exact implementation.
- **Protocol version** (currently `0.1`) identifies the evaluation and reporting procedure, independently of task maturity.

## Registered tasks

| Benchmark ID | Task | Maturity | Core suite | Terminal success metric |
| --- | --- | --- | --- | --- |
| `SwarmBots-WallEasy-v0` | fixed 0.2 m wall | Beta | no | yes |
| `SwarmBots-WallMedium-v0` | fixed 0.3 m wall | Beta | yes | yes |
| `SwarmBots-WallHard-v0` | fixed 0.4 m wall | Beta | no | yes |
| `SwarmBots-POWallEasy-v0` | connected locomotion over a hidden 0.25 m wall | Beta | no | yes |
| `SwarmBots-POWallMedium-v0` | connected locomotion over a hidden 0.3 m wall | Beta | yes | yes |
| `SwarmBots-POWallHard-v0` | connected locomotion over a hidden 0.4 m wall | Beta | no | yes |
| `SwarmBots-Bridge-v0` | narrow movable bridge traversal | Alpha | yes | yes |
| `SwarmBots-FindOpening-v0` | exploration within an episode to find a hidden opening | Beta | yes | yes |
| `SwarmBots-Climb-v0` | platform climbing | Alpha | yes | yes |
| `SwarmBots-VerticalReach-v0` | elevated goal reaching | Alpha | yes | yes |
| `SwarmBots-PayloadPlane-v0` | single-payload transport | Alpha | no | no; report return |
| `SwarmBots-PayloadStep-v0` | payload transport over a step | Alpha | yes | yes |
| `SwarmBots-DualPayloadPlane-v0` | two-payload transport | Alpha | no | no; report return |
| `SwarmBots-MultiPayloadGoal-v0` | variable payload-to-goal assignment | Alpha | yes | yes |
| `SwarmBots-MoveTo-v0` | sampled-goal navigation | Alpha | no | no; report return |

Wall and PO-wall tasks primarily challenge connected locomotion through coordinated movement and physical connections during obstacle traversal. Fixed walls provide three heights: 0.2, 0.3, and 0.4 m. PO-wall adds difficulty through partial observability and randomized hidden geometry, with heights of 0.25, 0.3, and 0.4 m. The hard PO-wall shares the medium variant's observations, rewards, and randomization, with a higher wall.

FindOpening focuses on partial observability and exploration within an episode. The task combines discovery of a hidden passage using observations gathered during that episode with coordinated swarm traversal through it.

Bridge adds a narrow movable support. Climb requires all active units to reach a platform goal; VerticalReach succeeds when at least one unit reaches the elevated goal volume. Payload tasks extend coordination to external objects, including multiple payloads with individual goals.

Use `swarmbots.CORE_BENCHMARK_IDS` or `swarmbots.ALL_BENCHMARK_IDS` to select a suite, and follow the [benchmark protocol](benchmark_protocol.md) for comparable scores. Returns must be reported per task because reward scales differ.

## Robot and assembly

The default unit has four articulated limbs, eight hinges, and four connectors. Continuous actuator actions control its movement; connector actions request connections or disconnections. Connections carry physical loads, so changing the connection graph changes the swarm's mechanics during an episode.

All registered scenarios use five padded agent slots, while reset pools may activate four or five units. Inactive slots are identified by `agent_mask`. Initial assemblies are sampled from the shared [morphology population](benchmark_protocol.md#morphology-population); the canonical evaluation does not use unseen morphologies.

Scenario constructors accept keyword overrides through `make_scenario(..., **scenario_kwargs)` or `make_env(..., scenario_kwargs={...})`. See [scenario customization](#scenario-customization) for an example.

## Observation contract

`local_obs` contains proprioception, connection state, and optional connector positions for every agent. `global_obs` contains shared task features that the actor is allowed to observe. `agent_mask` distinguishes active units from padding.

`hidden_local_vars` and `hidden_global_vars` are explicitly privileged. They support centralized critics, auxiliary losses, and diagnostics, but using them as actor inputs invalidates a benchmark result. The official evaluator removes both keys before invoking a policy.

For the default four-limb, eight-hinge robot, each agent's `local_obs` has 71 float32 features, concatenated in this order (Python slice notation):

| Slice | Meaning |
| --- | --- |
| `0:3` | Root position `(x, y, z)` in world coordinates, metres |
| `3:9` | Root orientation: first two rotation-matrix columns, each in `(x, y, z)` order |
| `9:25` | Eight hinge angles as interleaved `(sin(angle), cos(angle))` pairs |
| `25:39` | MuJoCo generalized velocity order: root linear velocity (m/s), root angular velocity (rad/s), then hinge velocities (rad/s) |
| `39:59` | Four connector records: `(disconnected, connected, sin(twist), cos(twist), disconnect_potential)` |
| `59:71` | Four connector positions in world coordinates, metres, flattened `(x, y, z)` |

Twist sine/cosine are zero when no valid connection exists. Disconnect potential is an accumulated control quantity, not a physical measurement. Use `agent_mask` to exclude inactive slots; custom joint/observation settings change these offsets. Actions use the same hinge and connector ordering as their observation records.

Shared task features in `global_obs` are positions in world coordinates unless stated otherwise:

| Task | Shared features |
| --- | --- |
| Wall, PO Wall, Bridge | Empty; privileged obstacle information is not actor input |
| FindOpening | Empty; privileged opening information is not actor input |
| Climb, VerticalReach | Goal position `(x, y, z)` |
| MoveTo | Goal position `(x, y)` |
| PayloadPlane, PayloadStep | Payload position `(x, y, z)`, followed by its six-dimensional orientation |
| DualPayloadPlane | Two consecutive payload position/orientation records |
| MultiPayloadGoal | One 12-feature record per payload slot: active flag, position (3), orientation (6), goal `(x, y)` |

Inactive MultiPayloadGoal records have zero positions/goals and identity orientation. These payload masks are separate from `agent_mask`. Query `env.single_observation_space` and `env.single_action_space` for the selected task; the corresponding spaces without `single_` include the world dimension.

## Success and termination

Success is a team-level boolean. Padding never counts toward success. For success-enabled tasks, `info["success"]` refers to the episode that just stepped, including its terminal step before SAME_STEP reset.

| Task | Success condition |
| --- | --- |
| Wall and PO Wall | Every active unit has `y > wall_y + wall_success_threshold` |
| Bridge | Every active unit has `y > success_y`, with no active unit below `fall_z_threshold` |
| FindOpening | Every active unit has `y > wall_y + opening_y_margin` |
| Climb | Every active unit's root lies within `goal_radius` of the 3D goal |
| VerticalReach | At least one active unit's root lies inside the goal box |
| PayloadStep | Payload centre reaches `payload_success_y` and the required height above the step |
| MultiPayloadGoal | Every active payload centre lies within `goal_radius` of its assigned 3D goal (goal height is the payload radius) |

PayloadPlane, DualPayloadPlane, and MoveTo have no success metric; report return. Episodes truncate at 500 control steps by default. Success terminates an episode for the success-enabled tasks; bridge falls and nonfinite simulator state can also terminate unsuccessfully. The default control period is `0.003 * 10 = 0.03` seconds. Reward terms are exposed in `info["reward_terms"]`; weights and thresholds are in `env.get_settings()` and exported evaluation metadata.

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
env.close()
```

Treat any override as a new task. Report every override and do not label the result with the unchanged canonical benchmark ID.
