# Benchmark protocol 0.1

Use this protocol to compare policies on the registered SwarmBots tasks. It standardizes task settings, evaluation episodes, permitted actor inputs, and reporting. Training algorithms and budgets are chosen by the experimenter and reported separately.

## Core suite

The eight-task core suite is exposed as `swarmbots.CORE_BENCHMARK_IDS`:

| Benchmark ID | Challenge |
| --- | --- |
| `SwarmBots-WallMedium-v0` | Fixed 0.3 m wall traversal |
| `SwarmBots-POWallMedium-v0` | Randomized hidden 0.3 m wall traversal |
| `SwarmBots-Bridge-v0` | Narrow movable bridge traversal |
| `SwarmBots-FindOpening-v0` | Hidden-opening exploration |
| `SwarmBots-Climb-v0` | Platform climbing |
| `SwarmBots-VerticalReach-v0` | Elevated goal reaching |
| `SwarmBots-PayloadStep-v0` | Payload transport over a step |
| `SwarmBots-MultiPayloadGoal-v0` | Multiple payloads delivered to assigned goals |

The full 14-task suite is exposed as `swarmbots.ALL_BENCHMARK_IDS`. See the [scenario catalog](scenarios.md) for all tasks and success conditions. Report per-task values. Reward scales differ between tasks, so an unnormalized average return across the suite is not meaningful.

## Evaluation

For each task and checkpoint:

1. Use the unmodified registered scenario and its 500-step episode limit.
2. Evaluate seeds `1000`, `1001`, `1002`, `1003`, and `1004`.
3. Run 256 parallel worlds and accept exactly the first episode from each world for every seed.
4. Use deterministic policy actions unless the experiment specifically studies stochastic evaluation.
5. Report mean episode return for every task and success rate where the registry declares one.
6. Report the mean and population standard deviation across the five seed-level means, plus all raw per-episode values.

This yields 1,280 evaluation episodes per task. Keeping one episode per lane avoids changing the sampled evaluation population when faster-terminating policies autoreset earlier.

All episodes start after the scenario's default physics settling, including the first episode following an explicit reset. Settling does not consume the 500 control steps or contribute to episode return. This corrects the pre-release behavior where explicit resets started unsettled while autoresets settled; results from that earlier behavior must be rerun for protocol 0.1.

The evaluator's defaults implement one 256-world seed at a time:

```python
from swarmbots import evaluate_policy

result = evaluate_policy(
    policy, benchmark_id, seed=1000, num_envs=256, num_episodes=256,
    action_mode="deterministic",
)
```

The evaluator accepts at most one episode per lane and returns episodes in lane order. It rejects `num_episodes > num_envs`. Smaller runs are useful for development but are not canonical protocol runs. Completed lanes still simulate while remaining lanes finish; their later rewards, successes, and episodes are ignored.

`action_mode` records how the caller configured the policy. It does not force deterministic actions or call `policy.eval()`. Put neural-network policies in evaluation mode and freeze observation normalization yourself.

See the [five-seed reporting example](api.md#five-seed-report) for JSON export and aggregation. A single evaluator call covers one seed; completing the protocol requires all five.

## Morphology population

The registered defaults share a fixed pool of 50 generated morphologies, with pool seeds `42000` through `42049`. Evaluation seeds `1000` through `1004` control sampling from that pool and task randomization; they do **not** generate unseen morphologies. Protocol 0.1 measures performance on this population and makes no held-out-morphology generalization claim. It also does not prescribe a training budget or number of training seeds; report both separately. Variation across evaluation seeds is not variation across independently trained policies.

For a generalization study, construct a separate morphology pool through `unit_start_locations`, keep its pool seeds disjoint from training, and report that configuration as a custom variant. Do not mix those scores with the canonical suite.

## Policy information boundary

An evaluated actor may consume only `local_obs`, `global_obs`, and `agent_mask`. It must not consume `hidden_local_vars`, `hidden_global_vars`, simulator state, future reset state, or reward decomposition terms. Recurrent state must be reset wherever `episode_starts` is true.

Centralized, partially centralized, and decentralized actors are all allowed. An actor may combine the permitted observations from any agents within the same world, including through attention or communication. The benchmark does not require decentralized execution; the information boundary applies to every actor architecture. Report which observations each agent's action can depend on and any communication restrictions used by the policy.

Critics may use privileged variables during training. Those variables must remain unavailable to the evaluated actor, regardless of how centralized it is.

## Reproducibility checklist

Publish or record:

- package version and Git commit;
- benchmark ID and protocol version;
- all scenario or episode-length overrides (canonical results should have none);
- policy checkpoint, model architecture, access to other agents' observations, communication restrictions, and deterministic/stochastic action mode;
- PyTorch, MuJoCo, MuJoCo Warp, Warp, CUDA, driver, and GPU versions;
- raw `EvaluationResult.to_dict()` outputs for all five seeds.

Changing active-unit distributions, morphology pools, connector semantics, observations, rewards, termination conditions, or physics parameters creates a different benchmark variant.

`EvaluationResult.to_dict()` includes `metadata` containing runtime package versions, Python/platform, device and GPU name, CUDA runtime, the resolved environment settings, requested overrides, actual world count, episode limit, action mode, and caller-supplied policy metadata and source revision. Supply `policy_metadata` with your checkpoint identifier/hash and architecture, and `source_revision` with the benchmark Git commit. Record the NVIDIA driver separately; it is not collected automatically.

`metadata.canonical_scenario` is false when scenario/environment overrides are supplied or the episode limit changes. This deliberately also flags performance-only overrides for review. `metadata.canonical_evaluation` additionally requires 256 worlds, a protocol seed, and declared deterministic actions. These flags describe one seed-level run, not completion of the five-seed protocol. For custom runs, the top-level `benchmark_id` identifies the base task only; report the overrides with it.
