# Agent Notes

Agents shall use this file to make notes for future instances. Write down important concepts, code architectures, etc. so future agents will have an easier time navigating the code base. KEEP THIS!

## TL;DR Architecture
- This repo has two main layers:
- `swarmbots/mj_env`: MuJoCo environment/scenario/swarm generation.
- `swarmbots/learn`: RL infra (PPO/MAT, action distributions, vector env wrappers, rollout buffer, logging, checkpointing).
- Main training entrypoints are in `scripts/` (`run_mat_*`).

## What Actually Runs (Training Flow)
- Script builds a `SwarmBotsEnv` factory (`make_env_fn`), then vectorizes it (`AsyncVectorEnv` or `WorkerPoolAsyncVectorEnv`), then wraps it.
- Typical wrapper order (important):
- `RecordEpisodeStatistics`
- `ProgressGuidanceEpisodeStatsWrapper`
- `FeatureWiseObsNormWrapper` (local/global/hidden_vars)
- `TransitionObsWrapper`
- `NormalizeReward` (optional; often disabled when PopArt is enabled)
- `SwarmBotsLearnEnvWrapper` (numpy->torch and action tensor->dict bridge)
- PPO loop:
- rollout collection (`collect_steps` or `collect_whole_episodes`) -> `PPORolloutBuffer`
- PPO update epochs
- optional world-model loss in `PPOWM`

## Hard Invariants You Should Not Break
- Vector env autoreset mode must be `NEXT_STEP` end-to-end.
- `BaseLearnEnvWrapper` enforces this and rollout accumulator logic depends on it.
- Observation dict keys required by learn side: `local_obs`, `global_obs`, `hidden_vars` (optional `agent_mask`).
- MAT policy currently expects `agent_mask` to be a contiguous true-prefix (`True...True, False...False`) and first agent always active.
- `TransitionObsWrapper` assumes dict obs and action layout compatible with swarm bots (it concatenates prev obs/actions with current obs).
- `FeatureWiseObsNormWrapper` normalizes only configured scalar indices and canonicalizes quaternion sign for configured quaternion starts.

## Environment / Scenario Notes
- `SwarmBotsEnv` delegates almost everything to scenario classes (`BaseScenario` + concrete scenarios).
- `SwarmBotsEnv` can shuffle agents per episode (`shuffle_agents=True`) by shuffling obs and unshuffling actions; if changing this, preserve inverse permutation logic.
- Unstable MuJoCo simulation is converted into terminal transition with zero-like obs + `simulation_unstable_reward` and info `error=simulation_unstable`.
- Reward weighting update path exists at runtime (`update_reward_weights`) with key validation in `BaseScenario`.
- Scenario presets (`default_wall`, `default_bridge`) are the canonical constructors used by training scripts.

## Swarm Generation Notes
- `HomogeneousSwarm` supports multiple placement modes:
- explicit coordinates / preset strings (e.g. `8:hourglass`)
- `PoissonDiscUnitLocationsConfig`
- `PreConnectedUnitLocationsConfig`
- `RandomWiggleUnitLocationsConfig`
- Inactive units are supported when `num_unit_probs` is provided (mask propagated via `agent_mask`).
- BaseScenario enforces inactive unit physics state and relocates inactive bodies to an off-area.

## World Model / Observation Indices
- `build_obs_indices(...)` (`swarmbots_obs_indices.py`) is the central place that maps obs layout to scalar/angle/rot6d/binary feature groups.
- World-model losses (`NextObsPredMixin`) depend on those indices; when obs layout changes, update this mapping first or training will fail/mis-train.
- Transition model in NOP/SPR policies uses these indices and optional multi-step windows from `ppo_wm_sampler.py`.

## Checkpoint / Resume Gotchas
- Checkpoints save:
- policy state
- optimizer state (optional)
- env normalization state (wrapper RMS stats)
- counters (`n_total_*`) and EMA fields
- `apply_env_state(...)` is intentionally strict: missing expected wrapper state raises, to avoid silently wrong normalization.
- Wrapper order/class/`obs_key` changes can break loading old checkpoints.

## Runtime Command Interface (During `learn()`)
- Interactive commands are supported (if `prompt_toolkit` + TTY): `set_lr`, `set_reward_weights`, `save`, `record`, `pause`, `stop`, plus PPO-specific commands (`set_clip_range`, `set_target_kl`, auto-lr toggles, etc.).
- Multiple commands can be chained with `;`.
- Command-triggered hyperparameter updates are logged to `command_log.jsonl` in run dir.

## Logging / Analysis
- Metrics are logged to `log.csv` (semicolon delimiter). Schema can expand during run (`MetricsLogger` rewrites header safely).
- `plot_logs/` has non-interactive and interactive plotting + metadata diff tools.
- `run_metadata_*.json` is written with de-dup logic; volatile fields (timesteps/script/load_path) are ignored for equality checks.

## Known Stale / Broken Files
- `recording/record.py` appears stale vs current APIs:
- imports `GSDEParams` from `hybrid_action_dist` (does not exist now)
- constructs `MATNOPPolicy` with legacy kwargs instead of config dataclasses
- `recording/record_random_obstacle_street.py` uses `unit_start_locations="random:6"` which is not a valid preset in `HomogeneousSwarm` current preset map.
- Treat `scripts/run_mat_*.py` as the up-to-date reference, not `recording/record.py`.

## Version / Dependency Note
- `AGENTS.md` says Python >= 3.11, but `pyproject.toml` currently declares `requires-python = ">=3.13"`.
- If environment/setup issues appear, check this mismatch first.

## Practical Guidance For Future Agents
- If changing obs composition in scenario/base env:
- update `build_obs_indices`
- verify FeatureWise normalization index groups
- verify TransitionObsWrapper output dims match policy expectations
- If changing wrappers/order:
- confirm `autoreset_mode` remains `NEXT_STEP`
- verify checkpoint env_state restore still matches wrappers
- For new training scripts, clone from latest `scripts/run_mat_nop_wall.py` pattern; it is the richest/most current setup (worker pool, auto-lr, world-model, detailed logging).

## Action Dist Gotcha
- In `make_proba_distribution(...)` (`swarmbots/learn/action_dists/hybrid_action_dist.py`), check subclass configs before base configs.
- `StickyBangZeroBangConfig` subclasses `BangZeroBangConfig`; if `BangZeroBangConfig` is checked first, sticky config will incorrectly instantiate `BangZeroBangActionDist` and stickiness is silently disabled.
- PPO rollout boundary nuance (`NEXT_STEP`): transitions at `is_final=True` are skipped from buffer. In `ppo_rollout.py`, `next_previous_actions` is masked with `is_final` so skipped boundary actions are not carried into the next stored transition's previous-action context.
- PPO sampler/rollout nuance for sticky action dists: `PPOEpisode` now stores `initial_previous_actions` (actions entering the first stored step of that episode chunk). `PPOSampler` prepends this value instead of hardcoded zeros when building `previous_actions`.
- `PPOEpisodeAccumulator.add(...)` now receives `previous_actions` from rollout and captures it when `step == 0` for an env. This fixes step-based rollouts where a chunk can start mid true environment episode.
- Recording path nuance: `record_policy(...)` (`swarmbots/learn/recording.py`) must pass and update `previous_actions` when `policy.requires_previous_actions()` is true (e.g., `StickyBangZeroBangActionDist` with `stickiness > 0`), and mask to zeros on done envs. Missing this causes `ValueError: previous_actions is required when stickiness > 0`.
- Recording init gotcha: do not use `env.n_agent_actions` in `record_policy(...)`; wrapped env chains (e.g. via `TransitionObsWrapper`) may not expose it. Use `env.action_space.total_agent_action_dim`.
- `ActionDist` has a base `get_hyper_parameters()` that auto-serializes simple runtime attrs.
- `HybridActionDistribution` and concrete dists now override `get_hyper_parameters()` for explicit runtime HP snapshots.
- `ActionDist.get_hyper_parameters()` is an abstract API; use `_base_hyper_parameters()` inside concrete dists for shared core fields.

## Scheduler Notes
- Generic scheduling infra now lives in `swarmbots/schedulers.py`:
- `ScheduledHyperParameter` = `{name, scheduler, get_value, apply, state}`.
- `SchedulerManager.step(...)` calls each scheduler with counters/metrics and applies requested updates.
- PPO integration: `PPO`/`PPOWM` accept `scheduler_manager`; schedulers run once per training iteration in `train()` and emit metrics:
- `scheduler_<name>_value`, `scheduler_<name>_event`, `scheduler_<name>_updated`.
- Callable serialization helper lives in `swarmbots/learn/serialization_utils.py` (`serialize_fn`), and is shared by PPO automatic-LR metadata + scheduler metadata serialization.
- Sticky action dist updates:
- `StickyBangZeroBangActionDist` now has `set_stickiness(...)` (updates cached log values too).
- `HybridActionDistribution` exposes `set_sub_stickiness(...)` and `set_all_stickiness(...)`, and keeps `continuous_configs` synced.
- Policy API:
- `BasePPOPolicy.set_action_stickiness(value, sub_dist_idx=None)` is the common setter used by schedulers.
- `get_hyper_parameters()` now reads runtime values from attributes/action_dist directly (no config-sync helper).

## Config / Hyperparameter Refactor
- Config dataclasses are now treated as initialization inputs only; runtime values are owned by module attributes.
- `self.hyper_parameters` caches were removed from `swarmbots/learn` policies/world-model modules.
- `get_hyper_parameters()` now computes from live attributes at call-time (especially action-dist/runtime loss weights), so command/scheduler changes are reflected without mutating config dicts.
- World-model policies (`MATNOPPolicy`, `MATSPRPolicy`) now override `get_hyper_parameters()` and include runtime world-model weights there.
- All `serialize_*_config` helpers were removed; hyperparameter dicts are now composed directly from attributes.
- `swarmbots/learn/config_serialization.py` was removed (no longer used).
- Dataclass hyperparameter serialization now lives in `swarmbots/learn/serialization_utils.py` (`serialize_dataclass`, `serialize_value`) and is reused by PPO/MAPPO/MAT policies plus hybrid action-dist config serialization.
- World-model hyperparameter snapshots now live with the world-model components:
- `TransformerTransitionModel.get_hyper_parameters()` serializes via `to_config()`.
- `NextObsPredMixin.get_next_obs_pred_hyper_parameters(...)` and `SPRMixin.get_spr_hyper_parameters(...)` collect runtime world-model settings, but policy-specific architecture metadata is passed in by the policy; do not store that metadata on the mixins.
