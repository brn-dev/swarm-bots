# Agent Notes

Agents shall use this file to make notes for future instances. Write down important concepts, code architectures, etc. so future agents will have an easier time navigating the code base. KEEP THIS!

## TL;DR Architecture
- Two main layers:
- `swarmbots/mj_env`: MuJoCo environment, scenarios, swarm generation.
- `swarmbots/learn`: RL/training stack (PPO/MAT, action distributions, wrappers, rollout buffer, logging, checkpoints).
- Main training entrypoints are in `scripts/`, especially `run_mat_*`.

## Training Flow
- Training scripts build a `SwarmBotsEnv` factory, vectorize it (`AsyncVectorEnv` or `WorkerPoolAsyncVectorEnv`), then wrap it for learning.
- Typical wrapper chain:
- `RecordEpisodeStatistics`
- `ProgressGuidanceEpisodeStatsWrapper`
- `FeatureWiseObsNormWrapper`
- `TransitionObsWrapper`
- `NormalizeReward` (not used together with PopArt)
- `SwarmBotsLearnEnvWrapper`
- PPO then collects rollouts, updates the policy, and may also train a world model (`PPOWM`).

## Hard Invariants
- Vector env autoreset mode must stay `NEXT_STEP` end-to-end. Rollout logic depends on it.
- Learn-side observations are dict-based and must provide `local_obs`, `global_obs`, and optionally `hidden_vars` / `agent_mask`.
- MAT currently expects `agent_mask` to be a contiguous true-prefix with the first agent active.
- If observation layout changes, update `build_obs_indices(...)` first. World-model losses and normalization rely on it.
- Wrapper order/class changes can break checkpoint restore because env state loading is intentionally strict.

## Environment / Scenario Notes
- `SwarmBotsEnv` delegates most environment logic to scenario classes.
- Agent shuffling is supported; preserve the shuffle/unshuffle pairing if touching that path.
- Unstable MuJoCo simulation is converted into a terminal transition with fallback observations/reward and `info["error"] = "simulation_unstable"`.
- Scenario presets such as `default_wall` and `default_bridge` are the canonical constructors used by training scripts.

## Swarm / Agent Notes
- `HomogeneousSwarm` supports preset layouts, explicit coordinates, and generated layouts such as Poisson-disc / pre-connected / random-wiggle.
- Inactive units are supported via `num_unit_probs`; the mask is propagated through `agent_mask`.
- Base scenario logic keeps inactive units physically out of the active area.

## Checkpoints / Runtime Controls
- Checkpoints include policy state, optional optimizer state, wrapper normalization state, and training counters.
- Interactive runtime commands exist during `learn()` (for example learning-rate, reward-weight, save, record, pause/stop controls) and are logged to `command_log.jsonl`.

## Logging / Analysis
- Training metrics go to `log.csv` using `;` as delimiter.
- `plot_logs/` contains plotting and run-metadata comparison utilities.

## Known Gotchas
- Treat `scripts/run_mat_*.py` as the current reference. Some files under `recording/` are stale relative to current APIs.
- `make_proba_distribution(...)` in `swarmbots/learn/action_dists/hybrid_action_dist.py` must check subclass configs before base configs; `StickyBangZeroBangConfig` is a subclass of `BangZeroBangConfig`.
- Recording / rollout paths for sticky action distributions must propagate `previous_actions`; zeroing only on done envs is the important behavior.
- Use `env.action_space.total_agent_action_dim` instead of `env.n_agent_actions` in wrapped recording code.

## Runtime Hyperparameters / Scheduling
- Runtime hyperparameters are owned by live module attributes, not by mutating config dataclasses after init.
- `get_hyper_parameters()` should report live runtime values.
- Generic scheduler infrastructure lives in `swarmbots/schedulers.py` and is integrated into PPO/PPOWM.
- Shared serialization helpers live in `swarmbots/learn/serialization_utils.py`.

## Version Note
- `AGENTS.md` says Python `>=3.11`, but `pyproject.toml` currently declares `>=3.13`. Check this first if setup behaves oddly.

## Practical Guidance
- If you change observation composition: update `build_obs_indices(...)`, then verify normalization and transition-wrapper expectations.
- If you change wrappers or vector env behavior: re-check `NEXT_STEP` autoreset assumptions and checkpoint restore.
- For new training scripts, start from the latest `scripts/run_mat_nop_wall.py` pattern.
