# Agent Notes

Agents shall use this file to make notes for future instances. Write down important concepts, code architectures, etc. so future agents will have an easier time navigating the code base. KEEP THIS!

## TL;DR Architecture
- Main layers:
- `swarmbots/mj_env`: MuJoCo env, scenarios, swarm generation.
- `swarmbots/learn`: RL/training stack (PPO/MAT, action dists, wrappers, rollout/samplers, checkpoints, logging).
- Canonical training reference is `scripts/run_mat_nop_wall.py`.
- Other `scripts/run_mat_*.py` files can be intentionally stale; do not assume they match the current MAT API.
- `recording/record.py` is also stale relative to the current MAT setup.

## Training Flow
- Script builds `SwarmBotsEnv` constructors and vectorizes (`AsyncVectorEnv` or `WorkerPoolAsyncVectorEnv`).
- Typical wrapper chain:
- `RecordEpisodeStatistics`
- `ProgressGuidanceEpisodeStatsWrapper`
- `FeatureWiseObsNormWrapper` (for local/global/hidden obs groups)
- `TransitionObsWrapper`
- `NormalizeReward` (skip when using PopArt)
- `SwarmBotsLearnEnvWrapper`
- `PPO.perform_iteration()` collects rollouts (`collect_whole_episodes` or `collect_steps`) and then trains.

## Core Class Structure (Current)
- Algorithm hierarchy:
- `BaseAlgorithm` -> `PPO`
- `PPO.train()` always uses `sampler = policy.make_sampler(episodes)`.
- `PPO.compute_loss()` always calls `policy.evaluate_actions(batch=...)`.

- Policy hierarchy:
- `BasePolicy` -> `BasePPOPolicy[Samples, SamplerConfig]`
- `BasePPOPolicy` defines the PPO-facing interface:
- `forward(...)`
- `_evaluate_actions(batch, ...)` / `evaluate_actions(batch, ...)`
- `make_sampler(episodes, config)`
- `after_optimizer_step()` hook (default no-op)
- Concrete policies:
- `PPOPolicy`: MLP actor + MLP/PopArt critic.
- `MATPolicy`: encoder/decoder transformer policy + DeepSet critic.
- `RMATPolicy`: `MATPolicy` decoder/critic with `RMATEncoder` (per-layer agent-axis transformer + time-axis sequence model). Rollout-time temporal state lives inside the policy. Under `NEXT_STEP`, terminal observations must be encoded with the pre-reset state; reset only after that forward pass so the next episode's first observation sees the reset. Step-rollout bootstrap value passes must snapshot/restore RMAT temporal state so the live rollout state is not advanced twice on the same observation.
- World-model composition is wrapper-first (not separate PPO algo classes):
- `NextObsPredWrapper(BasePPOPolicy[PPOWMSamples, PPOWMSamplerConfig], NextObsPredMixin)`
- `SPRWrapper(BasePPOPolicy[PPOWMSamples, PPOWMSamplerConfig], SPRMixin)`
- Wrappers delegate action/value to wrapped `MATPolicy` and add WM losses in `evaluate_actions(...)`.
- `BasePPOPolicy._policy_actions(...)` normalizes action batch shape from `(B,N,A)` or `(B,T,N,A)` to `(B,N,A)` for policy eval.

- Rollout/sampler structure:
- `PPORolloutBuffer` builds `PPOEpisode` objects and computes GAE.
- `PPOSampler` flattens episodes into `PPOSamples`.
- `PPOWMSampler` extends `PPOSampler` with multi-step windows and returns `PPOWMSamples` (next obs, validity masks, WM masks).
- Shared WM target construction now lives in `swarmbots/learn/algos/world_modeling/wm_sampler_helper.py`; use it for both flat and recurrent WM samplers so next-obs windows, shifted masks, and padding stay identical.
- `RPPOWMSampler` in `swarmbots/learn/algos/r_mat/r_ppo_wm_sampler.py` chunks `PPOEpisode` segments into fixed-length right-padded sequences with `time_mask`; unlike flat `PPOWMSamples`, it keeps current PPO `actions` separate from multi-step `wm_actions`.
- `RPPOWMSampler` now also supports burn-in via `burn_in_length`; it emits overlapping windows plus `loss_time_mask` so burn-in steps update recurrent state but do not contribute to PPO loss.
- PPO loss/reduction code now understands recurrent `loss_time_mask` (falling back to `time_mask`), so padded or burn-in `(B,T,...)` slices are ignored for policy loss, metrics, and extra-loss reduction.

- Action distribution structure:
- `HybridActionSpace` / `VectorHybridActionSpace` define sub-spaces and `total_agent_action_dim`.
- `HybridActionDistribution` is a container of per-subspace `ActionDist`s created by `make_proba_distribution(...)`.
- It handles split/concat, aggregate `log_prob`, prefixed metrics/losses (`act0_*`, `act1_*`, ...), entropy/stickiness runtime updates.
- Important factory gotcha: check subclass configs before base configs (`StickyBangZeroBangConfig` before `BangZeroBangConfig`).

## World-Model Integration
- `PPOWM` is gone; use base `PPO` with WM policy wrappers.
- WM configs live on wrappers:
- `NOPWorldModelConfig` for `NextObsPredWrapper`
- `SPRWorldModelConfig` for `SPRWrapper`
- `world_model_config` is required in both wrappers.
- `world_model_num_next_steps` belongs to WM wrapper state/config, not `MATPolicyConfig`.
- PPO runtime commands route to policy/wrapper state:
- `set_wm_loss_coef`
- `set_wm_num_next_steps`
- `set_wm_target_tau`
- SPR target encoder updates run through `PPO._after_optimizer_step()` -> `policy.after_optimizer_step()`.
- Current wall training setup wraps `MATPolicy` with `NextObsPredWrapper`; it does not use a separate MAT-specific WM policy class anymore.

## Hard Invariants
- Vector env autoreset must be `NEXT_STEP` end-to-end.
- Learn-side obs must include `local_obs`, `global_obs`, `hidden_local_vars`, `hidden_global_vars`; `agent_mask` optional but supported.
- `MATPolicy` currently requires `agent_mask` as contiguous true-prefix, and agent 0 must be active.
- `RMATPolicy` currently trains with zero-initialized recurrent state plus sampler burn-in windows, not stored rollout hidden states. That is an approximation, but much better than zero-init with non-overlapping chunks. Strict TBPTT / exact rollout-state replay still is not implemented.
- If observation layout changes, update `build_obs_indices(...)` first (WM targets + normalization depend on it).
- Wrapper order/class changes can break env-state restore because checkpoint env-state matching is strict by wrapper class/order (and `obs_key` for feature-normalization wrappers).

## Environment / Scenario Notes
- `SwarmBotsEnv` delegates most behavior to scenario classes.
- Agent shuffling path must preserve shuffle/unshuffle pairing.
- Unstable MuJoCo simulation is converted to terminal transition with fallback obs/reward and `info["error"] = "simulation_unstable"`.
- Canonical scenario constructors in scripts are preset-based (`default_wall`, `default_bridge`).
- `ObstacleStreetScenario` wall-pass reward is normalized by active unit count and threshold count; adding thresholds should not inflate total wall reward.

## Swarm Notes
- `HomogeneousSwarm` supports preset layouts, explicit coordinates, Poisson-disc/pre-connected/random-wiggle generation.
- Inactive units are controlled by `num_unit_probs`; this propagates through `agent_mask`.
- Base scenario logic keeps inactive units physically out of active area.

## Runtime, Checkpoints, Logging
- Runtime hyperparameters are live attributes; mutating config dataclasses after init does nothing.
- `get_hyper_parameters()` should report current live values.
- Checkpoints include policy state, optional optimizer state, env wrapper normalization state, and training counters.
- Interactive commands in `learn()` support lr/loss/reward/save/record/pause/stop, and updates are persisted to `command_log.jsonl`.
- Generic schedulers are under `swarmbots/learn/scheduling/` and integrated via `SchedulerManager` in PPO.
- Training logs go to `log.csv` with `;` delimiter.
- Plot tooling is in `plot_logs/`.
- Current wall script logs per-joint continuous action stats via `metrics_action_splitters` and drives sticky-action annealing through `SchedulerManager`.

## Version Note
- `AGENTS.md` says Python `>=3.11`, but `pyproject.toml` currently requires `>=3.13`.

## Practical Guidance
- For new training work, start from `scripts/run_mat_nop_wall.py`, not the other MAT scripts.
- `scripts/run_mat_nop_wall.py` currently assumes `cwd == scripts/` for relative paths like `../runs/...`; launcher wrappers should add repo root to `PYTHONPATH` instead of switching cwd to repo root.
- If you change wrappers/vector-env behavior, re-check `NEXT_STEP` assumptions and checkpoint restore.
- If you change observation composition, verify `build_obs_indices(...)`, normalization wrappers, and WM target configs together.
