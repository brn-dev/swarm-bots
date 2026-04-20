# Agent Notes

Agents shall use this file to make notes for future instances. Write down important concepts, code architectures, etc. so future agents will have an easier time navigating the code base. KEEP THIS!

## TL;DR Architecture
- Main layers:
- `swarmbots/mj_env`: MuJoCo env, scenarios, swarm generation.
- `swarmbots/mjw_env`: MJWarp batched GPU env path for wall training. Current scope is intentionally narrower than `mj_env`: obstacle-street wall scenario only, preconnected swarm pool only, quantized twists only, minimal contacts only, capsules only.
- `swarmbots/learn`: RL/training stack (PPO/MAT, action dists, wrappers, rollout/samplers, checkpoints, logging).
- Canonical training reference is `scripts/run_mat_nop_wall.py`.
- Other `scripts/run_mat_*.py` files can be intentionally stale; do not assume they match the current MAT API.
- `recording/record.py` is also stale relative to the current MAT setup.

## Training Flow
- Script builds `SwarmBotsEnv` constructors and vectorizes (`AsyncVectorEnv` or `WorkerPoolAsyncVectorEnv`).
- Typical wrapper chain:
- `SwarmBotsLearnEnvWrapper` (tensor/device boundary)
- `TorchRecordEpisodeStatisticsWrapper`
- `TorchProgressGuidanceEpisodeStatsWrapper`
- `TorchFeatureWiseObsNormWrapper` (for local/global/hidden obs groups)
- `TorchTransitionObsWrapper`
- `TorchNormalizeRewardWrapper` (skip when using PopArt)
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
- `RMATPolicy`: `MATPolicy` decoder/critic with `RMATEncoder` (per-layer agent-axis transformer + time-axis sequence model). Rollout-time temporal state lives inside the policy. Under the current `SAME_STEP` pipeline, reset masks are queued after a done step and consumed on the next episode's first observation. Step-rollout bootstrap value passes still snapshot/restore RMAT temporal state so the live rollout state is not advanced twice on bootstrap observations.
- World-model composition is wrapper-first (not separate PPO algo classes):
- `NextObsPredWrapper(BasePPOPolicy[PPOWMSamples, PPOWMSamplerConfig], NextObsPredMixin)`
- `SPRWrapper(BasePPOPolicy[PPOWMSamples, PPOWMSamplerConfig], SPRMixin)`
- Wrappers delegate action/value to wrapped `MATPolicy` and add WM losses in `evaluate_actions(...)`.
- Important recurrent gotcha: WM wrappers must delegate `make_sampler(...)` to the wrapped policy. If a wrapper hardcodes `PPOWMSampler`, RMAT silently falls back to flat samples and crashes/misbehaves.
- Important recurrent gotcha: WM wrappers must also delegate temporal-state hooks (`reset_temporal_state`, snapshot/restore, and `after_optimizer_step`) to the wrapped policy. Otherwise wrapped RMAT loses SAME_STEP episode-boundary resets and step-rollout bootstrap snapshot/restore.
- `NextObsPredMixin.compute_next_obs_pred_loss(...)` now flattens recurrent RMAT batches `(B, S, ...) -> (B*S, ...)`; recurrent NOP uses `RPPOWMSamples.wm_actions`, not the PPO current-step `actions`.
- `SPRWrapper` intentionally rejects `RMATPolicy`; recurrent SPR target latents would need history-aware target encoding, which is not implemented.
- WM sampler runtime checks should use `BaseWMSampler`, not concrete `PPOWMSampler`; recurrent RMAT uses `RPPOWMSampler`, which is a different class but still a valid WM sampler.
- Shared recurrent WM flattening now lives in `swarmbots/learn/algos/world_modeling/wm_recurrent_batch.py`; use that for both NOP and SPR instead of duplicating `(B, S, ...) -> (B*S, ...)` logic.
- `BasePPOPolicy._policy_actions(...)` normalizes action batch shape from `(B,N,A)` or `(B,T,N,A)` to `(B,N,A)` for policy eval.

- Rollout/sampler structure:
- `PPORolloutBuffer` builds `PPOEpisode` objects and computes GAE.
- PPO rollout is now `SAME_STEP`: every env step is stored immediately, done environments are finalized on that same step, and truncated/terminated bootstrap observations come from `infos["final_obs"]`, not the reset observation batch.
- Raw Gymnasium same-step env infos for done steps arrive under `infos["final_info"]`; rollout code must unwrap episode stats from there when they were not injected by later wrappers.
- Same-step `infos["final_obs"]` must be transformed through the learn-side observation wrappers too. `TorchFeatureWiseObsNormWrapper` has to normalize `final_obs` without updating RMS twice, and `TorchTransitionObsWrapper` has to stack transition features onto single-env `final_obs` entries before PPO bootstrap uses them.
- `collect_steps()` can emit partial `PPOEpisode`s that start mid true env episode. `PPOEpisode.is_true_episode_start` is explicit rollout bookkeeping for this; do not infer it from chunk index or `initial_previous_actions`.
- `PPOSampler` flattens episodes into `PPOSamples`.
- `PPOWMSampler` extends `PPOSampler` with multi-step windows and returns `PPOWMSamples` (next obs, validity masks, WM masks).
- Shared WM target construction now lives in `swarmbots/learn/algos/world_modeling/wm_sampler_helper.py`; use it for both flat and recurrent WM samplers so next-obs windows, shifted masks, and padding stay identical.
- WM samples expose PPO current-step `actions` separately from multi-step `wm_actions`; wrappers must use `wm_actions` for WM losses, never guess from `actions`.
- `RPPOWMSampler` in `swarmbots/learn/algos/r_mat/r_ppo_wm_sampler.py` chunks `PPOEpisode` segments into fixed-length right-padded sequences with `time_mask`. Its `is_true_episode_start` flag only means the chunk begins at a true env episode boundary, not merely the start of a partial rollout segment.
- `RPPOWMSampler` supports burn-in via `burn_in_length`; it emits overlapping windows plus `time_loss_mask` so burn-in steps update recurrent state but do not contribute to PPO loss.
- PPO loss/reduction code understands recurrent `time_loss_mask` (falling back to `time_mask`), so padded or burn-in `(B,T,...)` slices are ignored for policy loss, critic loss, metrics, and extra-loss reduction. PPO `value_loss_fn` must expose `reduction="none"` so masking can happen centrally.
- Recurrent WM wrappers must combine `wm_target_time_mask` with `time_loss_mask`, otherwise burn-in roots still train WM losses.

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
- Vector env autoreset must be `SAME_STEP` end-to-end.
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
- MuJoCo scenario reward plumbing is now stripped down: `swarmbots/mj_env/scenarios/base_scenario.py` only keeps `progress_reward_weight`, `guidance_reward_weight`, and `units_without_connections_reward_weight`. The old hinge-qvel, double-connection, actuator-activation, movement/height, and connector reward knobs were removed there.
- `swarmbots/mjw_env/mjw_swarm_bots_vector_env.py` is a `gymnasium.vector.VectorEnv`, not a single-env API. It keeps one MJWarp model plus batched `Data` on GPU and exposes `action_backend="torch"` so the learn wrapper sends GPU torch tensors directly.
- MJW `make_data(...)` workspace defaults in `swarmbots/mjw_env/mjw_swarm_bots_vector_env.py` should stay lean and scale with swarm size, not raw `nv`/huge blanket multipliers. The env API should only expose per-world `nconmax`; current default sizing is `nconmax = max(24, total_connectors + 2*num_units)` and `njmax = max(160, 5*nconmax + 2*num_units)`. For the wall setup, oversized `nconmax`/`njmax` noticeably slow MJW throughput.
- MJW reset behavior now has three paths in `swarmbots/mjw_env/mjw_swarm_bots_vector_env.py`: the default direct in-place reset path, an optional one-shot synchronous `settle_initial_reset` path for the very first full reset, and the CPU-settled reset prefetch path when `scenario.reset_settle_time > 0` for predicted truncations. The hot-path optimization is only for predicted truncations: before a step that will truncate next, the env samples the exact reset spec on the main torch RNG, starts a background CPU MuJoCo settle job, and then installs the settled full-state snapshot if it is ready by the SAME_STEP autoreset point. Unexpected done cases and not-ready jobs fall back to the direct reset path.
- MJW reset behavior now also has an optional one-shot `settle_initial_reset` path. When enabled and `reset()` is called for all worlds before training starts, the env synchronously settles all sampled resets on CPU and installs those settled snapshots as the initial state. This increases startup latency substantially, but gives settled initial episodes.
- Scenario-specific MJW logic now lives behind a runtime layer in `swarmbots/mjw_env/scenarios/base_mjw_scenario.py`. The vector env only owns batched physics, connector matching/disconnects, common local-obs assembly, and SAME_STEP orchestration. Obstacle-street reset sampling/application, hidden state, reward bookkeeping, and CPU settle snapshots moved to `swarmbots/mjw_env/scenarios/mjw_obstacle_street_runtime.py`. If you add another MJW scenario, implement `create_runtime(...)` on the scenario and keep scenario state out of `mjw_swarm_bots_vector_env.py`.
- Hot-path MJW connector matching now lives in `swarmbots/mjw_env/mjw_kernels.py`: Warp kernels handle per-connector candidate search and reset pose writes, while disconnect application stays batched torch indexing on GPU. The connector matching policy is mutual-nearest activation among newly activated connectors, not the old Python greedy scan from `mj_env`.
- MJW Warp kernels should use dedicated `int32` metadata/index tensors (`connector_body_indices`, `connector_unit_idx`, `unit_qpos_adr`, transient reset world/pool ids). Keep the torch-side state/index tensors as `int64` only where PyTorch advanced indexing actually needs it. Mixing `int64` throughout the Warp kernels causes codegen friction and worse GPU code.
- MJW connector matching precomputes per-step connector frames (`position`, `x/y/z` axes) into `vec3` work buffers before running the candidate-search kernel. Re-reading `xmat` inside the candidate inner loop is much worse.
- The MJW env hot path now reuses GPU scratch buffers for `local_obs`, hidden-local threshold observations, and disconnect updates. Avoid reintroducing `torch.any(...)` Python branches or rebuilding `torch.arange(...)`/`torch.cat(...)` tensors every step in `mjw_swarm_bots_vector_env.py`.
- PPO sampler minibatch indices should be created on the same device as the stored sample tensors. CPU `torch.randperm(...)` plus CUDA tensor indexing technically works, but it keeps minibatch selection host-driven and adds avoidable sync overhead.
- PPO rollout bootstrap extraction now accepts `infos["final_obs"]` as a dict of batched tensors in addition to Gym's legacy object-array style. That change was made for the MJW env to avoid host-side per-env observation packing on done steps.
- The torch-side wrapper chain now preserves tensor-native SAME_STEP infos for the MJW path: `final_obs`, `_final_obs`, `_episode`, and episode stats stay as torch tensors through `TorchEnvWrapper` / `TorchTransitionObsWrapper` / episode-stat wrappers. Legacy Gym object-array `final_obs` is still supported for the old `mj_env` vector env path.
- Recording reward overlays should use `infos["reward_terms"]` / `scenario_state["reward_terms"]` for scenario-specific reward breakdowns instead of hardcoding labels in generic recorder/env code. Keep legacy flat keys like `progress_reward` / `guidance_reward` only for existing wrappers and logs.
- `swarmbots/learn/tensor_conversion.py` no longer has explicit Warp-array support. The learn-side boundary expects torch / NumPy / generic DLPack-capable objects; the MJW env already exposes torch tensors, so passing raw Warp arrays into the learn wrappers is no longer a supported path.
- `HomogeneousSwarm(..., quantize_connection_twist=N)` prebuilds `N` weld equalities per connector pair with evenly spaced twists; `BaseScenario` then activates the nearest prebuilt constraint by toggling `data.eq_active` only. With `None`, legacy behavior remains and the single weld constraint's twist is still written into `model.eq_data` at activation time.
- `ObstacleStreetScenario` wall-pass reward is normalized by active unit count and threshold count; adding thresholds should not inflate total wall reward.
- MJW obstacle-street wall-pass bookkeeping must keep `next_threshold_for_unit` monotonic after reset. Do not overwrite it with the current crossed-threshold count each step, or units can backtrack across a threshold and farm wall reward repeatedly.
- `FeatureWiseObsNormWrapper` is copy-on-write for the configured obs key; it must not mutate incoming observation arrays.
- The canonical learn-side wrappers are now torch-side (`Torch*Wrapper` classes). `SwarmBotsLearnEnvWrapper` is the only generic conversion boundary; wrappers above it must stay tensor-native so device-native env outputs do not get forced through NumPy first.
- `SwarmBotsLearnEnvWrapper` now also owns the action conversion boundary. It reads `env.action_backend` and converts policy actions to that backend. Torch interop for non-NumPy backends uses DLPack-style zero-copy paths where available so device actions do not bounce through host NumPy.
- `BaseLearnEnvWrapper` handles NumPy, torch, and objects exposing `__dlpack__`.

## Swarm Notes
- `HomogeneousSwarm` supports preset layouts, explicit coordinates, Poisson-disc/pre-connected/random-wiggle generation.
- Inactive units are controlled by `num_unit_probs`; this propagates through `agent_mask`.
- Base scenario logic keeps inactive units physically out of active area.
- MuJoCo `HomogeneousSwarm` now exposes `segment_1_ratio`, `minimal_contacts`, and `use_cylinders`; in `swarmbots/mj_env/swarm/unit.py`, `minimal_contacts=True` disables collisions on the short first limb segments and connector tips and adds same-unit excludes between long segments / main body, while `use_cylinders=False` switches limb and connector geoms to capsules.

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
- `scripts/run_mat_nop_wall_mjw.py` is the MJWarp wall-training entrypoint. It uses one batched `MJWSwarmBotsVectorEnv` directly on CUDA, supports live recording through the `record` command on the training env, and now also supports per-env first-episode staggering via `first_episode_lengths`, matching the old CPU script's startup spreading behavior.
- MJW wall config now exposes physics cadence cleanly on the scenario itself: `MJWObstacleStreetScenario` owns both `timestep` and `action_repeat`, and `MJWSwarmBotsVectorEnv` reads `scenario.action_repeat` instead of owning a separate cadence default. Use that path for cadence experiments instead of monkeypatching `build_model()`.
- Default MJW cadence now just lives inside `swarmbots/mjw_env/scenarios/mjw_scenario_presets.py::DEFAULT_KWARGS` alongside the other scenario defaults (`"timestep": 0.002`, `"action_repeat": 15`). Keep benchmark/training entrypoints reading from that instead of scattering hardcoded `15`s or passing `action_repeat` directly to the vector env.
- MJW live recording now exists, but it is intentionally not action-replay into `mj_env`. `MJWSwarmBotsVectorEnv.start_video_recording(...)` records exact live MJW episodes by copying a small selected subset of world states (`qpos/qvel/eq_active/mocap/time`) into CPU MuJoCo render slots and writing MP4s asynchronously. Use this for `record` on MJW runs; do not try to reproduce episodes by replaying actions in plain MuJoCo.
- Reward text overlay for recorded videos is shared between the legacy `swarmbots/learn/recording.py` path and MJW live recording via `swarmbots/recording_overlay.py`; keep styling/placement changes there instead of duplicating them.
- MJW live recording default camera should stay scenario-owned, not recorder-owned. The generic recorder asks the scenario for `get_default_recording_camera_config()`. Do not bake obstacle-street framing back into `mjw_live_recording.py`, and do not base the wall camera on `model.stat.extent`; the huge ground plane makes the swarm invisible.
- CPU wall-training scripts now construct `WorkerPoolAsyncVectorEnv(..., copy=False)`. That avoids a parent-process `deepcopy(self.observations)` on every vector `step_wait/reset_wait`, which was throttling `mj_env` throughput on one core.
- `WorkerPoolAsyncVectorEnv` now defaults to `check_spaces=False` to reduce constructor latency with many envs/workers. With this fast path, worker-space validation is skipped at startup; set `check_spaces=True` if you need strict early mismatch detection.
- `scripts/benchmarks/benchmark_mj_env_vs_mjw_env.py` benchmarks raw wall-env throughput for `mj_env` vs `mjw_env` without PPO overhead. It times create/reset/step throughput, synchronizes CUDA correctly for MJW, and keeps running when one backend/startup case fails.
- `scripts/benchmarks/benchmark_mjw_step_overhead.py` benchmarks MJW per-step overhead by comparing full `env.step(...)` throughput against a physics-only path (`_run_physics` + actuator writes, without connector/reward/obs/reset logic).
- `scripts/benchmarks/benchmark_mjw_default_workspace_caps.py` benchmarks MJW full-step throughput for the current lean workspace-cap defaults versus the previous oversized defaults (`nconmax=max(128, total_connectors*8)`, `njmax=max(512, nv*8 + nconmax*6)`), and reports the resolved caps for each variant.
- `scripts/benchmarks/benchmark_mjw_physics_cadence_sweep.py` benchmarks MJW throughput over explicit `(timestep, action_repeat)` pairs and reports unstable terminations plus resolved workspace caps. Use that for constant-ish env-step-duration cadence sweeps such as `0.002x15`, `0.003x10`, `0.004x8`, `0.005x6`.
- `scripts/benchmarks/benchmark_mjw_settled_resets.py` benchmarks slowdown from non-initial MJW settled resets by comparing `reset_settle_time=0` vs `>0` at short episode lengths, with `settle_initial_reset=False` so startup settle cost is intentionally excluded.
- On the current Windows workstation, `mj_env` with `WorkerPoolAsyncVectorEnv(num_workers=23)` failed to start at `2048` envs because worker processes hit MuJoCo memory allocation failures during env construction. `mjw_env` at `2048` envs did run.
- `scripts/run_mat_nop_wall.py` currently assumes `cwd == scripts/` for relative paths like `../runs/...`; launcher wrappers should add repo root to `PYTHONPATH` instead of switching cwd to repo root.
- On Windows, `WorkerPoolAsyncVectorEnv` workers spawned from `scripts/run_mat_nop_wall.py` re-import that script as their main module. Keep top-level imports in that script limited to env-construction dependencies only; moving PPO/policy/torch-heavy imports inside `main()` materially reduces worker startup latency.
- `SwarmBotsEnv` now reuses `scenario.dummy_model` / `scenario.dummy_data` as the live env model/data instead of recompiling the same MuJoCo scene immediately in `__init__`.
- If you change wrappers/vector-env behavior, re-check `SAME_STEP` `final_obs` handling and checkpoint restore.
- Checkpoint env-state restore now has aliases for old NumPy/Gym normalization wrapper names (`FeatureWiseObsNormWrapper`, `NormalizeReward`) to the new torch-side normalization wrappers, so old checkpoints can still restore running stats into the new wrapper chain.
- If you change observation composition, verify `build_obs_indices(...)`, normalization wrappers, and WM target configs together.
