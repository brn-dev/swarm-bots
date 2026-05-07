# Agent Notes

Agents shall use this file to make notes for future instances. Write down important concepts, code architectures, etc. so future agents will have an easier time navigating the code base. KEEP THIS!

## TL;DR Architecture
- Main layers:
- `swarmbots/mj_env`: MuJoCo env, scenarios, swarm generation.
- `swarmbots/mjw_env`: MJWarp batched GPU env path. It still has a narrower scenario set than `mj_env`, but now covers obstacle-street wall, bridge, payload-plane, and move-to scenarios on the same batched env/runtime architecture; preconnected swarm pool only, quantized twists only, minimal contacts only, capsules only.
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
- `ActionDist` now exposes `compile_friendly`. `HybridActionDistribution` is compile-friendly only if all sub-dists are. Conservative current split: simple Bernoulli/BangZeroBang/DiagGaussian/Beta-mixture/Beta-mixture-style dists are marked compile-friendly; squashed/predicted-std/GSDE variants are not.
- `MATPolicyConfig` now has opt-in partial `torch.compile` support via `compile_modules` / `compile_mode`, but the exact compile surface depends on `policy.action_dist.compile_friendly`. It always compiles the encoder, query/context/memory token encoders, decoder `forward`, actor head, critic, and the latent/value training core. It only compiles the autoregressive rollout helper `MATPolicy._generate_actions_impl()` and the full training eval core `MATPolicy._evaluate_actions_core_impl()` when the action-dist stack opts in as compile-friendly. Otherwise action-dist update/log-prob/extra-loss work stays eager on purpose.
- MAT compile gotcha: normalize actor latents handed to the action-dist with `.contiguous()`. Decoder step slices can otherwise produce per-agent `latent_pi` tensors with different strides, which causes TorchDynamo recompiles inside `HybridActionDistribution.get_actions_with_log_probs(...)` during compiled rollout.
- MAT compile gotcha: avoid passing the Python `agent` index through compile-friendly action-dist paths unless sampling actually depends on it. Otherwise Dynamo can recompile `HybridActionDistribution.get_actions_with_log_probs(...)` once per agent value even when the current sub-dists ignore `agent`. `ActionDist.sampling_depends_on_agent` is the seam for this; currently GSDE is the important `True` case.
- MAT compile gotcha: keep Python autoregressive loop indices out of separately compiled token-helper frames. Select per-agent decoder embeddings in the rollout loop and pass the embedding tensor into `_encode_context_tokens(...)`; passing `start_agent_idx=i` makes Dynamo guard on each `i` value and can hit the recompile limit.
- PPO rollout bootstrap should use a value-only policy path, not `policy(..., deterministic=True)`. Reusing full forward for bootstrap values drags compiled action-dist code into an otherwise critic-only path and can trigger needless Dynamo recompiles on boolean control flow like `deterministic`.
- Sticky action-dist compile gotcha: mutable scheduler-controlled scalars like `stickiness` must live in module buffers/tensors, not Python floats. If compiled hot paths branch or do math on a Python `self.stickiness`, TorchDynamo guards on the exact value and recompiles every scheduler update.
- Sticky action-dist compile gotcha: keep `requires_previous_actions()` stable for sticky distributions even when annealing stickiness down to `0.0`. Letting it flip from `True` to `False` changes MAT sampler / compiled eval inputs from tensor to `None` mid-run and causes avoidable Dynamo recompiles or failures.
- `NextObsPredWrapper` now has its own optional compile surface via `NOPWorldModelConfig.compile_modules` / `compile_mode`. The useful seam is "compile almost all of `compute_next_obs_pred_loss(...)` but leave Python metric extraction eager": wrapper-owned WM modules plus `NextObsPredMixin._compute_next_obs_pred_loss_impl(...)` are compiled, while `.item()` logging stays outside the compiled graph.
- Canonical wall training entrypoints (`scripts/run_mat_nop_wall.py` and `scripts/run_mat_nop_wall_mjw.py`) now explicitly enable MAT policy compilation with `compile_modules=True` and `compile_mode="default"`. On unsupported Windows setups that can still fail inside `torch.compile`; the current base MAT compile guard only checks `torch.compile` availability, `cl.exe`, and a non-empty mode.
- MAT-family policies should swap encoders via `MATPolicy._build_encoder_config()` / `MATPolicy._build_encoder()`, not by replacing `self.encoder` after `super().__init__()`. Optional compile runs at the end of base init and assumes the final encoder is already installed.
- MAT-family policies can also swap the action-distribution wrapper via `MATPolicy._build_action_dist(...)`. That is the clean seam for benchmarks or variants that need identical sub-dists but different top-level `compile_friendly` exposure.
- MAT critic context is `hidden_global_vars`, not public `global_obs`. Public global obs is already folded into encoder tokens. Envs like payload can have `global_obs_dim > 0` and `hidden_global_vars_dim == 0`; the DeepSet critic must not enable `context_in_elements` in that case.
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
- PPO step-rollout accumulator capacity should be sized to rollout-segment length, not true env episode length. `collect_steps()` resets the buffer each rollout, so the live per-env segment bound is `ceil(n_steps_per_rollout / n_envs)` (capped by true `max_episode_length`). Whole-episode rollout still needs full episode capacity.
- `PPOSampler` flattens episodes into `PPOSamples`.
- `PPOWMSampler` extends `PPOSampler` with multi-step windows and returns `PPOWMSamples` (next obs, validity masks, WM masks).
- Flat `PPOWMSampler` now groups equal-length episodes and builds WM windows with one batched `unfold` per length bucket, then scatters back into the original flattened sample order. If you touch WM-window construction, keep that ordering invariant aligned with `PPOSampler`'s episode-by-episode concatenation.
- `PPOWMSamplerConfig` now also has optional helper-compile flags: `compile_wm_window_helper` and `wm_window_helper_compile_mode`. They compile only the pure tensor helper in `wm_sampler_helper.py`, not the sampler constructor. `RPPOWMSamplerConfig` inherits the same flags. On this Windows machine, CPU Inductor can fail for that path due to missing `omp.h`; the intended target is CUDA.
- `wm_sampler_helper.ensure_wm_window_helper_compile_available(...)` is now cached per compile mode, so repeated sampler creation with the same mode does not keep re-running `shutil.which("cl.exe")`.
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
- Obstacle-street progress reward has a category multiplier plus components: `progress_reward_weight` scales the whole progress category, while `forward_reward_weight` only scales the forward-distance component. Current CPU/MJW wall presets both use the newer wall defaults `forward_reward_weight=1.0`, `forward_reward_max_y=1.5`, and `wall_pass_reward_weight=10.0`.
- Exposed per-step reward info (`progress_reward`, `forward_reward`, `wall_pass_reward`, `guidance_reward`, and `reward_terms`) should contain weighted values only. Keep raw component values internal to scenario/runtime state.
- `BaseScenario` no longer owns a generic `compute_progress(...)` helper. It only requires `compute_progress_reward(...)`, and each scenario owns both its reset-time progress baseline and its per-step reward delta logic.
- Payload now lives in its own dedicated plane scenario on both backends: `swarmbots/mj_env/scenarios/payload_plane_scenario.py::PayloadPlaneScenario` and `swarmbots/mjw_env/scenarios/mjw_payload_plane_scenario.py::MJWPayloadPlaneScenario`. Those scenarios expose payload world position as `global_obs` `(x, y, z)` and use payload-y forward progress plus a per-step centerline penalty on `|x|^power`. `ObstacleStreetScenario` still has no payload and still keeps `global_obs` empty.
- Move-to now lives in its own dedicated plane scenario on both backends: `swarmbots/mj_env/scenarios/move_to_scenario.py::MoveToScenario` and `swarmbots/mjw_env/scenarios/mjw_move_to_scenario.py::MJWMoveToScenario`. It exposes the absolute goal `(x, y)` as `global_obs` and uses mean active-unit progress toward the goal based on `-max(distance_to_goal - goal_radius, 0)`, so movement inside the radius gives no further forward reward.
- Shared MJ/MJW scenario preset kwargs live in `swarmbots/scenario_presets_kwargs.py`; keep one complete kwargs dict per mirrored scenario (`*_SCENARIO_KWARGS`) plus `COMMON_SCENARIO_KWARGS`, instead of splitting reward kwargs from geometry/obs defaults or duplicating defaults in `scenario_presets.py` / `mjw_scenario_presets.py`. MJW-only kwargs like reward-kernel compilation still belong in `mjw_scenario_presets.py`.
- Payload centerline reward now uses a dead-zone tolerance: penalty is based on `max(abs(payload_x) - payload_centering_tolerance, 0) ** payload_centering_penalty_power`. CPU/MJW defaults use `payload_centering_penalty_weight=0.05` and `payload_centering_tolerance=0.25` so the dense x penalty does not swamp forward progress.
- Payload plane scenarios support `payload_shape` values `"sphere"`, `"box"`, and `"capsule"`. The payload presets default to `"box"` so touching a round object does not trivially roll it forward; `"capsule"` is supported but remains rounded and can still roll.
- Bridge now has an MJW scenario/runtime pair: `swarmbots/mjw_env/scenarios/mjw_bridge_scenario.py::MJWBridgeScenario` and `swarmbots/mjw_env/scenarios/mjw_bridge_runtime.py::BridgeMJWScenarioRuntime`. It exposes sampled `bridge_x` as `hidden_global_vars`, uses mean active-unit y progress, and reports a scenario termination when any active unit falls below `fall_z_threshold`.
- Move-to scenario goal sampling uses `swarmbots.move_to_goal_config`: `AbsoluteGoalConfig(x, y)` for fixed / absolute targets or `RelativePolarGoalConfig(distance, angle)` for offsets from the sampled swarm start. The CPU/MJW default presets use `RelativePolarGoalConfig(distance=UniformDistParams(2.0, 4.0), angle=UniformDistParams(0, 2*pi))` so the goal is per-episode and never directly on the spawn.
- CPU/MJW MoveTo scenarios support `visualize_goal=True` to add a non-colliding visible goal cylinder. The MAT/NOP MoveTo launchers enable it so recordings show the target.
- MJW scenario-specific model handles should go through `scenario.build_runtime_metadata(host_model=...)` and then into `create_runtime(..., runtime_metadata=...)`. Keep `MJWModelMetadata` for env/swarm-generic indices only; do not put one-off scenario bodies like payload indices there.
- MJW inactive-unit parking should be scenario-owned via `scenario.inactive_area_location`, not reconstructed from wall-specific fields like `street_width` inside `mjw_swarm_bots_vector_env.py`.
- `swarmbots/mjw_env/mjw_swarm_bots_vector_env.py` is a `gymnasium.vector.VectorEnv`, not a single-env API. It keeps one MJWarp model plus batched `Data` on GPU and exposes `action_backend="torch"` so the learn wrapper sends GPU torch tensors directly.
- MJW scenarios can now return scenario-owned terminations through `MJWStepResult.terminations`; `MJWSwarmBotsVectorEnv.step()` ORs those with unstable simulation terminations and suppresses truncation for worlds that terminated on the same step.
- MJW `make_data(...)` workspace defaults in `swarmbots/mjw_env/mjw_swarm_bots_vector_env.py` should stay lean and scale with swarm size, not raw `nv`/huge blanket multipliers. The env API should only expose per-world `nconmax`; current default sizing is `nconmax = max(24, total_connectors + 2*num_units)` and `njmax = max(160, 5*nconmax + 2*num_units)`. For the wall setup, oversized `nconmax`/`njmax` noticeably slow MJW throughput.
- MJW reset behavior now has three paths in `swarmbots/mjw_env/mjw_swarm_bots_vector_env.py`: the default direct in-place reset path, an optional one-shot synchronous `settle_initial_reset` path for the very first full reset, and the CPU-settled reset prefetch path when `scenario.reset_settle_time > 0` for predicted truncations. The hot-path optimization is only for predicted truncations: before a step that will truncate next, the env samples the exact reset spec on the main torch RNG, starts a background CPU MuJoCo settle job, and then installs the settled full-state snapshot if it is ready by the SAME_STEP autoreset point. Unexpected done cases and not-ready jobs fall back to the direct reset path.
- MJW reset behavior now also has an optional one-shot `settle_initial_reset` path. When enabled and `reset()` is called for all worlds before training starts, the env synchronously settles all sampled resets on CPU and installs those settled snapshots as the initial state. This increases startup latency substantially, but gives settled initial episodes.
- Scenario-specific MJW logic now lives behind a runtime layer in `swarmbots/mjw_env/scenarios/base_mjw_scenario.py`. The vector env only owns batched physics, connector matching/disconnects, common local-obs assembly, and SAME_STEP orchestration. Obstacle-street reset sampling/application, hidden state, reward bookkeeping, and CPU settle snapshots moved to `swarmbots/mjw_env/scenarios/mjw_obstacle_street_runtime.py`. If you add another MJW scenario, implement `create_runtime(...)` on the scenario and keep scenario state out of `mjw_swarm_bots_vector_env.py`.
- Hot-path MJW connector matching now lives in `swarmbots/mjw_env/mjw_kernels.py`: Warp kernels handle per-connector candidate search and reset pose writes, while disconnect application stays batched torch indexing on GPU. The connector matching policy is mutual-nearest activation among newly activated connectors, not the old Python greedy scan from `mj_env`.
- MJW Warp kernels should use dedicated `int32` metadata/index tensors (`connector_body_indices`, `connector_unit_idx`, `unit_qpos_adr`, transient reset world/pool ids). Keep the torch-side state/index tensors as `int64` only where PyTorch advanced indexing actually needs it. Mixing `int64` throughout the Warp kernels causes codegen friction and worse GPU code.
- MJW connector matching precomputes per-step connector frames (`position`, `x/y/z` axes) into `vec3` work buffers before running the candidate-search kernel. Re-reading `xmat` inside the candidate inner loop is much worse.
- The MJW env hot path now reuses GPU scratch buffers for `local_obs`, hidden-local threshold observations, and disconnect updates. Avoid reintroducing `torch.any(...)` Python branches or rebuilding `torch.arange(...)`/`torch.cat(...)` tensors every step in `mjw_swarm_bots_vector_env.py`.
- MJW obstacle-street reward math now has an opt-in `torch.compile` path on the scenario (`compile_reward_kernel`, `reward_kernel_compile_mode`), but only the pure tensor reward kernel is compiled; env orchestration/state mutation stays eager Python. On Windows, fail early if `cl.exe` is not on `PATH` because `torch.compile` will otherwise die later inside Inductor.
- MJW obstacle-street reward-kernel compile is now default-on only when the local environment actually supports it. `swarmbots/mjw_env/scenarios/mjw_scenario_presets.py::should_compile_reward_kernel_by_default()` gates it on `torch.compile`, a working Triton install, and `cl.exe` on Windows. Do not flip `compile_reward_kernel=True` blindly in shared defaults unless you want MJW startup to hard-fail on unsupported machines.
- PPO sampler minibatch indices should be created on the same device as the stored sample tensors. CPU `torch.randperm(...)` plus CUDA tensor indexing technically works, but it keeps minibatch selection host-driven and adds avoidable sync overhead.
- PPO rollout bootstrap extraction now accepts `infos["final_obs"]` as a dict of batched tensors in addition to Gym's legacy object-array style. That change was made for the MJW env to avoid host-side per-env observation packing on done steps.
- The torch-side wrapper chain now preserves tensor-native SAME_STEP infos for the MJW path: `final_obs`, `_final_obs`, `_episode`, and episode stats stay as torch tensors through `TorchEnvWrapper` / `TorchTransitionObsWrapper` / episode-stat wrappers. Legacy Gym object-array `final_obs` is still supported for the old `mj_env` vector env path.
- Recording reward overlays should use `infos["reward_terms"]` / `scenario_state["reward_terms"]` for scenario-specific reward breakdowns instead of hardcoding labels in generic recorder/env code. Keep legacy flat keys like `progress_reward` / `guidance_reward` only for existing wrappers and logs.
- `swarmbots/learn/tensor_conversion.py` no longer has explicit Warp-array support. The learn-side boundary expects torch / NumPy / generic DLPack-capable objects; the MJW env already exposes torch tensors, so passing raw Warp arrays into the learn wrappers is no longer a supported path.
- `HomogeneousSwarm(..., quantize_connection_twist=N)` prebuilds `N` weld equalities per connector pair with evenly spaced twists; `BaseScenario` then activates the nearest prebuilt constraint by toggling `data.eq_active` only. With `None`, legacy behavior remains and the single weld constraint's twist is still written into `model.eq_data` at activation time.
- Pre-connected swarm generation must sample directly from `connection_twist_values` when twist quantization is enabled. Sampling a continuous twist and only snapping at connection activation leaves the generated unit orientation inconsistent with the actual weld constraint.
- `ObstacleStreetScenario` wall-pass reward is normalized by active unit count and threshold count; adding thresholds should not inflate total wall reward.
- Obstacle-street now has optional `forward_reward_max_y` in both `mj_env` and `mjw_env`. It clamps per-unit forward-progress contribution before averaging, and it must only affect forward reward bookkeeping/baselines, not wall-pass threshold crossing logic.
- MJW obstacle-street wall-pass bookkeeping must keep `next_threshold_for_unit` monotonic after reset. Do not overwrite it with the current crossed-threshold count each step, or units can backtrack across a threshold and farm wall reward repeatedly.
- `FeatureWiseObsNormWrapper` is copy-on-write for the configured obs key; it must not mutate incoming observation arrays.
- The canonical learn-side wrappers are now torch-side (`Torch*Wrapper` classes). `SwarmBotsLearnEnvWrapper` is the only generic conversion boundary; wrappers above it must stay tensor-native so device-native env outputs do not get forced through NumPy first.
- `SwarmBotsLearnEnvWrapper` now also owns the action conversion boundary. It reads `env.action_backend` and converts policy actions to that backend. Torch interop for non-NumPy backends uses DLPack-style zero-copy paths where available so device actions do not bounce through host NumPy.
- `BaseLearnEnvWrapper` handles NumPy, torch, and objects exposing `__dlpack__`.

## Swarm Notes
- `HomogeneousSwarm` supports preset layouts, explicit coordinates, Poisson-disc/pre-connected/random-wiggle generation.
- Inactive units are controlled by `num_unit_probs`; this propagates through `agent_mask`.
- Base scenario logic keeps inactive units physically out of active area.
- Fixed preconnected swarm pools now support a runtime `active_pool_size` curriculum on both CPU `mj_env` and GPU `mjw_env`. The full pool must still be known up front; curriculum only restricts reset sampling to the first `active_pool_size` seeds/configurations and can be updated live without rebuilding models or MJW pool tensors.
- CPU/MJW scenarios expose optional `randomize_initial_swarm_z_rotation` (`bool`, default `False`). When enabled, each reset samples a full-circle yaw and applies it as a rigid rotation to active unit poses before settling. In MJW the sampled yaw lives in the common reset batch/spec as `initial_z_rotation`; keep direct GPU resets and CPU-settled reset snapshots applying the same value.
- MuJoCo `HomogeneousSwarm` now exposes `segment_1_ratio`, `minimal_contacts`, and `use_cylinders`; in `swarmbots/mj_env/swarm/unit.py`, `minimal_contacts=True` disables collisions on the short first limb segments and connector tips and adds same-unit excludes between long segments / main body, while `use_cylinders=False` switches limb and connector geoms to capsules.

## Runtime, Checkpoints, Logging
- Runtime hyperparameters are live attributes; mutating config dataclasses after init does nothing.
- `get_hyper_parameters()` should report current live values.
- Checkpoints include policy state, optional optimizer state, env wrapper normalization state, and training counters.
- Interactive commands in `learn()` support lr/loss/reward/save/record/pause/stop, and updates are persisted to `command_log.jsonl`.
- `BaseAlgorithm.learn(...)` now also supports `post_iteration_hooks`, called after each `perform_iteration()` with `(algorithm, metrics, rollout_steps)`. Use that seam for one-shot side effects tied to training progress; do not abuse `SchedulerManager` for non-numeric actions like recording.
- Separate `record` runs built from `make_record_env` should keep the record env on the compiled policy's active device. For MAT/NOP with `compile_modules=True`, forcing recording to CPU while the policy is compiled on CUDA can trigger a second CPU Inductor compile during `record_policy(...)` and fail on Windows.
- Generic schedulers are under `swarmbots/learn/scheduling/` and integrated via `SchedulerManager` in PPO.
- Training logs go to `log.csv` with `;` delimiter.
- `BaseAlgorithm.learn(..., compress_metrics_log_on_exit=True)` now optionally gzips `log.csv` to `log.csv.gz` on graceful run exit (normal max-step finish or `stop`), then deletes the plain CSV. Default stays off.
- Canonical wall training scripts and the 4096 env-sweep experiment send an optional Discord webhook notification when `ppo.learn(...)` exits or raises. Set `SWARMBOTS_DISCORD_WEBHOOK_URL`; implementation lives in `swarmbots/learn/discord_notifications.py` and uses only stdlib `urllib`.
- Plot tooling is in `plot_logs/`.
- Plot tooling now reads compressed logs in memory too: plain `.csv`, `.zip` (expects exactly one CSV inside or a uniquely preferred `log.csv`), plus `.gz/.bz2/.xz`. Do not extract archives to temp files just to inspect or plot logs.
- Generic grouped experiment result plots live in `plot_logs/experiment_results.py`. It expects `RUN_DIR/group/run/log.csv*`, labels only groups as `group (n=...)`, supports an optional group order, and writes high-DPI return/final-loss PNGs. Group curves use the union of run x-values and nan-aware reductions so the line continues after shorter runs finish with only the still-active runs contributing. The 4096 wall batch sweep wrapper is `experiments/mat_nop_wall_batch4096_env_sweep/plot_results.py`.
- Current wall script logs per-joint continuous action stats via `metrics_action_splitters` and drives sticky-action annealing through `SchedulerManager`.
- Gymnasium vector env info packing adds boolean `_key` masks for every info key, including nested dicts like `info["reward_terms"]`. Recording/overlay code that iterates nested reward-term dicts must ignore underscore-prefixed entries such as `_forward`, `_wall`, `_guidance`; those are presence masks, not rewards.

## Version Note
- `AGENTS.md` still says Python `>=3.11`, but `pyproject.toml` / `.python-version` now target Python `3.13`.
- Packaging split in `pyproject.toml`: base runtime deps are the core env/training stack; MJWarp lives under extras (`mjw` / `warp`), plotting under `plot`, and local tooling/tests under `[dependency-groups].dev`.
- `pyproject.toml` now uses `uv` package sources so `torch` resolves from PyTorch's CUDA 12.8 wheel index on Linux/Windows. That is `uv`-specific behavior; plain `pip install .` does not honor `[tool.uv.sources]`.
- `swarmbots.mj_env` / `swarmbots.mjw_env` now bootstrap MuJoCo rendering on headless Linux by defaulting `MUJOCO_GL=egl` when `MUJOCO_GL` is unset and neither `DISPLAY` nor `WAYLAND_DISPLAY` exists. Explicit env vars still win, so override with `MUJOCO_GL=osmesa` on servers without working EGL.

## Practical Guidance
- For new training work, start from `scripts/run_mat_nop_wall.py`, not the other MAT scripts.
- Payload and MoveTo MAT/NOP training now have first-class entrypoints too: `scripts/run_mat_nop_payload.py` / `scripts/run_mat_nop_move_to.py` for CPU MuJoCo and `scripts/run_mat_nop_payload_mjw.py` / `scripts/run_mat_nop_move_to_mjw.py` for MJWarp. They mirror the canonical wall launchers but use scenario-specific presets and run directories.
- `scripts/run_mat_nop_wall_mjw.py` is the MJWarp wall-training entrypoint. It uses one batched `MJWSwarmBotsVectorEnv` directly on CUDA, supports live recording through the `record` command on the training env, and now also supports per-env first-episode staggering via `first_episode_lengths`, matching the old CPU script's startup spreading behavior.
- `experiments/mat_nop_wall_batch4096_env_sweep/scripts/common.py::install_scheduled_recordings` takes an explicit `{percentage: episode_count}` schedule with percentage keys in `0..100`; keep the zero-percent episode count in that mapping, not as a hook special case.
- CPU MuJoCo cadence is now scenario-owned too: `swarmbots/mj_env/scenarios/base_scenario.py` owns `timestep` and `action_repeat`, presets expose them via `swarmbots/mj_env/scenarios/scenario_presets.py::DEFAULT_KWARGS`, `BaseScenario.build()` applies `model.opt.timestep`, and `SwarmBotsEnv` no longer accepts an `action_repeat` constructor arg.
- MJW wall config now exposes physics cadence cleanly on the scenario itself: `MJWObstacleStreetScenario` owns both `timestep` and `action_repeat`, and `MJWSwarmBotsVectorEnv` reads `scenario.action_repeat` instead of owning a separate cadence default. Use that path for cadence experiments instead of monkeypatching `build_model()`.
- Default MJW cadence now just lives inside `swarmbots/mjw_env/scenarios/mjw_scenario_presets.py::DEFAULT_KWARGS` alongside the other scenario defaults (`"timestep": 0.003`, `"action_repeat": 10`). Keep benchmark/training entrypoints reading from that instead of scattering hardcoded values or passing `action_repeat` directly to the vector env.
- CPU `swarmbots/mj_env/scenarios/scenario_presets.py` now mirrors the MJW wall/payload defaults where the backends share the same knobs: cadence `0.003x10`, wall reward defaults, payload reward defaults, `quantize_connection_twist=8`, and the fixed preconnected 4/5-unit pool seeded with `42_000..42_049`.
- MJW live recording now exists, but it is intentionally not action-replay into `mj_env`. `MJWSwarmBotsVectorEnv.start_video_recording(...)` records exact live MJW episodes by copying a small selected subset of world states (`qpos/qvel/eq_active/mocap/time`) into CPU MuJoCo render slots and writing MP4s asynchronously. Use this for `record` on MJW runs; do not try to reproduce episodes by replaying actions in plain MuJoCo.
- Reward text overlay for recorded videos is shared between the legacy `swarmbots/learn/recording.py` path and MJW live recording via `swarmbots/recording_overlay.py`; keep styling/placement changes there instead of duplicating them.
- MJW live recording default camera should stay scenario-owned, not recorder-owned. The generic recorder asks the scenario for `get_default_recording_camera_config()`. Do not bake obstacle-street framing back into `mjw_live_recording.py`, and do not base the wall camera on `model.stat.extent`; the huge ground plane makes the swarm invisible.
- MJW payload live recording uses a farther scenario camera than the original payload default: distance is based on `swarm.max_unit_extent * 10 + payload_radius * 4` with a `[6, 14]` clamp, and lookat tracks the payload y-offset. Keep it wide enough that small swarm/payload movement does not leave the frame.
- CPU wall-training scripts now construct `WorkerPoolAsyncVectorEnv(..., copy=False)`. That avoids a parent-process `deepcopy(self.observations)` on every vector `step_wait/reset_wait`, which was throttling `mj_env` throughput on one core.
- `WorkerPoolAsyncVectorEnv` now defaults to `check_spaces=False` to reduce constructor latency with many envs/workers. With this fast path, worker-space validation is skipped at startup; set `check_spaces=True` if you need strict early mismatch detection.
- `WorkerPoolAsyncVectorEnv` now also has `env_clone_group_keys`. Within one worker, envs with the same non-`None` key can reuse a worker-local prototype via `env.clone_for_worker_pool()` instead of rerunning the full constructor. `SwarmBotsEnv`/`BaseScenario` implement that fast path by deep-copying the compiled MuJoCo model/data instead of recompiling the scenario, which is much faster when one worker hosts many identical `mj_env` instances. Do not group envs whose constructors differ in stateful per-env attrs like `first_episode_length`; cloning copies the prototype env's value and breaks startup episode staggering.
- `SwarmBotsEnv.reset(seed=...)` now reseeds `scenario.rng` too. That matters because scenario reset sampling uses `scenario.rng`, not Gymnasium's `env.np_random`. If you touch reset seeding or env cloning, preserve that link or per-env reset seeds stop affecting placements/orientations.
- `scripts/benchmarks/benchmark_mj_env_vs_mjw_env.py` benchmarks raw wall-env throughput for `mj_env` vs `mjw_env` without PPO overhead. It times create/reset/step throughput, synchronizes CUDA correctly for MJW, and keeps running when one backend/startup case fails.
- On Windows, worker processes spawned from benchmark/training scripts re-import the script's top-level module. Keep `torch`, MJW, PPO, and similar heavy imports out of top level unless the worker env constructors really need them, or `mj_env` startup will pay for those imports in every worker.
- `scripts/benchmarks/benchmark_mjw_step_overhead.py` benchmarks MJW per-step overhead by comparing full `env.step(...)` throughput against a physics-only path (`_run_physics` + actuator writes, without connector/reward/obs/reset logic).
- `scripts/benchmarks/benchmark_mjw_step_overhead.py` no longer treats "physics only" as `actuators + _run_physics` on a live rollout, because that froze the connector topology and could make the supposed stripped-down path slower than full step. The connector-overhead part of the benchmark now branches from identical live GPU snapshots and compares `actuators_only + physics` vs `_apply_actions + physics` from the same pre-step state. That measures connector-step impact without the old trajectory/topology confound.
- `scripts/benchmarks/benchmark_mjw_default_workspace_caps.py` benchmarks MJW full-step throughput for the current lean workspace-cap defaults versus the previous oversized defaults (`nconmax=max(128, total_connectors*8)`, `njmax=max(512, nv*8 + nconmax*6)`), and reports the resolved caps for each variant.
- `scripts/benchmarks/benchmark_mjw_physics_cadence_sweep.py` benchmarks MJW throughput over explicit `(timestep, action_repeat)` pairs and reports unstable terminations plus resolved workspace caps. Use that for constant-ish env-step-duration cadence sweeps such as `0.002x15`, `0.003x10`, `0.004x8`, `0.005x6`.
- `scripts/benchmarks/benchmark_mjw_settled_resets.py` benchmarks slowdown from non-initial MJW settled resets by comparing `reset_settle_time=0` vs `>0` at short episode lengths, with `settle_initial_reset=False` so startup settle cost is intentionally excluded.
- `scripts/benchmarks/benchmark_mat_compile.py` benchmarks MAT rollout/training forward speed with `compile_modules` on/off and compares the same sticky-beta + Bernoulli action-dist stack in two exposures: normal compile-friendly vs a benchmark-only top-level wrapper that forces `compile_friendly=False` without changing the actual sub-dists.
- `scripts/benchmarks/benchmark_mat_action_cpu_vs_gpu.py` benchmarks `MATPolicy.act()` only, using synthetic wall-shaped observations and treating `num_envs` as the action batch size. Use it when you want CPU-vs-CUDA action-computation throughput without env stepping, rollout buffering, or critic/log-prob overhead.
- `scripts/benchmarks/benchmark_nop_compile.py` benchmarks `NextObsPredWrapper` compile speedups separately from MAT: it measures both the internal NOP loss core (`_compute_next_obs_pred_loss_fn`) and full `evaluate_actions()` under four cases (`policy_compile_modules` on/off crossed with `NOPWorldModelConfig.compile_modules` on/off). Use that before attributing full PPO training speed changes to NOP compile alone.
- `scripts/benchmarks/benchmark_ppo_wm_sampler.py` compares the old flat WM sampler's per-episode serial window construction against the current grouped batched implementation using a reference copy of the old constructor. It benchmarks fixed-length, bucketed-length, and ragged-length episode mixes, with and without agent masks.
- `scripts/benchmarks/benchmark_ppo_wm_sampler.py` should default to the sampler workload implied by `scripts/run_mat_nop_wall_mjw.py`, not the env's true episode length: `StepsRolloutMode(4096)` with `n_envs=1024` means flat WM sampler inputs are mostly rollout segments of length `4`, while the true env `episode_length` is `512`.
- Experiment entrypoints under `experiments/*/scripts/` should not import helpers from unrelated script files. If a script-local helper is needed there, either move it into a proper package module or keep it in the experiment's shared script. When run directly, these scripts should also add the repo root to `sys.path` for package imports and derive run/output paths from `__file__` instead of assuming a specific working directory.
- `scripts/benchmarks/benchmark_ppo_wm_sampler_helper_compile.py` now benchmarks the actual public helper path `build_wm_episode_windows_batch(..., compile_modules=False/True)` rather than manually compiling the private tensor kernel. Use that benchmark for realistic compile-vs-eager numbers; the older microkernel-only measurement was too optimistic.
- `scripts/benchmarks/benchmark_ppo_rollout_accumulator.py` compares the current preallocated `PPOEpisodeAccumulator` against a benchmark-local list-based accumulator that stores per-env step tensors and stacks on episode materialization. It reports constructor cost separately from the hot-path `add()` and full rollout-cycle costs; use the hot-path numbers, not constructor time, for training decisions because PPO reuses one accumulator across iterations.
- `scripts/benchmarks/benchmark_ppo_sampler_compile.py` benchmarks `torch.compile` on the flat `PPOSampler` concatenation block using a benchmark-local function that still takes `list[PPOEpisodeSegment]`. The useful cases are `fixed_mask_prev`, `staggered_mask_prev`, and especially `changing_staggered_mask_prev`, because compile wins on one frozen episode ordering are not trustworthy if reordered mixed-length episode lists trigger recompiles.
- On the current Windows workstation, `mj_env` with `WorkerPoolAsyncVectorEnv(num_workers=23)` failed to start at `2048` envs because worker processes hit MuJoCo memory allocation failures during env construction. `mjw_env` at `2048` envs did run.
- `scripts/run_mat_nop_wall.py` currently assumes `cwd == scripts/` for relative paths like `../runs/...`; launcher wrappers should add repo root to `PYTHONPATH` instead of switching cwd to repo root.
- On Windows, `WorkerPoolAsyncVectorEnv` workers spawned from `scripts/run_mat_nop_wall.py` re-import that script as their main module. Keep top-level imports in that script limited to env-construction dependencies only; moving PPO/policy/torch-heavy imports inside `main()` materially reduces worker startup latency.
- `SwarmBotsEnv` now reuses `scenario.dummy_model` / `scenario.dummy_data` as the live env model/data instead of recompiling the same MuJoCo scene immediately in `__init__`.
- If you change wrappers/vector-env behavior, re-check `SAME_STEP` `final_obs` handling and checkpoint restore.
- Checkpoint env-state restore now has aliases for old NumPy/Gym normalization wrapper names (`FeatureWiseObsNormWrapper`, `NormalizeReward`) to the new torch-side normalization wrappers, so old checkpoints can still restore running stats into the new wrapper chain.
- If you change observation composition, verify `build_obs_indices(...)`, normalization wrappers, and WM target configs together.
