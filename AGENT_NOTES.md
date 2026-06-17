# Agent Notes

Keep only durable architecture notes and gotchas. Prefer deleting stale detail over preserving trivia.

## Map

- `swarmbots/mj_env`: CPU MuJoCo envs, scenarios, and swarm generation.
- `swarmbots/mjw_env`: MJWarp batched GPU envs.
- `swarmbots/learn`: PPO/MAT, action dists, wrappers, rollout, checkpoints, logging.
- Main training entrypoint: `scripts/run_mat_qcs_nop_wall.py`; older `scripts/run_mat_*.py` and `recording/record.py` may be stale.

## Wrappers And Autoreset

- Env autoreset must stay `SAME_STEP`. Read `gymnasium_autoreset.md` before touching rollout/reset logic.
- Under `SAME_STEP`, done-step bootstrap obs comes from `infos["final_obs"]`, not the returned reset obs. Terminal stats may be in `infos["final_info"]`.
- Learn wrappers must transform `final_obs`; obs norm must not update RMS twice.
- Canonical learn wrapper order: `SwarmBotsLearnEnvWrapper -> TorchRecordEpisodeStatisticsWrapper -> TorchProgressGuidanceEpisodeStatsWrapper -> TorchFeatureWiseObsNormWrapper -> TorchTransitionObsWrapper -> TorchNormalizeRewardWrapper`, unless PopArt handles normalization.
- `SwarmBotsLearnEnvWrapper` is the conversion boundary and owns `env.action_backend` action conversion.
- Learn obs must contain `local_obs`, `global_obs`, `hidden_local_vars`, `hidden_global_vars`; `agent_mask` is optional.
- Observation layout changes must update `build_obs_indices(...)` and `_global_scalar_indices(...)`; WM targets and normalization depend on them.
- `TorchShuffleAgentsWrapper` owns agent-order randomization. Under `SAME_STEP`, shuffle `final_obs` with the old permutation, then resample done-env permutations for returned reset obs.
- `TorchTransitionObsWrapper` zeros previous-transition features on returned reset obs but preserves terminal transition features in `final_obs`.
- Env obs buffers are not durable. Async/shared-memory envs, GPU envs, and `copy=False` wrappers may mutate them after the next env call; snapshot anything retained.

## Rollout, PPO, MAT

- `collect_steps()` chunks may start mid-episode. Use `PPOEpisodeSegment.is_true_episode_start`; do not infer from chunk position.
- PPO `rollout_warmup_steps_per_env` performs a bufferless startup rollout, keeps only `PPORolloutState`, and does not increment timesteps/iterations or emit episode metrics.
- Rollout bootstrap should use a value-only path, not full deterministic action generation.
- `MATQCSPolicy` is the main transformer policy; `PPOPolicy` is the plain MLP policy; `MATDecPolicy` is decoderless despite the name.
- `MATQCCPolicy` keeps query/context streams separate and is not state-dict compatible with `MATQCSDecoder`'s interleaved `CONTEXT_TOKENS_ONLY` implementation.
- Recurrent MAT variants share `RMATPolicyMixin`: `RMATQCSPolicy`, `RMATQCCPolicy`, and decoderless `RMATDecPolicy` all use `RMATEncoder` plus env-major `RPPOWMSampler` TBPTT rows.
- `MATOrigPolicy` uses shifted previous-agent actions, requires contiguous true-prefix `agent_mask`, and must zero inactive-agent log-probs in rollout and `evaluate_actions()`.
- `MATQCSPolicy` supports arbitrary inactive positions if every row has at least one active agent. `assume_agent_mask_is_active_prefix=True` enables the cheap prefix path for QCS/QCC configs.
- For MAT customization, override `_build_encoder*()` / `_build_action_dist(...)`; do not mutate fields after `super().__init__()`.
- `ActionDist.compile_friendly` gates MAT compile coverage. Mutable action-dist scalars must be tensors/buffers, not Python floats.
- Sticky dists must keep `requires_previous_actions()` structurally stable even when annealed to zero.

## Recurrent And World Models

- Recurrent state is explicit and lives in `PPORolloutState`, not inside the policy. Policy temporal APIs take and return opaque tensor trees with a leading environment batch dimension.
- Recurrent step rollouts use env-major TBPTT rows. Episode segments carry env/time metadata plus their exact detached initial state; `RPPOWMSampler` regroups segments and applies `episode_start_mask` inside the sequence.
- Store one initial recurrent state per TBPTT row. Do not restore burn-in or per-step state storage. `sequence_length` must cover the fixed number of rollout steps per env.
- Under `SAME_STEP`, terminal bootstrap evaluates `final_obs` from the post-current-observation state without committing the resulting state. The returned reset obs consumes the done reset mask on the next rollout step.
- WM integration is policy-wrapper based: `NextObsPredWrapper` / `SPRWrapper`.
- WM wrappers must delegate sampler, explicit temporal-state APIs, and `after_optimizer_step()` to the wrapped policy. Flat WM samplers break RMAT.
- WM losses use `wm_actions`, not PPO current-step `actions`; recurrent padding uses `time_mask`.
- NOP global-observation prediction is optional. It is enabled by explicit global scalar/rot6d target indices and pools predicted agent latents across active agents before global heads.
- Shared recurrent WM flattening lives in `swarmbots/learn/algos/world_modeling/wm_recurrent_batch.py`.

## Scenarios And Swarms

- `SwarmBotsEnv` delegates most behavior to scenarios. Scenario cadence is scenario-owned; do not add env-level `action_repeat`.
- Scenarios own reset baselines and per-step progress deltas; no generic `BaseScenario.compute_progress(...)`.
- Render overlays are visual-only via `BaseScenario.add_render_geoms(scene)` after `Renderer.update_scene()`; do not use them for physics/model geometry.
- Key `global_obs` layouts: payload `(x, y, z, rot6d)`, dual-payload two payload poses, multi-payload goal records `(is_active, x, y, z, rot6d, goal_x, goal_y)`, move-to absolute goal `(x, y)`, climb top-face center goal `(x, y, z)`.
- Multi-payload goal inactive records keep `is_active=0` and zero scalar fields, but use identity rot6d so WM/global rot6d targets remain valid. `is_active` is not a global scalar target.
- Find-opening barrier segments use `wall_segment_width`, independent of `street_width`; their inner edges must remain anchored to the sampled opening in both MJ and MJW.
- Find-opening wall exploration reward cells are an internal episode latch: cells split `street_width` along x, cover the strip immediately before the wall front face, and are reset-latched from initial unit positions.
- Payload-step scenarios keep the single-payload global obs layout; the elevated step is fixed scenario geometry/settings, and the height reward uses a latch so crossing onto the step does not create a negative potential drop.
- `SwarmBotsEnv.reset(seed=...)` must reseed `scenario.rng`; reset sampling does not use Gymnasium `env.np_random`.

## MJW

- MJW is the primary version. CPU is only for completeness. Take much care in MJW!
- `MJWSwarmBotsVectorEnv` is a real `gymnasium.vector.VectorEnv` with one MJWarp model, batched GPU `Data`, and `action_backend="torch"`.
- Scenario-specific MJW logic belongs in scenario runtime classes, not `mjw_swarm_bots_vector_env.py`.
- MJW scenario metadata goes through `scenario.build_runtime_metadata(...)`; keep `MJWModelMetadata` generic.
- Scenario terminations flow through `MJWStepResult.terminations`.
- Reset paths: direct, one-shot `settle_initial_reset`, and a background CPU-settled snapshot buffer. MJW samples reset specs on the main thread and settles them in the executor; done-world resets consume ready snapshots and only block on buffer misses.
- Hot-path connector matching is kernelized in `swarmbots/mjw_env/mjw_kernels.py`; keep Warp indices `int32` unless PyTorch indexing forces `int64`.
- Reuse GPU scratch buffers. Avoid rebuilding tensors or Python branching in the MJW step path.
- Live MJW recording uses one shared `mujoco.Renderer`; `max_parallel_episodes` only caps concurrent episodes and should not multiply renderer VRAM.

## Runtime

- Runtime hyperparameters are live attributes. Mutating config dataclasses after init does nothing.
- Activation factories with `ParameterLearnMode.PER_FEATURE` need explicit feature counts; use `make_activation(...)` in generic `act_fn_cls` paths.
- hidden/projection/transformer-FF gains `1.0`, output/action/value/prediction heads `0.01`.
- Remote run logs are reachable vis WSL SSH at `brn@server2026`, `~/swarm-bots/runs`. Logs are huge so use scripts/code to summarize data and otherwise fetch sparsely. 

## Version And Install

- Target Python is `3.13` (`pyproject.toml` / `.python-version`), even if older docs mention `3.11`.
- Use `uv`; plain `pip install .` misses PyTorch CUDA package sources.
- Headless Linux defaults `MUJOCO_GL=egl` for `swarmbots.mj_env` / `swarmbots.mjw_env` when unset.
