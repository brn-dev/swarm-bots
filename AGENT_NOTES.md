# Agent Notes

Keep only durable architecture notes and gotchas. Prefer deleting stale detail over preserving trivia.

## Map

- `swarmbots/mj_env`: CPU MuJoCo envs, scenarios, and swarm generation. `swarmbots/mjw_env`: MJWarp batched GPU envs and the primary env path. `swarmbots/learn`: PPO/MAT, recurrent rollout, wrappers, action dists, checkpoints, logging, and world-model wrappers.
- Main training entrypoint: `scripts/run_mat_qcs_nop_wall.py`; older `scripts/run_mat_*.py` and `recording/record.py` may be stale.

## Autoreset And Wrappers

- Env autoreset must stay `SAME_STEP`. Read `gymnasium_autoreset.md` before touching rollout/reset logic.
- Under `SAME_STEP`, done-step bootstrap obs is `infos["final_obs"]`, not the returned reset obs. Terminal stats may be in `infos["final_info"]`.
- Wrappers must transform `final_obs` consistently with normal obs. Obs norm must not update RMS twice.
- Canonical learn wrapper order: `SwarmBotsLearnEnvWrapper -> TorchRecordEpisodeStatisticsWrapper -> TorchProgressGuidanceEpisodeStatsWrapper -> TorchFeatureWiseObsNormWrapper -> TorchTransitionObsWrapper -> TorchNormalizeRewardWrapper`, unless PopArt handles reward normalization.
- `SwarmBotsLearnEnvWrapper` is the conversion boundary and owns `env.action_backend` action conversion. Learn obs must contain `local_obs`, `global_obs`, `hidden_local_vars`, `hidden_global_vars`; `agent_mask` is optional.
- Observation layout changes must update `build_obs_indices(...)` and `_global_scalar_indices(...)`; normalization and world-model targets depend on them.
- `TorchShuffleAgentsWrapper` owns agent-order randomization. For done envs, shuffle `final_obs` with the old permutation, then resample permutations for returned reset obs.
- `TorchTransitionObsWrapper` zeros previous-transition features on returned reset obs but preserves terminal transition features in `final_obs`.
- Env obs buffers are not durable. Async/shared-memory envs, GPU envs, and `copy=False` wrappers may mutate them after the next env call; snapshot anything retained.

## Rollout, PPO, MAT, Recurrent

- `collect_steps()` chunks may start mid-episode. Use `PPOEpisodeSegment.is_true_episode_start`; do not infer from chunk position.
- PPO warmup rollout keeps only `PPORolloutState`; it must not increment timesteps/iterations or emit episode metrics.
- Rollout bootstrap should use a value-only path, not full deterministic action generation.
- Off-policy rollout/replay infrastructure lives in `swarmbots.learn.algos.off_policy`; it stores episode segments/chunks with explicit `next_*` observations, `terminations`, `truncations`, `previous_actions`, and `next_previous_actions`. Shared SAME_STEP final-observation and episode-info handling lives in `swarmbots.learn.rollout_utils`.
- `OffPolicyReplayBuffer` is ring-backed with per-env transition slots plus one extra per-env observation slot, so normal `next_obs` is shared with the next transition's current obs instead of duplicated. Terminal SAME_STEP `final_obs` lives in a separate terminal-observation ring. Keep transition-slot and observation-slot indexing separate when changing replay sampling.
- Recurrent state is explicit in `PPORolloutState`, not inside the policy. Temporal APIs take/return opaque tensor trees with leading env batch dimension.
- Step PPO rollouts use `PPORolloutBatch` for plain flat PPO configs and exact flat WM configs, keeping fixed `(env, step, ...)` tensors through training. Recurrent MAT still uses episode segments for TBPTT row construction.
- Recurrent rollouts use env-major TBPTT rows. Store one initial recurrent state per row; do not restore burn-in or store per-step recurrent state.
- Under `SAME_STEP`, terminal bootstrap evaluates `final_obs` from the post-current-observation state without committing the resulting state. The returned reset obs consumes the done reset mask on the next rollout step.
- `MATQCBasePolicy` owns shared query/context-style MAT PPO plumbing; `MATQCSPolicy` and `MATQCXPolicy` are sibling variants. `MATQCSPolicy` is the main transformer policy; `PPOPolicy` is the plain MLP policy; `MATDecPolicy` is decoderless despite the name.
- `MATOrigPolicy` uses shifted previous-agent actions, requires contiguous true-prefix `agent_mask`, and must zero inactive-agent log-probs in rollout and `evaluate_actions()`.
- `MATQCSPolicy` supports arbitrary inactive positions if every row has at least one active agent. `assume_agent_mask_is_active_prefix=True` enables the cheap prefix path for QCS/QCC configs.
- `MATQCXPolicy` is QCC-style query-to-context only: action encoding is policy-side via `action_encoder_dims` (`None` = direct projection to decoder width, `[]` = identity/raw action width, non-empty dims define the full MLP and final action-token width), decoder input is projected once to decoder width, query encoders are optional per-layer modules defaulting to identity, and context encoders are per decoder layer with context-token norm enabled by default. QCX does not do context-context attention and does not support decoder agent embeddings yet.
- Recurrent MAT variants share `RMATPolicyMixin` and env-major `RPPOWMSampler` TBPTT rows. `MATQCCPolicy` is not state-dict compatible with `MATQCSDecoder`'s interleaved `CONTEXT_TOKENS_ONLY` implementation. QCC parallel evaluation omits the final context/action token like QCS/QCX; `MATQCCDecoderConfig.tie_query_context_and_context_self_attention=False` adds separate context-self attention parameters.
- PyTorch-only xLSTM memory-core adapters live in `swarmbots.learn.algos.xlstm`: `MLSTMTemporalSequenceModel` and `SLSTMTemporalSequenceModel` match the `TemporalSequenceModel` contract without replacing the full xLSTM block stack. Projection biases are disabled by default so reset zero-input rows do not emit bias-only signals; sLSTM keeps an output head norm. mLSTM uses a stabilized parallel sequence path with reset masks and padded/invalid rows folded into the parallel formula; invalid steps are state no-ops with zero output. `RMATEncoderConfig.temporal_model_cls/config` may be either a single spec or one spec per RMAT layer.
- MJW experiment common accepts `r_mat_qcs`, `r_mat_qcc`, `r_mat_qcx`, and `r_mat_dec`; recurrent sampler batch size is the env-row count (`num_envs`), while rollout samples remain `num_envs * rollout_steps_per_env`.
- For MAT customization, override `_build_encoder*()` / `_build_action_dist(...)`; do not mutate fields after `super().__init__()`.
- `ActionDist.compile_friendly` gates MAT compile coverage. Mutable action-dist scalars must be tensors/buffers, not Python floats. Sticky dists must keep `requires_previous_actions()` structurally stable even when annealed to zero.
- `ReparameterizedSignMagnitudeKumaraswamyActionDist` is the cheap pathwise alternative to sign/magnitude beta: it uses two disjoint action intervals with inverse-CDF Kumaraswamy samples, so `sample()` has gradients through mixture probability and shape parameters while `log_prob()` stays exact for the piecewise density. `ReparameterizedSquashedGaussianMixtureActionDist` is a more general bounded mixture using inverse-CDF solve in tanh-Gaussian latent space plus a custom implicit-gradient backward; it is more expensive and marked non-compile-friendly.

## World Models

- World-model integration is policy-wrapper based: `NextObsPredWrapper` / `SPRWrapper`.
- WM wrappers must delegate sampler, explicit temporal-state APIs, and `after_optimizer_step()` to the wrapped policy. Flat WM samplers break RMAT.
- WM losses use `wm_actions`, not PPO current-step `actions`; recurrent padding uses `time_mask`.
- NOP global-observation prediction is optional. It needs explicit global scalar/rot6d target indices and pools predicted agent latents across active agents before global heads.
- Shared recurrent WM flattening lives in `swarmbots/learn/algos/world_modeling/wm_recurrent_batch.py`.

## Scenarios And Swarms

- `SwarmBotsEnv` delegates most behavior to scenarios. Scenario cadence, reset baselines, and per-step progress deltas are scenario-owned; do not add env-level `action_repeat` or generic `BaseScenario.compute_progress(...)`.
- Render overlays are visual-only via `BaseScenario.add_render_geoms(scene)` after `Renderer.update_scene()`; do not use them for physics/model geometry.
- Scenario `global_obs` layouts are contract-like: changing them requires updating learn observation indices and world-model target selection, not just the scenario code.
- `SwarmBotsEnv.reset(seed=...)` must reseed `scenario.rng`; reset sampling does not use Gymnasium `env.np_random`.

## MJW

- `MJWSwarmBotsVectorEnv` is a real `gymnasium.vector.VectorEnv` with one MJWarp model, batched GPU `Data`, and `action_backend="torch"`.
- Scenario-specific MJW logic belongs in scenario runtime classes, not `mjw_swarm_bots_vector_env.py`. Scenario metadata goes through `scenario.build_runtime_metadata(...)`; keep `MJWModelMetadata` generic.
- Scenario terminations flow through `MJWStepResult.terminations`.
- Vertical reach is not climb: the tall wall is effectively unclimbable, the green protruding goal box is visual/non-colliding, success is any active unit entering it, and the main shaping reward is max active-unit height inside the front-wall reach column. The box-distance reward is secondary. Global obs/success use the goal box center, while horizontal shaping defaults to the wall/box contact point (`horizontal_goal_at_wall_contact=True`).
- Reset paths: direct, one-shot `settle_initial_reset`, and a background CPU-settled snapshot buffer. MJW samples reset specs on the main thread and settles them in the executor; done-world resets consume ready snapshots and only block on buffer misses.
- Hot-path connector matching is kernelized in `swarmbots/mjw_env/mjw_kernels.py`; keep Warp indices `int32` unless PyTorch indexing forces `int64`.
- Reuse GPU scratch buffers. Avoid rebuilding tensors or Python branching in the MJW step path.
- Live MJW recording uses one shared `mujoco.Renderer`; `max_parallel_episodes` only caps concurrent episodes and should not multiply renderer VRAM.
- MuJoCo render resolution above 640x480 needs `model.vis.global_.offwidth/offheight` raised before constructing `mujoco.Renderer`; `ensure_mujoco_offscreen_framebuffer(...)` owns this.

## Runtime And Install

- Runtime hyperparameters are live attributes. Mutating config dataclasses after init does nothing.
- Activation factories with `ParameterLearnMode.PER_FEATURE` need explicit feature counts; use `make_activation(...)` in generic `act_fn_cls` paths.
- Initialization gains: hidden/projection/transformer-FF `1.0`; output/action/value/prediction heads `0.01`.
- Remote run logs are reachable via WSL SSH at `brn@server2026`, `~/swarm-bots/runs`. Logs are huge; summarize with scripts/code and fetch sparsely.
- Target Python is `3.13` (`pyproject.toml` / `.python-version`), even if older docs mention `3.11`.
- Use `uv`; plain `pip install .` misses PyTorch CUDA package sources.
- Headless Linux defaults `MUJOCO_GL=egl` for `swarmbots.mj_env` / `swarmbots.mjw_env` when unset.
