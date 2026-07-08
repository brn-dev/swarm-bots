# Agent Notes

Keep only durable architecture notes and gotchas. Delete stale detail instead of preserving trivia.

## Map

- `swarmbots/mj_env` is the CPU MuJoCo path; `swarmbots/mjw_env` is the MJWarp batched GPU path; `swarmbots/learn` contains PPO/MAT/RMAT, rollout, replay, wrappers, action dists, checkpointing, logging, and world-model wrappers.
- Scripts under `scripts/` are examples/entrypoints. Shared modern MJW experiment setup is in `experiments/mjw_experiment_common.py`; old experiment variants are not automatically canonical.

## SAME_STEP And Wrappers

- Vector envs and learn wrappers assume `gymnasium.vector.AutoresetMode.SAME_STEP`; read `gymnasium_autoreset.md` before touching reset/rollout code.
- On done steps the returned obs is already reset obs. Terminal obs is `infos["final_obs"]` with `infos["_final_obs"]`; terminal stats may be in `infos["final_info"]`.
- Obs wrappers must transform `final_obs` like normal obs, but obs norm must not update RMS from `final_obs`.
- Standard learn wrapper order: `SwarmBotsLearnEnvWrapper -> optional TorchShuffleAgentsWrapper -> TorchRecordEpisodeStatisticsWrapper -> TorchProgressGuidanceEpisodeStatsWrapper -> TorchFeatureWiseObsNormWrapper(s) -> optional TorchTransitionObsWrapper -> TorchNormalizeRewardWrapper`, skipping reward norm when PopArt owns it.
- `SwarmBotsLearnEnvWrapper` owns numpy/torch conversion and `env.action_backend`. Learn obs require `local_obs`, `global_obs`, `hidden_local_vars`, `hidden_global_vars`; `agent_mask` is optional.
- Observation layout changes must update `build_obs_indices(...)` and related global-scalar target logic. Normalization and world-model targets depend on those indices.
- Done `final_obs` in `TorchShuffleAgentsWrapper` uses the old permutation; returned reset obs gets the new one. `TorchTransitionObsWrapper` keeps terminal transition features in `final_obs` but zeros reset obs.
- Env obs buffers are not durable; async/shared-memory/GPU envs and `copy=False` conversions can mutate them after the next env call.

## Rollout And Replay

- `collect_steps()` chunks can start mid-episode. Use `PPOEpisodeSegment.is_true_episode_start`; do not infer from chunk position.
- Recurrent state is explicit in `PPORolloutState`. SAME_STEP terminal bootstrapping evaluates `final_obs` without committing the resulting recurrent state; reset obs consumes the done mask on the next rollout step.
- Flat PPO configs use `PPORolloutBatch`; recurrent MAT/RMAT uses env-major TBPTT rows with one initial recurrent state per row, not per-step stored recurrent state.
- Off-policy rollout/replay lives in `swarmbots.learn.algos.off_policy`. `OffPolicyReplayBuffer` is a per-env ring stream with `capacity + 1` obs slots; terminal `final_obs` is sparse by transition slot while returned reset obs stays in the stream.
- Off-policy recurrent replay stores optional temporal-state checkpoints on observation slots via `temporal_state_store_interval`; `sample_episode_segments(...)` starts at checkpointed obs and returns burn-in + train windows that do not cross episode boundaries before the final transition.
- Keep collecting into a nonempty replay buffer with the returned `OffPolicyRolloutState`; resetting the env into that buffer without state breaks observation-slot continuity.
- `OffPolicyReplayBatch` has no `next_previous_actions`; use sampled `actions` when conditioning on `next_*` obs. Pass `rollout_device` explicitly when moving env, policy inference, and resumed rollout state.
- gSDE off-policy rollouts need `gsde_reset_mode`; `OffPolicyRolloutState.gsde_noise_initialized` forces full batch-shaped noise after random warmup, reset, or rollout-device moves.
- SAC lives in `swarmbots.learn.algos.sac` and reuses off-policy rollout/replay. It intentionally rejects non-`Box` action sub-spaces; use continuous connector actions for TMASAC/SAC, not Bernoulli connectors.
- TMASAC SAC actors currently accept the reparameterizable continuous configs `BetaConfig`, `PredictedStdConfig`, `SquashedDiagGaussianConfig`, `GSDEConfig`, `ReparameterizedSignMagnitudeKumaraswamyConfig`, and `ReparameterizedSquashedGaussianMixtureConfig`. SAC with `GSDEConfig` requires an explicit `gsde_reset_mode`.
- Action distributions follow the PyTorch naming convention: `sample()` is ordinary non-reparameterized sampling, `rsample()` keeps the actor-gradient path. `ActionDist.get_actions_with_log_probs(..., use_rsample=True)` is for SAC actor updates; rollout and target-Q sampling use the default ordinary `sample()` path.
- `TMASACPolicy` uses separate actor/critic MAT encoders by default, can share the observation encoder with `share_observation_encoder=True`, and that shared encoder is critic-owned: actor reads detach its latents so only critic/NOP critic gradients update the encoder. SAC NOP modules are selected by latent source (`critic`, `actor`, or `both`); `both` means separate actor/critic projections, transition models, and losses. Multi-step SAC NOP uses `SAC(nop_steps=4)` by default and samples a separate contiguous episode segment batch for NOP losses when NOP is enabled; if replay has no valid contiguous NOP window yet, SAC skips only the auxiliary NOP loss and still runs the main actor/critic update.

## Policies And Action Dists

- `MATQCBasePolicy` holds shared QCS/QCX/QCC PPO plumbing. QCS is the main transformer policy; QCC subclasses QCS with its own decoder; QCX is query-to-context only; `MATDecPolicy` is decoderless despite the name.
- `MATEncoder` owns its custom `MATEncoderLayer` stack directly. `transformer_ff_hidden_dims` configures hidden MLP widths before the final `d_model` projection; `transformer_ff_init_gain=None` intentionally preserves cloned identical layer initialization.
- `MATOrigPolicy` requires contiguous true-prefix `agent_mask` and shifted previous-agent actions; inactive-agent log-probs must be zeroed in rollout and `evaluate_actions()`.
- QCS/QCC/QCX support arbitrary inactive positions if each row has at least one active agent. `assume_agent_mask_is_active_prefix=True` selects the cheaper prefix path.
- RMAT variants share `RMATPolicyMixin` and `RPPOWMSampler` env-major TBPTT rows. `RMATEncoder` owns its custom `RMATEncoderLayer` stack directly; `transformer_ff_hidden_dims` also applies to the optional inter-module MLP. Temporal cores must honor reset masks. The default LSTM temporal core/output projection are biasless so reset zero-input rows do not emit learned initial-state priors; `RMATEncoderConfig.use_temporal_output_projection=False` removes the extra temporal output projection.
- For MAT customization, override `_build_encoder*()` and `_build_action_dist(...)`; do not mutate inherited fields after `super().__init__()`.
- `ActionDist.compile_friendly` controls torch.compile coverage. Mutable action-dist scalars must be tensors/buffers, and sticky dists must keep `requires_previous_actions()` structurally stable even when annealed to zero.

## World Models

- World models are policy wrappers: `NextObsPredWrapper` and `SPRWrapper`. They must delegate sampler selection, temporal-state APIs, and `after_optimizer_step()` to the wrapped policy; flat WM samplers break RMAT.
- WM losses use `wm_actions`, not PPO current-step `actions`; recurrent padding uses `time_mask` via `wm_recurrent_batch.py`.
- NOP global-observation prediction needs explicit global scalar/rot6d target indices and pools predicted agent latents across active agents before global heads. `SPRWrapper` currently rejects RMAT; use `NextObsPredWrapper`.

## Scenarios And MJW

- `SwarmBotsEnv` delegates cadence, reset baselines, and progress deltas to scenarios. Do not add env-level `action_repeat` or generic `BaseScenario.compute_progress(...)`.
- Scenario `global_obs` layouts are contracts; changing them requires learn obs-index and WM target updates. `SwarmBotsEnv.reset(seed=...)` must reseed `scenario.rng`.
- Render overlays are visual only via `BaseScenario.add_render_geoms(scene)` after `Renderer.update_scene()`; do not use them for physics/model geometry.
- `MJWSwarmBotsVectorEnv` is a real `gymnasium.vector.VectorEnv` with one MJWarp model, batched GPU `Data`, and `action_backend="torch"`.
- Scenario-specific MJW logic belongs in scenario runtime classes. Runtime metadata flows through `scenario.build_runtime_metadata(...)`; keep `MJWModelMetadata` generic. Terminations flow through `MJWStepResult.terminations`.
- MJW reset modes: direct, one-shot `settle_initial_reset`, and background CPU-settled snapshot buffer. Done-world resets consume ready snapshots and block only on buffer misses.
- Keep MJW step hot paths kernelized and allocation-light: connector matching is in `mjw_kernels.py`, Warp indices stay `int32` unless PyTorch indexing forces `int64`, and GPU scratch buffers should be reused.
- Live MJW recording uses one shared `mujoco.Renderer`; `max_parallel_episodes` caps concurrency, not renderer count. Resolutions above 640x480 need `ensure_mujoco_offscreen_framebuffer(...)` before renderer construction.
- Vertical reach is not climb: the green goal box is visual/non-colliding, success is any active unit entering it, and main shaping is max active-unit height inside the front-wall reach column.

## Runtime

- Runtime hyperparameters are live attributes; mutating config dataclasses after init does nothing.
- Activation factories with `ParameterLearnMode.PER_FEATURE` need explicit feature counts; generic activation paths should use `make_activation(...)`.
- Initialization gains: hidden/projection/transformer-FF `1.0`; output/action/value/prediction heads `0.01`.
- Target Python is `3.13`; use `uv` because plain `pip install .` misses PyTorch CUDA package sources.
- Run tests through uv, e.g. `uv run python -m pytest tests/test_off_policy_replay.py`. In Codex desktop `cmd.exe`, `.venv\Scripts\python.exe` can fail with `Access is denied` because it is a uv-managed launcher; do not treat that as a repo failure.
- Headless Linux defaults `MUJOCO_GL=egl` for `swarmbots.mj_env` and `swarmbots.mjw_env` when unset.
