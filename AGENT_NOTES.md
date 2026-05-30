# Agent Notes

Keep only durable architecture notes and gotchas. Prefer deleting stale detail over preserving trivia.

## Map

- `swarmbots/mj_env`: CPU MuJoCo envs, scenarios, and swarm generation.
- `swarmbots/mjw_env`: MJWarp batched GPU envs.
- `swarmbots/learn`: PPO/MAT, action dists, wrappers, rollout, checkpoints, logging.
- Main training entrypoint: `scripts/run_mat_qcs_nop_wall.py`; older `scripts/run_mat_*.py` and `recording/record.py` may be stale.

## Wrappers And Autoreset

- Canonical learn wrapper order: `SwarmBotsLearnEnvWrapper -> TorchRecordEpisodeStatisticsWrapper -> TorchProgressGuidanceEpisodeStatsWrapper -> TorchFeatureWiseObsNormWrapper -> TorchTransitionObsWrapper -> TorchNormalizeRewardWrapper` unless PopArt handles normalization.
- `SwarmBotsLearnEnvWrapper` is the generic conversion boundary and owns `env.action_backend` action conversion.
- Learn obs must contain `local_obs`, `global_obs`, `hidden_local_vars`, `hidden_global_vars`; `agent_mask` is optional.
- Observation layout changes must update `build_obs_indices(...)` and `_global_scalar_indices(...)`; WM targets and normalization depend on them.
- Env autoreset must stay `SAME_STEP`. Read `gymnasium_autoreset.md` before touching rollout/reset logic.
- Under SAME_STEP, done-step bootstrap obs comes from `infos["final_obs"]`, not the returned reset obs. Terminal stats may be in `infos["final_info"]`.
- Learn wrappers must also transform `final_obs`. Obs norm must not update RMS twice.
- `TorchShuffleAgentsWrapper` owns agent-order randomization. Under SAME_STEP, shuffle `final_obs` with the old permutation, then resample done-env permutations for returned reset obs.
- `TorchTransitionObsWrapper` zeros previous-transition features on returned reset obs but preserves terminal transition features in `final_obs`.
- Env obs buffers are not durable. Async/shared-memory envs, GPU envs, and `copy=False` wrappers may mutate them after the next env call; snapshot anything retained.

## Rollout, PPO, MAT

- `collect_steps()` chunks may start mid-episode. Use `PPOEpisode.is_true_episode_start`; do not infer from chunk position.
- Rollout bootstrap should use a value-only path, not full deterministic action generation.
- `MATQCSPolicy` is the main transformer policy; `PPOPolicy` is the plain MLP policy; `MATDecPolicy` is decoderless despite the name.
- MAT variants:
  - `MATQCCPolicy` keeps query/context streams separate and is not state-dict compatible with `MATQCSDecoder`'s interleaved `CONTEXT_TOKENS_ONLY` implementation.
  - `MATOrigPolicy` uses shifted previous-agent actions, requires contiguous true-prefix `agent_mask`, and must zero inactive-agent log-probs in rollout and `evaluate_actions()`.
  - `MATQCSPolicy` supports arbitrary inactive positions if every row has at least one active agent. `assume_agent_mask_is_active_prefix=True` enables the cheap prefix path for QCS/QCC configs.
- For MAT-family customization, override `_build_encoder*()` / `_build_action_dist(...)`; do not mutate fields after `super().__init__()`.
- `ActionDist.compile_friendly` gates MAT compile coverage. Mutable action-dist scalars must be tensors/buffers, not Python floats.
- Sticky dists must keep `requires_previous_actions()` structurally stable even when annealed to zero.

## Recurrent And World Models

- `RMATQCSPolicy` stores rollout temporal state inside the policy. Under SAME_STEP, reset masks are queued after done and consumed on the next episode's first obs.
- Step-rollout bootstrap must snapshot/restore RMAT state. RMAT training uses zero-init plus burn-in, not exact rollout hidden-state replay.
- WM integration is policy-wrapper based: `NextObsPredWrapper` / `SPRWrapper`.
- WM wrappers must delegate sampler, temporal-state hooks, and `after_optimizer_step()` to the wrapped policy. Flat WM samplers break RMAT.
- WM losses use `wm_actions`, not PPO current-step `actions`; recurrent WM losses use `time_loss_mask`.
- NOP global-observation prediction is optional. It is enabled by explicit global scalar/rot6d target indices and pools predicted agent latents across active agents before global heads.
- Shared recurrent WM flattening lives in `swarmbots/learn/algos/world_modeling/wm_recurrent_batch.py`.

## Scenarios And Swarms

- `SwarmBotsEnv` delegates most behavior to scenarios. Scenario cadence is scenario-owned; do not add env-level `action_repeat`.
- Scenarios own reset baselines and per-step progress deltas; no generic `BaseScenario.compute_progress(...)`.
- Render overlays are visual-only via `BaseScenario.add_render_geoms(scene)` after `Renderer.update_scene()`; do not use them for physics/model geometry.
- Key `global_obs` layouts: payload `(x, y, z, rot6d)`, dual-payload two payload poses, move-to absolute goal `(x, y)`, climb top-face center goal `(x, y, z)`.
- Move-to payload transfer adapters expand goals into payload-shaped global obs for normalization/checkpoint compatibility. They intentionally keep global rot6d target indices empty.
- Random swarm z rotation must carry sampled yaw through all MJW reset paths.

## MJW

- `MJWSwarmBotsVectorEnv` is a real `gymnasium.vector.VectorEnv` with one MJWarp model, batched GPU `Data`, and `action_backend="torch"`.
- Scenario-specific MJW logic belongs in scenario runtime classes, not `mjw_swarm_bots_vector_env.py`.
- MJW scenario metadata goes through `scenario.build_runtime_metadata(...)`; keep `MJWModelMetadata` generic.
- Scenario terminations flow through `MJWStepResult.terminations`.
- Reset paths: direct, one-shot `settle_initial_reset`, CPU-settled prefetch for predicted truncations.
- Hot-path connector matching is kernelized in `swarmbots/mjw_env/mjw_kernels.py`; keep Warp indices `int32` unless PyTorch indexing forces `int64`.
- Reuse GPU scratch buffers. Avoid rebuilding tensors or Python branching in the MJW step path.

## Runtime

- Runtime hyperparameters are live attributes. Mutating config dataclasses after init does nothing.
- Checkpoints include policy state, optional optimizer state, env-wrapper normalization state, and training counters.
- Activation factories with `ParameterLearnMode.PER_FEATURE` need explicit feature counts; use `make_activation(...)` in generic `act_fn_cls` paths.
- MAT-QCS/NOP `proper_init_1`: hidden/projection/transformer-FF gains `1.0`, output/action/value/prediction heads `0.01`.
- CPU wall training should use `WorkerPoolAsyncVectorEnv(..., copy=False)`. Do not clone envs whose constructors differ in per-env state like `first_episode_length`.
- Medium wall presets enable `wall_climb_reward`: it tracks signed per-unit height-potential deltas in the approach band before each wall, using `sqrt(approach)` so the reward engages earlier; falls, backing away, or leaving the band can pay the shaping reward back. After a unit first crosses a wall's `wall_y`, that unit-wall climb potential is latched done, the crossing-induced potential drop is forgiven, and retries/backtracking for that wall no longer receive climb reward.
- Obstacle-street `forward_reward_wall_boost_factor` multiplies only positive per-unit forward deltas near/above a wall when the unit center is at least `wall_height + margin`; default margin is `swarm.body_radius`, and factor `1.0` disables it.
- `SwarmBotsEnv.reset(seed=...)` must reseed `scenario.rng`; reset sampling does not use Gymnasium `env.np_random`.
- On Windows, workers re-import the script top-level module. Keep heavy PPO/torch/MJW imports out of top level unless env construction needs them.
- Remote `brn@server2026` SSH works through WSL (`wsl ssh brn@server2026 ...`); Remote runs live under `~/git/swarm-bots/runs`. Use these to inspect live logs.

## Version And Install

- Target Python is `3.13` (`pyproject.toml` / `.python-version`), even if older docs mention `3.11`.
- Use `uv`; plain `pip install .` misses PyTorch CUDA package sources.
- Headless Linux defaults `MUJOCO_GL=egl` for `swarmbots.mj_env` / `swarmbots.mjw_env` when unset.
