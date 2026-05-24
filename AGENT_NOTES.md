# Agent Notes

Keep only durable architecture notes and gotchas. Prefer deleting stale detail over preserving trivia.

## Map

- `swarmbots/mj_env`: CPU MuJoCo envs/scenarios/swarm generation.
- `swarmbots/mjw_env`: MJWarp batched GPU envs.
- `swarmbots/learn`: PPO/MAT, action dists, wrappers, rollout, checkpoints, logging.
- Main training entrypoint: `scripts/run_mat_nop_wall.py`. Other `scripts/run_mat_*.py` and `recording/record.py` may be stale.
- New training scripts should usually start from `scripts/run_mat_nop_wall*.py`, `scripts/run_mat_nop_payload*.py`, or `scripts/run_mat_nop_move*.py`.

## Wrappers And Autoreset

- Canonical learn wrapper order: `SwarmBotsLearnEnvWrapper -> TorchRecordEpisodeStatisticsWrapper -> TorchProgressGuidanceEpisodeStatsWrapper -> TorchFeatureWiseObsNormWrapper -> TorchTransitionObsWrapper -> TorchNormalizeRewardWrapper` unless PopArt handles normalization.
- `SwarmBotsLearnEnvWrapper` is the generic conversion boundary and owns `env.action_backend` action conversion.
- Learn obs must contain `local_obs`, `global_obs`, `hidden_local_vars`, `hidden_global_vars`; `agent_mask` is optional.
- Observation layout changes must update `build_obs_indices(...)` and `_global_scalar_indices(...)`; WM targets and normalization depend on them.
- Env autoreset must stay `SAME_STEP`. Read `gymnasium_autoreset.md` before touching rollout/reset logic.
- Done-step bootstrap obs comes from `infos["final_obs"]`, not the returned reset obs. Terminal stats may be in `infos["final_info"]`.
- Learn wrappers must also transform `final_obs`. Obs norm must not update RMS twice.
- `TorchShuffleAgentsWrapper` owns agent-order randomization. Under SAME_STEP, shuffle `final_obs` with the old permutation, then resample done-env permutations for returned reset obs.
- `TorchTransitionObsWrapper` zeros previous-transition features on returned reset obs but preserves terminal transition features in `final_obs`.
- Env obs buffers are not durable. Async/shared-memory envs, GPU envs, and `copy=False` wrappers may mutate them after the next env call; snapshot anything retained.

## Rollout, PPO, MAT

- `collect_steps()` chunks may start mid-episode. Use `PPOEpisode.is_true_episode_start`; do not infer from chunk position.
- Rollout bootstrap should use a value-only path, not full deterministic action generation.
- `PPO.train()` uses `policy.make_sampler(episodes)`; `PPO.compute_loss()` uses `policy.evaluate_actions(batch=...)`.
- `MATPolicy` is the main transformer policy; `PPOPolicy` is the plain MLP policy.
- `MATDecPolicy` is decoderless despite the name.
- `MATOrigPolicy` uses shifted previous-agent actions, so it requires contiguous true-prefix `agent_mask`. It must zero inactive-agent log-probs in rollout and `evaluate_actions()`.
- `MATPolicy` supports arbitrary inactive positions if every row has at least one active agent; `MATDecoderConfig.assume_agent_mask_is_active_prefix=True` enables the cheap prefix path.
- For MAT-family customization, override `_build_encoder*()` / `_build_action_dist(...)`; do not mutate fields after `super().__init__()`.
- Keep actor latents contiguous before action dists. Pass `tgt_is_causal` explicitly to `nn.TransformerDecoder`.
- `ActionDist.compile_friendly` gates MAT compile coverage. Mutable action-dist scalars must be tensors/buffers, not Python floats.
- Sticky dists must keep `requires_previous_actions()` structurally stable even when annealed to zero.
- Default MAT/NOP training scripts use non-sticky `LeftRightBetaConfig`; opt into `sticky_lr_beta` only for comparison runs.
- Squashed Gaussian/gSDE entropy uses pre-squash Gaussian entropy proxies. `BetaActionDist` maps `[0, 1]` samples to repo-standard `[-1, 1]`.
- Transformer encoder/decoder layers clone identical prototype params unless explicitly reinitialized via configured transformer FF init gain.

## Recurrent And World Models

- `RMATPolicy` stores rollout temporal state inside the policy. Under SAME_STEP, reset masks are queued after done and consumed on the next episode's first obs.
- Step-rollout bootstrap must snapshot/restore RMAT state. RMAT training uses zero-init plus burn-in, not exact rollout hidden-state replay.
- WM integration is policy-wrapper based: `NextObsPredWrapper` / `SPRWrapper`.
- WM wrappers must delegate sampler, temporal-state hooks, and `after_optimizer_step()` to the wrapped policy. Flat WM samplers break RMAT.
- `SPRWrapper` does not support `RMATPolicy`.
- WM losses use `wm_actions`, not PPO current-step `actions`; recurrent WM losses use `time_loss_mask`.
- NOP global-observation prediction is optional. It is enabled by explicit global scalar/rot6d target indices and pools predicted agent latents across active agents before global heads; local-only configs should not allocate or require global NOP heads/tensors.
- Shared recurrent WM flattening: `swarmbots/learn/algos/world_modeling/wm_recurrent_batch.py`.

## Scenarios And Swarms

- `SwarmBotsEnv` delegates most behavior to scenarios. Scenario cadence is scenario-owned; do not add env-level `action_repeat`.
- Reward `info` should expose weighted values only; keep raw components internal.
- Scenarios own reset baselines and per-step progress deltas; no generic `BaseScenario.compute_progress(...)`.
- Render overlays are visual-only via `BaseScenario.add_render_geoms(scene)` after `Renderer.update_scene()`; do not use them for physics/model geometry.
- Payload `global_obs`: `(x, y, z, rot6d)`. Dual-payload: two payload poses. Move-to: absolute goal `(x, y)`.
- Move-to payload transfer adapters expand goals into payload-shaped global obs for normalization/checkpoint compatibility.
- Real payload layouts expose `global_rot6d_indices` for NOP global targets; move-to payload adapters intentionally keep global rot6d target indices empty.
- Shared mirrored scenario kwargs belong in `swarmbots/scenario_presets/scenario_presets_kwargs.py`.
- Payload presets default to `"box"` to avoid trivial rolling.
- Wall-pass reward stays normalized by active unit and threshold counts. Optional `wall_pass_reward_skew` changes crossing-rank payout but preserves the all-active-units total. `forward_reward_max_y` must not affect threshold crossing.
- Fixed preconnected swarm pools support runtime `active_pool_size` curriculum on CPU and MJW.
- Random swarm z rotation must carry sampled yaw through all MJW reset paths.
- Twist quantization prebuilds weld equalities and toggles `data.eq_active`; preconnected generation must sample quantized twist values.

## MJW

- `MJWSwarmBotsVectorEnv` is a real `gymnasium.vector.VectorEnv` with one MJWarp model, batched GPU `Data`, and `action_backend="torch"`.
- Scenario-specific MJW logic belongs in scenario runtime classes, not `mjw_swarm_bots_vector_env.py`.
- MJW scenario metadata goes through `scenario.build_runtime_metadata(...)`; keep `MJWModelMetadata` generic.
- Scenario terminations flow through `MJWStepResult.terminations`.
- Keep workspace defaults lean: `nconmax = max(32, total_connectors + 2*num_units)`, `njmax = max(160, 5*nconmax + 4*num_units)`.
- Use `MJWSwarmBotsVectorEnv(ccd_iterations=...)` for box/convex-heavy runs with CCD warnings.
- Reset paths: direct, one-shot `settle_initial_reset`, CPU-settled prefetch for predicted truncations.
- Hot-path connector matching is kernelized in `swarmbots/mjw_env/mjw_kernels.py`; keep Warp indices `int32` unless PyTorch indexing forces `int64`.
- Reuse GPU scratch buffers. Avoid rebuilding tensors or Python branching in the MJW step path.
- MJW live recording copies live MJW state into CPU MuJoCo render slots. Do not replay MJW actions in plain `mj_env`.

## Runtime

- Runtime hyperparameters are live attributes. Mutating config dataclasses after init does nothing.
- Checkpoints include policy state, optional optimizer state, env-wrapper normalization state, and training counters.
- Use `BaseAlgorithm.learn(..., post_iteration_hooks=...)` for one-shot side effects.
- Logs use `log.csv` with `;` delimiter. Plot tooling reads `.csv`, `.zip`, `.gz`, `.bz2`, `.xz`.
- `plot_logs.experiment_results.plot_experiment_results()` supports `extra_plot_selections` for additional subset plots without duplicating experiment-specific plotting code.
- Gymnasium vector-info packing adds boolean `_key` masks for every info field. Ignore underscore-prefixed entries in reward overlays.
- `scripts/utils/run_repeated.sh` registers under `.run/run_repeated`; `scripts/utils/show_run_repeated.sh` lists active registrations; `scripts/utils/stop_run_repeated.sh` creates the stop token.
- Activation factories with `ParameterLearnMode.PER_FEATURE` need explicit feature counts; use `make_activation(...)` in generic `act_fn_cls` paths.
- MAT/NOP `proper_init_1`: hidden/projection/transformer-FF gains `1.0`, output/action/value/prediction heads `0.01`.
- Plain `SquaredReLU` with default tiny MLP init can collapse MAT/NOP feature scales and critic gradients.
- CPU wall training should use `WorkerPoolAsyncVectorEnv(..., copy=False)`. Do not clone envs whose constructors differ in per-env state like `first_episode_length`.
- `SwarmBotsEnv.reset(seed=...)` must reseed `scenario.rng`; reset sampling does not use Gymnasium `env.np_random`.
- On Windows, workers re-import the script top-level module. Keep heavy PPO/torch/MJW imports out of top level unless env construction needs them.
- Recording camera defaults should stay scenario-owned. For compiled MAT/NOP, record envs should stay on the policy's active device.
- Remote `brn@server2026` SSH works through WSL (`wsl ssh brn@server2026 ...`); Windows `ssh` may hang or fail auth. Remote runs live under `~/git/swarm-bots/runs`.

## Version And Install

- Target Python is `3.13` (`pyproject.toml` / `.python-version`), even if older docs mention `3.11`.
- Use `uv`; plain `pip install .` misses PyTorch CUDA package sources.
- Headless Linux defaults `MUJOCO_GL=egl` for `swarmbots.mj_env` / `swarmbots.mjw_env` when unset.
