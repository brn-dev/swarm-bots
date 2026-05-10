# Agent Notes

Agents use this file for durable codebase notes. Keep only architecture, invariants, entrypoints, and real gotchas.

## TL;DR
- Main layers:
  - `swarmbots/mj_env`: CPU MuJoCo envs, scenarios, swarm generation.
  - `swarmbots/mjw_env`: MJWarp batched GPU envs. Narrower than `mj_env`, but covers wall, bridge, payload-plane, and move-to on one batched runtime architecture.
  - `swarmbots/learn`: PPO/MAT, action dists, wrappers, rollout, checkpoints, logging.
- Canonical training entrypoint is `scripts/run_mat_nop_wall.py`.
- Other `scripts/run_mat_*.py` files and `recording/record.py` can be stale. Do not assume they match current MAT APIs.

## Training Flow
- Typical wrapper chain:
  - `SwarmBotsLearnEnvWrapper`
  - `TorchRecordEpisodeStatisticsWrapper`
  - `TorchProgressGuidanceEpisodeStatsWrapper`
  - `TorchFeatureWiseObsNormWrapper`
  - `TorchTransitionObsWrapper`
  - `TorchNormalizeRewardWrapper` unless using PopArt
- `PPO.perform_iteration()` collects rollouts (`collect_whole_episodes` or `collect_steps`) and then trains.

## PPO / Policy Invariants
- Core hierarchy is `BaseAlgorithm -> PPO` and `BasePolicy -> BasePPOPolicy`.
- `PPO.train()` always uses `policy.make_sampler(episodes)`.
- `PPO.compute_loss()` always uses `policy.evaluate_actions(batch=...)`.
- `MATPolicy` is the main transformer policy; `PPOPolicy` is the plain MLP policy.
- World-model integration is wrapper-first, not algorithm-specific:
  - `NextObsPredWrapper(BasePPOPolicy, NextObsPredMixin)`
  - `SPRWrapper(BasePPOPolicy, SPRMixin)`

## MAT / Compile Gotchas
- `ActionDist.compile_friendly` gates how much of MAT can be compiled. `HybridActionDistribution` is compile-friendly only if every sub-dist is.
- If MAT-family policies replace encoder or top-level action-dist wrapper, do it through `_build_encoder*()` / `_build_action_dist(...)`, not by mutating fields after `super().__init__()`.
- Keep actor latents passed into action-dists contiguous. Non-contiguous decoder slices trigger Dynamo recompiles.
- Pass `tgt_is_causal` explicitly to `nn.TransformerDecoder`; causal auto-detection graph-breaks.
- Do not thread Python agent indices through compile-friendly action-dist paths unless sampling really depends on them.
- PPO rollout bootstrap should use a value-only path, not full `policy(..., deterministic=True)`.
- Mutable scheduler-driven action-dist scalars must live in tensors/buffers, not Python floats.
- For sticky distributions, keep `requires_previous_actions()` structurally stable even when stickiness anneals to `0`.

## Recurrent / WM Gotchas
- `RMATPolicy` keeps rollout-time temporal state inside the policy.
- Under SAME_STEP autoreset, RMAT reset masks are queued after a done step and consumed on the next episode's first observation.
- Step-rollout bootstrap must snapshot/restore RMAT temporal state.
- Current RMAT training still uses zero-init plus burn-in windows, not exact rollout hidden-state replay.
- WM wrappers must delegate `make_sampler(...)`, temporal-state hooks, and `after_optimizer_step()` to the wrapped policy. Hardcoding flat WM samplers breaks RMAT.
- `SPRWrapper` intentionally does not support `RMATPolicy`.
- WM losses must use `wm_actions`, not PPO current-step `actions`.
- Recurrent WM losses must mask with `time_loss_mask`.
- Shared recurrent WM flattening lives in `swarmbots/learn/algos/world_modeling/wm_recurrent_batch.py`.

## Rollout / Env Invariants
- Vector env autoreset must stay `SAME_STEP` end-to-end.
- Done-step bootstrap observations come from `infos["final_obs"]`, not the reset observation batch.
- Raw Gymnasium done-step stats may arrive under `infos["final_info"]`; rollout code must unwrap them when later wrappers did not inject stats.
- Learn-side wrappers must transform `final_obs` too. `TorchFeatureWiseObsNormWrapper` must not update RMS twice, and `TorchTransitionObsWrapper` must stack transition features onto single-env `final_obs`.
- `collect_steps()` may emit rollout segments that start mid true episode. Use `PPOEpisode.is_true_episode_start`; do not infer from chunk position.
- Step-rollout accumulator capacity is bounded by rollout segment length, not true env episode length.

## Observation / Wrapper Invariants
- Learn-side obs must contain `local_obs`, `global_obs`, `hidden_local_vars`, `hidden_global_vars`; `agent_mask` is optional but supported.
- `MATPolicy` currently assumes a contiguous true-prefix `agent_mask`, and agent 0 must be active.
- If observation layout changes, update `build_obs_indices(...)` first. WM targets and normalization depend on it.
- Keep `_global_scalar_indices(...)` in sync with every non-empty `global_obs` layout.
- `Torch*Wrapper` classes are the canonical learn-side wrappers. `SwarmBotsLearnEnvWrapper` is the only generic conversion boundary.
- `SwarmBotsLearnEnvWrapper` also owns action conversion via `env.action_backend`; torch interop uses zero-copy DLPack-style paths where possible.
- `FeatureWiseObsNormWrapper` is copy-on-write for its obs key and must not mutate incoming arrays.
- Checkpoint env-state restore is strict on wrapper class/order and, for feature normalization, `obs_key`.

## Scenario / Reward Notes
- `SwarmBotsEnv` delegates most behavior to scenario classes.
- Reward info exposed in `info` should be weighted values only. Keep raw reward components internal.
- `BaseScenario` no longer has a generic `compute_progress(...)`; scenarios own both reset baselines and per-step progress deltas.
- Dedicated scenario pairs now exist on both backends:
  - payload: `PayloadPlaneScenario` / `MJWPayloadPlaneScenario`
  - move-to: `MoveToScenario` / `MJWMoveToScenario`
  - bridge: MJW also has `MJWBridgeScenario`
- Payload `global_obs` is `(x, y, z, rot6d)`. Move-to `global_obs` is absolute goal `(x, y)`.
- `MoveToPayloadGlobalObsAdapter` in `scripts/run_mat_nop_move_to_payload_mjw.py` expands move-to goals into payload-shaped global obs so normalization/checkpoint state stays compatible.
- Move-to pretraining intentionally anneals sticky actions to zero and keeps them at zero for the payload phase.
- Shared mirrored scenario kwargs belong in `swarmbots/scenario_presets/scenario_presets_kwargs.py`.
- Payload supports `"sphere"`, `"box"`, and `"capsule"` shapes; presets default to `"box"` to avoid trivial rolling.
- Obstacle-street wall-pass reward must stay normalized by active unit count and threshold count.
- `forward_reward_max_y` clamps forward-progress contribution only; it must not affect threshold crossing logic.

## Swarm / Connection Notes
- `HomogeneousSwarm` supports preset layouts, explicit coordinates, Poisson-disc generation, and preconnected pools.
- Fixed preconnected swarm pools support runtime `active_pool_size` curriculum on both CPU and MJW.
- `randomize_initial_swarm_z_rotation=True` rotates active units rigidly on reset; MJW reset specs must carry the sampled yaw through both direct and CPU-settled reset paths.
- `quantize_connection_twist=N` prebuilds weld equalities per connector pair and activates the nearest one by toggling `data.eq_active`.
- When twist quantization is enabled, pre-connected swarm generation must sample directly from the quantized twist values.
- `minimal_contacts=True` removes several self-collision cases; `use_cylinders=False` switches limb and connector geoms to capsules.

## MJW-Specific Notes
- `MJWSwarmBotsVectorEnv` is a true `gymnasium.vector.VectorEnv` with one MJWarp model plus batched `Data` on GPU and `action_backend="torch"`.
- Scenario-specific MJW logic belongs in scenario runtime classes, not in `mjw_swarm_bots_vector_env.py`.
- MJW scenario metadata goes through `scenario.build_runtime_metadata(...)`; keep `MJWModelMetadata` generic, not scenario-specific.
- Scenario-owned terminations flow through `MJWStepResult.terminations`.
- Workspace defaults must stay lean. Current default sizing is `nconmax = max(24, total_connectors + 2*num_units)` and `njmax = max(160, 5*nconmax + 2*num_units)`.
- Reset behavior has three paths: direct reset, optional one-shot `settle_initial_reset`, and CPU-settled prefetch for predicted truncations.
- Hot-path connector matching is kernelized in `swarmbots/mjw_env/mjw_kernels.py`; keep Warp-side metadata/index tensors `int32` unless PyTorch indexing forces `int64`.
- Keep hot-path GPU scratch buffers reused. Avoid rebuilding tensors or adding Python-side branching back into the step path.
- Obstacle-street reward-kernel compile is optional and should only default on when local support checks pass.

## Runtime / Logging
- Runtime hyperparameters are live attributes. Mutating config dataclasses after init does nothing.
- `get_hyper_parameters()` should report current live values.
- Checkpoints include policy state, optional optimizer state, env-wrapper normalization state, and training counters.
- `BaseAlgorithm.learn(...)` supports `post_iteration_hooks`; use that for one-shot side effects instead of abusing schedulers.
- `learn()` interactive commands include lr/loss/reward/save/record/pause/stop and persist to `command_log.jsonl`.
- For compiled MAT/NOP policies, separate record envs should stay on the policy's active device.
- Logs use `log.csv` with `;` delimiter. Optional graceful-exit compression writes `log.csv.gz`.
- Plot tooling in `plot_logs/` reads plain `.csv` plus `.zip`, `.gz`, `.bz2`, and `.xz` directly.
- Gymnasium vector-info packing adds boolean `_key` masks for every info field, including nested reward-term dicts. Ignore underscore-prefixed entries when rendering reward overlays.

## Practical Guidance
- For new training work, start from:
  - `scripts/run_mat_nop_wall.py`
  - `scripts/run_mat_nop_wall_mjw.py`
  - `scripts/run_mat_nop_payload*.py`
  - `scripts/run_mat_nop_move*.py`
- Scenario cadence is scenario-owned on both backends. Do not pass or stash a separate env-level `action_repeat`.
- MJW live recording records exact live MJW episodes by copying selected world state into CPU MuJoCo render slots. Do not try to replay MJW actions in plain `mj_env`.
- Recording camera defaults should stay scenario-owned.
- CPU wall training should use `WorkerPoolAsyncVectorEnv(..., copy=False)`.
- `WorkerPoolAsyncVectorEnv` defaults to `check_spaces=False` and supports `env_clone_group_keys`; do not clone envs whose constructors differ in per-env state like `first_episode_length`.
- `SwarmBotsEnv.reset(seed=...)` must reseed `scenario.rng`, because reset sampling uses `scenario.rng`, not Gymnasium's `env.np_random`.
- On Windows, worker processes re-import the script's top-level module. Keep heavy PPO/torch/MJW imports out of top level unless env construction really needs them.
- If you touch wrappers or vector-env behavior, re-check SAME_STEP `final_obs` handling and checkpoint restore.

## Version Note
- Real target is Python `3.13` (`pyproject.toml` / `.python-version`), even if `AGENTS.md` still mentions `3.11`.
- `pyproject.toml` uses `uv` package sources for PyTorch CUDA wheels. Plain `pip install .` does not honor that.
- On headless Linux, `swarmbots.mj_env` and `swarmbots.mjw_env` default `MUJOCO_GL=egl` when unset and no display server is present.
