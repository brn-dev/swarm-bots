# Agent Notes

Keep only durable architecture notes and costly gotchas. Delete stale implementation history.

## Codebase Map

- `swarmbots/mj_env` is the CPU MuJoCo path, `swarmbots/mjw_env` is the batched GPU MJWarp path, and `swarmbots/learn` contains algorithms, rollout/replay, wrappers, action distributions, checkpointing, logging, and world-model wrappers.
- `scripts/` contains examples and entry points. Shared MJW experiment setup is in `experiments/mjw_experiment_common.py`; shared external-benchmark setup is in `experiments/external_benchmarks/benchmark_experiment_common.py`. The thesis MAPPO baseline deliberately uses the MAT-independent policy with agent attention disabled so its MLP structure matches the MAT comparisons.
- Standard TMASAC architecture variants are centralized in `experiments/tmasac_experiment_common.py`; scenario-specific TMASAC suites should only bind their scenario name/kwargs and run name. `tmasac_shared_encoder` uses a two-layer width-256 shared MAT followed by one-layer actor and critic encoders. `slstm_shared_encoder` puts the sLSTM memory in the analogous shared RMAT, keeps both downstream encoders feed-forward and one layer deep, and does not pass a separate actor-state summary to the critic. Shared encoders are critic-owned, actor inputs are detached, and the current learning-sequence shared forward is reused by the actor, critic, and shared-latent NOP. Keep the compiled shared encoder and downstream actor in separate AOTAutograd graphs: detach alone does not make two backwards through outputs of one compiled graph independent. Their NOP predicts from the shared latent with the first action included in the transition rollout. Thesis algorithm comparisons and their action-distribution/NOP choices are centralized in `experiments/thesis_experiment_common.py`; use `run_thesis_ppo_experiment(...)` for PPO ablations that only override rollout geometry. `experiments/thesis_plot_common.py` keeps ablations out of the main comparison and defines baseline-vs-ablation pair plots with the shared 100M-step cutoff. Thesis final-metric JSON scripts use `experiments/thesis_result_summary.py` and the same group sources and cutoff as their plot scripts.
- `experiments/evaluate_thesis_mjw_unseen_morphologies.py` evaluates the final PO-wall TMASAC and find-opening sLSTM-TMASAC checkpoints on fully pre-connected, unseen topology pools. Its default checkpoint discovery mirrors both the thesis and matching legacy plot sources; topology seeds start at 1,000,000 and therefore do not overlap the training preset's 42,000--42,049 pool. It strictly aligns legacy `torch.compile` checkpoint keys containing `_orig_mod` with current policy state-dict keys before loading, and reports progress by completed episode rather than environment step.
- Thesis final-metric JSON summarizes the first per-run timestep at which success EMA reaches at least 25%, 50%, 75%, 90%, 95%, and 98%. Each threshold reports the number of runs that reached it plus the population mean and standard deviation of their crossing timesteps; thresholds not reached within the plot cutoff receive `null` statistics.
- `.ref` contains reference implementations for xLSTM, Multi-Agent Transformer, and Stable Baselines3.

## Autoreset And Observation Wrappers

- Vector environments and learning wrappers use Gymnasium `AutoresetMode.SAME_STEP`. Read `gymnasium_autoreset.md` before changing reset or rollout code.
- On a done step, the returned observation is the reset observation. The terminal observation is `infos["final_obs"]`, selected by `infos["_final_obs"]`; terminal statistics may be in `infos["final_info"]`.
- Observation wrappers must transform `final_obs` like ordinary observations, but normalization RMS statistics must not be updated from terminal observations.
- `TorchShuffleAgentsWrapper` transforms terminal observations with the old permutation and reset observations with the new one. `TorchTransitionObsWrapper` retains terminal transition features in `final_obs` and zeros them in reset observations.
- Environment observation buffers are not durable. Async/shared-memory/GPU envs and `copy=False` conversions can mutate them after the next environment call.
- Learn observations require `local_obs`, `global_obs`, `hidden_local_vars`, and `hidden_global_vars`; `agent_mask` is optional. Layout changes must also update `build_obs_indices(...)` and global-scalar world-model targets.
- Standard wrapper order is learn conversion, optional agent shuffle, episode statistics, progress statistics, observation normalization, optional transition observations, then reward normalization. Skip reward normalization when PopArt owns it.

## Rollout, Replay, And SAC

- Rollout chunks may begin mid-episode. Use `PPOEpisodeSegment.is_true_episode_start`; never infer episode starts from chunk position.
- Recurrent state lives in `PPORolloutState`. SAME_STEP terminal bootstrapping evaluates `final_obs` without committing that recurrent state; the reset observation consumes the done mask on the next rollout step.
- Recurrent MAT/RMAT PPO uses env-major TBPTT rows with one initial recurrent state per row. Padding and world-model losses use the time mask.
- `OffPolicyReplayBuffer` is a per-environment ring stream with `capacity + 1` observation slots. Returned reset observations remain in the stream; terminal `final_obs` values live in a compact pool referenced by index.
- Continue collection into a nonempty replay buffer with the returned `OffPolicyRolloutState`. Resetting the environment into that buffer without its state breaks observation-slot continuity. During random recurrent collection, still advance the recurrent policy state while discarding its actions.
- Replay batches store physical environment actions. `OffPolicyReplayBatch` has no `next_previous_actions`; use sampled `actions` when conditioning on next observations.
- Recurrent replay uses compact temporal-state checkpoints and cached candidate starts. Ordinary episode segments do not cross episode boundaries; recurrent SAC streams may do so and reset recurrence from `episode_start_mask`. Burn-in is context-only.
- `SAC` supports feed-forward policies; `RecurrentSAC` handles recurrent and segment policies. SAC checkpoints deliberately omit replay and rollout state, so loading clears both and refills replay before updates resume while preserving global counters and optimizer schedules.
- SAC actors require action distributions registered as pathwise or straight-through. Rollout and target-Q sampling use `sample()`; actor updates use `rsample()` through `get_actions_with_log_probs(..., use_rsample=True)`.
- SAC entropy log-probabilities and targets are per-agent averages. Action-distribution auxiliary losses remain separate and are summed per replay row. Continuous SAC rejects non-`Box` action spaces.
- gSDE stores standard-normal temporal noise and applies the current standard deviation at sampling time. Preserve `OffPolicyRolloutState.gsde_noise_state` across training and use `resolve_gsde_reset_mode(...)` for reset-mode validation.

## Policies, Recurrent Models, And World Models

- `MATQCBasePolicy` owns shared QCS/QCX/QCC PPO plumbing. QCC subclasses QCS with its own decoder; QCX is query-to-context only; `MATIndPolicy` is the decoderless encoder-actor variant. `MATDecPolicy` has separate critic and actor MAT encoders; the actor encoder uses local and global observations but has no agent attention, so it cannot mix other agents' local tokens and `act()` never runs the centralized critic encoder.
- `MATOrigPolicy` requires a contiguous active-agent prefix and shifted previous-agent actions. QCS/QCC/QCX support arbitrary inactive positions if every row has an active agent; `assume_agent_mask_is_active_prefix=True` selects the cheaper prefix path.
- Recurrent temporal cores must honor reset masks. Reset state with overwrite/selection, not multiplication by zero, because multiplication does not clear NaN or Inf values.
- `ActionDist` action heads are always `nn.Linear`, even when input and output widths match, so initialization gain is honored. Add continuous action-distribution construction and actor-gradient capabilities to the registry in `hybrid_action_dist.py`, not policy-specific type lists.
- Mutable action-distribution scalars must be tensors or buffers. Sticky distributions must keep `requires_previous_actions()` structurally stable even when their effect anneals to zero.
- `TernarySignMagnitudeBetaActionDist` is a mixed distribution over a negative Beta branch, a point mass at zero, and a positive Beta branch. Its rollout sampler is exact categorical; its SAC sampler uses hard straight-through Gumbel selection, and `log_prob(0)` is the zero component's categorical log probability.
- Existing sign-magnitude distributions use the legacy negative-branch transform `action = -1 + U`, with `log_prob` evaluating `U = action + 1`. Despite the magnitude naming, changing this to `action = -U` changes existing policies and requires an explicit migration.
- TMASAC distribution-specific grids live under each suite's `scripts/<distribution>/` directory; their local `common.py` adapter pins the distribution while parent common code remains the source of experiment settings. Non-default distributions are appended to `variant_name` to prevent run-directory collisions.
- `NextObsPredWrapper` and `SPRWrapper` are policy wrappers and must delegate sampler selection, temporal-state APIs, and `after_optimizer_step()`. World-model losses use `wm_actions`, not PPO current-step actions. `SPRWrapper` does not support RMAT.
- Global next-observation prediction must use explicit scalar/rot6d target indices, pool only valid active-agent latents, and exclude rows without valid next-active agents.

## Scenarios And MJW

- `SwarmBotsEnv` delegates cadence, reset baselines, and progress deltas to scenarios. Scenario-specific MJW behavior belongs in runtime classes and flows through `scenario.build_runtime_metadata(...)`; keep `MJWModelMetadata` generic.
- Scenario observation layouts are contracts. Changing them requires corresponding normalization, policy-adapter, replay, and world-model target updates. `SwarmBotsEnv.reset(seed=...)` must reseed `scenario.rng`.
- `MultiScenarioVectorEnv` combines fixed groups of homogeneous SAME_STEP vector environments. It requires matching local-observation, agent-mask, and action spaces; pads other observation fields; and carries `scenario_id` through normal, terminal, replay, recurrent, and successor paths.
- External cooperative adapters live in `swarmbots/external_benchmark_envs`. MaMuJoCo is adapted from PettingZoo; VMAS retains batched Torch execution with explicit per-lane SAME_STEP resets. Both expose actor observations as `local_obs`, centralized state as `hidden_global_vars`, and average rewards across agents.
- `MJWSwarmBotsVectorEnv` is a Gymnasium `VectorEnv` backed by one MJWarp model and batched GPU data. Its reset modes are direct, one-shot settled reset, and background CPU-settled snapshot buffering.
- Keep MJW hot paths fixed-shape, kernelized, and allocation-light. CUDA connector bookkeeping must not reintroduce `nonzero()` or variable-length indexing. Reuse GPU scratch buffers and keep Warp indices `int32` unless Torch indexing requires `int64`.
- Non-finite MJW `qpos` or `qvel` terminates and resets that world, masks its terminal observation on-device, and emits one Discord warning per run.
- Fixed-shape MJW tensor operations and transition-observation masking compile by default on CUDA and remain eager on CPU. Keep Warp launches and variable-cardinality bookkeeping outside compiled graphs.
- Render overlays are visual only and are added with `BaseScenario.add_render_geoms(scene)` after `Renderer.update_scene()`. Live MJW recording shares one `mujoco.Renderer`; resolutions above 640x480 require `ensure_mujoco_offscreen_framebuffer(...)` first.

## Runtime Conventions

- Put agent-created temporary files under repository-root `.tmp/`.
- Runtime hyperparameters are live attributes; mutating configuration dataclasses after initialization has no effect.
- Canonicalize devices with `as_device(...)`; PyTorch treats `cuda` and `cuda:0` as unequal, which can cause redundant transfers or optimizer replacement.
- MJ and MJW scenarios default to continuous connector actions. Legacy binary experiments must set `continuous_connector_actions=False` explicitly.
- Activation factories using `ParameterLearnMode.PER_FEATURE` require an explicit feature count. Generic paths should call `make_activation(...)`.
- Initialization gains are `1.0` for hidden/projection/transformer feed-forward layers and `0.01` for output/action/value/prediction heads.
- Current Python is 3.11, with a planned move to 3.13. Use the existing `.venv` directly for tests and linting as specified in `AGENTS.md`; use `uv` only to manage dependencies.
- Headless Linux defaults `MUJOCO_GL=egl` for `swarmbots.mj_env` and `swarmbots.mjw_env` when unset.
