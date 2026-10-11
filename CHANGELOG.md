# Changelog

## Unreleased

- Scale payload-row clearance with sampled spawn coordinates and row span so float32 rounding cannot reintroduce capsule overlap far from the origin.
- Capture rollout action diagnostics before replay training replaces the cached distribution, preserving rollout entropy and standard-deviation values without extra policy evaluation or duplicated metrics work.
- Exclude padded timesteps and unused single-sample successor slots from recurrent SAC entropy and standard-deviation diagnostics across eager, compiled, shared-encoder, and segment sampling, including multiple action samples and observations without agent masks.
- Run CUDA simulation, connectors, resets, and deferred steps on the caller's active PyTorch stream; use Gymnasium's close lifecycle and reject negative scalar first-episode limits.
- Build padded simulator models independently of sampled assemblies, including populations with zero probability of a full swarm; resolve payload spawn spacing from actual shape footprints to prevent capsule penetration.
- Preserve training RNG during evaluation environment construction; retain scenario IDs and adapt global observation widths for scenario-conditioned benchmark policies, and refresh transition histories after normalized partial resets.
- Roll back recording state after first-render failures and close every scenario child before propagating cleanup errors; correct the documented continuous connector action contract.
- Remove whole-episode PPO collection and SPR, use fixed-step PPO collection by default, require zero policy dropout, and make disabling gSDE step resets retain episode-only resets.
- Remove distribution-specific entropy extras, coefficients, floors, magnitude weighting, commands, and Gaussian quadrature. PPO uses its importance-weighted Monte Carlo entropy term with default `mc_ent_coef=1e-3`; SAC retains its existing entropy objective and automatic temperature tuning. Keep entropy diagnostics outside the loss configuration, with underlying Gaussian entropy labeled `ent_normal`, exclude inactive agents and recurrent padding, and snapshot reusable environment masks.
- Keep rejected off-policy parameter commands atomic and shared NOP batch-size settings consistent; complete recurrent TD3 helper annotations.
- Retain NOP metrics as detached device tensors and snapshot reusable outputs, deferring CPU conversion to metric aggregation instead of synchronizing every update.
- Condense agent notes to navigation and costly gotchas, removing duplicated defaults and implementation history.
- Place connector frames on capsule surfaces by default, preserving robot geometry and mass properties; expose `legacy_connector_positioning=True` to restore the original frames and assembly spacing.
- Make asynchronous settled-reset sampling independent of worker completion timing, and retain agent-shuffle RNG state during repeated device setup.
- Compute PPO rollout likelihoods from stored bounded actions, matching training recomputation for saturated Gaussian and gSDE actions while retaining SAC's cached pre-tanh path.
- Reject undersized goal radii and nonpositive/nonfinite settling timestep scales; require Bridge success above the destination platform and sample collision-free MultiPayloadGoal assignments.
- These benchmark corrections change dynamics, initial populations and task outcomes; rerun evaluations rather than combine scores from before and after the changes.
- Refresh simulator body frames after each control step on CPU and CUDA; preserve physics options during reset settling and reject nonpositive episode limits.
- Require PayloadStep success to occur over the finite step, support scalar distributions for dual payload offsets, and bound truncated-normal rejection sampling.
- Support transition-observation histories in benchmark policy adapters; reject oversized PPO batches and correct the optional Monte Carlo entropy gradient.
- Finish evaluation episode capture before returning, clean up resources after video writer failures, and remove frame striding and its FPS mode options.
- Discard interrupted live-recording episodes after explicit resets while preserving the requested complete-episode count; fix recording filenames for fractional evaluation milestones.
- Invalidate ready and in-flight settled-reset snapshots when the active morphology pool changes, and reject pool changes while a simulator step is pending.
- Reject empty swarms in configuration, replay insertion, and SAC entropy reduction; propagate hybrid action-space seeds to all component spaces.
- Preserve custom PPO scheduler state, enabled flags, and current values in checkpoints; require matching scheduler names and order when restoring.
- Persist runtime changes to Predicted Std Gaussian's base standard deviation, while retaining constructor defaults for older checkpoints.
- Restore compressed metrics logs before appending on resume, preserving earlier rows and columns; retain singleton samples for deferred summary-statistic histograms.
- Preserve raw predictions when updating biasless PopArt statistics, and preserve NOP rotation-conversion gradients at zero rotation.
- Remove PPO's `virtual_mini_batches` option and commands; configure the sampler's actual batch size instead.
- Remove the NumPy observation-normalization and episode-statistics wrappers, worker-pool CPU environment, and separate-environment learning recording helper/fallback; use Torch wrappers and live vector recording.
- Package policy inspection, architecture diagrams, and log/experiment plotting under `swarmbots.tools`, with installed `swarmbots-inspect-policies`, `swarmbots-diagram-policies`, `swarmbots-plot-logs`, and `swarmbots-plot-experiments` commands and equivalent Python module entry points.
- Added offline policy architecture galleries with SVG, Mermaid, and Graphviz exports across model scales; retain complete parameter audits during partial diagram selections and support stacked SwiGLU layouts.
- Added fixed `2.5M NOP0.75M`, `5M NOP1M`, and `10M NOP2M` model-scale tiers, with regular presets defaulting to `5M NOP1M`; expose `list_model_scales()` and save resolved layouts and versions in run/checkpoint metadata. Use `model_scale=None` or CLI `--model-scale legacy` for explicit widths and older checkpoints.
- Replace `ppo_small`, `mappo_small`, and `mappo_mlp_small` preset names with model-scale selection; limit the default `list_variants()` and parameter sweep to 18 core presets while keeping optional presets available by name and listing them with `include_hidden=True`.
- Added policy parameter and layer-layout audits, compact snapshots, and model-scale comparisons; `trainable_minus_nop` excludes frozen targets and shared parameters are counted once.
- Added independent `full`, `minimal`, and `return_success` console and CSV/W&B logging levels, defaulting to `minimal` console output and `full` persistence, with custom console selections and retained scenario outcomes.
- Added `swarmbots.plotting` and the optional `plot` extra for compressed training-log reading, individual-run plots, and equal-seed mean/standard-deviation comparisons over finite sample ranges; sort and deduplicate samples before optional smoothing.
- Added runnable train/evaluate/record workflows for public presets and recurrent TD3; rename the random-policy example to `examples/evaluate_random_policy.py`.
- Correct recurrent TMASAC critics to retain their full feed-forward capacity instead of inheriting the recurrent actor's split budget.
- Added `mappo_mlp` with a flattened MLP critic; `mappo` retains a Deep Set critic. Both critic families support shared-encoder NOP and PopArt.
- Condition MAPPO and off-policy baseline Deep Set element encoders on global context while retaining direct context after pooling; `mappo_critic_context_in_elements=False` and `baseline_critic_context_in_elements=False` restore their previous checkpoint layouts.
- Added optional joint local/global observation embeddings for MAT/RMAT encoders and transformer critics via `mat_joint_obs_embedding=True`, retaining additive embeddings by default; reconstruct joint-embedding checkpoints with the same setting.
- Checkpoint recordings preserve configured DDPG/TD3 exploration noise; Python and CLI overrides support older checkpoints.
- Added cooperative MADDPG, MATD3, and MASAC baselines with large flattened MLP or masked Deep Set critics, plus TMATD3 and its decentralized-actor variant.
- Added deterministic target actors, bounded exploration and target smoothing, persistent TD3 update delays, and deterministic-policy support in benchmark evaluation.
- Added recurrent TD3 actors and optional recurrent TMATD3 critics, with sequence replay, burn-in, and separate online/target actor history states.
- Feed-forward critics for recurrent TD3 condition on detached actor state by default; use `td3_actor_state_critic_input_config=None` for checkpoints trained without that input.
- Expose `critic_independent_encoders` for TMASAC/TMATD3 transformer twin critics, which share an encoder by default; set it to `True` to reconstruct independent-encoder checkpoints.
- Consolidate SAC/DDPG/TD3 rollout, replay, NOP, and checkpoint helpers under a shared off-policy lifecycle, retaining historical SAC imports; add opt-in delayed SAC actor/temperature updates and target smoothing with unchanged SAC defaults.
- Added optional actor and critic NOP losses for Deep Set and transformer off-policy baselines, with multi-step replay and checkpoint continuation; flattened off-policy MLP critics reject NOP.
- Enabled critic NOP by default for the Deep Set and transformer off-policy baselines; marked their MLP variants as (no NOP) in the learning guide.
- Documented the baseline architectures and team-objective adaptation, with DDPG, TD3, MADDPG, MATD3, and Stable-Baselines3 citations and implementation references; add a standalone software citation in `CITATION.bib`.

## 0.1.0a3 - 2026-10-05

- Added direct references for PPO, MAPPO, MAT, and SAC, with the published thesis credited as the source of the SwarmBots TMASAC design and MAAC, SACHA, MDAC, and MATRS attributed as related prior work.
- Added a reference guide and reusable BibTeX entries for learning components and the main scientific-software dependencies; linked the guide from the README, learning documentation, and citation metadata, and included the bibliography in source distributions.
- Aligned task-family ordering between the README and scenario catalog, placing FindOpening's partial-observability exploration alongside the PO-wall tasks.
- Recorded the release procedure in internal agent notes.

## 0.1.0a2 - 2026-10-04

- Documented the benchmark's overall alpha status, scenario maturity criteria, and how maturity differs from task, package, and protocol versions; invited suggestions and research feedback.
- Added `SwarmBots-POWallHard-v0`, a randomized hidden 0.4 m wall with the medium PO-wall variant's observation and reward settings, bringing the full suite to 15 tasks.
- Marked all fixed-wall and PO-wall difficulty variants and FindOpening beta and all other registered scenarios alpha; exposed maturity through the registry, task discovery CLI, and evaluation metadata.
- Separated PO-wall's connected locomotion focus from FindOpening's partial observability and exploration within an episode in the task overview, scenario descriptions, and registry categories.
- Added an email contact for feedback and clarified CUDA setup for Windows projects that install the package from PyPI.

## 0.1.0a1 - 2026-09-26

- Added built-in PPO/SAC, MAT variants, TMASAC, recurrent policies, NOP, action distributions, and training infrastructure, with reusable task/variant training and benchmark evaluation helpers.
- Standardized learning preset names, removed obsolete query/context decoder implementations and duplicate aliases, and added `tmasac_dec` with a decentralized MLP actor and transformer critic. `mappo` now selects MAPPO; its former MAT-IND preset is `mat_ind_no_attention`.
- Kept action distributions independent of architecture preset names: on-policy defaults use Signed-Magnitude Beta and SAC defaults use `predicted_std_gaussian`. Renamed the Gaussian distribution module, classes, and selector to include `Gaussian` explicitly.
- Limited public action distributions to SMB, Gumbel SMB, Beta, Gaussian variants, gSDE, and categorical/Bernoulli support; removed experimental distributions and their sampling interfaces.
- Unified explicit and automatic resets to use scenario settling, including partial and repeated seeded resets. Removed the one-shot `settle_initial_reset` constructor argument; old unsettled-first-episode evaluation results need rerunning.
- Added default-settling and multi-episode lifecycle integration coverage, a `smoke --compiled` option, and GPU setup instructions.
- Published the standalone MJWarp benchmark environment and scenario suite.
- Added canonical benchmark IDs, a stable construction API, evaluation utilities, and a discovery/smoke-test CLI.
- Fixed evaluation to accept exactly the first episode per lane and reject episode counts larger than the world count.
- Added evaluation settings/runtime metadata, canonical-run flags, and a five-seed JSON reporting example.
- Fixed stale MultiPayloadGoal observations after reset; expanded integration coverage to every registered task.
- Fixed CUDA graph setup to read the device from simulation data, allowing GPU environments to initialize.
- Gated publication on CI and installed-wheel checks; documented GPU validation, observation features, success conditions, and the fixed morphology population.
- Added wall traversal and opening exploration demonstrations.
