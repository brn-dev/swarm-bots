# Changelog

## Unreleased

These changes describe the current source checkout relative to the published `0.1.0a3` release.

### Benchmark tasks and evaluation

- Make wall forward shaping continuous across the approach and far face. Lowering after clearing the wall preserves earned forward progress.
- Add `SwarmBots-WallMixed-v0` with equally likely 0.3 m and 0.4 m walls sampled at episode reset. The sampled height controls collision geometry and shaping, and is available to the actor as a public observation.
- Add `SwarmBots-PayloadGoal-v0` for single-payload delivery to a sampled goal. PayloadGoal and PayloadGoalSequence support optional smooth pseudo-Huber approach shaping over all active agents; the [scenario catalog](docs/scenarios.md) describes their observations, rewards, and success conditions.
- Add `SwarmBots-PayloadGoalSequence-v0` for ordered delivery with an easy first leg, randomized arrangement heading, and precise held goals. Optional payload-to-target displacement, payload XY velocity, an inside-goal flag, and bounded hold-progress rewards support learning; episode metrics report partial completion. See the [scenario catalog](docs/scenarios.md) for observation and reward options.
- Update connector placement, initial assembly grounding, and reset settling. Together with the reward and success-condition changes below, these changes require fresh evaluations before comparing scores with earlier revisions.
- Remove the isolation penalty from all scenarios. Rewards measure task progress and outcomes; connection topology affects behavior through physics.
- Require Bridge success above the destination platform and PayloadStep success over the finite step at the required height. Restrict PayloadStep climb shaping to the platform width.
- Improve collision clearance for payload resets and sample non-overlapping MultiPayloadGoal assignments. Batched payload sampling changes exact trajectories for existing seeds.
- Rename `SwarmBots-MoveTo-v0` to `SwarmBots-MoveToGoal-v0`.
- Add the active-agent count as the first privileged global observation in every task. This changes the critic observation layout, including for checkpoints whose tensor shapes still match.
- Add optional `NEXT_STEP` autoreset for external learners; the default and included learners retain `SAME_STEP`.
- Improve seeded partial resets, asynchronous settling, morphology-pool updates, and simulator behavior on the caller's CUDA stream.

### Baseline learning algorithms

- Allocate PPO episode storage and recurrent world-model windows only when the policy uses them. Recurrent PPO batches uninterrupted rollout segments without an intermediate copy and honors policy-defined world-model samplers.
- Extend default recurrent replay exploration through the first training sequence. Measure actor gradient norms on actor updates and preserve frozen critic parameters during those updates.
- Evenly stagger initial training episode endings to smooth reset workloads. This changes training trajectories for existing seeds; evaluation keeps full episodes.
- Add cooperative MADDPG, MATD3, MASAC, and transformer TD3 baselines, including recurrent TD3 actors and optional recurrent critics.
- Make `mappo__deepset_critic` the default training preset, with a local actor and centralized Deep Set critic. Standardize preset modifiers with double underscores; see the [current preset names](docs/learning.md#select-a-variant-and-task).
- Add fixed model-size tiers: `2.5M`, `5M-2L`, `5M-3L`, `10M-3L`, `10M-4L`, and `10M-5L`, each with an auxiliary-prediction budget. The default is `5M-2L NOP1M`; `custom` selects explicit widths in CLI tools.
- Align encoder, action-head, and critic budgets across baseline families. Add independent MAT actor/critic encoders, matched-budget depth choices, recurrent memory, and gated Deep Set critic controls. The [model reference](docs/policy_parameters/model_scale.md) explains controlled comparisons and remaining differences.
- Enable critic-side next-observation prediction (NOP) for public scalars and rotations in the core MAT, MAPPO, and off-policy presets. PPO measures scalar changes in the prediction origin's normalization frame, including targets drawn from stacked history. All presets exclude inactive payloads and undefined connector twists from losses.
- Set registered presets' discount to `0.995` and align off-policy optimization, warmup, replay, and auxiliary-loss defaults. SAC, DDPG, and TD3 retain their distinct objectives and update schedules.
- Resolve the default action distribution, optimizer, and replay settings from the selected policy implementation when overriding a preset. Custom-width recurrent MAT actors now apply their actor-specific encoder settings.
- Apply small scheduled learning-rate changes and reach the configured endpoint exactly, including PPO's cosine warmup.
- Derive PPO minibatches and recurrent sequence lengths from the effective collection horizon. Training metadata records the actual number of vector steps for PPO and off-policy presets.
- Track centered PopArt variance in double precision, validate the initial scale and mixing weight, and reject invalid targets or overflowing readout updates before changing the critic. This changes the PPO PopArt checkpoint format and requires a fresh checkpoint.
- Replace the off-policy constructor's `gradient_steps` argument with `replay_exposure`, defaulting to eight scored replay uses per new transition. Collection and batch settings must produce a positive whole update count; the default workload uses 1,024 parallel worlds and 1,024 learning positions per batch.
- Keep raw observations and rewards in replay while normalization statistics adapt throughout training. Normalize sampled transitions and recurrent histories with current statistics; evaluation uses frozen statistics.
- Simplify PPO to fixed-step collection and Monte Carlo entropy regularization. Recurrent and feed-forward PPO now normalize advantages identically for the same valid transitions.
- Remove whole-episode collection, SPR, virtual minibatches, distribution-specific entropy controls, and stochastic core-policy activations that invalidate rollout likelihoods.

### Research workflows

- Add standalone PyTorch PPO training independent of `swarmbots.learn`, plus train/evaluate/record examples for the included baselines and recurrent TD3. Preset training examples default to ten million transitions and 1,024 parallel worlds, with separate output directories for each run.
- Expand run and checkpoint metadata with training settings, scheduler/controller configuration, and source provenance for reproducibility. Add saved-checkpoint five-seed reporting and clearer checkpoint-continuation examples.
- Add compressed-log plotting, method comparisons across training seeds, and independent console/persistent logging profiles.
- Add installed policy-inspection, architecture-diagram, and plotting commands. Publish an [online architecture viewer](https://brn-dev.github.io/swarm-bots/policy-diagrams/); local tools generate current parameter counts and model diagrams.
- Focus documentation on research goals, benchmark capabilities, custom-learner integration, baseline training, and evaluation. Consolidate release notes around changes relevant to benchmark users.

### Reliability and compatibility

- Require Pillow 13 or newer for its security fixes. Record videos directly through imageio-ffmpeg, removing MoviePy's conflicting Pillow version bound.
- Improve numerical stability for bounded-action likelihoods, gradient clipping, recurrent memory, empty attention inputs, and inactive or padded observations and actions. Correct auxiliary-loss masking and single/twin-critic loss scaling.
- Improve compiled-policy action and recurrent-state handling, time-limit bootstrapping, and evaluation normalization. Skip unnecessary SAC actor sampling and actor gradients on delayed updates while preserving shared critic gradients. Reuse QCX autoregressive projections with deterministic activations.
- Count only newly reset lanes when updating observation statistics after an explicit partial reset. Preserve custom observation transforms during partial resets and benchmark evaluation. Recurrent SAC and TD3 reserve terminal-observation bootstraps for time limits where the episode did not also terminate.
- Preserve terminal observations in environment order when every lane finishes and a caller supplies the completed lane indices in a different order. Reject inconsistent completed-lane indices before changing replay and reject terminal observations with mismatched dimensions.
- Benchmark evaluation uses frozen observation statistics on the actor's device, including after training with CPU rollouts and a CUDA learner. On-policy evaluation also preserves custom observation hooks on isolated wrapper copies.
- Preserve moving averages, rollout warmup status, and best-model eligibility across continued training. Best-checkpoint retries preserve candidates with unordered module state, and short consecutive training calls keep separate best-model rotations.
- Count completed PPO and off-policy updates as they happen so a later minibatch failure preserves the update schedule on continuation. Scheduled changes to PPO's discount and GAE lambda now take effect in the next rollout.
- Honor custom PPO agent reductions and recurrent SAC shared-encoder samplers, including on delayed actor updates.
- Pool PPO KL and clipping diagnostics over valid likelihood ratios and count ratios strictly outside the clipping bounds for the learning-rate controller. KL early stopping handles small and large changes in joint log likelihood and skips updates when the estimated divergence is nonfinite.
- Report PPO critic explained variance consistently across reward scales, including low-variance returns. Compute joint likelihood-ratio summaries in double precision for multi-agent updates.
- Restore custom PyTorch module state, including lazy parameters and buffers, and validate Adam moments and normalization before loading them into the trainer. SAC checks its automatic entropy coefficient at initialization and restore.
- Atomic replacement preserves POSIX file permissions and accepts long filenames for checkpoints, quick-inspection JSON sidecars, expanded metric CSVs, and recordings.
- Make stochastic benchmark evaluation and recording reproducible from their seed while restoring the caller's CPU and CUDA RNG state, including pending CUDA seeds. Observable wall heights recover on reset after physics failures.
- Seed every CUDA device used for training rollouts, optimizer updates, or replay sampling. Benchmark evaluation, recording, and frozen training evaluations support running the actor and simulator on different devices.
- Preserve the trainer's gSDE buffers and custom exploration state across frozen evaluations, including when using PyTorch inference mode.
- Fix signed-Beta mode selection, improve likelihood accuracy for concentrated Beta components, and correct bounded Gaussian PPO likelihoods near saturation, including under autocast. Signed-Beta policies now draw negative magnitudes directly, and individual draws for existing seeds can change in both families.
- Enable plain and signed-Beta action sampling under CPU FP16 and BF16 autocast.
- Stabilize narrow Gaussian gradients in PPO and SAC, including under FP16 autocast, and score built-in hybrid heads in the emitted action precision. Joint sampling preserves custom action transforms, scoring hooks, and the likelihood range of replacement actions.
- Preserve rollout action, score, and conditioning values when PPO storage widens. PPO scores each vector step in its emitted action precision, and recurrent PPO preserves previous-action values at rollout boundaries.
- PPO rebases reduced-precision scalar targets in wider arithmetic and preserves the normalization frame of stacked observations when running statistics are frozen or resumed.
- Align Gumbel signed-Beta SAC actor and target sampling. Match gSDE sampling to its variance floor and resample each selected lane once.
- Restore learning rates, schedulers, optimizer parameter ownership, and SAC entropy temperature when continuing checkpoints. PPO optimizer restoration rejects mismatched scheduler and automatic learning-rate controller configurations. Off-policy checkpoints taken before the first optimizer update restart random exploration to fill their new replay buffer.
- Full checkpoint loads require the saved adjustable policy loss weights and validate policy parameters, including FP8 weights, before updating the trainer. Resuming restores the auxiliary prediction coefficients even when optimizer state is omitted.
- Preserve gSDE reset schedules across training, evaluation, and recording; add recording overrides and selective CUDA noise resets with a Torch fallback. Selective sampling changes exact seeded CUDA trajectories.
- Record complete episodes and report recurrent episode starts on reset observations for both benchmark autoreset modes. Frozen evaluations wait for successful video encoding and keep interrupted milestones eligible for retry.
- Support `NEXT_STEP` autoreset when recording a learning checkpoint.
- Frozen evaluation reports the latest milestone reached by each rollout. CPU-only evaluations preserve pending CUDA seeds without initializing CUDA.
- Hook failures preserve completed iterations' logs and scheduled checkpoints. Retrying `learn` with the same hooks can finish publishing an already completed evaluation.
- Finish pending simulator steps when collection is interrupted so `learn` can retry from fresh episodes. PPO discards the incomplete rollout, and off-policy learners refill replay to restore observation and history continuity.
- Evaluation metrics aggregate with buffered training metrics. CSV schemas can grow while a Windows reader holds the file open.
- Require positive integer timestep budgets and preserve task-specific episode counters in mixed-scenario training and frozen evaluation reports.
- Correct episode durations after autoreset, episode-weighted success rates across logging windows, compressed-log continuation, and plot exports.
- Frozen evaluations retain returns, lengths, and success flags supplied as NumPy scalars and batch the transfer of episode statistics from CUDA. Training and evaluation also accept bfloat16 and float8 vector statistics.
- Require current checkpoint structures and matching task, observation layout, model settings, and normalization. Remove historical migrations, aliases, and obsolete presets/options. Current fixed layouts use `v1`; that label alone does not establish compatibility with earlier development checkpoints. Preserve compiled/uncompiled weight-key alignment.

## 0.1.0a3 - 2026-10-05

- Added direct references for PPO, MAPPO, MAT, and SAC, with the published thesis credited as the source of the SwarmBots TMASAC design and MAAC, SACHA, MDAC, and MATRS attributed as related prior work.
- Added a reference guide and reusable BibTeX entries for learning components and the main scientific-software dependencies; linked the guide from the README, learning documentation, and citation metadata, and included the bibliography in source distributions.
- Aligned task-family ordering between the README and scenario catalog, placing FindOpening's partial-observability exploration alongside the PO-wall tasks.

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
