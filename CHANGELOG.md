# Changelog

## Unreleased

These changes describe the current source checkout relative to the published `0.1.0a3` release.

### Benchmark tasks and evaluation

- Add `SwarmBots-WallMixed-v0` with sampled, observable wall heights, `SwarmBots-PayloadGoal-v0` for single-goal delivery, and `SwarmBots-PayloadGoalSequence-v0` for ordered delivery with goal holds. The [scenario catalog](docs/scenarios.md) describes their observations, rewards, and options.
- Make wall forward shaping continuous across approach, traversal, and descent. Remove isolation penalties from all scenarios so connection topology affects behavior through physics.
- Update connector placement, initial assembly grounding, reset settling, and payload collision clearance. These dynamics and reset changes require fresh evaluations before comparisons with earlier revisions.
- Require Bridge success above the destination platform and PayloadStep success over the finite step at the required height. Restrict PayloadStep climb shaping to the platform width.
- Rename `SwarmBots-MoveTo-v0` to `SwarmBots-MoveToGoal-v0` and prepend the active-agent count to privileged global observations. The changed observation layout can invalidate older checkpoints even when tensor shapes match.
- Add optional `NEXT_STEP` autoreset for external learners and checkpoint recording; included training retains `SAME_STEP`.

### Baseline learning algorithms

- Add cooperative MADDPG, MATD3, MASAC, and transformer TD3 baselines, including recurrent actors and optional recurrent critics.
- Make `mappo__deepset_critic` the default training preset and standardize preset modifiers with double underscores. See the [current preset names](docs/learning.md#select-a-variant-and-task).
- Add fixed model-size tiers from 2.5M to 10M, matched-budget encoder-depth choices, independent MAT actor/critic encoders, recurrent memory, and gated Deep Set controls. The [model reference](docs/policy_parameters/model_scale.md) explains their budgets and comparison limits.
- Enable critic-side next-observation prediction (NOP) in core MAT, MAPPO, and off-policy presets, including public task scalars and rotations. Correct prediction targets under changing normalization and exclude inactive payloads and undefined connector twists.
- Align off-policy optimization, replay, and auxiliary-loss defaults; set registered presets' discount to `0.995`. Replace the off-policy constructor's `gradient_steps` argument with `replay_exposure`; the [training guide](docs/learning.md#customize-training) covers collection and batch settings.
- Keep raw observations and rewards in replay with adaptive normalization; evaluation uses frozen statistics. Recurrent presets collect a full burn-in and learning sequence before switching from random exploration.
- Simplify PPO to fixed-step collection and Monte Carlo entropy regularization, with consistent advantage normalization across feed-forward and recurrent policies. Remove whole-episode collection, SPR, virtual minibatches, and distribution-specific entropy controls.
- Evenly stagger initial training episode endings and derive default PPO minibatches and recurrent sequences from the selected collection horizon. Policy overrides now select their matching action-distribution, optimizer, and replay defaults.

### Research workflows

- Add standalone PyTorch PPO training and train/evaluate/record examples for included baselines and recurrent TD3. Preset examples default to ten million transitions and 1,024 parallel worlds, with separate output directories per run.
- Expand training metadata with resolved settings, scheduler/controller configuration, and source provenance. Add saved-checkpoint five-seed reporting and continuation examples.
- Add compressed-log plotting, training-seed comparisons, configurable logging profiles, and installed policy-inspection and diagram tools. An [online architecture viewer](https://brn-dev.github.io/swarm-bots/policy-diagrams/) shows the current baseline layouts.
- Make seeded stochastic evaluation and recording reproducible, preserve training normalization and exploration state, and support actors and simulators on different devices. Use the [evaluation protocol](docs/benchmark_protocol.md) for comparable results.
- Refocus documentation on benchmark capabilities and research workflows, with detailed task and model references separate from introductory guides.

### Reliability and compatibility

- Improve action-distribution likelihoods, mixed-precision training, auxiliary losses, gradient clipping, and PPO diagnostics. Reduce unnecessary rollout storage and delayed-actor work while preserving compiled and recurrent policy behavior.
- Improve checkpoint continuation for optimizer schedules, entropy temperature, auxiliary-loss weights, moving averages, and warmup state. Replay is not checkpointed and must refill after loading.
- **Older PPO PopArt checkpoints require a fresh start** after the centered-variance format change. Other checkpoints also require matching task, observation layout, model, and normalization settings; the `v1` layout label alone does not establish compatibility with earlier development checkpoints.
- Improve interrupted training and evaluation recovery, complete-episode recording, terminal-observation handling, and custom-policy/wrapper support. Preserve completed updates, logs, and scheduled checkpoints across recoverable failures.
- Correct episode durations, success-rate aggregation, partial-task progress metrics, compressed-log continuation, and plot exports.
- Preserve gSDE reset schedules through evaluation, recording, and continuation, with recording overrides and selective CUDA noise resets. Sampling and reset corrections change exact seeded trajectories; compare runs from the same implementation.
- Replace MoviePy with imageio-ffmpeg for video recording and require Pillow 13 or newer.

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
