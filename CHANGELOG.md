# Changelog

## Unreleased

These changes describe the current source checkout relative to the published `0.1.0a3` release.

### Benchmark tasks and evaluation

- Update connector placement, initial assembly grounding, and reset settling. Together with the reward and success-condition changes below, these changes require fresh evaluations before comparing scores with earlier revisions.
- Remove the isolation penalty from all scenarios. Rewards measure task progress and outcomes; connection topology affects behavior through physics.
- Require Bridge success above the destination platform and PayloadStep success over the finite step at the required height. Restrict PayloadStep climb shaping to the platform width.
- Improve collision clearance for payload resets and sample non-overlapping MultiPayloadGoal assignments. Batched payload sampling changes exact trajectories for existing seeds.
- Rename `SwarmBots-MoveTo-v0` to `SwarmBots-MoveToGoal-v0`.
- Add the active-agent count as the first privileged global observation in every task. This changes the critic observation layout, including for checkpoints whose tensor shapes still match.
- Add optional `NEXT_STEP` autoreset for external learners; the default and included learners retain `SAME_STEP`.
- Improve seeded partial resets, asynchronous settling, morphology-pool updates, and simulator behavior on the caller's CUDA stream.

### Baseline learning algorithms

- Add cooperative MADDPG, MATD3, MASAC, and transformer TD3 baselines, including recurrent TD3 actors and optional recurrent critics.
- Make `mappo__deepset_critic` the default training preset, with a local actor and centralized Deep Set critic. Standardize preset modifiers with double underscores; see the [current preset names](docs/learning.md#select-a-variant-and-task).
- Add fixed model-size tiers: `2.5M`, `5M-2L`, `5M-3L`, `10M-3L`, `10M-4L`, and `10M-5L`, each with an auxiliary-prediction budget. The default is `5M-2L NOP1M`; `custom` selects explicit widths in CLI tools.
- Align encoder, action-head, and critic budgets across baseline families. Add independent MAT actor/critic encoders, matched-budget depth choices, recurrent memory, and gated Deep Set critic controls. The [model reference](docs/policy_parameters/model_scale.md) explains controlled comparisons and remaining differences.
- Enable critic-side next-observation prediction (NOP) in the core MAT, MAPPO, and off-policy presets. Include public task scalars and rotations, predict per-step changes, and exclude inactive payloads and undefined connector twists from losses.
- Set registered presets' discount to `0.995` and align off-policy optimization, warmup, replay, and auxiliary-loss defaults. SAC, DDPG, and TD3 retain their distinct objectives and update schedules.
- Replace the off-policy constructor's `gradient_steps` argument with `replay_exposure`, defaulting to eight scored replay uses per new transition. Collection and batch settings must produce a positive whole update count; the default workload uses 1,024 parallel worlds and 1,024 learning positions per batch.
- Keep raw observations and rewards in replay while normalization statistics adapt throughout training. Normalize sampled transitions and recurrent histories with current statistics; evaluation uses frozen statistics.
- Simplify PPO to fixed-step collection and Monte Carlo entropy regularization. Remove whole-episode collection, SPR, virtual minibatches, distribution-specific entropy controls, and stochastic core-policy activations that invalidate rollout likelihoods.

### Research workflows

- Add standalone PyTorch PPO training independent of `swarmbots.learn`, plus train/evaluate/record examples for the included baselines and recurrent TD3. Preset training examples default to ten million transitions and 1,024 parallel worlds, with separate output directories for each run.
- Record training seeds, construction options, resolved settings, and source provenance before updates. Include scheduler/controller configurations, dataclass callback fields, bound receivers, partial arguments, and standard operator-helper settings in run/checkpoint metadata, including bound `.__call__` methods and inherited classmethods. Preserve dictionary key types, callable data contents, and enum callback/argument identity; preserve metadata safety and isolation across parallel and nested collections. Describe opaque runtime arguments and cyclic back-references by type without retaining live objects in checkpoints. Add saved-checkpoint five-seed reporting and clearer checkpoint-continuation examples.
- Add compressed-log plotting, method comparisons across training seeds, and independent console/persistent logging profiles.
- Add installed policy-inspection, architecture-diagram, and plotting commands. Publish an [online architecture viewer](https://brn-dev.github.io/swarm-bots/policy-diagrams/); local tools generate current parameter counts and model diagrams.
- Focus documentation on research goals, benchmark capabilities, custom-learner integration, baseline training, and evaluation. Consolidate release notes around changes relevant to benchmark users.

### Reliability and compatibility

- Improve numerical stability for bounded-action likelihoods, gradient clipping, recurrent memory, empty attention inputs, and inactive or padded observations. Correct auxiliary-loss masking and single/twin-critic loss scaling.
- Improve compiled-policy action and recurrent-state handling, time-limit bootstrapping, and evaluation normalization. Reduce unnecessary SAC actor sampling on delayed updates and reuse QCX autoregressive projections with deterministic activations.
- Restore learning rates, schedulers, optimizer parameter ownership, and SAC entropy temperature consistently when continuing checkpoints, including weights-only temperature restoration. Replay contents are not checkpointed.
- Preserve gSDE reset schedules across training, evaluation, and recording; add recording overrides and selective CUDA noise resets with a Torch fallback. Selective sampling changes exact seeded CUDA trajectories.
- Improve recording cleanup, complete-episode capture, and recurrent-state resets. Correct metric aggregation, mixed-scenario success rates, compressed-log continuation, and plot exports.
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
