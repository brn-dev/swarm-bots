# Changelog

## Unreleased

- Added `mappo_mlp` and `mappo_mlp_small` with flattened MLP critics; `mappo` and `mappo_small` retain Deep Set critics. Both critic families support shared-encoder NOP and PopArt.
- Checkpoint recordings preserve configured DDPG/TD3 exploration noise; Python and CLI overrides support older checkpoints.
- Added cooperative MADDPG, MATD3, and MASAC baselines with large flattened MLP or masked Deep Set critics, plus TMATD3 and its decentralized-actor variant.
- Added deterministic target actors, bounded exploration and target smoothing, persistent TD3 update delays, and deterministic-policy support in benchmark evaluation.
- Added optional actor and critic NOP losses for Deep Set and transformer off-policy baselines, with multi-step replay and checkpoint continuation; flattened off-policy MLP critics reject NOP.
- Enabled critic NOP by default for the Deep Set and transformer off-policy baselines; marked their MLP variants as (no NOP) in the learning guide.
- Documented the baseline architectures and team-objective adaptation, with DDPG, TD3, MADDPG, MATD3, and Stable-Baselines3 citations and implementation references.

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
