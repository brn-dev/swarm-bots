# Changelog

## 0.1.0a1 - 2026-09-26

- Unified explicit and automatic resets to use scenario settling, including partial and repeated seeded resets. Removed the one-shot `settle_initial_reset` constructor argument; old unsettled-first-episode evaluation results need rerunning.
- Added default-settling and multi-episode lifecycle integration coverage, a `smoke --compiled` option, and GPU setup instructions.
- Published the standalone MJWarp benchmark environment and scenario suite.
- Added canonical benchmark IDs, a stable construction API, evaluation utilities, and a discovery/smoke-test CLI.
- Removed training algorithms, thesis experiment configurations, results, vendored research references, and non-MJW benchmark adapters.
- Fixed evaluation to accept exactly the first episode per lane and reject episode counts larger than the world count.
- Added evaluation settings/runtime metadata, canonical-run flags, and a five-seed JSON reporting example.
- Fixed stale MultiPayloadGoal observations after reset; expanded integration coverage to every registered task.
- Fixed CUDA graph setup to read the device from simulation data, allowing GPU environments to initialize.
- Gated publication on CI and installed-wheel checks; documented GPU validation, observation features, success conditions, and the fixed morphology population.
- Added wall traversal and opening exploration demonstrations.
