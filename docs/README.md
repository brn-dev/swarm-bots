# SwarmBots documentation

SwarmBots supports training and comparison of policies for robots that connect, move together, and separate during an episode. The benchmark works with included learners and custom learning algorithms.

These guides describe the current source checkout, including changes since the published `0.1.0a3` release. The [versioned release guides](https://github.com/brn-dev/swarm-bots/tree/v0.1.0a3/docs) cover that package's APIs, presets, and tools.

## Start here

1. [Install SwarmBots and run a smoke test](../README.md#install).
2. [Check GPU setup](gpu_setup.md) before CUDA training, evaluation, or recording.
3. [Train, evaluate, and record a first policy](../examples/README.md#train-evaluate-and-record-a-preset).
4. Open the run's `evaluation.json` and `videos/` to inspect the result, then [plot its training log](log_plotting.md).

The example evaluates one seed. Reportable benchmark scores follow the [full evaluation protocol](benchmark_protocol.md).

## Guides by topic

| Task | Guide |
| --- | --- |
| Tasks and success conditions | [Scenario catalog](scenarios.md) |
| Included learners and training settings | [Learning baselines](learning.md) |
| Runnable workflows and saved outputs | [Examples](../examples/README.md) |
| Checkpoint continuation | [Checkpoint continuation](learning.md#continue-a-checkpoint) |
| Custom policies and learning algorithms | [Python API](api.md) |
| Standalone PyTorch training | [Standalone PPO example](../examples/README.md#train-without-the-included-learning-framework) |
| Policy comparisons and score exports | [Benchmark protocol](benchmark_protocol.md), [five-seed report](api.md#five-seed-report) |
| MP4 recording | [Recording](recording.md) |
| Learning curves and training-seed comparisons | [Log plotting](log_plotting.md) |
| Task listing and installed utilities | [Command-line tools](tools.md) |
| CUDA and compilation setup | [GPU setup and troubleshooting](gpu_setup.md) |
| Citations and attribution | [References and attribution](references.md) |

The environment and evaluator work independently of the included learners. Custom algorithms can use SwarmBots without importing `swarmbots.learn`.

## Model reference

The [online diagram viewer](https://brn-dev.github.io/swarm-bots/policy-diagrams/), [parameter inspection guide](policy_parameters/README.md), [fixed model layouts](policy_parameters/model_scale.md), and [diagram reading guide](policy_parameters/diagrams/README.md) explain the dimensions and parameter counts behind the training presets. The online viewer requires no installation or local diagram data.

## Repository layout

| Path | Contents |
| --- | --- |
| `swarmbots/benchmark/` | Task registry, evaluation, recording, and the main command-line interface |
| `swarmbots/mjw_env/`, `swarmbots/scenario_presets/` | MuJoCo Warp simulation and task definitions |
| `swarmbots/learn/` | Included learners and training presets |
| `swarmbots/plotting/`, `swarmbots/tools/` | Log analysis, parameter inspection, and architecture diagrams |
| `examples/` | Runnable training, evaluation, and recording workflows |
| `docs/` | User guides and focused model references |
| `tests/` | Automated checks for the simulator, learners, and tools |

## Version status and help

**SwarmBots is alpha.** Wall, PO-wall, and FindOpening tasks are beta after extensive internal testing; other tasks are alpha. [Scenario maturity and versioning](scenarios.md#scenario-maturity-and-versioning) explains task readiness and comparisons across revisions.

[GitHub issues](https://github.com/brn-dev/swarm-bots/issues) and email to [Dominik Baron](mailto:dominik.b4ron@gmail.com) are the channels for bugs and feedback. A useful report includes the task ID, version or commit, configuration, seed, and observed behavior.
