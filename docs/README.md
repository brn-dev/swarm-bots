# SwarmBots documentation

SwarmBots is a multi-agent reinforcement learning benchmark for self-assembling modular robots. Its tasks study cooperative locomotion, exploration with partial observations, and payload transport while policies control both movement and physical connections.

These guides describe the current source checkout, including changes since the published `0.1.0a3` release. The [versioned release guides](https://github.com/brn-dev/swarm-bots/tree/v0.1.0a3/docs) cover that package's APIs, presets, and tools.

## Using the benchmark

1. Choose a research task from the [scenario catalog](scenarios.md), including its observations, success conditions, and maturity.
2. [Install SwarmBots and run a smoke test](../README.md#install), with [GPU setup](gpu_setup.md) for CUDA runs.
3. Integrate a method through the [environment and policy API](api.md), or [train an included baseline](../examples/README.md#train-evaluate-and-record-a-preset).
4. Evaluate with the [benchmark protocol](benchmark_protocol.md), inspect [videos](recording.md), and compare [learning curves](log_plotting.md).

The environment and evaluator work independently of `swarmbots.learn`. The [standalone PyTorch PPO example](../examples/README.md#train-without-the-included-learning-framework) provides a complete custom-learner integration. Included training examples produce checkpoints, logs, evaluation JSON, and videos; their one-seed evaluation can be extended to a [five-seed benchmark report](learning.md#five-seed-report-from-a-saved-checkpoint).

## Guides by topic

| Task | Guide |
| --- | --- |
| Tasks and success conditions | [Scenario catalog](scenarios.md) |
| Baseline RL algorithms and training settings | [Learning baselines](learning.md) |
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

## Model reference

The baseline suite includes PPO, MAPPO, MAT, SAC, DDPG, and TD3-based methods, with communication, recurrence, and auxiliary-prediction controls. The [learning guide](learning.md) explains their use and adaptations. The [experimental comparison reference](policy_parameters/model_scale.md#experimental-comparisons) identifies controlled pairs and differences that affect interpretation.

The [online diagram viewer](https://brn-dev.github.io/swarm-bots/policy-diagrams/), [parameter inspection guide](policy_parameters/README.md), [fixed model layouts](policy_parameters/model_scale.md), and [diagram reading guide](policy_parameters/diagrams/README.md) explain the dimensions and parameter counts behind the training presets. The online viewer requires no installation or local diagram data.

## Version status and help

**SwarmBots is alpha.** Wall, PO-wall, and FindOpening tasks are beta after extensive internal testing; other tasks are alpha. [Scenario maturity and versioning](scenarios.md#scenario-maturity-and-versioning) explains task readiness and comparisons across revisions.

[GitHub issues](https://github.com/brn-dev/swarm-bots/issues) and email to [Dominik Baron](mailto:dominik.b4ron@gmail.com) are the channels for bugs and feedback. A useful report includes the task ID, version or commit, configuration, seed, and observed behavior.
