# SwarmBots documentation

SwarmBots benchmarks cooperative control of robots that can change their physical connections during an episode. Start with the scenario catalog to explore the tasks, then use the Python API to run environments with your own policies. The evaluation protocol defines how to report comparable results.

**The benchmark is alpha and under active development.** All fixed-wall and PO-wall variants (easy, medium, and hard), plus FindOpening, are beta; all other registered scenarios are alpha. Beta reflects extensive internal testing, with feedback from other researchers still pending. Suggestions and research feedback are welcome through [GitHub issues](https://github.com/brn-dev/swarm-bots/issues) or by emailing Dominik Baron at [dominik.b4ron@gmail.com](mailto:dominik.b4ron@gmail.com).

| Goal | Guide |
| --- | --- |
| Explore the tasks, robot mechanics, observations, and success conditions | [Scenario catalog](scenarios.md) |
| Check scenario readiness and understand task/package versioning | [Scenario maturity and versioning](scenarios.md#scenario-maturity-and-versioning) |
| Select the core suite and report comparable scores | [Benchmark protocol 0.1](benchmark_protocol.md) |
| Create environments and evaluate a custom policy | [Benchmark Python API](api.md) |
| Configure CUDA and verify simulation and compilation | [GPU setup](gpu_setup.md) |
| Record policy behavior as videos | [Policy recording](recording.md) |
| Train an included baseline or customize a learning preset | [Reference learning baselines](learning.md) |
| Cite algorithms, understand TMASAC's related work, and credit scientific software | [References and attribution](references.md) |

The benchmark environment and evaluator work independently of the included learning implementations. The learning guide describes the reference baselines and their training presets.

For an installation check, use the [README quick start](../README.md#quick-start). For evaluation reports, the [five-seed example](api.md#five-seed-report) exports raw episodes and metadata to JSON; replace its uniform-random policy with your own trained policy.
