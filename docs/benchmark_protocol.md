# Benchmark protocol 0.1

This protocol defines comparable evaluation, not a required training procedure.

## Core suite

The core suite is exposed as `swarmbots.CORE_BENCHMARK_IDS` and contains:

- fixed and partially observable medium walls;
- bridge traversal and hidden-opening exploration;
- climbing and vertical reach;
- payload-over-step and multi-payload goal transport.

The full suite is exposed as `swarmbots.ALL_BENCHMARK_IDS`. Report per-task values. Reward scales differ between tasks, so an unnormalized average return across the suite is not meaningful.

## Evaluation

For each task and checkpoint:

1. Use the unmodified registered scenario and its 500-step episode limit.
2. Evaluate seeds `1000`, `1001`, `1002`, `1003`, and `1004`.
3. Run 256 parallel worlds and accept exactly the first episode from each world for every seed.
4. Use deterministic policy actions unless the experiment specifically studies stochastic evaluation.
5. Report mean episode return for every task and success rate where the registry declares one.
6. Report the mean and standard deviation across the five seed-level means, plus all raw per-episode values.

This yields 1,280 evaluation episodes per task. Keeping one episode per lane avoids changing the sampled evaluation population when faster-terminating policies autoreset earlier.

The evaluator's defaults implement one 256-world seed at a time:

```python
result = evaluate_policy(policy, benchmark_id, seed=1000, num_envs=256, num_episodes=256)
```

## Policy information boundary

An evaluated actor may consume only `local_obs`, `global_obs`, and `agent_mask`. It must not consume `hidden_local_vars`, `hidden_global_vars`, simulator state, future reset state, or reward decomposition terms. Recurrent state must be reset wherever `episode_starts` is true.

Centralized critics may use privileged variables during training. This is centralized training with decentralized execution, not privileged execution.

## Reproducibility checklist

Publish or record:

- package version and Git commit;
- benchmark ID and protocol version;
- all scenario or episode-length overrides (canonical results should have none);
- policy checkpoint, model architecture, and deterministic/stochastic action mode;
- PyTorch, MuJoCo, MuJoCo Warp, Warp, CUDA, driver, and GPU versions;
- raw `EvaluationResult.to_dict()` outputs for all five seeds.

Changing active-unit distributions, morphology pools, connector semantics, observations, rewards, termination conditions, or physics parameters creates a different benchmark variant.
