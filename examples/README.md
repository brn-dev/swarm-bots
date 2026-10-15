# Examples

Run these scripts from the repository root after installing `swarmbots`. The commands below use `python`; in a source checkout, use `.venv/bin/python` or `.venv\Scripts\python.exe` on Windows.

## Evaluate a random policy

```bash
python examples/evaluate_random_policy.py SwarmBots-WallEasy-v0 --device cuda --output runs/wall-random.json
```

This evaluates a uniform-random policy over seeds `1000` through `1004` and writes raw episodes, runtime metadata, and aggregate statistics to JSON. Use `--num-envs 2 --seeds 1000 --device cpu` for a smaller run.

## Train without the included learning framework

```bash
python examples/train_standalone_ppo.py SwarmBots-WallEasy-v0 --device cuda
```

This minimal PyTorch PPO loop has no imports from `swarmbots.learn` and no additional dependencies. A small shared network uses `local_obs` and `global_obs` for both the per-agent actor and the team critic. Gaussian actions with a learned standard deviation are squashed through `tanh` for both actuators and connectors. Inactive agents contribute neither actions, likelihoods, nor pooled critic features.

The loop stores pre-tanh Gaussian samples: the squash Jacobian cancels in PPO's likelihood ratio, so no inverse `tanh` or Jacobian calculation is needed. It copies reusable observation buffers and bootstraps time-limit truncations from `info["final_obs"]`, while stopping advantage traces at all episode boundaries.

Defaults are one million world transitions, 256 parallel worlds, and 128 steps per rollout. Each rollout gets four training epochs using minibatches of up to 1024 world transitions. The script prints mean reward per transition; hyperparameters are directly in the code. Tensor operations and rewards run eagerly, and physics uses MuJoCo Warp. Settings illustrate integration and are not tuned benchmark baselines.

For a short CPU execution check:

```bash
python examples/train_standalone_ppo.py --device cpu --num-envs 2 --total-timesteps 8 --rollout-steps 4
```

To use the trained model in your own code, call `train()` and pass its deterministic `act` method to the evaluator:

```python
from examples.train_standalone_ppo import train
from swarmbots import evaluate_policy

model = train("SwarmBots-WallEasy-v0", device="cuda")
result = evaluate_policy(
    model.act, "SwarmBots-WallEasy-v0", device="cuda",
    num_envs=256, num_episodes=256, seed=1000, action_mode="deterministic",
)
print(result.mean_return, result.success_rate)
```

Run this snippet from the repository root; `examples` is checkout code rather than an installed package API. This evaluates one seed; follow the [evaluation protocol](../docs/benchmark_protocol.md) for reportable results.

## Train, evaluate, and record a preset

`train_policy.py` accepts every name returned by `swarmbots.learn.list_variants()`:

```bash
python examples/train_policy.py SwarmBots-WallEasy-v0 --variant mappo
python examples/train_policy.py SwarmBots-WallEasy-v0 --variant mat_ind
python examples/train_policy.py SwarmBots-WallEasy-v0 --variant tmasac
python examples/train_policy.py SwarmBots-WallEasy-v0 --variant matd3__deepset_critic
python examples/train_policy.py SwarmBots-WallEasy-v0 --variant maddpg__deepset_critic
python examples/train_policy.py SwarmBots-POWallMedium-v0 --variant tmasac__slstm
```

Each command trains for one million individual environment transitions using 256 parallel CUDA worlds, evaluates deterministic actions in a fresh environment at seed `1000`, and records two complete episodes at 640 × 480. Training follows the selected preset's defaults, including its action distribution and NOP setting. Fixed tiers are `2.5M NOP0.75M`, `5M NOP1M` (default), and `10M NOP2M`; use `--model-scale 2.5M` or `--model-scale 10M` to select the added tiers. Dimensions are declared per architecture. Use `--model-scale legacy` for explicitly customized widths. Use `--no-nop` to disable NOP or `--compile` to compile policy/world-model modules.

Outputs go to `runs/<benchmark-id>/<variant>/`: training metrics and metadata, checkpoints in `models/`, `evaluation.json`, and MP4s in `videos/`. Use `--run-dir` to select a separate directory for each experiment. Adjust `--total-timesteps`, `--num-envs`, `--eval-envs`, and `--video-episodes` to change the workload; `--device cpu` selects CPU execution. Off-policy presets collect at least 10,000 transitions before learning starts, and recurrent learners also need enough history for their replay segments.

Console metrics default to `minimal`, while persisted CSV/W&B metrics default to `full`. Select them independently with `--console-log-level` and `--persistent-log-level`; both accept `full`, `minimal`, and `return_success`. For example, add `--console-log-level return_success --persistent-log-level minimal` to reduce both outputs. The same flags apply to `train_recurrent_td3.py`. See [logging levels](../docs/learning.md#logging-levels) for retained fields and Python settings.

The example shows a single evaluation seed. For a complete protocol report, evaluate a fresh `as_benchmark_policy(trainer)` adapter at each seed `1000` through `1004`, with 256 episodes per seed, and aggregate the results as in `evaluate_random_policy.py`. See the [evaluation protocol](../docs/benchmark_protocol.md).

## Train recurrent TD3

```bash
python examples/train_recurrent_td3.py SwarmBots-POWallMedium-v0 --variant tmatd3
python examples/train_recurrent_td3.py SwarmBots-POWallMedium-v0 --variant tmatd3_dec --recurrent-critic
python examples/train_recurrent_td3.py SwarmBots-POWallMedium-v0 --variant matd3__deepset_critic
```

This example enables the recurrent actor through `policy_kwargs` and shows sequence-replay settings through `algorithm_kwargs`: 32 burn-in steps, 64 learning steps, and a stored actor-state interval of 16 steps. By default, feed-forward critics receive detached actor state. `--recurrent-critic` gives TMATD3's transformer critic its own recurrent history. The same training, evaluation, and recording workflow is reused, with separate default output directories for the two critic choices.

## Record an existing policy

```bash
python examples/record_policy.py SwarmBots-WallEasy-v0 --checkpoint runs/my-run/models/model_final.pt --variant mappo --output videos/mappo
```

Replace the checkpoint path with the file saved by your run. Architecture, distribution, and NOP settings must match when rebuilding a checkpoint. `record_policy.py` also accepts custom `module:factory` policies; see [recording](../docs/recording.md). Every script provides `--help`.

Policy inspection, architecture diagrams, and log plotting are available as [installed command-line tools](../docs/tools.md).
