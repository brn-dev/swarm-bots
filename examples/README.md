# Examples

Run these scripts from the repository root after [installing the checkout](../README.md#install). CUDA is recommended; check [GPU setup](../docs/gpu_setup.md) first.

The commands below use `.venv/bin/python`. On Windows, use `.\.venv\Scripts\python.exe` instead. No environment activation is needed. Every script supports `--help`.

## Train, evaluate, and record a preset

Start with one complete run:

```bash
.venv/bin/python examples/train_policy.py SwarmBots-WallEasy-v0 --variant mappo__deepset_critic --run-dir runs/first-mappo
```

On Windows:

```powershell
.\.venv\Scripts\python.exe examples/train_policy.py SwarmBots-WallEasy-v0 --variant mappo__deepset_critic --run-dir runs/first-mappo
```

The script trains for one million environment transitions using 1,024 parallel CUDA worlds, evaluates deterministic actions at seed `1000`, then records two episodes at 640 × 480. A short run demonstrates the workflow; it may not learn to solve the task.

Inspect the saved outputs:

| Path under `runs/first-mappo/` | Contents |
| --- | --- |
| `log.csv` or its compressed counterpart | Learning curves and diagnostics |
| `run_metadata_*.json` | Settings and runtime information |
| `models/` | Checkpoints; the final file ends in `_steps_final.pt` |
| `evaluation.json` | Evaluation summary, individual episodes, and metadata |
| `videos/` | MP4 recordings |

An explicit `--run-dir` must not already exist. Omit it to create a separate directory automatically under `runs/<benchmark-id>/<variant>/<scale>/seed-<seed>/`. Repeating a configuration then creates a new run. Use the [Python continuation workflow](../docs/learning.md#continue-a-checkpoint) to resume training.

Change the task or choose another [learning preset](../docs/learning.md#select-a-variant-and-task):

```bash
.venv/bin/python examples/train_policy.py SwarmBots-WallEasy-v0 --variant mat_qcx
.venv/bin/python examples/train_policy.py SwarmBots-WallEasy-v0 --variant tmasac
.venv/bin/python examples/train_policy.py SwarmBots-WallEasy-v0 --variant matd3__deepset_critic
.venv/bin/python examples/train_policy.py SwarmBots-POWallMedium-v0 --variant tmasac__slstm
```

| Option | Effect |
| --- | --- |
| `--total-timesteps 10000000` | Sets the training budget across all worlds |
| `--num-envs 64` | Reduces parallel training worlds and memory use |
| `--eval-envs 64` | Sets the number of development evaluation worlds |
| `--video-episodes 1` | Sets the number of recorded episodes |
| `--model-scale 2.5M` | Selects a smaller model; other choices are `5M-2L` (default), `5M-3L`, `10M-3L`, `10M-4L`, and `10M-5L` |
| `--no-nop` | Disables auxiliary next-observation prediction |
| `--compile` | Enables policy compilation in addition to the environment's default CUDA compilation |
| `--seed 43` | Sets the training seed |
| `--console-log-level return_success` | Limits console metrics to progress and return/success |
| `--persistent-log-level minimal` | Reduces saved metrics; the default is `full` |

SAC, DDPG, and TD3 collect at least 10,000 transitions before learning starts; recurrent policies also need enough history. A very short run may end before any optimizer updates. CPU runs use `--device cpu` and can be slow even with few worlds.

The script's evaluation covers one seed. The [saved-checkpoint report recipe](../docs/learning.md#five-seed-report-from-a-saved-checkpoint) evaluates all five protocol seeds and exports reportable scores.

## Evaluate a random policy

```bash
.venv/bin/python examples/evaluate_random_policy.py SwarmBots-WallEasy-v0 --device cuda --output runs/wall-random.json
```

This evaluates a uniform-random policy over all five protocol seeds and writes raw episodes, runtime metadata, and aggregate statistics to JSON. It is a stochastic sanity check. Adapt its reporting loop to a trained policy for a benchmark report.

For a smaller development run:

```bash
.venv/bin/python examples/evaluate_random_policy.py SwarmBots-WallEasy-v0 --num-envs 2 --seeds 1000 --device cpu --output runs/random-check.json
```

Full episodes can still be slow on CPU. Use the [smoke test](../README.md#quick-start) for an installation check.

## Train without the included learning framework

```bash
.venv/bin/python examples/train_standalone_ppo.py SwarmBots-WallEasy-v0 --device cuda
```

This minimal PyTorch PPO example uses the environment API directly, without `swarmbots.learn` or additional dependencies. It provides a starting point for custom-learner integration. It demonstrates public observation inputs, bounded actions, active-agent masks, rollout collection, and episode handling.

Defaults are one million transitions, 1,024 worlds, and 128 steps per rollout. It prints mean reward per transition; it does not produce the preset example's checkpoint, evaluation, and video outputs. The settings illustrate integration and are not tuned reference baselines.

For a short CPU execution check:

```bash
.venv/bin/python examples/train_standalone_ppo.py --device cpu --num-envs 2 --total-timesteps 8 --rollout-steps 4
```

To evaluate the trained model, call it from Python at the repository root:

```python
from examples.train_standalone_ppo import train
from swarmbots import evaluate_policy

model = train("SwarmBots-WallEasy-v0", device="cuda")
result = evaluate_policy(
    model.act,
    "SwarmBots-WallEasy-v0",
    device="cuda",
    num_envs=256,
    num_episodes=256,
    seed=1000,
    action_mode="deterministic",
)
print(result.mean_return, result.success_rate)
```

The `examples` imports are checkout code, not installed package APIs. This evaluates one seed; use the full [evaluation protocol](../docs/benchmark_protocol.md) for a report.

## Train recurrent TD3

```bash
.venv/bin/python examples/train_recurrent_td3.py SwarmBots-POWallMedium-v0 --variant tmatd3
.venv/bin/python examples/train_recurrent_td3.py SwarmBots-POWallMedium-v0 --variant tmatd3_dec --recurrent-critic
.venv/bin/python examples/train_recurrent_td3.py SwarmBots-POWallMedium-v0 --variant matd3__deepset_critic
```

This uses the same train → evaluate → record workflow with a recurrent actor and sequence replay. `--recurrent-critic` also gives the critic memory and is available for `tmatd3` and `tmatd3_dec`. The workload, output-directory, scale, and logging options above apply here too.

Checkpoint reconstruction requires `policy_kwargs={"td3_recurrent_actor": True}`; a run with `--recurrent-critic` additionally requires `"td3_recurrent_critic": True`. These settings apply to Python continuation and evaluation as well as recording. The [recording guide](../docs/recording.md#included-baseline-checkpoints) includes the corresponding `--policy-kwargs` command.

## Record an existing policy

Find the final checkpoint in the run's `models/` directory and substitute its actual filename below:

```bash
.venv/bin/python examples/record_policy.py SwarmBots-WallEasy-v0 --checkpoint runs/first-mappo/models/model_1000000_steps_final.pt --variant mappo__deepset_critic --output recordings/mappo
```

The saved step count can exceed the requested training budget by one collection batch. Match the checkpoint's task, model scale, architecture, action distribution, and NOP setting. The script also accepts custom `module:factory` policies; see [recording](../docs/recording.md).

The [log plotting guide](../docs/log_plotting.md) covers learning curves and seed comparisons. Policy inspection and architecture diagrams are available as [installed command-line tools](../docs/tools.md).
