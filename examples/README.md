# Examples

Run these scripts from the repository root after installing `swarmbots`. The commands below use `python`; in a source checkout, use `.venv/bin/python` or `.venv\Scripts\python.exe` on Windows.

## Evaluate a random policy

```bash
python examples/evaluate_random_policy.py SwarmBots-WallEasy-v0 --device cuda --output runs/wall-random.json
```

This evaluates a uniform-random policy over seeds `1000` through `1004` and writes raw episodes, runtime metadata, and aggregate statistics to JSON. Use `--num-envs 2 --seeds 1000 --device cpu` for a smaller run.

## Train, evaluate, and record a preset

`train_policy.py` accepts every name returned by `swarmbots.learn.list_variants()`:

```bash
python examples/train_policy.py SwarmBots-WallEasy-v0 --variant mappo
python examples/train_policy.py SwarmBots-WallEasy-v0 --variant mat_ind
python examples/train_policy.py SwarmBots-WallEasy-v0 --variant tmasac
python examples/train_policy.py SwarmBots-WallEasy-v0 --variant matd3_deepset
python examples/train_policy.py SwarmBots-WallEasy-v0 --variant maddpg_mlp
python examples/train_policy.py SwarmBots-POWallMedium-v0 --variant tmasac_slstm
```

Each command trains for one million individual environment transitions using 256 parallel CUDA worlds, evaluates deterministic actions in a fresh environment at seed `1000`, and records two complete episodes at 640 × 480. Training follows the selected preset's defaults, including its action distribution and NOP setting. Fixed tiers are `2.5M NOP0.75M`, `5M NOP1M` (default), and `10M NOP2M`; use `--model-scale 2.5M` or `--model-scale 10M` to select the added tiers. Dimensions are declared per architecture. Use `--model-scale legacy` for explicitly customized widths. Use `--no-nop` to disable NOP or `--compile` to compile policy/world-model modules. MLP off-policy critics already disable NOP and reject `--nop`.

Outputs go to `runs/<benchmark-id>/<variant>/`: training metrics and metadata, checkpoints in `models/`, `evaluation.json`, and MP4s in `videos/`. Use `--run-dir` to select a separate directory for each experiment. Adjust `--total-timesteps`, `--num-envs`, `--eval-envs`, and `--video-episodes` to change the workload; `--device cpu` selects CPU execution. Off-policy presets collect at least 10,000 transitions before learning starts, and recurrent learners also need enough history for their replay segments.

Console metrics default to `minimal`, while persisted CSV/W&B metrics default to `full`. Select them independently with `--console-log-level` and `--persistent-log-level`; both accept `full`, `minimal`, and `return_success`. For example, add `--console-log-level return_success --persistent-log-level minimal` to reduce both outputs. The same flags apply to `train_recurrent_td3.py`. See [logging levels](../docs/learning.md#logging-levels) for retained fields and Python settings.

The example shows a single evaluation seed. For a complete protocol report, evaluate a fresh `as_benchmark_policy(trainer)` adapter at each seed `1000` through `1004`, with 256 episodes per seed, and aggregate the results as in `evaluate_random_policy.py`. See the [evaluation protocol](../docs/benchmark_protocol.md).

## Train recurrent TD3

```bash
python examples/train_recurrent_td3.py SwarmBots-POWallMedium-v0 --variant tmatd3
python examples/train_recurrent_td3.py SwarmBots-POWallMedium-v0 --variant tmatd3_dec --recurrent-critic
python examples/train_recurrent_td3.py SwarmBots-POWallMedium-v0 --variant matd3_deepset
```

This example enables the recurrent actor through `policy_kwargs` and shows sequence-replay settings through `algorithm_kwargs`: 32 burn-in steps, 64 learning steps, and a stored actor-state interval of 16 steps. By default, feed-forward critics receive detached actor state. `--recurrent-critic` gives TMATD3's transformer critic its own recurrent history. The same training, evaluation, and recording workflow is reused, with separate default output directories for the two critic choices.

## Record an existing policy

```bash
python examples/record_policy.py SwarmBots-WallEasy-v0 --checkpoint runs/my-run/models/model_final.pt --variant mappo --output videos/mappo
```

Replace the checkpoint path with the file saved by your run. Architecture, distribution, and NOP settings must match when rebuilding a checkpoint. `record_policy.py` also accepts custom `module:factory` policies; see [recording](../docs/recording.md). Every script provides `--help`.

Policy inspection, architecture diagrams, and log plotting are available as [installed command-line tools](../docs/tools.md).
