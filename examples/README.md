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

Each command trains for one million individual environment transitions using 256 parallel CUDA worlds, evaluates deterministic actions in a fresh environment at seed `1000`, and records two complete episodes at 640 × 480. Training follows the selected preset's defaults, including its action distribution and NOP setting. Use `--no-nop` to disable NOP or `--compile` to compile policy/world-model modules. MLP off-policy critics already disable NOP and reject `--nop`.

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

## Count policy parameters

```bash
python examples/inspect_policy_parameters.py SwarmBots-WallEasy-v0 --output-dir docs/policy_parameters
python examples/inspect_policy_parameters.py SwarmBots-POWallMedium-v0 --variants tmasac tmasac_slstm tmasac_lstm
```

The script instantiates every registered preset by default, using the same policy factory as training, without allocating a trainer, replay buffer, or optimizer. It reports actor, critic, shared encoder, next-observation prediction (NOP), frozen target copies, and other parameters. Detailed component tables separate recurrent temporal modules, critic encoders/Q heads, and NOP projections/transition/prediction heads. Shared parameters count once; buffers are excluded. CPU execution and eager modules are the defaults.

The main-policy processing comparison excludes NOP and targets. It splits MLPs, standalone linear projections, attention, recurrent modules, normalization, embeddings, and other parameters, by actor, critic, and shared encoder. **MLP + linear** includes transformer feed-forward blocks, SwiGLU gates, observation/latent projections, and action/value heads. Attention's Q/K/V/output projections and complete recurrent modules (including input/gate projections) count in their respective categories. This affine budget helps compare processing capacity across variants; it does not measure FLOPs or effective capacity.

The **Trainable − NOP** column counts the trainable total minus trainable NOP parameters. It excludes all frozen parameters, including target networks. The **Online total** in the processing comparison excludes NOP and targets but can include other frozen main-policy parameters.

`--output-dir` saves a Markdown report, summary and component CSVs, `processing.csv` (processing counts by role), `processing_components.csv` (classified modules), and losslessly compressed `counts.json.gz` with resolved hyperparameters and task shapes. Read the JSON using `json.load(gzip.open(path, "rt", encoding="utf-8"))`. Use `--no-nop` or JSON `--policy-kwargs`, `--scenario-kwargs`, and `--env-kwargs` to match custom settings. Parameter counts depend on the task's observation/action shapes. See the [35-variant WallEasy results](../docs/policy_parameters/report.md) for the measured default sizes.

## Plot training logs

Install `swarmbots[plot]` to use the plotting examples:

```bash
python examples/plot_logs.py runs/SwarmBots-WallEasy-v0/mappo --output plots/mappo.png
python examples/plot_logs.py runs/my-run --list-columns
python examples/plot_logs.py runs/seed-42 runs/seed-43 --columns ep_rew_ema --labels "Seed 42" "Seed 43"
python examples/plot_experiment_results.py runs/wall-easy --output-dir plots/wall-easy --formats png pdf
python examples/plot_experiment_results.py --group MAPPO runs/mappo/seed-42 runs/mappo/seed-43 --group TMASAC runs/tmasac/seed-42 runs/tmasac/seed-43
```

`plot_logs.py` overlays scalar metrics from files or run directories; its defaults show return and success EMAs. `--show` opens a Matplotlib window. Compressed CSVs are accepted directly. `plot_experiment_results.py` saves per-metric grouped mean/std plots and individual-seed plots, using either an experiment root or explicit named groups. Both support custom columns, smoothing, and a training-step cutoff. See [log plotting](../docs/log_plotting.md) for layouts, aggregation semantics, and the Python API.
