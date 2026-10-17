# Reference learning baselines

`swarmbots.learn` provides included policy learners, checkpoint continuation, evaluation adapters, and recording helpers. The presets are starting configurations, not published reference scores or required benchmark settings.

The [runnable examples](../examples/README.md) combine training, evaluation, and recording in one script. The [Python API](api.md) supports custom-learner integration.

The presets and options below describe the current source checkout. The published `0.1.0a3` release has a [versioned learning guide](https://github.com/brn-dev/swarm-bots/blob/v0.1.0a3/docs/learning.md); its preset names and available options differ.

## Train a policy

```python
from swarmbots.learn import train

trainer = train(
    "SwarmBots-WallEasy-v0",
    "mappo",
    num_envs=256,
    device="cuda",
    total_timesteps=1_000_000,
    run_dir="runs/wall-easy/mappo",
)
```

Run this after [installation](../README.md#install). `total_timesteps` counts transitions across all parallel worlds: one step with 256 worlds adds 256 transitions. A short run checks the workflow; reaching good performance may need a larger budget and tuning.

`train` saves logs and checkpoints when `run_dir` is set and closes the simulator when it finishes. The returned trainer retains the learned policy and observation-normalization statistics for evaluation. Choose a separate run directory for each experiment.

## Select a variant and task

The training helpers accept every [registered task](scenarios.md). `list_variants()` returns the available policy names:

```python
from swarmbots.learn import list_variants

print(list_variants())
```

| Preset | What it runs |
| --- | --- |
| `ppo` | PPO with a joint multilayer perceptron (MLP) actor |
| `mappo` | MAPPO with independent actor/critic encoders and a centralized team critic |
| `mat_orig`, `mat_ind`, `mat_dec` | PPO with autoregressive, independent-head, or decentralized multi-agent transformer (MAT) actors |
| `mat_qcx` | PPO with the query/context autoregressive MAT decoder |
| `mat_ind__lstm`, `mat_qcx__lstm` | MAT policies with LSTM memory |
| `maddpg__deepset_critic` | DDPG with a shared per-agent actor and a Deep Set critic |
| `matd3__deepset_critic` | TD3 with a shared per-agent actor and Deep Set critics |
| `masac__deepset_critic` | SAC with a shared per-agent actor and Deep Set critics |
| `tmatd3`, `tmatd3_dec` | TD3 with transformer or decentralized actors and transformer critics |
| `tmasac`, `tmasac_dec` | SAC with transformer or decentralized actors and transformer critics |
| `tmasac__lstm`, `tmasac__slstm` | TMASAC with recurrent memory |

A decentralized actor uses its own agent's local observations and the public task observations. Other actors can combine public observations across agents. All evaluated actors exclude privileged fields.

`list_variants(include_hidden=True)` also lists NOP-disabled controls and gated Deep Set presets (`maddpg__gated_deepset_critic`, `matd3__gated_deepset_critic`, and `masac__gated_deepset_critic`) for architecture comparisons. Training and recording commands accept these exact names.

The [references guide](references.md) maps learners to their original papers. The [method differences](#differences-from-the-original-multi-agent-methods) explain how these adaptations affect comparisons with original implementations.

## Training defaults

The Python helper and the runnable example use different workload defaults:

| Setting | `train(...)` | `examples/train_policy.py` |
| --- | --- | --- |
| Training budget | 100 million transitions | 1 million transitions |
| Parallel training worlds | 1024 | 256 |
| Device | CUDA | CUDA |
| Training seed | 42 | 42 |
| Model scale | `5M-2L NOP1M` | `5M-2L NOP1M` |
| Policy compilation | Disabled | Disabled |
| Evaluation and videos | Separate evaluation and recording helpers | One evaluation seed, then two videos |

Tasks normally have a 500-control-step episode limit. Training spreads the first episode endings across worlds to avoid simultaneous resets; subsequent episodes use the full limit. Evaluation and recording start with full episodes.

CUDA environments compile tensor operations by default, independently of policy compilation. The [GPU setup guide](gpu_setup.md) covers the compiler prerequisites for ordinary CUDA runs.

SAC, DDPG, and TD3 collect at least 10,000 transitions before learning starts. Recurrent learners also need enough history for their replay sequences. Early random actions or missing losses during this warmup are expected.

## Model scale

The `model_scale` argument selects a fixed layout; `model_scale="2.5M"` selects the smallest tier:

| Short name | Intended main-policy budget | Auxiliary prediction budget |
| --- | --- | --- |
| `2.5M` | 2.5 million parameters | 0.75 million |
| `5M-2L` (default) | 5 million | 1 million |
| `5M-3L` | 5 million, with deeper attention/Deep Set encoders | 1 million |
| `10M-3L` | 10 million, with three encoder blocks | 2 million |
| `10M-4L` | 10 million, with four encoder blocks | 2 million |
| `10M-5L` | 10 million, with five encoder blocks | 2 million |

NOP means **next-observation prediction**, an auxiliary training objective enabled by default in the core presets. MAT, recurrent MAT, and MAPPO train NOP from critic-encoder features; the prediction loss updates the critic encoder without updating the actor. Plain PPO retains its shared feature encoder. `use_nop=False` disables it without changing the main policy's dimensions.

The scale names describe approximate feed-forward budgets, not exact counts or GPU memory requirements. Recurrent memory and optional gates add parameters; targets and optimizer state need additional memory. Fewer parallel worlds or a smaller model scale reduce memory requirements.

`model_scale=None` (CLI: `--model-scale custom`) enables custom widths; fixed scales take precedence over width overrides. The [model layouts](policy_parameters/model_scale.md) and [parameter inspection guide](policy_parameters/README.md) cover detailed comparisons.

## Customize training

Pass common settings directly and learner settings through `algorithm_kwargs`:

```python
from swarmbots.learn import train

trainer = train(
    "SwarmBots-POWallMedium-v0",
    "mat_qcx",
    num_envs=256,
    total_timesteps=10_000_000,
    run_dir="runs/po-wall/mat-qcx",
    model_scale="2.5M",
    use_nop=False,
    algorithm_kwargs={"learning_rate": 1e-4},
    learn_kwargs={"log_interval": 10},
)
```

| Argument | Purpose |
| --- | --- |
| `num_envs`, `device`, `seed` | Workload, hardware, and reproducibility |
| `total_timesteps`, `run_dir`, `load_path` | Budget, output location, and checkpoint continuation |
| `model_scale`, `use_nop`, `continuous_action_dist` | Model size, auxiliary prediction, and stochastic action sampling |
| `algorithm_kwargs` | Learner-specific learning rate, batch size, replay, or exploration settings |
| `learn_kwargs` | Logging, saving, W&B configuration, and evaluation hooks |
| `policy_kwargs` | Architecture experiments beyond the presets |
| `scenario_kwargs`, `env_kwargs`, `episode_length` | Custom task or simulator settings; report these as custom variants |
| `compile_modules=True` | Policy compilation after checking [GPU setup](gpu_setup.md) |

PPO, SAC, and TD3 accept different learner options. The [recurrent TD3 example](../examples/README.md#train-recurrent-td3) supplies the recurrent actor and replay settings.

Gated Deep Set presets gate the critic encoder and keep their Q heads ungated by default. `policy_kwargs={"critic_deepset_head_gating": True}` enables Q-head reception gates independently for architecture ablations; the same setting works with ordinary Deep Set and transformer off-policy presets.

### Logging levels

The default console output is `minimal`; persisted CSV/W&B metrics use `full`. Select them independently through `learn_kwargs`:

```python
learn_kwargs = {
    "logging_console_level": "return_success",
    "logging_persistence_level": "minimal",
}
```

| Level | Contents |
| --- | --- |
| `full` | All available metrics and statistics |
| `minimal` | Progress, return/success, core losses, learning rate, and algorithm diagnostics |
| `return_success` | Return/success metrics and progress counters |

Console labels include `steps` (environment transitions), `ret` (return moving average), `succ%` (success moving average in percent), `lr` (learning rate), and `fps` (throughput). Fields appear only when available. Tasks without a success condition do not produce success scores.

For custom console selection, `logging_console_keys` accepts metric names; `[]` silences metric rows and `None` shows all metrics. Persisted fields keep their original names. The [plotting guide](log_plotting.md) covers CSV column listing and charts.

W&B and interactive prompts are optional. In a checkout, install them with `uv sync --extra wandb --extra prompt`; in an application project, use `uv add "swarmbots[wandb,prompt]==0.1.0a3"`. W&B starts only when explicitly configured through logging options.

## Action distributions

PPO/MAT presets use Signed-Magnitude Beta (`sign_magnitude_beta`) by default. SAC presets use `predicted_std_gaussian`. DDPG/TD3 use deterministic actions with exploration noise during training.

`continuous_action_dist` selects the sampling distribution for distribution experiments. PPO supports `sign_magnitude_beta`, `gumbel_softmax_sign_magnitude_beta`, `beta`, `predicted_std_gaussian`, `gsde`, and `squashed_diag_gaussian`. SAC needs distributions with differentiable sampling; `gumbel_softmax_sign_magnitude_beta` provides its signed-magnitude counterpart. Checkpoint reconstruction requires the distribution used during training.

PPO's entropy bonus is controlled by `algorithm_kwargs={"mc_ent_coef": 1e-3}` (the default); `0.0` disables it. SAC presets use automatic entropy-temperature tuning. DDPG/TD3 exploration is controlled by `algorithm_kwargs={"exploration_noise": 0.1}` (the default), and omitted during deterministic evaluation.

## Continue a checkpoint

Recreate the task, variant, model scale, NOP setting, action distribution, and architecture used for training:

```python
from swarmbots.learn import train

trainer = train(
    "SwarmBots-WallEasy-v0",
    "mappo",
    load_path="runs/wall-easy/mappo/models/model_1000000_steps_final.pt",
    total_timesteps=2_000_000,
    run_dir="runs/wall-easy/mappo-continuation",
    num_envs=256,
    device="cuda",
)
```

Replace the path with the actual checkpoint saved by the training run. Final checkpoints are named `model_<steps>_steps_final.pt`; the actual counter can pass the requested budget by a collection batch.

`total_timesteps` is the final total counter, so resuming at one million with a target of two million adds about one million transitions. Checkpoints restore model parameters, saved optimizer state, counters, and normalization. Loading requires the current checkpoint format and matching model settings. Off-policy replay buffers are rebuilt and refilled after loading.

The Python API supports continuation in an existing run directory. The training example scripts require a fresh directory for each invocation. Checkpoints require matching architectures and observation layouts; matching a variant name alone is insufficient. The [layout versions](policy_parameters/model_scale.md) and [changelog](../CHANGELOG.md) record these changes.

## Evaluate a trained policy

Use a fresh adapter to freeze normalization and handle recurrent state for each evaluation:

```python
from swarmbots import evaluate_policy
from swarmbots.learn import as_benchmark_policy

result = evaluate_policy(
    as_benchmark_policy(trainer, deterministic=True),
    "SwarmBots-WallEasy-v0",
    num_envs=256,
    num_episodes=256,
    seed=1000,
    device="cuda",
    action_mode="deterministic",
)
print(result.mean_return, result.success_rate)
```

Use the policy's device. The adapter shares the trained actor, so run evaluation after training or between updates. It preserves the actor's mode and training normalization statistics.

This evaluates one seed. For a [complete benchmark report](benchmark_protocol.md), repeat with a fresh adapter at each seed `1000` through `1004` and aggregate the five results as in the [reporting example](api.md#five-seed-report). Include the checkpoint and architecture in `policy_metadata`, and the benchmark commit in `source_revision`.

### Five-seed report from a saved checkpoint

The first training script exits after its single-seed evaluation. The following standalone recipe loads the saved MAPPO checkpoint and writes `runs/first-mappo/benchmark_report.json` with all five protocol seeds, raw episodes, runtime metadata, and aggregate scores. Run it from the source checkout root after the [first training example](../examples/README.md#train-evaluate-and-record-a-preset):

```python
import json
from pathlib import Path
import statistics
import subprocess

from swarmbots import evaluate_policy
from swarmbots.learn import as_benchmark_policy, make_training

run_dir = Path("runs/first-mappo")
checkpoint = next((run_dir / "models").glob("*_steps_final.pt"))
training_seed = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))["metadata"]["policy"]["training_seed"]
benchmark_id = "SwarmBots-WallEasy-v0"
source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
trainer = make_training(benchmark_id, "mappo", num_envs=2, device="cuda")
try:
    trainer.load(checkpoint, restore_optimizer_state=False)
    results = [
        evaluate_policy(
            as_benchmark_policy(trainer, deterministic=True),
            benchmark_id,
            num_envs=256,
            num_episodes=256,
            seed=seed,
            device="cuda",
            action_mode="deterministic",
            source_revision=source_revision,
            policy_metadata={
                "variant": "mappo",
                "checkpoint": str(checkpoint),
                "training_seed": training_seed,
                "training_timesteps": trainer.n_total_timesteps,
                "settings": trainer.policy.get_hyper_parameters(),
            },
        )
        for seed in range(1000, 1005)
    ]
finally:
    trainer.env.close()

seed_returns = [result.mean_return for result in results]
seed_successes = [result.success_rate for result in results if result.success_rate is not None]
report = {
    "benchmark_id": benchmark_id,
    "seeds": [result.seed for result in results],
    "mean_return": statistics.fmean(seed_returns),
    "std_seed_mean_return": statistics.pstdev(seed_returns),
    "mean_success_rate": statistics.fmean(seed_successes) if seed_successes else None,
    "std_seed_success_rate": statistics.pstdev(seed_successes) if seed_successes else None,
    "results": [result.to_dict() for result in results],
}
output = run_dir / "benchmark_report.json"
output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(output)
```

The two-world trainer restores the policy and normalization; each evaluation uses a separate 256-world environment. This recipe performs no training updates. The report includes the restored training-step count and the training seed from the first script's `evaluation.json`. Other checkpoints require their original task, variant, model scale, NOP, distribution, and architecture settings in `make_training`.

The recorded Git revision identifies the checkout used for evaluation. The training commit needs a separate record alongside the original checkpoint, `evaluation.json`, and run metadata; the default run metadata does not capture that commit or the training seed.

## Record a checkpoint

```bash
swarmbots record SwarmBots-WallEasy-v0 --checkpoint runs/wall-easy/mappo/models/model_1000000_steps_final.pt --variant mappo --episodes 3 --output recordings/mappo
```

Use the actual checkpoint path and matching training settings. The Python equivalent is `swarmbots.learn.record_checkpoint(...)`. The [recording guide](recording.md) covers model-scale options, custom policies, and camera settings; [log plotting](log_plotting.md) covers learning curves.

## Custom training loops

`make_training` exposes the learning loop for custom control. The caller must close its environment:

```python
from swarmbots.learn import make_training

trainer = make_training("SwarmBots-Bridge-v0", "mat_dec", num_envs=256)
try:
    trainer.learn(
        max_total_timesteps=10_000_000,
        run_dir="runs/bridge/mat-dec",
        enable_command_prompt=False,
    )
finally:
    trainer.env.close()
```

The included collectors handle active-agent masks, terminal observations, and immediate autoreset. The [episode-handling guide](api.md#episode-lifecycle) and [standalone PPO example](../examples/README.md#train-without-the-included-learning-framework) cover independent learners.

## Differences from the original multi-agent methods

These learners adapt the original algorithms to SwarmBots' cooperative team objective. The following choices affect comparisons with original implementations:

- MAPPO and MAT, including `mat_orig`, estimate one shared team value per world instead of separate per-agent values.
- MADDPG/MATD3 use team critics. Their shared actor updates all active agents' current actions jointly through the team critic, so gradients flow through the entire joint action.
- MAT actors and critics use separate encoders. The `mat_orig` name identifies its autoregressive architecture lineage, not an exact reproduction of the original method.

The [references and implementation sources](references.md) identify the original formulations. The [model reference](policy_parameters/README.md) covers model dimensions, critic interactions, parameter accounting, and recurrent layouts. The [published thesis](https://resolver.obvsg.at/urn:nbn:at:at-ubl:1-108602) describes TMASAC, MAT-QCX, NOP, and signed-magnitude actions.
