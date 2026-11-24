# Baseline reinforcement learning algorithms

SwarmBots includes a suite of PPO, MAPPO, MAT, SAC, DDPG, and TD3-based learners for training policies and comparing multi-agent architectures. The suite spans centralized and decentralized actors, attention and Deep Set communication, recurrent memory, and optional next-observation prediction (NOP). Fixed model-size tiers support comparisons across these choices; the [comparison reference](policy_parameters/model_scale.md#experimental-comparisons) describes matched components and remaining differences.

`swarmbots.learn` provides training presets, checkpoint continuation, evaluation adapters, and recording helpers. The presets are starting configurations, not published reference scores or required benchmark settings. The [runnable examples](../examples/README.md) combine training, evaluation, and recording in one script. Custom algorithms use the separate [benchmark Python API](api.md).

The presets and options below describe the current source checkout. The published `0.1.0a3` release has a [versioned learning guide](https://github.com/brn-dev/swarm-bots/blob/v0.1.0a3/docs/learning.md); its preset names and available options differ.

## Train a policy

```python
from swarmbots.learn import train

trainer = train(
    "SwarmBots-WallEasy-v0",
    "mappo__deepset_critic",
    num_envs=1024,
    device="cuda",
    total_timesteps=1_000_000,
    run_dir="runs/wall-easy/mappo",
)
```

Run this after [installation](../README.md#install). `total_timesteps` counts transitions across all parallel worlds: one step with the default 1,024 worlds adds 1,024 transitions. A short run checks the workflow; reaching good performance may need a larger budget and tuning.

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
| `mappo__deepset_critic` | MAPPO with a local MLP actor and a centralized Deep Set critic; privileged inputs enter before critic encoding |
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
| Policy variant | `mappo__deepset_critic` | `mappo__deepset_critic` |
| Training budget | 100 million transitions | 10 million transitions |
| Parallel training worlds | 1024 | 1024 |
| Device | CUDA | CUDA |
| Training seed | 42 | 42 |
| Model scale | `5M-2L NOP1M` | `5M-2L NOP1M` |
| Policy compilation | Disabled | Disabled |
| Evaluation and videos | Separate evaluation and recording helpers | One evaluation seed, then two videos |

Training is designed for high CUDA parallelism, with 1,024 parallel worlds as the recommended starting configuration. Small CPU workloads serve execution checks and debugging.

Tasks normally have a 500-control-step episode limit. Training evenly spaces the first episode endings across that limit, with either `floor(num_envs / episode_length)` or `ceil(num_envs / episode_length)` scheduled endings per step. For 1,024 worlds and 500 steps, this gives two or three endings per step. Subsequent episodes use the full limit; early success or failure can change their spacing. Evaluation and recording start with full episodes.

CUDA environments compile tensor operations by default, independently of policy compilation. The [GPU setup guide](gpu_setup.md) covers the compiler prerequisites for ordinary CUDA runs.

SAC, DDPG, and TD3 collect at least 10,000 transitions before learning starts. Recurrent presets use random actions for at least one full burn-in and learning sequence per environment. With the default 1,024 worlds, 32 burn-in steps and 64 learning steps, this covers 98,304 transitions. Early random actions or missing losses during this warmup are expected.

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

Depth labels count actor and critic encoder blocks. Deeper layouts use narrower features to preserve comparable parameter budgets, with common encoder dimensions across MAT, MAPPO, and off-policy families. Action heads and decoders remain one block deep; autoregressive attention adds capacity above the matched non-attention budgets. The [fixed-layout reference](policy_parameters/model_scale.md) lists dimensions and allocation rules.

NOP means **next-observation prediction**, an auxiliary training objective enabled by default in the core presets. MAT, recurrent MAT, and MAPPO train NOP from critic-encoder features; the prediction loss updates the critic encoder without updating the actor. Plain PPO retains its shared feature encoder. `use_nop=False` disables it without changing the main policy's dimensions.

Default targets include public local observations and public global scalars and rotations, such as payload poses and task goals. Inactive MultiPayloadGoal records are excluded from global losses using their payload-active flags. Adding global predictors changes NOP checkpoint shapes for tasks with public global observations; checkpoints require the matching target configuration.

NOP predicts per-step scalar, angle, and rotation changes; binary targets predict next-state flags. PPO scalar deltas use the prediction origin's normalization frame so changing statistics do not create artificial motion. Angle targets require defined connector twists at both endpoints. Custom scalar losses with per-target payload masks must return unreduced losses, such as `nn.MSELoss(reduction="none")`.

The scale names describe approximate feed-forward budgets, not exact counts or GPU memory requirements. Recurrent memory and optional gates add parameters; targets and optimizer state need additional memory. A smaller model scale reduces model and optimizer memory while retaining parallel collection.

`model_scale=None` (CLI: `--model-scale custom`) enables custom widths; fixed scales take precedence over width overrides. The [model layouts](policy_parameters/model_scale.md) and [parameter inspection guide](policy_parameters/README.md) cover detailed comparisons.

## Customize training

Pass common settings directly and learner settings through `algorithm_kwargs`:

```python
from swarmbots.learn import train

trainer = train(
    "SwarmBots-POWallMedium-v0",
    "mat_qcx",
    num_envs=1024,
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

The PPO presets collect four vector steps per rollout by default. To request a total across all worlds, import `StepsRolloutMode` from `swarmbots.learn.algos.ppo.ppo` and pass `algorithm_kwargs={"rollout_mode": StepsRolloutMode(n)}`. Collection rounds the requested `n` transitions up to a whole vector step. With `sampler_config=None`, the default feed-forward minibatch spans the collected rollout, and the default recurrent sequence spans the collected steps per world. A supplied sampler config takes precedence.

Off-policy default batches target 1,024 learning positions independently of environment count and rollout length. Feed-forward learners sample 1,024 transitions; recurrent learners divide this budget by their effective `learning_steps`, rounding up to at least one whole sequence. The default 64-step learning horizon therefore uses 16 sequences per update. An explicit `algorithm_kwargs={"batch_size": ...}` overrides this default and counts transitions for feed-forward learners or sequences for recurrent learners. Burn-in adds computation outside the scored learning-position budget.

`algorithm_kwargs={"replay_exposure": 8.0}` sets average scored replay uses per new transition after learning starts, replacing the constructor's `gradient_steps` setting. Each collection cycle performs `replay_exposure × collected_transitions / learning_positions_per_batch` updates; this must be a positive whole number. The default 1,024 worlds, one-step collection, and 1,024-position batch give eight updates. If fewer worlds produce fractional updates, increase `num_envs` to a compatible workload. Burn-in, separate NOP batches, and actor/target delays are outside this ratio; individual transitions need not receive equal replay uses.

All registered learning presets use `gamma=0.995`. At the default 0.03-second control interval, this gives an approximately six-second discount horizon. `algorithm_kwargs={"gamma": ...}` overrides it; low-level learner constructors retain their own defaults.

Changes to `PPO.gamma` or `PPO.gae_lambda` take effect when the next rollout computes advantages and returns. With PopArt disabled, the reward-normalization wrapper estimates reward scale using its own discount, initialized from the construction settings.

Off-policy observation and reward statistics adapt throughout training. Replay and transition histories remain raw; rollout inputs and sampled transitions, burn-in, and NOP windows use current normalization statistics. Sampling does not update statistics. Checkpoint continuation resumes their adaptation, while evaluation uses frozen statistics. Recurrent burn-in reduces the mismatch caused by changing policy weights and normalization, but remains an approximation.

Custom environments accept one `TorchFeatureWiseObsNormWrapper` per observation key; combine feature groups for the same key in that wrapper. Normalization compiles by default on CUDA. Setting `compile_tensor_operations=False` on a participating wrapper keeps the combined pass eager; CPU normalization is always eager.

Recurrent SAC/TD3 replay supports at most one time-limit truncation per learning sequence. Training requires `episode_length >= learning_steps`; shorter limits fail before environment construction. A shorter learning horizon requires an explicit `learning_steps` override and a compatible `temporal_state_store_interval`. Burn-in is excluded from this restriction, and genuine terminations do not count as time-limit truncations. Other sources of early truncation can still make individual windows ineligible.

`make_training` shares these defaults across SAC, DDPG and TD3 presets, including hidden controls:

| Setting | Shared off-policy default |
| --- | --- |
| Actor/critic optimizer and learning rate | Adam, `5e-5` |
| Actor/critic learning-rate warmup | 500 critic updates, starting at 1% of the base rate |
| Learning positions per batch / replay exposure | 1,024 / 8 scored replay uses per new transition |
| Rollout | One vector-environment step per iteration |
| Discount / target mixing coefficient | `gamma=0.995` / `tau=0.005` |
| Maximum gradient norm | `2.0` |
| Learning starts | `max(10_000, 4 × rollout transitions per iteration)`, raised for recurrent presets to at least `num_envs × (burn_in_steps + learning_steps)` |
| Replay capacity per environment | Twice the episode length, enlarged for learning starts, batch positions and recurrent segment storage |
| NOP when enabled | Three-step prediction, loss coefficient `0.1`, reusing the critic replay batch |

All SAC presets use automatic temperature tuning with initial coefficient `0.05`. Within each world, SAC averages log probabilities over active agents. The `"auto_0.1"` setting gives a target entropy of `-0.1 × D` where `D` is the action dimension of one agent, so the target is independent of swarm size. The temperature optimizer uses the base actor/critic learning rate throughout actor/critic warmup. Actor and target evaluation each sample one action, and actor, temperature and target updates occur every critic update. DDPG and TD3 use exploration noise `0.1`. TD3 retains a two-update actor/target delay and target smoothing noise `0.2`, clipped at `0.5`. DDPG uses one critic and updates its actor and targets on every critic update. SAC and DDPG use unsmoothed target actions.

SAC, DDPG and TD3 average the main critic loss across Q heads before adding NOP. The common auxiliary coefficient therefore has the same relative scale for single and twin critics.

Recurrent SAC and TD3 share 32 burn-in steps, 64 learning steps, and temporal-state snapshots every 16 steps stored in float16. Their minimum history requirement can delay the first update. Matching batch positions does not match temporal correlation or compute; [experimental comparison details](policy_parameters/model_scale.md#experimental-comparisons) describe these differences. PPO/MAT retain their on-policy rollout, epoch and learning-rate schedules.

Gated Deep Set presets gate the critic encoder and keep their Q heads ungated by default. `policy_kwargs={"critic_deepset_head_gating": True}` enables Q-head reception gates independently for architecture ablations; the same setting works with ordinary Deep Set and transformer off-policy presets.

### Recurrent layer settings

For custom recurrent MAT-IND and MAT-QCX models, `rmat_actor_d_model` sets actor width independently of the critic. `rmat_actor_transformer_ff_config` and `rmat_actor_inter_module_mlp` configure the actor's feed-forward layers.

`policy_kwargs` accepts shared temporal defaults through `rmat_temporal_model_cls` and `rmat_temporal_model_config`. Recurrent MAT and TD3 critics can override these independently with `rmat_critic_temporal_model_cls` and `rmat_critic_temporal_model_config`. An omitted critic override inherits the corresponding shared setting. A single class or config applies to every layer; a per-layer sequence must match that encoder's resolved depth.

Fixed-scale MAT actors and critics have matching depths. At `5M-2L`, both have two recurrent layers, and independent config sequences take this form:

```python
from swarmbots.learn.algos.r_mat.temporal_sequence_model import LSTMTemporalSequenceModelConfig

policy_kwargs = {
    "rmat_temporal_model_config": [
        LSTMTemporalSequenceModelConfig(),
        LSTMTemporalSequenceModelConfig(),
    ],
    "rmat_critic_temporal_model_config": [
        LSTMTemporalSequenceModelConfig(),
        LSTMTemporalSequenceModelConfig(),
    ],
}
```

Per-layer class sequences follow the same rule, with a matching config type for each selected temporal model. A TMASAC temporal stack containing sLSTM uses the scale's sLSTM memory width, including mixed and per-layer class sequences. These widths support both the default four heads and an explicit eight-head configuration; other head counts must divide the selected memory width.

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

Buffered success rates pool the episodes that reported an outcome. Each reported rate includes its episode count in `ep_success_rate__n`, including in the reduced persistence profiles. Windows with no reported outcomes leave both fields empty. Scenario episode counts include every completed episode for that scenario.

PPO's iteration-level clip fraction and approximate KL use all valid likelihood ratios evaluated during the iteration. Minibatches with fewer valid ratios contribute proportionally less. KL early stopping still checks each minibatch before its optimizer step. The automatic learning-rate controller uses the pooled clip fraction.

W&B and interactive prompts are optional. In a checkout, install them with `uv sync --extra wandb --extra prompt`; in an application project, use `uv add "swarmbots[wandb,prompt]==0.1.0a3"`. W&B starts only when explicitly configured through logging options.

The interactive prompt is read-only. Available inspections include `show_hps`, `show_id`, `show_run_path`, `pid`, `show_rw`, `eta`, and `record_status`; semicolons separate multiple names. The Python equivalent is `trainer.execute_command("show_hps")`, which returns whether the inspection succeeded. Runtime commands cannot change settings, pause or stop training, save checkpoints, or start recordings. Construction options define the training configuration; Python APIs provide checkpoint saving and recording.

Run metadata is written before the first update. Runs created with `make_training` or `train` retain the training seed and construction options, resolved environment/scenario settings, policy and algorithm settings, and source revision. Source provenance includes the Git commit, dirty flag and a content hash of uncommitted tracked and untracked files; installations without an available Git checkout record an unknown revision. A dirty-source hash identifies the source state but does not replace preserving that checkout.

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
    "mappo__deepset_critic",
    load_path="runs/wall-easy/mappo/models/model_1000000_steps_final.pt",
    total_timesteps=2_000_000,
    run_dir="runs/wall-easy/mappo-continuation",
    num_envs=1024,
    device="cuda",
)
```

Replace the path with the actual checkpoint saved by the training run. Final checkpoints are named `model_<steps>_steps_final.pt`; the actual counter can pass the requested budget by a collection batch.

`total_timesteps` is the final total counter, so resuming at one million with a target of two million adds about one million transitions. Loading requires the current checkpoint format and matching model settings. Checkpoints restore model parameters, counters, normalization, and optional optimizer state. When restoring the PPO optimizer, configure the same schedulers and automatic learning-rate controller settings used for training.

Full checkpoint loads restore SAC entropy temperature and adjustable auxiliary-loss coefficients even without optimizer state. Custom off-policy objectives also require the original per-target weights in `NextObsPredConfig`. Partial policy weight transfers retain the destination's NOP coefficients.

Off-policy replay starts empty after loading. A checkpoint taken before the first optimizer update uses random actions until `learning_starts`; a trained policy refills replay with its usual exploration.

Return/success moving averages and best-model eligibility carry across training calls and checkpoint continuation. `learn(..., episode_return_ema_alpha=...)` sets the smoothing factor for both averages; the default is `0.01`.

When rebuilding a trainer whose rollout warmup is unfinished, supply the original `rollout_warmup_steps_per_env`.

The `.pt` checkpoint is authoritative for saved counters and scores; the adjacent `.pt.json` is a quick-inspection summary.

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

The evaluation device selects where the simulation runs. The adapter transfers public observations to the actor's device, freezes observation statistics there on its first call, and returns actions to the simulator. The adapter shares the trained actor and preserves its training mode, so run evaluation after training or between updates.

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
training_metadata = json.loads((run_dir / "run_metadata_0.json").read_text(encoding="utf-8"))
training_seed = training_metadata["training_seed"]
benchmark_id = "SwarmBots-WallEasy-v0"
source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
trainer = make_training(benchmark_id, "mappo__deepset_critic", num_envs=2, device="cuda")
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
                "variant": "mappo__deepset_critic",
                "checkpoint": str(checkpoint),
                "training_seed": training_seed,
                "training_source_revision": training_metadata["source_revision"],
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

The two-world trainer restores the policy and normalization; each evaluation uses a separate 256-world environment. This recipe performs no training updates. The report includes the restored training-step count, training seed and training source provenance from the original run metadata. Other checkpoints require their original task, variant, model scale, NOP, distribution, and architecture settings in `make_training`.

The evaluation revision identifies the checkout used for evaluation; the training source provenance identifies the original training checkout. These can differ when evaluation happens later.

## Record a checkpoint

```bash
swarmbots record SwarmBots-WallEasy-v0 --checkpoint runs/wall-easy/mappo/models/model_1000000_steps_final.pt --variant mappo__deepset_critic --episodes 3 --output recordings/mappo
```

Use the actual checkpoint path and matching training settings. The Python equivalent is `swarmbots.learn.record_checkpoint(...)`. The [recording guide](recording.md) covers model-scale options, custom policies, and camera settings; [log plotting](log_plotting.md) covers learning curves.

## Custom training loops

`make_training` exposes the learning loop for custom control. The caller must close its environment:

```python
from swarmbots.learn import make_training

trainer = make_training("SwarmBots-Bridge-v0", "mat_dec", num_envs=1024)
try:
    trainer.learn(
        max_total_timesteps=10_000_000,
        run_dir="runs/bridge/mat-dec",
        enable_command_prompt=False,
    )
finally:
    trainer.env.close()
```

`trainer.learn` accepts one positive integer budget. Use `max_total_timesteps` for an absolute target or `additional_timesteps` for a further number of transitions from the current counter. A rollout can carry the counter past either target.

A `ScheduledEvaluationHook` evaluates the latest milestone reached by a completed rollout. After a recoverable hook failure, retry `learn` with the same trainer, hooks, and absolute target; completed training updates are retained.

The included collectors handle active-agent masks, terminal observations, and immediate autoreset. The [episode-handling guide](api.md#episode-lifecycle) and [standalone PPO example](../examples/README.md#train-without-the-included-learning-framework) cover independent learners.

## Differences from the original multi-agent methods

These learners adapt the original algorithms to SwarmBots' cooperative team objective. The following choices affect comparisons with original implementations:

- MAPPO and MAT, including `mat_orig`, estimate one shared team value per world instead of separate per-agent values.
- MADDPG/MATD3 use team critics. Their shared actor updates all active agents' current actions jointly through the team critic, so gradients flow through the entire joint action.
- MAT actors and critics use separate encoders. The `mat_orig` name identifies its autoregressive architecture lineage, not an exact reproduction of the original method.

The [references and implementation sources](references.md) identify the original formulations. The [model reference](policy_parameters/README.md) covers model dimensions, critic interactions, parameter accounting, and recurrent layouts. The [published thesis](https://resolver.obvsg.at/urn:nbn:at:at-ubl:1-108602) describes TMASAC, MAT-QCX, NOP, and signed-magnitude actions.
