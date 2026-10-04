# Optional learning baselines

`swarmbots.learn` provides optional baselines and training helpers for SwarmBots experiments. You can use every registered task with Proximal Policy Optimization (PPO), Soft Actor-Critic (SAC), multi-agent transformer (MAT) policies, transformer-based SAC (TMASAC), and recurrent variants with LSTM, sLSTM, or mLSTM modules.

The benchmark's [tasks](scenarios.md) and [evaluation protocol](benchmark_protocol.md) are independent of these implementations. To integrate your own learner, use the [benchmark Python API](api.md). The presets below provide starting configurations; their training defaults are not required benchmark settings or published reference scores.

For more information about TMASAC, MAT-QCX, NOP, and SMB, see the [published master's thesis](https://resolver.obvsg.at/urn:nbn:at:at-ubl:1-108602). Its source and supplementary material are available in the [thesis repository](https://github.com/brn-dev/msc-thesis-swarmbots-qcx-tmasac-nop-smb).

The included on-policy learners build on [PPO](https://arxiv.org/abs/1707.06347), [MAPPO](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html), and [MAT](https://proceedings.neurips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html). TMASAC combines the thesis's architecture with [SAC](https://proceedings.mlr.press/v80/haarnoja18b.html) and its [automatic-temperature formulation](https://arxiv.org/abs/1812.05905). See [references and attribution](references.md) for citation guidance, prior attention-based multi-agent SAC methods, component credits, and reusable BibTeX entries.

The package also includes next-observation prediction (NOP), self-predictive representations (SPR), action distributions, rollout collection, replay, truncated backpropagation through time (TBPTT), PopArt value normalization, observation normalization, learning-rate scheduling, checkpointing, metrics, evaluation, and video recording. The examples below use the package's built-in presets; architectures and optimizer settings can be customized through the same API.

PyTorch 2.6 or newer is required. W&B logging and interactive command prompts are optional:

```bash
uv sync --extra wandb --extra prompt
```

## Select a variant and task

Every registered benchmark ID can be passed to the training helpers:

```python
from swarmbots.learn import list_variants, train

print(list_variants())
trainer = train(
    "SwarmBots-POWallMedium-v0",
    "mat_qcx",
    num_envs=1024,
    device="cuda",
    compile_modules=True,
    total_timesteps=100_000_000,
    run_dir="runs/po-wall-medium/mat-qcx",
)
```

This `mat_qcx` preset includes NOP and Signed-Magnitude Beta (SMB) action distributions. `train` closes the simulator when training ends or fails; the returned trainer retains the policy and normalization statistics. The run directory contains metrics, metadata, and the final checkpoint with optimizer and normalization state.

| Variant | Learner and architecture |
| --- | --- |
| `ppo`, `ppo_small` | PPO with a joint MLP actor |
| `mappo`, `mappo_small` | MAPPO with a per-agent MLP actor and centralized critic |
| `mat_orig`, `mat_ind`, `mat_dec` | PPO with autoregressive, independent-head, or decentralized MAT |
| `mat_qcx` | PPO with the QCX autoregressive query/context decoder |
| `mat_ind_no_attention` | MAT-IND with agent attention disabled |
| `mat_ind_lstm`, `mat_qcx_lstm` | Recurrent PPO with LSTM and TBPTT |
| `tmasac` | SAC with a transformer actor and transformer twin critic |
| `tmasac_dec` | SAC with a per-agent MLP actor and the same transformer twin critic as `tmasac` |
| `tmasac_slstm`, `tmasac_lstm` | Recurrent TMASAC with sLSTM or LSTM and actor-state critic inputs |
| `tmasac_shared_encoder`, `tmasac_slstm_shared_encoder` | Feed-forward or sLSTM TMASAC with a shared observation encoder |
| `tmasac_slstm_no_residual` | sLSTM TMASAC without a residual connection around the temporal module |
| `tmasac_swiglu`, `tmasac_slstm_swiglu` | Feed-forward or sLSTM TMASAC with SwiGLU feed-forward layers |
| `tmasac_lstm_no_actor_state` | LSTM TMASAC without actor-state critic inputs |
| `tmasac_segment` | Feed-forward TMASAC trained on episode segments |
| `mat_qcx_no_nop`, `tmasac_no_nop`, `tmasac_slstm_no_nop` | Corresponding architecture with NOP disabled |

`list_variants()` returns only these canonical names. Architecture names come first, followed by the recurrent module and any ablation suffix. The old development names and duplicate aliases are no longer accepted. `mappo` selects the MAPPO implementation; its former MAT-IND configuration is now `mat_ind_no_attention`.

The decentralized actors in `mat_dec` and `tmasac_dec` use each agent's local observation plus the task's non-privileged global observation. They do not mix other agents' local observations or actions. The `tmasac_dec` actor uses residual MLP blocks with agent attention disabled; its critic retains agent attention, and its NOP and optimizer defaults match `tmasac`.

The sLSTM presets use a residual connection around the temporal module; `tmasac_slstm_no_residual` disables it. They disable memory-strength inputs to the critic. To enable those inputs, set `policy_kwargs["tmasac_actor_state_critic_input_config"]` to an `ActorStateCriticInputConfig` with `include_slstm_memory_strength=True`. Import that configuration class from `swarmbots.learn.algos.sac.recurrent_tmasac_policy`.

## Training defaults

Task settings come from the [scenario registry](scenarios.md). Architecture and optimizer settings come from the selected learning preset. Unless overridden, the helpers use:

| Setting | Default |
| --- | --- |
| Parallel worlds | 1024 |
| Device | `cuda`; select `cpu` explicitly for development |
| Random seed | 42 |
| Episode limit | 500 control steps, with physics settling before each episode |
| Morphology population | The registered task's pool; see the [benchmark protocol](benchmark_protocol.md#morphology-population) |
| Policy/world-model compilation | Disabled; enable with `compile_modules=True` |
| Initial episode staggering | Disabled |
| Rollout warmup | 0 steps |
| Rollout steps per world per iteration | 4 for PPO, 1 for SAC |
| Training budget in `train` | 100 million transitions |

Most presets enable NOP. On-policy presets (PPO, MAPPO, and MAT) use Signed-Magnitude Beta (`sign_magnitude_beta`). All SAC presets, including recurrent TMASAC, use Predicted Std Gaussian (`predicted_std_gaussian`). Action distributions are configured independently of variant names through `continuous_action_dist`; `use_nop` overrides NOP. Learning rate, replay capacity, minibatch sizes, and recurrent sequence settings depend on the selected preset and can be overridden with `algorithm_kwargs`.

## Customize training

Use `make_training` when you need access to the environment, policy, or learning loop:

```python
from swarmbots.learn import make_training

trainer = make_training(
    "SwarmBots-Bridge-v0",
    "mat_dec",
    num_envs=256,
    device="cuda",
    use_nop=False,
    continuous_action_dist="predicted_std_gaussian",
    rollout_steps_per_env=16,
    algorithm_kwargs={"n_epochs": 4, "learning_rate": 1e-4},
)
try:
    trainer.learn(
        max_total_timesteps=10_000_000,
        run_dir="runs/bridge/mat-dec",
        enable_command_prompt=False,
    )
finally:
    trainer.env.close()
```

- `scenario_kwargs` overrides scenario construction; `env_kwargs` configures the vector simulator.
- `policy_kwargs` overrides the preset policy builder's architecture arguments, such as `enc_d_model`, `dec_d_model`, `mat_encoder_transformer_ff_config`, `rmat_temporal_model_cls`, and `rmat_temporal_model_config`. Low-level policy config dataclasses and classes remain available under `swarmbots.learn.algos` for custom architectures.
- `algorithm_kwargs` overrides PPO/SAC/RecurrentSAC constructor arguments, including learning rate, minibatches, rollout warmup, replay capacity, burn-in, and learning sequence length. For recurrent PPO, sampler batch sizes count sequence rows rather than transitions.
- `learn_kwargs` on `train` forwards logging, save, and evaluation-hook settings to `BaseAlgorithm.learn`. Custom `extra_run_metadata` is merged with the task and variant identifiers. `total_timesteps` and `run_dir` belong to `train` and take precedence over the forwarded options. Logging to W&B starts only when explicitly configured.
- `compile_modules=True` enables policy/world-model compilation. Configure the CUDA compiler first; see [GPU setup](gpu_setup.md).

The simulator and learning wrappers use Gymnasium `SAME_STEP` autoreset. Done-step returned observations belong to reset episodes; terminal observations are in `final_obs` with `_final_obs`. The supplied collectors, replay buffers, and wrappers already handle this convention.

## Action distributions

The `continuous_action_dist` selector supports six distributions: `sign_magnitude_beta`, `gumbel_softmax_sign_magnitude_beta`, `beta`, `predicted_std_gaussian`, `gsde`, and `squashed_diag_gaussian`. Discrete action spaces use categorical or Bernoulli distributions.

Predicted Std Gaussian predicts both the mean and standard deviation from the actor's features. The action factory applies tanh squashing to keep actions within the simulator's bounds. Its low-level classes are `PredictedStdGaussianConfig` and `PredictedStdGaussianActionDist` in `swarmbots.learn.action_dists.predicted_std_gaussian_action_dist`.

To compare SAC with SMB, keep the same architecture and override the distribution:

```python
trainer = make_training(
    "SwarmBots-POWallMedium-v0",
    "tmasac_slstm",
    continuous_action_dist="gumbel_softmax_sign_magnitude_beta",
    device="cuda",
)
```

Similarly, use `"mat_qcx"` with `continuous_action_dist="gsde"` to select state-dependent exploration.

Standard diagonal/squashed Gaussians, discrete/categorical distributions, and Bernoulli distributions are available directly under `swarmbots.learn.action_dists`. Binary connector spaces select Bernoulli; continuous connector spaces use the continuous distribution. The registry's default connectors are continuous. SAC requires a pathwise or straight-through gradient estimator; ordinary SMB's categorical sample is intended for PPO, and Gumbel SMB is its SAC counterpart. Unsupported SAC distributions raise an error during construction.

## Continue a checkpoint

Recreate the same variant, task, and architecture, then load the checkpoint:

```python
trainer = train(
    "SwarmBots-POWallMedium-v0",
    "tmasac_slstm",
    load_path="runs/po-wall/slstm/models/model_100000000_steps_final.pt",
    total_timesteps=150_000_000,
    run_dir="runs/po-wall/slstm-continuation",
    device="cuda",
    compile_modules=True,
)
```

`total_timesteps` is the absolute final counter, so this example adds 50 million transitions. Checkpoints restore model parameters, optimizer state, counters, and normalization. SAC replay is intentionally not checkpointed; it is rebuilt and refilled after loading. Lower-level `trainer.load(...)` also supports the existing policy-state transforms and transfer options.

## Evaluate a trained policy

Use the benchmark adapter to freeze observation normalization, maintain per-lane recurrent state, and convert packed policy actions to the simulator's action dictionary:

```python
from swarmbots import evaluate_policy
from swarmbots.learn import as_benchmark_policy

result = evaluate_policy(
    as_benchmark_policy(trainer, deterministic=True),
    "SwarmBots-POWallMedium-v0",
    num_envs=256,
    num_episodes=256,
    seed=1000,
    device="cuda",
    action_mode="deterministic",
)
print(result.to_dict())
```

Create a fresh adapter for each evaluation call and use the policy's device. It passes only `local_obs`, `global_obs`, and `agent_mask` to the actor; privileged fields remain unavailable. Normalization statistics and the actor's training mode are preserved. The adapter owns its recurrent state and previous actions, resetting them at episode boundaries. Stochastic gSDE evaluation resamples noise each step. The actor is shared, so do not train and evaluate it concurrently.

This example evaluates one protocol seed. For a complete [benchmark report](benchmark_protocol.md), repeat it with a fresh adapter for each seed `1000` through `1004` and aggregate the five results as shown in the [reporting example](api.md#five-seed-report). Record the checkpoint identifier and architecture in `policy_metadata`, and the benchmark Git commit in `source_revision`.

For evaluation during training, `FrozenEvaluationRunner` runs a policy with frozen normalization and `ScheduledEvaluationHook` triggers it at configured intervals. These helpers are available in `swarmbots.learn.evaluation`; use `evaluate_policy` for protocol scores.

## Record a checkpoint

`swarmbots record <benchmark-id> --checkpoint <path> --variant <variant> --episodes 5` records a saved policy with frozen normalization. The Python equivalent is `swarmbots.learn.record_checkpoint(...)`. Recording builds only the policy and environment, and handles recurrent state without allocating training buffers. See [recording](recording.md) for the checkout script, custom policies, and rendering options.
