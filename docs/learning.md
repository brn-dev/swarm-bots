# Reference learning baselines

For charts from the training CSVs, including comparisons across seeds, see [log plotting](log_plotting.md).

`swarmbots.learn` provides reference baselines and training helpers for SwarmBots experiments. You can use every registered task with PPO/MAPPO, DDPG/MADDPG, TD3/MATD3, SAC/MASAC, multi-agent transformer (MAT) policies, transformer-based SAC (TMASAC) and TD3 (TMATD3), and recurrent variants with LSTM, sLSTM, or mLSTM modules.

The benchmark's [tasks](scenarios.md) and [evaluation protocol](benchmark_protocol.md) are independent of these implementations. To integrate your own learner, use the [benchmark Python API](api.md). The presets below provide starting configurations; their training defaults are not required benchmark settings or published reference scores.

For more information about TMASAC, MAT-QCX, NOP, and SMB, see the [published master's thesis](https://resolver.obvsg.at/urn:nbn:at:at-ubl:1-108602). Its source and supplementary material are available in the [thesis repository](https://github.com/brn-dev/msc-thesis-swarmbots-qcx-tmasac-nop-smb).

The included on-policy learners build on [PPO](https://arxiv.org/abs/1707.06347), [MAPPO](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html), and [MAT](https://proceedings.neurips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html). TMASAC combines the thesis's architecture with [SAC](https://proceedings.mlr.press/v80/haarnoja18b.html) and its [automatic-temperature formulation](https://arxiv.org/abs/1812.05905). See [references and attribution](references.md) for citation guidance, prior attention-based multi-agent SAC methods, component credits, and reusable BibTeX entries.

The package also includes next-observation prediction (NOP), self-predictive representations (SPR), action distributions, rollout collection, replay, truncated backpropagation through time (TBPTT), PopArt value normalization, observation normalization, learning-rate scheduling, checkpointing, metrics, evaluation, and video recording. The examples below use the package's built-in presets; architectures and optimizer settings can be customized through the same API.

PyTorch 2.6 or newer is required. W&B logging and interactive command prompts are optional:

```bash
uv sync --extra wandb --extra prompt
```

## Differences from the original multi-agent methods

The included multi-agent learners are adaptations for SwarmBots' cooperative team objective, not fully faithful reproductions of the original formulations. In particular, MAT (including `mat_orig`), MAPPO, and the MADDPG/MATD3 baselines use global shared team value functions here, rather than the per-agent value estimates used in their original formulations. On-policy critics produce one scalar team V per world; off-policy critics produce one scalar team Q per critic, with TD3/SAC retaining two independent Q networks. A centralized critic can still estimate separate values for each agent in the original methods; centralization alone does not imply our shared team-value objective.

The [original MAT implementation](https://github.com/PKU-MARL/Multi-Agent-Transformer/blob/main/mat/algorithms/mat/algorithm/ma_transformer.py) produces a value estimate for each agent token using a shared network. SwarmBots' `mat_orig` pools its token-level value predictions into a single team value, while the other MAT variants use a pooled team critic. Sharing network parameters and sharing a single team value are different choices; the `mat_orig` name identifies the autoregressive architecture lineage, not an exact reproduction of the original value formulation.

The actor update also differs for multi-agent TD3 (MATD3) and MADDPG. In the original implementations, an update for agent i replaces only that agent's replay action with its current actor action and evaluates its per-agent critic; all other agents' actions remain fixed to their replay values. Our shared actor instead generates current actions for all active agents at the replayed observations and passes the entire joint action through the shared team critic at once. The policy gradient therefore flows through every active agent's action. MATD3/TMATD3 use the first of their twin team critics for this actor objective. See [implementation sources](references.md#deterministic-baseline-implementation-sources).

These changes affect the value objective and policy gradient and can affect learning behavior and results. Report the shared team-value formulation, joint actor update, and selected architecture when comparing these baselines with the original methods.

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

For runnable training → evaluation → video-recording workflows, see the [examples](../examples/README.md). `examples/train_policy.py` accepts any public preset; `examples/train_recurrent_td3.py` demonstrates recurrent TD3 actor/critic options and sequence replay settings.

| Variant | Learner and architecture |
| --- | --- |
| `ppo`, `ppo_small` | PPO with a joint MLP actor |
| `mappo`, `mappo_small` | MAPPO with a per-agent MLP actor and masked Deep Set critic |
| `mappo_mlp`, `mappo_mlp_small` | Same MAPPO actor sizes with a flattened MLP critic |
| `mat_orig`, `mat_ind`, `mat_dec` | PPO with autoregressive, independent-head, or decentralized MAT |
| `mat_qcx` | PPO with the QCX autoregressive query/context decoder |
| `mat_ind_no_attention` | MAT-IND with agent attention disabled |
| `mat_ind_lstm`, `mat_qcx_lstm` | Recurrent PPO with LSTM and TBPTT |
| `maddpg_mlp` (no NOP) | DDPG with a shared per-agent deterministic actor and one centralized MLP critic |
| `maddpg_deepset` | DDPG with a shared per-agent deterministic actor and one centralized Deep Set critic |
| `matd3_mlp` (no NOP) | TD3 with a shared per-agent deterministic actor and two independent centralized MLP critics |
| `matd3_deepset` | TD3 with a shared per-agent deterministic actor and two independent centralized Deep Set critics |
| `masac_mlp` (no NOP) | SAC with a shared per-agent stochastic actor and two independent centralized MLP critics |
| `masac_deepset` | SAC with a shared per-agent stochastic actor and two independent centralized Deep Set critics |
| `tmatd3` | TD3 with a transformer actor and two independent transformer critics |
| `tmatd3_dec` | TD3 with a per-agent MLP actor and the same transformer critics as `tmatd3` |
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

MAT/RMAT observation encoders and the transformer critics in TMASAC/TMATD3 use separate local and global projections added together by default. Set `policy_kwargs={"mat_joint_obs_embedding": True}` to concatenate the separately normalized local and global inputs before the per-agent token projection. Joint mode uses the local encoder's feed-forward configuration; the presets use MLP input encoders. Action-conditioned critics include public and privileged globals in this projection, after any optional observation/action pre-encoders. The switch also applies to recurrent and shared observation encoders, and to MAT-based actors in the other off-policy baselines. It introduces no new variant names. Low-level `MATEncoderConfig.joint_obs_embedding` (inherited by `RMATEncoderConfig`) controls the same setting.

The default `mat_joint_obs_embedding=False` retains the additive architecture and existing checkpoint weight shapes and keys. Reconstruct a joint-embedding checkpoint with the switch enabled, including when using `record_checkpoint`. Encoders without global inputs retain their existing layout in either mode. `mat_orig` already concatenates local and global observations, and PPO/MAPPO use their existing input encoders independently of this switch.

All MAPPO presets include non-privileged global observations in the shared per-agent encoder. `mappo` and `mappo_small` use Deep Set critics that condition each element encoder on privileged global variables, while retaining a direct privileged-global input to the value regressor after masked mean pooling. `mappo_mlp` and `mappo_mlp_small` flatten the masked per-agent features and local privileged variables into a fixed-slot MLP input, alongside privileged globals. Their critic hidden widths are `[256, 128]` and `[192, 128]`, respectively; their input layer grows with the padded agent count. MLP critics depend on agent order; Deep Set critics are permutation invariant. Corresponding regular/small presets have identical actor configurations, PPO settings, and action distributions.

NOP remains enabled by default for both MAPPO critic types: it uses the shared observation-encoder features, which remain independent of privileged inputs. PopArt is supported by both critics. Set `policy_kwargs={"mappo_critic_context_in_elements": False}` for the previous Deep Set layout or to reconstruct checkpoints trained with that layout; early conditioning changes the critic element encoder's input weight shapes when privileged globals are present. Low-level `MAPPOCriticConfig.context_in_elements` controls the same option for Deep Set critics; the MLP critic receives its global context directly in its input.

The sLSTM presets use a residual connection around the temporal module; `tmasac_slstm_no_residual` disables it. They disable memory-strength inputs to the critic. To enable those inputs, set `policy_kwargs["tmasac_actor_state_critic_input_config"]` to an `ActorStateCriticInputConfig` with `include_slstm_memory_strength=True`. Import that configuration class from `swarmbots.learn.algos.sac.recurrent_tmasac_policy`.

The standard recurrent TMASAC presets (`tmasac_lstm`, `tmasac_slstm`, and their architecture variants) already condition their feed-forward critic on the actor's detached hidden state and memory. `tmasac_lstm_no_actor_state` disables this input explicitly. The recurrent shared observation encoder preset supplies temporal latents directly to actor and critic instead of adding a separate actor-state projection. Plain feed-forward TMASAC has no recurrent actor state.

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
| Rollout steps per world per iteration | 4 for PPO, 1 for SAC/DDPG/TD3 |
| Training budget in `train` | 100 million transitions |

Learning presets enable NOP by default, except for the flattened MLP baselines marked **(no NOP)** and the explicit `*_no_nop` ablations. Deep Set and transformer MADDPG/MATD3/MASAC/TMATD3 presets use critic NOP by default; flattened MLP critics reject NOP. These off-policy baselines do not support PopArt. On-policy presets (PPO, MAPPO, and MAT) use Signed-Magnitude Beta (`sign_magnitude_beta`). All SAC presets, including MASAC and recurrent TMASAC, use Predicted Std Gaussian (`predicted_std_gaussian`). For stochastic policies, action distributions are configured independently of variant names through `continuous_action_dist`; `use_nop` overrides NOP on supported policies. DDPG/TD3 use deterministic tanh actors with Gaussian exploration noise. Learning rate, replay capacity, minibatch sizes, and recurrent sequence settings depend on the selected preset and can be overridden with `algorithm_kwargs`.

SAC, MASAC, and TMASAC retain `policy_delay=1`, `target_policy_noise=0.0`, and `target_noise_clip=0.0`. The actor and temperature update every critic step, and target actions receive no additional noise. These defaults also apply to recurrent SAC. You can opt into delayed updates and target smoothing through `algorithm_kwargs`:

```python
algorithm_kwargs={
    "policy_delay": 2,
    "target_policy_noise": 0.2,
    "target_noise_clip": 0.5,
}
```

For SAC, the delay applies to the actor, actor NOP, and automatic temperature updates. Critics and critic/shared-encoder NOP still update every step. Target networks retain their separate `target_update_interval` schedule. Target smoothing adds clipped Gaussian noise in normalized action coordinates, scales it to the physical action bounds, and keeps inactive actions zero. It changes the actions evaluated by the target critic; the entropy term uses the original policy sample's log-probability. This is an experimental modification of the SAC target, so compare it against the unchanged defaults for your task. Set both noise options to positive values to enable smoothing.

## DDPG, TD3 and centralized critic baselines

The names use underscores in Python: for example, MASAC-mlp is `masac_mlp`, and TMATD3-dec is `tmatd3_dec`. `DDPG` and `TD3` are available directly from `swarmbots.learn.algos.td3`. They support the single-agent case as well as joint swarm actions; the public multi-agent presets use the `maddpg` and `matd3` prefixes.

All of these presets learn a scalar team value from SwarmBots' shared reward. Actor parameters are shared across agents, and the team policy gradient differentiates through every active agent's action. This is a cooperative adaptation of MADDPG/MATD3: the original implementations use separate actors and per-agent critics and hold other agents' replay actions fixed during each actor's update. These presets do not implement mixed competitive rewards or policy ensembles. See [references and implementation sources](references.md#deterministic-baseline-implementation-sources).

The critic architecture suffix selects:

- **`_mlp`:** zero inactive local observations, privileged local fields, and actions; flatten them in padded agent-slot order; concatenate global observations, privileged global fields, and the complete agent mask. Each Q network has three 512-wide hidden layers. This critic is sensitive to agent order and is built for the task's fixed maximum number of slots.
- **`_deepset`:** encode each active agent's local observation, privileged local fields, and action using three shared 512-wide layers. Masked mean pooling produces a set representation. Concatenate global observations, privileged global fields, and the active-agent count, then apply three more 512-wide layers and a scalar Q head. Each critic is invariant to a joint permutation of agent observations, actions, and masks.

The decentralized actors use the same per-agent residual MLP backbone as `tmasac_dec`: 256-dimensional features, two residual blocks with 512/512 feed-forward layers, and a 128-wide action head. They receive local and non-privileged global observations. `tmatd3` enables attention across agents in this backbone; `tmatd3_dec` disables actor attention. Both use the action-conditioned transformer critic from TMASAC, sharing one encoder between their two Q heads by default. Their default transformer width is 256, with four attention heads and two layers. Privileged observations are confined to the critics.

Transformer critic sharing is configurable for both algorithms with `policy_kwargs["critic_independent_encoders"]`. TMASAC and TMATD3 both default to `False`, sharing one action-conditioned encoder between their two Q heads. Set it to `True` for independent critic encoders. This option is separate from TMASAC's optional shared observation encoder between actor and critic. Changing encoder sharing changes checkpoint parameter shapes and keys; reconstruct a checkpoint with the setting used during training. Earlier TMATD3 checkpoints trained with independent encoders require `critic_independent_encoders=True`.

DDPG uses one critic, an actor target, and a critic target; actor and targets update on every gradient step. TD3 uses two critics, the minimum target Q, target-action smoothing (`target_policy_noise=0.2`, `target_noise_clip=0.5`), and actor/target updates every second critic step (`policy_delay=2`). Both optimize the actor against Q1 and use `tau=0.005`. The delay counter continues across training iterations and checkpoint reloads. Rollout exploration defaults to `exploration_noise=0.1`; all noise scales use normalized action coordinates and are converted to the physical action bounds before clipping. Inactive actions remain zero. Deterministic evaluation omits exploration noise.

```python
from swarmbots.learn import make_training

trainer = make_training(
    "SwarmBots-WallEasy-v0",
    "matd3_deepset",
    policy_kwargs={
        "baseline_critic_hidden_dims": (512, 512, 512),
        "baseline_critic_element_hidden_dims": (512, 512, 512),
    },
    algorithm_kwargs={
        "exploration_noise": 0.1,
        "policy_delay": 2,
        "target_policy_noise": 0.2,
        "target_noise_clip": 0.5,
    },
)
```

`baseline_critic_element_hidden_dims` controls the Deep Set element encoder; `baseline_critic_hidden_dims` controls the flattened MLP or the Deep Set value regressor. DDPG fixes the delay to one and target smoothing to zero. DDPG/TD3 require finite continuous Box action spaces, including continuous connectors. Passing `continuous_action_dist` or SAC entropy options to a deterministic preset raises an error. MASAC retains SAC's distribution choices, entropy tuning, and multiple-action-sample support.

Deep Set baseline critics condition each element encoder on the non-privileged global observations, privileged global variables, and active-agent count. The same context also enters the value regressor directly after masked mean pooling. This gives the per-agent NOP features access to global conditions. Set `policy_kwargs={"baseline_critic_context_in_elements": False}` for the previous layout, where global context enters only after pooling. Use this setting when reconstructing checkpoints trained with that previous layout; early conditioning changes the element encoder's input weight shapes. Flattened MLP and transformer critics retain their existing input layouts.

Next-observation prediction is enabled by default on `maddpg_deepset`, `matd3_deepset`, `masac_deepset`, `tmatd3`, and `tmatd3_dec`. Pass `use_nop=False` to `make_training` or `train` to disable it. NOP defaults to the critic source. Set `policy_kwargs={"tmasac_nop_latent_source": "actor"}` or `"both"` to train actor representations as well; this option applies to DDPG/TD3 too. Deep Set NOP uses per-agent features before pooling, and transformer NOP uses action-conditioned encoder features. Independent critic features are concatenated along the feature axis; shared transformer critics supply the common encoder features once. Critic NOP follows `nop_skip_first_transition_for_critic`, which defaults to true because these features already include the current action.

NOP uses the existing replay windows and episode masks. `algorithm_kwargs` accepts `nop_batch_size` and `independent_nop_sampling` when NOP is enabled. DDPG trains actor and critic NOP every step. TD3 trains critic NOP every step and actor NOP on delayed actor updates. Auxiliary losses train only online networks. The `*_mlp` presets reject `use_nop=True` for every latent source.

When loading or recording a checkpoint created with NOP disabled, reconstruct the policy with `use_nop=False`.

TD3 can use recurrent actors with any MATD3/TMATD3 critic. Enable `policy_kwargs["td3_recurrent_actor"]`; the existing `rmat_temporal_model_cls`, `rmat_temporal_model_config`, and other `rmat_*` options configure its temporal module. TMATD3 additionally supports a recurrent critic through `td3_recurrent_critic=True`, which requires the recurrent actor option. Existing presets remain feed-forward unless these flags are set.

With a recurrent actor and a feed-forward critic, TD3 now supplies actor state to the critic by default, matching standard recurrent TMASAC. LSTM inputs contain hidden and cell states; sLSTM inputs contain hidden state and normalized memory. A critic-owned projection encodes these detached features, so critic training does not update actor parameters through this input. The target critic receives the target actor's corresponding state, including at truncation successors. This applies to MLP, Deep Set, and transformer critics. Set `policy_kwargs["td3_actor_state_critic_input_config"]=None` to disable it, or provide an `ActorStateCriticInputConfig` from `swarmbots.learn.algos.off_policy.actor_state_critic` to customize it. A recurrent critic uses its own history and defaults to no additional actor-state input. Actor-state inputs support LSTM and sLSTM; disable them for other temporal modules. Earlier recurrent TD3 checkpoints without actor-state conditioning require this option set to `None`.

```python
trainer = make_training(
    "SwarmBots-WallEasy-v0",
    "tmatd3",
    policy_kwargs={
        "td3_recurrent_actor": True,
        "td3_recurrent_critic": True,
        "critic_independent_encoders": False,
    },
    algorithm_kwargs={
        "batch_size": 16,
        "burn_in_steps": 32,
        "learning_steps": 64,
        "temporal_state_store_interval": 32,
    },
)
```

The trainer selects `RecurrentTD3` automatically, defaulting to 16 replay sequences per batch, 32 burn-in steps, 64 learning steps, and float16 temporal-state anchors every 16 steps. Override these values with `algorithm_kwargs`. `RecurrentTD3`, `RecurrentTD3Policy`, and `RecurrentTD3PolicyConfig` are also available from `swarmbots.learn.algos.td3`. Recurrent SAC and TD3 share sequence replay, burn-in, episode reset masks, truncation bootstrap handling, and NOP windows. TD3 stores separate online and target actor history states in replay; action-conditioned critic histories advance using replay actions, with policy actions evaluated on branches from that history. Its actor and target updates retain the TD3 policy delay.

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
- `algorithm_kwargs` overrides PPO/SAC/RecurrentSAC/DDPG/TD3 constructor arguments, including learning rate, minibatches, rollout warmup, replay capacity, exploration noise, policy delay, burn-in, and learning sequence length. For recurrent PPO, sampler batch sizes count sequence rows rather than transitions.
- `learn_kwargs` on `train` forwards logging, save, and evaluation-hook settings to `BaseAlgorithm.learn`. Custom `extra_run_metadata` is merged with the task and variant identifiers. `total_timesteps` and `run_dir` belong to `train` and take precedence over the forwarded options. Logging to W&B starts only when explicitly configured.
- `compile_modules=True` enables policy/world-model compilation. Configure the CUDA compiler first; see [GPU setup](gpu_setup.md).

The simulator and learning wrappers use Gymnasium `SAME_STEP` autoreset. Done-step returned observations belong to reset episodes; terminal observations are in `final_obs` with `_final_obs`. The supplied collectors, replay buffers, and wrappers already handle this convention.

### Logging levels

Choose console and persistence levels independently, through both `train(...)` and `trainer.learn(...)`. The defaults are **`minimal` for the console and `full` for persistence**. Persistence covers both CSV and W&B. All algorithms and recurrent variants use the same presets:

| Level | Console | CSV/W&B |
| --- | --- | --- |
| `full` | All available metric fields with their original names and summary statistics | All metrics |
| `minimal` | Progress, return/success EMAs, core losses, algorithm health diagnostics, learning rate, and throughput, with short labels | The important training metrics plus raw return/success metrics and timestamp/counter context |
| `return_success` | Progress counters, return/success EMAs, mean episode return, best return EMA, and current episode success rate | Only return/success metrics and context: timestamps, learning-start time, iterations, environment steps, optimizer-update counts, and episode counts |

```python
from swarmbots.learn import train

trainer = train(
    "SwarmBots-WallEasy-v0",
    "matd3_deepset",
    total_timesteps=1_000_000,
    run_dir="runs/wall-easy/td3",
    learn_kwargs={
        "logging_console_level": "minimal",
        "logging_persistence_level": "return_success",
    },
)
```

The same settings go directly to `trainer.learn(...)`. Console selection does not restrict persisted fields, and persistence selection does not restrict console output. W&B's configured step field is retained by reduced persistence levels. `logging_ignore_keys_for_persistence` can explicitly exclude more metrics.

Reduced console presets use iteration (`it`), environment steps (`steps`), total optimizer updates (`upd`), return EMA (`ret`), and success EMA in percent (`succ%`). Minimal adds learning rate (`lr`) and throughput (`fps`). TD3/DDPG/SAC show critic and actor losses (`q_loss`, `pi_loss`), policy and target Q values (`q_pi`, `q_tgt`), and any scaled NOP losses (`c_nop`, `a_nop`). Replay size (`replay`), random-action collection (`rnd`), skipped training (`skip`), and the fraction of optimizer steps that update the actor (`pi_upd`) expose warmup and delayed-update behavior. `skip` appears only when emitted; `rnd` and `skip` use 0/1 flags. SAC also shows its entropy coefficient (`alpha`), estimated policy entropy (`ent`), and target entropy (`ent_tgt`). PPO adds policy/value losses (`pi_loss`, `v_loss`), approximate KL (`kl`), clip fraction (`clip`), explained variance (`ev`), and the scaled world-model loss (`nop`) when present. `clip` is a fraction, not a percentage. Summary-statistic metrics show their means. Unavailable algorithm-specific fields are skipped, and EMAs print `n/a` before they are available. The `return_success` console preset also shows `ret_mean`, `best_ret`, and `succ_batch%` when present; log messages carry their timestamp in the console prefix.

Persisted metrics retain their original names and all available statistics for selected metrics, including means, standard deviations, counts, and histograms. Reduced persistence presets also retain per-scenario return/success metrics and episode counts. When resuming an existing CSV, earlier rows and columns remain; excluded fields are blank in newly written rows. Levels affect metric records; checkpoints, run metadata, and messages such as checkpoint saves are independent of these settings.

The training examples expose both choices:

```bash
python examples/train_policy.py SwarmBots-WallEasy-v0 --variant matd3_deepset --console-log-level return_success --persistent-log-level minimal
```

For a custom console selection, override the preset with `logging_console_keys`:

```python
trainer = train(
    "SwarmBots-WallEasy-v0",
    "matd3_deepset",
    total_timesteps=1_000_000,
    run_dir="runs/wall-easy/td3",
    learn_kwargs={
        "logging_console_keys": [
            ("timesteps", ",d", "steps"),
            ("ep_rew_ema", ".2f", "ret"),
            ("ep_success_rate_ema", ".1f", "succ%"),
            ("learning_rate", ".2e", "lr"),
            ("fps", ".0f", "fps"),
        ],
    },
)
```

An entry can be a key string, `(key, format)`, or `(key, format, display_name)`. Use `None` as the format for automatic formatting. For a metric containing summary statistics, such as `critic_loss`, use `SummaryStatisticsFormat(mean=".3f")` from `swarmbots.learn.summary_statistics` to select the mean and its precision. Missing explicitly requested keys warn once so typos are visible. Shared field selections and display formats live in `swarmbots.learn.logging_levels`.

Pass `logging_console_keys=None` to show every metric, `logging_console_keys=[]` to silence metric rows, or `logging_console_keys="default"` (the default) to follow `logging_console_level`. Custom console selections do not change `logging_persistence_level`. With `trainer.learn(...)`, pass the option directly instead of through `learn_kwargs`.

## Action distributions

For stochastic policies, the `continuous_action_dist` selector supports six distributions: `sign_magnitude_beta`, `gumbel_softmax_sign_magnitude_beta`, `beta`, `predicted_std_gaussian`, `gsde`, and `squashed_diag_gaussian`. Discrete action spaces use categorical or Bernoulli distributions. DDPG/TD3 instead use a deterministic action head and the noise settings above.

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

`total_timesteps` is the absolute final counter, so this example adds 50 million transitions. Checkpoints restore model parameters, optimizer state, counters, and normalization, including DDPG/TD3 actor and critic target networks. SAC/DDPG/TD3 replay is intentionally not checkpointed; it is rebuilt and refilled after loading. Lower-level `trainer.load(...)` also supports the existing policy-state transforms and transfer options.

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
