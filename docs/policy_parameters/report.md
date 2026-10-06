# Policy parameter counts

Task: `SwarmBots-WallEasy-v0`. Instantiated 35 of 35 requested registered variants, with one environment on `cpu` and compilation disabled. Generated 2026-10-06.

Counts are actual unique PyTorch parameters, including weights and biases. Actor includes action-distribution heads and learned distribution parameters. Shared encoder is counted separately from both actor and critic. NOP (next-observation prediction) counts its additional projections, transition model, and prediction heads; its source encoder remains in actor, critic, or shared encoder. Targets are the frozen actor/critic/shared-encoder copies used by off-policy learning.

Actor + critic + shared encoder + NOP + targets + other = total. **Trainable − NOP** is the trainable total minus trainable NOP parameters. It excludes all frozen parameters, including target networks. Trainable excludes frozen parameters. Actor execution needs actor + shared encoder. Counts describe the entire policy, not a separate network per agent. Buffers (including normalization/PopArt state), optimizer state, replay, and activations are excluded. Parameter MiB below uses the tensors' actual dtypes and is not runtime or checkpoint memory.

Sizes depend on task shapes and architecture settings. These are the registered preset defaults unless overrides below are nonempty. A residual-connection ablation can have the same count.

Overrides: `{"continuous_action_dist": null, "env_kwargs": {}, "policy_kwargs": {}, "scenario_kwargs": {}, "use_nop": null}`.

| Variant | Actor | Critic | Shared encoder | NOP | Targets | Other | Total | Trainable | Trainable − NOP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ppo | 44,072 | 317,441 | 790,336 | 417,716 | 0 | 0 | 1,569,565 | 1,569,565 | 1,151,849 |
| ppo_small | 43,880 | 157,441 | 545,824 | 394,100 | 0 | 0 | 1,141,245 | 1,141,245 | 747,145 |
| mappo | 58,696 | 166,273 | 1,007,232 | 471,092 | 0 | 0 | 1,703,293 | 1,703,293 | 1,232,201 |
| mappo_small | 55,480 | 130,609 | 660,864 | 471,092 | 0 | 0 | 1,318,045 | 1,318,045 | 846,953 |
| mappo_mlp | 58,696 | 366,337 | 1,007,232 | 471,092 | 0 | 0 | 1,903,357 | 1,903,357 | 1,432,265 |
| mappo_mlp_small | 55,480 | 274,817 | 660,864 | 471,092 | 0 | 0 | 1,462,253 | 1,462,253 | 991,161 |
| mat_orig | 409,160 | 199,427 | 1,204,736 | 471,092 | 0 | 0 | 2,284,415 | 2,284,415 | 1,813,323 |
| mat_ind | 42,184 | 198,913 | 1,204,736 | 471,092 | 0 | 0 | 1,916,925 | 1,916,925 | 1,445,833 |
| mat_dec | 737,096 | 1,403,649 | 0 | 471,092 | 0 | 0 | 2,611,837 | 2,611,837 | 2,140,745 |
| mat_qcx | 623,048 | 198,913 | 1,204,736 | 471,092 | 0 | 0 | 2,497,789 | 2,497,789 | 2,026,697 |
| mat_ind_lstm | 42,184 | 198,913 | 2,911,232 | 471,092 | 0 | 0 | 3,623,421 | 3,623,421 | 3,152,329 |
| mat_qcx_lstm | 623,048 | 198,913 | 2,911,232 | 471,092 | 0 | 0 | 4,204,285 | 4,204,285 | 3,733,193 |
| maddpg_mlp | 1,238,156 | 752,129 | 0 | 0 | 1,990,285 | 0 | 3,980,570 | 1,990,285 | 1,990,285 |
| maddpg_deepset | 1,238,156 | 1,360,897 | 0 | 832,564 | 2,599,053 | 0 | 6,030,670 | 3,431,617 | 2,599,053 |
| matd3_mlp | 1,238,156 | 1,504,258 | 0 | 0 | 2,742,414 | 0 | 5,484,828 | 2,742,414 | 2,742,414 |
| matd3_deepset | 1,238,156 | 2,721,794 | 0 | 1,292,340 | 3,959,950 | 0 | 9,212,240 | 5,252,290 | 3,959,950 |
| masac_mlp | 1,239,704 | 1,504,258 | 0 | 0 | 1,504,258 | 0 | 4,248,220 | 2,743,962 | 2,743,962 |
| masac_deepset | 1,239,704 | 2,721,794 | 0 | 1,292,340 | 2,721,794 | 0 | 7,975,632 | 5,253,838 | 3,961,498 |
| tmatd3 | 1,764,492 | 2,129,922 | 0 | 602,676 | 3,894,414 | 0 | 8,391,504 | 4,497,090 | 3,894,414 |
| tmatd3_dec | 1,238,156 | 2,129,922 | 0 | 602,676 | 3,368,078 | 0 | 7,338,832 | 3,970,754 | 3,368,078 |
| tmasac | 1,766,040 | 2,129,922 | 0 | 602,676 | 2,129,922 | 0 | 6,628,560 | 4,498,638 | 3,895,962 |
| tmasac_dec | 1,239,704 | 2,129,922 | 0 | 602,676 | 2,129,922 | 0 | 6,102,224 | 3,972,302 | 3,369,626 |
| tmasac_shared_encoder | 1,023,640 | 1,387,522 | 1,730,048 | 602,676 | 3,117,570 | 0 | 7,861,456 | 4,743,886 | 4,141,210 |
| tmasac_swiglu | 1,776,472 | 2,140,354 | 0 | 602,676 | 2,140,354 | 0 | 6,659,856 | 4,519,502 | 3,916,826 |
| tmasac_slstm | 2,032,280 | 2,392,578 | 0 | 602,676 | 2,392,578 | 0 | 7,420,112 | 5,027,534 | 4,424,858 |
| tmasac_slstm_no_residual | 2,032,280 | 2,392,578 | 0 | 602,676 | 2,392,578 | 0 | 7,420,112 | 5,027,534 | 4,424,858 |
| tmasac_slstm_shared_encoder | 1,023,640 | 1,387,522 | 1,994,752 | 602,676 | 3,382,274 | 0 | 8,390,864 | 5,008,590 | 4,405,914 |
| tmasac_lstm | 2,816,152 | 2,392,578 | 0 | 602,676 | 2,392,578 | 0 | 8,203,984 | 5,811,406 | 5,208,730 |
| tmasac_slstm_swiglu | 2,041,176 | 2,403,010 | 0 | 602,676 | 2,403,010 | 0 | 7,449,872 | 5,046,862 | 4,444,186 |
| mat_ind_no_attention | 42,184 | 198,913 | 678,400 | 471,092 | 0 | 0 | 1,390,589 | 1,390,589 | 919,497 |
| mat_qcx_no_nop | 623,048 | 198,913 | 1,204,736 | 0 | 0 | 0 | 2,026,697 | 2,026,697 | 2,026,697 |
| tmasac_no_nop | 1,766,040 | 2,129,922 | 0 | 0 | 2,129,922 | 0 | 6,025,884 | 3,895,962 | 3,895,962 |
| tmasac_slstm_no_nop | 2,032,280 | 2,392,578 | 0 | 0 | 2,392,578 | 0 | 6,817,436 | 4,424,858 | 4,424,858 |
| tmasac_lstm_no_actor_state | 2,420,376 | 1,604,610 | 0 | 602,676 | 1,604,610 | 0 | 6,232,272 | 4,627,662 | 4,024,986 |
| tmasac_segment | 1,240,728 | 1,604,610 | 0 | 602,676 | 1,604,610 | 0 | 5,052,624 | 3,448,014 | 2,845,338 |

## Main-policy MLP, projection, attention, and recurrent parameters

This comparison includes only online actor, critic, shared encoder, and any other main-policy parameters. It excludes all NOP modules and frozen target copies. Shared parameters count once. The **MLP + linear** column is the affine parameter budget to compare when matching feed-forward processing across variants. It includes biases and action/value output heads.

Classification:

- **MLP:** affine weights/biases in MLPs, SwiGLU/GLU gate/value/output projections, and transformer feed-forward blocks. Transformer MLPs remain here even when their layer also has attention.
- **Linear projection:** remaining standalone affine layers, including observation/token/latent projections, action and value heads, and PopArt affine weights/biases. A projection outside a recurrent module, such as `temporal_output_projection` or critic actor-state encoding, remains in MLP + linear.
- **Attention:** complete attention modules, including Q/K/V and output projections; these affine parameters are excluded from MLP + linear.
- **Recurrent:** complete LSTM/sLSTM/other temporal modules, including input/gate projections, recurrent kernels, biases, and any internal normalization. These are excluded from MLP + linear.
- **Normalization / embedding / other:** parameters outside the above categories; these are also excluded from MLP + linear. They are separated in the role and component tables below.

MLP + linear + attention + recurrent + remainder = online main-policy total. Remainder is normalization + embedding + other. Parameter counts measure stored affine capacity; they do not measure FLOPs, parameter reuse per timestep, or effective model capacity. Attention's matrix operations can add substantial compute without additional parameters. Matching MLP + linear totals alone can also hide different actor/critic allocations or layer shapes; use the role and module breakdowns when choosing an experimental control.

| Variant | Actor MLP + linear | Critic MLP + linear | Shared MLP + linear | Total MLP + linear | Attention | Recurrent | Remainder | Online total | Trainable − NOP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ppo | 44,072 | 317,441 | 790,336 | 1,151,849 | 0 | 0 | 0 | 1,151,849 | 1,151,849 |
| ppo_small | 43,880 | 157,441 | 545,824 | 747,145 | 0 | 0 | 0 | 747,145 | 747,145 |
| mappo | 58,696 | 166,273 | 1,007,232 | 1,232,201 | 0 | 0 | 0 | 1,232,201 | 1,232,201 |
| mappo_small | 55,480 | 130,609 | 660,864 | 846,953 | 0 | 0 | 0 | 846,953 | 846,953 |
| mappo_mlp | 58,696 | 366,337 | 1,007,232 | 1,432,265 | 0 | 0 | 0 | 1,432,265 | 1,432,265 |
| mappo_mlp_small | 55,480 | 274,817 | 660,864 | 991,161 | 0 | 0 | 0 | 991,161 | 991,161 |
| mat_orig | 142,920 | 198,915 | 675,840 | 1,017,675 | 790,528 | 0 | 5,120 | 1,813,323 | 1,813,323 |
| mat_ind | 42,184 | 198,913 | 675,840 | 916,937 | 526,336 | 0 | 2,560 | 1,445,833 | 1,445,833 |
| mat_dec | 734,536 | 874,753 | 0 | 1,609,289 | 526,336 | 0 | 5,120 | 2,140,745 | 2,140,745 |
| mat_qcx | 291,016 | 198,913 | 675,840 | 1,165,769 | 856,064 | 0 | 4,864 | 2,026,697 | 2,026,697 |
| mat_ind_lstm | 42,184 | 198,913 | 1,332,736 | 1,573,833 | 526,336 | 1,048,576 | 3,584 | 3,152,329 | 3,152,329 |
| mat_qcx_lstm | 291,016 | 198,913 | 1,332,736 | 1,822,665 | 856,064 | 1,048,576 | 5,888 | 3,733,193 | 3,733,193 |
| maddpg_mlp | 1,235,596 | 752,129 | 0 | 1,987,725 | 0 | 0 | 2,560 | 1,990,285 | 1,990,285 |
| maddpg_deepset | 1,235,596 | 1,360,897 | 0 | 2,596,493 | 0 | 0 | 2,560 | 2,599,053 | 2,599,053 |
| matd3_mlp | 1,235,596 | 1,504,258 | 0 | 2,739,854 | 0 | 0 | 2,560 | 2,742,414 | 2,742,414 |
| matd3_deepset | 1,235,596 | 2,721,794 | 0 | 3,957,390 | 0 | 0 | 2,560 | 3,959,950 | 3,959,950 |
| masac_mlp | 1,237,144 | 1,504,258 | 0 | 2,741,402 | 0 | 0 | 2,560 | 2,743,962 | 2,743,962 |
| masac_deepset | 1,237,144 | 2,721,794 | 0 | 3,958,938 | 0 | 0 | 2,560 | 3,961,498 | 3,961,498 |
| tmatd3 | 1,235,596 | 1,601,026 | 0 | 2,836,622 | 1,052,672 | 0 | 5,120 | 3,894,414 | 3,894,414 |
| tmatd3_dec | 1,235,596 | 1,601,026 | 0 | 2,836,622 | 526,336 | 0 | 5,120 | 3,368,078 | 3,368,078 |
| tmasac | 1,237,144 | 1,601,026 | 0 | 2,838,170 | 1,052,672 | 0 | 5,120 | 3,895,962 | 3,895,962 |
| tmasac_dec | 1,237,144 | 1,601,026 | 0 | 2,838,170 | 526,336 | 0 | 5,120 | 3,369,626 | 3,369,626 |
| tmasac_shared_encoder | 758,936 | 1,122,818 | 1,201,152 | 3,082,906 | 1,052,672 | 0 | 5,632 | 4,141,210 | 4,141,210 |
| tmasac_swiglu | 1,246,552 | 1,610,434 | 0 | 2,856,986 | 1,052,672 | 0 | 7,168 | 3,916,826 | 3,916,826 |
| tmasac_slstm | 1,237,656 | 1,863,682 | 0 | 3,101,338 | 1,052,672 | 264,704 | 6,144 | 4,424,858 | 4,424,858 |
| tmasac_slstm_no_residual | 1,237,656 | 1,863,682 | 0 | 3,101,338 | 1,052,672 | 264,704 | 6,144 | 4,424,858 | 4,424,858 |
| tmasac_slstm_shared_encoder | 758,936 | 1,122,818 | 1,201,152 | 3,082,906 | 1,052,672 | 264,704 | 5,632 | 4,405,914 | 4,405,914 |
| tmasac_lstm | 1,237,656 | 1,863,682 | 0 | 3,101,338 | 1,052,672 | 1,048,576 | 6,144 | 5,208,730 | 5,208,730 |
| tmasac_slstm_swiglu | 1,246,552 | 1,873,090 | 0 | 3,119,642 | 1,052,672 | 264,704 | 7,168 | 4,444,186 | 4,444,186 |
| mat_ind_no_attention | 42,184 | 198,913 | 675,840 | 916,937 | 0 | 0 | 2,560 | 919,497 | 919,497 |
| mat_qcx_no_nop | 291,016 | 198,913 | 675,840 | 1,165,769 | 856,064 | 0 | 4,864 | 2,026,697 | 2,026,697 |
| tmasac_no_nop | 1,237,144 | 1,601,026 | 0 | 2,838,170 | 1,052,672 | 0 | 5,120 | 3,895,962 | 3,895,962 |
| tmasac_slstm_no_nop | 1,237,656 | 1,863,682 | 0 | 3,101,338 | 1,052,672 | 264,704 | 6,144 | 4,424,858 | 4,424,858 |
| tmasac_lstm_no_actor_state | 842,904 | 1,075,714 | 0 | 1,918,618 | 1,052,672 | 1,048,576 | 5,120 | 4,024,986 | 4,024,986 |
| tmasac_segment | 711,832 | 1,075,714 | 0 | 1,787,546 | 1,052,672 | 0 | 5,120 | 2,845,338 | 2,845,338 |

## Reproduce

Run from the repository root with the installed package:

```bash
python examples/inspect_policy_parameters.py SwarmBots-WallEasy-v0 --output-dir docs/policy_parameters
```

This command counts all current registered defaults. Use `--variants` to select a subset, `--no-nop` to disable NOP, or `--policy-kwargs`, `--scenario-kwargs`, and `--env-kwargs` with JSON objects to match custom architectures/tasks. For example, `--policy-kwargs '{"enc_d_model": 128, "dec_d_model": 64}'`. Reapply the overrides recorded above to reproduce a customized report. The losslessly compressed `counts.json.gz` output records resolved policy hyperparameters; CSVs contain overall counts, components, processing by role, and processing components. Failed variants are listed explicitly and cause a nonzero exit code.

Read the compressed JSON in Python with:

```python
import gzip, json
with gzip.open("docs/policy_parameters/counts.json.gz", "rt", encoding="utf-8") as stream:
    counts = json.load(stream)
```

## Task shapes

- `{"actions_per_agent": 12, "actuators_per_agent": 8, "connectors_per_agent": 4, "continuous_connector_actions": true, "global_obs_dim": 0, "hidden_global_vars_dim": 1, "hidden_local_vars_dim": 4, "local_obs_dim": 71, "n_agents": 5}`

## Component breakdowns

Each component table is additive. Recurrent temporal-model weights are split from the encoder backbone; critic encoders, Q heads, and NOP heads are separated.

<details>
<summary>ppo — 1,569,565 parameters</summary>

Policy: `PPOPolicy`. Parameter storage: 5.987 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `action_dist` | 6,984 | 6,984 | 0 |
| actor | `actor.mlp` | 37,088 | 37,088 | 0 |
| critic | `critic.value_features` | 317,184 | 317,184 | 0 |
| critic | `critic.value_head` | 257 | 257 | 0 |
| next_obs_prediction | `local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `pre_predictors_transform` | 41,216 | 41,216 | 0 |
| next_obs_prediction | `pre_transition_transform` | 37,056 | 37,056 | 0 |
| next_obs_prediction | `transition_model` | 332,736 | 332,736 | 0 |
| shared_encoder | `shared_encoder` | 790,336 | 790,336 | 0 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 37,088 | 6,984 | 44,072 | 0 | 0 | 0 | 0 | 0 | 44,072 |
| critic | 317,184 | 257 | 317,441 | 0 | 0 | 0 | 0 | 0 | 317,441 |
| shared_encoder | 790,336 | 0 | 790,336 | 0 | 0 | 0 | 0 | 0 | 790,336 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | linear_projection | `action_dist.distributions.0.output_net` | 4,656 |
| actor | linear_projection | `action_dist.distributions.1.output_net` | 2,328 |
| actor | mlp | `actor.mlp` | 37,088 |
| critic | linear_projection | `critic.value_head` | 257 |
| critic | mlp | `critic.value_features` | 317,184 |
| shared_encoder | mlp | `shared_encoder.mlp` | 790,336 |

</details>

<details>
<summary>ppo_small — 1,141,245 parameters</summary>

Policy: `PPOPolicy`. Parameter storage: 4.354 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `action_dist` | 4,680 | 4,680 | 0 |
| actor | `actor.mlp` | 39,200 | 39,200 | 0 |
| critic | `critic.value_features` | 157,280 | 157,280 | 0 |
| critic | `critic.value_head` | 161 | 161 | 0 |
| next_obs_prediction | `local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `pre_predictors_transform` | 37,120 | 37,120 | 0 |
| next_obs_prediction | `pre_transition_transform` | 25,760 | 25,760 | 0 |
| next_obs_prediction | `transition_model` | 324,512 | 324,512 | 0 |
| shared_encoder | `shared_encoder` | 545,824 | 545,824 | 0 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 39,200 | 4,680 | 43,880 | 0 | 0 | 0 | 0 | 0 | 43,880 |
| critic | 157,280 | 161 | 157,441 | 0 | 0 | 0 | 0 | 0 | 157,441 |
| shared_encoder | 545,824 | 0 | 545,824 | 0 | 0 | 0 | 0 | 0 | 545,824 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | linear_projection | `action_dist.distributions.0.output_net` | 3,120 |
| actor | linear_projection | `action_dist.distributions.1.output_net` | 1,560 |
| actor | mlp | `actor.mlp` | 39,200 |
| critic | linear_projection | `critic.value_head` | 161 |
| critic | mlp | `critic.value_features` | 157,280 |
| shared_encoder | mlp | `shared_encoder.mlp` | 545,824 |

</details>

<details>
<summary>mappo — 1,703,293 parameters</summary>

Policy: `MAPPOPolicy`. Parameter storage: 6.498 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `action_dist` | 9,288 | 9,288 | 0 |
| actor | `actor.mlp` | 49,408 | 49,408 | 0 |
| critic | `critic.deepset` | 166,144 | 166,144 | 0 |
| critic | `critic.popart_head` | 129 | 129 | 0 |
| next_obs_prediction | `local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `pre_predictors_transform` | 49,408 | 49,408 | 0 |
| next_obs_prediction | `pre_transition_transform` | 65,792 | 65,792 | 0 |
| next_obs_prediction | `transition_model` | 349,184 | 349,184 | 0 |
| shared_encoder | `shared_encoder` | 1,007,232 | 1,007,232 | 0 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 49,408 | 9,288 | 58,696 | 0 | 0 | 0 | 0 | 0 | 58,696 |
| critic | 166,144 | 129 | 166,273 | 0 | 0 | 0 | 0 | 0 | 166,273 |
| shared_encoder | 1,007,232 | 0 | 1,007,232 | 0 | 0 | 0 | 0 | 0 | 1,007,232 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | linear_projection | `action_dist.distributions.0.output_net` | 6,192 |
| actor | linear_projection | `action_dist.distributions.1.output_net` | 3,096 |
| actor | mlp | `actor.mlp` | 49,408 |
| critic | linear_projection | `critic.popart_head` | 129 |
| critic | mlp | `critic.deepset.element_encoder` | 99,968 |
| critic | mlp | `critic.deepset.set_decoder.0` | 66,176 |
| shared_encoder | mlp | `shared_encoder.mlp` | 1,007,232 |

Tied parameter aliases are counted once; see `counts.json.gz` for their names.

</details>

<details>
<summary>mappo_small — 1,318,045 parameters</summary>

Policy: `MAPPOPolicy`. Parameter storage: 5.028 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `action_dist` | 8,136 | 8,136 | 0 |
| actor | `actor.mlp` | 47,344 | 47,344 | 0 |
| critic | `critic.deepset` | 130,480 | 130,480 | 0 |
| critic | `critic.popart_head` | 129 | 129 | 0 |
| next_obs_prediction | `local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `pre_predictors_transform` | 49,408 | 49,408 | 0 |
| next_obs_prediction | `pre_transition_transform` | 65,792 | 65,792 | 0 |
| next_obs_prediction | `transition_model` | 349,184 | 349,184 | 0 |
| shared_encoder | `shared_encoder` | 660,864 | 660,864 | 0 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 47,344 | 8,136 | 55,480 | 0 | 0 | 0 | 0 | 0 | 55,480 |
| critic | 130,480 | 129 | 130,609 | 0 | 0 | 0 | 0 | 0 | 130,609 |
| shared_encoder | 660,864 | 0 | 660,864 | 0 | 0 | 0 | 0 | 0 | 660,864 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | linear_projection | `action_dist.distributions.0.output_net` | 5,424 |
| actor | linear_projection | `action_dist.distributions.1.output_net` | 2,712 |
| actor | mlp | `actor.mlp` | 47,344 |
| critic | linear_projection | `critic.popart_head` | 129 |
| critic | mlp | `critic.deepset.element_encoder` | 83,888 |
| critic | mlp | `critic.deepset.set_decoder.0` | 46,592 |
| shared_encoder | mlp | `shared_encoder.mlp` | 660,864 |

Tied parameter aliases are counted once; see `counts.json.gz` for their names.

</details>

<details>
<summary>mappo_mlp — 1,903,357 parameters</summary>

Policy: `MAPPOPolicy`. Parameter storage: 7.261 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `action_dist` | 9,288 | 9,288 | 0 |
| actor | `actor.mlp` | 49,408 | 49,408 | 0 |
| critic | `critic.value_features` | 366,208 | 366,208 | 0 |
| critic | `critic.value_head` | 129 | 129 | 0 |
| next_obs_prediction | `local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `pre_predictors_transform` | 49,408 | 49,408 | 0 |
| next_obs_prediction | `pre_transition_transform` | 65,792 | 65,792 | 0 |
| next_obs_prediction | `transition_model` | 349,184 | 349,184 | 0 |
| shared_encoder | `shared_encoder` | 1,007,232 | 1,007,232 | 0 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 49,408 | 9,288 | 58,696 | 0 | 0 | 0 | 0 | 0 | 58,696 |
| critic | 366,208 | 129 | 366,337 | 0 | 0 | 0 | 0 | 0 | 366,337 |
| shared_encoder | 1,007,232 | 0 | 1,007,232 | 0 | 0 | 0 | 0 | 0 | 1,007,232 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | linear_projection | `action_dist.distributions.0.output_net` | 6,192 |
| actor | linear_projection | `action_dist.distributions.1.output_net` | 3,096 |
| actor | mlp | `actor.mlp` | 49,408 |
| critic | linear_projection | `critic.value_head` | 129 |
| critic | mlp | `critic.value_features` | 366,208 |
| shared_encoder | mlp | `shared_encoder.mlp` | 1,007,232 |

</details>

<details>
<summary>mappo_mlp_small — 1,462,253 parameters</summary>

Policy: `MAPPOPolicy`. Parameter storage: 5.578 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `action_dist` | 8,136 | 8,136 | 0 |
| actor | `actor.mlp` | 47,344 | 47,344 | 0 |
| critic | `critic.value_features` | 274,688 | 274,688 | 0 |
| critic | `critic.value_head` | 129 | 129 | 0 |
| next_obs_prediction | `local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `pre_predictors_transform` | 49,408 | 49,408 | 0 |
| next_obs_prediction | `pre_transition_transform` | 65,792 | 65,792 | 0 |
| next_obs_prediction | `transition_model` | 349,184 | 349,184 | 0 |
| shared_encoder | `shared_encoder` | 660,864 | 660,864 | 0 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 47,344 | 8,136 | 55,480 | 0 | 0 | 0 | 0 | 0 | 55,480 |
| critic | 274,688 | 129 | 274,817 | 0 | 0 | 0 | 0 | 0 | 274,817 |
| shared_encoder | 660,864 | 0 | 660,864 | 0 | 0 | 0 | 0 | 0 | 660,864 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | linear_projection | `action_dist.distributions.0.output_net` | 5,424 |
| actor | linear_projection | `action_dist.distributions.1.output_net` | 2,712 |
| actor | mlp | `actor.mlp` | 47,344 |
| critic | linear_projection | `critic.value_head` | 129 |
| critic | mlp | `critic.value_features` | 274,688 |
| shared_encoder | mlp | `shared_encoder.mlp` | 660,864 |

</details>

<details>
<summary>mat_orig — 2,284,415 parameters</summary>

Policy: `MATOrigPolicy`. Parameter storage: 8.714 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `action_dist` | 9,288 | 9,288 | 0 |
| actor | `decoder` | 366,976 | 366,976 | 0 |
| actor | `encoder_decoder_projection` | 32,896 | 32,896 | 0 |
| critic | `critic.popart_head` | 2 | 2 | 0 |
| critic | `critic.value_head` | 199,425 | 199,425 | 0 |
| next_obs_prediction | `local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `pre_predictors_transform` | 49,408 | 49,408 | 0 |
| next_obs_prediction | `pre_transition_transform` | 65,792 | 65,792 | 0 |
| next_obs_prediction | `transition_model` | 349,184 | 349,184 | 0 |
| shared_encoder | `encoder.backbone` | 1,204,736 | 1,204,736 | 0 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 99,072 | 43,848 | 142,920 | 264,192 | 0 | 2,048 | 0 | 0 | 409,160 |
| critic | 198,913 | 2 | 198,915 | 0 | 0 | 512 | 0 | 0 | 199,427 |
| shared_encoder | 675,840 | 0 | 675,840 | 526,336 | 0 | 2,560 | 0 | 0 | 1,204,736 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | attention | `decoder.blocks.0.attn1` | 66,048 |
| actor | attention | `decoder.blocks.0.attn2` | 66,048 |
| actor | attention | `decoder.blocks.1.attn1` | 66,048 |
| actor | attention | `decoder.blocks.1.attn2` | 66,048 |
| actor | linear_projection | `action_dist.distributions.0.output_net` | 6,192 |
| actor | linear_projection | `action_dist.distributions.1.output_net` | 3,096 |
| actor | linear_projection | `decoder.action_encoder.0` | 1,664 |
| actor | linear_projection | `encoder_decoder_projection` | 32,896 |
| actor | mlp | `decoder.blocks.0.mlp` | 33,024 |
| actor | mlp | `decoder.blocks.1.mlp` | 33,024 |
| actor | mlp | `decoder.head` | 33,024 |
| actor | normalization | `decoder.blocks.0.ln1` | 256 |
| actor | normalization | `decoder.blocks.0.ln2` | 256 |
| actor | normalization | `decoder.blocks.0.ln3` | 256 |
| actor | normalization | `decoder.blocks.1.ln1` | 256 |
| actor | normalization | `decoder.blocks.1.ln2` | 256 |
| actor | normalization | `decoder.blocks.1.ln3` | 256 |
| actor | normalization | `decoder.head.2` | 256 |
| actor | normalization | `decoder.ln` | 256 |
| critic | linear_projection | `critic.popart_head` | 2 |
| critic | mlp | `critic.value_head` | 198,913 |
| critic | normalization | `critic.value_head.2` | 512 |
| shared_encoder | attention | `encoder.layers.0.self_attn` | 263,168 |
| shared_encoder | attention | `encoder.layers.1.self_attn` | 263,168 |
| shared_encoder | mlp | `encoder.layers.0.feedforward` | 262,912 |
| shared_encoder | mlp | `encoder.layers.1.feedforward` | 262,912 |
| shared_encoder | mlp | `encoder.local_obs_encoder` | 150,016 |
| shared_encoder | normalization | `encoder.layers.0.norm1` | 512 |
| shared_encoder | normalization | `encoder.layers.0.norm2` | 512 |
| shared_encoder | normalization | `encoder.layers.1.norm1` | 512 |
| shared_encoder | normalization | `encoder.layers.1.norm2` | 512 |
| shared_encoder | normalization | `encoder.norm` | 512 |

</details>

<details>
<summary>mat_ind — 1,916,925 parameters</summary>

Policy: `MATIndPolicy`. Parameter storage: 7.312 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `action_dist` | 9,288 | 9,288 | 0 |
| actor | `actor_head` | 32,896 | 32,896 | 0 |
| critic | `critic.deepset` | 198,656 | 198,656 | 0 |
| critic | `critic.popart_head` | 257 | 257 | 0 |
| next_obs_prediction | `local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `pre_predictors_transform` | 49,408 | 49,408 | 0 |
| next_obs_prediction | `pre_transition_transform` | 65,792 | 65,792 | 0 |
| next_obs_prediction | `transition_model` | 349,184 | 349,184 | 0 |
| shared_encoder | `encoder.backbone` | 1,204,736 | 1,204,736 | 0 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 32,896 | 9,288 | 42,184 | 0 | 0 | 0 | 0 | 0 | 42,184 |
| critic | 198,656 | 257 | 198,913 | 0 | 0 | 0 | 0 | 0 | 198,913 |
| shared_encoder | 675,840 | 0 | 675,840 | 526,336 | 0 | 2,560 | 0 | 0 | 1,204,736 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | linear_projection | `action_dist.distributions.0.output_net` | 6,192 |
| actor | linear_projection | `action_dist.distributions.1.output_net` | 3,096 |
| actor | mlp | `actor_head` | 32,896 |
| critic | linear_projection | `critic.popart_head` | 257 |
| critic | mlp | `critic.deepset.element_encoder` | 132,864 |
| critic | mlp | `critic.deepset.set_decoder.0` | 65,792 |
| shared_encoder | attention | `encoder.layers.0.self_attn` | 263,168 |
| shared_encoder | attention | `encoder.layers.1.self_attn` | 263,168 |
| shared_encoder | mlp | `encoder.layers.0.feedforward` | 262,912 |
| shared_encoder | mlp | `encoder.layers.1.feedforward` | 262,912 |
| shared_encoder | mlp | `encoder.local_obs_encoder` | 150,016 |
| shared_encoder | normalization | `encoder.layers.0.norm1` | 512 |
| shared_encoder | normalization | `encoder.layers.0.norm2` | 512 |
| shared_encoder | normalization | `encoder.layers.1.norm1` | 512 |
| shared_encoder | normalization | `encoder.layers.1.norm2` | 512 |
| shared_encoder | normalization | `encoder.norm` | 512 |

Tied parameter aliases are counted once; see `counts.json.gz` for their names.

</details>

<details>
<summary>mat_dec — 2,611,837 parameters</summary>

Policy: `MATDecPolicy`. Parameter storage: 9.963 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `action_dist` | 9,288 | 9,288 | 0 |
| actor | `actor_encoder.backbone` | 678,400 | 678,400 | 0 |
| actor | `actor_head` | 49,408 | 49,408 | 0 |
| critic | `critic.deepset` | 198,656 | 198,656 | 0 |
| critic | `critic.popart_head` | 257 | 257 | 0 |
| critic | `encoder.backbone` | 1,204,736 | 1,204,736 | 0 |
| next_obs_prediction | `local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `pre_predictors_transform` | 49,408 | 49,408 | 0 |
| next_obs_prediction | `pre_transition_transform` | 65,792 | 65,792 | 0 |
| next_obs_prediction | `transition_model` | 349,184 | 349,184 | 0 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 725,248 | 9,288 | 734,536 | 0 | 0 | 2,560 | 0 | 0 | 737,096 |
| critic | 874,496 | 257 | 874,753 | 526,336 | 0 | 2,560 | 0 | 0 | 1,403,649 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | linear_projection | `action_dist.distributions.0.output_net` | 6,192 |
| actor | linear_projection | `action_dist.distributions.1.output_net` | 3,096 |
| actor | mlp | `actor_encoder.layers.0.feedforward` | 262,912 |
| actor | mlp | `actor_encoder.layers.1.feedforward` | 262,912 |
| actor | mlp | `actor_encoder.local_obs_encoder` | 150,016 |
| actor | mlp | `actor_head` | 49,408 |
| actor | normalization | `actor_encoder.layers.0.norm1` | 512 |
| actor | normalization | `actor_encoder.layers.0.norm2` | 512 |
| actor | normalization | `actor_encoder.layers.1.norm1` | 512 |
| actor | normalization | `actor_encoder.layers.1.norm2` | 512 |
| actor | normalization | `actor_encoder.norm` | 512 |
| critic | attention | `encoder.layers.0.self_attn` | 263,168 |
| critic | attention | `encoder.layers.1.self_attn` | 263,168 |
| critic | linear_projection | `critic.popart_head` | 257 |
| critic | mlp | `critic.deepset.element_encoder` | 132,864 |
| critic | mlp | `critic.deepset.set_decoder.0` | 65,792 |
| critic | mlp | `encoder.layers.0.feedforward` | 262,912 |
| critic | mlp | `encoder.layers.1.feedforward` | 262,912 |
| critic | mlp | `encoder.local_obs_encoder` | 150,016 |
| critic | normalization | `encoder.layers.0.norm1` | 512 |
| critic | normalization | `encoder.layers.0.norm2` | 512 |
| critic | normalization | `encoder.layers.1.norm1` | 512 |
| critic | normalization | `encoder.layers.1.norm2` | 512 |
| critic | normalization | `encoder.norm` | 512 |

Tied parameter aliases are counted once; see `counts.json.gz` for their names.

</details>

<details>
<summary>mat_qcx — 2,497,789 parameters</summary>

Policy: `MATQCXPolicy`. Parameter storage: 9.528 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `action_dist` | 9,288 | 9,288 | 0 |
| actor | `action_encoder` | 18,176 | 18,176 | 0 |
| actor | `decoder` | 595,584 | 595,584 | 0 |
| critic | `critic.deepset` | 198,656 | 198,656 | 0 |
| critic | `critic.popart_head` | 257 | 257 | 0 |
| next_obs_prediction | `local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `pre_predictors_transform` | 49,408 | 49,408 | 0 |
| next_obs_prediction | `pre_transition_transform` | 65,792 | 65,792 | 0 |
| next_obs_prediction | `transition_model` | 349,184 | 349,184 | 0 |
| shared_encoder | `encoder.backbone` | 1,204,736 | 1,204,736 | 0 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 248,832 | 42,184 | 291,016 | 329,728 | 0 | 2,304 | 0 | 0 | 623,048 |
| critic | 198,656 | 257 | 198,913 | 0 | 0 | 0 | 0 | 0 | 198,913 |
| shared_encoder | 675,840 | 0 | 675,840 | 526,336 | 0 | 2,560 | 0 | 0 | 1,204,736 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | attention | `decoder.layers.0.memory_attn` | 98,816 |
| actor | attention | `decoder.layers.0.query_context_attn` | 66,048 |
| actor | attention | `decoder.layers.1.memory_attn` | 98,816 |
| actor | attention | `decoder.layers.1.query_context_attn` | 66,048 |
| actor | linear_projection | `action_dist.distributions.0.output_net` | 6,192 |
| actor | linear_projection | `action_dist.distributions.1.output_net` | 3,096 |
| actor | linear_projection | `decoder.input_projection` | 32,896 |
| actor | mlp | `action_encoder` | 18,176 |
| actor | mlp | `decoder.layers.0` | 65,920 |
| actor | mlp | `decoder.layers.0.context_encoder` | 49,408 |
| actor | mlp | `decoder.layers.1` | 65,920 |
| actor | mlp | `decoder.layers.1.context_encoder` | 49,408 |
| actor | normalization | `decoder.layers.0.context_token_norm` | 256 |
| actor | normalization | `decoder.layers.0.norm1` | 256 |
| actor | normalization | `decoder.layers.0.norm2` | 256 |
| actor | normalization | `decoder.layers.0.norm3` | 256 |
| actor | normalization | `decoder.layers.1.context_token_norm` | 256 |
| actor | normalization | `decoder.layers.1.norm1` | 256 |
| actor | normalization | `decoder.layers.1.norm2` | 256 |
| actor | normalization | `decoder.layers.1.norm3` | 256 |
| actor | normalization | `decoder.norm` | 256 |
| critic | linear_projection | `critic.popart_head` | 257 |
| critic | mlp | `critic.deepset.element_encoder` | 132,864 |
| critic | mlp | `critic.deepset.set_decoder.0` | 65,792 |
| shared_encoder | attention | `encoder.layers.0.self_attn` | 263,168 |
| shared_encoder | attention | `encoder.layers.1.self_attn` | 263,168 |
| shared_encoder | mlp | `encoder.layers.0.feedforward` | 262,912 |
| shared_encoder | mlp | `encoder.layers.1.feedforward` | 262,912 |
| shared_encoder | mlp | `encoder.local_obs_encoder` | 150,016 |
| shared_encoder | normalization | `encoder.layers.0.norm1` | 512 |
| shared_encoder | normalization | `encoder.layers.0.norm2` | 512 |
| shared_encoder | normalization | `encoder.layers.1.norm1` | 512 |
| shared_encoder | normalization | `encoder.layers.1.norm2` | 512 |
| shared_encoder | normalization | `encoder.norm` | 512 |

Tied parameter aliases are counted once; see `counts.json.gz` for their names.

</details>

<details>
<summary>mat_ind_lstm — 3,623,421 parameters</summary>

Policy: `RMATIndPolicy`. Parameter storage: 13.822 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `action_dist` | 9,288 | 9,288 | 0 |
| actor | `actor_head` | 32,896 | 32,896 | 0 |
| critic | `critic.deepset` | 198,656 | 198,656 | 0 |
| critic | `critic.popart_head` | 257 | 257 | 0 |
| next_obs_prediction | `local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `pre_predictors_transform` | 49,408 | 49,408 | 0 |
| next_obs_prediction | `pre_transition_transform` | 65,792 | 65,792 | 0 |
| next_obs_prediction | `transition_model` | 349,184 | 349,184 | 0 |
| shared_encoder | `encoder.backbone` | 1,862,656 | 1,862,656 | 0 |
| shared_encoder | `encoder.temporal_model` | 1,048,576 | 1,048,576 | 0 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 32,896 | 9,288 | 42,184 | 0 | 0 | 0 | 0 | 0 | 42,184 |
| critic | 198,656 | 257 | 198,913 | 0 | 0 | 0 | 0 | 0 | 198,913 |
| shared_encoder | 1,201,664 | 131,072 | 1,332,736 | 526,336 | 1,048,576 | 3,584 | 0 | 0 | 2,911,232 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | linear_projection | `action_dist.distributions.0.output_net` | 6,192 |
| actor | linear_projection | `action_dist.distributions.1.output_net` | 3,096 |
| actor | mlp | `actor_head` | 32,896 |
| critic | linear_projection | `critic.popart_head` | 257 |
| critic | mlp | `critic.deepset.element_encoder` | 132,864 |
| critic | mlp | `critic.deepset.set_decoder.0` | 65,792 |
| shared_encoder | attention | `encoder.layers.0.self_attn` | 263,168 |
| shared_encoder | attention | `encoder.layers.1.self_attn` | 263,168 |
| shared_encoder | linear_projection | `encoder.layers.0.temporal_output_projection` | 65,536 |
| shared_encoder | linear_projection | `encoder.layers.1.temporal_output_projection` | 65,536 |
| shared_encoder | mlp | `encoder.layers.0.feedforward` | 262,912 |
| shared_encoder | mlp | `encoder.layers.0.inter_module_feedforward` | 262,912 |
| shared_encoder | mlp | `encoder.layers.1.feedforward` | 262,912 |
| shared_encoder | mlp | `encoder.layers.1.inter_module_feedforward` | 262,912 |
| shared_encoder | mlp | `encoder.local_obs_encoder` | 150,016 |
| shared_encoder | normalization | `encoder.layers.0.attention_norm` | 512 |
| shared_encoder | normalization | `encoder.layers.0.feedforward_norm` | 512 |
| shared_encoder | normalization | `encoder.layers.0.inter_module_feedforward_norm` | 512 |
| shared_encoder | normalization | `encoder.layers.1.attention_norm` | 512 |
| shared_encoder | normalization | `encoder.layers.1.feedforward_norm` | 512 |
| shared_encoder | normalization | `encoder.layers.1.inter_module_feedforward_norm` | 512 |
| shared_encoder | normalization | `encoder.norm` | 512 |
| shared_encoder | recurrent | `encoder.layers.0.temporal_model` | 524,288 |
| shared_encoder | recurrent | `encoder.layers.1.temporal_model` | 524,288 |

Tied parameter aliases are counted once; see `counts.json.gz` for their names.

</details>

<details>
<summary>mat_qcx_lstm — 4,204,285 parameters</summary>

Policy: `RMATQCXPolicy`. Parameter storage: 16.038 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `action_dist` | 9,288 | 9,288 | 0 |
| actor | `action_encoder` | 18,176 | 18,176 | 0 |
| actor | `decoder` | 595,584 | 595,584 | 0 |
| critic | `critic.deepset` | 198,656 | 198,656 | 0 |
| critic | `critic.popart_head` | 257 | 257 | 0 |
| next_obs_prediction | `local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `pre_predictors_transform` | 49,408 | 49,408 | 0 |
| next_obs_prediction | `pre_transition_transform` | 65,792 | 65,792 | 0 |
| next_obs_prediction | `transition_model` | 349,184 | 349,184 | 0 |
| shared_encoder | `encoder.backbone` | 1,862,656 | 1,862,656 | 0 |
| shared_encoder | `encoder.temporal_model` | 1,048,576 | 1,048,576 | 0 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 248,832 | 42,184 | 291,016 | 329,728 | 0 | 2,304 | 0 | 0 | 623,048 |
| critic | 198,656 | 257 | 198,913 | 0 | 0 | 0 | 0 | 0 | 198,913 |
| shared_encoder | 1,201,664 | 131,072 | 1,332,736 | 526,336 | 1,048,576 | 3,584 | 0 | 0 | 2,911,232 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | attention | `decoder.layers.0.memory_attn` | 98,816 |
| actor | attention | `decoder.layers.0.query_context_attn` | 66,048 |
| actor | attention | `decoder.layers.1.memory_attn` | 98,816 |
| actor | attention | `decoder.layers.1.query_context_attn` | 66,048 |
| actor | linear_projection | `action_dist.distributions.0.output_net` | 6,192 |
| actor | linear_projection | `action_dist.distributions.1.output_net` | 3,096 |
| actor | linear_projection | `decoder.input_projection` | 32,896 |
| actor | mlp | `action_encoder` | 18,176 |
| actor | mlp | `decoder.layers.0` | 65,920 |
| actor | mlp | `decoder.layers.0.context_encoder` | 49,408 |
| actor | mlp | `decoder.layers.1` | 65,920 |
| actor | mlp | `decoder.layers.1.context_encoder` | 49,408 |
| actor | normalization | `decoder.layers.0.context_token_norm` | 256 |
| actor | normalization | `decoder.layers.0.norm1` | 256 |
| actor | normalization | `decoder.layers.0.norm2` | 256 |
| actor | normalization | `decoder.layers.0.norm3` | 256 |
| actor | normalization | `decoder.layers.1.context_token_norm` | 256 |
| actor | normalization | `decoder.layers.1.norm1` | 256 |
| actor | normalization | `decoder.layers.1.norm2` | 256 |
| actor | normalization | `decoder.layers.1.norm3` | 256 |
| actor | normalization | `decoder.norm` | 256 |
| critic | linear_projection | `critic.popart_head` | 257 |
| critic | mlp | `critic.deepset.element_encoder` | 132,864 |
| critic | mlp | `critic.deepset.set_decoder.0` | 65,792 |
| shared_encoder | attention | `encoder.layers.0.self_attn` | 263,168 |
| shared_encoder | attention | `encoder.layers.1.self_attn` | 263,168 |
| shared_encoder | linear_projection | `encoder.layers.0.temporal_output_projection` | 65,536 |
| shared_encoder | linear_projection | `encoder.layers.1.temporal_output_projection` | 65,536 |
| shared_encoder | mlp | `encoder.layers.0.feedforward` | 262,912 |
| shared_encoder | mlp | `encoder.layers.0.inter_module_feedforward` | 262,912 |
| shared_encoder | mlp | `encoder.layers.1.feedforward` | 262,912 |
| shared_encoder | mlp | `encoder.layers.1.inter_module_feedforward` | 262,912 |
| shared_encoder | mlp | `encoder.local_obs_encoder` | 150,016 |
| shared_encoder | normalization | `encoder.layers.0.attention_norm` | 512 |
| shared_encoder | normalization | `encoder.layers.0.feedforward_norm` | 512 |
| shared_encoder | normalization | `encoder.layers.0.inter_module_feedforward_norm` | 512 |
| shared_encoder | normalization | `encoder.layers.1.attention_norm` | 512 |
| shared_encoder | normalization | `encoder.layers.1.feedforward_norm` | 512 |
| shared_encoder | normalization | `encoder.layers.1.inter_module_feedforward_norm` | 512 |
| shared_encoder | normalization | `encoder.norm` | 512 |
| shared_encoder | recurrent | `encoder.layers.0.temporal_model` | 524,288 |
| shared_encoder | recurrent | `encoder.layers.1.temporal_model` | 524,288 |

Tied parameter aliases are counted once; see `counts.json.gz` for their names.

</details>

<details>
<summary>maddpg_mlp — 3,980,570 parameters</summary>

Policy: `TD3Policy`. Parameter storage: 15.185 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `actor.action_net` | 1,548 | 1,548 | 0 |
| actor | `actor.encoder` | 1,203,712 | 1,203,712 | 0 |
| actor | `actor.head` | 32,896 | 32,896 | 0 |
| critic | `critic.0` | 752,129 | 752,129 | 0 |
| targets | `actor_target.action_net` | 1,548 | 0 | 1,548 |
| targets | `actor_target.encoder` | 1,203,712 | 0 | 1,203,712 |
| targets | `actor_target.head` | 32,896 | 0 | 32,896 |
| targets | `critic_target.0` | 752,129 | 0 | 752,129 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 1,234,048 | 1,548 | 1,235,596 | 0 | 0 | 2,560 | 0 | 0 | 1,238,156 |
| critic | 752,129 | 0 | 752,129 | 0 | 0 | 0 | 0 | 0 | 752,129 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | linear_projection | `actor.action_net` | 1,548 |
| actor | mlp | `actor.encoder.layers.0.feedforward` | 525,568 |
| actor | mlp | `actor.encoder.layers.1.feedforward` | 525,568 |
| actor | mlp | `actor.encoder.local_obs_encoder` | 150,016 |
| actor | mlp | `actor.head.head` | 32,896 |
| actor | normalization | `actor.encoder.layers.0.norm1` | 512 |
| actor | normalization | `actor.encoder.layers.0.norm2` | 512 |
| actor | normalization | `actor.encoder.layers.1.norm1` | 512 |
| actor | normalization | `actor.encoder.layers.1.norm2` | 512 |
| actor | normalization | `actor.encoder.norm` | 512 |
| critic | mlp | `critic.0.network` | 752,129 |

</details>

<details>
<summary>maddpg_deepset — 6,030,670 parameters</summary>

Policy: `TD3Policy`. Parameter storage: 23.005 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `actor.action_net` | 1,548 | 1,548 | 0 |
| actor | `actor.encoder` | 1,203,712 | 1,203,712 | 0 |
| actor | `actor.head` | 32,896 | 32,896 | 0 |
| critic | `critic.0` | 1,360,897 | 1,360,897 | 0 |
| next_obs_prediction | `critic_nop.local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `critic_nop.local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `critic_nop.local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `critic_nop.local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `critic_nop.pre_predictors_transform` | 82,176 | 82,176 | 0 |
| next_obs_prediction | `critic_nop.pre_transition_transform` | 328,704 | 328,704 | 0 |
| next_obs_prediction | `critic_nop.transition_model` | 414,976 | 414,976 | 0 |
| targets | `actor_target.action_net` | 1,548 | 0 | 1,548 |
| targets | `actor_target.encoder` | 1,203,712 | 0 | 1,203,712 |
| targets | `actor_target.head` | 32,896 | 0 | 32,896 |
| targets | `critic_target.0` | 1,360,897 | 0 | 1,360,897 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 1,234,048 | 1,548 | 1,235,596 | 0 | 0 | 2,560 | 0 | 0 | 1,238,156 |
| critic | 1,360,384 | 513 | 1,360,897 | 0 | 0 | 0 | 0 | 0 | 1,360,897 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | linear_projection | `actor.action_net` | 1,548 |
| actor | mlp | `actor.encoder.layers.0.feedforward` | 525,568 |
| actor | mlp | `actor.encoder.layers.1.feedforward` | 525,568 |
| actor | mlp | `actor.encoder.local_obs_encoder` | 150,016 |
| actor | mlp | `actor.head.head` | 32,896 |
| actor | normalization | `actor.encoder.layers.0.norm1` | 512 |
| actor | normalization | `actor.encoder.layers.0.norm2` | 512 |
| actor | normalization | `actor.encoder.layers.1.norm1` | 512 |
| actor | normalization | `actor.encoder.layers.1.norm2` | 512 |
| actor | normalization | `actor.encoder.norm` | 512 |
| critic | linear_projection | `critic.0.network.deepset.set_decoder.1` | 513 |
| critic | mlp | `critic.0.network.deepset.element_encoder` | 571,392 |
| critic | mlp | `critic.0.network.deepset.set_decoder.0` | 788,992 |

</details>

<details>
<summary>matd3_mlp — 5,484,828 parameters</summary>

Policy: `TD3Policy`. Parameter storage: 20.923 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `actor.action_net` | 1,548 | 1,548 | 0 |
| actor | `actor.encoder` | 1,203,712 | 1,203,712 | 0 |
| actor | `actor.head` | 32,896 | 32,896 | 0 |
| critic | `critic.0` | 752,129 | 752,129 | 0 |
| critic | `critic.1` | 752,129 | 752,129 | 0 |
| targets | `actor_target.action_net` | 1,548 | 0 | 1,548 |
| targets | `actor_target.encoder` | 1,203,712 | 0 | 1,203,712 |
| targets | `actor_target.head` | 32,896 | 0 | 32,896 |
| targets | `critic_target.0` | 752,129 | 0 | 752,129 |
| targets | `critic_target.1` | 752,129 | 0 | 752,129 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 1,234,048 | 1,548 | 1,235,596 | 0 | 0 | 2,560 | 0 | 0 | 1,238,156 |
| critic | 1,504,258 | 0 | 1,504,258 | 0 | 0 | 0 | 0 | 0 | 1,504,258 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | linear_projection | `actor.action_net` | 1,548 |
| actor | mlp | `actor.encoder.layers.0.feedforward` | 525,568 |
| actor | mlp | `actor.encoder.layers.1.feedforward` | 525,568 |
| actor | mlp | `actor.encoder.local_obs_encoder` | 150,016 |
| actor | mlp | `actor.head.head` | 32,896 |
| actor | normalization | `actor.encoder.layers.0.norm1` | 512 |
| actor | normalization | `actor.encoder.layers.0.norm2` | 512 |
| actor | normalization | `actor.encoder.layers.1.norm1` | 512 |
| actor | normalization | `actor.encoder.layers.1.norm2` | 512 |
| actor | normalization | `actor.encoder.norm` | 512 |
| critic | mlp | `critic.0.network` | 752,129 |
| critic | mlp | `critic.1.network` | 752,129 |

</details>

<details>
<summary>matd3_deepset — 9,212,240 parameters</summary>

Policy: `TD3Policy`. Parameter storage: 35.142 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `actor.action_net` | 1,548 | 1,548 | 0 |
| actor | `actor.encoder` | 1,203,712 | 1,203,712 | 0 |
| actor | `actor.head` | 32,896 | 32,896 | 0 |
| critic | `critic.0` | 1,360,897 | 1,360,897 | 0 |
| critic | `critic.1` | 1,360,897 | 1,360,897 | 0 |
| next_obs_prediction | `critic_nop.local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `critic_nop.local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `critic_nop.local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `critic_nop.local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `critic_nop.pre_predictors_transform` | 147,712 | 147,712 | 0 |
| next_obs_prediction | `critic_nop.pre_transition_transform` | 591,360 | 591,360 | 0 |
| next_obs_prediction | `critic_nop.transition_model` | 546,560 | 546,560 | 0 |
| targets | `actor_target.action_net` | 1,548 | 0 | 1,548 |
| targets | `actor_target.encoder` | 1,203,712 | 0 | 1,203,712 |
| targets | `actor_target.head` | 32,896 | 0 | 32,896 |
| targets | `critic_target.0` | 1,360,897 | 0 | 1,360,897 |
| targets | `critic_target.1` | 1,360,897 | 0 | 1,360,897 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 1,234,048 | 1,548 | 1,235,596 | 0 | 0 | 2,560 | 0 | 0 | 1,238,156 |
| critic | 2,720,768 | 1,026 | 2,721,794 | 0 | 0 | 0 | 0 | 0 | 2,721,794 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | linear_projection | `actor.action_net` | 1,548 |
| actor | mlp | `actor.encoder.layers.0.feedforward` | 525,568 |
| actor | mlp | `actor.encoder.layers.1.feedforward` | 525,568 |
| actor | mlp | `actor.encoder.local_obs_encoder` | 150,016 |
| actor | mlp | `actor.head.head` | 32,896 |
| actor | normalization | `actor.encoder.layers.0.norm1` | 512 |
| actor | normalization | `actor.encoder.layers.0.norm2` | 512 |
| actor | normalization | `actor.encoder.layers.1.norm1` | 512 |
| actor | normalization | `actor.encoder.layers.1.norm2` | 512 |
| actor | normalization | `actor.encoder.norm` | 512 |
| critic | linear_projection | `critic.0.network.deepset.set_decoder.1` | 513 |
| critic | linear_projection | `critic.1.network.deepset.set_decoder.1` | 513 |
| critic | mlp | `critic.0.network.deepset.element_encoder` | 571,392 |
| critic | mlp | `critic.0.network.deepset.set_decoder.0` | 788,992 |
| critic | mlp | `critic.1.network.deepset.element_encoder` | 571,392 |
| critic | mlp | `critic.1.network.deepset.set_decoder.0` | 788,992 |

</details>

<details>
<summary>masac_mlp — 4,248,220 parameters</summary>

Policy: `MASACPolicy`. Parameter storage: 16.206 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `_actor_encoder.backbone` | 1,203,712 | 1,203,712 | 0 |
| actor | `action_dist` | 3,096 | 3,096 | 0 |
| actor | `actor_head` | 32,896 | 32,896 | 0 |
| critic | `critic.q1` | 752,129 | 752,129 | 0 |
| critic | `critic.q2` | 752,129 | 752,129 | 0 |
| targets | `critic_target.q1` | 752,129 | 0 | 752,129 |
| targets | `critic_target.q2` | 752,129 | 0 | 752,129 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 1,234,048 | 3,096 | 1,237,144 | 0 | 0 | 2,560 | 0 | 0 | 1,239,704 |
| critic | 1,504,258 | 0 | 1,504,258 | 0 | 0 | 0 | 0 | 0 | 1,504,258 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | linear_projection | `action_dist.distributions.0.action_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.0.log_std_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.1.action_net` | 516 |
| actor | linear_projection | `action_dist.distributions.1.log_std_net` | 516 |
| actor | mlp | `_actor_encoder.layers.0.feedforward` | 525,568 |
| actor | mlp | `_actor_encoder.layers.1.feedforward` | 525,568 |
| actor | mlp | `_actor_encoder.local_obs_encoder` | 150,016 |
| actor | mlp | `actor_head.head` | 32,896 |
| actor | normalization | `_actor_encoder.layers.0.norm1` | 512 |
| actor | normalization | `_actor_encoder.layers.0.norm2` | 512 |
| actor | normalization | `_actor_encoder.layers.1.norm1` | 512 |
| actor | normalization | `_actor_encoder.layers.1.norm2` | 512 |
| actor | normalization | `_actor_encoder.norm` | 512 |
| critic | mlp | `critic.q1.network` | 752,129 |
| critic | mlp | `critic.q2.network` | 752,129 |

</details>

<details>
<summary>masac_deepset — 7,975,632 parameters</summary>

Policy: `MASACPolicy`. Parameter storage: 30.425 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `_actor_encoder.backbone` | 1,203,712 | 1,203,712 | 0 |
| actor | `action_dist` | 3,096 | 3,096 | 0 |
| actor | `actor_head` | 32,896 | 32,896 | 0 |
| critic | `critic.q1` | 1,360,897 | 1,360,897 | 0 |
| critic | `critic.q2` | 1,360,897 | 1,360,897 | 0 |
| next_obs_prediction | `critic_nop.local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `critic_nop.local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `critic_nop.local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `critic_nop.local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `critic_nop.pre_predictors_transform` | 147,712 | 147,712 | 0 |
| next_obs_prediction | `critic_nop.pre_transition_transform` | 591,360 | 591,360 | 0 |
| next_obs_prediction | `critic_nop.transition_model` | 546,560 | 546,560 | 0 |
| targets | `critic_target.q1` | 1,360,897 | 0 | 1,360,897 |
| targets | `critic_target.q2` | 1,360,897 | 0 | 1,360,897 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 1,234,048 | 3,096 | 1,237,144 | 0 | 0 | 2,560 | 0 | 0 | 1,239,704 |
| critic | 2,720,768 | 1,026 | 2,721,794 | 0 | 0 | 0 | 0 | 0 | 2,721,794 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | linear_projection | `action_dist.distributions.0.action_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.0.log_std_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.1.action_net` | 516 |
| actor | linear_projection | `action_dist.distributions.1.log_std_net` | 516 |
| actor | mlp | `_actor_encoder.layers.0.feedforward` | 525,568 |
| actor | mlp | `_actor_encoder.layers.1.feedforward` | 525,568 |
| actor | mlp | `_actor_encoder.local_obs_encoder` | 150,016 |
| actor | mlp | `actor_head.head` | 32,896 |
| actor | normalization | `_actor_encoder.layers.0.norm1` | 512 |
| actor | normalization | `_actor_encoder.layers.0.norm2` | 512 |
| actor | normalization | `_actor_encoder.layers.1.norm1` | 512 |
| actor | normalization | `_actor_encoder.layers.1.norm2` | 512 |
| actor | normalization | `_actor_encoder.norm` | 512 |
| critic | linear_projection | `critic.q1.network.deepset.set_decoder.1` | 513 |
| critic | linear_projection | `critic.q2.network.deepset.set_decoder.1` | 513 |
| critic | mlp | `critic.q1.network.deepset.element_encoder` | 571,392 |
| critic | mlp | `critic.q1.network.deepset.set_decoder.0` | 788,992 |
| critic | mlp | `critic.q2.network.deepset.element_encoder` | 571,392 |
| critic | mlp | `critic.q2.network.deepset.set_decoder.0` | 788,992 |

</details>

<details>
<summary>tmatd3 — 8,391,504 parameters</summary>

Policy: `TD3Policy`. Parameter storage: 32.011 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `actor.action_net` | 1,548 | 1,548 | 0 |
| actor | `actor.encoder` | 1,730,048 | 1,730,048 | 0 |
| actor | `actor.head` | 32,896 | 32,896 | 0 |
| critic | `critic.encoder` | 1,734,656 | 1,734,656 | 0 |
| critic | `critic.q1` | 197,633 | 197,633 | 0 |
| critic | `critic.q2` | 197,633 | 197,633 | 0 |
| next_obs_prediction | `critic_nop.local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `critic_nop.local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `critic_nop.local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `critic_nop.local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `critic_nop.pre_predictors_transform` | 49,408 | 49,408 | 0 |
| next_obs_prediction | `critic_nop.pre_transition_transform` | 197,376 | 197,376 | 0 |
| next_obs_prediction | `critic_nop.transition_model` | 349,184 | 349,184 | 0 |
| targets | `actor_target.action_net` | 1,548 | 0 | 1,548 |
| targets | `actor_target.encoder` | 1,730,048 | 0 | 1,730,048 |
| targets | `actor_target.head` | 32,896 | 0 | 32,896 |
| targets | `critic_target.encoder` | 1,734,656 | 0 | 1,734,656 |
| targets | `critic_target.q1` | 197,633 | 0 | 197,633 |
| targets | `critic_target.q2` | 197,633 | 0 | 197,633 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 1,234,048 | 1,548 | 1,235,596 | 526,336 | 0 | 2,560 | 0 | 0 | 1,764,492 |
| critic | 1,600,512 | 514 | 1,601,026 | 526,336 | 0 | 2,560 | 0 | 0 | 2,129,922 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | attention | `actor.encoder.layers.0.self_attn` | 263,168 |
| actor | attention | `actor.encoder.layers.1.self_attn` | 263,168 |
| actor | linear_projection | `actor.action_net` | 1,548 |
| actor | mlp | `actor.encoder.layers.0.feedforward` | 525,568 |
| actor | mlp | `actor.encoder.layers.1.feedforward` | 525,568 |
| actor | mlp | `actor.encoder.local_obs_encoder` | 150,016 |
| actor | mlp | `actor.head.head` | 32,896 |
| actor | normalization | `actor.encoder.layers.0.norm1` | 512 |
| actor | normalization | `actor.encoder.layers.0.norm2` | 512 |
| actor | normalization | `actor.encoder.layers.1.norm1` | 512 |
| actor | normalization | `actor.encoder.layers.1.norm2` | 512 |
| actor | normalization | `actor.encoder.norm` | 512 |
| critic | attention | `critic.encoder.layers.0.self_attn` | 263,168 |
| critic | attention | `critic.encoder.layers.1.self_attn` | 263,168 |
| critic | linear_projection | `critic.q1.deepset.set_decoder.1` | 257 |
| critic | linear_projection | `critic.q2.deepset.set_decoder.1` | 257 |
| critic | mlp | `critic.encoder.global_encoder` | 66,304 |
| critic | mlp | `critic.encoder.layers.0.feedforward` | 525,568 |
| critic | mlp | `critic.encoder.layers.1.feedforward` | 525,568 |
| critic | mlp | `critic.encoder.local_action_encoder` | 88,320 |
| critic | mlp | `critic.q1.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q1.deepset.set_decoder.0` | 65,792 |
| critic | mlp | `critic.q2.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q2.deepset.set_decoder.0` | 65,792 |
| critic | normalization | `critic.encoder.layers.0.norm1` | 512 |
| critic | normalization | `critic.encoder.layers.0.norm2` | 512 |
| critic | normalization | `critic.encoder.layers.1.norm1` | 512 |
| critic | normalization | `critic.encoder.layers.1.norm2` | 512 |
| critic | normalization | `critic.encoder.norm` | 512 |

</details>

<details>
<summary>tmatd3_dec — 7,338,832 parameters</summary>

Policy: `TD3Policy`. Parameter storage: 27.995 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `actor.action_net` | 1,548 | 1,548 | 0 |
| actor | `actor.encoder` | 1,203,712 | 1,203,712 | 0 |
| actor | `actor.head` | 32,896 | 32,896 | 0 |
| critic | `critic.encoder` | 1,734,656 | 1,734,656 | 0 |
| critic | `critic.q1` | 197,633 | 197,633 | 0 |
| critic | `critic.q2` | 197,633 | 197,633 | 0 |
| next_obs_prediction | `critic_nop.local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `critic_nop.local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `critic_nop.local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `critic_nop.local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `critic_nop.pre_predictors_transform` | 49,408 | 49,408 | 0 |
| next_obs_prediction | `critic_nop.pre_transition_transform` | 197,376 | 197,376 | 0 |
| next_obs_prediction | `critic_nop.transition_model` | 349,184 | 349,184 | 0 |
| targets | `actor_target.action_net` | 1,548 | 0 | 1,548 |
| targets | `actor_target.encoder` | 1,203,712 | 0 | 1,203,712 |
| targets | `actor_target.head` | 32,896 | 0 | 32,896 |
| targets | `critic_target.encoder` | 1,734,656 | 0 | 1,734,656 |
| targets | `critic_target.q1` | 197,633 | 0 | 197,633 |
| targets | `critic_target.q2` | 197,633 | 0 | 197,633 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 1,234,048 | 1,548 | 1,235,596 | 0 | 0 | 2,560 | 0 | 0 | 1,238,156 |
| critic | 1,600,512 | 514 | 1,601,026 | 526,336 | 0 | 2,560 | 0 | 0 | 2,129,922 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | linear_projection | `actor.action_net` | 1,548 |
| actor | mlp | `actor.encoder.layers.0.feedforward` | 525,568 |
| actor | mlp | `actor.encoder.layers.1.feedforward` | 525,568 |
| actor | mlp | `actor.encoder.local_obs_encoder` | 150,016 |
| actor | mlp | `actor.head.head` | 32,896 |
| actor | normalization | `actor.encoder.layers.0.norm1` | 512 |
| actor | normalization | `actor.encoder.layers.0.norm2` | 512 |
| actor | normalization | `actor.encoder.layers.1.norm1` | 512 |
| actor | normalization | `actor.encoder.layers.1.norm2` | 512 |
| actor | normalization | `actor.encoder.norm` | 512 |
| critic | attention | `critic.encoder.layers.0.self_attn` | 263,168 |
| critic | attention | `critic.encoder.layers.1.self_attn` | 263,168 |
| critic | linear_projection | `critic.q1.deepset.set_decoder.1` | 257 |
| critic | linear_projection | `critic.q2.deepset.set_decoder.1` | 257 |
| critic | mlp | `critic.encoder.global_encoder` | 66,304 |
| critic | mlp | `critic.encoder.layers.0.feedforward` | 525,568 |
| critic | mlp | `critic.encoder.layers.1.feedforward` | 525,568 |
| critic | mlp | `critic.encoder.local_action_encoder` | 88,320 |
| critic | mlp | `critic.q1.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q1.deepset.set_decoder.0` | 65,792 |
| critic | mlp | `critic.q2.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q2.deepset.set_decoder.0` | 65,792 |
| critic | normalization | `critic.encoder.layers.0.norm1` | 512 |
| critic | normalization | `critic.encoder.layers.0.norm2` | 512 |
| critic | normalization | `critic.encoder.layers.1.norm1` | 512 |
| critic | normalization | `critic.encoder.layers.1.norm2` | 512 |
| critic | normalization | `critic.encoder.norm` | 512 |

</details>

<details>
<summary>tmasac — 6,628,560 parameters</summary>

Policy: `TMASACPolicy`. Parameter storage: 25.286 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `_actor_encoder.backbone` | 1,730,048 | 1,730,048 | 0 |
| actor | `action_dist` | 3,096 | 3,096 | 0 |
| actor | `actor_head` | 32,896 | 32,896 | 0 |
| critic | `critic.encoder` | 1,734,656 | 1,734,656 | 0 |
| critic | `critic.q1` | 197,633 | 197,633 | 0 |
| critic | `critic.q2` | 197,633 | 197,633 | 0 |
| next_obs_prediction | `critic_nop.local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `critic_nop.local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `critic_nop.local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `critic_nop.local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `critic_nop.pre_predictors_transform` | 49,408 | 49,408 | 0 |
| next_obs_prediction | `critic_nop.pre_transition_transform` | 197,376 | 197,376 | 0 |
| next_obs_prediction | `critic_nop.transition_model` | 349,184 | 349,184 | 0 |
| targets | `critic_target.encoder` | 1,734,656 | 0 | 1,734,656 |
| targets | `critic_target.q1` | 197,633 | 0 | 197,633 |
| targets | `critic_target.q2` | 197,633 | 0 | 197,633 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 1,234,048 | 3,096 | 1,237,144 | 526,336 | 0 | 2,560 | 0 | 0 | 1,766,040 |
| critic | 1,600,512 | 514 | 1,601,026 | 526,336 | 0 | 2,560 | 0 | 0 | 2,129,922 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | attention | `_actor_encoder.layers.0.self_attn` | 263,168 |
| actor | attention | `_actor_encoder.layers.1.self_attn` | 263,168 |
| actor | linear_projection | `action_dist.distributions.0.action_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.0.log_std_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.1.action_net` | 516 |
| actor | linear_projection | `action_dist.distributions.1.log_std_net` | 516 |
| actor | mlp | `_actor_encoder.layers.0.feedforward` | 525,568 |
| actor | mlp | `_actor_encoder.layers.1.feedforward` | 525,568 |
| actor | mlp | `_actor_encoder.local_obs_encoder` | 150,016 |
| actor | mlp | `actor_head.head` | 32,896 |
| actor | normalization | `_actor_encoder.layers.0.norm1` | 512 |
| actor | normalization | `_actor_encoder.layers.0.norm2` | 512 |
| actor | normalization | `_actor_encoder.layers.1.norm1` | 512 |
| actor | normalization | `_actor_encoder.layers.1.norm2` | 512 |
| actor | normalization | `_actor_encoder.norm` | 512 |
| critic | attention | `critic.encoder.layers.0.self_attn` | 263,168 |
| critic | attention | `critic.encoder.layers.1.self_attn` | 263,168 |
| critic | linear_projection | `critic.q1.deepset.set_decoder.1` | 257 |
| critic | linear_projection | `critic.q2.deepset.set_decoder.1` | 257 |
| critic | mlp | `critic.encoder.global_encoder` | 66,304 |
| critic | mlp | `critic.encoder.layers.0.feedforward` | 525,568 |
| critic | mlp | `critic.encoder.layers.1.feedforward` | 525,568 |
| critic | mlp | `critic.encoder.local_action_encoder` | 88,320 |
| critic | mlp | `critic.q1.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q1.deepset.set_decoder.0` | 65,792 |
| critic | mlp | `critic.q2.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q2.deepset.set_decoder.0` | 65,792 |
| critic | normalization | `critic.encoder.layers.0.norm1` | 512 |
| critic | normalization | `critic.encoder.layers.0.norm2` | 512 |
| critic | normalization | `critic.encoder.layers.1.norm1` | 512 |
| critic | normalization | `critic.encoder.layers.1.norm2` | 512 |
| critic | normalization | `critic.encoder.norm` | 512 |

</details>

<details>
<summary>tmasac_dec — 6,102,224 parameters</summary>

Policy: `TMASACPolicy`. Parameter storage: 23.278 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `_actor_encoder.backbone` | 1,203,712 | 1,203,712 | 0 |
| actor | `action_dist` | 3,096 | 3,096 | 0 |
| actor | `actor_head` | 32,896 | 32,896 | 0 |
| critic | `critic.encoder` | 1,734,656 | 1,734,656 | 0 |
| critic | `critic.q1` | 197,633 | 197,633 | 0 |
| critic | `critic.q2` | 197,633 | 197,633 | 0 |
| next_obs_prediction | `critic_nop.local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `critic_nop.local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `critic_nop.local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `critic_nop.local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `critic_nop.pre_predictors_transform` | 49,408 | 49,408 | 0 |
| next_obs_prediction | `critic_nop.pre_transition_transform` | 197,376 | 197,376 | 0 |
| next_obs_prediction | `critic_nop.transition_model` | 349,184 | 349,184 | 0 |
| targets | `critic_target.encoder` | 1,734,656 | 0 | 1,734,656 |
| targets | `critic_target.q1` | 197,633 | 0 | 197,633 |
| targets | `critic_target.q2` | 197,633 | 0 | 197,633 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 1,234,048 | 3,096 | 1,237,144 | 0 | 0 | 2,560 | 0 | 0 | 1,239,704 |
| critic | 1,600,512 | 514 | 1,601,026 | 526,336 | 0 | 2,560 | 0 | 0 | 2,129,922 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | linear_projection | `action_dist.distributions.0.action_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.0.log_std_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.1.action_net` | 516 |
| actor | linear_projection | `action_dist.distributions.1.log_std_net` | 516 |
| actor | mlp | `_actor_encoder.layers.0.feedforward` | 525,568 |
| actor | mlp | `_actor_encoder.layers.1.feedforward` | 525,568 |
| actor | mlp | `_actor_encoder.local_obs_encoder` | 150,016 |
| actor | mlp | `actor_head.head` | 32,896 |
| actor | normalization | `_actor_encoder.layers.0.norm1` | 512 |
| actor | normalization | `_actor_encoder.layers.0.norm2` | 512 |
| actor | normalization | `_actor_encoder.layers.1.norm1` | 512 |
| actor | normalization | `_actor_encoder.layers.1.norm2` | 512 |
| actor | normalization | `_actor_encoder.norm` | 512 |
| critic | attention | `critic.encoder.layers.0.self_attn` | 263,168 |
| critic | attention | `critic.encoder.layers.1.self_attn` | 263,168 |
| critic | linear_projection | `critic.q1.deepset.set_decoder.1` | 257 |
| critic | linear_projection | `critic.q2.deepset.set_decoder.1` | 257 |
| critic | mlp | `critic.encoder.global_encoder` | 66,304 |
| critic | mlp | `critic.encoder.layers.0.feedforward` | 525,568 |
| critic | mlp | `critic.encoder.layers.1.feedforward` | 525,568 |
| critic | mlp | `critic.encoder.local_action_encoder` | 88,320 |
| critic | mlp | `critic.q1.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q1.deepset.set_decoder.0` | 65,792 |
| critic | mlp | `critic.q2.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q2.deepset.set_decoder.0` | 65,792 |
| critic | normalization | `critic.encoder.layers.0.norm1` | 512 |
| critic | normalization | `critic.encoder.layers.0.norm2` | 512 |
| critic | normalization | `critic.encoder.layers.1.norm1` | 512 |
| critic | normalization | `critic.encoder.layers.1.norm2` | 512 |
| critic | normalization | `critic.encoder.norm` | 512 |

</details>

<details>
<summary>tmasac_shared_encoder — 7,861,456 parameters</summary>

Policy: `TMASACPolicy`. Parameter storage: 29.989 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `_actor_encoder.backbone` | 987,648 | 987,648 | 0 |
| actor | `action_dist` | 3,096 | 3,096 | 0 |
| actor | `actor_head` | 32,896 | 32,896 | 0 |
| critic | `critic.encoder` | 992,256 | 992,256 | 0 |
| critic | `critic.q1` | 197,633 | 197,633 | 0 |
| critic | `critic.q2` | 197,633 | 197,633 | 0 |
| next_obs_prediction | `critic_nop.local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `critic_nop.local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `critic_nop.local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `critic_nop.local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `critic_nop.pre_predictors_transform` | 49,408 | 49,408 | 0 |
| next_obs_prediction | `critic_nop.pre_transition_transform` | 197,376 | 197,376 | 0 |
| next_obs_prediction | `critic_nop.transition_model` | 349,184 | 349,184 | 0 |
| shared_encoder | `shared_observation_encoder.backbone` | 1,730,048 | 1,730,048 | 0 |
| targets | `critic_target.encoder` | 992,256 | 0 | 992,256 |
| targets | `critic_target.q1` | 197,633 | 0 | 197,633 |
| targets | `critic_target.q2` | 197,633 | 0 | 197,633 |
| targets | `shared_observation_encoder_target` | 1,730,048 | 0 | 1,730,048 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 755,840 | 3,096 | 758,936 | 263,168 | 0 | 1,536 | 0 | 0 | 1,023,640 |
| critic | 1,122,304 | 514 | 1,122,818 | 263,168 | 0 | 1,536 | 0 | 0 | 1,387,522 |
| shared_encoder | 1,201,152 | 0 | 1,201,152 | 526,336 | 0 | 2,560 | 0 | 0 | 1,730,048 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | attention | `_actor_encoder.layers.0.self_attn` | 263,168 |
| actor | linear_projection | `action_dist.distributions.0.action_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.0.log_std_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.1.action_net` | 516 |
| actor | linear_projection | `action_dist.distributions.1.log_std_net` | 516 |
| actor | mlp | `_actor_encoder.layers.0.feedforward` | 525,568 |
| actor | mlp | `_actor_encoder.local_obs_encoder` | 197,376 |
| actor | mlp | `actor_head.head` | 32,896 |
| actor | normalization | `_actor_encoder.layers.0.norm1` | 512 |
| actor | normalization | `_actor_encoder.layers.0.norm2` | 512 |
| actor | normalization | `_actor_encoder.norm` | 512 |
| critic | attention | `critic.encoder.layers.0.self_attn` | 263,168 |
| critic | linear_projection | `critic.q1.deepset.set_decoder.1` | 257 |
| critic | linear_projection | `critic.q2.deepset.set_decoder.1` | 257 |
| critic | mlp | `critic.encoder.global_encoder` | 66,304 |
| critic | mlp | `critic.encoder.layers.0.feedforward` | 525,568 |
| critic | mlp | `critic.encoder.local_action_encoder` | 135,680 |
| critic | mlp | `critic.q1.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q1.deepset.set_decoder.0` | 65,792 |
| critic | mlp | `critic.q2.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q2.deepset.set_decoder.0` | 65,792 |
| critic | normalization | `critic.encoder.layers.0.norm1` | 512 |
| critic | normalization | `critic.encoder.layers.0.norm2` | 512 |
| critic | normalization | `critic.encoder.norm` | 512 |
| shared_encoder | attention | `shared_observation_encoder.layers.0.self_attn` | 263,168 |
| shared_encoder | attention | `shared_observation_encoder.layers.1.self_attn` | 263,168 |
| shared_encoder | mlp | `shared_observation_encoder.layers.0.feedforward` | 525,568 |
| shared_encoder | mlp | `shared_observation_encoder.layers.1.feedforward` | 525,568 |
| shared_encoder | mlp | `shared_observation_encoder.local_obs_encoder` | 150,016 |
| shared_encoder | normalization | `shared_observation_encoder.layers.0.norm1` | 512 |
| shared_encoder | normalization | `shared_observation_encoder.layers.0.norm2` | 512 |
| shared_encoder | normalization | `shared_observation_encoder.layers.1.norm1` | 512 |
| shared_encoder | normalization | `shared_observation_encoder.layers.1.norm2` | 512 |
| shared_encoder | normalization | `shared_observation_encoder.norm` | 512 |

</details>

<details>
<summary>tmasac_swiglu — 6,659,856 parameters</summary>

Policy: `TMASACPolicy`. Parameter storage: 25.405 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `_actor_encoder.backbone` | 1,740,480 | 1,740,480 | 0 |
| actor | `action_dist` | 3,096 | 3,096 | 0 |
| actor | `actor_head` | 32,896 | 32,896 | 0 |
| critic | `critic.encoder` | 1,745,088 | 1,745,088 | 0 |
| critic | `critic.q1` | 197,633 | 197,633 | 0 |
| critic | `critic.q2` | 197,633 | 197,633 | 0 |
| next_obs_prediction | `critic_nop.local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `critic_nop.local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `critic_nop.local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `critic_nop.local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `critic_nop.pre_predictors_transform` | 49,408 | 49,408 | 0 |
| next_obs_prediction | `critic_nop.pre_transition_transform` | 197,376 | 197,376 | 0 |
| next_obs_prediction | `critic_nop.transition_model` | 349,184 | 349,184 | 0 |
| targets | `critic_target.encoder` | 1,745,088 | 0 | 1,745,088 |
| targets | `critic_target.q1` | 197,633 | 0 | 197,633 |
| targets | `critic_target.q2` | 197,633 | 0 | 197,633 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 1,243,456 | 3,096 | 1,246,552 | 526,336 | 0 | 3,584 | 0 | 0 | 1,776,472 |
| critic | 1,609,920 | 514 | 1,610,434 | 526,336 | 0 | 3,584 | 0 | 0 | 2,140,354 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | attention | `_actor_encoder.layers.0.self_attn` | 263,168 |
| actor | attention | `_actor_encoder.layers.1.self_attn` | 263,168 |
| actor | linear_projection | `action_dist.distributions.0.action_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.0.log_std_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.1.action_net` | 516 |
| actor | linear_projection | `action_dist.distributions.1.log_std_net` | 516 |
| actor | mlp | `_actor_encoder.layers.0.feedforward` | 530,272 |
| actor | mlp | `_actor_encoder.layers.1.feedforward` | 530,272 |
| actor | mlp | `_actor_encoder.local_obs_encoder` | 150,016 |
| actor | mlp | `actor_head.head` | 32,896 |
| actor | normalization | `_actor_encoder.layers.0.feedforward.pre_norms.1` | 512 |
| actor | normalization | `_actor_encoder.layers.0.norm1` | 512 |
| actor | normalization | `_actor_encoder.layers.0.norm2` | 512 |
| actor | normalization | `_actor_encoder.layers.1.feedforward.pre_norms.1` | 512 |
| actor | normalization | `_actor_encoder.layers.1.norm1` | 512 |
| actor | normalization | `_actor_encoder.layers.1.norm2` | 512 |
| actor | normalization | `_actor_encoder.norm` | 512 |
| critic | attention | `critic.encoder.layers.0.self_attn` | 263,168 |
| critic | attention | `critic.encoder.layers.1.self_attn` | 263,168 |
| critic | linear_projection | `critic.q1.deepset.set_decoder.1` | 257 |
| critic | linear_projection | `critic.q2.deepset.set_decoder.1` | 257 |
| critic | mlp | `critic.encoder.global_encoder` | 66,304 |
| critic | mlp | `critic.encoder.layers.0.feedforward` | 530,272 |
| critic | mlp | `critic.encoder.layers.1.feedforward` | 530,272 |
| critic | mlp | `critic.encoder.local_action_encoder` | 88,320 |
| critic | mlp | `critic.q1.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q1.deepset.set_decoder.0` | 65,792 |
| critic | mlp | `critic.q2.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q2.deepset.set_decoder.0` | 65,792 |
| critic | normalization | `critic.encoder.layers.0.feedforward.pre_norms.1` | 512 |
| critic | normalization | `critic.encoder.layers.0.norm1` | 512 |
| critic | normalization | `critic.encoder.layers.0.norm2` | 512 |
| critic | normalization | `critic.encoder.layers.1.feedforward.pre_norms.1` | 512 |
| critic | normalization | `critic.encoder.layers.1.norm1` | 512 |
| critic | normalization | `critic.encoder.layers.1.norm2` | 512 |
| critic | normalization | `critic.encoder.norm` | 512 |

</details>

<details>
<summary>tmasac_slstm — 7,420,112 parameters</summary>

Policy: `RecurrentTMASACPolicy`. Parameter storage: 28.305 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `_actor_encoder.backbone` | 1,731,584 | 1,731,584 | 0 |
| actor | `_actor_encoder.temporal_model` | 264,704 | 264,704 | 0 |
| actor | `action_dist` | 3,096 | 3,096 | 0 |
| actor | `actor_head` | 32,896 | 32,896 | 0 |
| critic | `critic.actor_state_encoder` | 197,120 | 197,120 | 0 |
| critic | `critic.encoder` | 1,800,192 | 1,800,192 | 0 |
| critic | `critic.q1` | 197,633 | 197,633 | 0 |
| critic | `critic.q2` | 197,633 | 197,633 | 0 |
| next_obs_prediction | `critic_nop.local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `critic_nop.local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `critic_nop.local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `critic_nop.local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `critic_nop.pre_predictors_transform` | 49,408 | 49,408 | 0 |
| next_obs_prediction | `critic_nop.pre_transition_transform` | 197,376 | 197,376 | 0 |
| next_obs_prediction | `critic_nop.transition_model` | 349,184 | 349,184 | 0 |
| targets | `critic_target.actor_state_encoder` | 197,120 | 0 | 197,120 |
| targets | `critic_target.encoder` | 1,800,192 | 0 | 1,800,192 |
| targets | `critic_target.q1` | 197,633 | 0 | 197,633 |
| targets | `critic_target.q2` | 197,633 | 0 | 197,633 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 1,234,560 | 3,096 | 1,237,656 | 526,336 | 264,704 | 3,584 | 0 | 0 | 2,032,280 |
| critic | 1,863,168 | 514 | 1,863,682 | 526,336 | 0 | 2,560 | 0 | 0 | 2,392,578 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | attention | `_actor_encoder.layers.0.self_attn` | 263,168 |
| actor | attention | `_actor_encoder.layers.1.self_attn` | 263,168 |
| actor | linear_projection | `action_dist.distributions.0.action_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.0.log_std_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.1.action_net` | 516 |
| actor | linear_projection | `action_dist.distributions.1.log_std_net` | 516 |
| actor | mlp | `_actor_encoder.layers.0.feedforward` | 262,912 |
| actor | mlp | `_actor_encoder.layers.0.inter_module_feedforward` | 262,912 |
| actor | mlp | `_actor_encoder.layers.1.feedforward` | 262,912 |
| actor | mlp | `_actor_encoder.layers.1.inter_module_feedforward` | 262,912 |
| actor | mlp | `_actor_encoder.local_obs_encoder` | 150,016 |
| actor | mlp | `actor_head.head` | 32,896 |
| actor | normalization | `_actor_encoder.layers.0.attention_norm` | 512 |
| actor | normalization | `_actor_encoder.layers.0.feedforward_norm` | 512 |
| actor | normalization | `_actor_encoder.layers.0.inter_module_feedforward_norm` | 512 |
| actor | normalization | `_actor_encoder.layers.1.attention_norm` | 512 |
| actor | normalization | `_actor_encoder.layers.1.feedforward_norm` | 512 |
| actor | normalization | `_actor_encoder.layers.1.inter_module_feedforward_norm` | 512 |
| actor | normalization | `_actor_encoder.norm` | 512 |
| actor | recurrent | `_actor_encoder.layers.0.temporal_model` | 132,352 |
| actor | recurrent | `_actor_encoder.layers.1.temporal_model` | 132,352 |
| critic | attention | `critic.encoder.layers.0.self_attn` | 263,168 |
| critic | attention | `critic.encoder.layers.1.self_attn` | 263,168 |
| critic | linear_projection | `critic.q1.deepset.set_decoder.1` | 257 |
| critic | linear_projection | `critic.q2.deepset.set_decoder.1` | 257 |
| critic | mlp | `critic.actor_state_encoder` | 197,120 |
| critic | mlp | `critic.encoder.global_encoder` | 66,304 |
| critic | mlp | `critic.encoder.layers.0.feedforward` | 525,568 |
| critic | mlp | `critic.encoder.layers.1.feedforward` | 525,568 |
| critic | mlp | `critic.encoder.local_action_encoder` | 153,856 |
| critic | mlp | `critic.q1.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q1.deepset.set_decoder.0` | 65,792 |
| critic | mlp | `critic.q2.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q2.deepset.set_decoder.0` | 65,792 |
| critic | normalization | `critic.encoder.layers.0.norm1` | 512 |
| critic | normalization | `critic.encoder.layers.0.norm2` | 512 |
| critic | normalization | `critic.encoder.layers.1.norm1` | 512 |
| critic | normalization | `critic.encoder.layers.1.norm2` | 512 |
| critic | normalization | `critic.encoder.norm` | 512 |

</details>

<details>
<summary>tmasac_slstm_no_residual — 7,420,112 parameters</summary>

Policy: `RecurrentTMASACPolicy`. Parameter storage: 28.305 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `_actor_encoder.backbone` | 1,731,584 | 1,731,584 | 0 |
| actor | `_actor_encoder.temporal_model` | 264,704 | 264,704 | 0 |
| actor | `action_dist` | 3,096 | 3,096 | 0 |
| actor | `actor_head` | 32,896 | 32,896 | 0 |
| critic | `critic.actor_state_encoder` | 197,120 | 197,120 | 0 |
| critic | `critic.encoder` | 1,800,192 | 1,800,192 | 0 |
| critic | `critic.q1` | 197,633 | 197,633 | 0 |
| critic | `critic.q2` | 197,633 | 197,633 | 0 |
| next_obs_prediction | `critic_nop.local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `critic_nop.local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `critic_nop.local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `critic_nop.local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `critic_nop.pre_predictors_transform` | 49,408 | 49,408 | 0 |
| next_obs_prediction | `critic_nop.pre_transition_transform` | 197,376 | 197,376 | 0 |
| next_obs_prediction | `critic_nop.transition_model` | 349,184 | 349,184 | 0 |
| targets | `critic_target.actor_state_encoder` | 197,120 | 0 | 197,120 |
| targets | `critic_target.encoder` | 1,800,192 | 0 | 1,800,192 |
| targets | `critic_target.q1` | 197,633 | 0 | 197,633 |
| targets | `critic_target.q2` | 197,633 | 0 | 197,633 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 1,234,560 | 3,096 | 1,237,656 | 526,336 | 264,704 | 3,584 | 0 | 0 | 2,032,280 |
| critic | 1,863,168 | 514 | 1,863,682 | 526,336 | 0 | 2,560 | 0 | 0 | 2,392,578 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | attention | `_actor_encoder.layers.0.self_attn` | 263,168 |
| actor | attention | `_actor_encoder.layers.1.self_attn` | 263,168 |
| actor | linear_projection | `action_dist.distributions.0.action_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.0.log_std_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.1.action_net` | 516 |
| actor | linear_projection | `action_dist.distributions.1.log_std_net` | 516 |
| actor | mlp | `_actor_encoder.layers.0.feedforward` | 262,912 |
| actor | mlp | `_actor_encoder.layers.0.inter_module_feedforward` | 262,912 |
| actor | mlp | `_actor_encoder.layers.1.feedforward` | 262,912 |
| actor | mlp | `_actor_encoder.layers.1.inter_module_feedforward` | 262,912 |
| actor | mlp | `_actor_encoder.local_obs_encoder` | 150,016 |
| actor | mlp | `actor_head.head` | 32,896 |
| actor | normalization | `_actor_encoder.layers.0.attention_norm` | 512 |
| actor | normalization | `_actor_encoder.layers.0.feedforward_norm` | 512 |
| actor | normalization | `_actor_encoder.layers.0.inter_module_feedforward_norm` | 512 |
| actor | normalization | `_actor_encoder.layers.1.attention_norm` | 512 |
| actor | normalization | `_actor_encoder.layers.1.feedforward_norm` | 512 |
| actor | normalization | `_actor_encoder.layers.1.inter_module_feedforward_norm` | 512 |
| actor | normalization | `_actor_encoder.norm` | 512 |
| actor | recurrent | `_actor_encoder.layers.0.temporal_model` | 132,352 |
| actor | recurrent | `_actor_encoder.layers.1.temporal_model` | 132,352 |
| critic | attention | `critic.encoder.layers.0.self_attn` | 263,168 |
| critic | attention | `critic.encoder.layers.1.self_attn` | 263,168 |
| critic | linear_projection | `critic.q1.deepset.set_decoder.1` | 257 |
| critic | linear_projection | `critic.q2.deepset.set_decoder.1` | 257 |
| critic | mlp | `critic.actor_state_encoder` | 197,120 |
| critic | mlp | `critic.encoder.global_encoder` | 66,304 |
| critic | mlp | `critic.encoder.layers.0.feedforward` | 525,568 |
| critic | mlp | `critic.encoder.layers.1.feedforward` | 525,568 |
| critic | mlp | `critic.encoder.local_action_encoder` | 153,856 |
| critic | mlp | `critic.q1.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q1.deepset.set_decoder.0` | 65,792 |
| critic | mlp | `critic.q2.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q2.deepset.set_decoder.0` | 65,792 |
| critic | normalization | `critic.encoder.layers.0.norm1` | 512 |
| critic | normalization | `critic.encoder.layers.0.norm2` | 512 |
| critic | normalization | `critic.encoder.layers.1.norm1` | 512 |
| critic | normalization | `critic.encoder.layers.1.norm2` | 512 |
| critic | normalization | `critic.encoder.norm` | 512 |

</details>

<details>
<summary>tmasac_slstm_shared_encoder — 8,390,864 parameters</summary>

Policy: `RecurrentTMASACPolicy`. Parameter storage: 32.009 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `_actor_encoder.backbone` | 987,648 | 987,648 | 0 |
| actor | `action_dist` | 3,096 | 3,096 | 0 |
| actor | `actor_head` | 32,896 | 32,896 | 0 |
| critic | `critic.encoder` | 992,256 | 992,256 | 0 |
| critic | `critic.q1` | 197,633 | 197,633 | 0 |
| critic | `critic.q2` | 197,633 | 197,633 | 0 |
| next_obs_prediction | `critic_nop.local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `critic_nop.local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `critic_nop.local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `critic_nop.local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `critic_nop.pre_predictors_transform` | 49,408 | 49,408 | 0 |
| next_obs_prediction | `critic_nop.pre_transition_transform` | 197,376 | 197,376 | 0 |
| next_obs_prediction | `critic_nop.transition_model` | 349,184 | 349,184 | 0 |
| shared_encoder | `shared_observation_encoder.backbone` | 1,730,048 | 1,730,048 | 0 |
| shared_encoder | `shared_observation_encoder.temporal_model` | 264,704 | 264,704 | 0 |
| targets | `critic_target.encoder` | 992,256 | 0 | 992,256 |
| targets | `critic_target.q1` | 197,633 | 0 | 197,633 |
| targets | `critic_target.q2` | 197,633 | 0 | 197,633 |
| targets | `shared_observation_encoder_target` | 1,730,048 | 0 | 1,730,048 |
| targets | `shared_observation_encoder_target.temporal_model` | 264,704 | 0 | 264,704 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 755,840 | 3,096 | 758,936 | 263,168 | 0 | 1,536 | 0 | 0 | 1,023,640 |
| critic | 1,122,304 | 514 | 1,122,818 | 263,168 | 0 | 1,536 | 0 | 0 | 1,387,522 |
| shared_encoder | 1,201,152 | 0 | 1,201,152 | 526,336 | 264,704 | 2,560 | 0 | 0 | 1,994,752 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | attention | `_actor_encoder.layers.0.self_attn` | 263,168 |
| actor | linear_projection | `action_dist.distributions.0.action_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.0.log_std_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.1.action_net` | 516 |
| actor | linear_projection | `action_dist.distributions.1.log_std_net` | 516 |
| actor | mlp | `_actor_encoder.layers.0.feedforward` | 525,568 |
| actor | mlp | `_actor_encoder.local_obs_encoder` | 197,376 |
| actor | mlp | `actor_head.head` | 32,896 |
| actor | normalization | `_actor_encoder.layers.0.norm1` | 512 |
| actor | normalization | `_actor_encoder.layers.0.norm2` | 512 |
| actor | normalization | `_actor_encoder.norm` | 512 |
| critic | attention | `critic.encoder.layers.0.self_attn` | 263,168 |
| critic | linear_projection | `critic.q1.deepset.set_decoder.1` | 257 |
| critic | linear_projection | `critic.q2.deepset.set_decoder.1` | 257 |
| critic | mlp | `critic.encoder.global_encoder` | 66,304 |
| critic | mlp | `critic.encoder.layers.0.feedforward` | 525,568 |
| critic | mlp | `critic.encoder.local_action_encoder` | 135,680 |
| critic | mlp | `critic.q1.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q1.deepset.set_decoder.0` | 65,792 |
| critic | mlp | `critic.q2.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q2.deepset.set_decoder.0` | 65,792 |
| critic | normalization | `critic.encoder.layers.0.norm1` | 512 |
| critic | normalization | `critic.encoder.layers.0.norm2` | 512 |
| critic | normalization | `critic.encoder.norm` | 512 |
| shared_encoder | attention | `shared_observation_encoder.layers.0.self_attn` | 263,168 |
| shared_encoder | attention | `shared_observation_encoder.layers.1.self_attn` | 263,168 |
| shared_encoder | mlp | `shared_observation_encoder.layers.0.feedforward` | 525,568 |
| shared_encoder | mlp | `shared_observation_encoder.layers.1.feedforward` | 525,568 |
| shared_encoder | mlp | `shared_observation_encoder.local_obs_encoder` | 150,016 |
| shared_encoder | normalization | `shared_observation_encoder.layers.0.attention_norm` | 512 |
| shared_encoder | normalization | `shared_observation_encoder.layers.0.feedforward_norm` | 512 |
| shared_encoder | normalization | `shared_observation_encoder.layers.1.attention_norm` | 512 |
| shared_encoder | normalization | `shared_observation_encoder.layers.1.feedforward_norm` | 512 |
| shared_encoder | normalization | `shared_observation_encoder.norm` | 512 |
| shared_encoder | recurrent | `shared_observation_encoder.layers.0.temporal_model` | 132,352 |
| shared_encoder | recurrent | `shared_observation_encoder.layers.1.temporal_model` | 132,352 |

</details>

<details>
<summary>tmasac_lstm — 8,203,984 parameters</summary>

Policy: `RecurrentTMASACPolicy`. Parameter storage: 31.296 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `_actor_encoder.backbone` | 1,731,584 | 1,731,584 | 0 |
| actor | `_actor_encoder.temporal_model` | 1,048,576 | 1,048,576 | 0 |
| actor | `action_dist` | 3,096 | 3,096 | 0 |
| actor | `actor_head` | 32,896 | 32,896 | 0 |
| critic | `critic.actor_state_encoder` | 197,120 | 197,120 | 0 |
| critic | `critic.encoder` | 1,800,192 | 1,800,192 | 0 |
| critic | `critic.q1` | 197,633 | 197,633 | 0 |
| critic | `critic.q2` | 197,633 | 197,633 | 0 |
| next_obs_prediction | `critic_nop.local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `critic_nop.local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `critic_nop.local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `critic_nop.local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `critic_nop.pre_predictors_transform` | 49,408 | 49,408 | 0 |
| next_obs_prediction | `critic_nop.pre_transition_transform` | 197,376 | 197,376 | 0 |
| next_obs_prediction | `critic_nop.transition_model` | 349,184 | 349,184 | 0 |
| targets | `critic_target.actor_state_encoder` | 197,120 | 0 | 197,120 |
| targets | `critic_target.encoder` | 1,800,192 | 0 | 1,800,192 |
| targets | `critic_target.q1` | 197,633 | 0 | 197,633 |
| targets | `critic_target.q2` | 197,633 | 0 | 197,633 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 1,234,560 | 3,096 | 1,237,656 | 526,336 | 1,048,576 | 3,584 | 0 | 0 | 2,816,152 |
| critic | 1,863,168 | 514 | 1,863,682 | 526,336 | 0 | 2,560 | 0 | 0 | 2,392,578 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | attention | `_actor_encoder.layers.0.self_attn` | 263,168 |
| actor | attention | `_actor_encoder.layers.1.self_attn` | 263,168 |
| actor | linear_projection | `action_dist.distributions.0.action_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.0.log_std_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.1.action_net` | 516 |
| actor | linear_projection | `action_dist.distributions.1.log_std_net` | 516 |
| actor | mlp | `_actor_encoder.layers.0.feedforward` | 262,912 |
| actor | mlp | `_actor_encoder.layers.0.inter_module_feedforward` | 262,912 |
| actor | mlp | `_actor_encoder.layers.1.feedforward` | 262,912 |
| actor | mlp | `_actor_encoder.layers.1.inter_module_feedforward` | 262,912 |
| actor | mlp | `_actor_encoder.local_obs_encoder` | 150,016 |
| actor | mlp | `actor_head.head` | 32,896 |
| actor | normalization | `_actor_encoder.layers.0.attention_norm` | 512 |
| actor | normalization | `_actor_encoder.layers.0.feedforward_norm` | 512 |
| actor | normalization | `_actor_encoder.layers.0.inter_module_feedforward_norm` | 512 |
| actor | normalization | `_actor_encoder.layers.1.attention_norm` | 512 |
| actor | normalization | `_actor_encoder.layers.1.feedforward_norm` | 512 |
| actor | normalization | `_actor_encoder.layers.1.inter_module_feedforward_norm` | 512 |
| actor | normalization | `_actor_encoder.norm` | 512 |
| actor | recurrent | `_actor_encoder.layers.0.temporal_model` | 524,288 |
| actor | recurrent | `_actor_encoder.layers.1.temporal_model` | 524,288 |
| critic | attention | `critic.encoder.layers.0.self_attn` | 263,168 |
| critic | attention | `critic.encoder.layers.1.self_attn` | 263,168 |
| critic | linear_projection | `critic.q1.deepset.set_decoder.1` | 257 |
| critic | linear_projection | `critic.q2.deepset.set_decoder.1` | 257 |
| critic | mlp | `critic.actor_state_encoder` | 197,120 |
| critic | mlp | `critic.encoder.global_encoder` | 66,304 |
| critic | mlp | `critic.encoder.layers.0.feedforward` | 525,568 |
| critic | mlp | `critic.encoder.layers.1.feedforward` | 525,568 |
| critic | mlp | `critic.encoder.local_action_encoder` | 153,856 |
| critic | mlp | `critic.q1.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q1.deepset.set_decoder.0` | 65,792 |
| critic | mlp | `critic.q2.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q2.deepset.set_decoder.0` | 65,792 |
| critic | normalization | `critic.encoder.layers.0.norm1` | 512 |
| critic | normalization | `critic.encoder.layers.0.norm2` | 512 |
| critic | normalization | `critic.encoder.layers.1.norm1` | 512 |
| critic | normalization | `critic.encoder.layers.1.norm2` | 512 |
| critic | normalization | `critic.encoder.norm` | 512 |

</details>

<details>
<summary>tmasac_slstm_swiglu — 7,449,872 parameters</summary>

Policy: `RecurrentTMASACPolicy`. Parameter storage: 28.419 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `_actor_encoder.backbone` | 1,740,480 | 1,740,480 | 0 |
| actor | `_actor_encoder.temporal_model` | 264,704 | 264,704 | 0 |
| actor | `action_dist` | 3,096 | 3,096 | 0 |
| actor | `actor_head` | 32,896 | 32,896 | 0 |
| critic | `critic.actor_state_encoder` | 197,120 | 197,120 | 0 |
| critic | `critic.encoder` | 1,810,624 | 1,810,624 | 0 |
| critic | `critic.q1` | 197,633 | 197,633 | 0 |
| critic | `critic.q2` | 197,633 | 197,633 | 0 |
| next_obs_prediction | `critic_nop.local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `critic_nop.local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `critic_nop.local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `critic_nop.local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `critic_nop.pre_predictors_transform` | 49,408 | 49,408 | 0 |
| next_obs_prediction | `critic_nop.pre_transition_transform` | 197,376 | 197,376 | 0 |
| next_obs_prediction | `critic_nop.transition_model` | 349,184 | 349,184 | 0 |
| targets | `critic_target.actor_state_encoder` | 197,120 | 0 | 197,120 |
| targets | `critic_target.encoder` | 1,810,624 | 0 | 1,810,624 |
| targets | `critic_target.q1` | 197,633 | 0 | 197,633 |
| targets | `critic_target.q2` | 197,633 | 0 | 197,633 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 1,243,456 | 3,096 | 1,246,552 | 526,336 | 264,704 | 3,584 | 0 | 0 | 2,041,176 |
| critic | 1,872,576 | 514 | 1,873,090 | 526,336 | 0 | 3,584 | 0 | 0 | 2,403,010 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | attention | `_actor_encoder.layers.0.self_attn` | 263,168 |
| actor | attention | `_actor_encoder.layers.1.self_attn` | 263,168 |
| actor | linear_projection | `action_dist.distributions.0.action_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.0.log_std_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.1.action_net` | 516 |
| actor | linear_projection | `action_dist.distributions.1.log_std_net` | 516 |
| actor | mlp | `_actor_encoder.layers.0.feedforward` | 265,136 |
| actor | mlp | `_actor_encoder.layers.0.inter_module_feedforward` | 265,136 |
| actor | mlp | `_actor_encoder.layers.1.feedforward` | 265,136 |
| actor | mlp | `_actor_encoder.layers.1.inter_module_feedforward` | 265,136 |
| actor | mlp | `_actor_encoder.local_obs_encoder` | 150,016 |
| actor | mlp | `actor_head.head` | 32,896 |
| actor | normalization | `_actor_encoder.layers.0.attention_norm` | 512 |
| actor | normalization | `_actor_encoder.layers.0.feedforward_norm` | 512 |
| actor | normalization | `_actor_encoder.layers.0.inter_module_feedforward_norm` | 512 |
| actor | normalization | `_actor_encoder.layers.1.attention_norm` | 512 |
| actor | normalization | `_actor_encoder.layers.1.feedforward_norm` | 512 |
| actor | normalization | `_actor_encoder.layers.1.inter_module_feedforward_norm` | 512 |
| actor | normalization | `_actor_encoder.norm` | 512 |
| actor | recurrent | `_actor_encoder.layers.0.temporal_model` | 132,352 |
| actor | recurrent | `_actor_encoder.layers.1.temporal_model` | 132,352 |
| critic | attention | `critic.encoder.layers.0.self_attn` | 263,168 |
| critic | attention | `critic.encoder.layers.1.self_attn` | 263,168 |
| critic | linear_projection | `critic.q1.deepset.set_decoder.1` | 257 |
| critic | linear_projection | `critic.q2.deepset.set_decoder.1` | 257 |
| critic | mlp | `critic.actor_state_encoder` | 197,120 |
| critic | mlp | `critic.encoder.global_encoder` | 66,304 |
| critic | mlp | `critic.encoder.layers.0.feedforward` | 530,272 |
| critic | mlp | `critic.encoder.layers.1.feedforward` | 530,272 |
| critic | mlp | `critic.encoder.local_action_encoder` | 153,856 |
| critic | mlp | `critic.q1.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q1.deepset.set_decoder.0` | 65,792 |
| critic | mlp | `critic.q2.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q2.deepset.set_decoder.0` | 65,792 |
| critic | normalization | `critic.encoder.layers.0.feedforward.pre_norms.1` | 512 |
| critic | normalization | `critic.encoder.layers.0.norm1` | 512 |
| critic | normalization | `critic.encoder.layers.0.norm2` | 512 |
| critic | normalization | `critic.encoder.layers.1.feedforward.pre_norms.1` | 512 |
| critic | normalization | `critic.encoder.layers.1.norm1` | 512 |
| critic | normalization | `critic.encoder.layers.1.norm2` | 512 |
| critic | normalization | `critic.encoder.norm` | 512 |

</details>

<details>
<summary>mat_ind_no_attention — 1,390,589 parameters</summary>

Policy: `MATIndPolicy`. Parameter storage: 5.305 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `action_dist` | 9,288 | 9,288 | 0 |
| actor | `actor_head` | 32,896 | 32,896 | 0 |
| critic | `critic.deepset` | 198,656 | 198,656 | 0 |
| critic | `critic.popart_head` | 257 | 257 | 0 |
| next_obs_prediction | `local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `pre_predictors_transform` | 49,408 | 49,408 | 0 |
| next_obs_prediction | `pre_transition_transform` | 65,792 | 65,792 | 0 |
| next_obs_prediction | `transition_model` | 349,184 | 349,184 | 0 |
| shared_encoder | `encoder.backbone` | 678,400 | 678,400 | 0 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 32,896 | 9,288 | 42,184 | 0 | 0 | 0 | 0 | 0 | 42,184 |
| critic | 198,656 | 257 | 198,913 | 0 | 0 | 0 | 0 | 0 | 198,913 |
| shared_encoder | 675,840 | 0 | 675,840 | 0 | 0 | 2,560 | 0 | 0 | 678,400 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | linear_projection | `action_dist.distributions.0.output_net` | 6,192 |
| actor | linear_projection | `action_dist.distributions.1.output_net` | 3,096 |
| actor | mlp | `actor_head` | 32,896 |
| critic | linear_projection | `critic.popart_head` | 257 |
| critic | mlp | `critic.deepset.element_encoder` | 132,864 |
| critic | mlp | `critic.deepset.set_decoder.0` | 65,792 |
| shared_encoder | mlp | `encoder.layers.0.feedforward` | 262,912 |
| shared_encoder | mlp | `encoder.layers.1.feedforward` | 262,912 |
| shared_encoder | mlp | `encoder.local_obs_encoder` | 150,016 |
| shared_encoder | normalization | `encoder.layers.0.norm1` | 512 |
| shared_encoder | normalization | `encoder.layers.0.norm2` | 512 |
| shared_encoder | normalization | `encoder.layers.1.norm1` | 512 |
| shared_encoder | normalization | `encoder.layers.1.norm2` | 512 |
| shared_encoder | normalization | `encoder.norm` | 512 |

Tied parameter aliases are counted once; see `counts.json.gz` for their names.

</details>

<details>
<summary>mat_qcx_no_nop — 2,026,697 parameters</summary>

Policy: `MATQCXPolicy`. Parameter storage: 7.731 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `action_dist` | 9,288 | 9,288 | 0 |
| actor | `action_encoder` | 18,176 | 18,176 | 0 |
| actor | `decoder` | 595,584 | 595,584 | 0 |
| critic | `critic.deepset` | 198,656 | 198,656 | 0 |
| critic | `critic.popart_head` | 257 | 257 | 0 |
| shared_encoder | `encoder.backbone` | 1,204,736 | 1,204,736 | 0 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 248,832 | 42,184 | 291,016 | 329,728 | 0 | 2,304 | 0 | 0 | 623,048 |
| critic | 198,656 | 257 | 198,913 | 0 | 0 | 0 | 0 | 0 | 198,913 |
| shared_encoder | 675,840 | 0 | 675,840 | 526,336 | 0 | 2,560 | 0 | 0 | 1,204,736 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | attention | `decoder.layers.0.memory_attn` | 98,816 |
| actor | attention | `decoder.layers.0.query_context_attn` | 66,048 |
| actor | attention | `decoder.layers.1.memory_attn` | 98,816 |
| actor | attention | `decoder.layers.1.query_context_attn` | 66,048 |
| actor | linear_projection | `action_dist.distributions.0.output_net` | 6,192 |
| actor | linear_projection | `action_dist.distributions.1.output_net` | 3,096 |
| actor | linear_projection | `decoder.input_projection` | 32,896 |
| actor | mlp | `action_encoder` | 18,176 |
| actor | mlp | `decoder.layers.0` | 65,920 |
| actor | mlp | `decoder.layers.0.context_encoder` | 49,408 |
| actor | mlp | `decoder.layers.1` | 65,920 |
| actor | mlp | `decoder.layers.1.context_encoder` | 49,408 |
| actor | normalization | `decoder.layers.0.context_token_norm` | 256 |
| actor | normalization | `decoder.layers.0.norm1` | 256 |
| actor | normalization | `decoder.layers.0.norm2` | 256 |
| actor | normalization | `decoder.layers.0.norm3` | 256 |
| actor | normalization | `decoder.layers.1.context_token_norm` | 256 |
| actor | normalization | `decoder.layers.1.norm1` | 256 |
| actor | normalization | `decoder.layers.1.norm2` | 256 |
| actor | normalization | `decoder.layers.1.norm3` | 256 |
| actor | normalization | `decoder.norm` | 256 |
| critic | linear_projection | `critic.popart_head` | 257 |
| critic | mlp | `critic.deepset.element_encoder` | 132,864 |
| critic | mlp | `critic.deepset.set_decoder.0` | 65,792 |
| shared_encoder | attention | `encoder.layers.0.self_attn` | 263,168 |
| shared_encoder | attention | `encoder.layers.1.self_attn` | 263,168 |
| shared_encoder | mlp | `encoder.layers.0.feedforward` | 262,912 |
| shared_encoder | mlp | `encoder.layers.1.feedforward` | 262,912 |
| shared_encoder | mlp | `encoder.local_obs_encoder` | 150,016 |
| shared_encoder | normalization | `encoder.layers.0.norm1` | 512 |
| shared_encoder | normalization | `encoder.layers.0.norm2` | 512 |
| shared_encoder | normalization | `encoder.layers.1.norm1` | 512 |
| shared_encoder | normalization | `encoder.layers.1.norm2` | 512 |
| shared_encoder | normalization | `encoder.norm` | 512 |

Tied parameter aliases are counted once; see `counts.json.gz` for their names.

</details>

<details>
<summary>tmasac_no_nop — 6,025,884 parameters</summary>

Policy: `TMASACPolicy`. Parameter storage: 22.987 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `_actor_encoder.backbone` | 1,730,048 | 1,730,048 | 0 |
| actor | `action_dist` | 3,096 | 3,096 | 0 |
| actor | `actor_head` | 32,896 | 32,896 | 0 |
| critic | `critic.encoder` | 1,734,656 | 1,734,656 | 0 |
| critic | `critic.q1` | 197,633 | 197,633 | 0 |
| critic | `critic.q2` | 197,633 | 197,633 | 0 |
| targets | `critic_target.encoder` | 1,734,656 | 0 | 1,734,656 |
| targets | `critic_target.q1` | 197,633 | 0 | 197,633 |
| targets | `critic_target.q2` | 197,633 | 0 | 197,633 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 1,234,048 | 3,096 | 1,237,144 | 526,336 | 0 | 2,560 | 0 | 0 | 1,766,040 |
| critic | 1,600,512 | 514 | 1,601,026 | 526,336 | 0 | 2,560 | 0 | 0 | 2,129,922 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | attention | `_actor_encoder.layers.0.self_attn` | 263,168 |
| actor | attention | `_actor_encoder.layers.1.self_attn` | 263,168 |
| actor | linear_projection | `action_dist.distributions.0.action_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.0.log_std_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.1.action_net` | 516 |
| actor | linear_projection | `action_dist.distributions.1.log_std_net` | 516 |
| actor | mlp | `_actor_encoder.layers.0.feedforward` | 525,568 |
| actor | mlp | `_actor_encoder.layers.1.feedforward` | 525,568 |
| actor | mlp | `_actor_encoder.local_obs_encoder` | 150,016 |
| actor | mlp | `actor_head.head` | 32,896 |
| actor | normalization | `_actor_encoder.layers.0.norm1` | 512 |
| actor | normalization | `_actor_encoder.layers.0.norm2` | 512 |
| actor | normalization | `_actor_encoder.layers.1.norm1` | 512 |
| actor | normalization | `_actor_encoder.layers.1.norm2` | 512 |
| actor | normalization | `_actor_encoder.norm` | 512 |
| critic | attention | `critic.encoder.layers.0.self_attn` | 263,168 |
| critic | attention | `critic.encoder.layers.1.self_attn` | 263,168 |
| critic | linear_projection | `critic.q1.deepset.set_decoder.1` | 257 |
| critic | linear_projection | `critic.q2.deepset.set_decoder.1` | 257 |
| critic | mlp | `critic.encoder.global_encoder` | 66,304 |
| critic | mlp | `critic.encoder.layers.0.feedforward` | 525,568 |
| critic | mlp | `critic.encoder.layers.1.feedforward` | 525,568 |
| critic | mlp | `critic.encoder.local_action_encoder` | 88,320 |
| critic | mlp | `critic.q1.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q1.deepset.set_decoder.0` | 65,792 |
| critic | mlp | `critic.q2.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q2.deepset.set_decoder.0` | 65,792 |
| critic | normalization | `critic.encoder.layers.0.norm1` | 512 |
| critic | normalization | `critic.encoder.layers.0.norm2` | 512 |
| critic | normalization | `critic.encoder.layers.1.norm1` | 512 |
| critic | normalization | `critic.encoder.layers.1.norm2` | 512 |
| critic | normalization | `critic.encoder.norm` | 512 |

</details>

<details>
<summary>tmasac_slstm_no_nop — 6,817,436 parameters</summary>

Policy: `RecurrentTMASACPolicy`. Parameter storage: 26.006 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `_actor_encoder.backbone` | 1,731,584 | 1,731,584 | 0 |
| actor | `_actor_encoder.temporal_model` | 264,704 | 264,704 | 0 |
| actor | `action_dist` | 3,096 | 3,096 | 0 |
| actor | `actor_head` | 32,896 | 32,896 | 0 |
| critic | `critic.actor_state_encoder` | 197,120 | 197,120 | 0 |
| critic | `critic.encoder` | 1,800,192 | 1,800,192 | 0 |
| critic | `critic.q1` | 197,633 | 197,633 | 0 |
| critic | `critic.q2` | 197,633 | 197,633 | 0 |
| targets | `critic_target.actor_state_encoder` | 197,120 | 0 | 197,120 |
| targets | `critic_target.encoder` | 1,800,192 | 0 | 1,800,192 |
| targets | `critic_target.q1` | 197,633 | 0 | 197,633 |
| targets | `critic_target.q2` | 197,633 | 0 | 197,633 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 1,234,560 | 3,096 | 1,237,656 | 526,336 | 264,704 | 3,584 | 0 | 0 | 2,032,280 |
| critic | 1,863,168 | 514 | 1,863,682 | 526,336 | 0 | 2,560 | 0 | 0 | 2,392,578 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | attention | `_actor_encoder.layers.0.self_attn` | 263,168 |
| actor | attention | `_actor_encoder.layers.1.self_attn` | 263,168 |
| actor | linear_projection | `action_dist.distributions.0.action_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.0.log_std_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.1.action_net` | 516 |
| actor | linear_projection | `action_dist.distributions.1.log_std_net` | 516 |
| actor | mlp | `_actor_encoder.layers.0.feedforward` | 262,912 |
| actor | mlp | `_actor_encoder.layers.0.inter_module_feedforward` | 262,912 |
| actor | mlp | `_actor_encoder.layers.1.feedforward` | 262,912 |
| actor | mlp | `_actor_encoder.layers.1.inter_module_feedforward` | 262,912 |
| actor | mlp | `_actor_encoder.local_obs_encoder` | 150,016 |
| actor | mlp | `actor_head.head` | 32,896 |
| actor | normalization | `_actor_encoder.layers.0.attention_norm` | 512 |
| actor | normalization | `_actor_encoder.layers.0.feedforward_norm` | 512 |
| actor | normalization | `_actor_encoder.layers.0.inter_module_feedforward_norm` | 512 |
| actor | normalization | `_actor_encoder.layers.1.attention_norm` | 512 |
| actor | normalization | `_actor_encoder.layers.1.feedforward_norm` | 512 |
| actor | normalization | `_actor_encoder.layers.1.inter_module_feedforward_norm` | 512 |
| actor | normalization | `_actor_encoder.norm` | 512 |
| actor | recurrent | `_actor_encoder.layers.0.temporal_model` | 132,352 |
| actor | recurrent | `_actor_encoder.layers.1.temporal_model` | 132,352 |
| critic | attention | `critic.encoder.layers.0.self_attn` | 263,168 |
| critic | attention | `critic.encoder.layers.1.self_attn` | 263,168 |
| critic | linear_projection | `critic.q1.deepset.set_decoder.1` | 257 |
| critic | linear_projection | `critic.q2.deepset.set_decoder.1` | 257 |
| critic | mlp | `critic.actor_state_encoder` | 197,120 |
| critic | mlp | `critic.encoder.global_encoder` | 66,304 |
| critic | mlp | `critic.encoder.layers.0.feedforward` | 525,568 |
| critic | mlp | `critic.encoder.layers.1.feedforward` | 525,568 |
| critic | mlp | `critic.encoder.local_action_encoder` | 153,856 |
| critic | mlp | `critic.q1.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q1.deepset.set_decoder.0` | 65,792 |
| critic | mlp | `critic.q2.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q2.deepset.set_decoder.0` | 65,792 |
| critic | normalization | `critic.encoder.layers.0.norm1` | 512 |
| critic | normalization | `critic.encoder.layers.0.norm2` | 512 |
| critic | normalization | `critic.encoder.layers.1.norm1` | 512 |
| critic | normalization | `critic.encoder.layers.1.norm2` | 512 |
| critic | normalization | `critic.encoder.norm` | 512 |

</details>

<details>
<summary>tmasac_lstm_no_actor_state — 6,232,272 parameters</summary>

Policy: `RecurrentTMASACPolicy`. Parameter storage: 23.774 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `_actor_encoder.backbone` | 1,335,808 | 1,335,808 | 0 |
| actor | `_actor_encoder.temporal_model` | 1,048,576 | 1,048,576 | 0 |
| actor | `action_dist` | 3,096 | 3,096 | 0 |
| actor | `actor_head` | 32,896 | 32,896 | 0 |
| critic | `critic.encoder` | 1,209,344 | 1,209,344 | 0 |
| critic | `critic.q1` | 197,633 | 197,633 | 0 |
| critic | `critic.q2` | 197,633 | 197,633 | 0 |
| next_obs_prediction | `critic_nop.local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `critic_nop.local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `critic_nop.local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `critic_nop.local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `critic_nop.pre_predictors_transform` | 49,408 | 49,408 | 0 |
| next_obs_prediction | `critic_nop.pre_transition_transform` | 197,376 | 197,376 | 0 |
| next_obs_prediction | `critic_nop.transition_model` | 349,184 | 349,184 | 0 |
| targets | `critic_target.encoder` | 1,209,344 | 0 | 1,209,344 |
| targets | `critic_target.q1` | 197,633 | 0 | 197,633 |
| targets | `critic_target.q2` | 197,633 | 0 | 197,633 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 708,736 | 134,168 | 842,904 | 526,336 | 1,048,576 | 2,560 | 0 | 0 | 2,420,376 |
| critic | 1,075,200 | 514 | 1,075,714 | 526,336 | 0 | 2,560 | 0 | 0 | 1,604,610 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | attention | `_actor_encoder.layers.0.self_attn` | 263,168 |
| actor | attention | `_actor_encoder.layers.1.self_attn` | 263,168 |
| actor | linear_projection | `_actor_encoder.layers.0.temporal_output_projection` | 65,536 |
| actor | linear_projection | `_actor_encoder.layers.1.temporal_output_projection` | 65,536 |
| actor | linear_projection | `action_dist.distributions.0.action_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.0.log_std_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.1.action_net` | 516 |
| actor | linear_projection | `action_dist.distributions.1.log_std_net` | 516 |
| actor | mlp | `_actor_encoder.layers.0.feedforward` | 262,912 |
| actor | mlp | `_actor_encoder.layers.1.feedforward` | 262,912 |
| actor | mlp | `_actor_encoder.local_obs_encoder` | 150,016 |
| actor | mlp | `actor_head.head` | 32,896 |
| actor | normalization | `_actor_encoder.layers.0.attention_norm` | 512 |
| actor | normalization | `_actor_encoder.layers.0.feedforward_norm` | 512 |
| actor | normalization | `_actor_encoder.layers.1.attention_norm` | 512 |
| actor | normalization | `_actor_encoder.layers.1.feedforward_norm` | 512 |
| actor | normalization | `_actor_encoder.norm` | 512 |
| actor | recurrent | `_actor_encoder.layers.0.temporal_model` | 524,288 |
| actor | recurrent | `_actor_encoder.layers.1.temporal_model` | 524,288 |
| critic | attention | `critic.encoder.layers.0.self_attn` | 263,168 |
| critic | attention | `critic.encoder.layers.1.self_attn` | 263,168 |
| critic | linear_projection | `critic.q1.deepset.set_decoder.1` | 257 |
| critic | linear_projection | `critic.q2.deepset.set_decoder.1` | 257 |
| critic | mlp | `critic.encoder.global_encoder` | 66,304 |
| critic | mlp | `critic.encoder.layers.0.feedforward` | 262,912 |
| critic | mlp | `critic.encoder.layers.1.feedforward` | 262,912 |
| critic | mlp | `critic.encoder.local_action_encoder` | 88,320 |
| critic | mlp | `critic.q1.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q1.deepset.set_decoder.0` | 65,792 |
| critic | mlp | `critic.q2.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q2.deepset.set_decoder.0` | 65,792 |
| critic | normalization | `critic.encoder.layers.0.norm1` | 512 |
| critic | normalization | `critic.encoder.layers.0.norm2` | 512 |
| critic | normalization | `critic.encoder.layers.1.norm1` | 512 |
| critic | normalization | `critic.encoder.layers.1.norm2` | 512 |
| critic | normalization | `critic.encoder.norm` | 512 |

</details>

<details>
<summary>tmasac_segment — 5,052,624 parameters</summary>

Policy: `SegmentTMASACPolicy`. Parameter storage: 19.274 MiB.

| Role | Module | Parameters | Trainable | Frozen |
| --- | --- | ---: | ---: | ---: |
| actor | `_actor_encoder.backbone` | 1,204,736 | 1,204,736 | 0 |
| actor | `action_dist` | 3,096 | 3,096 | 0 |
| actor | `actor_head` | 32,896 | 32,896 | 0 |
| critic | `critic.encoder` | 1,209,344 | 1,209,344 | 0 |
| critic | `critic.q1` | 197,633 | 197,633 | 0 |
| critic | `critic.q2` | 197,633 | 197,633 | 0 |
| next_obs_prediction | `critic_nop.local_angles_predictor` | 1,548 | 1,548 | 0 |
| next_obs_prediction | `critic_nop.local_binaries_predictor` | 516 | 516 | 0 |
| next_obs_prediction | `critic_nop.local_rot6ds_predictor` | 387 | 387 | 0 |
| next_obs_prediction | `critic_nop.local_scalars_predictor` | 4,257 | 4,257 | 0 |
| next_obs_prediction | `critic_nop.pre_predictors_transform` | 49,408 | 49,408 | 0 |
| next_obs_prediction | `critic_nop.pre_transition_transform` | 197,376 | 197,376 | 0 |
| next_obs_prediction | `critic_nop.transition_model` | 349,184 | 349,184 | 0 |
| targets | `critic_target.encoder` | 1,209,344 | 0 | 1,209,344 |
| targets | `critic_target.q1` | 197,633 | 0 | 197,633 |
| targets | `critic_target.q2` | 197,633 | 0 | 197,633 |

Online main-policy processing by role (NOP/targets excluded):

| Role | MLP | Linear projection | MLP + linear | Attention | Recurrent | Normalization | Embedding | Other | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor | 708,736 | 3,096 | 711,832 | 526,336 | 0 | 2,560 | 0 | 0 | 1,240,728 |
| critic | 1,075,200 | 514 | 1,075,714 | 526,336 | 0 | 2,560 | 0 | 0 | 1,604,610 |

Online processing components (all seven kinds are additive):

| Role | Kind | Module | Parameters |
| --- | --- | --- | ---: |
| actor | attention | `_actor_encoder.layers.0.self_attn` | 263,168 |
| actor | attention | `_actor_encoder.layers.1.self_attn` | 263,168 |
| actor | linear_projection | `action_dist.distributions.0.action_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.0.log_std_net` | 1,032 |
| actor | linear_projection | `action_dist.distributions.1.action_net` | 516 |
| actor | linear_projection | `action_dist.distributions.1.log_std_net` | 516 |
| actor | mlp | `_actor_encoder.layers.0.feedforward` | 262,912 |
| actor | mlp | `_actor_encoder.layers.1.feedforward` | 262,912 |
| actor | mlp | `_actor_encoder.local_obs_encoder` | 150,016 |
| actor | mlp | `actor_head.head` | 32,896 |
| actor | normalization | `_actor_encoder.layers.0.norm1` | 512 |
| actor | normalization | `_actor_encoder.layers.0.norm2` | 512 |
| actor | normalization | `_actor_encoder.layers.1.norm1` | 512 |
| actor | normalization | `_actor_encoder.layers.1.norm2` | 512 |
| actor | normalization | `_actor_encoder.norm` | 512 |
| critic | attention | `critic.encoder.layers.0.self_attn` | 263,168 |
| critic | attention | `critic.encoder.layers.1.self_attn` | 263,168 |
| critic | linear_projection | `critic.q1.deepset.set_decoder.1` | 257 |
| critic | linear_projection | `critic.q2.deepset.set_decoder.1` | 257 |
| critic | mlp | `critic.encoder.global_encoder` | 66,304 |
| critic | mlp | `critic.encoder.layers.0.feedforward` | 262,912 |
| critic | mlp | `critic.encoder.layers.1.feedforward` | 262,912 |
| critic | mlp | `critic.encoder.local_action_encoder` | 88,320 |
| critic | mlp | `critic.q1.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q1.deepset.set_decoder.0` | 65,792 |
| critic | mlp | `critic.q2.deepset.element_encoder` | 131,584 |
| critic | mlp | `critic.q2.deepset.set_decoder.0` | 65,792 |
| critic | normalization | `critic.encoder.layers.0.norm1` | 512 |
| critic | normalization | `critic.encoder.layers.0.norm2` | 512 |
| critic | normalization | `critic.encoder.layers.1.norm1` | 512 |
| critic | normalization | `critic.encoder.layers.1.norm2` | 512 |
| critic | normalization | `critic.encoder.norm` | 512 |

</details>
