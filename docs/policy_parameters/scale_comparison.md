# Model-scale comparison

WallEasy, CPU, compilation disabled. Counts include unique online weights and biases; frozen targets are excluded. NOP is reported separately. Snapshots cover 17 core presets, two NOP controls and three gated Deep Set presets. Layout versions are small v6, 5M-2L v1, 5M-3L v2, and large v7.

The scale is the feed-forward reference budget. DDPG has one ordinary Q head; recurrent policies add temporal modules to unchanged backbones; gated policies add reception gates. Recurrent 3L memory widths are reduced to match the corresponding 2L recurrent totals; feed-forward backbones remain unchanged. Gate additions are not compensated by shrinking other components.

| Variant | 2.5M main | Small NOP | 5M-2L main | Medium NOP | 10M main | Large NOP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ppo | 2.681M | 0.668M | 4.927M | 0.973M | 10.261M | 1.947M |
| mappo | 2.388M | 0.668M | 4.524M | 0.973M | 9.364M | 1.947M |
| mat_orig | 2.391M | 0.651M | 4.793M | 0.957M | 10.164M | 1.873M |
| mat_ind | 2.480M | 0.668M | 4.545M | 0.973M | 9.776M | 1.897M |
| mat_dec | 2.480M | 0.668M | 4.546M | 0.973M | 9.972M | 1.897M |
| mat_qcx | 2.366M | 0.651M | 4.821M | 0.957M | 9.603M | 1.873M |
| mat_ind__lstm | 3.808M | 0.668M | 6.907M | 0.973M | 13.318M | 1.897M |
| mat_qcx__lstm | 3.325M | 0.651M | 6.666M | 0.957M | 12.370M | 1.873M |
| maddpg__deepset_critic | 2.357M | 0.684M | 4.371M | 1.006M | 10.425M | 1.947M |
| matd3__deepset_critic | 2.624M | 0.684M | 4.654M | 1.006M | 11.023M | 1.947M |
| masac__deepset_critic | 2.625M | 0.684M | 4.656M | 1.006M | 11.026M | 1.947M |
| tmatd3 | 2.616M | 0.684M | 4.636M | 1.006M | 10.801M | 1.947M |
| tmatd3_dec | 2.617M | 0.684M | 4.637M | 1.006M | 10.997M | 1.947M |
| tmasac | 2.618M | 0.684M | 4.638M | 1.006M | 10.803M | 1.947M |
| tmasac_dec | 2.618M | 0.684M | 4.638M | 1.006M | 11.000M | 1.947M |
| tmasac__slstm | 2.928M | 0.684M | 5.199M | 1.006M | 11.694M | 1.947M |
| tmasac__lstm | 3.369M | 0.684M | 5.983M | 1.006M | 12.870M | 1.947M |
| maddpg__deepset_critic__gated | 2.502M | 0.684M | 4.659M | 1.006M | 10.879M | 1.947M |
| matd3__deepset_critic__gated | 2.798M | 0.684M | 4.971M | 1.006M | 11.542M | 1.947M |
| masac__deepset_critic__gated | 2.799M | 0.684M | 4.973M | 1.006M | 11.545M | 1.947M |

## Five-million depth comparison

Only attention/Deep Set encoder stacks vary between these choices. MLP depths, decoder stacks, Q/V readout templates, and NOP core stay fixed. Three-layer widths are reduced to keep the feed-forward budgets comparable. PPO/MAPPO are unchanged controls. Recurrent 3L temporal modules use smaller hidden states and projections to match complete 2L recurrent counts within 0.5%. NOP source-adapter counts follow source width.

| Variant | 5M-2L main | 5M-3L main | 2L NOP | 3L NOP |
| --- | ---: | ---: | ---: | ---: |
| ppo | 4.927M | 4.927M | 0.973M | 0.973M |
| mappo | 4.524M | 4.524M | 0.973M | 0.973M |
| mat_orig | 4.793M | 4.858M | 0.957M | 0.957M |
| mat_ind | 4.545M | 4.574M | 0.973M | 0.973M |
| mat_dec | 4.546M | 4.560M | 0.973M | 0.973M |
| mat_qcx | 4.821M | 4.886M | 0.957M | 0.957M |
| mat_ind__lstm | 6.907M | 6.937M | 0.973M | 0.973M |
| mat_qcx__lstm | 6.666M | 6.669M | 0.957M | 0.957M |
| maddpg__deepset_critic | 4.371M | 4.269M | 1.006M | 0.989M |
| matd3__deepset_critic | 4.654M | 4.544M | 1.006M | 0.989M |
| masac__deepset_critic | 4.656M | 4.545M | 1.006M | 0.989M |
| tmatd3 | 4.636M | 4.540M | 1.006M | 0.989M |
| tmatd3_dec | 4.637M | 4.526M | 1.006M | 0.989M |
| tmasac | 4.638M | 4.542M | 1.006M | 0.989M |
| tmasac_dec | 4.638M | 4.528M | 1.006M | 0.989M |
| tmasac__slstm | 5.199M | 5.207M | 1.006M | 0.989M |
| tmasac__lstm | 5.983M | 5.969M | 1.006M | 0.989M |
| mat_qcx__no_nop | 4.821M | 4.886M | 0.000M | 0.000M |
| tmasac__no_nop | 4.638M | 4.542M | 0.000M | 0.000M |
| maddpg__deepset_critic__gated | 4.659M | 4.568M | 1.006M | 0.989M |
| matd3__deepset_critic__gated | 4.971M | 4.872M | 1.006M | 0.989M |
| masac__deepset_critic__gated | 4.973M | 4.874M | 1.006M | 0.989M |

`5M-2L` is the unchanged former 5M architecture. The unqualified `5M` aliases have been removed. [Layout details](model_scale.md#five-million-encoder-depth-comparison).

## Shared critic comparison

Deep Set and transformer critics use the same token embeddings, local/global input placement, encoder width, number of blocks, normalization, residuals and local FFNs. The shared encoder changes only the communication operator. Deep Set messages expand to 1.25 times token width and receive nonlinear processing before and after masked mean pooling.

Every Q head has the same projection, two expanding Deep Set interactions, masked mean pooling and two-hidden-layer scalar regressor. DDPG uses one head; TD3 and SAC use two private heads. Actor parameters are separate from critic parameters.

| 5M-2L variant | Actor | Critic | Difference from matching transformer critic |
| --- | ---: | ---: | ---: |
| maddpg__deepset_critic | 1.765M | 2.606M | one fewer Q head |
| matd3__deepset_critic | 1.765M | 2.889M | +0.60% |
| masac__deepset_critic | 1.766M | 2.889M | +0.60% |
| tmatd3_dec | 1.765M | 2.872M | +0.00% |
| tmasac_dec | 1.766M | 2.872M | +0.00% |

The communication weight approximation is 4.0625D² for expanding Deep Set versus 4D² for attention. Measured complete critic counts, including embeddings and heads, stay close across all three tiers. This controls parameter allocation, not expressive power, optimization difficulty or compute.


## Recurrent and gated additions

LSTM and sLSTM preserve the corresponding feed-forward widths, depths, embeddings, local FFNs and readouts. Temporal modules use common insertion order, outer normalization and residual settings, without an extra inter-module FFN. In 5M-3L v2, bias-free memory input/output projections permit narrower temporal states; actor-state critic adapters follow the actual state width. MAT keeps independent actor/critic histories; off-policy canonical recurrent actors keep the explicit detached actor-state bridge to their ordinary critic. The bridge is additional capacity and input information, not a claim of an actor-only recurrence ablation.

Gated presets retain all ungated processing dimensions. Gates consume each normalized token and the processed global message, use a D/4-wide hidden layer, and select expanded message features before bias-free compression. They introduce receiver-dependent feature selection while retaining one pooled message. Their added parameters are visible in the table.

## Training controls

TMASAC is the reference. MASAC shares its actor/critic learning rate (`5e-5`), initial automatic temperature (`0.05`), inherited temperature learning rate and target entropy (`auto_0.1`). Explicit overrides remain available. On-policy decoder learning rates retain the empirically motivated multiplier of `0.25`.

TMASAC/TMATD3 retain centralized attention actors. Compare their decentralized `*_dec` controls against MASAC/MATD3 to isolate critic communication; comparing the primary centralized variants also changes actor communication. Parameter matching does not match compute. Recurrent replay counts sequences, while feed-forward replay counts transitions: the default 16 sequences × 64 learning steps have a different replay exposure from a feed-forward batch unless configured explicitly. Use independent training seeds for uncertainty estimates; evaluation seeds are not training replicas.

Training examples reserve unique output directories by scale, training seed, NOP mode and run ID, including repeated invocations of the same configuration. Existing explicit output directories are rejected. Actual resolved hyperparameters remain in each run's metadata.

## Registry and checkpoints

`list_variants(include_hidden=True)` adds the two NOP controls (`mat_qcx__no_nop`, `tmasac__no_nop`) and three `*__deepset_critic__gated` presets. `mat_ind__no_attention` is removed. Independent Q encoders are a low-level option, not a registered preset; selecting them duplicates full-width encoders without resizing the primary encoder or heads.

Expanding interactions, common Q heads and additive recurrent layouts replace earlier architecture versions. Existing weights require retraining or manual migration; `legacy` selects current explicit-width templates, not old architectures. Checkpoints record their resolved configuration and layout version. The [historical audit](preset_audit.md) describes earlier designs.

NOP uses a common auxiliary core with source-width-dependent adapters. MAT attaches it to actor features; off-policy defaults attach it to shared pre-head critic features. PPO/MAPPO have shared MLP backbones and different input organizations, so their totals do not establish the same component control as the aligned off-policy critics.

Full role counts and layer dimensions are stored in the compressed snapshots. Verbose reports are generated under `generated/<scale>/`. Regenerate counts and diagrams with:

```bash
swarmbots-diagram-policies --refresh --include-hidden
```

See the [fixed layouts](model_scale.md), [inspector instructions](README.md), and [diagram reading guide](diagrams/README.md).
