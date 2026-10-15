# Model-scale comparison

WallEasy, CPU, compilation disabled. Counts are unique online main-policy and auxiliary parameters; frozen targets are excluded. The current registry has 17 core presets and three optional attention/NOP controls. All MAT variants have independent actor and critic encoders. Recurrent MAT maintains a separate temporal history on each side. Layout versions are 2.5M v5, 5M v6, and 10M v6.

| Tier | Nominal main | Actual main range | Nominal NOP | Actual NOP range |
| --- | ---: | ---: | ---: | ---: |
| 2.5M | 2.5M | 2.364M–2.925M | 0.75M | 0.651M–0.684M |
| 5M | 5M | 4.517M–5.196M | 1M | 0.940M–1.006M |
| 10M | 10M | 9.347M–11.485M | 2M | 1.873M–1.947M |

| Variant | 2.5M main | Small NOP | 5M main | Medium NOP | 10M main | Large NOP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ppo | 2.681M | 0.668M | 4.927M | 0.973M | 10.261M | 1.947M |
| mappo | 2.384M | 0.668M | 4.517M | 0.973M | 9.347M | 1.947M |
| mat_orig | 2.391M | 0.651M | 4.793M | 0.957M | 10.164M | 1.873M |
| mat_ind | 2.478M | 0.668M | 4.541M | 0.973M | 9.768M | 1.897M |
| mat_dec | 2.478M | 0.668M | 4.542M | 0.973M | 9.965M | 1.897M |
| mat_qcx | 2.364M | 0.651M | 4.817M | 0.957M | 9.595M | 1.873M |
| mat_ind__lstm | 2.501M | 0.651M | 4.741M | 0.957M | 10.556M | 1.897M |
| mat_qcx__lstm | 2.671M | 0.651M | 4.778M | 0.940M | 10.494M | 1.873M |
| maddpg__deepset_critic | 2.540M | 0.668M | 4.752M | 0.973M | 10.231M | 1.947M |
| matd3__deepset_critic | 2.656M | 0.668M | 4.688M | 0.973M | 10.693M | 1.947M |
| masac__deepset_critic | 2.658M | 0.668M | 4.689M | 0.973M | 10.695M | 1.947M |
| tmatd3 | 2.613M | 0.684M | 4.633M | 1.006M | 10.793M | 1.947M |
| tmatd3_dec | 2.613M | 0.684M | 4.633M | 1.006M | 10.989M | 1.947M |
| tmasac | 2.615M | 0.684M | 4.635M | 1.006M | 10.795M | 1.947M |
| tmasac_dec | 2.615M | 0.684M | 4.635M | 1.006M | 10.992M | 1.947M |
| tmasac__slstm | 2.925M | 0.684M | 5.196M | 1.006M | 11.097M | 1.947M |
| tmasac__lstm | 2.729M | 0.684M | 4.925M | 1.006M | 11.485M | 1.947M |

## Independent MAT budgets

Every MAT actor includes its own encoder and its action head or autoregressive decoder. Every MAT critic includes its own encoder and team-V head. No MAT main-policy weights are attributed to a shared role. Encoder and readout widths are rebalanced together rather than duplicating the previous shared encoder without paying for it.

| MAT variant | 5M actor | 5M critic | 5M shared |
| --- | ---: | ---: | ---: |
| mat_orig | 1.813M | 2.980M | 0 |
| mat_ind | 1.772M | 2.769M | 0 |
| mat_dec | 1.772M | 2.769M | 0 |
| mat_qcx | 2.047M | 2.769M | 0 |
| mat_ind__lstm | 1.874M | 2.867M | 0 |
| mat_qcx__lstm | 1.911M | 2.867M | 0 |

MAT-IND and decentralized MAT use a small private action head after their own encoder. Autoregressive actors use smaller encoder widths so the decoder fits inside the complete actor allowance. The ordinary MAT Deep Set critics share one critic template across IND, DEC and QCX; recurrent IND/QCX use the same private recurrent-critic template. Original MAT retains token-wise V prediction followed by masked pooling.

The nominal actor/critic allocation remains 40:60. This is a reviewed starting budget, not an established optimum. MAT actor NOP consumes actor encoder features and trains only the actor representation and auxiliary model. Off-policy defaults retain critic NOP. PPO and MAPPO retain their own shared MLP backbone outside the MAT family. Sharing an encoder between two Q heads is also distinct from sharing it between actor and critic; the former remains configurable in off-policy policies.

## Registry and checkpoint changes

`*__mlp_critic`, SwiGLU, shared-encoder TMASAC, optional recurrent no-residual/no-NOP/no-actor-state controls, and segment-training variants are no longer registered. The canonical `tmasac__lstm`, `tmasac__slstm`, `mat_ind__lstm`, and `mat_qcx__lstm` remain. Plain `ppo` and `mappo` remain registered. Their low-level MLP/GLU/shared/segment implementation classes and configuration machinery are retained, but removed names are rejected by the training helpers and CLIs.

`list_variants(include_hidden=True)` adds only `mat_ind__no_attention`, `mat_qcx__no_nop`, and `tmasac__no_nop`. The [historical preset audit](preset_audit.md) describes older architectures and a larger registry; it does not describe these layouts.

Previous MAT checkpoints require retraining or manual migration. Their shared `encoder` weights have no automatic mapping to the new `actor_encoder` and `critic_encoder`, and recurrent state is now a mapping with separate `actor` and `critic` entries. Checkpoints record the current resolved actor and critic encoder configs and scale version. Runtime fitting and automatic migration are not provided.

## Remaining comparison limits

The main ranges remain broad size families. Counts do not establish equal compute or optimal width/depth. Flattened PPO depends on padded agent count, so inspect the actual task shape when comparing it with structured policies. The small NOP tier retains its larger projection/predictor allowance; all tiers use one common auxiliary core with source-width-dependent adapters. Actor state conditioning and recurrent residual choices remain part of the canonical recurrent TMASAC architectures and should be reported when comparing them.

Full role counts, layer dimensions and processing budgets are generated under `generated/<scale>/`. Regenerate current audits and diagrams with:

```bash
swarmbots-diagram-policies --refresh
```

See the [fixed layouts](model_scale.md), [inspector instructions](README.md), and [diagram reading guide](diagrams/README.md).
