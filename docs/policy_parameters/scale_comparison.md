# Model-scale comparison

WallEasy, CPU, compilation disabled. These are unique online main-policy and auxiliary parameters; frozen targets are excluded. All tiers use fixed declared layouts with four normalized residual interaction blocks in standalone DDPG/TD3/SAC Deep Set critics and two in on-policy critics and transformer Q readouts. Element widths are rebalanced for the deeper critics. Layouts are 2.5M v3, 5M v4, and 10M v4; the 5M QCX decoder retains its `192 → 256 → 192` FFN, and 10M retains three main encoder/decoder and NOP transition blocks, with four blocks in FF-only actors.

| Tier | Nominal main | Actual main range | Nominal NOP | Actual NOP range |
| --- | ---: | ---: | ---: | ---: |
| 2.5M | 2.5M | 2.352M–2.924M | 0.75M | 0.651M–0.684M |
| 5M | 5M | 4.267M–5.195M | 1M | 0.957M–1.006M |
| 10M | 10M | 9.243M–11.483M | 2M | 1.897M–1.996M |

The small tier keeps the 5M tier's projection and prediction widths, with a wider feed-forward expansion in its smaller transformer. Its actual NOP capacity stays above a linear reduction to 0.5M.

| Variant | 2.5M main | Small NOP | 5M main | Medium NOP | 10M main | Large NOP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ppo | 2.681M | 0.668M | 4.927M | 0.973M | 10.261M | 1.947M |
| mappo | 2.383M | 0.668M | 4.516M | 0.973M | 9.345M | 1.947M |
| mat_orig | 2.452M | 0.668M | 4.520M | 0.973M | 10.035M | 1.897M |
| mat_ind | 2.377M | 0.668M | 4.267M | 0.973M | 9.243M | 1.897M |
| mat_dec | 2.478M | 0.668M | 4.541M | 0.973M | 9.964M | 1.897M |
| mat_qcx | 2.422M | 0.668M | 4.544M | 0.973M | 9.542M | 1.897M |
| mat_ind_lstm | 2.352M | 0.651M | 4.311M | 0.957M | 9.637M | 1.897M |
| mat_qcx_lstm | 2.372M | 0.651M | 4.539M | 0.957M | 9.936M | 1.897M |
| maddpg_deepset | 2.538M | 0.668M | 4.750M | 0.973M | 10.228M | 1.947M |
| matd3_deepset | 2.515M | 0.684M | 4.832M | 1.006M | 10.116M | 1.996M |
| masac_deepset | 2.516M | 0.684M | 4.834M | 1.006M | 10.118M | 1.996M |
| tmatd3 | 2.612M | 0.684M | 4.632M | 1.006M | 10.791M | 1.947M |
| tmatd3_dec | 2.612M | 0.684M | 4.632M | 1.006M | 10.988M | 1.947M |
| tmasac | 2.614M | 0.684M | 4.634M | 1.006M | 10.794M | 1.947M |
| tmasac_dec | 2.614M | 0.684M | 4.634M | 1.006M | 10.990M | 1.947M |
| tmasac_shared_encoder | 2.739M | 0.668M | 4.705M | 0.973M | 10.080M | 1.897M |
| tmasac_slstm | 2.924M | 0.684M | 5.195M | 1.006M | 11.096M | 1.947M |
| tmasac_lstm | 2.728M | 0.684M | 4.924M | 1.006M | 11.483M | 1.947M |

Full role breakdowns, module widths, and processing counts can be [generated locally](README.md) from the compact audits. See [layout declarations and rationale](model_scale.md).

Reproduce the added-tier audits:

```bash
swarmbots-inspect-policies --model-scale 2.5M --output-dir docs/policy_parameters/generated/2.5M
swarmbots-inspect-policies --model-scale 10M --output-dir docs/policy_parameters/generated/10M
```
