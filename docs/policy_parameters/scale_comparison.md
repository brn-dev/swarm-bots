# Model-scale comparison

WallEasy, CPU, compilation disabled. These are unique online main-policy and auxiliary parameters; frozen targets are excluded. All three tiers use fixed declared layouts. The 5M v2 layout uses `192 → 256 → 192` FFNs in the QCX decoder blocks. The 10M v2 tier uses three main attention/recurrent/decoder blocks and three NOP transition blocks, with four blocks in FF-only actors.

| Tier | Nominal main | Actual main range | Nominal NOP | Actual NOP range |
| --- | ---: | ---: | ---: | ---: |
| 2.5M | 2.5M | 2.365M–2.790M | 0.75M | 0.651M–0.815M |
| 5M | 5M | 4.460M–5.521M | 1M | 0.957M–1.170M |
| 10M | 10M | 9.489M–11.183M | 2M | 1.897M–2.389M |

The small tier keeps the 5M tier's projection and prediction widths, with a wider feed-forward expansion in its smaller transformer. Its actual NOP capacity stays above a linear reduction to 0.5M.

| Variant | 2.5M main | Small NOP | 5M main | Medium NOP | 10M main | Large NOP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ppo | 2.681M | 0.668M | 4.926M | 0.973M | 10.260M | 1.947M |
| mappo | 2.493M | 0.668M | 4.712M | 0.973M | 9.787M | 1.947M |
| mat_orig | 2.452M | 0.668M | 4.520M | 0.973M | 10.035M | 1.897M |
| mat_ind | 2.488M | 0.668M | 4.464M | 0.973M | 9.539M | 1.897M |
| mat_dec | 2.609M | 0.668M | 4.688M | 0.973M | 10.422M | 1.897M |
| mat_qcx | 2.533M | 0.668M | 4.742M | 0.973M | 9.839M | 1.897M |
| mat_ind_lstm | 2.426M | 0.651M | 4.460M | 0.957M | 9.933M | 1.897M |
| mat_qcx_lstm | 2.446M | 0.651M | 4.687M | 0.957M | 10.233M | 1.897M |
| maddpg_deepset | 2.365M | 0.750M | 4.789M | 1.104M | 9.489M | 2.192M |
| matd3_deepset | 2.554M | 0.815M | 4.487M | 1.170M | 10.194M | 2.389M |
| masac_deepset | 2.556M | 0.815M | 4.488M | 1.170M | 10.197M | 2.389M |
| tmatd3 | 2.478M | 0.684M | 4.957M | 1.006M | 10.491M | 1.947M |
| tmatd3_dec | 2.478M | 0.684M | 4.958M | 1.006M | 10.688M | 1.947M |
| tmasac | 2.479M | 0.684M | 4.959M | 1.006M | 10.494M | 1.947M |
| tmasac_dec | 2.479M | 0.684M | 4.959M | 1.006M | 10.690M | 1.947M |
| tmasac_shared_encoder | 2.605M | 0.668M | 5.031M | 0.973M | 9.780M | 1.897M |
| tmasac_slstm | 2.790M | 0.684M | 5.521M | 1.006M | 10.796M | 1.947M |
| tmasac_lstm | 2.594M | 0.684M | 5.249M | 1.006M | 11.183M | 1.947M |

Full role breakdowns, module widths, and processing counts can be [generated locally](README.md) from the compact audits. See [layout declarations and rationale](model_scale.md).

Reproduce the added-tier audits:

```bash
python examples/inspect_policy_parameters.py --model-scale 2.5M --output-dir docs/policy_parameters/generated/2.5M
python examples/inspect_policy_parameters.py --model-scale 10M --output-dir docs/policy_parameters/generated/10M
```
