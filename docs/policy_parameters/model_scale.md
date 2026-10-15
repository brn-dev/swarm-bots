# Fixed model-scale layouts

The supported scales select explicit architecture layouts in [model_scale.py](../../swarmbots/learn/presets/model_scale.py). There is no runtime parameter fitting or task-dependent hidden-width search.

| Canonical name | Main budget | NOP budget | Current version |
| --- | ---: | ---: | --- |
| `2.5M NOP0.75M` | 2.5M | 0.75M | `2.5+0.75M-v5` |
| `5M NOP1M` (default) | 5M | 1M | `5+1M-v6` |
| `10M NOP2M` | 10M | 2M | `10+2M-v6` |

Short names `2.5M`, `5M`, and `10M` select these tiers. `model_scale=None` (CLI `legacy`) uses current explicit-width builder arguments. It does not restore old shared MAT architectures. Fixed-scale dimensions take precedence over low-level width arguments.

## Actor and critic ownership

MAT-Orig, MAT-IND, MAT-DEC, MAT-QCX and both recurrent MAT variants always have separate `actor_encoder` and `critic_encoder` modules. Config `encoder_config` describes the actor; `critic_encoder_config` describes the critic. A missing critic config uses the actor's architecture as a template, creating a separate module with independent weights. Privileged observations enter only the value head; they never enter actor encoding or decoding. MAT-DEC disables actor attention while retaining an attention critic.

Recurrent MAT uses independent recurrent encoders and a temporal-state mapping with `actor` and `critic` histories. Rollout advances both, actor-only inference advances the actor history, and value-only bootstrap advances the critic history. Reset masks apply to each history when it is advanced. Sequence training uses both saved initial histories, episode reset masks and time masks. NOP consumes actor encoder features, so its gradients do not update the value encoder or head.

The nominal complete-actor/complete-critic budgets are 1M/1.5M, 2M/3M and 4M/6M. MAT no longer uses half-shared attribution. PPO/MAPPO retain their shared MLP backbone: nominal shared/private-actor/private-critic budgets are 1M/0.5M/1M, 2M/1M/2M and 4M/2M/4M, with half the backbone attributed to each side for accounting. Both sides still execute the full backbone.

## MAT tier dimensions

All attention encoders and decoders use four heads. Counts include weights, biases, attention, recurrence, normalization and output heads; frozen targets, buffers, optimizer state and NOP are excluded from main counts.

| Component | 2.5M | 5M | 10M |
| --- | --- | --- | --- |
| Main encoder / decoder blocks | 2 | 2 | 3 |
| Decentralized actor blocks | 3 | 3 | 4 |
| MAT-IND / MAT-DEC actor D / FF | 192 / `[384,384]` | 256 / `[512,512]` | 256 / `[768,768]` |
| MAT-IND / MAT-DEC action head | 128 | 128 | 192 |
| Orig / QCX actor encoder D / FF | 128 / `[256,256]` | 192 / `[384,384]` | 192 / `[512,512]` |
| MAT-Orig decoder D / FF | 128 / 128 | 192 / 192 | 256 / 256 |
| QCX decoder D / FF | 128 / 128 | 192 / 256 | 192 / 512 |
| Feed-forward MAT critic encoder D / FF | 192 / `[384,384]` | 256 / `[512,512]` | 256 / `[768,768]` |
| Deep Set V element / interaction width | 128 | 192 | 256 |
| Deep Set V regressor | `[512,384]` | `[768,512]` | `[1024,768]` |
| Original MAT token-wise V hidden widths | `[512,512,384]` | `[768,768,512]` | `[1024,1024,768]` |
| Recurrent IND actor D / each FF stage | 128 / 512 | 192 / 512 | 256 / 512 |
| Recurrent QCX actor D / each FF stage | 128 / 256 | 128 / 384 | 192 / 384 |
| Recurrent MAT critic D / each FF stage | 128 / 512 | 192 / 512 | 256 / 512 |

The IND actor pays for its encoder and only a small action head. Autoregressive actors reserve more of their budget for the decoder and therefore use a smaller encoder. All MAT Deep Set value heads account for a private critic encoder; recurrent critics keep their own history and the same V-readout template. Original MAT retains token-wise prediction and masked pooling. These layouts are starting points; establishing the best width/depth trade-off requires training experiments.

## Other families

PPO/MAPPO retain their existing fixed MLP layouts and latent widths 192/256/384. Standalone Deep Set DDPG/TD3/SAC use the common decentralized actor and four set interactions. On-policy value heads and transformer Q readouts use two set interactions. Deep Set and transformer twins default to one action-conditioned critic encoder with private Q heads; `critic_independent_encoders=True` creates separate critic encoders. Actor and critic parameters remain separate in registered TMASAC/TMATD3 presets.

Ordinary TMASAC actors use D=192/256/256, FF `[384,384]` / `[512,512]` / `[768,768]`, and 2/2/3 blocks. Shared twin transformer critics use D=256/384/384 with FF=384/512/1536. The canonical LSTM actor uses D=128/192/256 and FF=512 in each FF stage. The sLSTM actor uses D=192/256/256 and FF=384/512/768 in each stage. Their feed-forward critics retain actor-state conditioning, projected to 192/256/384 features. Independent twin-critic layouts and custom Deep Set interaction controls remain available through policy kwargs.

## Common NOP core

| Component | 2.5M | 5M | 10M |
| --- | --- | --- | --- |
| Latent / transition width | 128 | 192 | 192 |
| Transition FF width | 512 | 384 | 768 |
| Transition blocks / heads | 2 / 4 | 2 / 4 | 3 / 4 |
| Source bridge hidden widths | `[256,256]` | `[256,256]` | `[384,384]` |
| Predictor hidden widths | `[192,192]` | `[192,192]` | `[256,256]` |

Only the source adapter input depends on the policy encoder width. MAT attaches NOP to the actor encoder; PPO/MAPPO attach it to their shared feature encoder; off-policy defaults attach it to critic features. `use_nop=False` preserves main-policy dimensions. Explicit actor-plus-critic NOP creates two auxiliary models rather than shrinking each to fit one allowance. NOP widths remain independent of prediction-target and action dimensions, while input/output matrices account for task-dependent count differences.

## Registry, counts and checkpoints

The default list contains 17 core presets. `list_variants(include_hidden=True)` also lists `mat_ind__no_attention`, `mat_qcx__no_nop` and `tmasac__no_nop`. MLP-suffixed, SwiGLU, shared-encoder, optional recurrent-control and segment presets are removed from the registry; their low-level implementation machinery remains. Plain PPO and MAPPO remain registered.

Names follow `base_variant__modifier__modifier`: major architecture names keep single underscores (`mat_qcx`, `tmasac_dec`), while additional modifiers use double underscores (`mat_qcx__lstm`, `matd3__deepset_critic`). Superseded spellings are rejected without aliases. This naming change preserves architecture, checkpoint weight keys and layout versions; see the [naming and migration guide](../learning.md).

Earlier MAT checkpoints need retraining or manual migration because encoder ownership, layer widths and recurrent-state structure changed. No automatic shared-to-independent migration or compatibility mode is provided. New checkpoints record resolved actor/critic configs and the layout version. Use matching task, distribution, NOP settings and version when reconstructing them.

See the [measured cross-scale counts](scale_comparison.md), [inspector commands](README.md), and [architecture diagrams](diagrams/README.md). Observation/action dimensions and padded agent slots can affect actual counts, particularly for flattened PPO; the hidden layouts stay fixed across tasks.

```bash
swarmbots-inspect-policies --model-scale 5M --output-dir docs/policy_parameters/generated/5M
swarmbots-diagram-policies --refresh
```
