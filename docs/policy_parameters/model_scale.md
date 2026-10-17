# Fixed model-scale layouts

The supported scales select explicit architecture layouts in [model_scale.py](../../swarmbots/learn/presets/model_scale.py). There is no runtime parameter fitting or task-dependent hidden-width search.

| Canonical name | Main budget | NOP budget | Current version |
| --- | ---: | ---: | --- |
| `2.5M NOP0.75M` | 2.5M | 0.75M | `2.5+0.75M-v1` |
| `5M-2L NOP1M` (default) | 5M | 1M | `5M-2L+1M-v1` |
| `5M-3L NOP1M` | 5M | 1M | `5M-3L+1M-v1` |
| `10M-3L NOP2M` | 10M | 2M | `10M-3L+2M-v1` |
| `10M-4L NOP2M` | 10M | 2M | `10M-4L+2M-v1` |
| `10M-5L NOP2M` | 10M | 2M | `10M-5L+2M-v1` |

Short names `2.5M`, `5M-2L`, `5M-3L`, `10M-3L`, `10M-4L`, and `10M-5L` select these layouts. `10M-3L` names the three-block 10M reference. `model_scale=None` (CLI `--model-scale custom`) enables explicit-width builder arguments. Fixed-scale dimensions take precedence over low-level width arguments.

The main budgets describe feed-forward reference architectures. DDPG uses one ordinary Q head and therefore has fewer parameters than the twin-head reference. Recurrent variants preserve their corresponding feed-forward backbones and add temporal modules, normalization and, for recurrent off-policy actors, critic state conditioning. The deeper 5M and 10M temporal modules are narrower to keep complete recurrent counts comparable within each budget. Gated variants add reception gates without shrinking other components.

## Five-million encoder-depth comparison

`5M-2L` is the two-block 5M reference. `5M-3L` adds a third block only to attention and action-conditioned Deep Set encoder stacks, with narrower feed-forward layers or tokens to retain comparable feed-forward capacity. Use the inspector to measure the resulting main-policy counts for the task being compared.

| Encoder | 5M-2L | 5M-3L |
| --- | --- | --- |
| Attention actor D / FF | 256 / `[512,512]`, 2 blocks | 256 / `[320,320]`, 3 blocks |
| Orig / QCX actor encoder D / FF | 192 / `[384,384]`, 2 blocks | 192 / `[256,256]`, 3 blocks |
| MAT critic encoder D / FF | 256 / `[512,512]`, 2 blocks | 256 / `[320,320]`, 3 blocks |
| Off-policy shared critic encoder D / FF | 384 / `[512]`, 2 blocks | 320 / `[384]`, 3 blocks |

FF-only actors keep three blocks with D=256 and FF `[512,512]`. Decoder stacks keep two blocks and their original widths. Q/V readout templates, their two Deep Set interactions, prediction heads, and the NOP core keep their architecture. The NOP source adapter follows its source encoder's width, so its parameter count can change slightly. PPO/MAPPO have no affected encoder stack and are identical across these choices. Gates remain additional capacity.

Recurrent main counts still exceed the feed-forward 5M reference; the 3L memory widths are chosen to keep the complete recurrent totals close to their 2L counterparts. The 3L layout uses memory widths 192 for MAT-IND actor/critic, 136 for MAT-QCX actor and 192 for its critic, 192 for TMASAC LSTM, and 168 for TMASAC sLSTM. Bias-free input/output projections connect each memory module to the unchanged encoder token width. The detached actor-state critic adapter takes the actual memory width; its output remains 256. The 2.5M, 5M-2L, and 10M-3L layouts retain memory width equal to token width.

Generate reports for both choices with the [inspector commands](README.md) to compare their current measured counts. The tier tables below describe the 2L five-million reference; this section lists the 3L differences.

## Ten-million encoder-depth comparison

`10M-3L` is the three-block reference layout. `10M-4L` and `10M-5L` increase only attention/recurrent and action-conditioned Deep Set encoder depth. Their narrower FF and memory modules keep the registered core presets' main counts within 1% of the 3L reference on WallEasy. Gated controls add capacity and remain within 3% across these depths.

| Encoder | 10M-3L | 10M-4L | 10M-5L |
| --- | --- | --- | --- |
| Attention actor / MAT critic D | 256 | 256 | 256 |
| Attention actor / MAT critic FF | `[768,768]` | `[608,608]` | `[480,480]` |
| Orig / QCX actor encoder D | 192 | 192 | 192 |
| Orig / QCX actor encoder FF | `[512,512]` | `[400,400]` | `[320,320]` |
| Off-policy shared critic encoder D / FF | 384 / `[1536]` | 384 / `[960]` | 384 / `[608]` |
| Encoder blocks | 3 | 4 | 5 |
| MAT-IND actor/critic LSTM memory | 256 | 204 | 184 |
| MAT-QCX actor LSTM memory | 192 | 144 | 128 |
| MAT-QCX critic LSTM memory | 256 | 204 | 184 |
| TMASAC LSTM actor memory | 256 | 192 | 176 |
| TMASAC sLSTM actor memory | 256 | 144 | 136 |

FF-only actors retain four blocks at D=256 and FF `[768,768]`. Decoder stacks retain three blocks and their original widths. Q/V readouts, their private Deep Set interactions, and the three-block 2M NOP core keep their architecture. PPO/MAPPO have no affected encoder stack and retain identical configurations across the three choices. Low-level shared-encoder policies change the shared stack's depth and FF widths while retaining two private blocks per side.

All hidden widths remain fixed declarations, independent of the task. The dimension tables below describe the 10M-3L reference; this section lists the deeper alternatives.

## Actor and critic ownership

MAT-Orig, MAT-IND, MAT-DEC, MAT-QCX and both recurrent MAT variants always have separate `actor_encoder` and `critic_encoder` modules. Config `encoder_config` describes the actor; `critic_encoder_config` describes the critic. A missing critic config uses the actor's architecture as a template, creating a separate module with independent weights. Privileged observations enter only the value head; they never enter actor encoding or decoding. MAT-DEC disables actor attention while retaining an attention critic.

Recurrent MAT uses independent recurrent encoders and a temporal-state mapping with `actor` and `critic` histories. Rollout advances both, actor-only inference advances the actor history, and value-only bootstrap advances the critic history. Reset masks apply to each history when it is advanced. Sequence training uses both saved initial histories, episode reset masks and time masks. NOP consumes critic encoder features before the value head, reusing the same forward pass. Its gradients update the critic encoder and leave actor parameters and private value-head parameters untouched. Recurrent NOP follows the critic's temporal state.

The nominal complete-actor/complete-critic budgets are 1M/1.5M, 2M/3M and 4M/6M. MAT no longer uses half-shared attribution. MAPPO also uses independent actor and critic MLP encoders, with the same complete-actor/complete-critic budget targets. Plain PPO retains its shared MLP backbone: nominal shared/private-actor/private-critic budgets are 1M/0.5M/1M, 2M/1M/2M and 4M/2M/4M, with half the backbone attributed to each side for accounting. Both sides still execute the full backbone.

## MAT tier dimensions

All attention encoders and decoders use four heads. Counts include weights, biases, attention, recurrence, normalization and output heads; frozen targets, buffers, optimizer state and NOP are excluded from main counts.

| Component | 2.5M | 5M-2L | 10M-3L |
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
| Recurrent IND actor D / FF | 192 / `[384,384]` | 256 / `[512,512]` | 256 / `[768,768]` |
| Recurrent QCX actor D / FF | 128 / `[256,256]` | 192 / `[384,384]` | 192 / `[512,512]` |
| Recurrent MAT critic D / FF | 192 / `[384,384]` | 256 / `[512,512]` | 256 / `[768,768]` |

The IND actor pays for its encoder and only a small action head. Autoregressive actors reserve more of their budget for the decoder and therefore use a smaller encoder. All MAT Deep Set value heads account for a private critic encoder; recurrent critics keep their own history and the same V-readout template. Original MAT retains token-wise prediction and masked pooling. These layouts are starting points; establishing the best width/depth trade-off requires training experiments.

## Other families

Plain PPO retains its fixed shared MLP layout. MAPPO uses independent per-agent MLP encoders with feature widths 192/256/384:

| MAPPO component | 2.5M | 5M-2L / 5M-3L | 10M-3L / 10M-4L / 10M-5L |
| --- | --- | --- | --- |
| Actor encoder hidden widths | `[512,384,384]` | `[768,512,512,384]` | `[1024,768,768,512]` |
| Critic encoder hidden widths | `[512,512,384,384]` | `[1024,768,512]` | `[1024,1024,768]` |
| Encoder feature width | 192 | 256 | 384 |

Both encoders use public local/global observations and have independent weights. Privileged local/global variables enter only the value head, preserving the actor boundary. MAPPO NOP uses the critic encoder's per-agent features before the value head. The actor encoder is shared across agents, not between actor and critic.

Off-policy Deep Set DDPG/TD3/SAC use the common decentralized actor. Their critic embeddings, token width, block count, normalization, residuals and local feed-forward networks match the transformer critic. Only the shared encoder's communication operator differs. Actor and critic parameters remain separate.

For low-level Deep Set policies, explicit `joint_critic_config.encoder_config` and `head_config` supply the structural settings used during scaling. Fixed layouts set their standard widths and depths while preserving embedding mode, token normalization, separate observation/action encoding, interaction controls and initialization gains. Missing nested configs use the top-level critic templates. Top-level `baseline_critic_element_hidden_dims` and `baseline_critic_hidden_dims` builder arguments apply only without a fixed scale.

At the 2.5M, 5M-2L, and 10M-3L reference layouts, ordinary TMASAC actors use D=192/256/256, FF `[384,384]` / `[512,512]` / `[768,768]`, and 2/2/3 blocks. Both shared critic families use D=256/384/384, FF=384/512/1536, and 2/2/3 blocks. Every off-policy Q head projects tokens to 128/128/192 features, applies two Deep Set interaction blocks, pools with a masked mean, and uses two regression hidden layers of that same width. DDPG uses one such head; TD3/SAC use two.

LSTM and sLSTM actors retain their feed-forward baseline dimensions and add recurrence at the same positions, with temporal residuals and layer normalization. They do not split or duplicate the existing FF network. Their ordinary critics retain actor-state conditioning, projected to 192/256/384 features. The low-level independent-encoder option remains available without resizing either encoder; it is not a registered variant.

## Expanding and gated Deep Set interactions

A token has width D; the pooled message has width M=5D/4. The global branch is pre-norm, `D -> M -> activation -> masked mean -> M -> activation -> D`, followed by residual addition. The final projection is linear and bias-free. A separately normalized local FF residual follows, matching the transformer block. For shared critic encoders, the two families execute the same local FF module implementation.

The expanding communication branch has approximately `2DM + M² = 4.0625D²` weights, compared with attention's `4D²`; biases and normalization are excluded from this approximation. This controls parameter allocation, not expressive power or runtime.

The optional `maddpg__gated_deepset_critic`, `matd3__gated_deepset_critic`, and `masac__gated_deepset_critic` presets retain the standard dimensions and add feature-wise reception gates only in the critic encoder. Their private Q heads remain ordinary Deep Sets with the same architecture as the corresponding ungated presets. Each gate receives the normalized local token and processed global message, uses a D/4-wide hidden layer and sigmoid outputs, and selects M-dimensional message features before compression. Gate parameters are additional capacity; no other component is shrunk. Pooling remains shared, so gating does not recover receiver-specific sender selection.

`deepset_interaction_config` exposes `message_dim`, local `hidden_dims`, `gated`, and `gate_hidden_dim`. For off-policy Deep Set presets, `gated` controls the encoder. The separate `critic_deepset_head_gating` builder option defaults to `False` and controls reception gates in the off-policy Q heads independently. `policy_kwargs={"critic_deepset_head_gating": True}` enables head gating for ablations, with or without a gated encoder. Fixed-scale configs record the resolved encoder and head settings, including message and gate dimensions. Explicit low-level configurations can intentionally depart from these defaults.

## Common NOP core

| Component | 2.5M | 5M-2L / 5M-3L | 10M-3L / 10M-4L / 10M-5L |
| --- | --- | --- | --- |
| Latent / transition width | 128 | 192 | 192 |
| Transition FF width | 512 | 384 | 768 |
| Transition blocks / heads | 2 / 4 | 2 / 4 | 3 / 4 |
| Source bridge hidden widths | `[256,256]` | `[256,256]` | `[384,384]` |
| Predictor hidden widths | `[192,192]` | `[192,192]` | `[256,256]` |

Only the source adapter input depends on the policy encoder width. MAT/RMAT and MAPPO attach NOP to critic encoders; plain PPO attaches it to its shared feature encoder; off-policy defaults attach it to critic features. `use_nop=False` preserves main-policy dimensions. Explicit actor-plus-critic NOP creates two auxiliary models rather than shrinking each to fit one allowance. NOP widths remain independent of prediction-target and action dimensions, while input/output matrices account for task-dependent count differences.

## Registry, counts and checkpoints

The default list contains 17 core presets. `list_variants(include_hidden=True)` also lists two NOP controls (`mat_qcx__no_nop`, `tmasac__no_nop`) and the three gated Deep Set presets. `mat_ind__no_attention` is removed. MLP-suffixed, SwiGLU, shared-encoder, optional recurrent-control and segment presets remain unregistered; their low-level implementation machinery remains. Plain PPO and MAPPO remain registered.

Names follow `base_variant__modifier__modifier`: major architecture names keep single underscores (`mat_qcx`, `tmasac_dec`), while additional modifiers use double underscores (`mat_qcx__lstm`, `matd3__deepset_critic`). Superseded spellings are rejected without aliases. This naming change preserves architecture, checkpoint weight keys and layout versions. The [available-variant reference](../learning.md#select-a-variant-and-task) lists current names; the [changelog](../../CHANGELOG.md) records historical changes.

Checkpoints record resolved actor/critic configs and the layout version. Loading requires the current checkpoint format, task, distribution, NOP settings, architecture and observation layout.

Generate current counts using the [inspector commands](README.md) or [architecture diagrams](diagrams/README.md). Observation/action dimensions and padded agent slots can affect actual counts, particularly for flattened PPO; the hidden layouts stay fixed across tasks.

```bash
swarmbots-inspect-policies --model-scale 5M-2L --output-dir docs/policy_parameters/generated/5M-2L
swarmbots-diagram-policies
```

## Experimental comparisons

TMASAC/TMATD3 use centralized attention actors. Compare their decentralized `*_dec` controls against MASAC/MATD3 to isolate critic communication; comparing the primary centralized variants also changes actor communication. The detached actor-state critic bridge adds both capacity and input information, so canonical recurrent off-policy presets are not actor-only recurrence ablations.

Parameter matching does not match compute or replay exposure. Recurrent replay counts sequences; feed-forward replay counts transitions. Configure their learning steps and batches explicitly when comparing them. Use independent training seeds for uncertainty estimates; evaluation seeds are not training replicas. Record resolved hyperparameters, model-scale versions and the Git commit with each experiment, alongside any generated count reports.
