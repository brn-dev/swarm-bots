# Fixed model-scale layouts

Three supported scales select explicit architecture layouts in [model_scale.py](../../swarmbots/learn/presets/model_scale.py):

| Main policy | Nominal NOP | Canonical name | Short name |
| ---: | ---: | --- | --- |
| 2.5M | 0.75M | `2.5M NOP0.75M` | `2.5M` or `2.5+0.75M` |
| 5M | 1M | `5M NOP1M` (default) | `5M` or `5+1M` |
| 10M | 2M | `10M NOP2M` | `10M` or `10+2M` |

There is no automatic parameter-count fitting, continuous width multiplier, or task-dependent size search. `list_model_scales()` lists the available names. Unregistered pairings are rejected; `model_scale=None` keeps the low-level builder's custom width arguments. The 5M layout is version `5+1M-v2`, which widens the QCX decoder block FFN to `192 → 256 → 192`.

The 10M layout is version `10+2M-v2` and grows depth as well as feed-forward capacity. Main attention/recurrent encoders and autoregressive decoders use **three blocks**; comparable FF-only actors use **four blocks**. Actor–critic shared variants use three shared blocks and two private blocks on each side. Regular transformer actors retain D=256 with FF `[768,768]`; shared twin-Q encoders use D=384 with a conventional fourfold FF expansion to 1536. This keeps the nominal separate actor/critic budgets at 4M/6M. Three blocks provide the depth increase within the reviewed parameter allowance.

The 2.5M tier preserves the 5M tier's 256-wide source bridge and 192-wide prediction MLPs. Its transition model uses D=128 with FF=512 and two layers: a larger expansion instead of reducing every auxiliary component. This gives about **0.65M–0.82M actual NOP parameters** on WallEasy, rather than the 0.5M allowance a linear reduction would suggest. The 10M tier uses a three-layer D=192, FF=768 transition model, bridge `[384,384]`, and prediction MLPs `[256,256]`, targeting about 2M. Every algorithm in a tier uses the same auxiliary core; source adapter input sizes account for the remaining count differences.

See the [cross-scale counts](scale_comparison.md) and [instructions for generating full audits and diagrams](README.md).


The 5M tier's nominal allocation remains:

| Layout | Shared encoder | Private actor | Combined private critic |
| --- | ---: | ---: | ---: |
| Separate encoders | 0M | 2M | 3M |
| Actor-critic shared encoder | 2M | 1M | 2M |

Half the shared encoder is attributed to each side, giving nominal 2M actor / 3M critic budgets. Attribution is accounting: both sides execute the full shared encoder, and its weights count once in the total. Counts include online weights and biases, attention, recurrence, embeddings, normalization and action/value heads. Frozen targets, buffers and optimizer state are excluded. These are reviewed starting configurations, not claims of optimal capacity or equal compute.

At 2.5M the targets are 1M actor / 1.5M critic, or 1M shared / 0.5M actor / 1M private critic. At 10M they are 4M actor / 6M critic, or 4M shared / 2M actor / 4M private critic.

## Tier dimensions

Each entry below is declared directly; no scale multiplier generates these widths or depths. Block/layer counts and architecture controls stay consistent within each comparison family in a tier.

| Component | 2.5M tier | 5M tier | 10M tier |
| --- | --- | --- | --- |
| Main attention/recurrent and decoder blocks | 2 | 2 | 3 |
| Decentralized FF-only actor blocks | 3 | 3 | 4 |
| Private blocks after an actor–critic shared encoder | 1 | 1 | 2 |
| Ordinary transformer D / FF hidden widths | 192 / `[384,384]` | 256 / `[512,512]` | 256 / `[768,768]` |
| PPO/MAPPO shared latent width | 192 | 256 | 384 |
| Dense LSTM D / FF width | 128 / 512 | 192 / 512 | 256 / 512 |
| Private shared-encoder actor D / FF widths | 192 / `[384,384]` | 256 / `[512,512]` | 256 / `[512,512]` |
| Off-policy actor head | 128 | 128 | 192 |
| MAT-Orig decoder D / FF width | 128 / 128 | 192 / 192 | 256 / 256 |
| QCX decoder D / FF width | 128 / 128 | 192 / 256 | 192 / 512 |
| Shared twin transformer critic D / FF | 256 / 384 | 384 / 512 | 384 / 1536 |
| Private critic after an actor–critic shared encoder D / FF | 256 / 384 | 384 / 512 | 384 / 1024 |
| Independent/recurrent critic D / FF | 192 / 384 | 256 / 512 | 256 / 1024 |
| Actor-state critic bridge | 192 | 256 | 384 |
| Head-only actor processing | `[512,384,384]` | `[768,512,512]` | `[1024,768,768]` |
| Shared V element/regressor MLPs | `[768,384]` | `[1024,512]` | `[1536,768]` |
| Joint single-Q hidden width (three layers per stage) | 512 | 768 | 1024 |
| Joint twin-Q hidden width (three layers per stage) | 384 | 512 | 768 |
| NOP transition blocks | 2 | 2 | 3 |
| NOP latent / transition D | 128 | 192 | 192 |
| NOP FF width | 512 | 384 | 768 |
| NOP source bridge hidden widths | `[256,256]` | `[256,256]` | `[384,384]` |
| NOP prediction MLP widths | `[192,192]` | `[192,192]` | `[256,256]` |

## Default 5M dimensions

All widths are powers of two or common intermediate sizes (128, 192, 256, 384, 512, 768, 1024, 1536). They do not change to absorb a small count discrepancy. The following table describes the default core variants; structural controls retain their settings while the fixed scale supplies dimensions.

| Family | Representation / actor | Critic |
| --- | --- | --- |
| PPO | Joint MLP shared backbone `[768,768,512,512]`, 256 features per agent; private actor `[768,512]` and latent 512 | Flattened V MLP `[1024,512]` |
| MAPPO | Per-agent shared backbone `[1024,768,768,512]`, latent 256; same private actor dimensions as PPO | Deep Set elements `[1024,512]`, pooled regressor `[1024,512]` |
| MAT-Orig / MAT-IND / MAT-QCX | Shared encoder D=256, two blocks, FF `[512,512]`; observation projections use D | IND private MLP `[768,512,512]`; Orig/QCX use a two-layer D=192 decoder. Shared IND/QCX V critics use the same Deep Set template as MAPPO. Orig retains token-wise V prediction with head `[1024,1024,512]` before mean pooling. |
| Recurrent MAT-IND / MAT-QCX | Shared LSTM encoder D=192, two blocks, FF 512 in each of its two FF stages; same private actor/decoder dimensions as feed-forward MAT | Same V-head template as feed-forward MAT; no automatic widening/narrowing when selecting the decoder |
| MAT-DEC | Separate actor D=256, three FF-only blocks with FF `[512,512]`, head 128; critic encoder D=256, two attention blocks, FF `[512,512]` | Smaller private Deep Set elements `[512,512]`, regressor `[768,512]`, because the critic also pays for its encoder |
| DDPG / TD3 / SAC with Deep Set critics | Common decentralized actor: D=256, three FF-only blocks, FF `[512,512]`, head 128 | Single Q: three 768-wide layers in each element/regressor stage. Twin Q: three 512-wide layers in each stage of each complete Q network. |
| TMASAC / TMATD3 | Transformer actor D=256, two attention/FF blocks, FF `[512,512]`, head 128 | Shared twin-Q encoder D=384, two blocks, FF 512; Q tails derive from critic width |
| Decentralized TMASAC / TMATD3 | Exactly the same three-block decentralized actor layout as the Deep Set algorithms | Exactly the same transformer critic as the corresponding attention actor variant |
| Recurrent TMASAC | sLSTM actor D=256; LSTM actor D=192; both two encoder blocks, FF 512 in each FF stage, head 128 | Same D=384 transformer critic as feed-forward TMASAC, plus a 256-wide actor-state bridge |
| Shared-encoder TMASAC | Shared D=256 encoder, two blocks, FF `[512,512]`; private actor D=256, one block, FF `[512,512]`, head 128 | Private critic D=384, one block, FF 512, twin Q tails |

Attention uses four heads throughout the fixed layouts. Original MAT and QCX decoders both use D=192 and two layers. QCX uses a `192 → 256 → 192` FFN in each decoder block, including the recurrent MAT-QCX variant. Original MAT retains a `192 → 192 → 192` decoder MLP.

## Why these comparisons are cleaner

- Decentralized actors retain the same observation processing, widths, FF topology and head across DDPG, TD3, SAC and decentralized transformer-critic variants. Only the algorithm, critic and action distribution change.
- Decentralized actors use one extra FF-only block at the same width and FF topology as the attention actors: two/three blocks in the smaller tiers, three/four at 10M. At 10M, attention actors have about 3.95M parameters and FF-only actors about 4.15M. The additional MLP block compensates for attention capacity with an ordinary processing block.
- MAT-IND and MAT-QCX share their encoder and V-critic layouts; recurrent versions share those private head/decoder layouts as well. Original MAT retains its own value-pooling semantics.
- LSTM uses D=192 because dense recurrent gates cost more than the sLSTM recurrence. Both retain FF=512 and head=128. Recurrent TMASAC's critic keeps its normal backbone instead of shrinking to pay for actor-state inputs.
- Switching twin transformer critics to independent encoders leaves the actor untouched. At 5M, the independent/recurrent critic layout uses D=256, two blocks, FF=512; at 10M it uses D=256, three blocks, FF=1024. Counts for optional structural combinations remain visible rather than triggering a new fit.
- At 10M, three shared blocks and two smaller private blocks put the shared-encoder TMASAC counts at about 3.90M shared / 1.83M actor / 4.05M critic. Attributing half the shared encoder to each side gives about 3.78M actor / 6.00M critic. QCX retains D=192 while gaining a third decoder block and FF=512, so its more involved blocks stay close to the private actor allowance; original MAT uses three D=256 blocks. PPO/MAPPO keep their existing 10M MLP layouts and 384-wide shared latents.

## One common NOP architecture

All default 5M-tier NOP models use latent and transition widths of **192**, **two transformer layers**, **four heads**, FF **384**, coembedding hidden width **192**, a source projection bridge **256 -> 256 -> 192**, and predictor processing **192 -> 192**. Only the bridge's input dimension depends on the actor/critic/shared source. All other auxiliary dimensions stay identical across algorithms and source widths.

`use_nop=False` leaves main-policy dimensions unchanged. Off-policy MLP critics omit NOP. Explicit actor-plus-critic NOP creates two copies of this common auxiliary layout; it is an optional control with a larger total auxiliary count, rather than silently halving both models.

## Counts and reproduction

The [current comparison](scale_comparison.md) audits the 18 core presets on WallEasy. Full reports and `layer_layout.csv` can be [generated locally](README.md) to expose every actual module's dimensions and count. Observation/action sizes and padded agent counts can change input/output matrix sizes, especially for flattened PPO/MLP networks. The declared hidden widths remain fixed across tasks. Actual totals are expected to differ moderately from the nominal 5M + 1M.

```python
from swarmbots.learn import make_training
trainer = make_training("SwarmBots-WallEasy-v0", "tmasac", model_scale="5+1M")
```

```bash
swarmbots-inspect-policies --model-scale "5+1M" --output-dir docs/policy_parameters/generated/5M
```

Only 18 core variants appear in the default list/sweep. MLP baselines, recurrent shared encoders, SwiGLU and other controls remain explicitly selectable; `list_variants(include_hidden=True)` lists them and the inspector's `--include-hidden` includes them. PPO/MAPPO `_small` names are removed. For custom dimensions use `model_scale=None` (CLI `legacy`); fixed-scale dimensions take precedence over low-level width arguments. Checkpoints record the selected tier, its layout version (`2.5+0.75M-v1`, `5+1M-v2`, or `10+2M-v2`), and resolved configurations. Reconstruct them with matching task, distribution, structural controls and layout version.

## Architecture diagrams

The [diagram gallery and reading guide](diagrams/README.md) cover all 18 core variants at all three scales. [Open the offline selector](diagrams/index.html), or use the individual SVG/Mermaid/Graphviz files. Regenerate from current presets with `swarmbots-diagram-policies --refresh`.
