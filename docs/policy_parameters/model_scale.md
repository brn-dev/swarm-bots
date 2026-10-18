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

The main budgets describe feed-forward reference architectures. DDPG uses one ordinary Q head and therefore has fewer parameters than the twin-head reference. Recurrent MAT preserves its feed-forward backbone and adds memory. Recurrent TMASAC splits the actor's MLP budget around memory, adding temporal modules, normalization and critic state conditioning. The deeper 5M and 10M temporal modules are narrower to keep complete recurrent counts comparable within each budget. Gated variants add reception gates without shrinking other components.

## Comparability limits

The layouts support comparisons with matched components; a common scale name does not make every complete policy parameter matched. The following distinctions apply before interpreting a performance difference:

| Comparison | Matched components | Remaining difference |
| --- | --- | --- |
| `mat_qcx` / `tmasac` versus their `__no_nop` controls | Complete main-policy architecture and parameter count | NOP adds auxiliary parameters and gradients to its source encoder |
| `tmasac_dec` versus `masac__deepset_critic`; `tmatd3_dec` versus `matd3__deepset_critic` | Decentralized actor, critic token width/depth, embeddings, local FF MLPs and private Q heads | Communication operator; expanding Deep Set communication has slightly more parameters than attention |
| TD3 versus SAC with the same actor/critic family | Encoder backbones and twin Q heads | Action distribution and learning algorithm |
| DDPG versus TD3/SAC Deep Set critics | Actor backbone, shared critic encoder and each individual Q-head architecture | DDPG has one Q head; twin-head methods have two |
| Recurrent MAT versus its feed-forward counterpart | Actor/critic non-temporal backbones, decoder and readouts | Memory, temporal projections and normalization add main parameters on both sides |
| Recurrent TMASAC versus `tmasac` | Actor token/attention layout, combined actor FF budget, critic block stack and Q heads | Actor memory, temporal projections/normalization and detached critic-state conditioning add main parameters |
| TMASAC LSTM versus sLSTM | Non-temporal architecture and complete main counts within 0.5% on WallEasy at fixed scales | Narrower dense LSTM memory versus wider four-head sLSTM memory; temporal projections and critic-state adapter inputs follow the memory width |
| MAT-IND versus MAT-QCX | Identical actor/critic observation encoders, matched head budgets within 1%, equal encoder-plus-head stage counts | Independent MLP action head versus autoregressive attention decoder; action conditioning and compute differ |
| MAT-IND versus MAT-DEC | Critic architecture, actor stage counts and matched encoder/head budgets | Removing actor attention is balanced by wider local FF layers |
| MAT-Orig versus MAT-QCX | Identical actor/critic observation encoders and matched actor-head and critic-role budgets | Decoder architecture and value readout differ; this compares complete policies |
| Gated versus ordinary Deep Set critics | Actor, critic backbone and Q heads | Reception gates add critic parameters; the increment grows with encoder depth |
| Depth choices within one variant | Approximately matched complete main counts, actor/critic allocation and NOP core | On-policy actor depth is split between encoder and decoder/head; widths trade against depth |
| MAPPO Deep Set versus MAT critics | Critic observation embeddings, width/depth/local FF layout and V readout; privileged inputs enter before encoding | Deep Set communication replaces attention; the actor remains a local MLP |
| PPO versus MAPPO/MAT/off-policy families | Approximate complete actor/critic allocation and common NOP core | Shared backbone, flattened inputs and late privileged-input placement |

The default sLSTM uses headwise input and recurrent projections, while the LSTM uses dense matrices. Fixed TMASAC layouts use narrower LSTM memory to match the complete main count, including temporal projections and critic-state conditioning. This compares parameter budgets while retaining different state widths and recurrent mixing. Explicit custom widths need not match. Recurrent and gated variants still add capacity to their feed-forward references; shrinking those backbones to fit the feed-forward total would change the component being controlled.

Actor/critic totals and their internal allocation matter alongside the complete count. Feed-forward MAT and MAPPO actors target the same 1M/2M/4M actor budgets as the off-policy methods, with approximately half assigned to encoding and half to action generation. IND, QCX and Original MAT use identical encoders within a tier; MAPPO and MAT-DEC match the component budgets with different observation-processing architectures. Original MAT retains a token-wise value MLP, other MAT variants use pooled Deep Set readouts, and off-policy methods use one or two private Q heads. PPO's half-shared attribution remains an accounting convention; actor inference executes the full shared backbone.

Counts describe stored parameters, not training compute, data exposure, memory usage or effective capacity. Recurrent unrolls, autoregressive decoding, repeated NOP transitions and attention reuse parameters. Comparisons use the same task, observation/action shapes and structural overrides, with main, NOP and frozen-target counts reported separately. The inspector's role and component tables expose the actual allocations rather than treating every tier as an exact budget.

## Five-million encoder-depth comparison

`5M-2L` is the two-stage 5M reference. Its actor features are 320-wide across MAT, MAPPO and off-policy families, compared with 256 at `5M-3L`. The wider shallow layout shifts capacity from FF hidden layers into attention and token features while retaining comparable parameter allocations. MAT/MAPPO actors split the stages between encoder and decoder/head; off-policy attention actors retain their encoder stack and small output head.

On WallEasy, ordinary feed-forward presets have 5.0–5.35M main parameters, excluding NOP and frozen targets. Each registered variant's 2L and 3L main counts stay within 1% of one another, including recurrent and gated variants. The layouts retain version `v1` as the baseline for fresh runs. Counts remain task-dependent, particularly for flattened PPO inputs.

| Encoder | 5M-2L | 5M-3L |
| --- | --- | --- |
| Off-policy attention actor D / FF | 320 / `[424,424]`, 2 blocks | 256 / `[384,384]`, 3 blocks |
| IND / QCX / Orig actor encoder D / FF | 320 / `[360,360]`, 1 block | 256 / `[216,216]`, 2 blocks |
| Off-policy FF-only actor D / FF | 320 / `[504,504]`, 3 blocks | 256 / `[560,560]`, 3 blocks |
| MAT/MAPPO encoder + decoder/head stages | 1 + 1 | 2 + 1 |
| MAT / MAPPO / off-policy critic encoder D / FF | 384 / `[784]`, 2 blocks | 320 / `[656]`, 3 blocks |

MAT action heads retain one stage at both 5M depths, with hidden widths 720/768 to account for the different feature widths. MAPPO uses one or two three-linear-layer encoder stages and the same MLP action head as IND. Plain PPO retains its shared backbone. Q/V readout structures and the NOP core retain their roles; widths maintain their budgets. The NOP source adapter follows its source encoder's width, so its parameter count can change slightly. Gates remain additional capacity.

On WallEasy, attention accounts for approximately 41–42% of the 2L MAT and off-policy actor encoders, up from approximately 26–27% at the former 256-wide layout. Complete feed-forward actors remain approximately 2M parameters; MAT/MAPPO retain approximately 1M each for encoding and action generation.

Recurrent MAT preserves each corresponding feed-forward actor/critic backbone and adds memory. IND and QCX use the same actor memory width: 232 at 2L and 160 at 3L. Critic memory widths remain 256/212. TMASAC uses LSTM widths 176/144 and sLSTM widths 280/232. Bias-free projections connect memory and token widths when they differ. Complete recurrent counts remain comparable across the two depths.

Generate reports for both choices with the [inspector commands](README.md) to compare their current measured counts. The actor allocation table below includes both depths.

## Ten-million encoder-depth comparison

`10M-3L` is the three-stage reference layout. The 4L and 5L choices increase combined actor depth and critic encoder depth. Their fixed widths keep registered core presets' main counts within 1% of the 3L reference on WallEasy. Gated controls add capacity and remain within 3% across these depths.

Ordinary feed-forward presets have 10.0–10.65M main parameters on WallEasy, excluding NOP and frozen targets. MAT and off-policy critics use the same observation-encoder widths; Original MAT and MAPPO have matched critic-role budgets while retaining different readout/encoder structures. DDPG retains one standard Q head. All layout versions remain `v1` for fresh runs.

| Encoder | 10M-3L | 10M-4L | 10M-5L |
| --- | --- | --- | --- |
| Attention actor D | 256 | 256 | 256 |
| Off-policy attention actor FF | `[776,776]` | `[608,608]` | `[488,488]` |
| IND / QCX / Orig actor encoder D | 256 | 256 | 256 |
| IND / QCX / Orig actor encoder FF | `[592,592]` | `[592,592]` | `[392,392]` |
| MAT/MAPPO encoder + decoder/head stages | 2 + 1 | 2 + 2 | 3 + 2 |
| MAT / MAPPO / off-policy critic encoder D / FF | 384 / `[1416]` | 384 / `[864]` | 384 / `[536]` |
| Critic encoder blocks | 3 | 4 | 5 |
| MAT-IND / MAT-QCX actor LSTM memory | 256 | 232 | 184 |
| MAT-IND / MAT-QCX critic LSTM memory | 256 | 220 | 196 |
| TMASAC LSTM actor memory | 176 | 152 | 140 |
| TMASAC sLSTM actor memory | 288 | 248 | 224 |

Off-policy FF-only actors retain four blocks at D=256 and FF `[752,752]`. MAT decoders and independent heads use one stage at 3L and two at 4L/5L. MAPPO follows the same stage allocation with local MLPs. Plain PPO, the Q-head template and the three-block 2M NOP core retain their architecture. Low-level shared-encoder policies retain two private blocks per side.

All hidden widths remain fixed declarations, independent of the task. The actor allocation table below includes every depth.

## Actor and critic ownership

MAT-Orig, MAT-IND, MAT-DEC, MAT-QCX and both recurrent MAT variants always have separate `actor_encoder` and `critic_encoder` modules. Config `encoder_config` describes the actor; `critic_encoder_config` describes the critic. A missing critic config uses the actor's architecture as a template, with a one-hidden-layer observation embedding and independent weights. MAT-DEC disables actor attention while retaining an attention critic.

MAT critic encoders use the off-policy observation layout: local public and privileged observations are concatenated before the local embedding MLP, and global public and privileged observations are concatenated before the global embedding. Separate global embeddings are added to local tokens; joint embedding concatenates the global inputs before local projection. Fixed scales use the same critic token width, attention depth, normalization, residuals and one-hidden-layer block FF as the off-policy critics. MAT values depend on state; off-policy Q critics additionally coembed actions. Privileged inputs enter only critic encoding. Value heads receive encoded features without appending the privileged inputs again, and actor encoding/decoding remains public-only.

Recurrent MAT uses independent recurrent encoders and a temporal-state mapping with `actor` and `critic` histories. Rollout advances both, actor-only inference advances the actor history, and value-only bootstrap advances the critic history. Reset masks apply to each history when it is advanced. Sequence training uses both saved initial histories, episode reset masks and time masks. NOP consumes critic encoder features before the value head, reusing the same forward pass. Its gradients update the critic encoder and leave actor parameters and private value-head parameters untouched. Recurrent NOP follows the critic's temporal state.

The nominal complete-actor/complete-critic budgets are 1M/1.5M, 2M/3M and 4M/6M. MAT no longer uses half-shared attribution. MAPPO uses an independent MLP actor encoder and Deep Set critic encoder, with the same complete-actor/complete-critic budget targets. Plain PPO retains its shared MLP backbone: nominal shared/private-actor/private-critic budgets are 1M/0.5M/1M, 2M/1M/2M and 4M/2M/4M, with half the backbone attributed to each side for accounting. Both sides still execute the full backbone.

## MAT actor allocations

Feed-forward MAT-IND, MAT-QCX, MAT-Orig, MAT-DEC and MAPPO target 1M/2M/4M complete actor parameters, comparable to off-policy actors. Approximately half goes to encoding and half to action generation. IND, QCX and Original MAT use identical observation encoders. Counts include action embeddings, input/output bridges and distribution projections; matching only the decoder block would miss these parameters.

| Component | 2.5M | 5M-2L | 5M-3L | 10M-3L | 10M-4L | 10M-5L |
| --- | --- | --- | --- | --- | --- | --- |
| Actor encoder stages | 1 | 1 | 2 | 2 | 2 | 3 |
| Decoder/head stages | 1 | 1 | 1 | 1 | 2 | 2 |
| IND / QCX / Orig encoder D | 192 | 320 | 256 | 256 | 256 | 256 |
| IND / QCX / Orig encoder FF, two hidden layers | 352 | 360 | 216 | 592 | 592 | 392 |
| MAT-DEC encoder FF, two hidden layers | 472 | 616 | 440 | 736 | 736 | 568 |
| IND / DEC / MAPPO head hidden width H | 536 | 720 | 768 | 1176 | 776 | 768 |
| Orig / QCX decoder D | 160 | 224 | 224 | 320 | 256 | 256 |
| QCX decoder FF hidden width | 344 | 408 | 544 | 976 | 408 | 376 |
| Original MAT decoder FF hidden width | 616 | 840 | 912 | 1392 | 792 | 760 |
| IND / QCX actor LSTM memory | 192 | 232 | 160 | 256 | 232 | 184 |

Each IND/DEC/MAPPO head stage is an independent `D → H → H → D` MLP chain. QCX projects decoder output back to the same actor feature width D; Original MAT uses the same final feature width. Complete action-head counts match within 1% on WallEasy. All attention modules use four heads. Original MAT exposes `dim_feedforward` separately from decoder token width, retaining its original attention ordering and token-wise value readout.

The stage count controls broad processing depth. Attention stages contain extra projections, residuals and normalization, so equal stage counts do not mean identical affine-operation counts, longest computational paths or FLOP. MAT-DEC uses wider local FF layers to compensate for removed attention, while retaining the same encoder/head stage allocation and critic as IND. Recurrent IND/QCX add memory to their corresponding feed-forward encoders without resizing those backbones.

## Other families

Plain PPO retains its shared MLP backbone and half-shared accounting. MAPPO has independent per-agent actor/critic encoders. Its actor encoder uses three linear transformations per stage, `input → E → E → D`, with LayerNorm after each learned-feature activation; the first stage consumes public observations, and later stages consume D-wide features. This keeps feature magnitudes comparable to MAT's normalized encoder outputs through the deeper MLP chain. It uses the exact same MLP head dimensions as IND.

| MAPPO component | 2.5M | 5M-2L | 5M-3L | 10M-3L | 10M-4L | 10M-5L |
| --- | --- | --- | --- | --- | --- | --- |
| Actor encoder E | 584 | 820 | 528 | 808 | 808 | 624 |
| Actor feature D | 192 | 320 | 256 | 256 | 256 | 256 |
| Critic token D | 256 | 384 | 320 | 384 | 384 | 384 |
| Critic block FF hidden width | 384 | 784 | 656 | 1416 | 864 | 536 |
| Critic Deep Set blocks | 2 | 2 | 3 | 3 | 4 | 5 |

`mappo__deepset_critic` retains its local MLP actor. Its critic concatenates public and privileged local observations before local embedding, and public and privileged global observations before global embedding. The encoder uses the same expanding Deep Set communication blocks and local FF/residual/normalization layout as the off-policy Deep Set critics, without action inputs. Its value head matches MAT's pooled Deep Set readout and receives encoded features only. NOP consumes those same critic features from the evaluation pass; value and NOP losses leave the actor unchanged.

MAPPO and IND use gain 1.0 in the MLP head and gain 0.01 at the action-distribution projection. MAPPO records `action_net_init_gain` separately, placing the small gain after feature activation. MAPPO critic-role counts match MAT within 1% on WallEasy; expanding Deep Set communication adds slightly more parameters than attention. Original MAT retains the tighter 0.5% critic-role match. The canonical preset replaces the former `mappo` name without aliases or checkpoint migrations; layout versions remain `v1` for fresh runs.

Off-policy Deep Set DDPG/TD3/SAC use the common decentralized actor. Critic embeddings, token widths, depth, local FF modules and private Q heads match their attention-critic counterparts. Communication is the controlled architectural difference; expanding Deep Set communication has slightly more parameters than attention. Every Q head projects to 128/128/192 features at the 2.5M/5M/10M tiers, uses two Deep Set interaction blocks, masked mean pooling and two equally wide regression layers. DDPG retains one head; TD3/SAC use two.

For low-level Deep Set policies, explicit nested encoder/head configs supply structural settings during scaling. Fixed layouts set standard widths/depths while preserving embedding mode, token normalization, observation/action encoding, interaction controls and initialization gains. Top-level baseline critic widths apply only without a fixed scale.

Each recurrent TMASAC actor block executes `attention → MLP → recurrence → MLP`, with residuals and normalization. Each MLP has one hidden layer (`D → H → D`), replacing the ordinary block's two-hidden-layer MLP (`D → F → F → D`). Each small MLP has approximately half the ordinary FF parameters, within 1% including biases. The two stages have independent weights. Complete LSTM/sLSTM main counts match within 0.5% on WallEasy.

| Layout | Actor token D | Ordinary FF hidden widths | Each recurrent FF hidden width H | LSTM memory | sLSTM memory |
| --- | ---: | --- | ---: | ---: | ---: |
| 2.5M | 192 | `[384,384]` | 384 | 132 | 216 |
| 5M-2L | 320 | `[424,424]` | 352 | 176 | 280 |
| 5M-3L | 256 | `[384,384]` | 336 | 144 | 232 |
| 10M-3L | 256 | `[776,776]` | 976 | 176 | 288 |
| 10M-4L | 256 | `[608,608]` | 664 | 152 | 248 |
| 10M-5L | 256 | `[488,488]` | 480 | 140 | 224 |

Without fixed scales, TMASAC retains two independent `[512]` actor MLPs around memory and the ordinary `[512,512]` critic FF. Actor-state conditioning projects to 192/256/384 features at fixed tiers. Low-level independent critic encoders remain available without resizing either encoder.

## Expanding and gated Deep Set interactions

A token has width D; the pooled message has width M=5D/4. The global branch is pre-norm, `D -> M -> activation -> masked mean -> M -> activation -> D`, followed by residual addition. The final projection is linear and bias-free. A separately normalized local FF residual follows, matching the transformer block. For shared critic encoders, the two families execute the same local FF module implementation.

The expanding communication branch has approximately `2DM + M² = 4.0625D²` weights, compared with attention's `4D²`; biases and normalization are excluded from this approximation. This controls parameter allocation, not expressive power or runtime.

The optional `maddpg__gated_deepset_critic`, `matd3__gated_deepset_critic`, and `masac__gated_deepset_critic` presets retain the standard dimensions and add feature-wise reception gates only in the critic encoder. Their private Q heads remain ordinary Deep Sets with the same architecture as the corresponding ungated presets. Each gate receives the normalized local token and processed global message, uses a D/4-wide hidden layer and sigmoid outputs, and selects M-dimensional message features before compression. Gate parameters are additional capacity; no other component is shrunk. Pooling remains shared, so gating does not recover receiver-specific sender selection.

`deepset_interaction_config` exposes `message_dim`, local `hidden_dims`, `gated`, and `gate_hidden_dim`. For off-policy Deep Set presets, `gated` controls the encoder. The separate `critic_deepset_head_gating` builder option defaults to `False` and controls reception gates in the off-policy Q heads independently. `policy_kwargs={"critic_deepset_head_gating": True}` enables head gating for ablations, with or without a gated encoder. Fixed-scale configs record the resolved encoder and head settings, including message and gate dimensions. Explicit low-level configurations can intentionally depart from these defaults.

## Common NOP core

| Component | 2.5M | 5M-2L / 5M-3L | 10M-3L / 10M-4L / 10M-5L |
| --- | --- | --- | --- |
| Latent / transition width | 128 | 192 | 256 |
| Transition FF hidden widths | `[512]` | `[384]` | `[384]` |
| Transition blocks / heads | 2 / 4 | 2 / 4 | 3 / 4 |
| Source bridge hidden widths | `[256,256]` | `[256,256]` | `[320,320]` |
| Predictor hidden widths | `[192,192]` | `[192,192]` | `[256,256]` |

The common 10M NOP uses 256-dimensional latents and attention, with one 384-wide hidden layer in each transition FF MLP (`256 → 384 → 256`). The three residual attention/MLP blocks provide depth, while each MLP expands intermediate features. This retains the approximately 2M auxiliary budget. All three 10M encoder-depth choices use this same three-block NOP architecture; the policy encoder depth does not change NOP depth. The transition MLP depth is recorded as `num_feedforward_hidden_layers` in transition-model metadata and configured through `transition_model_num_feedforward_hidden_layers` in both NOP wrappers.

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

Parameter matching does not match compute or replay exposure. Recurrent replay counts sequences; feed-forward replay counts transitions. A recurrent batch has `batch_size × learning_steps` nominal learning positions before masking, with burn-in positions adding computation without scored losses. The nominal replay exposure per collected transition is `gradient_steps × learning_positions_per_batch / rollout_steps_per_iteration`. Equal exposure still leaves differences in temporal correlation, state reconstruction and optimizer batch structure.

The default off-policy batch budget is 1,024 learning positions, independent of environment count and rollout length. Feed-forward learners sample 1,024 transitions. Recurrent learners convert the budget to `ceil(1024 / learning_steps)` whole sequences, with a minimum of one sequence, after applying the learning-horizon override. The default 64-step horizon uses 16 sequences. With eight gradient steps and the example script's default 256 environments and one-step rollout, both have a nominal replay exposure of 32 learning positions per collected transition. At the Python API's default 1,024 environments, both have an exposure of eight. Comparisons use the same environment count and rollout schedule.

Learning horizons that do not divide 1,024 round up by less than one sequence. Horizons longer than 1,024 require at least one complete sequence. An explicit `algorithm_kwargs["batch_size"]` keeps its learner-specific meaning: transitions for feed-forward learners and sequences for recurrent learners. Custom batch sizes, gradient counts or `rollout_steps_per_iteration` overrides can deliberately change replay exposure. Burn-in remains additional computation and is excluded from the learning-position budget.

All off-policy presets share Adam at `5e-5`, the same 500-update learning-rate warmup from 1% of that rate, gradient clipping at `2.0`, discount `0.99`, target mixing coefficient `0.005`, and eight gradient steps. Replay capacity follows the same episode-length and fill-budget rule, enlarged when necessary for recurrent segments. Enabled NOP modules share a three-step horizon and loss coefficient `0.1`. SAC presets also share their entropy, action-sampling and update-frequency settings. TD3 retains delayed actor/target updates and smoothing; DDPG retains a single critic without delay or smoothing. Equal target mixing coefficients therefore do not imply equal target-update frequencies. The [training reference](../learning.md) lists the settings and algorithm-specific differences.

First-update timing also differs. Recurrent replay waits for at least `burn_in_steps + learning_steps` vector steps, in addition to the ordinary learning-start threshold. With the recurrent defaults this is 96 vector steps, or 24,576 collected transitions at 256 environments. The ordinary threshold is 10,000 transitions at that parallelism; rollout chunking and state-anchored segment availability can delay the actual first update. Both the configured threshold and observed first update belong in the experiment record.

Recurrent MAT inherits the four-step on-policy rollout default, which also sets its TBPTT sequence length. Hidden state persists across rollouts, but gradients traverse only the current unroll and there is no separate burn-in stage. Longer-memory comparisons need an explicitly chosen rollout/unroll length; a result with the short default does not establish the value of longer-horizon memory.

Original MAT's token-wise value readout still makes it a complete-policy baseline rather than an isolated decoder test, despite matching actor encoders and parameter allocations. MAPPO combines a local MLP actor with a Deep Set critic, so comparisons with MAT change actor processing and critic communication together. The smallest NOP tier uses approximately 0.67–0.69M parameters on WallEasy against its nominal 0.75M allowance. Hidden widths remain fixed across tasks, so task-specific measured counts accompany these budget labels.

Independent training seeds estimate uncertainty; evaluation seeds are not training replicas. Experiment records include resolved hyperparameters, model-scale versions and the Git commit, alongside measured main/NOP counts, temporal horizons, replay exposure and first-update timing.
