# Fixed model-scale layouts

The supported scales select explicit architecture layouts in [model_scale.py](../../swarmbots/learn/presets/model_scale.py). Hidden widths are fixed declarations, independent of task shapes. Parameter counting and width fitting are inspection tools, never part of policy construction.

| Canonical name | Main budget | NOP budget | Current version |
| --- | ---: | ---: | --- |
| `2.5M NOP0.75M` | 2.5M | 0.75M | `2.5+0.75M-v1` |
| `5M-2L NOP1M` (default) | 5M | 1M | `5M-2L+1M-v1` |
| `5M-3L NOP1M` | 5M | 1M | `5M-3L+1M-v1` |
| `10M-3L NOP2M` | 10M | 2M | `10M-3L+2M-v1` |
| `10M-4L NOP2M` | 10M | 2M | `10M-4L+2M-v1` |
| `10M-5L NOP2M` | 10M | 2M | `10M-5L+2M-v1` |

Short names omit the NOP suffix. `model_scale=None` (CLI `--model-scale custom`) enables explicit-width builders. Fixed scales define the current fresh-run architectures at `v1`; older layouts have no aliases or checkpoint migrations. Checkpoints require matching resolved architecture settings.

## Component allocation

MAT, MAPPO, SAC, TD3 and DDPG assign approximately 40% of the base main budget to the actor. Its observation encoder receives about 80–90% of that allocation, followed by a small action head. Actor encoder widths and depths match across methods within a layout. Critic encoders and individual heads are sized consistently; a single critic omits one ordinary head instead of reallocating its parameters elsewhere. QCX and Original MAT decoder attention, recurrence, detached actor-state conditioning and reception gates add capacity above the base budgets. NOP and frozen targets are counted separately.

| Component | Common structure | Method-specific operation |
| --- | --- | --- |
| Actor observation embeddings | Same local/global MLP dimensions and output width | Public inputs only; normalization/activation settings retain each family's configuration |
| Actor encoder blocks | Same feature width, block count and two-hidden-layer local FF MLP | Attention, or an additional `D -> 2D -> D` residual MLP for local actors |
| Action generation | Common encoder/output widths and matched non-attention head budgets | One residual MLP block, or one QCX/Original MAT decoder block with additional attention |
| Critic observation encoder | Same embeddings, feature width, block count and local FF MLP | Attention or expanding Deep Set communication; Q encoders additionally receive actions |
| Pooled V/Q readouts | Projection, two Deep Set interactions, masked mean and two regression hidden layers | Each pooled V or Q head has the same architecture and widths; single critics omit the second head |
| NOP | Common projection, transition and predictor layouts within a budget | Source adapters follow actual source-feature widths |

The depth label counts encoder blocks in both actor and critic. A `2L` transformer actor has two attention/FF encoder blocks followed by its separate action head. Decentralized encoders retain the same block count and FF widths; each missing attention operation receives a local residual MLP with the same leading parameter count, approximately `4D²`. This preserves the allocation between ordinary FF processing and the processing assigned to communication, while local replacements remain independent across agents. Critic readout interactions and actor head blocks are reported separately from encoder depth. Plain PPO retains its flattened MLP baseline.

## Actor encoder and action-generation layouts

| Layout | Actor width D | Encoder blocks | Encoder FF hidden widths | MLP head hidden width | QCX feature width | QCX FF hidden width | Original MAT FF hidden width |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 2.5M | 192 | 2 | `[320,320]` | 384 | 128 | 248 | 312 |
| 5M-2L | 256 | 2 | `[512,512]` | 512 | 160 | 392 | 472 |
| 5M-3L | 256 | 3 | `[320,320]` | 512 | 160 | 392 | 472 |
| 10M-3L | 320 | 3 | `[584,584]` | 640 | 192 | 544 | 640 |
| 10M-4L | 320 | 4 | `[408,408]` | 640 | 192 | 544 | 640 |
| 10M-5L | 320 | 5 | `[280,280]` | 640 | 192 | 544 | 640 |

Every fixed layout uses one local residual MLP head block or one autoregressive decoder block, independently of encoder depth. MAPPO uses the same local encoder blocks as MAT-DEC and decentralized off-policy actors. The 5M depth choices share 256-wide actor streams and identical heads; all 10M depth choices share 320-wide streams and identical heads. Encoder FF widths narrow as block count increases to retain the base actor budget.

MAT-IND, MAT-DEC, MAPPO and independent SAC/TD3/DDPG actors share a generic per-agent [residual MLP](../../swarmbots/learn/nn_components/residual_mlp.py) head. It computes `x = x + Linear2(GELU(Linear1(LayerNorm(x))))`, with a constant D-wide residual stream and a `D -> H -> D` branch, then a final LayerNorm and the action distribution/readout. There is no activation after the residual addition and no separate context or encoder-memory branch. At `5M-2L`, one `256 -> 512 -> 256` block plus final normalization uses 263,936 parameters before the action distribution. Custom-width builders retain their explicit MLP layouts unless a `ResidualMLPConfig` is supplied.

QCX retains the identical actor encoder and matches the non-attention head budget through its own configuration: a single action projection, a single projection of concatenated decoder/action context, a narrower decoder stream and separately declared FF width. Its causal context attention and cross-attention to full-width encoder features are additional capacity. Original MAT follows the same accounting with its native two-attention decoder. Local heads do not imitate these attention paths. All MLPs, input/output adapters, normalization and action readouts belong to the non-attention head budget; final action distributions cause small task-dependent count differences.

At `5M-2L`, QCX projects encoder queries `256 -> 160`, actions `A -> 160`, and concatenated decoder/action context `320 -> 160`. Both attentions use four 40-wide heads. Memory-attention K/V inputs retain the full encoder width, `256 -> 160` inside attention. Its FF branch is `160 -> 392 -> 160`, followed by final decoder normalization and a `160 -> 256` actor projection with GELU. There is no separate memory MLP. IND and QCX have approximately 282K non-attention head/readout parameters on WallEasy; QCX adds 236,800 decoder-attention parameters. Complete actors are 2,012,488 and 2,248,720 parameters, respectively.

| Layout | Additional QCX decoder attention | Additional Original MAT decoder attention |
| --- | ---: | ---: |
| 2.5M | 148,480 | 132,096 |
| 5M-2L / 5M-3L | 236,800 | 206,080 |
| 10M-3L / 10M-4L / 10M-5L | 345,600 | 296,448 |

These counts include attention Q/K/V/output projections and biases at the default settings. They are already included in actual actor/main totals, rather than added to those totals again. Parameter reports and diagrams identify this additional capacity explicitly. Nominal scale budgets describe the base architecture; autoregressive comparisons match encoders and non-attention heads, rather than total actor parameters.

### Five-million encoder-depth comparison

`5M-2L` and `5M-3L` use 256-wide actor features, two and three actor encoder blocks, and identical one-block action heads. All token-based actor families follow these depths, including decentralized and off-policy actors. Critic depth also increases from two to three blocks; NOP and plain PPO's shared architecture retain their budget-specific layouts.

### Ten-million encoder-depth comparison

The 10M actor and critic encoder depths are three, four and five blocks. Actor streams stay 320-wide and action heads remain identical one-block structures. Attention and local actors use the same encoder block counts. The NOP core retains three blocks at every depth. Base main counts stay comparable by adjusting encoder FF widths.

## Critic layouts

| Layout | Encoder width | Blocks | Encoder FF hidden width | Each V/Q readout width |
| --- | ---: | ---: | ---: | ---: |
| 2.5M | 256 | 2 | 384 | 128 |
| 5M-2L | 384 | 2 | 784 | 128 |
| 5M-3L | 320 | 3 | 656 | 128 |
| 10M-3L | 384 | 3 | 1416 | 192 |
| 10M-4L | 384 | 4 | 864 | 192 |
| 10M-5L | 384 | 5 | 536 | 192 |

Each pooled V and Q readout has the same feature widths, two interaction blocks and two regression hidden layers. Twin critics add a second ordinary head to the shared encoder. Single V and DDPG critics have fewer parameters by approximately one Q head; their remaining components are not enlarged to compensate. Observation/action conditioning can cause small encoder-count differences. Original MAT retains token-wise value prediction and pooling, with a separately sized value MLP matching one ordinary head. On WallEasy, critic-role counts match within 1% after accounting for the extra twin-Q head.

Pooled V/Q heads expose Deep Set message width, local updates, gates and normalization. Shared critic encoders expose message width and gates through the interaction config, while the surrounding encoder config owns LayerNorm and local FF processing; the nested communication module omits duplicate normalization and uses a residual update. Explicit nested off-policy encoder/head configs retain these structural controls; fixed scales choose their standard feature widths and block counts. Low-level flattened MLP critics also use the same per-head width for single and twin configurations. Independent twin encoders duplicate capacity without resizing either encoder.

## Recurrence

Fixed recurrent actors preserve the corresponding feed-forward observation encoder, action decoder and FF parameter shapes, then add memory, temporal adapters and normalization. Fixed TMASAC follows the same additive rule as recurrent MAT instead of moving FF capacity into two MLPs around memory. The custom-width TMASAC builder retains its configurable inter-module MLP structure.

Recurrent MAT adds memory to both actor and critic encoders. Default recurrent TMASAC adds actor memory and conditions a feed-forward critic on detached actor state. Their memory placement and critic information paths therefore differ, even when the same recurrent cell family is selected.

| Layout | MAT actor LSTM memory | TMASAC LSTM memory | TMASAC sLSTM memory |
| --- | ---: | ---: | ---: |
| 2.5M | 192 | 137 | 224 |
| 5M-2L | 208 | 166 | 264 |
| 5M-3L | 160 | 130 | 208 |
| 10M-3L | 280 | 184 | 288 |
| 10M-4L | 232 | 156 | 248 |
| 10M-5L | 204 | 140 | 216 |

TMASAC uses narrower dense LSTM memory than headwise sLSTM memory to match complete main counts within 0.5% on WallEasy. Temporal projections and detached critic-state adapters follow the actual memory width. Per-layer temporal specifications follow encoder depth; fixed actor and critic depths agree, while custom layouts can select different depths.

The optional recurrent TD3 critic uses the same per-layer memory width as recurrent MAT's critic: 192/256/212/256/220/196 across the six layouts. Enabling it adds memory to the unchanged critic backbone and retains the ordinary Q heads. Different recurrence placements remain complete-method comparisons; their total added capacity need not be identical.

## Comparability limits

Plain PPO remains a shared, flattened MLP baseline: half-shared actor/critic attribution is an accounting convention, and its internals do not match the token-based families. Converting it to independent token encoders would change the baseline architecture. Original MAT retains a distinct value readout. Attention and Deep Set communication have similar parameter budgets but different mixing behavior. A local MLP replaces communication capacity without adding inter-agent information. Q heads receive actions; V heads do not. Recurrence and gates are explicit added capacity, and optional independent encoders duplicate the full encoder.

Matched parameter counts do not establish matched compute, effective capacity or optimization. Autoregressive generation, recurrent unrolls and NOP transitions reuse parameters. Comparisons use the same task shapes and structural overrides, with component, main, auxiliary and frozen-target counts reported separately. `use_nop=False` preserves the complete main architecture.

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

The cleanest comparisons hold the learner, action distribution, NOP settings and task shapes fixed. The controls below refer to the fixed-scale layouts with common structural settings. Custom-width mode (`model_scale=None`) can select different encoder/head structures and needs a separate component comparison.

| Comparison | Controlled components | Intended difference |
| --- | --- | --- |
| `mat_ind` / `mat_dec` | Critic, action head, actor FF layout and training defaults | Actor attention versus matched local MLP processing |
| `mat_dec` / `mappo__deepset_critic` | Local actor structure and V readout | Attention versus Deep Set critic communication |
| `tmasac` / `tmasac_dec`; `tmatd3` / `tmatd3_dec` | Critic, actor/head budgets and learner settings | Actor communication |
| `tmasac_dec` / `masac__deepset_critic`; `tmatd3_dec` / `matd3__deepset_critic` | Local actor and private Q heads | Critic communication |
| `mat_ind` / `mat_qcx` | Actor observation encoder, critic, distribution, non-attention head budget and learner settings | Autoregressive action generation with additional decoder attention |
| Ordinary / gated Deep Set presets | Backbone widths/depth, actor and private heads | Additional reception gates and their parameters |
| NOP enabled / disabled within one variant | Complete main-policy architecture | Auxiliary objective and its encoder gradients |

The actor, critic and decoder feature layers use the common initialization gains, separately from the small action/value output gains. Independent MAT/MAPPO/off-policy MLP heads apply their configured head gain to every affine layer. Attention retains geometry-dependent Xavier initialization; common feature gains do not imply identical variance for packed and separate attention projections. Original MAT uses the same Xavier variance for its separate attention projections as packed PyTorch QKV projections. Plain PPO maps observation hidden/projection gains to its shared MLP, the actor-head gain to all actor feature layers, and the critic-regressor gain to its flat critic features; its architecture has no separate per-agent critic projection. PPO also forwards the shared action/value readout gains and PopArt configuration. All high-level PPO components receive the same base learning rate by default; `algorithm_kwargs["parameter_lr_multipliers"]` enables explicit component-specific tuning. A decoder name alone does not reduce its learning rate.

At fixed scales, `mat_dec` and `mappo__deepset_critic` construct their complete actors before their different critic encoders, so matching seeds give identical actor encoders, action heads and distribution weights. QCX context-token normalization defaults to enabled and respects explicit `MATNormalizationConfig.normalize_context_tokens` overrides; it normalizes learned context features rather than raw inputs.

The off-policy critic-communication pairs also preserve identical complete actor weights under matching seeds/settings: `tmasac_dec` with `masac__deepset_critic`, and `tmatd3_dec` with `matd3__deepset_critic`. TD3's target actors match as well.

Independent MLP heads and QCX share pre-norm residual processing, but QCX applies a final projection/activation after its decoder normalization. Original MAT retains its post-normalized decoder and token-wise V readout. These choices can produce different feature variances and gradient paths despite matching initialization gains and parameter budgets. Original MAT is a complete baseline, not an isolated decoder ablation. The shared flattened PPO backbone also changes parameter sharing and NOP gradient routing. Cross-learner comparisons measure complete methods: PPO, SAC and TD3/DDPG retain different objectives, policy distributions, replay/on-policy data use and optimizer schedules.

Plain PPO uses joint-agent likelihood ratios and KL, whereas MAT/MAPPO clip per-agent ratios. Identical clipping ranges and KL targets therefore do not imply identical effective trust regions or adaptive learning-rate signals. Plain PPO is a complete architecture/objective baseline.

Depth choices reallocate a fixed budget from FF processing to additional interaction blocks. Actor width and action heads stay fixed within each budget tier; the 5M critic layouts also change encoder width from 384 to 320. These are matched-budget encoder-depth comparisons, not pure depth ablations; report the measured attention/MLP proportions with any depth result. QCX and Original MAT retain the same additional decoder-attention allowance across the depths of a budget tier.

TMASAC/TMATD3 use centralized attention actors. Compare their decentralized `*_dec` controls against MASAC/MATD3 to isolate critic communication; comparing the primary centralized variants also changes actor communication. The detached actor-state critic bridge adds both capacity and input information, so canonical recurrent off-policy presets are not actor-only recurrence ablations.

Parameter matching does not match compute or replay exposure. Recurrent replay counts sequences; feed-forward replay counts transitions. A recurrent batch has `batch_size × learning_steps` nominal learning positions before masking, with burn-in positions adding computation without scored losses. The nominal replay exposure per collected transition is `gradient_steps × learning_positions_per_batch / rollout_steps_per_iteration`. Equal exposure still leaves differences in temporal correlation, state reconstruction and optimizer batch structure.

The default off-policy batch budget is 1,024 learning positions, independent of environment count and rollout length. Feed-forward learners sample 1,024 transitions. Recurrent learners convert the budget to `ceil(1024 / learning_steps)` whole sequences, with a minimum of one sequence, after applying the learning-horizon override. The default 64-step horizon uses 16 sequences. The Python API and training scripts default to 1,024 environments and a one-step off-policy rollout, collecting 1,024 transitions per cycle. With eight gradient steps, both learner types have a nominal replay exposure of eight learning positions per collected transition. Comparisons use the same environment count and rollout schedule; reducing the environment count without changing the update settings increases this exposure.

Learning horizons that do not divide 1,024 round up by less than one sequence. Horizons longer than 1,024 require at least one complete sequence. An explicit `algorithm_kwargs["batch_size"]` keeps its learner-specific meaning: transitions for feed-forward learners and sequences for recurrent learners. Custom batch sizes, gradient counts or `rollout_steps_per_iteration` overrides can deliberately change replay exposure. Burn-in remains additional computation and is excluded from the learning-position budget.

All off-policy presets share Adam at `5e-5`, the same 500-update learning-rate warmup from 1% of that rate, gradient clipping at `2.0`, discount `0.99`, target mixing coefficient `0.005`, and eight gradient steps. Replay capacity follows the same episode-length and fill-budget rule, enlarged when necessary for recurrent segments. Enabled NOP modules share a three-step horizon and loss coefficient `0.1`. SAC presets also share their entropy, action-sampling and update-frequency settings. TD3 retains delayed actor/target updates and smoothing; DDPG retains a single critic without delay or smoothing. Equal target mixing coefficients therefore do not imply equal target-update frequencies. The [training reference](../learning.md) lists the settings and algorithm-specific differences.

First-update timing also differs. Recurrent replay waits for at least `burn_in_steps + learning_steps` vector steps, in addition to the ordinary learning-start threshold. With the recurrent defaults this is 96 vector steps, or 98,304 collected transitions at the default 1,024 environments. The ordinary threshold is 10,000 transitions at that parallelism; rollout chunking and state-anchored segment availability can delay the actual first update. Both the configured threshold and observed first update belong in the experiment record. A controlled feed-forward/recurrent replay comparison can set the same explicit `learning_starts` for both methods, at least `(burn_in_steps + learning_steps) * num_envs`, and verify the observed first update and update count. At the defaults this is 98,304 transitions. This protocol choice intentionally extends the feed-forward warmup; it is not imposed on every preset.

Recurrent MAT inherits the four-step on-policy rollout default, which also sets its TBPTT sequence length. Hidden state persists across rollouts, but gradients traverse only the current unroll and there is no separate burn-in stage. Longer-memory comparisons need an explicitly chosen rollout/unroll length; a result with the short default does not establish the value of longer-horizon memory.

The declared three-step NOP horizon also does not guarantee equal auxiliary exposure. A four-step PPO rollout has only `4 + 3 + 2 = 9` valid future-target positions before episode cuts, while replay can supply longer complete windows. Recurrent replay scores complete prediction windows inside its learning segment; critic NOP skips its first transition when the encoded inputs already contain the executed action. NOP comparisons therefore hold the learner and rollout/replay protocol fixed, or report scored origins/future positions alongside the horizon. Longer PPO rollouts reduce truncation at rollout boundaries.

Comparing `mat_ind` with `mappo__deepset_critic` changes both actor and critic communication; the `mat_dec` control separates those changes. The smallest NOP tier uses approximately 0.67–0.69M parameters on WallEasy against its nominal 0.75M allowance. Hidden widths remain fixed across tasks, so task-specific measured counts accompany these budget labels.

Independent training seeds estimate uncertainty; evaluation seeds are not training replicas. A complete comparison record contains training/evaluation seeds, resolved hyperparameters, model-scale versions and the Git commit, alongside measured main/NOP counts, temporal horizons, replay exposure and first-update timing. Standard training metadata omits the training seed and commit; the training example's `evaluation.json` records its training seed after evaluation. Complete records therefore require those fields to be captured explicitly.
