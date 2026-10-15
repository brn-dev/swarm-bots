# Policy architecture diagrams

Run the generator below, then open `docs/policy_parameters/diagrams/index.html` in a browser to choose a variant and scale. The gallery is standalone and works offline. Each of the **17 core variants × 3 scales** has its own SVG diagram, Mermaid source (`.mmd`), Graphviz source (`.dot`), and Markdown page. `diagrams.json` stores the nodes, nested `content` (text or sub-component blocks), repetition counts, parameter subtotals, and dependencies for other renderers. All rendered outputs are local and ignored by Git; this reading guide and the generator are retained.

## Reading the diagrams

- Shapes are per world; batch dimension B is omitted. `N` is padded agents, `D` is feature width, and `K` is future prediction steps. The audited WallEasy shapes are N=5, public local=71, public global=0, privileged local=4, privileged global=2 (including the active-agent count), and actions=12 per agent.
- Blue blocks are actor-critic shared representations. Green blocks belong to the actor, orange to the critic, and purple to NOP. Privileged inputs have no actor path.
- The SVG gallery separates actor/public features, critic, and NOP into rows sized from the actual component contents. Read each main path left to right; twin critics are stacked with their scalar outputs aligned. Detached actor-state bridges sit between the actor and critic rows. Connections use distinct attachment points and orthogonal routes around boxes and connection labels; action feedback takes an outer return path. The same geometry is recorded in JSON and used by the optional raster preview. Mermaid and Graphviz choose their own overall placement.
- Bordered insets group the internals of a sub-component. **Block × L** encloses exactly the operations repeated L times; input projections and final heads sit outside that inset. **Once** means once per encoder/decoder pass. The NOP transition coembedding and head run once per predicted step, outside its block stack. Top-to-bottom order shows the main stages. Mermaid uses nested subgraphs and Graphviz uses nested tables to retain these scopes.
- The 10M tier uses three main attention/recurrent blocks, three decoder blocks, and three NOP transition blocks. FF-only actors use four blocks; actor–critic shared variants have three shared blocks followed by two private blocks on each side. Each diagram records the corresponding audited depths and widths.
- Parenthesized numbers beside dimensions are exact parameter counts, including weights and biases: `element MLP (once): 384 → 384 → 384 (295,680)`. Counts within a repeated inset describe one block; its **Per block** subtotal includes all those operations and normalization. Different blocks have independent weights. Reuse over agents or timesteps does not multiply the count.
- Each learned component shows **Total: … parameters**, including all its child modules, normalization, and small projections. The header gives exact actor, critic, shared, main, NOP, and combined online totals. Box totals are additive; counts inside a box are included subtotals. Shared encoders count once, outside both private actor and critic totals. Feature extraction/concatenation adds zero parameters. V/Q scalar outputs have their learned readouts counted in the preceding critic box; the action box counts the final action distribution/readout.
- Solid arrows show data dependencies. Dashed arrows show temporal memory, detached actor state, causal previous-agent actions, or policy actions used in Q evaluation. They do not show gradients or optimizer updates.
- Transformer and Deep Set twin critics default to **one shared action-conditioned encoder and two independent Q heads**. Deep Set TD3/SAC pool the shared features once and send them to separate scalar MLP readouts; NOP consumes the shared final per-agent features once before pooling. Transformer Q heads retain private Deep Set interactions and pooling after the shared attention encoder. Deep Set DDPG has one complete Q network. With `critic_independent_encoders=True`, twin critics instead concatenate their independently encoded per-agent features for NOP.
- On-policy critics estimate a single team V and do not consume actions. Deep Sets first project elements once, then run the **Set interaction × L** inset. Each interaction performs global processing followed by local processing: `z = global LayerNorm(h)`, masked mean of `message MLP(z)`, then a global MLP and broadcast residual `u = h + g`. A separate `local LayerNorm(u)` and per-agent MLP produce the default local residual `h_next = u + local MLP(v)`. The local MLP therefore receives the global message from the same block. Masked agents are zeroed before the next block. Final per-agent features remain available before the separate **Team pooling · once** or **Team readout · once** inset. Standalone DDPG/TD3/SAC Deep Set encoders default to four interactions; on-policy critics and transformer Q readouts use two. PPO flattens agent features; original MAT predicts token-wise V then pools.
- Original MAT and QCX are autoregressive across agent order. QCX projects encoder features into decoder queries while retaining the full encoder width for cross-attention memory. Causal action embeddings see earlier agents only.
- Recurrent encoders keep state per temporal block. The core presets apply attention → inter-module MLP → LSTM/sLSTM → final MLP within each block; the diagram follows the configured temporal/attention order. Standard recurrent TMASAC also sends a detached actor-state projection to its feed-forward critic. MAT uses independent actor and critic encoders. Recurrent MAT shows a separate temporal history on each side; actor NOP consumes actor encoder features.
- NOP consumes source features and a **known rollout/replay action sequence**. Its transition model is reused across future steps, with a latent residual. Prediction heads output transformed scalar/angle/rotation/binary groups; raw future observations supervise the loss and are not model inputs. Their output dimensions therefore need not sum to the raw observation width.
- Frozen target networks, activation functions, and optimization edges are omitted for clarity. Masks and normalization are shown where they explain the Deep Set update; other normalization internals are omitted. TD3/DDPG also maintain a target actor; SAC/TD3/DDPG maintain target critics. Those are separate frozen copies, outside the online main counts shown here.

## Regenerate

From the repository root:

```bash
swarmbots-diagram-policies
swarmbots-diagram-policies --refresh
swarmbots-diagram-policies --variants tmasac tmatd3 --scales 5M
```

The default command reads the compact `counts.json.gz` snapshots included in the repository. `--refresh` reconstructs policies from current presets, updates these snapshots, and writes full parameter/layer reports to `../generated/<scale>/` before drawing. Rebuilding an audit retains all core presets and any cached hidden presets still registered, even when `--variants` selects fewer diagrams. `--include-hidden` also draws the remaining attention/NOP controls. The geometry uses ordinary Python and SVG and needs no Graphviz, Mermaid CLI, browser service, or external network. Optional `png_preview` needs Pillow. See the [output storage guide](../README.md).

Each `.mmd` works in Mermaid-compatible Markdown. A `.dot` can be rerendered with Graphviz:

```bash
dot -Tsvg 5M/tmasac.dot -o tmasac.svg
```

The graph connections are explicit architecture rules checked against the policy implementation. Dimensions, MLP chains, and counts come from the actual constructed policies' audited configuration, layer, and module metadata. Module counts deduplicate aliased weights within each subtree; they are inclusive subtotals and must not be summed with their children. The builder assigns each online parameter to one box and checks reconciliation against the role totals. If a new architecture changes its dependencies, update the graph builder and its semantic tests as well as refreshing the size audit.

[Model-scale layouts](../model_scale.md) · [Cross-scale counts](../scale_comparison.md)

## Individual diagrams

These links become available after running the generator.

| Variant | 2.5M | 5M | 10M |
| --- | --- | --- | --- |
| ppo | [SVG](2.5M/ppo.svg) · [source](2.5M/ppo.mmd) | [SVG](5M/ppo.svg) · [source](5M/ppo.mmd) | [SVG](10M/ppo.svg) · [source](10M/ppo.mmd) |
| mappo | [SVG](2.5M/mappo.svg) · [source](2.5M/mappo.mmd) | [SVG](5M/mappo.svg) · [source](5M/mappo.mmd) | [SVG](10M/mappo.svg) · [source](10M/mappo.mmd) |
| mat_orig | [SVG](2.5M/mat_orig.svg) · [source](2.5M/mat_orig.mmd) | [SVG](5M/mat_orig.svg) · [source](5M/mat_orig.mmd) | [SVG](10M/mat_orig.svg) · [source](10M/mat_orig.mmd) |
| mat_ind | [SVG](2.5M/mat_ind.svg) · [source](2.5M/mat_ind.mmd) | [SVG](5M/mat_ind.svg) · [source](5M/mat_ind.mmd) | [SVG](10M/mat_ind.svg) · [source](10M/mat_ind.mmd) |
| mat_dec | [SVG](2.5M/mat_dec.svg) · [source](2.5M/mat_dec.mmd) | [SVG](5M/mat_dec.svg) · [source](5M/mat_dec.mmd) | [SVG](10M/mat_dec.svg) · [source](10M/mat_dec.mmd) |
| mat_qcx | [SVG](2.5M/mat_qcx.svg) · [source](2.5M/mat_qcx.mmd) | [SVG](5M/mat_qcx.svg) · [source](5M/mat_qcx.mmd) | [SVG](10M/mat_qcx.svg) · [source](10M/mat_qcx.mmd) |
| mat_ind__lstm | [SVG](2.5M/mat_ind__lstm.svg) · [source](2.5M/mat_ind__lstm.mmd) | [SVG](5M/mat_ind__lstm.svg) · [source](5M/mat_ind__lstm.mmd) | [SVG](10M/mat_ind__lstm.svg) · [source](10M/mat_ind__lstm.mmd) |
| mat_qcx__lstm | [SVG](2.5M/mat_qcx__lstm.svg) · [source](2.5M/mat_qcx__lstm.mmd) | [SVG](5M/mat_qcx__lstm.svg) · [source](5M/mat_qcx__lstm.mmd) | [SVG](10M/mat_qcx__lstm.svg) · [source](10M/mat_qcx__lstm.mmd) |
| maddpg__deepset_critic | [SVG](2.5M/maddpg__deepset_critic.svg) · [source](2.5M/maddpg__deepset_critic.mmd) | [SVG](5M/maddpg__deepset_critic.svg) · [source](5M/maddpg__deepset_critic.mmd) | [SVG](10M/maddpg__deepset_critic.svg) · [source](10M/maddpg__deepset_critic.mmd) |
| matd3__deepset_critic | [SVG](2.5M/matd3__deepset_critic.svg) · [source](2.5M/matd3__deepset_critic.mmd) | [SVG](5M/matd3__deepset_critic.svg) · [source](5M/matd3__deepset_critic.mmd) | [SVG](10M/matd3__deepset_critic.svg) · [source](10M/matd3__deepset_critic.mmd) |
| masac__deepset_critic | [SVG](2.5M/masac__deepset_critic.svg) · [source](2.5M/masac__deepset_critic.mmd) | [SVG](5M/masac__deepset_critic.svg) · [source](5M/masac__deepset_critic.mmd) | [SVG](10M/masac__deepset_critic.svg) · [source](10M/masac__deepset_critic.mmd) |
| tmatd3 | [SVG](2.5M/tmatd3.svg) · [source](2.5M/tmatd3.mmd) | [SVG](5M/tmatd3.svg) · [source](5M/tmatd3.mmd) | [SVG](10M/tmatd3.svg) · [source](10M/tmatd3.mmd) |
| tmatd3_dec | [SVG](2.5M/tmatd3_dec.svg) · [source](2.5M/tmatd3_dec.mmd) | [SVG](5M/tmatd3_dec.svg) · [source](5M/tmatd3_dec.mmd) | [SVG](10M/tmatd3_dec.svg) · [source](10M/tmatd3_dec.mmd) |
| tmasac | [SVG](2.5M/tmasac.svg) · [source](2.5M/tmasac.mmd) | [SVG](5M/tmasac.svg) · [source](5M/tmasac.mmd) | [SVG](10M/tmasac.svg) · [source](10M/tmasac.mmd) |
| tmasac_dec | [SVG](2.5M/tmasac_dec.svg) · [source](2.5M/tmasac_dec.mmd) | [SVG](5M/tmasac_dec.svg) · [source](5M/tmasac_dec.mmd) | [SVG](10M/tmasac_dec.svg) · [source](10M/tmasac_dec.mmd) |
| tmasac__slstm | [SVG](2.5M/tmasac__slstm.svg) · [source](2.5M/tmasac__slstm.mmd) | [SVG](5M/tmasac__slstm.svg) · [source](5M/tmasac__slstm.mmd) | [SVG](10M/tmasac__slstm.svg) · [source](10M/tmasac__slstm.mmd) |
| tmasac__lstm | [SVG](2.5M/tmasac__lstm.svg) · [source](2.5M/tmasac__lstm.mmd) | [SVG](5M/tmasac__lstm.svg) · [source](5M/tmasac__lstm.mmd) | [SVG](10M/tmasac__lstm.svg) · [source](10M/tmasac__lstm.mmd) |
