# Policy architecture diagrams

The generator produces a standalone, offline gallery at `docs/policy_parameters/diagrams/index.html`, organized by variant and scale. Each selected variant and scale has its own SVG diagram, Mermaid source (`.mmd`), Graphviz source (`.dot`), and Markdown page. `diagrams.json` stores the nodes, nested `content` (text or sub-component blocks), repetition counts, parameter subtotals, and dependencies for other renderers. All rendered outputs are local and ignored by Git; this reading guide and the generator are retained.

## Reading the diagrams

- Shapes are per world; batch dimension B is omitted. `N` is padded agents, `D` is feature width, and `K` is future prediction steps. Input and action widths come from the current WallEasy environment and are shown in each diagram and its generated report.
- Blue blocks are actor-critic shared representations. Green blocks belong to the actor, orange to the critic, and purple to NOP. Privileged inputs have no actor path.
- The SVG gallery separates actor/public features, critic, and NOP into rows sized from the actual component contents. Read each main path left to right; twin critics are stacked with their scalar outputs aligned. Detached actor-state bridges sit between the actor and critic rows. Connections use distinct attachment points and orthogonal routes around boxes and connection labels; action feedback takes an outer return path. The same geometry is recorded in JSON and used by the optional raster preview. Mermaid and Graphviz choose their own overall placement.
- Bordered insets group the internals of a sub-component. **Block × L** encloses exactly the operations repeated L times; input projections and final heads sit outside that inset. **Once** means once per encoder/decoder pass. The NOP transition coembedding and head run once per predicted step, outside its block stack. Top-to-bottom order shows the main stages. Mermaid uses nested subgraphs and Graphviz uses nested tables to retain these scopes.
- `5M-2L` is the two-block 5M reference. `5M-3L` uses three attention/Deep Set encoder blocks with narrower widths; decoder and FF-only actor depths, Q/V heads, and NOP core stay fixed. PPO/MAPPO therefore have identical diagrams across the two choices.
- The 10M layouts are `10M-3L`, `10M-4L`, and `10M-5L`. Their main attention/recurrent and shared Deep Set encoder stacks use three, four, or five blocks, with widths adjusted for comparable counts. All three retain three decoder blocks, three NOP transition blocks, and four FF-only actor blocks. Low-level shared-encoder policies retain two private blocks on each side. `10M-3L` names the three-block reference; the old name is rejected.
- Parenthesized numbers beside dimensions are exact parameter counts, including weights and biases: `linear: 2 → 3 (9)`. Counts within a repeated inset describe one block; its **Per block** subtotal includes all those operations and normalization. Different blocks have independent weights. Reuse over agents or timesteps does not multiply the count.
- Each learned component shows **Total: … parameters**, including all its child modules, normalization, and small projections. The header gives exact actor, critic, shared, main, NOP, and combined online totals. Box totals are additive; counts inside a box are included subtotals. Shared encoders count once, outside both private actor and critic totals. Feature extraction/concatenation adds zero parameters. V/Q scalar outputs have their learned readouts counted in the preceding critic box; the action box counts the final action distribution/readout.
- Solid arrows show data dependencies. Dashed arrows show temporal memory, detached actor state, causal previous-agent actions, or policy actions used in Q evaluation. They do not show gradients or optimizer updates.
- Transformer and Deep Set twin critics use **one shared action-conditioned encoder and two private Q heads**. Both families put two Deep Set interaction blocks and final pooling inside each Q head. Gated Deep Set presets add reception gates only to the critic encoder by default; their Q heads retain ordinary ungated Deep Set interactions. The optional `critic_deepset_head_gating=True` builder setting adds reception gates to Q heads independently for ablations. NOP consumes the masked shared per-agent features before those heads. DDPG uses the same encoder/head template with one Q head. Low-level independent encoders duplicate the full encoder and concatenate pre-head features for NOP.
- Optional observation/action MLPs encode the two inputs separately before concatenation and the joint token projection. Optional token LayerNorm follows the token projections once, before the repeated encoder blocks. Raw observation/action inputs have no LayerNorm; feature-wise environment normalization remains separate. Off-policy Q heads do not support PopArt.
- On-policy critics estimate a team V without actions. Expanding Deep Set interactions normalize D-wide tokens, project to M=5D/4 with activation, pool masked messages, process the pooled M-wide vector nonlinearly, then compress with a bias-free linear projection before global residual addition. A separately normalized local FF residual follows. Gated blocks show a reception gate from local token plus global message before compression. Shared critic encoders have the same 2/2/3 depth and local FFNs as transformers; private V/Q heads use two interactions. PPO flattens features; original MAT predicts token-wise V then pools.
- Original MAT and QCX are autoregressive across agent order. QCX projects encoder features into decoder queries while retaining the full encoder width for cross-attention memory. Causal action embeddings see earlier agents only.
- Recurrent encoders keep independent state per temporal block and add memory to unchanged feed-forward backbones. Canonical blocks execute attention, LSTM/sLSTM, then the existing FFN; no inter-module FFN is added. Recurrent TMASAC also sends a detached actor-state projection to its ordinary critic. MAT retains independent actor/critic encoders and temporal histories. MAT/RMAT NOP consumes critic-encoder features and trains only the critic encoder. MAPPO also has independent actor/critic MLP encoders and critic-side NOP; plain PPO retains its shared backbone.
- NOP consumes source features and a **known rollout/replay action sequence**. Its transition model is reused across future steps, with a latent residual. Prediction heads output transformed scalar/angle/rotation/binary groups; raw future observations supervise the loss and are not model inputs. Their output dimensions therefore need not sum to the raw observation width.
- Frozen target networks, activation functions, and optimization edges are omitted for clarity. Masks and normalization are shown where they explain the Deep Set update; other normalization internals are omitted. TD3/DDPG also maintain a target actor; SAC/TD3/DDPG maintain target critics. Those are separate frozen copies, outside the online main counts shown here.

## Regenerate

From the repository root:

```bash
swarmbots-diagram-policies
swarmbots-diagram-policies --include-hidden
swarmbots-diagram-policies --variants tmasac tmatd3 --scales 5M-2L
```

Every invocation constructs the selected current policies and writes full parameter/layer reports to `../generated/<scale>/` before drawing. Only the requested variants and scales are inspected; previous count exports are never used as inputs. `--include-hidden` also draws the NOP controls and gated Deep Set presets. The geometry uses ordinary Python and SVG and needs no Graphviz, Mermaid CLI, browser service, or external network. Optional `png_preview` needs Pillow. See the [output storage guide](../README.md).

Each `.mmd` works in Mermaid-compatible Markdown. A `.dot` can be rerendered with Graphviz:

```bash
dot -Tsvg 5M-2L/tmasac.dot -o tmasac.svg
```

The graph connections are explicit architecture rules checked against the policy implementation. Dimensions, MLP chains, and counts come from the actual constructed policies' audited configuration, layer, and module metadata. Module counts deduplicate aliased weights within each subtree; they are inclusive subtotals and must not be summed with their children. The builder assigns each online parameter to one box and checks reconciliation against the role totals. If a new architecture changes its dependencies, update the graph builder and its semantic tests. Counts and dimensions are measured again on the next invocation.

[Model-scale layouts](../model_scale.md) · [Parameter reports](../README.md)
