# Feed-forward preset audit

Audited on 2026-10-06 after the recurrent TMASAC critic fix. The audit instantiated all **35 registered variants**, plus **six TD3 configurations** with recurrent actors and/or recurrent critics. It used the default `SwarmBots-WallEasy-v0` observation/action shapes and disabled NOP during construction. All 35 default main-policy processing counts matched [the saved report](report.md).

The [compressed audit JSON](preset_audit.json.gz) records the 41 configurations, actor/critic/shared parameter counts, and every MAT/RMAT encoder layer's resolved affine shapes, block counts, and temporal output projections. Counts below exclude NOP, targets, attention, recurrent modules, normalization, and embeddings. Read the JSON with `json.load(gzip.open(path, "rt", encoding="utf-8"))`.

## Routing result

No additional instance of the same actor-config-to-critic leak was found in TD3, DDPG, MASAC, PPO, MAPPO, or on-policy MAT. There are, however, other feed-forward budget differences that matter when selecting experimental controls.

| Family / configuration | Resolved behavior |
| --- | --- |
| DDPG / MATD3 / MASAC MLP and Deep Set critics | Critic hidden dimensions come from their own `JointCriticConfig`. They retain their configured three 512-wide hidden layers; recurrent actor FF overrides do not remove critic layers. |
| TMATD3 with a recurrent actor and ordinary critic | Actor-specific FF overrides stay on the actor. The critic retains `[512, 512]`, including its middle `512 -> 512` layer. Actor-state conditioning adds a separate critic projection and larger input rather than removing FF capacity. |
| TMATD3 with a recurrent actor and recurrent critic | The critic retains the full `[512, 512]` config **in both** recurrent FF blocks. This increases critic FF capacity relative to its ordinary critic. |
| `mat_ind_lstm` / `mat_qcx_lstm` | The shared encoder has two FF blocks per layer instead of one. Its separate critic head retains the same parameter shapes as the corresponding ordinary MAT preset. |
| PPO / MAPPO | Actor/shared feature dimensions and critic dimensions are set independently in the explicit preset branches. Their registered variants do not route a smaller recurrent FF config into a critic. |
| `tmasac_slstm_shared_encoder` | The recurrent shared encoder has one full `[512, 512]` FF block per layer; its inter-module MLP is disabled. Actor and critic tails keep their full ordinary FF blocks. Its affine budget matches `tmasac_shared_encoder`. |

The relevant construction paths are [the policy factory](../../swarmbots/learn/presets/policy_factory.py), [RMATEncoderLayer](../../swarmbots/learn/algos/r_mat/r_mat_encoder.py), and [the recurrent transformer critic](../../swarmbots/learn/algos/off_policy/recurrent_transformer_critic.py).

## Other capacity differences

| Configuration | Main-policy MLP + linear parameters | Difference from the listed ordinary baseline |
| --- | ---: | ---: |
| `mat_ind` | 916,937 | baseline |
| `mat_ind_lstm` | 1,573,833 | +656,896 |
| `mat_qcx` | 1,165,769 | baseline |
| `mat_qcx_lstm` | 1,822,665 | +656,896 |
| `tmatd3` | 2,836,622 | baseline |
| `tmatd3` + recurrent actor | 3,230,350 | +393,728 |
| `tmatd3` + recurrent actor and critic | 4,149,902 | +1,313,280 |

For both recurrent MAT presets, the increase is **525,824** parameters for the added shared-encoder FF blocks plus **131,072** for temporal output projections outside the recurrent modules. The critic-head affine budget remains **198,913** in both the ordinary and recurrent variants.

For recurrent-actor TMATD3, the actor adds **131,072** affine projection parameters and the ordinary critic adds **262,656** for actor-state conditioning. With a recurrent critic instead, actor-state conditioning is disabled by default; the critic adds **1,051,136** parameters in its extra full FF blocks and **131,072** in temporal output projections relative to the ordinary, unconditioned critic. These are actual affine budget changes in addition to recurrent parameters.

## TMASAC controls that bypass the canonical preset

Two existing controls in [the variant registry](../../swarmbots/learn/presets/variants.py) use the low-level policy builder directly instead of `make_tmasac_options`. Consequently, they still resolve the generic FF config to a single hidden layer `[512]` rather than the canonical TMASAC `[512, 512]` block.

| Variant | Actor MLP + linear | Critic MLP + linear | Combined |
| --- | ---: | ---: | ---: |
| `tmasac` | 1,237,144 | 1,601,026 | 2,838,170 |
| `tmasac_segment` | 711,832 | 1,075,714 | 1,787,546 |
| `tmasac_lstm` | 1,237,656 | 1,863,682 | 3,101,338 |
| `tmasac_lstm_no_actor_state` | 842,904 | 1,075,714 | 1,918,618 |

`tmasac_lstm_no_actor_state` also has one actor FF block per layer instead of the canonical recurrent actor's two blocks, and enables temporal output projections. Its actor/critic changes therefore extend beyond removing actor-state conditioning. `tmasac_segment` is a feed-forward policy with a segment-replay interface, but its actor and critic FF depths differ from `tmasac`.

For an isolated no-actor-state comparison, instantiate the canonical `tmasac_lstm` preset with `tmasac_actor_state_critic_input_config=None`. This preserves its actor and full critic FF blocks while removing the conditioning path. Selecting `tmasac_lstm_no_actor_state` currently selects the other architecture recorded above.

## Verification and reproduction

Added regression coverage that instantiates online and target TD3 policies, supplies the smaller actor-only FF override, and verifies that ordinary and recurrent critics keep the full generic FF architecture. MLP/Deep Set critic parameter shapes are checked independently of actor-state conditioning. Additional tests verify that recurrent MAT retains the ordinary MAT critic-head architecture and adds a full encoder FF block. The new audit tests, the existing TMASAC FF regressions, and the parameter-count tests passed together: **27 tests**. Ruff also passed.

The existing parameter-count script reproduces the main comparisons:

```bash
python examples/inspect_policy_parameters.py --variants mat_ind mat_ind_lstm mat_qcx mat_qcx_lstm --no-nop
python examples/inspect_policy_parameters.py --variants tmatd3 tmatd3_dec --no-nop --policy-kwargs '{"td3_recurrent_actor": true}'
python examples/inspect_policy_parameters.py --variants tmatd3 tmatd3_dec --no-nop --policy-kwargs '{"td3_recurrent_actor": true, "td3_recurrent_critic": true}'
```

NOP toggles do not change the main-policy affine counts. Equal total affine counts alone do not establish equal actor/critic allocations or equal compute; the resolved layer shapes in the JSON make those differences visible.
