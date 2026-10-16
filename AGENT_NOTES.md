# Agent notes

Keep navigation and costly gotchas here; defaults/layouts belong in source, history in CHANGELOG.md. Learning paths below are relative to `swarmbots/learn`.

## Documentation style

Public docs introduce capabilities, applications, and practical workflows; implementation detail belongs in focused references. Prefer reader-neutral, subject-focused, natural declarative prose and active voice. Avoid explicit/implied second-person address, unnecessary imperatives, and framing around what the reader can or should do; do not mechanically substitute passive voice. Reader-directed language fits genuine tutorials, procedures, and conversational explanations.

## Navigation

| Area | Entry points |
| --- | --- |
| Benchmark and simulation | `swarmbots/benchmark/`, `swarmbots/mjw_env/`, `swarmbots/scenario_presets/` |
| Learning API and construction | Lazy `swarmbots.learn` exports; `training.py`; `presets/variants.py`, `policy_factory.py` |
| Policy layouts and defaults | `presets/model_scale.py`, `sac.py`, `tmasac.py`; `algos/off_policy/` owns shared SAC/DDPG/TD3 machinery |
| Shared critic interactions | `nn_components/deep_set.py`; `DeepSetInteractionConfig` |
| Inspection and plotting | `swarmbots/tools/`, `swarmbots/plotting/`; guides in `docs/policy_parameters/` |

## Policy and checkpoint invariants

- Preset names use `base_variant__modifier`: architecture names retain single underscores (`mat_qcx`, `tmasac_dec`); modifiers use double underscores (`mat_qcx__lstm`, `masac__deepset_critic`). Removed names have no aliases. Registry membership comes from `variants.py`; retained low-level builders are not necessarily public presets. Actor/critic encoder sharing differs from sharing between Q heads.
- Model scales are fixed, versioned, task-independent declarations, never runtime fitting. `5M-2L` replaces rejected `5M` aliases; `5M-3L` changes encoder stacks, retaining other architectures. The 10M family is `10M-3L`/`10M-4L`/`10M-5L`; old `10M` aliases are rejected. The 4L/5L choices keep decoder/readout/NOP and FF-only actors fixed, with narrower FF/memory modules for comparable counts. Registry order is budget then encoder depth and also controls gallery order. Recurrence/gates add capacity. Fixed widths override custom widths but preserve explicit structural configs; `model_scale=None` / CLI `legacy` uses current builders, not historical architectures. `ScaleLayout.slug` separates equal-budget layouts.
- MAT owns independent actor/critic encoders: `encoder_config` is actor, `critic_encoder_config` critic. RMAT state maps `actor`/`critic` histories; rollout advances both, inference actor only, bootstrap critic only. MAT/RMAT NOP consumes the attached critic-encoder output from the same evaluation pass; use `nop_latent_dim` for its source width. Recurrent NOP follows critic history, not actor history. Compiled joint evaluation may materialize zero gradients for unused outputs; isolation tests check zero gradients and unchanged weights. Recurrent models add memory to the feed-forward backbone; critic state adapters use actual temporal `hidden_dim`, not token width.
- Deep Set/transformer off-policy critics share embedding/local processing and private Q-head construction (`build_q_network`). Independent encoders duplicate capacity without resizing. Copied message/gate weights need independent initialization. NOP consumes masked pre-head features once, concatenating only independent encoders; re-encoding must pass `agent_mask`. Flattened MLP critics reject NOP; off-policy Deep Set heads reject PopArt.
- MAPPO has independent `actor_encoder` and `critic_encoder` MLPs (`mappo_encoder.py`); `actor_config.latent_dim` and `critic_encoder_config.latent_dim` can differ. NOP uses critic features, value/NOP gradients never update the actor, and inference runs only the needed encoder. Privileged variables enter the value head. Plain PPO alone keeps the shared MLP backbone. Fixed-scale MAPPO budgets are 40% actor / 60% critic; changed architecture/NOP semantics bump layout versions.
- Raw-input LayerNorm erases scalar signals; preserve observation/action features before projection. Feature-wise observation normalization and learned-feature normalization are separate.
- Checkpoint reconstruction requires matching task, layout/version, distribution, NOP and structural settings. Changed observation layouts can invalidate checkpoints even when shapes match. Recurrent TD3 additionally needs `td3_recurrent_actor` and the original `td3_recurrent_critic` setting.
- Loading must restore LR controllers/custom schedulers and their enabled state, not just Adam LR. Off-policy replay is not checkpointed. gSDE schedules live in `gsde_reset.py`, persist in algorithm `hyper_parameters`, and default to episode-only when absent; initialize noise before masked resets.

## Learning and evaluation

- SAME_STEP returns reset observations; terminals are `final_obs` selected by `_final_obs`. Truncation bootstrap uses terminals; recurrent PPO requires TBPTT. Observation buffers are reused: retained data needs copies; integer-indexed reset writes must assign back to source tensors.
- `base_policy.policy_inference_step` brackets complete MAT/RMAT CUDA graph inference and snapshots outputs before subsequent calls reuse storage. Boundaries stay outside compiled submodules.
- PPO collects fixed-step rollouts and rejects core-policy dropout; NOP dropout is independent. Entropy regularization belongs to learners: PPO's masked joint-importance-weighted `mc_ent_coef`, SAC temperature tuning. SAC needs pathwise/straight-through sampling; ordinary SMB is for PPO, Gumbel SMB for SAC. Squashed Gaussian likelihoods use stable pre-tanh Jacobians.
- Diagnostics need masks from their own cached parameter batch, captured before replay updates replace it. Exclude recurrent padding/unused truncation successor slots. NOP metrics stay detached on-device until aggregation; reusable compiled outputs need snapshots.
- TD3/SAC delays use persistent update counters; SAC delay also gates temperature. Critic NOP updates each step, actor NOP follows actor delay. Recurrent TD3 preserves separate online/target actor histories; critics consume matching detached states. Candidate-action branches must not overwrite executed-action history.
- `train` saves/closes; `make_training` leaves closing to the caller. Example scripts require fresh output directories; Python supports continuation. `total_timesteps` is an absolute final counter. Default run metadata omits training seed/commit; the training example's `evaluation.json` records seed and final steps.
- A fresh `as_benchmark_policy` per evaluation/recording owns temporal/action/noise state and freezes normalization without mutating training statistics. It shares the actor: no concurrent training. Scenario-conditioned evaluation needs explicit `scenario_id`; supplied IDs win, and public globals are padded before normalization.
- Protocol evaluation takes the first episode per lane in lane order, requiring `num_episodes <= num_envs`. Recording recycles lanes, discards interrupted episodes, closes environments and waits for writers. Actors may combine public observations across agents, never privileged fields. Evaluation seeds sample the existing morphology pool, not unseen morphologies.
- Logging console/persistence profiles are independent. Filtering follows aggregation; preserve scenario-success presence masks, CSV columns/history and restored compressed logs.
- The standalone PPO example stays independent of `swarmbots.learn`; stored pre-tanh samples avoid inverse-tanh saturation because squash Jacobians cancel in PPO ratios. Biasless PopArt needs its offset; zero-angle rotation gradients need analytic sinc coefficients.

## Simulator invariants

- `MJWSwarmBotsVectorEnv` prepends raw active-agent count at `hidden_global_vars[..., 0]`; scenario fields start at 1. Normalization leaves the count raw; critics must not append it again. Joint/observation hinge counts derive from `mjw_env/swarm/unit.py`.
- All resets share `_reset_worlds`. Positive settling uses CPU-settled snapshots; zero disables it. Reseeding clears prefetch; pool changes invalidate pending snapshots. FIFO reservations include ready/queued/in-flight work: worker timing must not change sampling order. Preserve shuffle/shared RNG streams during device setup and partial reseeding.
- First-episode staggering belongs in training construction, outside `_make_policy_env`, so evaluation/recording retain full episodes. Explicit first-episode keys, including `None`, suppress generated limits; staggering uses a separate RNG.
- Wall resets intentionally include partially solved assemblies; milestones latch without retroactive rewards. Live reward commands accept only declared `reward_weights` and validate all scenario children before mutation; other reward settings affect cached potentials.
- Surface connector positioning is default; legacy positioning changes spacing. Pool grounding uses `get_unit_rest_min_z`, assuming zero hinge angles and yaw-only resets; numeric `z_pos` retains fixed-height offset semantics. Payload spacing needs shape-aware footprints and float32 clearance at actual sampled origins.
- CUDA runtime/reset work follows PyTorch's current stream; physics capture uses a capturable Warp stream and `data.qpos.device`. Padded model templates are separate from sampled assemblies. Different task shapes share Dynamo caches: isolate processes or reset compiler state after closing environments, never silently reset callers' policies.

## Tools and release hazards

- Diagram generation inspects current policies every invocation. Only explanatory guides are tracked; generated counts/reports/diagrams stay local and excluded by Git and `MANIFEST.in`. No snapshots or `--refresh`. Integration tests cache metadata per variant/scale.
- Diagram `ComponentBlock` insets contain repeated operations/inclusive subtotals; outer `Node` boxes own additive counts. One-time projections/readouts stay outside repeated blocks. `policy_diagram_layout.py` owns shared SVG/Pillow geometry and reserved routing space.
- Plotting imports Matplotlib lazily, preserves the caller's backend and direct `plot_logs` export despite the module-name collision. Aggregation interpolates finite ranges without extrapolation, keeps last finite duplicate-X, applies cutoff first, and uses population std; contributing seed counts vary with X.
- CUDA dependencies are pinned together in the source `cuda` group; published metadata uses standard PyTorch. CPU CI needs EGL (`MUJOCO_GL=egl`); OSMesa caused native crashes. Installed-wheel checks run outside the checkout; CPU CI does not cover CUDA.
- Windows sandbox caches belong under `.tmp` (`TEMP`/`TMP`, Inductor, Warp). Python 3.13's `0o700` temp ACLs may exclude the sandbox identity; local `.tmp/sitecustomize.py` can map mkdir to `0o777`, inherited through `PYTHONPATH`. Never package this workaround.
- Release: unused alpha via `uv version <version> --no-sync`; align CITATION/README pin/dated changelog. Run `uv sync --locked --extra dev`, release metadata tests, CUDA tests (`.venv/Scripts/python.exe -m pytest -m cuda`) and `git diff --check`; stop on failure. Review/stage intended changes, commit/push main, then push an annotated version tag.
- **Tag pushes publish to PyPI** through `.github/workflows/release.yml`. Verify publication; corrections require a new version, never overwritten tags/distributions. Contributor/release procedures stay out of public docs.
