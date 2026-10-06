# Record benchmark policy behavior

Use `swarmbots record` or the checkout script `examples/record_policy.py` to inspect how a policy moves, reconfigures the swarm, and interacts with obstacles or payloads. Both support custom policies and included-baseline checkpoints. They record complete episodes with the simulator's live MP4 recorder, including reward overlays and the scenario's default camera, and wait for video encoding before returning.

Videos illustrate behavior. Use the [benchmark evaluator](api.md#policy-evaluation) and [reporting protocol](benchmark_protocol.md) for quantitative comparisons.

## Custom policy factories

Pass an importable factory with `--policy module:function`:

```bash
swarmbots record SwarmBots-Bridge-v0 --policy my_policy:make_policy --episodes 3 --device cuda
```

The factory receives these keyword arguments and returns a benchmark policy callable:

```python
import torch

def make_policy(*, benchmark_id: str, device: torch.device, seed: int, deterministic: bool) -> MyPolicy:
    return MyPolicy(benchmark_id, device=device, seed=seed, deterministic=deterministic)
```

The returned callable receives `(observations, episode_starts)` and returns an action dictionary. Observations contain only `local_obs`, `global_obs`, and `agent_mask`. Configure evaluation mode and frozen normalization in the factory. `--stochastic` passes `deterministic=False`; the factory controls how that affects its policy. See the [policy interface](api.md#policy-evaluation) for details.

## Included-baseline checkpoints

Choose any ID from `swarmbots list` and the learning variant used to train the checkpoint:

```bash
swarmbots record SwarmBots-POWallMedium-v0 \
  --checkpoint runs/po-wall/slstm/models/model_100000000_steps_final.pt \
  --variant tmasac_slstm --episodes 5 --parallel 2 --device cuda \
  --output recordings/po-wall/slstm
```

From a Windows checkout, the equivalent is:

```powershell
.\.venv\Scripts\python.exe examples/record_policy.py SwarmBots-POWallMedium-v0 --checkpoint runs/po-wall/slstm/models/model_100000000_steps_final.pt --variant tmasac_slstm --episodes 5 --parallel 2 --device cuda --output recordings/po-wall/slstm
```

Policy weights and observation normalization are restored. Recording constructs the policy and environment without allocating a trainer, replay buffer, or optimizer. Recurrent state, previous actions, and distribution state reset at episode boundaries. Compiled checkpoint keys also load into eager policies.

DDPG/TD3 checkpoints saved by the trainer retain the configured exploration noise for `--stochastic` recordings. Use `--exploration-noise 0.4` (Python: `exploration_noise=0.4`) to override it. Older checkpoints without saved policy settings use the default `0.1` unless overridden. Deterministic recording omits exploration noise.

The selected task's observation/action spaces, architecture, action distribution, and NOP settings must match the checkpoint. The command does not infer them from checkpoint metadata. Use `--policy-kwargs '{"enc_d_model":32,"dec_d_model":16}'`, `--continuous-action-dist predicted_std_gaussian`, or `--no-use-nop` when needed. If PopArt was customized through training's `algorithm_kwargs`, pass the matching `use_popart` in recording's `--policy-kwargs`.

## Recording options

- `--episodes` counts completed videos; `--parallel` caps the number of simulator worlds. More episodes reuse worlds after reset.
- Actions use deterministic modes by default for learning checkpoints; `--stochastic` samples instead.
- `--seed` defaults to 1000 and seeds the simulator and PyTorch. `--device auto` selects CUDA when available, otherwise CPU.
- `--width`/`--height` default to 1280×720. `--camera -1` uses the scenario camera; camera names and numeric IDs are accepted.
- `--fps` defaults to 30. `--frame-stride 2` captures every second step and halves playback FPS to preserve duration. Use `--fps-mode fixed` to keep the requested FPS.
- `--episode-length` overrides the registered task's limit. `--scenario-kwargs` and `--env-kwargs` accept JSON objects for scalar/list overrides.
- `--model-scale "5M NOP1M"` selects the checkpoint's layout (the default); `2.5M` and `10M` select the additional fixed tiers; `--model-scale legacy` uses explicit widths for earlier checkpoints. Match training's scale and architecture settings.
- `--compile-policy` opts into policy compilation; use the [GPU setup guide](gpu_setup.md) first.
- The default directory is `recordings/<benchmark-id>/<timestamp>`. `--output` and `--prefix` select the directory and filename prefix. Reusing a directory and prefix can replace matching video filenames.

## Python helpers

`record_policy` accepts any benchmark policy callable:

```python
from swarmbots import record_policy

record_policy(
    policy,
    "SwarmBots-Bridge-v0",
    video_folder="recordings/bridge",
    num_episodes=3,
    device="cuda",
)
```

To record an included baseline from a loaded trainer, use its policy adapter:

```python
from swarmbots import record_policy
from swarmbots.learn import as_benchmark_policy

record_policy(
    as_benchmark_policy(trainer, deterministic=True),
    "SwarmBots-POWallMedium-v0",
    video_folder="recordings/po-wall",
    num_episodes=5,
    max_parallel_episodes=2,
    device="cuda",
)
```

Create a fresh adapter for each recording and use the policy's device. To load a checkpoint directly:

```python
from swarmbots.learn import record_checkpoint

record_checkpoint(
    "runs/po-wall/slstm/models/model_100000000_steps_final.pt",
    "SwarmBots-POWallMedium-v0",
    "tmasac_slstm",
    video_folder="recordings/po-wall/slstm",
    num_episodes=5,
    device="cuda",
)
```

Both helpers return the output directory after closing the simulator and finishing the writers. They accept `scenario_kwargs` and `env_kwargs`; `record_checkpoint` also accepts `model_scale`, `policy_kwargs`, `use_nop`, and `continuous_action_dist`. Use Python for configuration objects such as `MJWPreConnectedUnitLocationsConfig`, which cannot be represented by the command's plain JSON overrides.
