# Policy video recording

The installed `swarmbots record` command and checkout script `examples/record_policy.py` produce MP4 videos of policy movement, unit connections, and interactions with obstacles or payloads.

Videos provide qualitative evidence of behavior. The [evaluator](api.md#policy-evaluation) produces quantitative scores.

Headless Linux recording requires [EGL/OpenGL libraries](gpu_setup.md#headless-video-recording) in addition to working simulation. Normal CUDA environments also require compiler setup, independently of the optional policy-compilation flag.

## Included-baseline checkpoints

Find a checkpoint in the training run's `models/` directory. Supply the task and variant used for training, replacing the example filename with the saved checkpoint:

```bash
swarmbots record SwarmBots-WallEasy-v0 --checkpoint runs/first-mappo/models/model_1000000_steps_final.pt --variant mappo --episodes 3 --output recordings/mappo
```

Without environment activation, use `.venv/bin/swarmbots` on Linux or `.\.venv\Scripts\swarmbots.exe` on Windows. The checkout script can also be run on Windows:

```powershell
.\.venv\Scripts\python.exe examples/record_policy.py SwarmBots-WallEasy-v0 --checkpoint runs/first-mappo/models/model_1000000_steps_final.pt --variant mappo --episodes 3 --output recordings/mappo
```

The command restores policy weights and observation normalization, records complete episodes, and waits for encoding before printing the output directory. It uses deterministic actions by default and resets recurrent state between episodes.

**Match the training configuration.** Recording does not infer the task, model scale, distribution, or architecture from checkpoint metadata. Add these options when the training defaults were changed:

| Training choice | Recording option |
| --- | --- |
| Smaller or different fixed scale | `--model-scale 2.5M`, `5M-3L`, `10M-3L`, `10M-4L`, or `10M-5L` |
| NOP disabled | `--no-use-nop` |
| Different action distribution | `--continuous-action-dist predicted_std_gaussian` (or the saved choice) |
| Custom architecture | `--policy-kwargs` with matching settings as a JSON object |
| Custom-width model | `--model-scale custom` plus matching architecture settings |
| Recurrent TD3 actor | `--policy-kwargs '{"td3_recurrent_actor":true}'` |
| Recurrent TD3 actor and critic | `--policy-kwargs '{"td3_recurrent_actor":true,"td3_recurrent_critic":true}'` |

The recurrent TD3 flags belong to the checkpoint's architecture; `--variant tmatd3` alone constructs a feed-forward policy. For a checkpoint from `train_recurrent_td3.py --variant tmatd3 --recurrent-critic`:

```bash
swarmbots record SwarmBots-POWallMedium-v0 --checkpoint checkpoints/recurrent-td3.pt --variant tmatd3 --policy-kwargs '{"td3_recurrent_actor":true,"td3_recurrent_critic":true}' --output recordings/recurrent-td3
```

Replace the checkpoint path with the saved file. The same dictionaries apply as `policy_kwargs` in Python checkpoint continuation, saved-policy evaluation, and `record_checkpoint`.

If PopArt was customized through training's `algorithm_kwargs`, supply matching `use_popart` through recording's `--policy-kwargs`. Loading requires the current checkpoint format and matching architecture and observation layouts.

## Recording options

| Option | Default and effect |
| --- | --- |
| `--episodes` | 5 completed videos |
| `--parallel` | Up to 4 worlds; additional episodes reuse them |
| `--output` | `recordings/<benchmark-id>/<timestamp>` |
| `--prefix` | Video filename prefix; use a new directory or prefix to avoid replacing videos |
| `--device` | `auto`: CUDA when available, otherwise CPU |
| `--seed` | 1000 |
| `--width`, `--height` | 1280 × 720 |
| `--camera` | `-1` for the scenario camera; names and numeric IDs are accepted |
| `--fps` | 30; changes playback speed, with every control step recorded |
| `--stochastic` | Action sampling instead of deterministic modes |
| `--compile-policy` | Policy compilation; requires working [GPU setup](gpu_setup.md) |

For DDPG/TD3 stochastic recordings, saved exploration-noise settings are restored; `--exploration-noise` overrides them. Stochastic gSDE recordings restore the saved noise-reset schedule, defaulting to episode-only resets when absent.

`--episode-length`, `--scenario-kwargs`, and `--env-kwargs` can change the task for qualitative inspection. JSON overrides accept scalar/list settings. Run `swarmbots record --help` for all options.

## Custom policy factories

Use `--policy module:function` with an importable factory:

```bash
swarmbots record SwarmBots-Bridge-v0 --policy my_policy:make_policy --episodes 3 --device cuda
```

The factory receives the following keyword arguments and returns a [benchmark policy callable](api.md#policy-evaluation):

```python
import torch

def make_policy(
    *, benchmark_id: str, device: torch.device, seed: int, deterministic: bool,
) -> MyPolicy:
    return MyPolicy(benchmark_id, device=device, seed=seed, deterministic=deterministic)
```

Replace `MyPolicy` with the policy class. Configure evaluation mode, frozen normalization, and deterministic or stochastic actions in the factory. The returned callable receives `(observations, episode_starts)` and returns an action dictionary; only public observations reach it. `--stochastic` passes `deterministic=False` to the factory.

## Python helpers

For an included baseline, load the checkpoint directly:

```python
from swarmbots.learn import record_checkpoint

video_folder = record_checkpoint(
    "runs/first-mappo/models/model_1000000_steps_final.pt",
    "SwarmBots-WallEasy-v0",
    "mappo",
    video_folder="recordings/mappo",
    num_episodes=3,
    device="cuda",
)
print(video_folder)
```

Use the actual checkpoint path. Pass `model_scale`, `policy_kwargs`, `use_nop`, and `continuous_action_dist` as needed to match training.

For a policy already loaded in Python:

```python
from swarmbots import record_policy
from swarmbots.learn import as_benchmark_policy

video_folder = record_policy(
    as_benchmark_policy(trainer, deterministic=True),
    "SwarmBots-WallEasy-v0",
    video_folder="recordings/mappo",
    num_episodes=3,
    max_parallel_episodes=2,
    device="cuda",
)
```

Here `trainer` is the result of [training from Python](learning.md#train-a-policy). Create a fresh adapter for each recording and use the policy's device. For a custom callable, pass it directly instead of the adapter.

Both helpers return the output directory after closing the simulator and finishing the video writers. The Python API also accepts scenario configuration objects that cannot be represented by the CLI's JSON options.
