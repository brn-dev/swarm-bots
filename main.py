import mujoco

from swarmbots import SimpleSwarmTetrahedronZX
from swarmbots import SwarmBotsEnv
from swarmbots import ObstacleStreetScenario

import numpy as np


swarm = SimpleSwarmTetrahedronZX(connection_torquescale=0.01)
scenario = ObstacleStreetScenario(swarm, payload_type='sphere', payload_start_location_offset=(0, 0, 1), seed=42)

opt = mujoco.MjvOption()

env = SwarmBotsEnv(
    scenario=scenario,
    render_mode="rgb_array",
    width=640,
    height=480,
    camera=0,
    scene_option=opt,
    action_repeat=15,
)

rng = np.random.default_rng()


frames = []
for i in range(3):
    done = False
    obs, info = env.reset()

    while not done:
        # Random action
        action = env.action_space.sample()
        # if len(frames) % 3 == 0:
        #     action['connectors'] = True
        obs, reward, terminated, truncated, info = env.step(action)

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        done = terminated or truncated

        # if info:
        #     print('err ' + str(mj_env.data.time))


        # if len(frames) % 50 == 0:
        #     mj_env.data.eq_active[:] = 0
        #     mj_env.data.eq_active[rng.integers(low=0, high=len(mj_env.data.eq_active))] = 1

    print(f"Recorded {len(frames)} frames")
    for _ in range(15):
        frames.append(np.zeros_like(frames[0]))
