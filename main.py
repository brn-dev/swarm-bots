import mujoco
from swarmbots.swarm_bots_env import SwarmBotsEnv
from swarmbots.swarm.simple_swarm import SimpleSwarm
from swarmbots.scenarios.obstacle_dungeon_scenario import ObstacleDungeonScenario
from rendering import display_video

import numpy as np


swarm = SimpleSwarm(connection_torquescale=0.01)
scenario = ObstacleDungeonScenario(swarm, payload_type=None, seed=42)

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
        obs, reward, terminated, truncated, info = env.step(action)

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        done = terminated or truncated

        if info:
            print('err ' + str(env.data.time))


        # if len(frames) % 50 == 0:
        #     env.data.eq_active[:] = 0
        #     env.data.eq_active[rng.integers(low=0, high=len(env.data.eq_active))] = 1

    print(f"Recorded {len(frames)} frames")
    for _ in range(15):
        frames.append(np.zeros_like(frames[0]))
