from swarmbots.envs.swarm_bot_env import SwarmBotEnv
from swarmbots.swarm.simple_swarm import SimpleSwarm
from swarmbots.scenarios.obstacle_dungeon_scenario import ObstacleDungeonScenario
from rendering import display_video

# Initialize swarm and scenario
swarm = SimpleSwarm()
scenario = ObstacleDungeonScenario()

# Initialize environment
env = SwarmBotEnv(
    swarm=swarm,
    scenario=scenario,
    render_mode="rgb_array",
    width=640,
    height=480,
)

obs, info = env.reset()

frames = []
FRAMERATE = 30
done = False

while not done:
    # Random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    # Record frame if it's time
    if len(frames) < env.data.time * FRAMERATE:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

    done = terminated or truncated

env.close()
print(f"Recorded {len(frames)} frames")
display_video(frames, FRAMERATE)