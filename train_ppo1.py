import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import imageio

from swarmbots.swarms_bot_env import SwarmBotsEnv
from swarmbots.scenarios.obstacle_dungeon_scenario import ObstacleDungeonScenario
from swarmbots.swarm.simple_swarm import SimpleSwarm

class FlattenMultiAgentWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        
        # Flatten observation space
        obs_shape = env.observation_space.shape
        # obs_shape is (n_agents, obs_per_agent)
        n_agents = obs_shape[0]
        obs_dim = obs_shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(n_agents * obs_dim,), 
            dtype=np.float32
        )
        
        # Flatten action space
        action_shape = env.action_space.shape
        # action_shape is (n_agents, act_per_agent)
        act_dim = action_shape[1]
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(n_agents * act_dim,), 
            dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs.flatten(), info
    
    def step(self, action):
        # Reshape action to (n_agents, act_dim)
        original_shape = self.env.action_space.shape
        reshaped_action = action.reshape(original_shape)
        
        obs, reward, terminated, truncated, info = self.env.step(reshaped_action)
        
        return obs.flatten(), reward, terminated, truncated, info

def make_env():
    rng = np.random.default_rng(42)
    swarm = SimpleSwarm(rng.integers(0, 10000000))
    scenario = ObstacleDungeonScenario(swarm=swarm, payload_type=None)
    env = SwarmBotsEnv(scenario=scenario, render_mode=None)
    env = FlattenMultiAgentWrapper(env)
    env = Monitor(env)
    return env

def train():
    vec_env = SubprocVecEnv([make_env] * 8)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=512,
        device="cpu",
        learning_rate=2e-5,
        target_kl=0.05
    )

    print("Starting training...")
    model.learn(total_timesteps=10_000_000)
    print("Training finished.")

    model.save("ppo_swarm_bots")
    vec_env.save("vecnormalize_swarm_bots.pkl")
    return model

def record(model_path="ppo_swarm_bots", vecnorm_path="vecnormalize_swarm_bots.pkl"):
    def make_eval_env():
        swarm = SimpleSwarm(42)
        scenario = ObstacleDungeonScenario(swarm=swarm, payload_type=None)
        env = SwarmBotsEnv(scenario=scenario, render_mode="rgb_array", width=640, height=480)
        env = FlattenMultiAgentWrapper(env)
        return env
    model = PPO.load(model_path)

    # Create a single-env VecEnv for evaluation
    eval_env = DummyVecEnv([make_eval_env])

    # Load VecNormalize with the saved stats, applied to this eval_env
    eval_env = VecNormalize.load(vecnorm_path, eval_env)

    # VERY IMPORTANT: don't keep updating stats during eval
    eval_env.training = False
    # And usually you don't want reward normalization in eval
    eval_env.norm_reward = False

    obs = eval_env.reset()
    images = []
    print("Recording rollout...")

    for _ in range(500):
        # obs is (1, obs_dim) here, which is fine
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(action)

        # Render from the underlying base env
        img = eval_env.envs[0].render()
        if img is not None:
            images.append(img)

        if dones[0]:
            break

    eval_env.close()

    if images:
        imageio.mimsave("swarm_bots_rollout_ppo.gif", images, fps=30)
        print("Saved swarm_bots_rollout_ppo.gif")
    else:
        print("No images captured.")

if __name__ == "__main__":
    train()
    record()

