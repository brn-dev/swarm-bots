import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import imageio

from swarmbots.swarm.simple_swarm_tetrahedron_zx import SimpleSwarmTetrahedronZX
from swarmbots.swarm_bots_env import SwarmBotsEnv
from swarmbots.scenarios.obstacle_street_scenario import ObstacleStreetScenario



class HybridActionWrapper(gym.Wrapper):
    """
    Wraps a SwarmBotsEnv to flatten the observation space and
    convert the hybrid Dict action space into a single flat Box space.
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env

        # --- 1. Flatten Observation Space ---
        # Obs is a Dict with 'local_obs' and 'global_obs'
        self.local_obs_shape = env.observation_space['local_obs'].shape
        self.global_obs_shape = env.observation_space['global_obs'].shape
        
        self.local_obs_dim = np.prod(self.local_obs_shape)
        self.global_obs_dim = np.prod(self.global_obs_shape)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.local_obs_dim + self.global_obs_dim,),
            dtype=np.float32
        )

        # --- 2. Flatten Action Space (Hybrid -> Box) ---
        # The env action space is a Dict with 'actuators' (Box) and 'connectors' (MultiBinary)
        self.original_action_space = env.action_space

        # Get shapes
        self.act_space = self.original_action_space['actuators']
        self.conn_space = self.original_action_space['connectors']

        self.act_shape = self.act_space.shape
        self.conn_shape = self.conn_space.shape

        self.act_dim_flat = np.prod(self.act_shape)
        self.conn_dim_flat = np.prod(self.conn_shape)

        total_action_dim = self.act_dim_flat + self.conn_dim_flat

        # Create a single continuous action space for ppo
        # We use [-1, 1] range. For binary actions, >0 will be treated as 1.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(total_action_dim,),
            dtype=np.float32
        )

    def _flatten_obs(self, obs):
        return np.concatenate([
            obs['global_obs'],
            obs['local_obs'].flatten(),
        ])

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._flatten_obs(obs), info

    def step(self, action):
        # 'action' is a flat float array from ppo

        # 1. Split into actuator and connector parts
        act_part_flat = action[:self.act_dim_flat]
        conn_part_flat = action[self.act_dim_flat:]

        # 2. Reshape actuators (Continuous)
        # ppo outputs in [-1, 1] which matches the actuator space usually
        actuators = act_part_flat.reshape(self.act_shape)

        # 3. Reshape and Threshold connectors (Binary)
        # Treat positive values as 1 (active), negative/zero as 0 (inactive)
        connectors_continuous = conn_part_flat.reshape(self.conn_shape)
        connectors = (connectors_continuous > 0).astype(self.conn_space.dtype)

        # 4. Construct Dictionary Action
        dict_action = {
            'actuators': actuators,
            'connectors': connectors
        }

        obs, reward, terminated, truncated, info = self.env.step(dict_action)

        return self._flatten_obs(obs), reward, terminated, truncated, info

def make_base_env(seed: int, render_mode: str | None):
    swarm = SimpleSwarmTetrahedronZX(connection_torquescale=10.0)
    scenario = ObstacleStreetScenario(
        swarm, 
        payload_type=None, 
        payload_size=(0.2, 0.2, 0.2),
        payload_start_location_offset=(0, 0, 1), 
        seed=seed
    )
    return SwarmBotsEnv(scenario=scenario, render_mode=render_mode)
    

def make_env():
    rng = np.random.default_rng(42)
    env = make_base_env(seed=rng.integers(0, 10000000), render_mode=None)
    # Use the new wrapper
    env = HybridActionWrapper(env)
    env = Monitor(env)
    return env

def record(model_path, vecnorm_path, save_dir):
    def make_eval_env():
        env = make_base_env(42, 'rgb_array')
        env = HybridActionWrapper(env)
        return env

    model = PPO.load(model_path)
    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecNormalize.load(vecnorm_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    obs = eval_env.reset()
    images = []
    print("Recording rollout...")

    for _ in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(action)
        img = eval_env.envs[0].render()
        if img is not None:
            images.append(img)
        if dones[0]:
            break

    eval_env.close()

    if images:
        gif_path = os.path.join(save_dir, "swarm_bots_rollout_ppo.gif")
        imageio.mimsave(gif_path, images, fps=30)
        print(f"Saved {gif_path}")
    else:
        print("No images captured.")

if __name__ == '__main__':
    vec_env = SubprocVecEnv([make_env] * 4)

    # Create run directory
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving run outputs to {run_dir}")

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    continue_training = False
    if continue_training:
        print('Continuing training on existing policy')
        vec_env = VecNormalize.load('vecnormalize_swarm_bots.pkl', vec_env)
        model = PPO.load("ppo_swarm_bots", env=vec_env, device='cpu', learning_rate=1e-5)
    else:
        print('Creating new policy')
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
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

    model_save_path = os.path.join(run_dir, "ppo_swarm_bots")
    vecnorm_save_path = os.path.join(run_dir, "vecnormalize_swarm_bots.pkl")

    model.save(model_save_path)
    vec_env.save(vecnorm_save_path)

    record(model_save_path, vecnorm_save_path, run_dir)



