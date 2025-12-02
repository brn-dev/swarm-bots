import gymnasium as gym
from gymnasium import spaces
import numpy as np
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

        # Create a single continuous action space for PPO
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
        # 'action' is a flat float array from PPO

        # 1. Split into actuator and connector parts
        act_part_flat = action[:self.act_dim_flat]
        conn_part_flat = action[self.act_dim_flat:]

        # 2. Reshape actuators (Continuous)
        # PPO outputs in [-1, 1] which matches the actuator space usually
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

def make_env():
    rng = np.random.default_rng(42)
    swarm = SimpleSwarmTetrahedronZX(connection_torquescale=10.0)
    scenario = ObstacleStreetScenario(swarm=swarm, payload_type=None, seed=rng.integers(0, 10000000))
    env = SwarmBotsEnv(scenario=scenario, render_mode=None)
    # Use the new wrapper
    env = HybridActionWrapper(env)
    env = Monitor(env)
    return env

def record(model_path="ppo_swarm_bots", vecnorm_path="vecnormalize_swarm_bots.pkl"):
    def make_eval_env():
        swarm = SimpleSwarmTetrahedronZX(connection_torquescale=10.0)
        scenario = ObstacleStreetScenario(swarm=swarm, payload_type=None, seed=42)
        env = SwarmBotsEnv(scenario=scenario, render_mode="rgb_array", width=640, height=480)
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
        imageio.mimsave("swarm_bots_rollout_ppo.gif", images, fps=30)
        print("Saved swarm_bots_rollout_ppo.gif")
    else:
        print("No images captured.")

vec_env = SubprocVecEnv([make_env] * 4)
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
model.learn(total_timesteps=20_000_000)
print("Training finished.")

model.save("ppo_swarm_bots")
vec_env.save("vecnormalize_swarm_bots.pkl")

record()

