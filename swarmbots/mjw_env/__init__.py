from swarmbots.utils.mujoco_bootstrap import configure_mujoco_gl_backend

configure_mujoco_gl_backend()

from swarmbots.mjw_env.mjw_swarm_bots_vector_env import MJWSwarmBotsVectorEnv

__all__ = ["MJWSwarmBotsVectorEnv"]
