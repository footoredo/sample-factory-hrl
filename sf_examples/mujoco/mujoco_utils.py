from typing import Optional

import numpy as np
import gym

from sample_factory.utils.utils import is_module_available
from sample_factory.algo.utils.hierarchical_utils import RecoveryRLEnv


def mujoco_available():
    return is_module_available("mujoco")


class MujocoSpec:
    def __init__(self, name, env_id):
        self.name = name
        self.env_id = env_id


MUJOCO_ENVS = [
    MujocoSpec("mujoco_hopper", "Hopper-v4"),
    MujocoSpec("mujoco_halfcheetah", "HalfCheetah-v4"),
    MujocoSpec("mujoco_humanoid", "Humanoid-v4"),
    MujocoSpec("mujoco_ant", "Ant-v4"),
    MujocoSpec("mujoco_standup", "HumanoidStandup-v4"),
    MujocoSpec("mujoco_doublependulum", "InvertedDoublePendulum-v4"),
    MujocoSpec("mujoco_pendulum", "InvertedPendulum-v4"),
    MujocoSpec("mujoco_reacher", "Reacher-v4"),
    MujocoSpec("mujoco_walker", "Walker2d-v4"),
    MujocoSpec("mujoco_pusher", "Pusher-v4"),
    MujocoSpec("mujoco_swimmer", "Swimmer-v4"),
]


def mujoco_env_by_name(name):
    for cfg in MUJOCO_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception("Unknown Mujoco env")


# class MultiRewardWrapper(gym.Wrapper):
#     @property
#     def num_rewards(self):
#         return 5
    
#     def step(self, action):
#         obs, reward, terminated, truncated, info = self.env.step(action)
#         # print(info.keys())
#         return obs, np.array([info['forward_reward'], -info['reward_ctrl'], np.abs(info['x_velocity']), np.abs(info['y_velocity']), info['distance_from_origin']]), terminated, truncated, info


class RecoveryWrapper(gym.Wrapper):
    @property
    def num_rewards(self):
        return 3
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # print(info.keys())
        # return obs, np.array([info['forward_reward'], info['reward_ctrl'] < -20, info['reward_ctrl']]), terminated, truncated, info
        return obs, np.array([info['forward_reward'], info['x_velocity'] > 2, -info['x_velocity']]), terminated, truncated, info


class DummyWrapper(gym.Wrapper):
    @property
    def num_rewards(self):
        return 1
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # print(info.keys())
        return obs, np.array([info['forward_reward']]), terminated, truncated, info


def make_mujoco_env(env_name, _cfg, _env_config, render_mode: Optional[str] = None, **kwargs):
    mujoco_spec = mujoco_env_by_name(env_name)
    env = gym.make(mujoco_spec.env_id, render_mode=render_mode)
    env = RecoveryWrapper(env)
    env = RecoveryRLEnv(env)
    return env
