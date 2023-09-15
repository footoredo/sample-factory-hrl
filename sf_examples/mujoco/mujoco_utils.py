from typing import Optional

import numpy as np
import gym
from gym.spaces import Box

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


OBSTACLE_SIZE = 4


class RecoveryWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = 3 + self.env.observation_space.shape[0]
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )
        self.obstacle = None
    
    def _add_obstacle_obs(self, obs, info):
        if self.obstacle is None:
            obstacle_obs = np.array([0, 0, 0])
        else:
            x_pos = info['x_position']
            y_pos = info['y_position']
            obstacle_obs = np.array([1, self.obstacle[0] - x_pos, self.obstacle[1] - y_pos])
        return np.concatenate((obs, obstacle_obs), 0)
    
    def _add_obstacle_info(self, info):
        info['has_obstacle'] = self.obstacle is not None
        if self.obstacle is not None:
            x_pos = info['x_position']
            y_pos = info['y_position']
            info['obstacle_x'] = self.obstacle[0]
            info['obstacle_y'] = self.obstacle[1]
            info['obstacle_distance'] = np.sqrt(np.square(self.obstacle[0] - x_pos) + np.square(self.obstacle[1] - y_pos))
        return info
    
    def _update_obstacle(self, info):
        x_pos = info['x_position']
        y_pos = info['y_position']
        rnd = (x_pos * 100) % 1
        if self.obstacle is None:
            if rnd < 0.1:
                rnd2 = (rnd * 10000) % 1
                rnd3 = (rnd2 * 10000) % 1
                obstacle_dx = rnd2 * 5 + 5
                obstacle_dy = rnd3 * 8 - 4
                self.obstacle = [x_pos + obstacle_dx, y_pos + obstacle_dy]
        elif x_pos > self.obstacle[0] + 5:
            self.obstacle = None

    def _check_collision(self, info):
        if self.obstacle is None:
            return False, 0
        else:
            obstacle_pos = np.array(self.obstacle)
            agent_pos = np.array([info['x_position'], info['y_position']])
            distance = np.sqrt(np.square(agent_pos - obstacle_pos).sum())
            return distance <= OBSTACLE_SIZE, distance
    
    def reset(self, **kargs):
        obs, info = self.env.reset(**kargs)
        obs[0] = 0  # mask x posistion
        self.obstacle = None
        return self._add_obstacle_obs(obs, info), self._add_obstacle_info(info)
    
    @property
    def reward_labels(self):
        return ["progress", "collision", "y_limit", "collision_reward", "y_limit_reward"]
    
    @property
    def reward_weights(self):
        return [[1., 0., 0., 0., 0.], [0., 0., 0., 3.0, 1.0]]
    
    @property
    def recovery_weights(self):
        return [[0., 0.5, 0.5, 0., 0.]]
    
    @property
    def suppression_weights(self):
        return [[0., 1., 0., 0., 0.], [0., 1., 0., 0., 0.]]
    
    @property
    def num_rewards(self):
        return 5
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._update_obstacle(info)

        # print(info.keys())
        z_pos = obs[2]
        health_pen = np.power(2 * (z_pos - 0.6), 4)
        health_rew = 1 - min(health_pen, 1)
        y_pos = info['y_position']
        y_pen = np.power(0.2 * y_pos, 4)
        y_rew = 5 - min(y_pen, 5)
        # return obs, np.array([info['forward_reward'], info['reward_ctrl'] < -20, info['reward_ctrl']]), terminated, truncated, info
        # return obs, np.array([info['forward_reward'], info['x_velocity'] > 2, -info['x_velocity']]), terminated, truncated, info
        # return obs, np.array([info['forward_reward'], int(terminated), -health_pen]), terminated, truncated, info
        obs[0] = 0  # mask x posistion

        y_limit_violation = y_pos < -5 or y_pos > 5

        collision, distance = self._check_collision(info)
        # collision_reward = 0 if collision else 5 + distance
        if self.obstacle is None:
            collision_reward = 0.
        else:
            collision_reward = np.log(max(distance, 0.1) / 5)
            # collision_reward = np.log(max(distance, 0.1) / 5) - 5 * int(distance <= OBSTACLE_SIZE)
            # collision_reward = distance - 5 * int(distance <= OBSTACLE_SIZE)

        info['collision'] = collision
        # if collision:
        #     self.obstacle = None
        
        # return self._add_obstacle_obs(obs), np.array([reward, y_pos < -5 or y_pos > 5, y_rew]), terminated, truncated, self._add_obstacle_info(info)
        return self._add_obstacle_obs(obs, info), np.array([info['forward_reward'], collision, y_limit_violation, collision_reward, y_rew]), terminated, truncated, self._add_obstacle_info(info)


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
    env = gym.make(mujoco_spec.env_id, render_mode=render_mode, exclude_current_positions_from_observation=False)
    env = RecoveryWrapper(env)
    env = RecoveryRLEnv(env)
    return env
