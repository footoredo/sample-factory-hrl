from typing import Optional
import argparse
import sys
import os
import csv
from typing import Any, Dict, Optional

import gym
import time
import numpy as np
import random
import string
import pathlib
import torch
from torch import nn
from copy import deepcopy

import joblib
import csv


def preemptive_save(obj, save_path, type="torch"):
    # temp_path = save_path + ".tmp"
    save_path = str(save_path)
    extension = save_path.split('.')[-1]
    temp_path = save_path[:-len(extension)] + "tmp." + extension
    if type == "torch":
        torch.save(obj, temp_path)
    elif type == "numpy":
        np.save(temp_path, obj)
    elif type == "joblib":
        joblib.dump(obj, temp_path, compress=3)
    elif type == "csv":
        with open(temp_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(obj)
    else:
        raise NotImplementedError
    os.replace(temp_path, save_path)


from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.envs.env_utils import RewardShapingInterface, TrainingInfoInterface, register_env
from sample_factory.train import run_rl
from sample_factory.model.encoder import Encoder
from sample_factory.model.model_utils import nonlinearity
from sample_factory.utils.typing import Config, ObsSpace

from crafter.env import Env as CrafterEnv


CRAFTER_KWARGS = dict(
    size=(84, 84),
)


class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to channels x weight x height
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation):
        return np.transpose(observation, axes=(2, 0, 1))


class NewGymCrafter(gym.Env):
    def __init__(self):
        super(NewGymCrafter, self).__init__()
        self.env = CrafterEnv(**CRAFTER_KWARGS)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, **kwargs):
        return self.env.reset(), {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        return obs, reward, done, False, info

    def render(self):
        pass


class CustomEnv(gym.Env, TrainingInfoInterface, RewardShapingInterface):
    def __init__(self, full_env_name, cfg, env_config, render_mode: Optional[str] = None):
        TrainingInfoInterface.__init__(self)
        self.name = full_env_name  # optional
        env = NewGymCrafter()
        env = ImageToPyTorch(env)
        self.env = env

        self.num_rewards = 2

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.reward_shaping = dict()

        self.render_mode = render_mode

        self.avg_health_lost_count = 0
        self.pen = 1

    def set_training_info(self, training_info):
        # print(training_info['policy_avg_stats'].keys())
        if 'health_lost_count' in training_info['policy_avg_stats']:
            self.avg_health_lost_count = training_info['policy_avg_stats']['health_lost_count']
            # print('avg_health_lost_count', self.avg_health_lost_count, np.log(self.avg_health_lost_count + 1))

    def reset(self, **kwargs):
        self.health_lost_count = 0
        self.last_health = 9
        self.episode_reward = 0
        return self.env.reset()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += reward
        health = info['inventory']['health']
        pen = 0
        if health < self.last_health:
            # reward -= self.pen
            pen = 1
            reward -= 0
            self.health_lost_count += 1
        self.last_health = health
        if terminated or truncated:
            episode_extra_stats = {
                f'unlock_{key}': int(value > 0) for key, value in info['achievements'].items()
            }
            episode_extra_stats['health_lost_count'] = self.health_lost_count
            episode_extra_stats['episode_reward'] = self.episode_reward
            episode_extra_stats['pen'] = self.pen
            info['episode_extra_stats'] = episode_extra_stats
            self.health_lost_count = 0
            self.last_health = 9
            self.episode_reward = 0
            self.pen += 0.02 * (self.avg_health_lost_count - 1)
            self.pen = min(max(self.pen, 0), 5)
        return obs, np.array([reward, pen], dtype=np.float32), terminated, truncated, info

    def render(self):
        pass

    def get_default_reward_shaping(self) -> Dict[str, Any]:
        return self.reward_shaping

    def set_reward_shaping(self, reward_shaping: Dict[str, Any], agent_idx) -> None:
        self.reward_shaping = reward_shaping


class CustomEncoder(Encoder):
    """Just an example of how to use a custom model component."""

    def __init__(self, cfg, obs_space):
        super().__init__(cfg)

        obs_shape = obs_space["obs"].shape

        conv_layers = [
            nn.Conv2d(obs_shape[0], 32, 8, stride=4),
            nonlinearity(cfg),
            nn.Conv2d(32, 64, 4, stride=2),
            nonlinearity(cfg),
            nn.Conv2d(64, 64, 3, stride=1),
            nonlinearity(cfg),
        ]

        self.conv_head = nn.Sequential(*conv_layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape)

    def forward(self, obs_dict):
        # we always work with dictionary observations. Primary observation is available with the key 'obs'
        main_obs = obs_dict["obs"]

        x = self.conv_head(main_obs)
        x = x.view(-1, self.conv_head_out_size)
        return x

    def get_out_size(self) -> int:
        return self.conv_head_out_size


def make_custom_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    """Factory function as required by the API."""
    return CustomEncoder(cfg, obs_space)


def make_custom_env_func(full_env_name, cfg=None, env_config=None, render_mode: Optional[str] = None):
    # print(env_config)
    return CustomEnv(full_env_name, cfg, env_config, render_mode=render_mode)

def register_custom_components():
    register_env("crafter", make_custom_env_func)
    global_model_factory().register_encoder_factory(make_custom_encoder)

def add_custom_env_args(_env, p: argparse.ArgumentParser, evaluation=False):
    # You can extend the command line arguments here
    p.add_argument("--crafter_monitor", default=False, action="store_true")
    p.add_argument("--save_replay", default=False, action="store_true")

def custom_env_override_defaults(_env, parser):
    # Modify the default arguments when using this env.
    # These can still be changed from the command line. See configuration guide for more details.
    parser.set_defaults(
        obs_scale=255.0,
        gamma=0.99,
        learning_rate=0.00025,
        lr_schedule="linear_decay",
        adam_eps=1e-5,  
    )

def parse_args(argv=None, evaluation=False):
    # parse the command line arguments to build
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_custom_env_args(partial_cfg.env, parser, evaluation=evaluation)
    custom_env_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg

def main():
    """Script entry point."""
    register_custom_components()
    cfg = parse_args()

    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())