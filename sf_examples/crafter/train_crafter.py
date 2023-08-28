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
    render_centering=False, 
    health_reward_coef=0.0, 
    immortal=True, 
    idle_death=100,
    add_meta=True
)

CRAFTER_ORIGINAL_KWARGS = dict(
    size=(84, 84),
    render_centering=False, 
    vanila=True,
    add_meta=True
)


class ReplayRecorder(gym.Wrapper):
    def __init__(self, env, env_id, save_dir, freq=1.0, batch_size=1000):
        self.env = env
        self.id = env_id
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = pathlib.Path(save_dir)
        self.freq = freq
        self.batch_size = batch_size
        self.counter = 0
        self.save_counter = 0
        self._check_sample()
        self._reset_buffer()

    def _check_sample(self):
        self.sample = np.random.random() < self.freq

    def _reset_buffer(self):
        self.buffer = {
            "action": [],
            "reward": [],
            "terminated": [],
            "truncated": [],
            "meta": []
        }
        if isinstance(self.observation_space, gym.spaces.Dict):
            for key in self.observation_space.keys():
                self.buffer[f"observation_{key}"] = []
        else:
            self.buffer["observation"] = []

    def _save_buffer(self):
        for key, datas in self.buffer.items():
            if isinstance(datas[0], np.ndarray):
                batched_data = np.stack(datas, axis=0)
                save_type = "numpy"
            elif isinstance(datas[0], dict):
                batched_data = datas
                save_type = "joblib"
            else:
                batched_data = np.array(datas)
                save_type = "numpy"
            save_type = "joblib"
            suffix = "npy" if save_type == "numpy" else "obj"
            preemptive_save(batched_data, self.save_dir / f"{key}.{self.id}.{self.save_counter}.{suffix}", type=save_type)
        self.save_counter += 1
        self._reset_buffer()

    def reset(self):
        self._check_sample()
        observation, info = self.env.reset()
        self.last_observation = deepcopy(observation)
        self.last_info = deepcopy(info)
        return observation, info
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        if self.sample:
            self.buffer["action"].append(action)
            self.buffer["reward"].append(reward)
            self.buffer["terminated"].append(terminated)
            self.buffer["truncated"].append(truncated)
            if isinstance(self.last_observation, dict):
                for key, value in self.last_observation.items():
                    self.buffer[f"observation_{key}"].append(deepcopy(value))
            else:
                self.buffer["observation"].append(deepcopy(self.last_observation))
            if "meta" in self.last_info:
                self.buffer["meta"].append(deepcopy(self.last_info['meta']))
            self.counter += 1
            if self.counter % self.batch_size == 0:
                self._save_buffer()
        self.last_observation = deepcopy(observation)
        self.last_info = deepcopy(info)

        if truncated:
            self._check_sample()

        return observation, reward, terminated, truncated, info


class CrafterMonitorWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, monitor_id, save_path, save_freq=10):
        super(CrafterMonitorWrapper, self).__init__(env)
        self.monitor_id = monitor_id
        os.makedirs(save_path, exist_ok=True)
        self.save_path = str(save_path) + f"/{monitor_id}_stats.csv"
        self.save_freq = max(1, save_freq)
        self.episode_step = 0
        self.last_save_timestamp = 0
        self.achv_names = []
        self.rows = []
        if os.path.exists(self.save_path):
            with open(self.save_path) as f:
                reader = csv.reader(f)
                header = reader.__next__()
                assert header[0] == "timestamp"
                assert header[1] == "episode_steps"
                self.achv_names = header[2:]
                for row in reader:
                    self.rows.append(row)
        self.counter = dict()
        self.save()

    def save(self):
        rows = [["timestamp", "episode_steps"] + self.achv_names] + self.rows
        preemptive_save(rows, self.save_path, type="csv")

    def add_cols(self, achvs):
        self.achv_names += achvs
        for i in range(len(self.rows)):
            self.rows[i] += [0] * len(achvs)

    def add_row(self):
        new_cols = []
        for name in self.counter.keys():
            if name not in self.achv_names:
                new_cols.append(name)
        if len(new_cols) > 0:
            self.add_cols(new_cols)
        stats = [0] * len(self.achv_names)
        for name, count in self.counter.items():
            stats[self.achv_names.index(name)] = count
        row = [time.time(), self.episode_step] + stats
        self.rows.append(row)
        self.counter = dict()

    def reset(self):
        self.episode_step = 0
        return self.env.reset()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.episode_step += 1
        # self.step_count += 1
        if truncated:
            achv = info["achievements"]
            for name, count in achv.items():
                if count >= 1:
                    if name not in self.counter:
                        self.counter[name] = 0
                    self.counter[name] += 1
            self.add_row()
            self.episode_step = 0
            if time.time() > self.last_save_timestamp + self.save_freq:
                self.save()
                self.last_save_timestamp = time.time()
        return observation, reward, terminated, truncated, info


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
        self.env = CrafterEnv(**CRAFTER_ORIGINAL_KWARGS)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, **kwargs):
        return self.env.reset(), {"meta": self.env.export()}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        return obs, reward, done, done, info

    def render(self):
        pass


class CustomEnv(gym.Env, TrainingInfoInterface, RewardShapingInterface):
    def __init__(self, full_env_name, cfg, env_config, render_mode: Optional[str] = None):
        TrainingInfoInterface.__init__(self)
        self.name = full_env_name  # optional
        env = NewGymCrafter()
        env = ImageToPyTorch(env)
        # if env_config is not None:  # not tmp_env
        #     if cfg.crafter_monitor:
        #         env = CrafterMonitorWrapper(env, env_config.env_id, str(cfg.train_dir) + '/' + str(cfg.experiment) + "/crafter_monitor", 30)
        #     if cfg.save_replay:
        #         env = ReplayRecorder(env, env_config.env_id, str(cfg.train_dir) + '/' + str(cfg.experiment) + "/replay", freq=1e7 / (2.5e8))
        self.env = env

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.reward_shaping = dict()

        self.render_mode = render_mode

    def reset(self, **kwargs):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if done:
            info['episode_extra_stats'] = {
                
            }

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