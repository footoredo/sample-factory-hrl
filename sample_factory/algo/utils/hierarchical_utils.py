from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import copy

import gym
import numpy as np
import torch
from gym import Wrapper, spaces
from gym.core import ActType, ObsType
from torch import Tensor

from sample_factory.algo.utils.tensor_utils import dict_of_lists_cat
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.envs.create_env import create_env
from sample_factory.envs.env_utils import (
    RewardShapingInterface,
    TrainingInfoInterface,
    find_training_info_interface,
    find_wrapper_interface,
)
from sample_factory.utils.dicts import dict_of_lists_append, list_of_dicts_to_dict_of_lists
from sample_factory.utils.typing import Config

Actions = Any
ListActions = Sequence[Actions]
TensorActions = Tensor

SeqBools = Sequence[bool]

DictObservations = Dict[str, Any]
DictOfListsObservations = Dict[str, Sequence[Any]]
DictOfTensorObservations = Dict[str, Tensor]
ListObservations = Sequence[Any]
ListOfDictObservations = Sequence[DictObservations]


def recursive_copy(data):
    if isinstance(data, np.ndarray):
        return data.copy()
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().copy()
    elif type(data) is list:
        return [recursive_copy(x) for x in data]
    elif type(data) is dict:
        return {key: recursive_copy(value) for key, value in data.items()}
    else:
        return data


class RecoveryRLEnv(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # assert env.num_rewards == 2

        self.num_agents = 2
        self.is_multiagent = True

    def _replicate(self, item):
        return [recursive_copy(item) for _ in range(self.num_agents)]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._replicate(obs), self._replicate(info)
    
    def step(self, actions):
        # assume action[0] is the actual action.
        _obs, _reward, _terminated, _truncated, _info = self.env.step(actions[0])
        if _terminated or _truncated:
            _obs, _ = self.env.reset()
        return self._replicate(_obs), self._replicate(_reward), self._replicate(_terminated), \
            self._replicate(_truncated), self._replicate(_info)


class MetaController:
    def __init__(self, cfg: Config, env_info: EnvInfo):
        self.cfg = cfg
        self.env_info = env_info

        self.num_agents = 2

        self.recovery_weights = self.env_info.recovery_weights

        self.last_observation = None
        self.policy_outputs = [None for _ in range(self.num_agents)]
        self.ready = [False for _ in range(self.num_agents)]

        self.episode_length = 0
        self.exploration_countdown = int(1e5)
        self.total_steps = 0

        self.policy_idx = None

        self.selected = False

    def _select_policy(self):
        if all(self.ready):
            self.total_steps += 1
            denormed_values = self.policy_outputs[0]["denormalized_values"]
            # if isinstance(denormed_values, torch.Tensor):
            #     denormed_values = denormed_values.detach().cpu().numpy()
            safety = np.dot(self.recovery_weights, denormed_values)
            rnd = safety * 100 % 1
            # if safety > 0.5 or rnd > (0.8 + 0.2 * self.total_steps / self.exploration_countdown):
            # if safety > 0.5 and self.total_steps > 1e4:
            if safety > 0.5:
            # if safety > 1e9:
            # if safety * 100 % 1 > 0.95:
                self.policy_idx = 1
            else:
                self.policy_idx = 0
            for i in range(self.num_agents):
                self.ready[i] = False
            self.selected = True

    def set_observation(self, observation):
        self.last_observation = observation

    def set_policy_outputs(self, policy_idx, policy_outputs):
        self.policy_outputs[policy_idx] = recursive_copy(policy_outputs)
        # print(policy_outputs, policy_idx)
        self.ready[policy_idx] = True
        self._select_policy()

    def process_actions(self, actions):
        assert self.selected
        self.selected = False
        return [recursive_copy(actions[self.policy_idx])] * len(actions)
    
    def process_infos(self, infos, done):
        self.episode_length += 1

        for i in range(len(infos)):
            if i == self.policy_idx:
                infos[i]["actual_acting"] = True
            else:
                infos[i]["actual_acting"] = False
            infos[i]["actual_episode_length"] = self.episode_length

        if done:
            self.episode_length = 0
        
        return infos