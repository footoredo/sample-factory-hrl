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


class RecoveryRLEnv(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.num_rewards == 2

        self.num_agents = 2
        self.is_multiagent = True

        self.num_rewards = 1

    def _replicate(self, item):
        return [copy.deepcopy(item) for _ in range(self.num_agents)]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._replicate(obs), self._replicate(info)
    
    def step(self, actions):
        # assume action[0] is the actual action.
        _obs, _reward, _terminated, _truncated, _info = self.env.step(actions[0])
        if _terminated or _truncated:
            _obs, _info = self.env.reset()
        return self._replicate(_obs), _reward[:, None], self._replicate(_terminated), \
            self._replicate(_truncated), self._replicate(_info)
