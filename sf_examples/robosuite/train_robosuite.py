import sys
from typing import Any, Dict, Optional
from collections import deque

import joblib

import gymnasium as gym
import numpy as np
from torch import nn

from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
import robosuite.utils.transform_utils as T
from robosuite.wrappers.tamp_gated_wrapper import TAMPGatedWrapper

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sf_examples.robosuite.robosuite_params import add_robosuite_env_args, robosuite_override_defaults


class RobosuiteGymEnv(gym.Env):
    def __init__(self, env_name, controller="OSC_POSE", tamp_ratio=10, eval=True):
        options = {}
        options["env_name"] = env_name
        if "TwoArm" in options["env_name"]:
            # Choose env config and add it to options
            options["env_configuration"] = choose_multi_arm_config()

            # If chosen configuration was bimanual, the corresponding robot must be Baxter. Else, have user choose robots
            if options["env_configuration"] == 'bimanual':
                options["robots"] = 'Baxter'
            else:
                options["robots"] = []

                # Have user choose two robots
                print("A multiple single-arm configuration was chosen.\n")

                for i in range(2):
                    print("Please choose Robot {}...\n".format(i))
                    options["robots"].append(choose_robots(exclude_bimanual=True))

        # Else, we simply choose a single (single-armed) robot to instantiate in the environment
        else:
            # options["robots"] = choose_robots(exclude_bimanual=True)
            options["robots"] = "Panda"
            
        options["controller_configs"] = load_controller_config(default_controller=controller)
        
        env = suite.make(
            **options,
            has_renderer=False,
            has_offscreen_renderer=True,
            # ignore_done=True,
            use_camera_obs=False,
            control_freq=20,
            horizon=500,
            reward_shaping=False,
        )
        
        # wrap with TAMP-gated wrapper
        env = TAMPGatedWrapper(
            env=env,
            htamp_grasp_conditions=False,
            htamp_constraints_path=None,
            htamp_noise_level=0,
        )
        
        self._env = env
        self._succ_counter = 0
        self._num_resets = 0
        self._tamp_ratio = tamp_ratio
        self._state_maxlen = 1000
        self._state_queue = deque(maxlen=self._state_maxlen)
        self._eval = eval
        # self._state_queue = list()
        
        if not self._eval:
            init_states = joblib.load("/home/footoredo/playground/sample-factory/Stack_init_states_100.joblib")
            for init_state in init_states:
                self._state_queue.append(init_state)
        
    @property
    def observation_space(self):
        obs_spec = self._env.observation_spec()
        obs_dict = dict()
        for key, value in obs_spec.items():
            obs_dict[key] = gym.spaces.Box(-np.inf, np.inf, shape=value.shape, dtype=value.dtype)
        return gym.spaces.Dict(obs_dict)
    
    @property
    def action_space(self):
        low, high = self._env.action_spec
        return gym.spaces.Box(low[:7], high[:7])
    
    def _populate_states(self, max_new=None):
        self._env.set_use_tamp(True)
        num_new = 0
        while len(self._state_queue) < self._state_maxlen:
            self._env.reset()
            self._state_queue.append(self._env.get_state())
            num_new += 1
            if num_new >= max_new:
                break
        self._env.set_use_tamp(False)

    def reset(self, **kwargs):
        self._num_resets += 1
        if not self._eval:
            if self._num_resets % self._tamp_ratio == 0: # and len(self._state_queue) > 0:
                # self._state_queue.popleft()
                print("num_resets", self._num_resets)
                # self._populate_states(max_new=1)
            init_state_ind = self.np_random.integers(len(self._state_queue))
            print(init_state_ind, self._state_queue[init_state_ind].keys())
            self._env.set_use_tamp(False)
            obs = self._env.reset(init_state=self._state_queue[init_state_ind])
        else:
            obs = self._env.reset()
        self._succ_counter = 0
        return obs, {}

    def step(self, action):
        obs, reward, done, _ = self._env.step(action)
        
        terminated = truncated = False
        
        if reward > 0.5:
            reward *= np.power(2.0, self._succ_counter)
            self._succ_counter += 1
            if self._succ_counter == 5:
                terminated = True
                # reward += 10
        else:
            if self._succ_counter > 0:
                terminated = True
                # reward -= 0.5
        
        # if reward > 0.5:
        #     # terminated = True
        #     self._succ_counter += 1
            
        # if self._succ_counter >= (1 if self._eval else 1):
        #     terminated = True
        
        if done:
            terminated = True
            # if self._env.timestep < self._env.horizon:
            #     terminated = True
            # else:
            #     truncated = True

        return obs, reward, terminated, truncated, dict()

    def render(self):
        return self._env.sim.render(height=512, width=512, camera_name="agentview")[::-1]

def make_robosuite_env(env_name, cfg=None, _env_config=None, render_mode: Optional[str] = None):
    return RobosuiteGymEnv(env_name)

def make_robosuite_env_eval(env_name, cfg=None, _env_config=None, render_mode: Optional[str] = None):
    return RobosuiteGymEnv(env_name, eval=True)


def register_robosuite_components(evaluation=False):
    if evaluation:
        register_env("Stack", make_robosuite_env_eval)
    else:
        register_env("Stack", make_robosuite_env)
    
    
def parse_robosuite_args(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_robosuite_env_args(partial_cfg.env, parser)
    robosuite_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg
    

def main():  # pragma: no cover
    """Script entry point."""
    register_robosuite_components()
    cfg = parse_robosuite_args()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
