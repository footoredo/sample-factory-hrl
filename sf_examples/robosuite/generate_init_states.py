import sys
from typing import Any, Dict, Optional
from collections import deque

import joblib

from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
import robosuite.utils.transform_utils as T
from robosuite.wrappers.tamp_gated_wrapper import TAMPGatedWrapper


def generate(env_name, num_inits):
    controller="OSC_POSE"
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
    
    init_states = []
    for _ in range(num_inits):
        env.reset()
        init_states.append(env.get_state())
        
    joblib.dump(init_states, f"/home/footoredo/playground/sample-factory/{env_name}_init_states.joblib")
    

if __name__ == "__main__":
    generate("Stack", 100)
