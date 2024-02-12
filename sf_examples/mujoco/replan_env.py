__credits__ = ["Kallinteris-Andreas"]

from typing import Dict, Tuple, Union

import numpy as np
import mujoco

import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


gym.envs.register(
    id="Pickplace-v0",
    entry_point="sf_examples.mujoco.replan_env:PickplaceEnv",
    max_episode_steps=500
)


DEFAULT_CAMERA_CONFIG = {
    "distance": 2.0,
}


class PickplaceEnv(MujocoEnv):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file: str = "/home/footoredo/playground/mujoco_mpc/build/mjpc/tasks/panda/pickplace/task.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        ctrl_cost_weight: float = 0.0,
        contact_cost_weight: float = 0.0,
        contact_force_range: Tuple[float, float] = (-1.0, 1.0),
        include_cfrc_ext_in_observation: bool = True,
        **kwargs,
    ):
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        
        self._contact_force_range = contact_force_range
        
        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,  # needs to be defined after
            default_camera_config=default_camera_config,
            camera_id=0,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.data.qpos.size + self.data.qvel.size
        obs_size += self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,
            "cfrc_ext": self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation,
        }

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def lift_height(self):
        return self.data.qpos[2] - 0.4301
    
    @property
    def lifted(self):
        return self.lift_height > 0.2
    
    @property
    def proximity(self):
        gripper_pos = self.data.site("pinch").xpos.copy()
        object_pos = self.data.qpos[:3].copy()
        dist = np.linalg.norm(gripper_pos - object_pos)
        return dist
    
    @property
    def gripper_joint(self):
        return self.data.joint("left_driver_joint").qpos[0]
    
    @property
    def gripper_displacement(self):
        return self.data.site("pinch").xpos.copy() - np.array([0.485437943, 1.20335101e-04, 0.538278488])

    def step(self, action):
        ctrl = np.zeros_like(action)
        ctrl[:3] = action[:3] * 0.05 + self.gripper_displacement
        ctrl[3:6] = 0.
        ctrl[6] = action[6]
        
        self.do_simulation(ctrl, self.frame_skip)
        
        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)
        terminated = self.lifted
        info = {
            "gripper_x": self.data.site("pinch").xpos[0],
            "gripper_y": self.data.site("pinch").xpos[1],
            "gripper_z": self.data.site("pinch").xpos[2],
            "object_x": self.data.qpos[0],
            "object_y": self.data.qpos[1],
            "object_z": self.data.qpos[2],
            "lift_height": self.lift_height,
            "gripper_joint": self.gripper_joint,
            **reward_info,
        }
        
        # print(action)
        # print(info)

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def _get_rew(self, action):
        # lift_reward = 1.0 if self.lifted else 0.0
        lift_reward = self.lift_height
        proximity_reward = np.exp(-self.proximity)
        rewards = lift_reward + proximity_reward
        
        # if self.proximity < 0.02:
        #     rewards += self.gripper_joint

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        costs = ctrl_cost + contact_cost

        reward = rewards - costs

        reward_info = {
            "reward_lift": lift_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_proximity": proximity_reward
        }

        return reward, reward_info

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._include_cfrc_ext_in_observation:
            contact_force = self.contact_forces[1:].flatten()
            return np.concatenate((position, velocity, contact_force))
        else:
            return np.concatenate((position, velocity))

    def reset_model(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        observation = self._get_obs()

        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }
        

if __name__ == "__main__":
    from PIL import Image
    env = PickplaceEnv(render_mode="rgb_array")
    env.reset()
    print(env.lift_height)
    print(env.observation_space)
    print(env.action_space)
    # action = np.random.random(env.action_space.shape)
    action = np.zeros(env.action_space.shape)
    action[-1] = 1.
    for _ in range(10):
        observation, reward, terminated, truncated, info = env.step(action)
    print(reward, terminated, info)
    im = Image.fromarray(env.render())
    im.save("obs.png")