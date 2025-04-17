import math
import pathlib

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from so100_mujoco_rl.envs.utils import joints_from_model, Joint, MUJOCO_SO100_PREFIX


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 1.25,
    "elevation": -25,
    "azimuth": 45,
}


"""
Most of the Env code will be common across different scenarios as the so100
doesn't change. The base class includes all this common code.
"""
class So100BaseEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    def __init__(self, env_filename: str, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)

        model = mujoco.MjModel.from_xml_path(str(pathlib.Path(__file__).parent.joinpath(env_filename)))
        self.joints = joints_from_model(model)
        observation_space = self.get_observation_space()

        MujocoEnv.__init__(
            self,
            str(pathlib.Path(__file__).parent.joinpath(env_filename)),
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            width=800,
            height=800,
            **kwargs,
        )

        self.loop_count = 0
        self.start_distance = None

    def get_observation_space(self):
        mins = [self.joints[i].range[0] for i in range(len(self.joints))]
        maxs = [self.joints[i].range[1] for i in range(len(self.joints))]

        observation_space = Box(
            np.array([*mins, -1.0, -1.0, -1.0]),
            np.array([*maxs, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        return observation_space

    def _set_action_space(self):
        self.action_space = Box(
            np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        return self.action_space

    def render(self):
        if self.mujoco_renderer.viewer is not None and self.render_mode == 'human':
            # in this case it's a Gymnasium Mujoco Viewer
            self.mujoco_renderer.viewer.add_overlay(
                gridpos=mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                text1="A value",
                text2="test"
            )
        return super().render()

    def get_joint_angles(self) -> list[float]:
        angles = [
            self.data.joint(MUJOCO_SO100_PREFIX + joint.name).qpos[0]
            for joint in self.joints
        ]
        return angles
    
    def get_end_effector_pos(self) -> list[float]:
        # if we just use xpos, then we get the orgin of the fixed jaw instead of
        # the tip of the fixed jaw
        # below we add a 100mm offset in the -y direction, and convert that into
        # world coordinates
        xmat = self.data.body(MUJOCO_SO100_PREFIX + 'Fixed_Jaw').xmat.reshape(3, 3)
        world_position = self.data.body(MUJOCO_SO100_PREFIX + 'Fixed_Jaw').xpos
        local_point = np.array([0,-0.1,0], dtype=np.float32)
        global_point = world_position + xmat @ local_point
        return global_point

    def get_block_pos(self) -> list[float]:
        pos = self.data.body('block_a').xpos
        return pos

    def _get_reward(self):
        reward = 0.5

        block_pos = self.get_block_pos()
        end_pos = self.get_end_effector_pos()
        # print(f"block_pos: {block_pos}")
        # print(f"end_pos: {end_pos}")
        distance = math.sqrt(
            (block_pos[0] - end_pos[0]) ** 2 +
            (block_pos[1] - end_pos[1]) ** 2 +
            (block_pos[2] - end_pos[2]) ** 2
        )

        if self.start_distance is None and self.loop_count > 1:
            initial_block_pos = self.get_block_pos()
            initial_end_pos = self.get_end_effector_pos()
            
            distance = math.sqrt(
                (initial_block_pos[0] - initial_end_pos[0]) ** 2 +
                (initial_block_pos[1] - initial_end_pos[1]) ** 2 +
                (initial_block_pos[2] - initial_end_pos[2]) ** 2
            )
            print("Setting start distance")
            print(f"start_distance: {distance}")
            print(f"init block_pos: {initial_block_pos}")
            print(f"init end_pos: {initial_end_pos}")
            print(dir(self.data.body(MUJOCO_SO100_PREFIX + 'Fixed_Jaw')))
            print(self.data.body(MUJOCO_SO100_PREFIX + 'Fixed_Jaw').xmat)
            print(type(self.data.body(MUJOCO_SO100_PREFIX + 'Fixed_Jaw').xmat))
            print(self.data.body(MUJOCO_SO100_PREFIX + 'Fixed_Jaw').xpos)

            # p = np.array([0,0,0], dtype=np.float32)
            # t = self.data.body(MUJOCO_SO100_PREFIX + 'Fixed_Jaw').xmat @ p
            # print(f"t: {t}")

            self.start_distance = distance

        if self.start_distance is not None:

            delta_distance_norm = (self.start_distance - distance) / self.start_distance
            reward += delta_distance_norm * 0.5

        # print("reward: ", reward)

        return reward

    def _get_obs(self):
        self.loop_count += 1

        block_pos = self.get_block_pos()
        end_pos = self.get_end_effector_pos()

        dx = block_pos[0] - end_pos[0]
        dy = block_pos[1] - end_pos[1]
        dz = block_pos[2] - end_pos[2]

        return np.array(
            [
                *self.get_joint_angles(),
                dx,
                dy,
                dz,
            ],
            dtype=np.float32
        ).ravel()

