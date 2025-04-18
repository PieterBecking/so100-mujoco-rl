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
        self.last_reward = 0.0
        self.last_distance = None
        self.last_joint_angles = None

    def get_observation_space(self):
        mins = [self.joints[i].range[0] for i in range(len(self.joints))]
        maxs = [self.joints[i].range[1] for i in range(len(self.joints))]

        more_mins = [-1.0] * 12
        more_maxs = [-1.0] * 12
        observation_space = Box(
            np.array([*more_mins, *mins, -1.0, -1.0, -1.0]),
            np.array([*more_maxs, *maxs, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # observation_space = Box(
        #     np.array([*mins, -1.0, -1.0, -1.0]),
        #     np.array([*maxs, 1.0, 1.0, 1.0]),
        #     dtype=np.float32
        # )
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
                text1="Reward",
                text2=f"{self.last_reward:.3f}",
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

        joint_angles = self.get_joint_angles()
        block_pos = self.get_block_pos()
        end_pos = self.get_end_effector_pos()
        # print(f"block_pos: {block_pos}")
        # print(f"end_pos: {end_pos}")
        distance = math.sqrt(
            (block_pos[0] - end_pos[0]) ** 2 +
            (block_pos[1] - end_pos[1]) ** 2 +
            (block_pos[2] - end_pos[2]) ** 2
        )

        if block_pos[1] < -0.1:
            # then the block is in front of the robot
            # so the second joint, pitch, should be greater than -0.5 * pi
            if self.last_joint_angles is not None and joint_angles[1] < -0.5 * math.pi:
                pitch = joint_angles[1]
                last_pitch = self.last_joint_angles[1]
                delta_pitch = pitch - last_pitch
                # print(f"delta_pitch: {delta_pitch * 100}")
                reward += (delta_pitch * 100)


        if self.start_distance is None and self.loop_count > 1:
            self.start_distance = distance

        if self.start_distance is not None:
            delta_distance_norm = (self.start_distance - distance) / self.start_distance
            reward += delta_distance_norm * 0.5

        # if self.last_distance is not None:
        #     delta_distance = self.last_distance - distance
        #     if delta_distance > 0:
        #         reward += delta_distance * 500
        #     else:
        #         reward -= delta_distance * 500
        self.last_distance = distance

        # print("reward: ", reward)
        self.last_reward = reward
        self.last_joint_angles = joint_angles
        return reward

    def _get_obs(self):
        self.loop_count += 1

        block_pos = self.get_block_pos()
        end_pos = self.get_end_effector_pos()
        # print(f"block_pos: {block_pos}")
        # print(f"end_pos: {end_pos}")

        dx = block_pos[0] - end_pos[0]
        dy = block_pos[1] - end_pos[1]
        dz = block_pos[2] - end_pos[2]

        joint_angles = self.get_joint_angles()

        # for i, joint in enumerate(self.joints):
        #     ja = joint_angles[i]
        #     if ja > joint.range[1] or ja < joint.range[0]:
        #         print(f"Joint {joint.name} out of range: {ja} ({joint.range[0]}, {joint.range[1]})")

        return np.array(
            [
                *np.cos(joint_angles),
                *np.sin(joint_angles),
                *joint_angles,
                dx,
                dy,
                dz,
            ],
            dtype=np.float32
        ).ravel()

