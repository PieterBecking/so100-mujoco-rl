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
        "render_fps": 31,
    }

    def __init__(self, env_filename: str, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)

        model = mujoco.MjModel.from_xml_path(str(pathlib.Path(__file__).parent.joinpath(env_filename)))
        self.joints = joints_from_model(model)
        observation_space = self.get_observation_space()

        MujocoEnv.__init__(
            self,
            str(pathlib.Path(__file__).parent.joinpath(env_filename)),
            16,
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
        self.last_end_pos = None
        self.last_wrist_pos = None

        self.reward_components = {}

    def get_observation_space(self):
        mins = [self.joints[i].range[0] for i in range(len(self.joints))]
        maxs = [self.joints[i].range[1] for i in range(len(self.joints))]

        xyz_mins = [-0.5, -0.5, -0.5]
        xyz_maxs = [0.5, 0.5, 0.5]

        observation_space = Box(
            np.array([*mins, -1.0, -1.0, -1.0, *xyz_mins, *xyz_mins]),
            np.array([*maxs, 1.0, 1.0, 1.0, *xyz_maxs, *xyz_maxs]),
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
                text1="Reward",
                text2=f"{self.last_reward:.3f}",
            )
            for key, value in self.reward_components.items():
                if isinstance(value, (float, np.floating)):
                    val_s = f"{value:.3f}"
                elif isinstance(value, (bool, int)):
                    val_s = f"{value}"
                else:
                    val_s = str(value)
                self.mujoco_renderer.viewer.add_overlay(
                    gridpos=mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                    text1=key,
                    text2=val_s,
                )
        return super().render()

    def get_joint_angles(self) -> list[float]:
        angles = [
            self.data.joint(MUJOCO_SO100_PREFIX + joint.name).qpos[0]
            for joint in self.joints
        ]
        return angles

    def get_wrist_pos(self) -> list[float]:
        world_position = self.data.body(MUJOCO_SO100_PREFIX + 'Wrist_Pitch_Roll').xpos
        return world_position

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

    def get_block_to_end_distance(self) -> float:
        """ Distance between the block and the end effector """
        block_pos = self.get_block_pos()
        end_pos = self.get_end_effector_pos()
        distance = math.sqrt(
            (block_pos[0] - end_pos[0]) ** 2 +
            (block_pos[1] - end_pos[1]) ** 2 +
            (block_pos[2] - end_pos[2]) ** 2
        )
        return distance

    def _get_joint_reward(self) -> float:
        reward = 0.0
        joint_angles = self.get_joint_angles()
        for i, joint in enumerate(self.joints):
            joint_range = joint.range
            joint_angle = joint_angles[i]
            reward += self._calculate_joint_penalty(joint_angle, joint_range)
        return reward

    def _calculate_joint_penalty(self, joint_angle: float, joint_range: tuple[float, float]) -> float:
        penalty = 0.0
        lower_threshold = joint_range[0] + 0.05 * (joint_range[1] - joint_range[0])
        upper_threshold = joint_range[1] - 0.05 * (joint_range[1] - joint_range[0])

        if joint_angle < lower_threshold:
            penalty -= (lower_threshold - joint_angle) * 10.0
        elif joint_angle > upper_threshold:
            penalty -= (joint_angle - upper_threshold) * 10.0

        return penalty

    def _get_reward(self):
        reward = 0.0

        joint_angles = self.get_joint_angles()
        block_pos = self.get_block_pos()
        end_pos = self.get_end_effector_pos()
        wrist_pos = self.get_wrist_pos()
        # print(f"block_pos: {block_pos}")
        # print(f"end_pos: {end_pos}")
        distance = math.sqrt(
            (block_pos[0] - end_pos[0]) ** 2 +
            (block_pos[1] - end_pos[1]) ** 2 +
            (block_pos[2] - end_pos[2]) ** 2
        )

        for k in self.reward_components.keys():
            self.reward_components[k] = 0.0

        if block_pos[1] < -0.1:
            # then the block is in front of the robot
            # so the second joint, pitch, should be greater than -0.5 * pi
            pitch = joint_angles[1]
            if self.last_joint_angles is not None and pitch < -0.5 * math.pi:
                pitch_reward = (pitch + 0.5 * math.pi) * 0.7
                reward += pitch_reward
                self.reward_components['rew pitch'] = pitch_reward

        if self.last_end_pos is not None:
            if end_pos[2] < 0.02:
                end_pos_z_reward = (end_pos[2] - 0.02) * 20.0
                reward += end_pos_z_reward
                self.reward_components['rew end pos z'] = end_pos_z_reward

        if self.last_wrist_pos is not None:
            if wrist_pos[2] < 0.04:
                wrist_pos_z_reward = (wrist_pos[2] - 0.04) * 10.0
                wrist_pos_z_reward = np.clip(wrist_pos_z_reward, -0.4, 0.4)
                reward += wrist_pos_z_reward
                self.reward_components['rew wrist pos z'] = wrist_pos_z_reward

        if self.start_distance is None and self.loop_count > 1:
            self.start_distance = distance

        if self.start_distance is not None:
            delta_distance_norm = (self.start_distance - distance) / 0.5
            reward += delta_distance_norm * 0.5
            self.reward_components['rew start dist'] = delta_distance_norm * 0.5

        if self.last_distance is not None:
            delta_distance = self.last_distance - distance
            delta_distance_reward = delta_distance * 500
            delta_distance_reward = np.clip(delta_distance_reward, -0.25, 0.25)
            reward += delta_distance_reward
            self.reward_components['rew last dist'] = delta_distance_reward
            # print(f"delta_distance: {delta_distance_reward}")

        joint_reward = self._get_joint_reward()
        reward += joint_reward
        self.reward_components['rew joint'] = joint_reward

        # print("reward: ", reward)
        self.last_distance = distance
        self.last_reward = reward
        self.last_joint_angles = joint_angles
        self.last_end_pos = end_pos
        self.last_wrist_pos = wrist_pos
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
                *joint_angles,
                dx,
                dy,
                dz,
                *block_pos,
                *end_pos
            ],
            dtype=np.float32
        ).ravel()

