import math
import mujoco
import numpy as np

from scipy.spatial.transform import Rotation

from so100_mujoco_rl.envs.env_base_01 import So100BaseEnv
from so100_mujoco_rl.envs.utils import JOINT_STEP_SCALE, MUJOCO_SO100_PREFIX, REST_POSITION

class Env06(So100BaseEnv):

    def __init__(self, **kwargs):
        So100BaseEnv.__init__(self, './model/env01.xml', **kwargs)

        self.block_pos = None
        self.last_block_pos = None

    def step(self, a):
        reward = self._get_reward()

        joint_angles = self.get_joint_angles()
        new_joint_angles = [
            joint_angles[i] + a[i] * JOINT_STEP_SCALE for i in range(len(joint_angles))
        ]

        for joint, new_angle in zip(self.joints, new_joint_angles):
            self.data.actuator(MUJOCO_SO100_PREFIX + joint.name).ctrl = new_angle

        if self.get_block_to_end_distance() < 0.03:
            # give it a bonus reward for reaching the block based on its distance
            # from the previous block
            block_distance = np.linalg.norm(
                np.array(self.block_pos) - np.array(self.last_block_pos)
            )
            reward += block_distance * 20

            self.set_random_block_position()

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        mujoco.mj_rnePostConstraint(self.model, self.data)

        terminated = False

        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()
        # print( f"ob: {ob}")
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return ob, reward, terminated, False, {}

    def set_random_block_position(self):
        # get a random block location that is at least 80mm away from the base origin, but
        # within 420mm of the base origin
        dist = np.random.uniform(0.22, 0.42)
        theta = np.random.uniform(0, 2 * np.pi)
        theta = -0.5 * np.pi + np.random.uniform(-0.25 * np.pi, 0.25 * np.pi)
        x = dist * math.cos(theta)
        y = dist * math.sin(theta)

        random_block_pos = [x, y, 0.0]
        self.data.joint('block_a_joint').qpos[0:3] = random_block_pos

        if self.last_block_pos is None:
            self.last_block_pos = random_block_pos
        else:
            self.last_block_pos = self.block_pos
        self.block_pos = random_block_pos

    def reset_model(self):
        self.start_distance = None
        self.loop_count = 0

        self.set_random_block_position()

        start_pos = REST_POSITION
        for i, joint in enumerate(self.joints):
            joint_name = MUJOCO_SO100_PREFIX + joint.name
            self.data.joint(joint_name).qpos[0] = start_pos[i]

        return self._get_obs()


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
            if self.last_joint_angles is not None and pitch < -0.7 * math.pi:
                pitch_reward = (pitch + 0.7 * math.pi) * 0.7
                reward += pitch_reward
                self.reward_components['rew pitch'] = pitch_reward

        if self.last_end_pos is not None:
            if end_pos[2] < 0.02:
                end_pos_z_reward = (end_pos[2] - 0.02) * 20.0
                reward += end_pos_z_reward
                self.reward_components['rew end pos z'] = end_pos_z_reward

        if self.last_wrist_pos is not None:
            if wrist_pos[2] < 0.08:
                wrist_pos_z_reward = (wrist_pos[2] - 0.08) * 10.0
                wrist_pos_z_reward = np.clip(wrist_pos_z_reward, -0.8, 0.8)
                reward += wrist_pos_z_reward
                self.reward_components['rew wrist pos z'] = wrist_pos_z_reward

        if self.start_distance is None and self.loop_count > 1:
            self.start_distance = distance

        detected_distance_reward = -distance
        detected_distance_reward += 0.02
        detected_distance_reward = min(detected_distance_reward, 0.0)
        self.reward_components['detected_distance_reward'] = detected_distance_reward
        reward += detected_distance_reward * 0.5

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
