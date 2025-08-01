import math
import mujoco
import numpy as np

from scipy.spatial.transform import Rotation

from so100_mujoco_rl.envs.env_base_06 import So100BaseEnv
from so100_mujoco_rl.envs.utils import JOINT_STEP_SCALE, MUJOCO_SO100_PREFIX, REST_POSITION

class Env06(So100BaseEnv):

    def __init__(self, **kwargs):
        So100BaseEnv.__init__(self, './model/env06.xml', **kwargs)

        self.block_pos = None
        self.last_block_pos = None

    def step(self, a):
        is_in_reach = self.get_block_to_end_distance() < 0.03
        reward = self._get_reward(is_in_reach)

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

            # self.set_random_block_position()

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
