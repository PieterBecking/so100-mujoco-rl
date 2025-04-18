import math
import mujoco
import numpy as np

from scipy.spatial.transform import Rotation

from so100_mujoco_rl.envs.env_base_01 import So100BaseEnv
from so100_mujoco_rl.envs.utils import JOINT_STEP_SCALE, MUJOCO_SO100_PREFIX, VALID_START_POSITIONS

class Env01(So100BaseEnv):

    def __init__(self, **kwargs):
        So100BaseEnv.__init__(self, './model/env01.xml', **kwargs)

    def step(self, a):
        reward = self._get_reward()

        joint_angles = self.get_joint_angles()
        new_joint_angles = [
            joint_angles[i] + a[i] * JOINT_STEP_SCALE for i in range(len(joint_angles))
        ]

        for joint, new_angle in zip(self.joints, new_joint_angles):
            self.data.actuator(MUJOCO_SO100_PREFIX + joint.name).ctrl = new_angle

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        mujoco.mj_rnePostConstraint(self.model, self.data)

        terminated = False

        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()
        # print( f"ob: {ob}")
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return ob, reward, terminated, False, {}

    def reset_model(self):
        self.start_distance = None
        self.loop_count = 0

        # get a random block location that is at least 80mm away from the base origin, but
        # within 420mm of the base origin
        dist = np.random.uniform(0.18, 0.42)
        theta = np.random.uniform(0, 2 * np.pi)
        theta = -0.5 * np.pi + np.random.uniform(-0.25 * np.pi, 0.25 * np.pi)
        x = dist * math.cos(theta)
        y = dist * math.sin(theta)

        random_block_pos = [x, y, 0.0]
        self.data.joint('block_a_joint').qpos[0:3] = random_block_pos

        # Get a random integer between 0 and the length of VALID_START_POSITIONS
        random_index = np.random.randint(0, len(VALID_START_POSITIONS))
        start_pos = VALID_START_POSITIONS[random_index]
        for i, joint in enumerate(self.joints):
            if joint.name == "Jaw":
                continue
            joint_name = MUJOCO_SO100_PREFIX + joint.name
            self.data.joint(joint_name).qpos[0] = start_pos[i]

        return self._get_obs()
