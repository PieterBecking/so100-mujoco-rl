
import math
import os
import time

import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer
from PIL import Image

from so100_mujoco_rl.envs.env_base_01 import So100BaseEnv
from so100_mujoco_rl.envs.utils import JOINT_STEP_SCALE, MUJOCO_SO100_PREFIX

import glfw
START_POSITION = [0.0, -2.04, 1.19, 1.5, -1.58, 0.5]

# The Gymnasium offscreen renderer doesn't quite work the way we need. It seems
# to not fix the camera to the end of the arm. Also doesn't quite respect the width
# and height we give it.
# Here we make a few changes so it does work the way we need
class EndCamOffScreenViewer(OffScreenViewer):

    def __init__(self, width, height, model, data):
        # these two lines are important
        # it uses these values when creating a new context for the offscreen
        # rendering
        old_w, old_h = model.vis.global_.offwidth, model.vis.global_.offheight
        model.vis.global_.offwidth = width
        model.vis.global_.offheight = height

        super().__init__(model, data, width, height)

        self._cam = self.get_end_camera()

        model.vis.global_.offwidth = old_w
        model.vis.global_.offheight = old_h

        self.old_con = None

    def get_end_camera(self):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        # name in so_arm100.xml is end_point_camera, but it gets given the so100_ prefix
        # when imported as an asset
        cam.fixedcamid = self.model.camera("so100_end_point_camera").id
        return cam

    def render(self):
        # we get issues with onscreen and offscreen rendering if we don't manage which context is
        # current. So stash whateved context is current, and restore that after we do the offscreen
        # rendering.
        self.old_con = glfw.get_current_context()
        self.make_context_current()

        # do the offscreen rendering
        mujoco.mjv_updateScene(self.model, self.data, self.vopt, None, self._cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
        r = mujoco.mjr_render(self.viewport, self.scn, self.con)
        image = np.zeros((self.viewport.height, self.viewport.width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(image, None, self.viewport, self.con)

        # restore the old context
        glfw.make_context_current(self.old_con)

        return image


class Env03(So100BaseEnv):

    def __init__(self, **kwargs):
        So100BaseEnv.__init__(self, './model/env01.xml', **kwargs)

        self.block_pos = None
        self.last_block_pos = None

        self.offscreen_viewer = EndCamOffScreenViewer(
            width=1080,
            height=1920,
            model=self.model,
            data=self.data,
        )

        self.image_folder = "./images"
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)
        self.last_image_save_time = 0

    def render(self):
        return super().render()

    def step(self, a):
        reward = self._get_reward()

        joint_angles = self.get_joint_angles()
        new_joint_angles = [
            joint_angles[i] + a[i] * JOINT_STEP_SCALE for i in range(len(joint_angles))
        ]

        for joint, new_angle in zip(self.joints, new_joint_angles):
            self.data.actuator(MUJOCO_SO100_PREFIX + joint.name).ctrl = new_angle

        if self.get_block_to_end_distance() < 0.02:
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
        dist = np.random.uniform(0.25, 0.42)
        theta = np.random.uniform(0, 2 * np.pi)
        theta = -0.5 * np.pi + np.random.uniform(-0.05 * np.pi, 0.05 * np.pi)
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

        start_pos = START_POSITION
        for i, joint in enumerate(self.joints):
            joint_name = MUJOCO_SO100_PREFIX + joint.name
            self.data.joint(joint_name).qpos[0] = start_pos[i]

        return self._get_obs()

    def get_end_camera(self):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = 0
        return cam

    def __save_render(self, img_data: np.ndarray):
        image = np.flipud(img_data)  # Flip the image vertically
        filename = f"{self.image_folder}/frame_{int(time.time() * 1000)}.png"
        Image.fromarray(image).save(filename)

    def _get_obs(self):
        self.loop_count += 1

        if int(time.time() * 1000) - self.last_image_save_time > 5000:
            img = self.offscreen_viewer.render()
            self.__save_render(img)
            self.last_image_save_time = int(time.time() * 1000)

        block_pos = self.get_block_pos()
        end_pos = self.get_end_effector_pos()

        dx = block_pos[0] - end_pos[0]
        dy = block_pos[1] - end_pos[1]
        dz = block_pos[2] - end_pos[2]

        joint_angles = self.get_joint_angles()

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

