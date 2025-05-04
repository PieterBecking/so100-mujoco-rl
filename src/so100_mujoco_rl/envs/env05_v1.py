import cv2
import mujoco
import numpy as np

from so100_mujoco_rl.envs.env_base_02 import END_CAM_RES_WIDTH, END_CAM_RES_HEIGHT
from so100_mujoco_rl.envs.env03_v1 import Env03
from so100_mujoco_rl.envs.utils import (
    JOINT_STEP_SCALE,
    MUJOCO_SO100_PREFIX
)

class Env05(Env03):

    def _get_obs(self):
        obs_center_x_f = -1.0
        obs_center_y_f = -1.0

        projected_block_pos = self.get_projected_block_position()
        if projected_block_pos is not None:
            center_x, center_y = projected_block_pos
            obs_center_x_f = center_x / END_CAM_RES_WIDTH
            obs_center_y_f = center_y / END_CAM_RES_HEIGHT

        if False:
            img = self.offscreen_viewer.render()

            # Draw a red crosshair in the middle of the frame
            center_x = img.shape[1] // 2
            center_y = img.shape[0] // 2
            img = cv2.line(img, (center_x - 20, center_y), (center_x + 20, center_y), (255, 0, 0), 4)
            img = cv2.line(img, (center_x, center_y - 20), (center_x, center_y + 20), (255, 0, 0), 4)

            if projected_block_pos is not None:
                # print(f"projected_block_pos: {projected_block_pos}")
                center_x, center_y = projected_block_pos
                img = cv2.line(img, (center_x - 10, center_y), (center_x + 10, center_y), (0, 0, 255), 6)
                img = cv2.line(img, (center_x, center_y - 10), (center_x, center_y + 10), (0, 0, 255), 6)

            self.last_offscreen_render = img

        joint_angles = self.get_joint_angles()

        self.loop_count += 1

        return np.array(
            [
                *joint_angles,
                obs_center_x_f,
                obs_center_y_f,
            ],
            dtype=np.float32
        ).ravel()

