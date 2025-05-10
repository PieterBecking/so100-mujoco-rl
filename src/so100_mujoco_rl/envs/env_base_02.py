"""
Base env class for environments that use the offscreen render and YOLO object detection
"""
import os
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from gymnasium.spaces import Box
from PIL import Image
from ultralytics import YOLO

from so100_mujoco_rl.envs.env_base_01 import So100BaseEnv
from so100_mujoco_rl.envs.utils import (
    JOINT_STEP_SCALE,
    MUJOCO_SO100_PREFIX,
    EndCamOffScreenViewer
)

END_CAM_RES_WIDTH = 1080
END_CAM_RES_HEIGHT = 1920


class So100OffscreenBaseEnv(So100BaseEnv):


    def __init__(self, **kwargs):
        So100BaseEnv.__init__(self, './model/env01.xml', **kwargs)

        self._set_initial_values()

        self.offscreen_viewer = EndCamOffScreenViewer(
            width=END_CAM_RES_WIDTH,
            height=END_CAM_RES_HEIGHT,
            model=self.model,
            data=self.data,
        )

        self.image_folder = "./images"
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)
        self.last_image_save_time = 0

        # load the YOLO model
        current_dir = Path(__file__).parent
        model_path = current_dir / "detect_models" / "best_sim.pt"
        self.yolo_model = YOLO(str(model_path))

        self.data.joint('block_a_joint').qpos[0:3] = self.block_target

    def _set_initial_values(self):
        raise NotImplementedError("This method should be implemented in a subclass")

    def get_observation_space(self):
        mins = [self.joints[i].range[0] for i in range(len(self.joints))]
        maxs = [self.joints[i].range[1] for i in range(len(self.joints))]

        # observation space is:
        # joint angles
        # detected object bbox center X (as a fraction of the image width)
        # detected object bbox center Y (as a fraction of the image height)
        observation_space = Box(
            np.array([*mins, 0.0, 0.0]),
            np.array([*maxs, 5.0, 5.0]),
            dtype=np.float32
        )
        return observation_space

    def render(self):
        r = super().render()
        if self.last_offscreen_render is not None and self.render_mode == "rgb_array":
            # overlay the last offscreen render on the bottom right of the image
            # only works when doing an rgb render (not human)
            overlay = cv2.resize(
                self.last_offscreen_render,
                (self.last_offscreen_render.shape[1] // 4, self.last_offscreen_render.shape[0] // 4),
                interpolation=cv2.INTER_LINEAR
            )
            overlay = np.flipud(overlay)
            r[-overlay.shape[0]:, :overlay.shape[1]] = overlay
        return r

    def get_joint_angles(self):
        return self.last_joint_angles

    def _get_projected_position(self, pos: np.ndarray) -> tuple[int, int] | None:
        cam_pos = self.data.camera(CAMERA_NAME).xpos
        cam_mat = self.data.camera(CAMERA_NAME).xmat.reshape(3, 3)
        rel_pos_world = pos - cam_pos

        rel_pos_camera = cam_mat.T @ rel_pos_world
        x, y, z = rel_pos_camera

        camera = self.offscreen_viewer.get_end_camera()
        fovy_deg = self.model.cam_fovy[camera.fixedcamid]
        fovy_rad = np.deg2rad(fovy_deg)

        fy = 0.5 * END_CAM_RES_HEIGHT / np.tan(fovy_rad / 2)
        fx = fy

        cx = END_CAM_RES_WIDTH / 2
        cy = END_CAM_RES_HEIGHT / 2

        u = fx * x / z + cx
        v = fy * y / z + cy

        if np.isnan(u) or np.isnan(v):
            return None

        u = int(u)
        v = int(v)
        if u < 0 or u >= END_CAM_RES_WIDTH or v < 0 or v >= END_CAM_RES_HEIGHT:
            return None

        # need to flip the projected coordinates to match those we'd usually get from using
        # yolo to do object detection. Presumably this is because there are different coordinate
        # systems in play
        u = END_CAM_RES_WIDTH - u
        v = END_CAM_RES_HEIGHT - v
        return int(u), int(v)

    def get_projected_block_position(self):
        # get the block position in the world frame
        block_pos = self.data.joint('block_a_joint').qpos[0:3]
        return self._get_projected_position(block_pos)

    def _get_cube_corners(
            self, center:list[float],
            width: float,
            height: float,
            depth: float
        ) -> np.ndarray:
        dx = width / 2
        dy = height / 2
        dz = depth / 2

        corners = np.array([
            [center[0] - dx, center[1] - dy, center[2] - dz],
            [center[0] - dx, center[1] - dy, center[2] + dz],
            [center[0] - dx, center[1] + dy, center[2] - dz],
            [center[0] - dx, center[1] + dy, center[2] + dz],
            [center[0] + dx, center[1] - dy, center[2] - dz],
            [center[0] + dx, center[1] - dy, center[2] + dz],
            [center[0] + dx, center[1] + dy, center[2] - dz],
            [center[0] + dx, center[1] + dy, center[2] + dz],
        ])

        return corners

    def get_projected_cube_bounding_box(self):
        cube_corners = self._get_cube_corners(
            self.data.joint('block_a_joint').qpos[0:3],
            0.02,
            0.02,
            0.02,
        )

        projected_corners = [
            self._get_projected_position(corner) for corner in cube_corners
        ]
        projected_corners = [
            corner for corner in projected_corners if corner is not None
        ]
        if not projected_corners:
            return None
        if len(projected_corners) < 2:
            return None

        min_x = min(corner[0] for corner in projected_corners)
        max_x = max(corner[0] for corner in projected_corners)
        min_y = min(corner[1] for corner in projected_corners)
        max_y = max(corner[1] for corner in projected_corners)

        return (min_x, min_y), (max_x, max_y)

    def _get_obs(self):
        obs_center_x_f = -1.0
        obs_center_y_f = -1.0

        if True:
            img = self.offscreen_viewer.render()
            self.last_offscreen_render = img
            tracker_path = Path(__file__).parent / "tracker.yaml"
            results = self.yolo_model.track(
                img,
                persist=True,
                device='mps',
                verbose=False,
                tracker=str(tracker_path),
                conf=0.25,
                iou=0.3,
            )

            for result in results:
                for box in result.boxes:
                    # if int(box.cls[0]) != 1:
                    #     continue
                    confidence = box.conf[0]
                    if confidence < 0.4:
                        continue
                    if self.tracking_id is None and box.id is not None and confidence > 0.5:
                        self.tracking_id = box.id[0]
                    if self.tracking_id is not None and box.id is not None and box.id[0] != self.tracking_id:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    center_x_f = center_x / img.shape[1]
                    center_y_f = center_y / img.shape[0]
                    width = x2 - x1
                    height = y2 - y1
                    width_f = width / img.shape[1]
                    height_f = height / img.shape[0]

                    obs_center_x_f = center_x_f
                    obs_center_y_f = center_y_f

            if obs_center_x_f == -1.0 and obs_center_y_f == -1.0:
                self.tracking_id = None

            self.cached_ob_center_x = obs_center_x_f
            self.cached_ob_center_y = obs_center_y_f
        else:
            # use the cached values from the last offscreen render / yolo detection
            obs_center_x_f = self.cached_ob_center_x
            obs_center_y_f = self.cached_ob_center_y

        if True:
            # print(results)
            # Draw detections back into the image
            if results is None:
                results = []
            for result in results:
                for box in result.boxes:
                    # if int(box.cls[0]) != 1:
                    #     continue
                    confidence = box.conf[0]
                    if confidence < 0.4:
                        continue
                    if int(box.cls[0]) != 1:
                        c = (255, 255, 0)
                    elif self.tracking_id is not None and box.id is not None and box.id[0] == self.tracking_id:
                        c = (0, 255, 0)
                    elif self.tracking_id is not None and box.id is not None and box.id[0] != self.tracking_id:
                        c = (255, 0, 0)
                    else:
                        c = (0, 0, 255)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = box.cls[0]
                    label_text = f"{self.yolo_model.names[int(label)]} {confidence:.2f}"

                    # Draw a red crosshair in the middle of the frame
                    center_x = img.shape[1] // 2
                    center_y = img.shape[0] // 2
                    img = cv2.line(img, (center_x - 20, center_y), (center_x + 20, center_y), (255, 0, 0), 4)
                    img = cv2.line(img, (center_x, center_y - 20), (center_x, center_y + 20), (255, 0, 0), 4)

                    # Draw bounding box
                    img = cv2.rectangle(img, (x1, y1), (x2, y2), c, 4)
                    # Put label text
                    img = cv2.putText(
                        img, label_text, (x1, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 1.0, c, 2
                    )

            self.last_image_save_time = int(time.time() * 1000)

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
