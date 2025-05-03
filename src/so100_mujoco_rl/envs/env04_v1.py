import os
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer
from gymnasium.spaces import Box
from PIL import Image
from ultralytics import YOLO

from so100_mujoco_rl.envs.env03_v1 import EndCamOffScreenViewer
from so100_mujoco_rl.envs.env_base_01 import So100BaseEnv
from so100_mujoco_rl.envs.utils import (
    JOINT_STEP_SCALE,
    MUJOCO_SO100_PREFIX,
    EndCamOffScreenViewer
)

START_POSITION = [0.0, -2.04, 1.19, 1.5, -1.58, 0.5]

# min xyz, and max xyz
BLOCK_SPACE_START = [
    [-0.05, -0.4, 0.01],
    [0.05, -0.3, 0.01]
]
BLOCK_SPACE_END = [
    [-0.35, -0.45, 0.01],
    [0.35, -0.25, 0.01]
]

END_CAM_RES_WIDTH = 1080
END_CAM_RES_HEIGHT = 1920




class Env04(So100BaseEnv):

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
        # the last good observation center x and y
        self.last_ob_center_x = -1.0
        self.last_ob_center_y = -1.0
        # number of steps a detected object has not been found
        self.last_ob_center_count = 0
        self.cached_ob_center_x = -1.0
        self.cached_ob_center_y = -1.0

        self.block_space_min = BLOCK_SPACE_START[0]
        self.block_space_max = BLOCK_SPACE_START[1]

        # initial block target is the center of the block space
        self.block_target = [
            (BLOCK_SPACE_START[0][i] + BLOCK_SPACE_START[1][i]) / 2 for i in range(3)
        ]
        self.block_target_dt = 0.01
        self.block_target_time = 0.0
        self.block_position_updated = False

        self.tracking_id = None
        

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
        return super().render()

    def _update_block_space(self, sim_time_fraction: float):
        # Interpolate box_space_min and box_space_max based on sim_time_fraction
        # self.block_space_min = [
        #     BLOCK_SPACE_START[0][i] + sim_time_fraction * (BLOCK_SPACE_END[0][i] - BLOCK_SPACE_START[0][i])
        #     for i in range(3)
        # ]
        # self.block_space_max = [
        #     BLOCK_SPACE_START[1][i] + sim_time_fraction * (BLOCK_SPACE_END[1][i] - BLOCK_SPACE_START[1][i])
        #     for i in range(3)
        # ]
        pass



    def _update_block_position(self):

        # Get a random block target position within block_space_min and block_space_max
        target_x = np.random.uniform(self.block_space_min[0], self.block_space_max[0])
        target_y = np.random.uniform(self.block_space_min[1], self.block_space_max[1])
        target_z = np.random.uniform(self.block_space_min[2], self.block_space_max[2])

        self.block_target = [target_x, target_y, target_z]

        self.data.joint('block_a_joint').qpos[0:3] = self.block_target

    def step(self, a):
        # fraction increases from 0 to 1 over 6 seconds, doesn't exceed 1.0
        sim_time_fraction = self.data.time / 12.0
        sim_time_fraction = min(sim_time_fraction, 1.0)

        self._update_block_space(sim_time_fraction)


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

        obs_center_x_f = ob[-2]
        obs_center_y_f = ob[-1]
        if obs_center_x_f == -1.0 and obs_center_y_f == -1.0:
            # nothing detected this time
            if self.last_ob_center_count > 30:
                # if we haven't detected anything for 10 steps, then terminate as we've lost
                # the cube
                terminated = True
            self.last_ob_center_count += 1

            # otherwise we'd be feeding in -1 and -1, which is garbage
            ob[-2] = self.last_ob_center_x
            ob[-1] = self.last_ob_center_y
        else:
            self.last_ob_center_x = obs_center_x_f
            self.last_ob_center_y = obs_center_y_f
            self.last_ob_center_count = 0

        reward = 0.5

        if self.last_ob_center_x is not None and self.last_ob_center_y is not None:
            # if we have a last detected object, then give a reward based on how far
            # the current detected object is from the last one
            detected_distance = np.sqrt(
                (0.5 - self.last_ob_center_x) ** 2 +
                (0.5 - self.last_ob_center_y) ** 2
            )
            detected_distance_bonus_reward = np.exp(-10 * detected_distance)
            reward += detected_distance_bonus_reward
            self.reward_components["r dist bonus"] = detected_distance_bonus_reward

            detected_distance_reward = -1.0 * detected_distance
            reward += detected_distance_reward
            self.reward_components["r detected dist"] = detected_distance_reward

            block_looked_at_reward = 0.0
            if detected_distance < 0.1 and not self.block_position_updated:
                self.block_position_updated = True
                self._update_block_position()
                block_looked_at_reward = 10.0
                self.tracking_id = None
                print(f"looked at block {self.loop_count}")
            self.reward_components["r looked at"] = block_looked_at_reward
            reward += block_looked_at_reward

        joint_reward = self._get_joint_reward()
        reward += joint_reward
        self.reward_components['rew joint'] = joint_reward

        # penalise any rotation of joint 4 (wrist roll), if it rotates it will only confuse
        # itself
        wrist_roll_penalty = self._calculate_joint_penalty(
            joint_angle=joint_angles[4],
            joint_range=[
                START_POSITION[4] - 0.2,
                START_POSITION[4] + 0.2
            ]
        )
        wrist_roll_penalty = np.clip(wrist_roll_penalty, -0.2, 0.0)
        reward += wrist_roll_penalty * 0.5
        self.reward_components['rew wrist roll'] = wrist_roll_penalty

        self.last_reward = reward

        self.reward_components["terminated"] = terminated

        ob[-2] = 5 * ob[-2]
        ob[-1] = 5 * ob[-1]

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return ob, reward, terminated, False, {}

    def reset_model(self):
        self.start_distance = None
        self.loop_count = 0

        self._set_initial_values()
        self.data.joint('block_a_joint').qpos[0:3] = self.block_target

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
        obs_center_x_f = -1.0
        obs_center_y_f = -1.0

        if self.loop_count % self.frame_skip == 0:
            img = self.offscreen_viewer.render()
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

                    self.block_position_updated = False

            if obs_center_x_f == -1.0 and obs_center_y_f == -1.0:
                self.tracking_id = None

            self.cached_ob_center_x = obs_center_x_f
            self.cached_ob_center_y = obs_center_y_f
        else:
            # use the cached values from the last offscreen render / yolo detection
            obs_center_x_f = self.cached_ob_center_x
            obs_center_y_f = self.cached_ob_center_y

        # if  int(time.time() * 1000) - self.last_image_save_time > 1000:
        if  self.loop_count % self.frame_skip == 0:
            # print(results)
            # Flip the image back vertically for correct drawing
            img = np.flipud(img)

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

                    # Adjust coordinates for flipped image
                    flipped_y1 = img.shape[0] - y1
                    flipped_y2 = img.shape[0] - y2

                    # Draw bounding box
                    img = cv2.rectangle(img.astype(np.uint8), (x1, flipped_y1), (x2, flipped_y2), c, 2)
                    # Put label text
                    img = cv2.putText(
                        img.astype(np.uint8), label_text, (x1 + 4, flipped_y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2
                    )

            # Put label text
            img = cv2.putText(
                img.astype(np.uint8), f"x = {obs_center_x_f:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2
            )
            img = cv2.putText(
                img.astype(np.uint8), f"y = {obs_center_y_f:.2f}", (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2
            )

            detected_distance = np.sqrt(
                (0.5 - obs_center_x_f) ** 2 +
                (0.5 - obs_center_y_f) ** 2
            )
            if detected_distance < 0.1:
                img = cv2.putText(
                    img.astype(np.uint8), f"Looked at block", (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
                )


            # Flip the image back again for saving
            img = np.flipud(img)

            self.__save_render(img)
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

