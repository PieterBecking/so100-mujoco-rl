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
BLOCK_SPEED_MIN = 0.0
BLOCK_SPEED_MAX = 2.0

END_CAM_RES_WIDTH = 1080
END_CAM_RES_HEIGHT = 1920


class Env03(So100BaseEnv):

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
        self.last_joint_angles = START_POSITION
        # the last good observation center x and y
        self.last_ob_center_x = None
        self.last_ob_center_y = None
        # number of steps a detected object has not been found
        self.last_ob_center_count = 0
        self.cached_ob_center_x = -1.0
        self.cached_ob_center_y = -1.0

        self.block_space_min = BLOCK_SPACE_START[0]
        self.block_space_max = BLOCK_SPACE_START[1]
        self.block_speed = BLOCK_SPEED_MIN
        # initial block target is the center of the block space
        self.block_target = [
            (BLOCK_SPACE_START[0][i] + BLOCK_SPACE_START[1][i]) / 2 for i in range(3)
        ]
        self.block_target_dt = 0.01
        self.block_target_time = 0.0

        self.tracking_id = None

        self.last_offscreen_render = None

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

    def _update_block_space(self, sim_time_fraction: float):
        # Interpolate box_space_min and box_space_max based on sim_time_fraction
        self.block_space_min = [
            BLOCK_SPACE_START[0][i] + sim_time_fraction * (BLOCK_SPACE_END[0][i] - BLOCK_SPACE_START[0][i])
            for i in range(3)
        ]
        self.block_space_max = [
            BLOCK_SPACE_START[1][i] + sim_time_fraction * (BLOCK_SPACE_END[1][i] - BLOCK_SPACE_START[1][i])
            for i in range(3)
        ]

    def _update_block_speed(self, sim_time_fraction: float):
        # Interpolate block speed based on sim_time_fraction
        if sim_time_fraction <= 0.05:
            self.block_speed = BLOCK_SPEED_MIN
        else:
            self.block_speed = BLOCK_SPEED_MIN + (sim_time_fraction - 0.05) * (BLOCK_SPEED_MAX - BLOCK_SPEED_MIN) / (1.0 - 0.05)

    def _update_block_target(self):
        # we also need to check that the block hasn't already reached the target, if it has
        # then it needs a new target otherwise it will just sit there
        current_block_pos = self.data.joint('block_a_joint').qpos[0:3]
        distance_to_target = np.linalg.norm(np.array(self.block_target) - np.array(current_block_pos))

        if self.data.time - self.block_target_time < self.block_target_dt and distance_to_target > 0.02:
            # still moving towards the last target
            return
        # Get a random block target position within block_space_min and block_space_max
        target_x = np.random.uniform(self.block_space_min[0], self.block_space_max[0])
        target_y = np.random.uniform(self.block_space_min[1], self.block_space_max[1])
        target_z = np.random.uniform(self.block_space_min[2], self.block_space_max[2])

        self.block_target = [target_x, target_y, target_z]
        self.block_target_dt = np.random.uniform(1.2, 5.1)
        self.block_target_time = self.data.time

    def _update_block_position(self):
        current_block_pos = self.data.joint('block_a_joint').qpos[0:3]

        # Calculate the direction vector from current_block_pos to block_target
        direction = np.array(self.block_target) - np.array(current_block_pos)
        distance = np.linalg.norm(direction)

        if distance > 0:
            # Normalize the direction vector
            direction = direction / distance

            # Calculate the new position based on block_speed
            step_distance = self.block_speed * self.model.opt.timestep
            new_block_pos = np.array(current_block_pos) + direction * min(step_distance, distance)

            # Update the block position
            self.data.joint('block_a_joint').qpos[0:3] = new_block_pos

    def _calculate_angular_velocity_penalty(self, joint_angles, last_joint_angles, timestep):
        penalty = 0.0
        angular_velocities = [
            (joint_angles[i] - last_joint_angles[i]) / timestep for i in range(len(joint_angles))
        ]
        if hasattr(self, "last_angular_velocities"):
            for i in range(len(angular_velocities)):
                # Calculate the change in angular velocity
                delta_angular_velocity = angular_velocities[i] - self.last_angular_velocities[i]
                # Penalize based on the magnitude of the change
                penalty += abs(delta_angular_velocity) * 0.0025
        # Store the current angular velocities for the next step
        self.last_angular_velocities = angular_velocities
        return -penalty

    def get_joint_angles(self):
        return self.last_joint_angles

    def step(self, a):
        # fraction increases from 0 to 1 over 6 seconds, doesn't exceed 1.0
        sim_time_fraction = self.data.time / 12.0
        sim_time_fraction = min(sim_time_fraction, 1.0)

        self._update_block_space(sim_time_fraction)
        self._update_block_speed(sim_time_fraction)
        self._update_block_target()
        self._update_block_position()

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
        else:
            self.last_ob_center_x = obs_center_x_f
            self.last_ob_center_y = obs_center_y_f
            self.last_ob_center_count = 0

        reward = 0.5

        if self.last_ob_center_x is not None and self.last_ob_center_y is not None:
            # if we have a last detected object, then give a reward based on how far
            # the current detected object is from the last one
            detected_distance_reward = -1 * np.sqrt(
                (0.5 - self.last_ob_center_x) ** 2 +
                (0.5 - self.last_ob_center_y) ** 2
            )
            reward += detected_distance_reward
            self.reward_components["r detected dist"] = detected_distance_reward

        joint_reward = self._get_joint_reward()
        reward += joint_reward
        self.reward_components['rew joint'] = joint_reward

        joint_acceleration_reward = self._calculate_angular_velocity_penalty(
            new_joint_angles,
            joint_angles,
            self.model.opt.timestep
        )
        joint_acceleration_reward = joint_acceleration_reward * sim_time_fraction
        self.reward_components['rew angular velocity'] = joint_acceleration_reward
        reward += joint_acceleration_reward

        self.last_reward = reward

        self.reward_components["terminated"] = terminated

        ob[-2] = 5 * ob[-2]
        ob[-1] = 5 * ob[-1]

        self.last_joint_angles = new_joint_angles

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

