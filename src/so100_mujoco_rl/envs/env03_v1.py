import mujoco
import numpy as np

from so100_mujoco_rl.envs.env_base_02 import So100OffscreenBaseEnv
from so100_mujoco_rl.envs.utils import (
    JOINT_STEP_SCALE,
    MUJOCO_SO100_PREFIX
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


class Env03(So100OffscreenBaseEnv):

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

            # need to zero out the blocks vecocity, because when we cancel out gravity
            # momentum causes the block to fly out to space
            self.data.joint('block_a_joint').qvel[0:3] = [0,0,0]

            # Get the mass of the block_a body
            block_a_body_id = self.model.body('block_a').id
            mass = self.model.body_mass[block_a_body_id]
            gravity = self.model.opt.gravity
            anti_gravity_force = -mass * gravity
            self.data.joint('block_a_joint').qfrc_applied[0:3] = anti_gravity_force

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

