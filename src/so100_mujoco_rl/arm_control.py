import time
from dataclasses import dataclass

import torch
from configs.so100 import So100Config
from lerobot.common.robot_devices.motors.feetech import (
    FeetechMotorsBus,
    TorqueMode
)
from lerobot.common.robot_devices.robots.utils import make_robot_from_config


@dataclass
class Joint:
    """ class for representing a joint"""
    name: str
    # radians, min and max
    range: tuple[float, float]

    def __repr__(self):
        return f"Joint({self.name}, {self.range})"


class ArmController:
    """
    Base class for controlling a robotic arm
    """
    def __init__(self, joints: list[Joint]):
        self.joints = joints
        # the position that this controller has been told to move to
        self.joint_set_positions = [0.0] * len(self.joints)
        # the position this controller is in right now
        self.joint_actual_positions = [0.0] * len(self.joints)
        # the position this controller provides to other controllers
        # when it is primary
        self.joint_output_positions = [0.0] * len(self.joints)

        self._primary = False
        self._name = "Base"
        self._controllable = False

        self.count = 0

    @property
    def primary(self) -> bool:
        return self._primary

    @primary.setter
    def primary(self, value: bool) -> None:
        self._primary = value
        self._primary_set()

    @property
    def name(self) -> str:
        return self._name

    @property
    def controllable(self) -> bool:
        """
        Returns True if the controller is in a state where
        if can be used to drive the other controllers.
        eg; real robot controller would be false until connected
        """
        return self._controllable

    def set_joint_actual_position(self, joint_name: str, position: float):
        for i, joint in enumerate(self.joints):
            if joint.name == joint_name:
                # Clamp the position within the joint's range
                clamped_position = max(joint.range[0], min(position, joint.range[1]))
                self.joint_actual_positions[i] = clamped_position
                break

    def get_joint_actual_position(self, joint_name: str) -> float:
        for i, joint in enumerate(self.joints):
            if joint.name == joint_name:
                return self.joint_actual_positions[i]
        raise ValueError(f"Joint {joint_name} not found")

    def get_joint_actual_positions(self) -> list[float]:
        return self.joint_actual_positions

    def set_joint_set_position(self, joint_name: str, position: float):
        for i, joint in enumerate(self.joints):
            if joint.name == joint_name:
                # Clamp the position within the joint's range
                clamped_position = max(joint.range[0], min(position, joint.range[1]))
                self.joint_set_positions[i] = clamped_position
                break

    def get_joint_set_position(self, joint_name: str) -> float:
        for i, joint in enumerate(self.joints):
            if joint.name == joint_name:
                return self.joint_set_positions[i]
        raise ValueError(f"Joint {joint_name} not found")

    def get_joint_set_positions(self) -> list[float]:
        return self.joint_set_positions

    def set_joint_set_positions(self, positions: list[float]):
        if len(positions) != len(self.joints):
            raise ValueError(f"Expected {len(self.joints)} joint positions, got {len(positions)}")
        # Clamp each position within the corresponding joint's range
        self.joint_set_positions = [
            max(joint.range[0], min(position, joint.range[1]))
            for joint, position in zip(self.joints, positions)
        ]

        
        if self.count % 1000 == 0:
            print(f"{self.joint_set_positions}")
        self.count += 1

    def reset(self):
        self.joint_set_positions = [0.0] * len(self.joints)
        self.joint_actual_positions = [0.0] * len(self.joints)
        self.joint_output_positions = [0.0] * len(self.joints)

    def update(self):
        self.joint_actual_positions = list(self.joint_set_positions)
        self.joint_output_positions = list(self.joint_set_positions)

    def set_positions(self):
        pass

    def _primary_set(self):
        """ override this function if the controller needs to do something when
        its state as primary is changed.
        """
        pass


class So100ArmController(ArmController):
    """
    Class for controlling the So100 robotic arm
    """
    def __init__(self, ):
        self.robot = None

        # we get a JointOutOfRangeError if any of the angle joints exceed +/- 270 deg (4.69 rad)
        # or -10 to 110 for the gripper
        joints = [
            Joint("shoulder_pan", (-4.69, 4.69)),
            Joint("shoulder_lift", (-4.69, 4.69)),
            Joint("elbow_flex", (-4.69, 4.69)),
            Joint("wrist_flex", (-4.69, 4.69)),
            Joint("wrist_roll", (-4.69, 4.69)),
            Joint("gripper", (-0.17, 1.9)),
        ]
        super().__init__(joints)

        self._name = "Robot"

    def connect(self, port: str, calibration_dir: str) -> None:
        # Create the So100 robot from the configuration
        robot = make_robot_from_config(
            So100Config(calibration_dir=calibration_dir, port=port)
        )
        robot.connect()
        # small delay to allow the robot to connect
        time.sleep(0.2)
        self.robot = robot

    def is_connected(self) -> bool:
        """
        Checks if the robot is connected
        """
        if self.robot is None:
            return False
        return self.robot.is_connected

    @property
    def controllable(self) -> bool:
        return self.is_connected()

    def update(self):
        super().update()
        if self.robot is None:
            return
        # Update the actual positions of the joints by reading from the robot
        # This is where you would read the actual positions of the joints from the robot
        # and update the joint_actual_positions attribute
        obs: torch.Tensor = self.robot.capture_observation()['observation.state']

        # print("obs")
        # print(obs)
        obs = torch.deg2rad(obs).tolist()
        # TODO: which motors should be flipped is available in the calibration config
        # kind of that is, the first one isn't reversed in the calibration so not sure
        # what's up
        obs[0] *= -1.0
        obs[1] *= -1.0
        obs[4] *= -1.0

        for i, joint in enumerate(self.joints):
            joint_actual_pos = obs[i]
            self.set_joint_actual_position(joint.name, joint_actual_pos)
        
        # set the output positions to be the actual robot positions
        self.joint_output_positions = list(self.joint_actual_positions)

    def set_positions(self):
        """
        Applies the set joint permissions to the So100 robot
        """
        if self.robot is None:
            return

        position_floats = list(self.joint_set_positions)
        position_floats[0] *= -1.0
        position_floats[1] *= -1.0
        position_floats[4] *= -1.0

        position_tensor = torch.FloatTensor(position_floats)
        position_tensor = torch.rad2deg(position_tensor)

        self.robot.send_action(position_tensor)

        # print("position_tensor")
        # print(position_tensor)

    def _primary_set(self):
        """ override this function if the controller needs to do something when
        its state as primary is changed.
        """
        if self.robot is None:
            return

        for name in self.robot.follower_arms:
            mb: FeetechMotorsBus =  self.robot.follower_arms[name]

            if self.primary:
                mb.write("Torque_Enable", TorqueMode.DISABLED.value)
                mb.write("Lock", 0)
            else:
                mb.write("Torque_Enable", TorqueMode.ENABLED.value)
                mb.write("Lock", 1)

