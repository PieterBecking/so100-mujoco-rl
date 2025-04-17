from dataclasses import dataclass

import mujoco


# this is the prefix we use when attaching the so-arm100 model into
# the `sim_scene.xml` file.
MUJOCO_SO100_PREFIX = "so100_"

JOINT_STEP_SCALE = 0.05

@dataclass
class Joint:
    """ class for representing a joint"""
    name: str
    # radians, min and max
    range: tuple[float, float]

    def __repr__(self):
        return f"Joint({self.name}, {self.range})"


def joints_from_model(model: mujoco.MjModel) -> list[Joint]:
    """
    Extracts joint details from a mujoco model
    """
    # get the number of joints
    num_joints = model.njnt

    # get joint names
    joint_names = []
    for i in range(num_joints):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        # there are other  non-so100 joints in the mujoco model
        # like the block free joints
        if name.startswith(MUJOCO_SO100_PREFIX):
            name = name[len(MUJOCO_SO100_PREFIX):]
            joint_names.append(name)

    # get joint ranges
    joint_ranges = model.jnt_range.reshape(-1, 2)

    joints: list[Joint] = []
    for i in range(len(joint_names)):
        j = Joint(joint_names[i], tuple(joint_ranges[i]))
        j.range = (j.range[0], j.range[1])
        joints.append(j)
    return joints

