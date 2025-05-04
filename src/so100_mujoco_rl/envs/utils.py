from dataclasses import dataclass

import mujoco

# this is the prefix we use when attaching the so-arm100 model into
# the `sim_scene.xml` file.
MUJOCO_SO100_PREFIX = "so100_"

JOINT_STEP_SCALE = 0.075

REST_POSITION = [0.0, -3.141, 3.117, 1.0, 0.0, 0.0]

VALID_START_POSITIONS = [
    [0.116, -2.848, 1.84, 1.198, -1.598, 0.191],
    [0.11504855751991272, -3.0602917671203613, 2.4727771282196045, -0.5859806537628174, -1.5968739986419678, 0.18762288987636566],
    [0.11504855751991272, -3.063359498977661, 2.474310874938965, -0.5844466686248779, -1.5968739986419678, 0.18762288987636566],
    [0.11658254265785217, -3.049553871154785, 2.420621633529663, 0.09817477315664291, -1.5846021175384521, 0.19053177535533905],
    [0.11658254265785217, -3.049553871154785, 2.420621633529663, 0.11198059469461441, -1.5846021175384521, 0.19053177535533905],
    [0.7209709882736206, -2.597029447555542, 1.8867963552474976, 0.21629129350185394, -1.5968739986419678, 0.19053177535533905],
    [0.731708824634552, -2.607767343521118, 1.9911071062088013, 1.1780972480773926, -1.5968739986419678, 0.18471401929855347],
    [0.731708824634552, -2.607767343521118, 1.9911071062088013, 1.1780972480773926, -1.5968739986419678, 0.18471401929855347],
    [0.7225049734115601, -2.437495470046997, 0.6519418358802795, -0.8682331442832947, -1.59073805809021, 0.18471401929855347],
    [0.6151263117790222, -2.7719032764434814, 0.029145635664463043, -0.8682331442832947, -1.59073805809021, 0.18471401929855347],
    [0.6151263117790222, -2.7719032764434814, 0.029145635664463043, -0.8682331442832947, -1.59073805809021, 0.18471401929855347],
    [0.03374757617712021, -2.932971239089966, 0.03067961521446705, 0.5905826091766357, -2.4190876483917236, 0.18907733261585236],
    [0.11044661700725555, -2.787243127822876, 1.718058466911316, -0.9295923709869385, -2.4221556186676025, 0.19053177535533905],
    [0.11044661700725555, -2.787243127822876, 1.718058466911316, -0.9295923709869385, -2.4221556186676025, 0.19053177535533905],
    [0.1702718734741211, -1.8116313219070435, 2.230407953262329, -0.22549517452716827, -2.161378860473633, 0.19053177535533905],
    [0.6902913451194763, -1.7978254556655884, 2.2319419384002686, -0.22549517452716827, -2.1629128456115723, 0.19053177535533905],
    [0.6902913451194763, -1.7978254556655884, 2.2319419384002686, -0.22549517452716827, -2.1629128456115723, 0.19053177535533905],
    [1.1903691291809082, -1.7057865858078003, 2.1629128456115723, 0.8605632185935974, -1.7241944074630737, 0.18616846203804016],
    [0.007669903803616762, -2.7488934993743896, 2.8808159828186035, 0.5445631742477417, -1.7257283926010132, 0.19198621809482574],
    [0.007669903803616762, -2.7488934993743896, 2.8808159828186035, 0.5445631742477417, -1.7257283926010132, 0.19198621809482574],
    [-0.04908738657832146, -3.0173401832580566, 2.702874183654785, -0.06442718952894211, -1.7257283926010132, 0.19198621809482574],
    [-0.07516505569219589, -2.7274177074432373, 0.5246214270591736, -1.3406991958618164, -1.7211264371871948, 0.19198621809482574],
    [-0.07516505569219589, -2.7274177074432373, 0.5246214270591736, -1.3406991958618164, -1.7211264371871948, 0.19198621809482574],
    [-0.06902913749217987, -2.730485677719116, 0.5077476501464844, -1.3284273147583008, -1.7226604223251343, 0.19198621809482574],
    [1.0154953002929688, -3.1293208599090576, 0.5046796798706055, -1.3406991958618164, -1.7195924520492554, 0.19198621809482574],
    [1.0154953002929688, -3.1293208599090576, 0.5046796798706055, -1.3406991958618164, -1.7195924520492554, 0.19198621809482574],
    [1.371378779411316, -2.471243143081665, 2.633845090866089, 0.5921165943145752, -1.7211264371871948, 0.19198621809482574],
    [2.0202527046203613, -1.023165225982666, 1.3176895380020142, 0.5905826091766357, -1.7211264371871948, 0.19198621809482574],
    [2.0202527046203613, -1.023165225982666, 1.3176895380020142, 0.5905826091766357, -1.7211264371871948, 0.19198621809482574],
    [0.5967185497283936, -2.178252696990967, 1.7165244817733765, 0.5905826091766357, -1.7211264371871948, 0.19198621809482574],
    [0.200951486825943, -2.5003886222839355, 0.9234564304351807, -1.339165210723877, -1.7195924520492554, 0.19198621809482574],
    [0.200951486825943, -2.5003886222839355, 0.9234564304351807, -1.339165210723877, -1.7195924520492554, 0.19198621809482574],
    [0.777728259563446, -2.842466354370117, 0.0, -1.3514370918273926, -1.718058466911316, 0.19198621809482574],
    [-0.5077476501464844, -2.7765052318573, 0.00920388475060463, -1.0860583782196045, -1.718058466911316, 0.19198621809482574],
    [-0.5077476501464844, -2.7765052318573, 0.00920388475060463, -1.0860583782196045, -1.718058466911316, 0.19198621809482574],
    [-0.5077476501464844, -2.764233350753784, 1.2394564151763916, 1.1520196199417114, -1.7211264371871948, 0.19198621809482574],
]


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
        if name.startswith(MUJOCO_SO100_PREFIX) and 'end_point_joint' not in name:
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


import glfw
import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer

from so100_mujoco_rl.envs.utils import MUJOCO_SO100_PREFIX

CAMERA_NAME = "so100_end_point_camera"

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
        cam.fixedcamid = self.model.camera(CAMERA_NAME).id
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
