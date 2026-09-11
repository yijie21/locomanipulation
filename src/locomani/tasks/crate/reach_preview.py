"""Kinematic preview of the planned reach: side pose -> Bezier -> hover, then
the straight descent, at four stand-offs. Top row: tracking camera; bottom row:
the ego head camera. No policy, no physics."""

import glob
import json
import os as _os
import sys

import cv2
import mujoco
import numpy as np

_SRC = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
sys.path.insert(0, _SRC)  # reach src/ for `python src/locomani/tasks/crate/reach_preview.py`
_os.chdir(_os.path.dirname(_SRC))  # repo root: the scene/output paths below are repo-relative
from locomani.primitives.arm_ik import ArmIK  # noqa: E402
from locomani.primitives.g1_joints import ARM_MJ  # noqa: E402

out = "outputs/scene/crate_bimanual"
meta = json.load(open(f"{out}/scene_meta.json"))
xml = f"{out}/scene_locomani.xml"
ik = ArmIK(xml)
m = mujoco.MjModel.from_xml_path(xml)
d = mujoco.MjData(m)
crate0 = np.asarray(meta["objects"]["blue_crate"]["init_pos"])
cq = meta["objects"]["blue_crate"]["init_quat"]
near_face = crate0[0] - 0.1347
PZ = 0.79
cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, meta["objects"]["blue_crate"]["body"])
cadr = int(m.jnt_qposadr[m.body_jntadr[cid]])
grip_q = [
    m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
    for n in (
        "left_hand_gripper_1_joint",
        "left_hand_gripper_2_joint",
        "right_hand_gripper_1_joint",
        "right_hand_gripper_2_joint",
    )
]
W, H = 320, 240
r = mujoco.Renderer(m, height=H, width=W)
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
cam.distance, cam.azimuth, cam.elevation = 2.0, 135.0, -20.0
ego = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
vopt = mujoco.MjvOption()
mujoco.mjv_defaultOption(vopt)
vopt.geomgroup[4] = 0


def place(q29, so):
    mujoco.mj_resetData(m, d)
    d.qpos[ik.base_qadr : ik.base_qadr + 3] = [near_face - so, 0, PZ]
    d.qpos[ik.base_qadr + 3 : ik.base_qadr + 7] = [1, 0, 0, 0]
    d.qpos[ik.qadr] = q29
    d.qpos[grip_q] = 0.0245
    d.qpos[cadr : cadr + 3] = crate0
    d.qpos[cadr + 3 : cadr + 7] = cq
    mujoco.mj_forward(m, d)


stands = (0.13, 0.18, 0.24, 0.31)
trajs = {so: np.load(f"outputs/reach/reach_preview/traj_so{so:.2f}.npy") for so in stands}
FPS = 20
T_REACH = 2.0
HOLD0 = 0.5
frames = []
n_reach = int(T_REACH * FPS)
for k in range(int(HOLD0 * FPS) + n_reach + int(0.5 * FPS)):
    row_t, row_e = [], []
    for so in stands:
        tr = trajs[so]
        i = 0 if k < HOLD0 * FPS else min(int((k - HOLD0 * FPS) / n_reach * (len(tr) - 1)), len(tr) - 1)
        place(tr[i], so)
        r.update_scene(d, camera=cam, scene_option=vopt)
        img = r.render().copy()
        cv2.putText(img, f"stand {so:.2f} m", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        row_t.append(img)
        r.update_scene(d, camera=ego, scene_option=vopt)
        row_e.append(r.render().copy())
    frames.append(np.vstack([np.hstack(row_t), np.hstack(row_e)]))
vw = cv2.VideoWriter(
    "outputs/reach/reach_preview/reach_preview.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W * 4, H * 2)
)
for f in frames:
    vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
vw.release()
print("wrote", len(frames), "frames")
