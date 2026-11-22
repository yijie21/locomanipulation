"""selfmodel_via_20261120: where the acting arm really went and where the reference said it would, as collision spheres.

    cd $REPO && $PY -I scripts/experiments/selfmodel_via_20261120/arm_spheres.py <dataset dir> <scene dir>

<dataset dir>/all.npz is a REACH_VIA / REACH_VIA_REPLAY dataset of reachwm_20261030/gen_reach.py, <scene dir> the scene it
was recorded in (REACH_SCENE_DIR). The spheres are cuRobo's for SIMPLE's G1 with Dex3 hands (SIMPLE_ROOT/data/robots/g1/
curobo/spheres/g1_29dof.yml): the right arm's 9 (shoulder pitch, roll, yaw, elbow, wrist pitch), the palm's 3 and the
fingers' 12, 24 in all. A left-hand segment uses the left links with the same spheres mirrored (y -> -y in the link frame),
so every segment has the same 24 in the same order. The palm link is not a body of the simulated robot: it sits on the
wrist-yaw link at the offset the URDF gives it.

Forward kinematics per 10 Hz tick from the recorded joints (S[:29], ScaleBFM's metadata order), the pelvis position
(W[0:3]) and its attitude (yaw W[3], roll and pitch from the gravity direction S[58:61]); per reference pose from RQ
(MuJoCo joint order) and RP (pelvis position and quaternion). The fingers stand at hand_q, the acting hand's joints at the
command. Everything is written in the segment's heading frame at the command (x ahead, y left, z above the floor), not
mirrored.

-> <dataset dir>/arm.npz: X (ticks, 24, 3) the spheres at every recorded tick, XR (reference poses, 24, 3) at every
reference pose, radius (24,), link (24,) the right arm's link names; n_steps and n_ref index them as in all.npz.
Printed: the rebuilt jaw centre against the recorded one (W[4:7] right, W[7:10] left), and the last reference pose's jaw
against the command -- the two checks that the joint orders and frames are right.
"""

import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import yaml

REPO = Path.cwd()
sys.path.insert(0, str(REPO / "src"))
from locomani.backends.scalebfm import runtime as RT  # noqa: E402
from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES  # noqa: E402

D, SCENE = Path(sys.argv[1]), Path(sys.argv[2])
SIMPLE = Path(os.environ.get("SIMPLE_ROOT", "/workspace/code/locomanipulation/third_party/SIMPLE"))
ARM = ("shoulder_pitch_link", "shoulder_roll_link", "shoulder_yaw_link", "elbow_link", "wrist_pitch_link")
HAND = ("hand_palm_link", "hand_thumb_0_link", "hand_thumb_1_link", "hand_thumb_2_link", "hand_middle_0_link",
        "hand_middle_1_link", "hand_index_0_link", "hand_index_1_link")  # fmt: skip
FINGERS = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
OFF = np.array(
    [float(v) for v in os.environ.get("REACH_HAND_OFF", "0.0415,0,0").split(",")]
)  # jaw centre, wrist-yaw frame

sph = yaml.safe_load(open(SIMPLE / "data/robots/g1/curobo/spheres/g1_29dof.yml"))
sph = sph.get("collision_spheres", sph)
urdf = ET.parse(SIMPLE / "data/robots/g1/g1_29dof_with_dex3.urdf").getroot()
palm = {}
for j in urdf.iter("joint"):
    if j.get("name", "").endswith("_hand_palm_joint"):
        assert j.find("origin").get("rpy", "0 0 0").split() == ["0", "0", "0"], "the palm link is turned on the wrist"
        palm[j.get("name")[: -len("_hand_palm_joint")]] = np.array(
            [float(v) for v in j.find("origin").get("xyz").split()]
        )
links = [n for n in ARM + HAND for _ in sph[f"right_{n}"]]
local = np.array([s["center"] for n in ARM + HAND for s in sph[f"right_{n}"]], float)
radius = np.array([s["radius"] for n in ARM + HAND for s in sph[f"right_{n}"]], np.float32)

m = mujoco.MjModel.from_xml_path(str(SCENE / "scene_locomani.xml"))
d = mujoco.MjData(m)
assert m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE, "the robot's first joint is its free root"


def adr(names):
    return [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in names]


Q_META = adr(json.load(open(RT.META))["joint_names"])  # S[:29]
Q_MJ = adr(MUJOCO_JOINT_NAMES)  # RQ
Q_HAND = {s: adr([f"{s}_hand_{f}_joint" for f in FINGERS]) for s in ("right", "left")}
body = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
BODY, MIRROR, SHIFT = {}, {}, {}
for s in ("right", "left"):
    ids, shift = [], []
    for n in links:
        b = body(f"{s}_wrist_yaw_link" if n == "hand_palm_link" else f"{s}_{n}")
        assert b >= 0, f"{s}_{n} is not a body of {SCENE}"
        ids.append(b)
        shift.append(palm[s] if n == "hand_palm_link" else np.zeros(3))
    BODY[s], SHIFT[s] = np.array(ids), np.array(shift)
    MIRROR[s] = np.array([1.0, 1.0 if s == "right" else -1.0, 1.0])
WRIST = {s: body(f"{s}_wrist_yaw_link") for s in ("right", "left")}


def pelvis_R(yaw, g):
    """The pelvis attitude from its yaw and the gravity direction in its frame (fk_check.py)."""
    pitch, roll = np.arcsin(np.clip(g[0], -1, 1)), np.arctan2(-g[1], -g[2])
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def quat_of(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def spheres(side, pos, quat, joints, qadr, fingers):
    """The side's 24 sphere centres and its jaw centre (world) with the pelvis at pos / quat and the joints set."""
    d.qpos[:] = 0
    d.qpos[:3], d.qpos[3:7] = pos, quat
    d.qpos[qadr] = joints
    d.qpos[Q_HAND[side]] = fingers
    mujoco.mj_kinematics(m, d)
    b = BODY[side]
    Rm = d.xmat[b].reshape(-1, 3, 3)
    X = d.xpos[b] + np.einsum("kij,kj->ki", Rm, SHIFT[side] + local * MIRROR[side])
    w = WRIST[side]
    return X, d.xpos[w] + d.xmat[w].reshape(3, 3) @ OFF


z = dict(np.load(D / "all.npz"))  # every array read once: an NpzFile decompresses the whole array again at each z[...]
assert "RQ" in z, f"{D}/all.npz has no reference plan: not a REACH_VIA dataset"
S, W, n = z["S"], z["W"], z["n_steps"].astype(int)
so, ro = np.r_[0, np.cumsum(n)[:-1]], np.r_[0, np.cumsum(z["n_ref"])[:-1]].astype(int)
X, XR = np.zeros((len(S), len(links), 3), np.float32), np.zeros((len(z["RQ"]), len(links), 3), np.float32)
e_jaw, e_ref = [], []
for i in range(len(n)):
    side = "right" if z["side"][i] > 0 else "left"
    p0, yaw0 = z["start"][i][:3].astype(float), float(z["start"][i][3])
    c, s = np.cos(yaw0), np.sin(yaw0)
    Rh = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    head = lambda x: np.c_[(x[:, :2] - p0[:2]) @ Rh[:2, :2], x[:, 2]]  # noqa: E731
    col = 4 if side == "right" else 7
    for k in range(so[i], so[i] + n[i]):
        q = quat_of(pelvis_R(float(W[k, 3]), S[k, 58:61].astype(float)))
        x, jaw = spheres(side, W[k, :3], q, S[k, :29], Q_META, z["hand_q"][i])
        X[k] = head(x)
        e_jaw.append(100 * np.linalg.norm(jaw - W[k, col : col + 3]))
    for k in range(ro[i], ro[i] + int(z["n_ref"][i])):
        x, jaw = spheres(side, z["RP"][k, :3], z["RP"][k, 3:7], z["RQ"][k], Q_MJ, z["hand_q"][i])
        XR[k] = head(x)
    e_ref.append(100 * np.linalg.norm(head(jaw[None])[0] - z["target"][i]))
np.savez_compressed(
    D / "arm.npz", X=X, XR=XR, radius=radius, link=np.array(links), n_steps=z["n_steps"], n_ref=z["n_ref"]
)  # fmt: skip
e_jaw, e_ref = np.array(e_jaw), np.array(e_ref)
print(f"{len(n)} segments, {len(S)} ticks, {len(XR)} reference poses, {len(links)} spheres -> {D / 'arm.npz'}")
print(f"rebuilt jaw centre against the recorded one: median {np.median(e_jaw):.3f} cm, 95 % {np.percentile(e_jaw, 95):.3f}, "
      f"max {e_jaw.max():.3f}")  # fmt: skip
ik = z["ik_err_cm"]
print(f"last reference pose's jaw against the command: median {np.median(e_ref):.3f} cm, 95 % {np.percentile(e_ref, 95):.3f} "
      f"(the plan's own IK miss: median {np.median(ik):.3f}, 95 % {np.percentile(ik, 95):.3f})")  # fmt: skip
