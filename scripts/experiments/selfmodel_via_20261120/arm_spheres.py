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

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_fk import ArmFK  # noqa: E402

D, SCENE = Path(sys.argv[1]), Path(sys.argv[2])
FK = ArmFK(SCENE)
links, radius = FK.links, FK.radius


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
        x, jaw = FK.spheres(side, W[k, :3], q, S[k, :29], "meta", z["hand_q"][i])
        X[k] = head(x)
        e_jaw.append(100 * np.linalg.norm(jaw - W[k, col : col + 3]))
    for k in range(ro[i], ro[i] + int(z["n_ref"][i])):
        x, jaw = FK.spheres(side, z["RP"][k, :3], z["RP"][k, 3:7], z["RQ"][k], "mj", z["hand_q"][i])
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
