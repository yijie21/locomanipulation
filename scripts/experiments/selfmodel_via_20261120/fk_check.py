"""Can the whole arm's path be rebuilt from the self-model data? FK from each tick's joints (S[:29]) and pelvis pose
(W[0:3] world xyz, W[3] yaw, roll/pitch from S[58:61], gravity in the pelvis frame) -> the right and left jaw centres,
against the recorded ones (W[4:7], W[7:10]). Run from the repo root: python -I fk_check.py <dataset dir> [n segments]"""

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path.cwd()
sys.path.insert(0, str(REPO / "src"))
from locomani.backends.scalebfm import runtime as RT  # noqa: E402
from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES  # noqa: E402

D, N = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 300
z = np.load(D / "all.npz")
S, W, n, so = z["S"], z["W"], z["n_steps"].astype(int), np.r_[0, np.cumsum(z["n_steps"].astype(int))[:-1]]
XML = (
    REPO / "outputs/scene" / (sys.argv[3] if len(sys.argv) > 3 else "walk_mars") / "scene_locomani.xml"
)  # the scene the data was recorded in (REACH_SCENE_DIR)
m = mujoco.MjModel.from_xml_path(str(XML))
d = mujoco.MjData(m)
free = m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
import json  # noqa: E402

JN = json.load(open(RT.META))["joint_names"]  # the state's joint order (runtime.Sim: meta["joint_names"])
qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JN]
print("same order as MUJOCO_JOINT_NAMES:", list(JN) == list(MUJOCO_JOINT_NAMES))
wr = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_wrist_yaw_link") for s in ("right", "left")}
pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
print("FK model", XML.parent.name, "free root", free, "bodies", m.nbody, "joints mapped", len(qadr))


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def pelvis_R(yaw, g):
    pitch, roll = np.arcsin(np.clip(g[0], -1, 1)), np.arctan2(-g[1], -g[2])
    cp, sp, cr, sr = np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz(yaw) @ Ry @ Rx


rng = np.random.default_rng(0)
segs = rng.choice(len(n), size=min(N, len(n)), replace=False)
errs = {o: {"right": [], "left": []} for o in ("dex1", "dex3")}
res = {"wrist": [], "pelvis": [], "world": []}  # dex3 right residual, recorded - rebuilt, in each frame
OFFS = {"dex1": np.array([0.1515, 0, 0]), "dex3": np.array([0.0415, 0, 0])}
for i in segs:
    for k in range(so[i], so[i] + n[i]):
        d.qpos[:] = 0
        if free:
            d.qpos[3] = 1
        for a, v in zip(qadr, S[k, :29]):
            d.qpos[a] = v
        mujoco.mj_kinematics(m, d)
        Rp0, p0 = d.xmat[pel].reshape(3, 3), d.xpos[pel]
        Rp = pelvis_R(float(W[k, 3]), S[k, 58:61].astype(float))
        for s, col in (("right", 4), ("left", 7)):
            Rw0, pw0 = d.xmat[wr[s]].reshape(3, 3), d.xpos[wr[s]]
            for o, off in OFFS.items():
                jaw_pel = Rp0.T @ (pw0 + Rw0 @ off - p0)
                jaw = W[k, :3] + Rp @ jaw_pel
                errs[o][s].append(100 * np.linalg.norm(jaw - W[k, col : col + 3]))
                if o == "dex3" and s == "right":
                    r = W[k, col : col + 3] - jaw
                    res["world"].append(r)
                    res["pelvis"].append(Rp.T @ r)
                    res["wrist"].append((Rp @ Rp0.T @ Rw0).T @ r)
for o in errs:
    for s in errs[o]:
        e = np.array(errs[o][s])
        print(
            f"offset {o} {s:5s} ticks {len(e)}  error cm median {np.median(e):.3f}  p95 {np.percentile(e, 95):.3f}  max {e.max():.3f}"
        )
for f, v in res.items():
    v = 100 * np.array(v)
    print(f"residual in {f:6s} frame: mean cm {np.round(v.mean(0), 3)}  spread (std) cm {np.round(v.std(0), 3)}")
