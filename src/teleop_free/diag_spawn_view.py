"""Real spawns, real frame positions: where does the bottle sit in the ego
frame when the VLA takes over, and how much drift is left before it leaves?

    python src/teleop_free/diag_spawn_view.py "outputs/logs/collect_*/manifest_*.jsonl"
    VLIM=1.0 ULIM=1.0 python ...      # slack to the physical frame edge,
                                      # not to the conservative central gate

Reads the collection manifests and evaluates the frustum against each spawn.

READ THE NUMBERS AS AN UPPER BOUND, not as what the camera sees. `init.robot_xy`
is the SAMPLED spawn point (grasp_gym writes it before the teleport), and the
robot then settles under SONIC by up to 0.35 m -- rejected past that -- with the
measured direction table-ward, i.e. toward MORE margin, plus some extra torso
pitch, i.e. toward less. The live post-settle check is grasp_gym's own gate
(`[view] object not in ego view at spawn`, ~12% of rounds at --spawn-edge
1.1 1.3), and `vla_round._tstart_view` records the same quantity again at
trace t=0, after the VLA prelude. What this script answers is the geometric
question those gates cannot: how much slack the spawn envelope has to begin
with, and in which direction it runs out.
"""

import glob
import json
import math
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, "/workspace/code/locomanipulation/src/teleop_free")
import paths  # noqa: E402

meta = json.load(open(f"{paths.REPO}/outputs/scene/sim/scene_meta.json"))
can = meta["objects"][meta["target_object"]]
z_obj = float(meta["surfaces"]["A"]["top"]) + can["height"] / 2.0
m = mujoco.MjModel.from_xml_path(f"{paths.REPO}/outputs/scene/sim/scene_locomani.xml")
d = mujoco.MjData(m)
cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
mujoco.mj_resetDataKeyframe(m, d, 0) if m.nkey else mujoco.mj_resetData(m, d)
mujoco.mj_forward(m, d)
C = np.array(d.cam_xpos[cam])
R = np.array(d.cam_xmat[cam]).reshape(3, 3)
P = np.array(d.xpos[pel])
Rp = np.array(d.xmat[pel]).reshape(3, 3)
half = np.tan(np.radians(float(m.cam_fovy[cam]) / 2.0))
ASPECT = 640.0 / 480.0


def uv(x, y):
    w = P + Rp @ np.array([x, y, z_obj - P[2]])
    v = R.T @ (w - C)
    if -v[2] <= 0.05:
        return 9.9, 9.9
    return v[0] / (-v[2]) / (half * ASPECT), v[1] / (-v[2]) / half


VLIM = float(os.environ.get("VLIM", 0.9))  # gym --view-gate-y
ULIM = float(os.environ.get("ULIM", 0.8))  # gym --view-gate-x


def back_slack(x, y, vlim=VLIM, ulim=ULIM):
    """extra straight-back metres before the gate breaks"""
    lo, hi = 0.0, 1.5
    if abs(uv(x, y)[1]) > vlim or abs(uv(x, y)[0]) > ulim:
        return 0.0
    for _ in range(40):
        mid = (lo + hi) / 2
        ux, uy = uv(x + mid, y)
        (lo, hi) = (mid, hi) if (abs(uy) <= vlim and abs(ux) <= ulim) else (lo, mid)
    return lo


rows = []
for f in sorted(glob.glob(sys.argv[1])):
    for line in open(f):
        r = json.loads(line)
        i = r.get("init") or {}
        if not i.get("robot_xy"):
            continue
        o = np.array(i["obj_xy"], float)
        p = np.array(i["robot_xy"], float)
        yaw = float(i["yaw"])
        dxy = o - p
        c, s = math.cos(-yaw), math.sin(-yaw)
        x = c * dxy[0] - s * dxy[1]
        y = s * dxy[0] + c * dxy[1]
        ux, uy = uv(x, y)
        rows.append((r.get("produced", False), math.hypot(*dxy), x, y, ux, uy, back_slack(x, y)))

A = np.array([[r[1], r[2], r[3], r[4], r[5], r[6]] for r in rows])
ok = np.array([r[0] for r in rows])
for tag, sel in (("all attempts", np.ones(len(A), bool)), ("produced", ok)):
    if not sel.any():
        continue
    B = A[sel]
    print(f"\n{tag}: n={sel.sum()}")
    for j, nm in enumerate(["dist  ", "x_rel ", "y_rel ", "u_x   ", "u_y   ", "backslack"]):
        v = B[:, j]
        print(
            f"  {nm} min {v.min():+.3f}  p10 {np.percentile(v, 10):+.3f}  "
            f"med {np.median(v):+.3f}  p90 {np.percentile(v, 90):+.3f}  "
            f"max {v.max():+.3f}"
        )
    bs = B[:, 5]
    for thr in (0.20, 0.30):
        print(
            f"  spawns with < {thr * 100:.0f} cm of backward slack: "
            f"{(bs < thr).sum()}/{len(bs)} ({100 * (bs < thr).mean():.0f}%)"
        )
