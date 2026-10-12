"""TODO 21.3.4: the right Dex1 hand with the jaw fully open (slide +0.0245, 9.5 cm), as surface points in the tool frame.

outputs/scene/hand_points_dex1.npy is sampled from the ghost hand, whose fingers sit at slide 0 (about 4.6 cm open), so
every graspnet grasp wider than that collided with its own fingers (one candidate of 184 on object 002, 2026-10-08).
Tool frame = wrist_yaw_link frame shifted by ready_grasp_policy.OFFSET (the grip centre), palm axes.

    PYTHONPATH=src "$VENV_SIM_PY" scripts/experiments/graspnet_bottle_20261008/hand_cloud.py outputs/graspnet_bottle_20261008
"""

import sys
from pathlib import Path

import mujoco
import numpy as np

from locomani import paths
from locomani.tasks.bottle.ready_grasp_policy import OFFSET

BODIES = ("right_wrist_yaw_link", "right_dex1_finger_1_link", "right_dex1_finger_2_link")
OPEN = 0.0245

m = mujoco.MjModel.from_xml_path(str(Path(paths.REPO) / "outputs" / "scene" / "sim_lowcom" / "scene_locomani.xml"))
d = mujoco.MjData(m)
for j in ("right_hand_gripper_1_joint", "right_hand_gripper_2_joint"):
    d.qpos[m.jnt_qposadr[m.joint(j).id]] = OPEN
mujoco.mj_kinematics(m, d)
w = m.body("right_wrist_yaw_link").id
Rw, pw = d.xmat[w].reshape(3, 3), d.xpos[w]
rng = np.random.default_rng(0)
points, part = [], []
for g in range(m.ngeom):
    body = m.body(m.geom_bodyid[g]).name
    if (
        body not in BODIES
        or not (m.geom_contype[g] or m.geom_conaffinity[g])
        or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
    ):
        continue
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
    f = m.mesh_face[m.mesh_faceadr[mid] : m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    n = int(np.clip(area.sum() / 2e-5, 50, 3000))
    k = rng.choice(len(f), n, p=area / area.sum())
    a, b = rng.random((2, n))
    flip = a + b > 1
    a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
    s = tri[k, 0] + a[:, None] * (tri[k, 1] - tri[k, 0]) + b[:, None] * (tri[k, 2] - tri[k, 0])
    world = s @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
    points.append((world - pw) @ Rw - OFFSET)
    part += [body] * n
pts = np.vstack(points)
out = Path(sys.argv[1]) / "hand_open_points.npy"
np.save(out, pts.astype(np.float32))
fingers = np.array([p != "right_wrist_yaw_link" for p in part])
inner = np.abs(pts[fingers, 1]).min()
print(
    f"{len(pts)} points; extent min {np.round(pts.min(0), 3)} max {np.round(pts.max(0), 3)}; finger inner faces at +-{inner:.4f} m"
)
print(f"wrote {out}")
