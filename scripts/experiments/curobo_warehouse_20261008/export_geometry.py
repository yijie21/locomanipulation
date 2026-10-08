"""CuRobo warehouse study, stage 2a: export the mars upper-body collision geometry for sphere fitting.

CuRobo runs in another venv (SIMPLE's) and only reads this file. For each upper-body link it writes the convex
hulls of the link's MuJoCo collision geoms in the link frame (the frame CuRobo's URDF uses, checked by
`fk_samples`). The Dex1 fingers are folded into `<side>_wrist_yaw_link` at a fixed slide (`--jaw`), since the
plan holds the jaws open. It also writes the MJCF joint limits and FK samples of a few links, so the CuRobo side
can verify that the URDF it loads has the same kinematics as the simulated model.

    PYTHONPATH=src:scripts/experiments/curobo_warehouse_20261008 "$VENV_SIM_PY" \
        scripts/experiments/curobo_warehouse_20261008/export_geometry.py \
        outputs/scene/warehouse_box_45x32x33_mars/scene_locomani.xml \
        outputs/curobo_warehouse_20261008/mars_geometry.json
"""

from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np
from cases import quat_wxyz_to_matrix
from scipy.spatial import ConvexHull

UPPER = [
    "torso_link",
    *[
        f"{s}_{link}"
        for s in ("left", "right")
        for link in (
            "shoulder_pitch_link",
            "shoulder_roll_link",
            "shoulder_yaw_link",
            "elbow_link",
            "wrist_roll_link",
            "wrist_pitch_link",
            "wrist_yaw_link",
        )
    ],
]
FK_LINKS = ["torso_link", "left_elbow_link", "right_elbow_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]


def transform(m, kind, idx):
    T = np.eye(4)
    if kind == "geom":
        T[:3, :3] = quat_wxyz_to_matrix(m.geom_quat[idx])
        T[:3, 3] = m.geom_pos[idx]
    else:
        T[:3, :3] = quat_wxyz_to_matrix(m.body_quat[idx])
        T[:3, 3] = m.body_pos[idx]
    return T


def geom_points(m, g):
    """Points whose convex hull is the geom's collision shape, in the geom frame."""
    kind, size = int(m.geom_type[g]), m.geom_size[g]
    if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
        mid = m.geom_dataid[g]
        return m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
    if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        return np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * size
    if kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        a = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        ring = np.c_[size[0] * np.cos(a), size[0] * np.sin(a)]
        return np.vstack([np.c_[ring, np.full(32, -size[1])], np.c_[ring, np.full(32, size[1])]])
    if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        u = np.random.default_rng(0).normal(size=(200, 3))
        u = size[0] * u / np.linalg.norm(u, axis=1, keepdims=True)
        return np.vstack([u + [0, 0, -size[1]], u + [0, 0, size[1]]])
    raise ValueError(f"geom {g}: unsupported type {kind}")


def hull(points):
    h = ConvexHull(points)
    used = np.unique(h.simplices)
    remap = -np.ones(len(points), int)
    remap[used] = np.arange(len(used))
    return points[used], remap[h.simplices]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("scene")
    ap.add_argument("out")
    ap.add_argument("--jaw", type=float, default=0.0245, help="finger slide qpos folded into the wrist link")
    ap.add_argument("--fk-samples", type=int, default=20)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(args.scene)
    d = mujoco.MjData(m)
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
    collides = lambda g: m.geom_contype[g] or m.geom_conaffinity[g]  # noqa: E731

    links = {}
    for link in UPPER:
        b = bid(link)
        parts = [(b, np.eye(4))]
        if link.endswith("wrist_yaw_link"):
            side = link.split("_")[0]
            for k in (1, 2):
                fb = bid(f"{side}_dex1_finger_{k}_link")
                T = transform(m, "body", fb)
                T[:3, 3] += T[:3, :3] @ m.jnt_axis[m.body_jntadr[fb]] * args.jaw
                parts.append((fb, T))
        geoms = []
        for body, T_link_body in parts:
            for g in range(m.ngeom):
                if m.geom_bodyid[g] != body or not collides(g):
                    continue
                T = T_link_body @ transform(m, "geom", g)
                pts = geom_points(m, g) @ T[:3, :3].T + T[:3, 3]
                verts, faces = hull(pts)
                geoms.append(
                    dict(
                        geom=int(g),
                        body=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body),
                        vertices=verts.tolist(),
                        faces=faces.tolist(),
                    )
                )
        links[link] = geoms
        print(link, len(geoms), "geoms")

    hinges = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    joints = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): [float(v) for v in m.jnt_range[j]] for j in hinges}
    qadr = [m.jnt_qposadr[j] for j in hinges]
    pelvis = bid("pelvis")
    rng = np.random.default_rng(1)
    samples = []
    for _ in range(args.fk_samples):
        q = np.array([rng.uniform(lo, hi) for lo, hi in joints.values()])
        d.qpos[:] = 0
        d.qpos[3] = 1.0
        d.qpos[qadr] = q
        mujoco.mj_kinematics(m, d)
        Rp, pp = d.xmat[pelvis].reshape(3, 3), d.xpos[pelvis]
        poses = {}
        for link in FK_LINKS:
            b = bid(link)
            poses[link] = dict(pos=(Rp.T @ (d.xpos[b] - pp)).tolist(), rot=(Rp.T @ d.xmat[b].reshape(3, 3)).tolist())
        samples.append(dict(joints=q.tolist(), poses=poses))

    with open(args.out, "w") as f:
        json.dump(dict(scene=args.scene, jaw=args.jaw, links=links, joints=joints, fk_samples=samples), f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
