"""Export the planned grasp — hand and object — as a model to open elsewhere.

Renders are a projection and a still frame is one viewpoint; a model can be
turned over. This writes the ghost hand at the pose `grasping.grasp_pose`
computes and the target object at its own pose, in world geometry, recentred on
the grip point so the file opens near the origin.

Wavefront OBJ, because it needs no libraries beyond what the sim already has
and every viewer reads it — Windows 3D Viewer, macOS Preview, Blender,
three.js. Two named groups (`gripper`, `object`) with an MTL beside it, so the
two are separable and coloured.

    python src/teleop_free/export_grasp_model.py [--out outputs/models]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import grasping  # noqa: E402
import paths
from diag_grasp_pose import SCENE_META, place_robot, stand_pose  # noqa: E402
from primitives import GRASP_CENTRE_OFFSET  # noqa: E402

GHOST_BODY = "ghost_hand"
CYL_SEGMENTS = 48


def mesh_triangles(model, data, g):
    """World-space (vertices, faces) of one mesh geom."""
    mi = model.geom_dataid[g]
    va, vn = model.mesh_vertadr[mi], model.mesh_vertnum[mi]
    fa, fn = model.mesh_faceadr[mi], model.mesh_facenum[mi]
    v = model.mesh_vert[va : va + vn]
    f = model.mesh_face[fa : fa + fn]
    w = v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]

    return w, f


def cylinder_triangles(pos, rot, radius, half_h, n=CYL_SEGMENTS):
    """A closed cylinder, since MuJoCo primitives carry no mesh to export."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([radius * np.cos(a), radius * np.sin(a), np.zeros(n)], 1)
    top = ring + [0, 0, half_h]
    bot = ring - [0, 0, half_h]
    verts = np.vstack([top, bot, [[0, 0, half_h]], [[0, 0, -half_h]]])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [[i, j, n + j], [i, n + j, n + i]]  # side
        faces += [[2 * n, j, i]]  # top cap
        faces += [[2 * n + 1, n + i, n + j]]  # bottom cap

    return verts @ rot.T + pos, np.array(faces)


def write_obj(path, groups) -> None:
    """groups: [(name, material, verts, faces)] — one OBJ, one MTL beside it."""
    mtl = os.path.splitext(os.path.basename(path))[0] + ".mtl"
    with open(path, "w") as f:
        f.write("# planned grasp: gripper at the pose, object where it stands\n")
        f.write(f"mtllib {mtl}\n")
        base = 1  # OBJ indices are 1-based and file-global
        for name, material, verts, faces in groups:
            f.write(f"\ng {name}\nusemtl {material}\n")
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for t in faces:
                f.write(f"f {t[0] + base} {t[1] + base} {t[2] + base}\n")
            base += len(verts)

    with open(os.path.join(os.path.dirname(path), mtl), "w") as f:
        for name, kd, alpha in (("gripper", (1.0, 0.15, 0.75), 0.85), ("object", (0.82, 0.28, 0.22), 1.0)):
            f.write(f"newmtl {name}\nKd {kd[0]} {kd[1]} {kd[2]}\nKa 0 0 0\nKs 0.1 0.1 0.1\nNs 20\nd {alpha}\n\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{paths.REPO}/outputs/models")
    ap.add_argument("--side", default="right", choices=["left", "right"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    meta = json.load(open(SCENE_META))
    model = mujoco.MjModel.from_xml_path(meta["scene_xml"])
    data = mujoco.MjData(model)

    obj_key = meta["target_object"]
    obj = meta["objects"][obj_key]
    surf = meta["surfaces"][obj["surface"]]

    stand, yaw = stand_pose(meta, obj_key)
    place_robot(model, data, stand, yaw)
    mujoco.mj_forward(model, data)
    plan = grasping.grasp_pose(
        obj_pos=np.array(obj["init_pos"]),
        obj_quat=obj["init_quat"],
        box=obj["box"],
        surface_top=surf["top"],
        pelvis_xy=stand,
        pelvis_yaw=yaw,
        approach_dir=(np.cos(yaw), np.sin(yaw)),
    )
    if not plan.ok:
        sys.exit(f"no grasp to export: {plan.why}")
    print(grasping.describe(plan))

    # Place the ghost exactly as the episode does.
    head = float(np.arctan2(plan.approach[1], plan.approach[0]))
    q = np.array([np.cos(head / 2), 0.0, 0.0, np.sin(head / 2)])
    rot = np.empty(9)
    mujoco.mju_quat2Mat(rot, q)
    rot = rot.reshape(3, 3)
    gb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GHOST_BODY)
    data.mocap_pos[model.body_mocapid[gb]] = plan.grip - rot @ GRASP_CENTRE_OFFSET
    data.mocap_quat[model.body_mocapid[gb]] = q
    mujoco.mj_forward(model, data)

    # Recentre on the grip point: the scene's world origin is metres away and
    # some viewers frame the origin rather than the geometry.
    origin = np.asarray(plan.grip, dtype=float)

    gv, gf = [], []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != gb:
            continue
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        w, f = mesh_triangles(model, data, g)
        gf.append(f + sum(len(x) for x in gv))
        gv.append(w - origin)
    gverts, gfaces = np.vstack(gv), np.vstack(gf)

    ob = obj["body_id"]
    cyl = [g for g in range(model.ngeom) if model.geom_bodyid[g] == ob]
    g0 = cyl[0]
    if model.geom_type[g0] == mujoco.mjtGeom.mjGEOM_CYLINDER:
        r, hh = float(model.geom_size[g0][0]), float(model.geom_size[g0][1])
        overts, ofaces = cylinder_triangles(data.geom_xpos[g0] - origin, data.geom_xmat[g0].reshape(3, 3), r, hh)
    else:
        ov, of = [], []
        for g in cyl:
            if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            w, f = mesh_triangles(model, data, g)
            of.append(f + sum(len(x) for x in ov))
            ov.append(w - origin)
        overts, ofaces = np.vstack(ov), np.vstack(of)

    path = os.path.join(args.out, "grasp_pose.obj")
    write_obj(path, [("gripper", "gripper", gverts, gfaces), ("object", "object", overts, ofaces)])

    print(f"\nwrote {path}")
    print(f"      {path[:-4]}.mtl")
    print(f"  gripper {len(gverts):6d} verts {len(gfaces):6d} faces")
    print(f"  {obj_key:8s}{len(overts):6d} verts {len(ofaces):6d} faces")
    print("  origin is the grip point; +z is up, +x is the approach direction rotated to world")
    print(f"  object spans {np.round(overts.min(0), 3)} to {np.round(overts.max(0), 3)} m about it")


if __name__ == "__main__":
    main()
