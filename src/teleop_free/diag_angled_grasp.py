"""Show what a tilted, standing grasp would look like, before building it.

The grasp in the pipeline holds the palm level and slides in horizontally. That
forces the crouch: with the palm flat the grip centre sits level with the wrist,
so reaching a 0.62 m table means putting the pelvis near 0.65 m, and the squat
is held by parking the free hand at the floor for the whole approach.

A person does not do that. The hand comes down at an angle, the web of the
thumb rides the top shoulder of the object, and the fingers close on the side.
The wrist stays high; only the fingers go low.

Geometrically that is one rotation: pitch the palm nose-down about its own y
axis, which is the pinch axis, so the jaw still straddles the cylinder. What it
buys is GRASP_CENTRE_OFFSET[0] * sin(pitch) of drop between the wrist and the
grip point — the arm reaches down without the body going down.

This renders the candidate at several pitches so the geometry can be judged
before any of it is wired into the episode.

    python src/teleop_free/diag_angled_grasp.py [--pitch 0,20,35,50] [--out DIR]
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
import paths  # noqa: E402
from diag_grasp_pose import SCENE_META, stand_pose  # noqa: E402
from export_grasp_model import cylinder_triangles, mesh_triangles  # noqa: E402
from primitives import (  # noqa: E402
    ARM_REACH_LIMIT,
    GRASP_CENTRE_OFFSET,
    SHOULDER_OFFSET,
)

GHOST_BODY = "ghost_hand"
STANDING_PELVIS_Z = 0.793  # what the robot holds with no crouch commanded
# Where on the cylinder to close. Level with the top shoulder is where a hand
# naturally lands on a can; a few centimetres down keeps the fingers on the
# barrel rather than sliding off the rim.
GRIP_BELOW_TOP = 0.030

VIEWS = [("front", 0.0, 8.0), ("side", 90.0, 8.0), ("top", 0.0, 88.0), ("iso", 40.0, 26.0)]


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)

    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def palm_frame(heading: float, pitch: float) -> np.ndarray:
    """Palm rotation: yaw to the heading, then nose-down about the pinch axis.

    The level palm is a pure yaw — +x along the approach, +y the pinch/entry
    axis, +z world up. Pitching about +y tips +x downward and leaves the jaw
    where it was, which is the whole point: a cylinder is symmetric about its
    own axis, so a jaw tilted off horizontal still closes on a chord of the
    same barrel.
    """
    yaw = rot_z(heading)
    c, s = np.cos(pitch), np.sin(pitch)
    # About +y, sign chosen so +x tips DOWN (verified by the printed fingertip
    # height, which has to fall as the pitch grows).
    pitch_y = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    return yaw @ pitch_y


def quat_of(rot: np.ndarray) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, rot.flatten())

    return q


def rasterise(groups, path, az, el, dist=0.55, size=460, centre=None):
    """Tiny z-buffer render. No libraries: the sim's env has no mesh viewer."""
    from PIL import Image

    centre = np.zeros(3) if centre is None else np.asarray(centre, float)
    a, e = np.deg2rad(az), np.deg2rad(el)
    fwd = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    eye = centre + fwd * dist
    z = fwd
    up = np.array([0.0, 0.0, 1.0]) if abs(el) < 85 else np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rot = np.stack([x, y, z])

    img = np.ones((size, size, 3)) * 0.10
    zbuf = np.full((size, size), 1e9)
    focal = 2.3 * size / 2
    for verts, faces, colour in groups:
        p = (verts - eye) @ rot.T
        d = -p[:, 2]
        dd = np.where(d > 1e-6, d, 1e-6)
        u = (p[:, 0] / dd) * focal + size / 2
        v = size / 2 - (p[:, 1] / dd) * focal
        a_, b_, c_ = faces[:, 0], faces[:, 1], faces[:, 2]
        n = np.cross(verts[b_] - verts[a_], verts[c_] - verts[a_])
        ln = np.linalg.norm(n, axis=1)
        ln[ln == 0] = 1.0
        shade = np.clip(np.abs((n / ln[:, None]) @ np.array([0.4, 0.5, 0.75])), 0.2, 1.0)
        for i in range(len(faces)):
            ia, ib, ic = a_[i], b_[i], c_[i]
            if d[ia] <= 0 or d[ib] <= 0 or d[ic] <= 0:
                continue
            xs = (u[ia], u[ib], u[ic])
            ys = (v[ia], v[ib], v[ic])
            x0, x1 = int(max(0, min(xs))), int(min(size - 1, max(xs)))
            y0, y1 = int(max(0, min(ys))), int(min(size - 1, max(ys)))
            if x1 < x0 or y1 < y0:
                continue
            px, py = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            e0 = (xs[1] - xs[0]) * (py - ys[0]) - (ys[1] - ys[0]) * (px - xs[0])
            e1 = (xs[2] - xs[1]) * (py - ys[1]) - (ys[2] - ys[1]) * (px - xs[1])
            e2 = (xs[0] - xs[2]) * (py - ys[2]) - (ys[0] - ys[2]) * (px - xs[2])
            m = ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))
            if not m.any():
                continue
            zz = (d[ia] + d[ib] + d[ic]) / 3
            reg = zbuf[y0 : y1 + 1, x0 : x1 + 1]
            m &= zz < reg
            if not m.any():
                continue
            reg[m] = zz
            img[y0 : y1 + 1, x0 : x1 + 1][m] = colour * shade[i]

    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def slab(centre, half, top, thickness=0.02):
    """A box for the table top, so the render shows what the fingers clear."""
    c = np.asarray(centre, float)
    h = np.asarray(half, float)
    lo = np.array([c[0] - h[0], c[1] - h[1], top - thickness])
    hi = np.array([c[0] + h[0], c[1] + h[1], top])
    v = np.array(
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ]
    )
    f = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 5, 1],
            [0, 4, 5],
            [1, 6, 2],
            [1, 5, 6],
            [2, 7, 3],
            [2, 6, 7],
            [3, 4, 0],
            [3, 7, 4],
        ]
    )

    return v, f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pitch", default="0,20,35,50", help="comma-separated degrees, nose-down")
    ap.add_argument("--out", default=f"{paths.REPO}/outputs/models/angled")
    ap.add_argument("--side", default="right", choices=["left", "right"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    meta = json.load(open(SCENE_META))
    model = mujoco.MjModel.from_xml_path(meta["scene_xml"])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    obj_key = meta["target_object"]
    obj = meta["objects"][obj_key]
    surf = meta["surfaces"][obj["surface"]]
    centre = np.array(obj["init_pos"], dtype=float)
    he = np.asarray(obj["box"]["half_extent"], dtype=float)
    radius, half_h = float(he[0]), float(he[2])

    stand, yaw = stand_pose(meta, obj_key)
    shoulder = SHOULDER_OFFSET * np.array([1.0, 1.0 if args.side == "right" else -1.0, 1.0])

    # Grip on the barrel, a little below the rim.
    top = centre[2] + half_h
    grip = np.array([centre[0], centre[1], top - GRIP_BELOW_TOP])

    gb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GHOST_BODY)
    mocap = model.body_mocapid[gb]
    ghost_geoms = [
        g for g in range(model.ngeom) if model.geom_bodyid[g] == gb and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
    ]

    tv, tf = slab(surf["xy"], surf["half_extent"], surf["top"])
    print(f"can  phi{radius * 200:.1f} x h{half_h * 200:.0f} cm at {np.round(centre, 3)}, top {top:.3f}")
    print(f"grip {np.round(grip, 3)}  ({(grip[2] - surf['top']) * 100:.1f} cm above the table)")
    print(f"standing pelvis {STANDING_PELVIS_Z:.3f}, no crouch\n")
    print(f"{'pitch':>6} {'wrist z':>8} {'drop':>7} {'shoulder':>9} {'fingertip z':>12}  verdict")

    for deg in [float(x) for x in args.pitch.split(",")]:
        rot = palm_frame(yaw, np.deg2rad(deg))
        wrist = grip - rot @ GRASP_CENTRE_OFFSET
        # Reach demand from the standing shoulder, the number that decides
        # whether the crouch can be dropped at all.
        local = rot_z(yaw).T @ np.array([wrist[0] - stand[0], wrist[1] - stand[1], wrist[2] - STANDING_PELVIS_Z])
        demand = float(np.linalg.norm(local - shoulder))

        data.mocap_pos[mocap] = wrist
        data.mocap_quat[mocap] = quat_of(rot)
        mujoco.mj_forward(model, data)

        gv, gf, lowest = [], [], 1e9
        for g in ghost_geoms:
            w, f = mesh_triangles(model, data, g)
            gf.append(f + sum(len(x) for x in gv))
            gv.append(w)
            lowest = min(lowest, float(w[:, 2].min()))
        gv, gf = np.vstack(gv), np.vstack(gf)

        ov, of = cylinder_triangles(centre, np.eye(3), radius, half_h)
        groups = [
            (tv, tf, np.array([0.55, 0.35, 0.20])),
            (ov, of, np.array([0.90, 0.35, 0.15])),
            (gv, gf, np.array([1.00, 0.25, 0.80])),
        ]

        ok = demand <= ARM_REACH_LIMIT and lowest >= surf["top"] - 0.002
        why = (
            "reachable standing"
            if ok
            else ("arm too short" if demand > ARM_REACH_LIMIT else "fingers through the table")
        )
        print(f"{deg:5.0f}d {wrist[2]:8.3f} {wrist[2] - grip[2]:7.3f} {demand:9.3f} {lowest:12.3f}  {why}")

        for name, az, el in VIEWS:
            rasterise(
                groups, os.path.join(args.out, f"pitch{deg:02.0f}_{name}.png"), az + np.rad2deg(yaw), el, centre=grip
            )

    print(f"\nwrote {args.out}/pitch*_{{front,side,top,iso}}.png (arm limit {ARM_REACH_LIMIT} m from the shoulder)")


if __name__ == "__main__":
    main()
