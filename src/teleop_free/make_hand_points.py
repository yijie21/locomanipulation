"""Bake the hand's shape into a point cloud, in the palm frame.

The adaptive grasp has to answer two questions for every candidate palm
orientation, thousands of times per episode: do the fingers clear the table,
and does the jaw straddle the object. Both are questions about where the hand's
geometry ends up once rotated, and the hand is 85k vertices of mesh — far too
much to rotate in a control loop, and MuJoCo is not loaded in the episode
process anyway.

So sample it once, offline, expressed relative to the GRASP CENTRE rather than
the wrist: that is the point the plan pins to the object, so it is the natural
origin for asking "and where is everything else, then".

    python src/teleop_free/make_hand_points.py [--n 400]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diag_grasp_pose import SCENE_META  # noqa: E402
from export_grasp_model import mesh_triangles  # noqa: E402
from primitives_const import GRASP_CENTRE_OFFSET_CONST  # noqa: E402

GHOST_BODY = "ghost_hand"
import paths
from primitives_const import HAND as _HAND

OUT = f"{paths.REPO}/outputs/scene/" + ("hand_points_dex1.npy" if _HAND == "dex1" else "hand_points.npy")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    meta = json.load(open(SCENE_META))
    model = mujoco.MjModel.from_xml_path(meta["scene_xml"])
    data = mujoco.MjData(model)

    gb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GHOST_BODY)
    # Park it at the origin with no rotation, so the vertices come out in the
    # ghost body's own frame — which is the wrist frame, by construction.
    data.mocap_pos[model.body_mocapid[gb]] = [0.0, 0.0, 0.0]
    data.mocap_quat[model.body_mocapid[gb]] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    pts = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != gb:
            continue
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        pts.append(mesh_triangles(model, data, g)[0])
    pts = np.vstack(pts) - np.asarray(GRASP_CENTRE_OFFSET_CONST)

    # Farthest-point sampling, not random: what matters is the hull — the
    # fingertips that hit the table and the jaw faces that hit the object — and
    # uniform random sampling of a dense mesh spends nearly every point on the
    # palm, which is the one part that never decides anything.
    keep = [int(np.argmin(pts[:, 2]))]  # start from the lowest point
    d = np.linalg.norm(pts - pts[keep[0]], axis=1)
    for _ in range(args.n - 1):
        i = int(np.argmax(d))
        keep.append(i)
        d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    out = pts[keep]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, out.astype(np.float32))
    print(f"{len(pts)} vertices -> {len(out)} points")
    print(f"  extent about the grasp centre {np.round(out.min(0), 3)} .. {np.round(out.max(0), 3)} m")
    print(f"  lowest point sits {-out[:, 2].min() * 100:.1f} cm below it")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
