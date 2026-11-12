"""The side of the object the camera did not see: the seen points mirrored to the back, kept only where the camera could not see.

One look gives the front of the object, and grasps planned on a front alone sit too shallow. The object is taken to be as deep
as it is wide, unless the look shows more depth than that (a top seen from above). The seen points are mirrored through the
vertical plane at half that depth, across the horizontal view direction. A mirrored point stays only if the camera could not
have seen it: its pixel shows something nearer (the object's front, or anything else), or it falls outside the image. A
mirrored point the camera looked through is dropped, and so is one under the lowest seen point.

--how mirror   the above; no truth.
--how true     points on the object's own surface at its true pose: privileged, the ceiling of any completion.

Both write a table of the cloud's size along the view direction against the object's own (complete_<tag>.json).

    python vision_complete.py <set> --see wristb100 --pose wb100 --tag wb100c
-> <set>/vision/see_<tag>_L<k>_s<j>.npz (pts, p_o, R_o)
"""

import argparse
import glob
import json
import math
import os
import re

import numpy as np

H, W = 480, 640


def thin(p: np.ndarray, voxel: float) -> np.ndarray:
    _, i = np.unique(np.floor(p / voxel).astype(np.int64), axis=0, return_index=True)
    return p[np.sort(i)]


def span(x: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(x, [2, 98])
    return float(lo), float(hi)


def mirror(P: np.ndarray, cam_p: np.ndarray, cam_R: np.ndarray, fovy: float, depth: np.ndarray, margin: float):
    """-> the mirrored points that stay, the horizontal view direction, the depth assumed (m)."""
    v = np.median(P, axis=0) - cam_p
    v[2] = 0.0
    v /= np.linalg.norm(v)
    lat = np.array([-v[1], v[0], 0.0])
    f0, f1 = span(P @ v)
    l0, l1 = span(P @ lat)
    D = max(f1 - f0, l1 - l0)
    Q = P + (2.0 * (f0 + 0.5 * D - P @ v))[:, None] * v
    Q = Q[Q[:, 2] >= np.percentile(P[:, 2], 2)]
    c = (Q - cam_p) @ cam_R  # camera frame, z forward
    f = 0.5 * H / math.tan(math.radians(fovy) / 2)
    u = np.floor(c[:, 0] * f / np.maximum(c[:, 2], 1e-6) + W / 2).astype(int)
    r = np.floor(c[:, 1] * f / np.maximum(c[:, 2], 1e-6) + H / 2).astype(int)
    inside = (c[:, 2] > 0) & (u >= 0) & (u < W) & (r >= 0) & (r < H)
    seen = np.zeros(len(Q), bool)  # the camera looked through this point: its pixel shows something farther
    d = depth[r[inside], u[inside]]
    seen[inside] = (d > 0) & (d > c[inside, 2] + margin)
    return Q[~seen], v, D


def surface(scene_xml: str, n: int, rng) -> np.ndarray:
    """Points on the object's own meshes, in its body frame."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(scene_xml)
    b = next(i for i in range(m.nbody) if m.body_dofnum[i] == 6 and m.body(i).name.endswith("_object"))
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
    vis = [g for g in geoms if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0]
    tri, area = [], []
    for g in vis or geoms:
        k = m.geom_dataid[g]
        V = m.mesh_vert[m.mesh_vertadr[k] : m.mesh_vertadr[k] + m.mesh_vertnum[k]]
        F = m.mesh_face[m.mesh_faceadr[k] : m.mesh_faceadr[k] + m.mesh_facenum[k]]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        T = (V @ R.reshape(3, 3).T + m.geom_pos[g])[F]
        tri.append(T)
        area.append(0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1))
    tri, area = np.concatenate(tri), np.concatenate(area)
    i = rng.choice(len(tri), n, p=area / area.sum())
    a, b_ = rng.random(n), rng.random(n)
    flip = a + b_ > 1
    a[flip], b_[flip] = 1 - a[flip], 1 - b_[flip]
    return tri[i, 0] + a[:, None] * (tri[i, 1] - tri[i, 0]) + b_[:, None] * (tri[i, 2] - tri[i, 0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("--see", required=True, help="the look's own files (camera pose, depth)")
    ap.add_argument("--pose", required=True, help="the object clouds to complete")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--how", default="mirror", choices=["mirror", "true"])
    ap.add_argument("--objects", default=os.environ.get("GRASP_ROOM_OBJECTS", ""))
    ap.add_argument(
        "--margin", type=float, default=0.01, help="a pixel this much farther than the point saw through it"
    )
    ap.add_argument("--voxel", type=float, default=0.003)
    ap.add_argument("--min-points", type=int, default=100)
    a = ap.parse_args()
    V = os.path.join(a.set, "vision")
    rng = np.random.default_rng(0)
    rows, surf = [], {}
    for f in sorted(glob.glob(os.path.join(V, f"see_{a.pose}_L*_s*.npz"))):
        k, j = map(int, re.search(r"_L(\d+)_s(\d+)\.npz$", f).groups())
        z, raw = np.load(f), np.load(os.path.join(V, f"see_{a.see}_L{k}_s{j}.npz"))
        P = z["pts"].astype(np.float64)
        row = dict(layout=k, slot=j, n_seen=len(P))
        slot = json.load(open(os.path.join(a.set, f"layout_{k}", "layout.json")))["slots"][j]
        if a.objects:  # the object's own surface: the table's truth column, and the cloud of --how true
            if slot["id"] not in surf:
                surf[slot["id"]] = surface(
                    os.path.join(a.objects, "scenes", f"obj_{slot['id']:03d}", "scene_locomani.xml"), 4000, rng
                )
            T = z["p_o"] + surf[slot["id"]] @ z["R_o"].T
        pts = P
        if a.how == "true":
            pts = T
        elif len(P) >= a.min_points:
            lb = os.path.join(V, f"lb_{a.see}_L{k}_s{j}.npz")
            depth = np.load(lb)["depth_lb"] if os.path.exists(lb) else raw["depth_rs"]
            Q, v, D = mirror(P, raw["cam_p"], raw["cam_R"], float(raw["fovy"]), depth, a.margin)
            pts = np.vstack([P, Q])
            row.update(n_back=len(Q), depth_cm=100 * D)
            if a.objects:
                (s0, s1), (t0, t1), (c0, c1) = span(P @ v), span(T @ v), span(pts @ v)
                row.update(seen_cm=100 * (s1 - s0), true_cm=100 * (t1 - t0), full_cm=100 * (c1 - c0),
                           back_err_cm=100 * (c1 - t1), back_err_seen_cm=100 * (s1 - t1), front_err_cm=100 * (s0 - t0))  # fmt: skip
        pts = thin(pts, a.voxel) if len(pts) else pts
        row["n"] = len(pts)
        np.savez_compressed(os.path.join(V, f"see_{a.tag}_L{k}_s{j}.npz"), pts=pts.astype(np.float32), p_o=z["p_o"],
                            R_o=z["R_o"])  # fmt: skip
        rows.append(row)
    json.dump(rows, open(os.path.join(V, f"complete_{a.tag}.json"), "w"))
    r = [x for x in rows if "true_cm" in x]
    if r:
        med = lambda key: float(np.median([x[key] for x in r]))  # noqa: E731
        print(f"{len(r)} completed; along the view, cm: object {med('true_cm'):.1f}, seen {med('seen_cm'):.1f}, "
              f"completed {med('full_cm'):.1f}; back face error: seen {med('back_err_seen_cm'):.1f}, "
              f"completed {med('back_err_cm'):.1f}; |completed| {float(np.median([abs(x['back_err_cm']) for x in r])):.1f}")  # fmt: skip
    print(len(rows), "clouds")


if __name__ == "__main__":
    main()
