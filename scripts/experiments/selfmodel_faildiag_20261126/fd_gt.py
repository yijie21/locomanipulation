"""selfmodel_faildiag_20261126: grasps on the TRUE object (measurement tool for the gtgrasp / gtservo groups).

Per (task, episode): the episode's start (hg_run.HGDriver.reset_hg: the scene and our jitter, a function of the episode
alone; no stepping), the target object's pose from the simulator, its visual mesh geoms (MuJoCo group 1; every mesh
geom of the object's body tree if it has none; box / cylinder / sphere geoms sampled analytically) sampled on the
surface, area-weighted, GGX_PTS (2048) points (rng seed = episode), sent to the GraspGen-X service with the look's
request (num_grasps 400, seed 0, the look's 5 cm outlier threshold is the server's). Saved: the grasps in the object
frame (grasp_G_obj, best score first), their scores, the object pose at the start (obj_T_reset), the cloud.
fd_run.py moves them to the object's simulator pose at the look and hands them to the plan in place of the look's.

    cd $G; LC_CODE_ROOT=$C0 PYTHONPATH=$G:$C0/src HG_GGX_PORT=18873 python fd_gt.py --task T --ep-list 30,31,...
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

import fd_run as FR

LC, LR, HG = FR.LC, FR.LR, FR.HG
GGX_PTS = 2048


def tri_area(v0, v1, v2):
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)


def sample_geom(m, d, g: int, rng, mj) -> tuple[np.ndarray, np.ndarray]:
    """(triangles (n,3,3) world, areas) for one geom; primitives as a fine triangle mesh of their surface."""
    R, p = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
    t = int(m.geom_type[g])
    if t == mj.mjtGeom.mjGEOM_MESH:
        mid = int(m.geom_dataid[g])
        v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        f = m.mesh_face[m.mesh_faceadr[mid] : m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
        T = v[f] @ R.T + p
    else:
        s = m.geom_size[g]
        pts = []
        if t == mj.mjtGeom.mjGEOM_BOX:
            u = rng.uniform(-1, 1, (4000, 3))
            ax = rng.integers(0, 3, 4000)
            u[np.arange(4000), ax] = np.sign(rng.uniform(-1, 1, 4000))
            pts = u * s[:3]
        elif t in (mj.mjtGeom.mjGEOM_CYLINDER, mj.mjtGeom.mjGEOM_CAPSULE):
            a = rng.uniform(0, 2 * np.pi, 4000)
            z = rng.uniform(-s[1], s[1], 4000)
            pts = np.c_[s[0] * np.cos(a), s[0] * np.sin(a), z]
        elif t == mj.mjtGeom.mjGEOM_SPHERE:
            u = rng.normal(size=(4000, 3))
            pts = s[0] * u / np.linalg.norm(u, axis=1, keepdims=True)
        else:
            return np.zeros((0, 3, 3)), np.zeros(0)
        P = np.asarray(pts) @ R.T + p
        T = np.repeat(P[:, None, :], 3, axis=1)  # degenerate "triangles": points with a nominal area
        return T, np.full(len(P), 1e-6)
    return T, tri_area(T[:, 0], T[:, 1], T[:, 2])


def cloud(dr, seed: int) -> tuple[np.ndarray, dict]:
    mj, m, d = dr.mj, dr.m, dr.d
    geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in dr.obj_tree]
    vis = [g for g in geoms if int(m.geom_group[g]) == 1 and int(m.geom_type[g]) == mj.mjtGeom.mjGEOM_MESH]
    use = vis or [g for g in geoms if int(m.geom_type[g]) == mj.mjtGeom.mjGEOM_MESH] or geoms
    rng = np.random.default_rng(seed)
    Ts, As = [], []
    for g in use:
        T, A = sample_geom(m, d, g, rng, mj)
        Ts.append(T)
        As.append(A)
    T, A = np.concatenate(Ts), np.concatenate(As)
    k = rng.choice(len(T), GGX_PTS, p=A / A.sum())
    r1, r2 = rng.uniform(size=(2, GGX_PTS))
    s1 = np.sqrt(r1)
    P = (1 - s1)[:, None] * T[k, 0] + (s1 * (1 - r2))[:, None] * T[k, 1] + (s1 * r2)[:, None] * T[k, 2]
    return P, dict(n_geoms=len(geoms), used=len(use), visual=bool(vis), extent_cm=(100 * (P.max(0) - P.min(0))).round(1).tolist())


def main() -> None:
    import hg_look as HL

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ep-list", required=True)
    a = ap.parse_args()
    eps = [int(v) for v in a.ep_list.split(",")]
    out = FR.GT_DIR / a.task
    out.mkdir(parents=True, exist_ok=True)
    dr = LR.make_driver("scalebfm", a.task)
    for ep in eps:
        f = out / f"ep{ep:+05d}.npz"
        if f.exists():
            continue
        t0 = time.time()
        dr.reset_hg(ep)
        dr.mj.mj_forward(dr.m, dr.d)
        T0 = FR.obj_T(dr)
        P, meta = cloud(dr, ep)
        rep = HL._call(HL.GGX_ADDR, dict(points=P.astype(np.float32), num_grasps=400, seed=0))
        G = np.asarray(rep["G"], float).reshape(-1, 4, 4)
        Gobj = np.linalg.inv(T0) @ G
        tmp = f.with_name(f.stem + ".tmp.npz")
        np.savez_compressed(tmp, grasp_G_obj=Gobj, grasp_score=np.asarray(rep["conf"], float), obj_T_reset=T0, cloud_w=P.astype(np.float32),
                            meta=np.array(json.dumps(dict(meta, n_grasps=len(G), ggx_s=rep["s"], obj=dr.obj_name))))  # fmt: skip
        os.replace(tmp, f)
        print(f"gt {a.task} ep {ep}: {len(G)} grasps, cloud {meta}, {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
