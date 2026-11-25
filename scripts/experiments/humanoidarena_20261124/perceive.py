"""The robot's look, for humanoidarena_20261124: one front-camera RGB-D frame -> points in the pelvis frame and the world,
SAM 3's masks for text prompts, the target's points, a grasp from a written geometric rule, the voxelized frame.

Nothing here reads the simulator's objects. The inputs are the camera's image and depth, where the camera sits on the
pelvis (the robot's own kinematics) and the pelvis's pose (the robot's own root state, which the brief allows); the
robot's own link positions drop its own pixels.

SAM 3 runs as a server (scripts/rsi/sam3_server.py, port 18832 on the 4090); the wire protocol is
locomani.tasks.scene.services' (pickled dicts over multiprocessing.connection, authkey below), re-declared here so this
module needs numpy only.
"""

from __future__ import annotations

import os
from multiprocessing.connection import Client as _Connect

import numpy as np

SAM3 = ("127.0.0.1", int(os.environ.get("SAM3_PORT", 18832)))
AUTHKEY = b"locomani-vision-rsi"
VOX = 0.02  # m, the obstacle voxels (curobo_avoid_20261120's)


class Sam:
    def __init__(self, address=SAM3):
        self.address, self.conn = tuple(address), None

    def call(self, request: dict) -> dict:
        for attempt in (0, 1):
            try:
                if self.conn is None:
                    self.conn = _Connect(self.address, authkey=AUTHKEY)
                self.conn.send(request)
                reply = self.conn.recv()
                break
            except (EOFError, OSError, ConnectionError):
                self.conn = None
                if attempt:
                    raise
        if "error" in reply:
            raise RuntimeError(f"SAM 3: {reply['error']}")
        return reply

    def masks(self, rgb, prompts, floor=0.3):
        """Every instance above `floor` over the prompts: [(score, mask, prompt)], best first."""
        res = self.call(dict(image=np.asarray(rgb, np.uint8), prompts=list(prompts)))["results"]
        out = []
        for p in prompts:
            for s, packed in res.get(p, []):
                if s >= floor:
                    n = int(np.prod(packed["shape"]))
                    m = np.unpackbits(packed["bits"], count=n).astype(bool).reshape(packed["shape"])
                    out.append((float(s), m, p))
        return sorted(out, key=lambda x: -x[0])


def cam_points(depth, K):
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(np.float64)
    return np.stack([(u + 0.5 - K[0, 2]) * z / K[0, 0], (v + 0.5 - K[1, 2]) * z / K[1, 1], z], axis=2)


def erode(m, n=2):
    m = m.copy()
    for _ in range(n):
        m[1:] &= m[:-1]
        m[:-1] &= m[1:]
        m[:, 1:] &= m[:, :-1]
        m[:, :-1] &= m[:, 1:]
    return m


class Look:
    """One frame: pts (H, W, 3) in the pelvis frame at the look, world (H, W, 3), ok (valid depth, not the robot)."""

    def __init__(self, L: dict, own_links=None, own_r=0.09):
        self.L = L
        d = np.asarray(L["depth"], float)
        ok = np.isfinite(d) & (d > 0.12) & (d < 4.0)
        Pc = cam_points(np.where(ok, d, 0.0), L["K"])
        self.pts = Pc @ np.asarray(L["cam_R"]).T + np.asarray(L["cam_p"])  # pelvis frame
        self.pR, self.pp = np.asarray(L["pelvis_R"]), np.asarray(L["pelvis_p"])
        self.world = self.pts @ self.pR.T + self.pp
        own = np.zeros_like(ok)
        if own_links is not None and len(own_links):  # the robot's own pixels: near its own links (pelvis frame)
            from scipy.spatial import cKDTree

            dist, _ = cKDTree(np.asarray(own_links, float)).query(self.pts.reshape(-1, 3))
            own = (dist < own_r).reshape(ok.shape)
        self.own = own & ok
        self.ok = ok & ~own
        self.rgb = L["rgb"]

    def object_points(self, mask, world=True, erode_px=2):
        m = erode(mask, erode_px) & self.ok
        return (self.world if world else self.pts)[m]

    def voxels(self, exclude=None, box=((0.05, 1.6), (-1.0, 1.0), (-0.9, 1.0))):
        """The frame's points (pelvis frame at the look) less `exclude` pixels, inside `box` (pelvis frame), as VOX
        voxel centres (world)."""
        m = self.ok.copy()
        if exclude is not None:
            m &= ~exclude
        P = self.pts[m]
        (x0, x1), (y0, y1), (z0, z1) = box
        P = P[(P[:, 0] > x0) & (P[:, 0] < x1) & (P[:, 1] > y0) & (P[:, 1] < y1) & (P[:, 2] > z0) & (P[:, 2] < z1)]
        V = (np.unique(np.floor(P / VOX).astype(np.int64), axis=0) + 0.5) * VOX
        return V @ self.pR.T + self.pp


def dilate(m, n=4):
    m = m.copy()
    for _ in range(n):
        m[1:] |= m[:-1]
        m[:-1] |= m[1:]
        m[:, 1:] |= m[:, :-1]
        m[:, :-1] |= m[:, 1:]
    return m


def min_area_rect(xy):
    """Smallest-area rectangle of 2-D points: centre, yaw of its first axis (in [-pi/4, pi/4]), half sizes."""
    from scipy.spatial import ConvexHull

    xy = np.asarray(xy, float)
    h = xy[ConvexHull(xy).vertices]
    best = None
    for i in range(len(h)):
        e = h[(i + 1) % len(h)] - h[i]
        a = np.arctan2(e[1], e[0])
        c, s = np.cos(a), np.sin(a)
        r = h @ np.array([[c, -s], [s, c]])
        lo, hi = r.min(0), r.max(0)
        area = np.prod(hi - lo)
        if best is None or area < best[0]:
            ctr = np.array([[c, -s], [s, c]]) @ ((lo + hi) / 2)
            best = (area, ctr, a, (hi - lo) / 2)
    _, ctr, a, half = best
    a = (a + np.pi / 4) % (np.pi / 2) - np.pi / 4  # a box's yaw is defined mod 90 deg
    return ctr, float(a), half


def support_level(world, ok, obj_mask, around=0.35):
    """The height of what the object stands on: the densest level of the points just below and around its foot."""
    obj = world[obj_mask & ok]
    lo, c = np.percentile(obj[:, 2], 2), np.median(obj[:, :2], axis=0)
    near = (
        ok
        & ~dilate(obj_mask, 3)
        & (np.linalg.norm(world[..., :2] - c, axis=2) < around)
        & (np.abs(world[..., 2] - lo) < 0.06)
    )
    z = world[near][:, 2]
    if len(z) < 50:
        return float(lo), int(len(z))
    h, e = np.histogram(z, bins=np.arange(lo - 0.06, lo + 0.0651, 0.005))
    top = 0.5 * (e[h.argmax()] + e[h.argmax() + 1])
    return float(np.median(z[np.abs(z - top) < 0.006])), int(len(z))


def plane_fit(P):
    """Least-squares plane: centroid, unit normal."""
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    return c, vt[2]
