"""Vision RSI D2 step 2c (docs/handoff/vision-rsi-design-20261023.md section 17.7): the map the robot builds from its own
sensors on the mapping tour -- geometry and surfaces -- scored against the scene file it replaces.

  points    head-camera depth (a D455-like stereo error model: sigma = z^2 * 0.08 px / (f * 95 mm), beyond 6 m
            dropped, the robot's own pixels removed) and LiDAR returns, placed with the estimated poses (map_poses.py, or the truth: --poses truth),
            the camera through the LiDAR-to-camera calibration of each frame (the head's pan and tilt are commanded,
            so known); one point cloud, 2 cm voxels (points.ply)
  grid      the walking map at the navigator's 5 cm: a cell is an obstacle when 3 or more points lie in it between
            0.08 and 1.7 m, floor seen when 3 or more lie below 0.04 m; free = floor seen and no obstacle; the rest
            unknown (counted as blocked: section 17.1, rule 4); the C-space inflates obstacles and unknown by the body
            radius (grid.npz, grid.png against the scene file's map)
  surfaces  horizontal patches at 0.25-1.35 m from the camera points (normals within 18 deg of up, 1 cm height peaks,
            DBSCAN in xy), each a minimum-area rectangle and a height (surfaces.json), matched to the scene file's
            surfaces of the room

Scores (map.json): the share of the true free cells seen free, true obstacle cells seen free (must be 0), the
C-space's agreement with the scene file's; per true surface found or not, the height and centre error.

    PYTHONPATH=src /workspace/envs/vision_map/bin/python scripts/rsi/map_build.py outputs/vision_rsi_20261023/tour/sensors \\
        outputs/vision_rsi_20261023/tour/map_kiss_loop --poses outputs/vision_rsi_20261023/tour/poses/kiss_loop.npy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from matplotlib.path import Path as MplPath
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import DBSCAN

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from map_common import MAX_DEPTH_M, kmat, se3, stereo_noise  # noqa: E402

from locomani.primitives.navigator import build as build_navigation  # noqa: E402

BODY_RADIUS_M = 0.35  # scene.world's
OBST_Z, FLOOR_Z, MIN_HITS = (0.08, 1.7), 0.04, 3
SURF_Z = (0.25, 1.35)
CELL_M = 0.05  # surface footprints compared on this grid


def load(sensors: Path, poses: str):
    lz = dict(np.load(sensors / "lidar.npz"))  # loaded once: an NpzFile decompresses an array on every access
    lidar_true = np.array([se3(R, p) for R, p in zip(lz["R"], lz["pos"], strict=True)])
    lidar_est = lidar_true if poses == "truth" else np.load(poses)
    cz = dict(np.load(sensors / "camera.npz"))
    return lz, lidar_true, lidar_est, cz


def camera_points(sensors: Path, cz, lt, le, lidar_t, stride: int, noise: bool, rng) -> np.ndarray:
    """World points from every head-camera frame, each placed by the estimated LiDAR pose of the nearest LiDAR frame
    times that moment's LiDAR-to-camera calibration."""
    K = kmat(cz["K"])
    h, w = 480, 640
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    rays = np.stack([(u - K[0, 2]) / K[0, 0], -(v - K[1, 2]) / K[1, 1], -np.ones_like(u, float)], -1)  # MuJoCo camera
    out = []
    for i, t in enumerate(cz["t"]):
        z = cv2.imread(str(sensors / "depth" / f"{i:05d}.png"), cv2.IMREAD_UNCHANGED)[::stride, ::stride] / 1000.0
        if noise:
            z = stereo_noise(z, K[1, 1], rng)
        ok = (z > 0.1) & (z < MAX_DEPTH_M)
        own = sensors / "self" / f"{i:05d}.png"
        if own.exists():  # the robot's own arms and hands (the self filter)
            ok &= cv2.imread(str(own), cv2.IMREAD_UNCHANGED)[::stride, ::stride] == 0
        n = int(np.argmin(np.abs(lidar_t - t)))
        cam = le[n] @ np.linalg.inv(lt[n]) @ se3(cz["R"][i], cz["pos"][i])
        p = rays[ok] * z[ok][:, None]
        out.append(p @ cam[:3, :3].T + cam[:3, 3])
    return np.concatenate(out)


def lidar_points(lz, le, every: int) -> np.ndarray:
    off, out = lz["offsets"], []
    for k in range(0, len(off) - 1, every):
        p = lz["points"][off[k] : off[k + 1]].astype(np.float64)
        out.append(p @ le[k][:3, :3].T + le[k][:3, 3])
    return np.concatenate(out)


def voxel(p: np.ndarray, size: float) -> np.ndarray:
    return np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p)).voxel_down_sample(size).points)


def grid(points: np.ndarray, occ) -> dict:
    """Obstacle / floor-seen counts per navigator cell, the free and C-space masks."""

    def counts(p):
        c = np.floor((p[:, :2] - occ.origin) / occ.res).astype(int)
        ok = (c[:, 0] >= 0) & (c[:, 0] < occ.nx) & (c[:, 1] >= 0) & (c[:, 1] < occ.ny)
        n = np.zeros((occ.nx, occ.ny), int)
        np.add.at(n, (c[ok, 0], c[ok, 1]), 1)
        return n

    z = points[:, 2]
    obstacle = counts(points[(z > OBST_Z[0]) & (z < OBST_Z[1])]) >= MIN_HITS
    floor = counts(points[z < FLOOR_Z]) >= MIN_HITS
    free = floor & ~obstacle
    clearance = distance_transform_edt(free, sampling=occ.res)  # distance to the nearest obstacle or unknown cell
    return dict(obstacle=obstacle, floor=floor, free=free, cspace=clearance >= BODY_RADIUS_M)


def room_mask(meta: dict, room: str, occ) -> np.ndarray:
    poly = MplPath(np.asarray(meta["room"]["rooms"][room]["polygon"], float))
    ij = np.stack(np.meshgrid(np.arange(occ.nx), np.arange(occ.ny), indexing="ij"), -1).reshape(-1, 2)
    xy = occ.origin + (ij + 0.5) * occ.res
    return poly.contains_points(xy).reshape(occ.nx, occ.ny)


def grid_scores(g: dict, occ, room: np.ndarray) -> dict:
    truth_free = ~occ.occupied & room
    truth_c = occ.free & room
    seen_free = g["free"] & room
    return dict(
        true_free_cells=int(truth_free.sum()),
        true_free_seen_free=round(float((seen_free & truth_free).sum() / max(truth_free.sum(), 1)), 3),
        true_obstacle_seen_free=int((seen_free & occ.occupied).sum()),
        cspace_true_free_cells=int(truth_c.sum()),
        cspace_recall=round(float((g["cspace"] & truth_c).sum() / max(truth_c.sum(), 1)), 3),
        cspace_extra_cells=int((g["cspace"] & room & ~occ.free).sum()),
    )


def grid_png(path: Path, g: dict, occ, room: np.ndarray) -> None:
    """Truth vs perceived, one pixel per cell: black true obstacle, white free in both, green free only in the map,
    red true free that the map left unknown or blocked, blue perceived obstacle the truth calls free."""
    img = np.full((occ.nx, occ.ny, 3), 90, np.uint8)
    tf, pf = ~occ.occupied & room, g["free"]
    img[occ.occupied & room] = (0, 0, 0)
    img[tf & pf] = (255, 255, 255)
    img[tf & ~pf] = (60, 60, 220)  # BGR red
    img[~tf & pf & room] = (60, 200, 60)
    img[tf & g["obstacle"]] = (220, 120, 40)
    cv2.imwrite(str(path), np.flipud(img.transpose(1, 0, 2)).repeat(3, 0).repeat(3, 1))


def surfaces(points: np.ndarray) -> list[dict]:
    """Horizontal patches between SURF_Z: points whose normal is near up, grouped by 1 cm height peaks and in xy."""
    sel = points[(points[:, 2] > SURF_Z[0]) & (points[:, 2] < SURF_Z[1])]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sel))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=20))
    up = np.abs(np.asarray(pc.normals)[:, 2]) > np.cos(np.radians(18))
    p = sel[up]
    hist, edges = np.histogram(p[:, 2], bins=np.arange(SURF_Z[0], SURF_Z[1] + 0.01, 0.01))
    peaks = [i for i in range(1, len(hist) - 1) if hist[i] >= 30 and hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1]]
    out, used = [], np.zeros(len(p), bool)
    for i in sorted(peaks, key=lambda i: -hist[i]):
        zc = 0.5 * (edges[i] + edges[i + 1])
        band = (np.abs(p[:, 2] - zc) < 0.015) & ~used
        if band.sum() < 30:
            continue
        idx = np.where(band)[0]
        lab = DBSCAN(eps=0.06, min_samples=10).fit_predict(p[idx, :2])
        for c in set(lab) - {-1}:
            pts = p[idx[lab == c]]
            (cx, cy), (w, h), ang = cv2.minAreaRect(pts[:, :2].astype(np.float32))
            if w * h < 0.02:
                continue
            used[idx[lab == c]] = True
            cells = np.unique(np.floor(pts[:, :2] / CELL_M).astype(int), axis=0)
            out.append(dict(xy=[round(float(cx), 3), round(float(cy), 3)], top=round(float(np.median(pts[:, 2])), 3),
                            half_extent=[round(w / 2, 3), round(h / 2, 3)], yaw=round(float(np.radians(ang)), 4),
                            area=round(float(w * h), 3), points=int(len(pts)), cells=cells.tolist()))  # fmt: skip
    return out


def box_cells(s: dict) -> set[tuple[int, int]]:
    """The CELL_M cells whose centres lie inside a surface's footprint."""
    from locomani.tasks.scene.tour import in_box

    r = float(np.hypot(*s["half_extent"])) + CELL_M
    lo = np.floor((np.asarray(s["xy"]) - r) / CELL_M).astype(int)
    hi = np.ceil((np.asarray(s["xy"]) + r) / CELL_M).astype(int)
    return {(i, j) for i in range(lo[0], hi[0]) for j in range(lo[1], hi[1])
            if in_box(((i + 0.5) * CELL_M, (j + 0.5) * CELL_M), s, 0.0)}  # fmt: skip


def surface_scores(found: list[dict], meta: dict, room: str, must: set[str]) -> tuple[dict, dict[str, int]]:
    """Each true surface of the room (tops 0.3-1.3 m) against the perceived patches within 3 cm of its height: found
    when they cover 30% or more of its footprint's cells; its patch is the one covering most. Returns the scores and
    the match (true key -> patch index) for mapped.perceived_meta."""
    rows, match = {}, {}
    cells = [{tuple(c) for c in f["cells"]} for f in found]
    for k, s in meta["surfaces"].items():
        if s.get("room") != room or s.get("half_extent") is None or not 0.3 <= float(s["top"]) <= 1.3:
            continue
        mine = box_cells(s)
        cover = {i: len(cells[i] & mine) / max(len(mine), 1) for i, f in enumerate(found)
                 if abs(f["top"] - float(s["top"])) < 0.03}  # fmt: skip
        total = min(1.0, sum(cover.values()))
        best = max(cover, key=cover.get) if cover else None
        ok = best is not None and total >= 0.3
        if ok:
            match[k] = best
        f = found[best] if ok else None
        rows[k] = dict(must=k in must, found=ok, cover=round(total, 2),
                       dz_cm=round(100 * (f["top"] - float(s["top"])), 2) if f else None,
                       dxy_cm=round(100 * float(np.linalg.norm(np.subtract(f["xy"], s["xy"]))), 1) if f else None,
                       area_ratio=round(f["area"] / max(4 * s["half_extent"][0] * s["half_extent"][1], 1e-6), 2) if f else None)  # fmt: skip
    found_n = sum(r["found"] for r in rows.values())
    out = dict(true=len(rows), found=found_n, must_found=f"{sum(r['found'] for r in rows.values() if r['must'])}"
               f"/{sum(r['must'] for r in rows.values())}", perceived=len(found), rows=rows)  # fmt: skip
    return out, match


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sensors", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--poses", default="truth", help="an estimated LiDAR pose file (.npy, map_poses.py) or 'truth'")
    ap.add_argument("--no-noise", action="store_true", help="the rendered depth as is")
    ap.add_argument("--stride", type=int, default=4, help="every n-th pixel")
    a = ap.parse_args(argv)
    info = json.loads((a.sensors / "sensors.json").read_text())
    tour = json.loads((Path(info["record"]) / "tour.json").read_text())
    meta = json.loads((REPO / tour["scene"] / "scene_meta.json").read_text())
    room = tour["config"]["room"]
    lz, lt, le, cz = load(a.sensors, a.poses)
    rng = np.random.default_rng(0)
    cam = voxel(camera_points(a.sensors, cz, lt, le, lz["t"], a.stride, not a.no_noise, rng), 0.02)
    lid = voxel(lidar_points(lz, le, 2), 0.02)
    points = voxel(np.concatenate([cam, lid]), 0.02)
    a.out.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(a.out / "points.ply"), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)))
    occ, _ = build_navigation(meta, body_radius=BODY_RADIUS_M)
    g = grid(points, occ)
    rm = room_mask(meta, room, occ)
    np.savez_compressed(a.out / "grid.npz", origin=occ.origin, res=occ.res, **g)
    grid_png(a.out / "grid.png", g, occ, rm)
    found = surfaces(cam)
    (a.out / "surfaces.json").write_text(json.dumps(found, indent=1))
    must = set(tour["config"]["must"])
    res = dict(poses=a.poses, noise=not a.no_noise, points=len(points), camera_points=len(cam), lidar_points=len(lid),
               grid=grid_scores(g, occ, rm))  # fmt: skip
    res["surfaces"], match = surface_scores(found, meta, room, must)
    (a.out / "surface_match.json").write_text(json.dumps(match, indent=1))
    (a.out / "map.json").write_text(json.dumps(res, indent=1))
    s = res["surfaces"]
    print(
        json.dumps(res["grid"]), f"surfaces {s['found']}/{s['true']} must {s['must_found']} perceived {s['perceived']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
