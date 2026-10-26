"""Vision RSI D2 step 2b (docs/handoff/vision-rsi-design-20261023.md section 17.7): the robot's pose along the mapping
tour from its own sensors, against the truth, offline on one recording (tour_sensors.py).

  kiss       LiDAR odometry: KISS-ICP (point-to-point ICP on a voxel map, constant-velocity prediction; no IMU, no loop
             closure). Stands in for FAST-LIO2, which needs ROS (section 17.4): a pip package on the same input.
  kiss_planar  KISS-ICP's x, y and heading; the height and the tilt (roll, pitch) from the robot's own body: its legs on
             the flat floor give the sensor's height (forward kinematics from the joint encoders and the feet in
             contact), the IMU's gravity gives the tilt. In the sim both are exact, so they come from the true pose.
             LiDAR odometry alone drifts in height when the floor is out of its view (section 17.3).
  kiss_loop  the same as kiss, plus a pose graph: keyframes every 0.5 m or 15 deg, loop edges by point-to-plane ICP between
             keyframes the odometry puts within 2 m of each other but 20 s or more apart, optimised by Open3D, every
             frame then moved with its keyframe.

The estimate starts at the tour's true start pose (the map frame is the start's). Reported per method: the position
error over the walk (RMSE, 95th percentile, max), the error at each viewpoint's head scan (where a map is looked at),
the end-to-start error (the tour closes its loop), the heading error, and the seconds per frame.

    /workspace/envs/vision_map/bin/python scripts/rsi/map_poses.py outputs/vision_rsi_20261023/tour/sensors \\
        outputs/vision_rsi_20261023/tour/poses [--methods kiss kiss_loop]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP

MAX_RANGE, MIN_RANGE, VOXEL = 20.0, 0.3, 0.10


def frames(sensors: Path):
    z = dict(np.load(sensors / "lidar.npz"))  # loaded once: an NpzFile decompresses an array on every access
    off, allp = z["offsets"], z["points"]
    pts = [allp[off[k] : off[k + 1]].astype(np.float64) for k in range(len(off) - 1)]
    true = np.tile(np.eye(4), (len(pts), 1, 1))
    true[:, :3, :3], true[:, :3, 3] = z["R"], z["pos"]
    return z["t"], pts, true


def kiss(pts) -> tuple[np.ndarray, float]:
    """Poses of the sensor relative to its first frame."""
    cfg = KISSConfig()
    cfg.data.max_range, cfg.data.min_range, cfg.data.deskew = MAX_RANGE, MIN_RANGE, False
    cfg.mapping.voxel_size = VOXEL
    odo = KissICP(cfg)
    out, t0 = [], time.time()
    for p in pts:
        odo.register_frame(p, np.zeros(len(p)))
        out.append(np.asarray(odo.last_pose).copy())
    return np.array(out), (time.time() - t0) / len(pts)


def _cloud(p: np.ndarray) -> o3d.geometry.PointCloud:
    c = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
    c = c.voxel_down_sample(VOXEL)
    c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))
    return c


def loop_closed(t, pts, odom: np.ndarray, *, near_m=2.0, apart_s=20.0, fitness=0.6, rmse_m=0.08):
    """Odometry plus loop edges through a pose graph. Returns (poses, keyframes, loop edges accepted)."""
    keys = [0]
    for k in range(1, len(pts)):
        d = np.linalg.inv(odom[keys[-1]]) @ odom[k]
        ang = np.degrees(np.arccos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1)))
        if np.linalg.norm(d[:3, 3]) > 0.5 or ang > 15:
            keys.append(k)
    # a keyframe's cloud: its own frame and its 4 neighbours, in its own frame (one MID-360 frame is sparse)
    clouds = []
    for k in keys:
        nb = [j for j in range(max(0, k - 2), min(len(pts), k + 3))]
        stack = [(np.linalg.inv(odom[k]) @ odom[j] @ np.c_[pts[j], np.ones(len(pts[j]))].T).T[:, :3] for j in nb]
        clouds.append(_cloud(np.concatenate(stack)))
    g = o3d.pipelines.registration.PoseGraph()
    for k in keys:
        g.nodes.append(o3d.pipelines.registration.PoseGraphNode(odom[k]))
    info_odo = np.eye(6) * 100.0
    for i in range(len(keys) - 1):
        rel = np.linalg.inv(odom[keys[i]]) @ odom[keys[i + 1]]
        g.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(i, i + 1, np.linalg.inv(rel), info_odo, uncertain=False)
        )
    loops = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if t[keys[j]] - t[keys[i]] < apart_s:
                continue
            if np.linalg.norm(odom[keys[i]][:3, 3] - odom[keys[j]][:3, 3]) > near_m:
                continue
            guess = np.linalg.inv(odom[keys[i]]) @ odom[keys[j]]
            r = o3d.pipelines.registration.registration_icp(
                clouds[j], clouds[i], 0.3, guess, o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )
            if r.fitness >= fitness and r.inlier_rmse <= rmse_m:
                info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
                    clouds[j], clouds[i], 0.3, r.transformation
                )
                g.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        i, j, np.linalg.inv(r.transformation), info, uncertain=True
                    )
                )
                loops.append((keys[i], keys[j], round(float(r.fitness), 3), round(float(r.inlier_rmse), 4)))
    o3d.pipelines.registration.global_optimization(
        g,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=0.3, edge_prune_threshold=0.25, reference_node=0
        ),  # fmt: skip
    )
    opt = [np.asarray(n.pose) for n in g.nodes]
    out = np.empty_like(odom)
    ki = 0
    for k in range(len(pts)):
        while ki + 1 < len(keys) and keys[ki + 1] <= k:
            ki += 1
        out[k] = opt[ki] @ np.linalg.inv(odom[keys[ki]]) @ odom[k]
    return out, keys, loops


def planar(rel: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Relative poses whose world height and tilt are the true ones, x, y and heading the estimate's."""
    est = true[0] @ rel
    out = est.copy()
    for k in range(len(est)):
        yaw_e = np.arctan2(est[k, 1, 0], est[k, 0, 0])
        yaw_t = np.arctan2(true[k, 1, 0], true[k, 0, 0])
        c, s = np.cos(yaw_e - yaw_t), np.sin(yaw_e - yaw_t)
        out[k, :3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ true[k, :3, :3]
        out[k, 2, 3] = true[k, 2, 3]
    return np.linalg.inv(true[0]) @ out


def score(t, est_rel: np.ndarray, true: np.ndarray, holds: list[dict]) -> dict:
    """Errors of the estimate anchored at the true start pose."""
    est = true[0] @ est_rel
    e = np.linalg.norm(est[:, :3, 3] - true[:, :3, 3], axis=1)
    eh = np.linalg.norm(est[:, :2, 3] - true[:, :2, 3], axis=1)
    yaw = lambda R: np.arctan2(R[:, 1, 0], R[:, 0, 0])  # noqa: E731
    dyaw = np.degrees(np.abs(np.angle(np.exp(1j * (yaw(est[:, :3, :3]) - yaw(true[:, :3, :3]))))))
    at = []
    for h in holds:
        idx = (t >= h["t0"]) & (t < h["t1"])
        at.append(round(100 * float(eh[idx].mean()), 2) if idx.any() else None)
    return dict(rmse_cm=round(100 * float(np.sqrt((e**2).mean())), 2), p95_cm=round(100 * float(np.percentile(e, 95)), 2),
                max_cm=round(100 * float(e.max()), 2), end_cm=round(100 * float(eh[-1]), 2),
                yaw_max_deg=round(float(dyaw.max()), 2), at_viewpoints_cm=at)  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sensors", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--methods", nargs="+", default=["kiss", "kiss_planar", "kiss_loop"])
    a = ap.parse_args(argv)
    info = json.loads((a.sensors / "sensors.json").read_text())
    tour = json.loads((Path(info["record"]) / "tour.json").read_text())
    t, pts, true = frames(a.sensors)
    a.out.mkdir(parents=True, exist_ok=True)
    res, cache = {}, {}
    for m in a.methods:
        t0 = time.time()
        if m in ("kiss", "kiss_loop", "kiss_planar") and "kiss" not in cache:
            cache["kiss"] = kiss(pts)
        if m == "kiss":
            rel, per = cache["kiss"]
            extra = dict(s_per_frame=round(per, 4))
        elif m == "kiss_planar":
            rel = planar(cache["kiss"][0], true)
            extra = {}
        elif m == "kiss_loop":
            rel, keys, loops = loop_closed(t, pts, cache["kiss"][0])
            extra = dict(keyframes=len(keys), loops=len(loops), loop_edges=loops[:50])
        else:
            raise SystemExit(f"unknown method {m}")
        res[m] = dict(score(t, rel, true, tour["holds"]), wall_s=round(time.time() - t0, 1), **extra)
        np.save(a.out / f"{m}.npy", true[0] @ rel)
        print(m, json.dumps({k: v for k, v in res[m].items() if k != "loop_edges"}), flush=True)
    (a.out / "poses.json").write_text(json.dumps(dict(sensors=str(a.sensors), frames=len(pts), methods=res), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
