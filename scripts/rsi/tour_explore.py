"""Vision RSI D2 step 2e (docs/handoff/vision-rsi-design-20261023.md section 17.5): the mapping walk without a planned
route -- frontier exploration on the walking grid the robot builds as it goes, in place of tour.py's route on the scene
file's map.

From the scene's robot start, over and over: the head camera's depth (MuJoCo on the full scene model, doors open, the
robot's own pixels removed, the stereo error model) goes into a 5 cm grid as in map_build.py (obstacle: 3 or more
points at 0.08-1.7 m; floor seen: 3 or more below 0.04 m; unknown = blocked); the frontiers are seen-free cells next to
never-seen ones, grouped 8-connected; the next goal is the C-space cell 0.6-1.2 m from the frontier group with the most
cells per metre of walk (frontier cells / (1 + path)); the robot walks there (scene.navigate on the grid so far, the
round's start disc free), stops, and pans its head (tour.SCAN's first two rows); a frontier still there when the robot
already stands at its goal is given up (it cannot be seen from the floor). It stops when no frontier group of
`--min-frontier` cells or more can be reached, or at `--limit-s`. The depth is placed with the true camera pose (step 2b
measured LiDAR odometry at about 2 cm; wiring it in is the next step).

Writes the same record as tour_record.py (trajectory.npz, tour.json with the stops as viewpoints and their scans as
holds), so tour_sensors.py and the map scripts run on it unchanged, plus explore.json: per leg the frontier, the goal,
the walk, and the share of the room's true free cells seen free so far.

    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \\
        scripts/rsi/tour_explore.py grasp_atomic outputs/vision_rsi_20261023/tour_explore/record
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import binary_dilation, label

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src")]

import family as F  # noqa: E402
from map_common import MAX_DEPTH_M, stereo_noise  # noqa: E402
from scene_round import make_world  # noqa: E402
from tour_sensors import FINGERS, H, W, head_quat, open_doors  # noqa: E402

from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES  # noqa: E402
from locomani.primitives.navigator import build as build_navigation  # noqa: E402
from locomani.tasks.scene import tour as T  # noqa: E402
from locomani.tasks.scene.mapped import raster, with_free_disc  # noqa: E402
from locomani.tasks.scene.navigate import walk_path  # noqa: E402
from locomani.tasks.scene.perceive import intrinsics  # noqa: E402

OBST_Z, FLOOR_Z, MIN_HITS, STRIDE = (0.08, 1.7), 0.04, 3, 4
PAN_SCAN = tuple(p for p in T.SCAN[:6])  # pans at the mount's tilt and at 30 deg
BODY_RADIUS_M = 0.35
GIVE_UP_M = 0.3  # a frontier whose goal is this close was scanned from there already
IMU = ("imu_gyro", "imu_acc", "imu_quat", "frame_pos")


class Eyes:
    """The head camera on the full scene model, posed from the sim's robot state; depth into the online grid."""

    def __init__(self, scene_dir: Path, occ):
        m = self.m = mujoco.MjModel.from_xml_path(str(scene_dir / "scene_locomani.xml"))
        open_doors(m)
        self.d = mujoco.MjData(m)
        pel = m.body("pelvis").id
        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self.root = int(m.jnt_qposadr[next(j for j in free if m.jnt_bodyid[j] == pel)])
        self.qj = np.array([m.jnt_qposadr[m.joint(n).id] for n in MUJOCO_JOINT_NAMES])
        self.qf = np.array([m.jnt_qposadr[m.joint(n).id] for n in FINGERS])
        self.cid = m.camera("head_camera").id
        self.mount = m.cam_quat[self.cid].copy()
        self.robot = m.body_rootid[m.geom_bodyid] == m.body_rootid[pel]
        self.r = mujoco.Renderer(m, H, W, max_geom=int(m.ngeom) + 1000)
        fx, fy, cx, cy = intrinsics(float(m.cam_fovy[self.cid]), W, H)
        self.fy = fy
        v, u = np.mgrid[0:H:STRIDE, 0:W:STRIDE]
        self.rays = np.stack([(u - cx) / fx, -(v - cy) / fy, -np.ones_like(u, float)], -1)
        self.origin, self.res, self.shape = occ.origin.copy(), occ.res, (occ.nx, occ.ny)
        self.obst = np.zeros(self.shape, np.int32)
        self.floor = np.zeros(self.shape, np.int32)
        self.rng = np.random.default_rng(0)
        self.frames = 0

    def look(self, row: dict, pan: float, tilt: float) -> None:
        m, d = self.m, self.d
        d.qpos[:] = m.qpos0
        d.qpos[self.root : self.root + 7] = row["robot"]
        d.qpos[self.qj], d.qpos[self.qf] = row["joints"], row["fingers"]
        m.cam_quat[self.cid] = head_quat(self.mount, pan, tilt)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_camlight(m, d)
        r = self.r
        r.enable_depth_rendering()
        r.update_scene(d, camera=self.cid)
        z = r.render()[::STRIDE, ::STRIDE].copy()
        r.disable_depth_rendering()
        r.enable_segmentation_rendering()
        r.update_scene(d, camera=self.cid)
        seg = r.render()[::STRIDE, ::STRIDE]
        r.disable_segmentation_rendering()
        mine = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[..., 0] >= 0)
        mine[mine] = self.robot[seg[..., 0][mine]]
        z = stereo_noise(z, self.fy, self.rng)
        ok = (z > 0.1) & (z < MAX_DEPTH_M) & ~mine
        R, p = d.cam_xmat[self.cid].reshape(3, 3), d.cam_xpos[self.cid]
        w = (self.rays[ok] * z[ok][:, None]) @ R.T + p
        c = np.floor((w[:, :2] - self.origin) / self.res).astype(int)
        inb = (c[:, 0] >= 0) & (c[:, 0] < self.shape[0]) & (c[:, 1] >= 0) & (c[:, 1] < self.shape[1])
        c, zz = c[inb], w[inb, 2]
        o = (zz > OBST_Z[0]) & (zz < OBST_Z[1])
        np.add.at(self.obst, (c[o, 0], c[o, 1]), 1)
        f = zz < FLOOR_Z
        np.add.at(self.floor, (c[f, 0], c[f, 1]), 1)
        self.frames += 1

    def stood_on(self, xy, radius: float = 0.3) -> None:
        """The floor under the robot is floor (the camera cannot see it, the feet stand on it)."""
        ij = np.stack(np.meshgrid(np.arange(self.shape[0]), np.arange(self.shape[1]), indexing="ij"), -1)
        near = np.linalg.norm(self.origin + (ij + 0.5) * self.res - np.asarray(xy, float), axis=-1) <= radius
        self.floor[near & (self.obst < MIN_HITS)] += MIN_HITS

    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        obstacle = self.obst >= MIN_HITS
        return obstacle, (self.floor >= MIN_HITS) & ~obstacle


def frontiers(free: np.ndarray, obstacle: np.ndarray, min_cells: int, given_up=None) -> list[np.ndarray]:
    """Groups (cell index arrays) of seen-free cells that touch a never-seen cell, less the ones given up."""
    unknown = ~free & ~obstacle
    edge = free & binary_dilation(unknown, structure=np.ones((3, 3), bool))
    if given_up is not None:
        edge &= ~given_up
    lab, n = label(edge, structure=np.ones((3, 3), bool))
    return [np.argwhere(lab == k) for k in range(1, n + 1) if (lab == k).sum() >= min_cells]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("family")
    ap.add_argument("out", type=Path)
    ap.add_argument("--min-frontier", type=int, default=12, help="cells (5 cm) in a frontier group worth a walk")
    ap.add_argument("--limit-s", type=float, default=420.0)
    ap.add_argument("--max-legs", type=int, default=40)
    ap.add_argument("--controller", default="scalebfm")
    a = ap.parse_args(argv)
    fam = F.load(a.family)
    meta = fam["meta"]
    world = make_world(fam, a.controller)
    m, sim = world.motor, world.sim
    start = np.asarray(meta["robot_start"]["xy"], float)
    yaw0 = float(np.radians(meta["robot_start"].get("yaw_deg", 0.0)))
    eyes = Eyes(fam["scene_dir"], world.occ)
    truth_free = ~world.occ.occupied & room_mask(meta, "living_dining_kitchen", world.occ)
    m.track = {}
    m.reset(start, yaw0)
    adr = {k: (int(sim.m.sensor(k).adr[0]), int(sim.m.sensor(k).dim[0])) for k in IMU}
    imu = {k: [] for k in ("t", *IMU)}
    tick, last_look = m.tick, [-1.0]

    def recorded() -> bool:
        ok = tick()
        imu["t"].append(m.t)
        for k, (i, n) in adr.items():
            imu[k].append(sim.d.sensordata[i : i + n].copy())
        if m.t - last_look[0] >= 0.2 - 1e-9:  # the camera at 5 Hz while walking, as tour_sensors renders it
            eyes.look(m.trace[-1], *T.MOUNT)
            eyes.stood_on(m.pelvis()[0][:2])
            last_look[0] = m.t
        return ok

    m.tick = recorded
    wall, legs, holds, stops = time.time(), [], [], []
    dwell = 0.4

    def scan_here(k: int) -> None:
        t0 = m.t
        for pan, tilt in PAN_SCAN:
            m.hold(dwell)
            eyes.look(m.trace[-1], pan, tilt)
        holds.append(dict(viewpoint=k, t0=round(t0, 3), t1=round(m.t, 3)))

    m.hold(0.4)
    scan_here(0)
    stops.append(dict(xy=start.round(3).tolist(), yaw=round(yaw0, 4), surfaces=[]))
    given_up = np.zeros((world.occ.nx, world.occ.ny), bool)  # frontier cells left unseen after a scan at their goal
    leg = 0
    while leg < a.max_legs:
        if m.t >= a.limit_s or m.fell_at is not None:
            break
        obstacle, free = eyes.grid()
        blocked = obstacle | ~free
        pm = with_free_disc(dict(meta, room=dict(meta["room"], walls=[], obstacles=[],
                                                 blocked=raster(blocked, eyes.origin, eyes.res)), surfaces={}),
                            m.pelvis()[0][:2], 0.5)  # fmt: skip
        occ, nav = build_navigation(pm, body_radius=BODY_RADIUS_M)
        here = m.pelvis()[0][:2]
        if not nav.set_goal_point(here):
            break
        best = None
        for grp in frontiers(free, obstacle, a.min_frontier, given_up):
            centre = eyes.origin + (grp.mean(axis=0) + 0.5) * eyes.res
            ring = occ.ring_mask(centre, 0.6, 1.2) & occ.free & (nav.dist < 1e4)
            if not ring.any():
                continue
            cells = np.argwhere(ring)
            dist = nav.dist[cells[:, 0], cells[:, 1]]
            j = int(np.argmin(dist))
            gain = len(grp) / (1.0 + float(dist[j]))
            if best is None or gain > best[0]:
                best = (gain, occ.xy_of(cells[j]), centre, len(grp), float(dist[j]), grp)
        seen = float((free & truth_free).sum() / max(truth_free.sum(), 1))
        if best is None:
            legs.append(dict(leg=leg, t=round(m.t, 2), done="no reachable frontier", seen_free=round(seen, 3)))
            print(json.dumps(legs[-1]), flush=True)
            break
        gain, goal, centre, size, dist, grp = best
        if dist < GIVE_UP_M:  # already standing at its goal and scanned from there: what is left cannot be seen
            given_up[grp[:, 0], grp[:, 1]] = True
            legs.append(dict(leg=leg, t=round(m.t, 2), gave_up=size, seen_free=round(seen, 3)))
            print(json.dumps(legs[-1]), flush=True)
            continue  # not a leg: nothing walked
        yaw = float(np.arctan2(*(centre - goal)[::-1]))
        out = walk_path(m, occ, nav, goal, yaw)
        legs.append(dict(leg=leg, t=round(m.t, 2), goal=np.round(goal, 2).tolist(), frontier_cells=size,
                         path_m=round(dist, 2), ok=bool(out.ok), why=out.why, seen_free=round(seen, 3)))  # fmt: skip
        print(json.dumps(legs[-1]), flush=True)
        if m.fell_at is not None:
            break
        stops.append(dict(xy=np.round(m.pelvis()[0][:2], 3).tolist(), yaw=round(yaw, 4), surfaces=[]))
        scan_here(len(stops) - 1)
        leg += 1
    if m.fell_at is None:  # back to the start: the loop for the pose graph
        obstacle, free = eyes.grid()
        pm = with_free_disc(dict(meta, room=dict(meta["room"], walls=[], obstacles=[],
                                                 blocked=raster(obstacle | ~free, eyes.origin, eyes.res)), surfaces={}),
                            start, 0.5)  # fmt: skip
        occ, nav = build_navigation(with_free_disc(pm, m.pelvis()[0][:2], 0.5), body_radius=BODY_RADIUS_M)
        walk_path(m, occ, nav, start, yaw0)
    m.tick = tick
    obstacle, free = eyes.grid()
    seen = float((free & truth_free).sum() / max(truth_free.sum(), 1))
    arrays = m.arrays()
    arrays.update({f"imu_{k}" if k != "t" else "imu_t": np.asarray(v, np.float64) for k, v in imu.items()})
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / "trajectory.npz", **arrays)
    cfg = T.TourConfig(scan=PAN_SCAN, scan_dwell_s=dwell)
    rec = dict(family=a.family, scene=str(fam["scene"]), controller=a.controller, config=asdict(cfg), mode="explore",
               start=dict(xy=start.tolist(), yaw=yaw0), viewpoints=stops, holds=holds, walks=legs,
               fell_at=m.fell_at, t_end=round(m.t, 2), wall_s=round(time.time() - wall, 1),
               seen_free=round(seen, 3), frames=eyes.frames)  # fmt: skip
    (a.out / "tour.json").write_text(json.dumps(rec, indent=1))
    (a.out / "explore.json").write_text(json.dumps(dict(legs=legs, seen_free=rec["seen_free"]), indent=1))
    print(f"t_end {m.t:.1f} s, wall {rec['wall_s']} s, legs {len(legs)}, seen free {seen:.3f}, fell {m.fell_at}")
    return 0 if m.fell_at is None else 1


def room_mask(meta: dict, room: str, occ) -> np.ndarray:
    from matplotlib.path import Path as MplPath

    poly = MplPath(np.asarray(meta["room"]["rooms"][room]["polygon"], float))
    ij = np.stack(np.meshgrid(np.arange(occ.nx), np.arange(occ.ny), indexing="ij"), -1).reshape(-1, 2)
    return poly.contains_points(occ.origin + (ij + 0.5) * occ.res).reshape(occ.nx, occ.ny)


if __name__ == "__main__":
    raise SystemExit(main())
