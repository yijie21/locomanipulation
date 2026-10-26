"""Vision RSI D2 step 2a (docs/handoff/vision-rsi-design-20261023.md section 17.7): the mapping tour's sensors, rendered
from its trace (tour_record.py) on the full scene model (visual meshes, ceiling included). The objects stay at the
scene's defaults, or at one round's placements (`--placements <pool start json>`): the walk does not touch them, so a
round's map comes from the same walk re-rendered (section 17.4).

  lidar   MID-360 frames at 10 Hz on the visual geoms (groups 0-2, what the cameras see), 20,000 rays each with 2 cm
          of range noise: points in the sensor frame, the hit geoms, the true sensor pose per frame (lidar.npz)
  camera  the head camera (640 x 480, MuJoCo) at 5 Hz while walking, and once per head-scan pose at each viewpoint
          (tour.SCAN: the sim has no head joints, so the scan is a render-time camera pose): RGB (rgb/*.jpg), depth
          along the view axis (depth/*.png, 16-bit millimetres), the robot's own pixels (self/*.png: what a self
          filter on the robot's model removes, as the LiDAR's does), the true camera pose and the head (pan, tilt)
          per frame (camera.npz, frames.json)
  imu     the IMU site's gyro, accelerometer, orientation and position after every 50 Hz step (imu.npz, from the trace)

Poses are MuJoCo's: a camera's rotation has x right, y up, z backward (OpenCV's is R @ diag(1, -1, -1)).

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    MUJOCO_GL=egl PYTHONPATH=src $PY scripts/rsi/tour_sensors.py outputs/vision_rsi_20261023/tour/record \\
        outputs/vision_rsi_20261023/tour/sensors lidar camera imu [--procs 12] [--placements <start json>]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src")]

from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES  # noqa: E402
from locomani.tasks.scene import lidar as L  # noqa: E402
from locomani.tasks.scene import tour as T  # noqa: E402
from locomani.tasks.scene.perceive import intrinsics  # noqa: E402

FINGERS = ("left_hand_gripper_1_joint", "left_hand_gripper_2_joint", "right_hand_gripper_1_joint",
           "right_hand_gripper_2_joint")  # fmt: skip
VISUAL = (1, 1, 1, 0, 0, 0)
LIDAR_HZ, CAMERA_HZ, W, H = 10.0, 5.0, 640, 480


def open_doors(m: mujoco.MjModel) -> list[str]:
    """Move the scene's doors that have no collision twin (adt_fix_*Door* without adt_fixcol_*: the room doors; the
    cabinets' doors have twins) to geom group 5, out of the sensors' groups and the renderer's default ones. The physics
    model has no such door, so the walking robot passes through them (the tour went through the bathroom door's panel
    and its LiDAR odometry drifted 31 cm against it); the sensors see them open."""
    names = {m.geom(g).name: g for g in range(m.ngeom)}
    out = []
    for n, g in names.items():
        if n.startswith("adt_fix_") and "Door" in n and n.replace("adt_fix_", "adt_fixcol_", 1) not in names:
            m.geom_group[g] = 5
            out.append(n)
    return out


class Replay:
    """The full scene model posed from the trace, row by row (open_doors applied)."""

    def __init__(self, scene_dir: Path, traj: dict, placements: dict | None, meta: dict):
        m = self.m = mujoco.MjModel.from_xml_path(str(scene_dir / "scene_locomani.xml"))
        self.doors = open_doors(m)
        self.d = mujoco.MjData(m)
        pel = m.body("pelvis").id
        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self.root = int(m.jnt_qposadr[next(j for j in free if m.jnt_bodyid[j] == pel)])
        self.qj = np.array([m.jnt_qposadr[m.joint(n).id] for n in MUJOCO_JOINT_NAMES])
        self.qf = np.array([m.jnt_qposadr[m.joint(n).id] for n in FINGERS])
        self.z = traj
        self.base = m.qpos0.copy()
        for k, (pos, quat) in (placements or {}).items():
            a = int(m.jnt_qposadr[m.body(meta["objects"][k]["body"]).jntadr[0]])
            self.base[a : a + 7] = [*pos, *quat]

    def row(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.z["t"], t), 0, len(self.z["t"]) - 1))

    def pose(self, i: int) -> None:
        d = self.d
        d.qpos[:] = self.base
        d.qpos[self.root : self.root + 7] = self.z["robot"][i]
        d.qpos[self.qj], d.qpos[self.qf] = self.z["joints"][i], self.z["fingers"][i]
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_camlight(self.m, d)


def _quat_axis(axis, deg: float) -> np.ndarray:
    a = np.radians(deg) / 2
    return np.r_[np.cos(a), np.sin(a) * np.asarray(axis, float)]


def _qmul(a, b) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def head_quat(mount: np.ndarray, pan: float, tilt: float) -> np.ndarray:
    """The head camera's local orientation for a head pose: pan about the parent body's z (left positive), tilt down
    from the horizon (the mount's own is tour.MOUNT's 52): about the camera's x, positive turning the view up."""
    return _qmul(_qmul(_quat_axis((0, 0, 1), pan), mount), _quat_axis((1, 0, 0), T.MOUNT[1] - tilt))


def _lidar_chunk(job) -> dict:
    scene_dir, traj, placements, meta, frames, seed = job
    rp = Replay(Path(scene_dir), traj, placements, meta)
    li = L.Mid360(rp.m, L.Mid360Config(geomgroup=VISUAL, seed=seed))
    out = dict(k=[], t=[], pts=[], geoms=[], pos=[], R=[])
    for k, t in frames:
        rp.pose(rp.row(t))
        li.frame = k  # the scan pattern of frame k, whichever process casts it
        f = li.scan(rp.d)
        out["k"].append(k)
        out["t"].append(t)
        out["pts"].append(((f["points"] - f["origin"]) @ f["rotation"]).astype(np.float32))  # sensor frame
        out["geoms"].append(f["geoms"].astype(np.int32))
        out["pos"].append(f["origin"])
        out["R"].append(f["rotation"])
    return out


def lidar(rec: Path, out: Path, scene_dir: Path, traj: dict, placements, meta, procs: int) -> dict:
    t_end = float(traj["t"][-1])
    frames = [(k, k / LIDAR_HZ) for k in range(int(t_end * LIDAR_HZ) + 1)]
    chunks = [frames[i::procs] for i in range(procs)]
    jobs = [(str(scene_dir), traj, placements, meta, c, 1000 + i) for i, c in enumerate(chunks)]
    with mp.get_context("spawn").Pool(procs) as pool:
        parts = pool.map(_lidar_chunk, jobs)
    rows = sorted(((k, p, i) for p in parts for i, k in enumerate(p["k"])), key=lambda x: x[0])
    pts = [p["pts"][i] for _, p, i in rows]
    np.savez_compressed(
        out / "lidar.npz",
        t=np.array([p["t"][i] for _, p, i in rows]),
        offsets=np.r_[0, np.cumsum([len(x) for x in pts])],
        points=np.concatenate(pts),
        geoms=np.concatenate([p["geoms"][i] for _, p, i in rows]),
        pos=np.array([p["pos"][i] for _, p, i in rows]),
        R=np.array([p["R"][i] for _, p, i in rows]),
    )
    return dict(frames=len(rows), points=int(sum(len(x) for x in pts)), hz=LIDAR_HZ)


def camera_schedule(tour: dict, t_end: float) -> list[dict]:
    """Walking frames at CAMERA_HZ outside the viewpoint holds (head at its mount), then one frame per scan pose in
    the middle of its dwell."""
    cfg, holds = tour["config"], tour["holds"]
    dwell = float(cfg["scan_dwell_s"])
    inside = lambda t: any(h["t0"] <= t < h["t1"] for h in holds)  # noqa: E731
    out = [dict(t=k / CAMERA_HZ, kind="walk", viewpoint=None, pan=T.MOUNT[0], tilt=T.MOUNT[1])
           for k in range(int(t_end * CAMERA_HZ) + 1) if not inside(k / CAMERA_HZ)]  # fmt: skip
    for h in holds:
        for j, (pan, tilt) in enumerate(cfg["scan"]):
            out.append(dict(t=h["t0"] + (j + 0.5) * dwell, kind="scan", viewpoint=h["viewpoint"], pan=pan, tilt=tilt))
    return sorted(out, key=lambda f: f["t"])


def camera(rec: Path, out: Path, scene_dir: Path, traj: dict, placements, meta, tour: dict) -> dict:
    rp = Replay(scene_dir, traj, placements, meta)
    m, d = rp.m, rp.d
    cid = m.camera("head_camera").id
    mount = m.cam_quat[cid].copy()
    r = mujoco.Renderer(m, H, W, max_geom=int(m.ngeom) + 1000)
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    (out / "self").mkdir(parents=True, exist_ok=True)
    robot = m.body_rootid[m.geom_bodyid] == m.body_rootid[m.body("pelvis").id]  # geoms of the robot's own tree
    sched = camera_schedule(tour, float(traj["t"][-1]))
    pos, rot = [], []
    for n, f in enumerate(sched):
        m.cam_quat[cid] = head_quat(mount, f["pan"], f["tilt"])
        rp.pose(rp.row(f["t"]))
        r.update_scene(d, camera=cid)
        rgb = r.render()
        r.enable_depth_rendering()
        r.update_scene(d, camera=cid)
        depth = r.render()
        r.disable_depth_rendering()
        r.enable_segmentation_rendering()
        r.update_scene(d, camera=cid)
        seg = r.render()
        r.disable_segmentation_rendering()
        mine = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[..., 0] >= 0)
        mine[mine] = robot[seg[..., 0][mine]]
        cv2.imwrite(str(out / "self" / f"{n:05d}.png"), mine.astype(np.uint8) * 255)
        cv2.imwrite(
            str(out / "rgb" / f"{n:05d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        mm = np.clip(np.round(depth * 1000.0), 0, 65535).astype(np.uint16)
        cv2.imwrite(str(out / "depth" / f"{n:05d}.png"), mm)
        pos.append(d.cam_xpos[cid].copy())
        rot.append(d.cam_xmat[cid].reshape(3, 3).copy())
        f["i"] = n
    m.cam_quat[cid] = mount
    r.close()
    fx, fy, cx, cy = intrinsics(float(m.cam_fovy[cid]), W, H)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    np.savez_compressed(out / "camera.npz", t=np.array([f["t"] for f in sched]), K=K, pos=np.array(pos), R=np.array(rot),
                        pan=np.array([f["pan"] for f in sched]), tilt=np.array([f["tilt"] for f in sched]))  # fmt: skip
    (out / "frames.json").write_text(json.dumps(sched))
    return dict(
        frames=len(sched), walk=sum(f["kind"] == "walk" for f in sched), scan=sum(f["kind"] == "scan" for f in sched)
    )


def imu(out: Path, traj: dict) -> dict:
    keys = ("imu_t", "imu_imu_gyro", "imu_imu_acc", "imu_imu_quat", "imu_frame_pos")
    np.savez_compressed(out / "imu.npz", **{k.removeprefix("imu_").removeprefix("imu_"): traj[k] for k in keys})
    return dict(samples=len(traj["imu_t"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("record", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("what", nargs="+", choices=("lidar", "camera", "imu"))
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--placements", type=Path, help="a pool start (json): the round's object placements")
    a = ap.parse_args(argv)
    tour = json.loads((a.record / "tour.json").read_text())
    traj = dict(np.load(a.record / "trajectory.npz"))
    scene_dir = REPO / tour["scene"]
    meta = json.loads((scene_dir / "scene_meta.json").read_text())
    placements = json.loads(a.placements.read_text())["start"]["placements"] if a.placements else None
    a.out.mkdir(parents=True, exist_ok=True)
    info = dict(record=str(a.record), placements=str(a.placements) if a.placements else None)
    for w in a.what:
        t0 = time.time()
        if w == "lidar":
            info[w] = lidar(a.record, a.out, scene_dir, traj, placements, meta, a.procs)
        elif w == "camera":
            info[w] = camera(a.record, a.out, scene_dir, traj, placements, meta, tour)
        else:
            info[w] = imu(a.out, traj)
        info[w]["wall_s"] = round(time.time() - t0, 1)
        print(w, json.dumps(info[w]), flush=True)
    prev = json.loads((a.out / "sensors.json").read_text()) if (a.out / "sensors.json").exists() else {}
    (a.out / "sensors.json").write_text(json.dumps({**prev, **info}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
