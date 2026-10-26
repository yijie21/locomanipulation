"""What the simulated head LiDAR (locomani.tasks.scene.lidar, Livox MID-360) sees of the apartment from the sealed
pool's starts, for the vision RSI mapping design (docs/handoff/vision-rsi-design-20261023.md section 17): with the lab
URDF's mount (upright, pitched 2.9 deg, -7 to 52 deg in elevation) the near floor and low furniture fall below the
field of view. Per start and mount: the share of returns on the floor, the nearest floor return, and the nearest return
at table height (0.4-1.1 m) and at coffee-table height (0.3-0.6 m), all horizontal distances from the sensor. The robot
stands at the start in the model's default joint pose; "inverted" turns the sensor upside down about its forward axis
(the field of view becomes about -52 to 7 deg), for the case the real unit is mounted that way. CPU only, about 0.5 s
per 20,000-ray frame.

    PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \\
        scripts/experiments/lidar_coverage_check_20261026.py --out outputs/vision_rsi_20261023/lidar_cover
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from locomani.tasks.scene import lidar as L

REPO = Path(__file__).resolve().parents[2]
SCENE = REPO / "outputs/scene_discovery/adt_apartment_mars_physics/scene_locomani.xml"
POOL = REPO / "outputs/vision_rsi_20261023/pool_sealed"
FLIP = np.diag([1.0, -1.0, -1.0])  # 180 deg about the sensor's x axis


def nearest(h: np.ndarray, mask: np.ndarray) -> float | None:
    return round(float(h[mask].min()), 2) if mask.any() else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=5, help="20,000-ray frames per start and mount")
    ap.add_argument("--root-z", type=float, default=0.79, help="pelvis height the robot stands at")
    a = ap.parse_args(argv)
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    pelvis = m.body("pelvis").id
    root = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.jnt_bodyid[j] == pelvis][0]
    adr = int(m.jnt_qposadr[root])
    rows = []
    for f in sorted(POOL.glob("pick_*.json")):
        s = json.loads(f.read_text())["start"]
        mujoco.mj_resetData(m, d)
        yaw = s["robot_yaw"]
        d.qpos[adr : adr + 7] = [*s["robot_xy"], a.root_z, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        mujoco.mj_forward(m, d)
        for mount in ("urdf", "inverted"):
            li = L.Mid360(m, L.Mid360Config(rays_per_frame=20000, noise_m=0.0))
            if mount == "inverted":
                li.mount_R = li.mount_R @ FLIP
            frames = [li.scan(d) for _ in range(a.frames)]
            pts = np.concatenate([fr["points"] for fr in frames])
            origin = frames[0]["origin"]
            h = np.linalg.norm(pts[:, :2] - origin[:2], axis=1)
            z = pts[:, 2]
            rows.append(dict(start=f.stem, mount=mount, sensor_z=round(float(origin[2]), 3),
                             returned=round(len(pts) / (a.frames * 20000), 3),
                             floor_share=round(float((z < 0.03).mean()), 3), floor_nearest_m=nearest(h, z < 0.03),
                             table_nearest_m=nearest(h, (z > 0.4) & (z < 1.1)),
                             low_nearest_m=nearest(h, (z > 0.3) & (z < 0.6))))  # fmt: skip
            print(json.dumps(rows[-1]), flush=True)
    summary = {}
    for mount in ("urdf", "inverted"):
        rs = [r for r in rows if r["mount"] == mount]
        summary[mount] = {
            k: dict(min=min(v), median=float(np.median(v)), max=max(v))
            for k in ("returned", "floor_share", "floor_nearest_m", "table_nearest_m", "low_nearest_m")
            if (v := [r[k] for r in rs if r[k] is not None])
        }
        summary[mount]["starts"] = len(rs)
        summary[mount]["sensor_z"] = rs[0]["sensor_z"]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "coverage.json").write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
