"""Vision RSI D2 step 2a (docs/handoff/vision-rsi-design-20261023.md section 17.7): walk the mapping tour
(locomani.tasks.scene.tour) once in MuJoCo with the scene's default object poses, from the scene's robot start and back
to it, and keep the trace for tour_sensors.py to render the sensors from (LiDAR, IMU, head camera) -- and, later, for
re-rendering per round with that round's object placements (the walk does not touch the objects, so its physics does
not depend on where they are).

Writes <out>/trajectory.npz (the motor trace at 50 Hz: t, robot, joints, fingers, ...; plus the IMU site's gyro,
accelerometer, orientation and position after every step) and <out>/tour.json (the plan, each viewpoint's head-scan
interval, every walk's outcome).

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \\
        scripts/rsi/tour_record.py grasp_atomic outputs/vision_rsi_20261023/tour/record
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src")]

import family as F  # noqa: E402
from scene_round import make_world  # noqa: E402

from locomani.tasks.scene import tour as T  # noqa: E402
from locomani.tasks.scene.navigate import walk_path  # noqa: E402

IMU = ("imu_gyro", "imu_acc", "imu_quat", "frame_pos")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("family")
    ap.add_argument("out", type=Path)
    ap.add_argument("--controller", default="scalebfm")
    a = ap.parse_args(argv)
    fam = F.load(a.family)
    meta = fam["meta"]
    targets = [fam["rows"][t]["goal"][0].split("(")[1].rstrip(")") for t in fam["tasks"]]
    cfg = T.TourConfig(must=tuple(sorted({meta["objects"][k]["surface"] for k in targets})))
    world = make_world(fam, a.controller)
    m, sim = world.motor, world.sim
    start = np.asarray(meta["robot_start"]["xy"], float)
    yaw0 = float(np.radians(meta["robot_start"].get("yaw_deg", 0.0)))
    vps, summary = T.plan(meta, world.occ, world.nav, start, cfg)
    print(json.dumps(summary), flush=True)

    m.track = {}
    m.reset(start, yaw0)
    adr = {k: (int(sim.m.sensor(k).adr[0]), int(sim.m.sensor(k).dim[0])) for k in IMU}
    imu = {k: [] for k in ("t", *IMU)}
    tick = m.tick

    def recorded() -> bool:
        ok = tick()
        imu["t"].append(m.t)
        for k, (i, n) in adr.items():
            imu[k].append(sim.d.sensordata[i : i + n].copy())
        return ok

    m.tick = recorded
    wall, holds, walks = time.time(), [], []
    dwell = len(cfg.scan) * cfg.scan_dwell_s
    for k, vp in enumerate([*vps, None]):
        goal, yaw = (start, yaw0) if vp is None else (vp.xy, vp.yaw)
        t0 = m.t
        out = walk_path(m, world.occ, world.nav, goal, yaw)
        pel = m.pelvis()[0][:2]
        walks.append(dict(to=k if vp is not None else "start", ok=bool(out.ok), why=out.why, t0=round(t0, 2),
                          t1=round(m.t, 2), err_m=round(float(np.linalg.norm(pel - goal)), 3)))  # fmt: skip
        print(json.dumps(walks[-1]), flush=True)
        if m.fell_at is not None:
            break
        if vp is not None:
            t0 = m.t
            m.hold(dwell)
            holds.append(dict(viewpoint=k, t0=round(t0, 3), t1=round(m.t, 3)))
    m.tick = tick
    arrays = m.arrays()
    arrays.update({f"imu_{k}" if k != "t" else "imu_t": np.asarray(v, np.float64) for k, v in imu.items()})
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / "trajectory.npz", **arrays)
    rec = dict(family=a.family, scene=str(fam["scene"]), controller=a.controller, config=asdict(cfg),
               start=dict(xy=start.tolist(), yaw=yaw0), summary=summary,
               viewpoints=[dict(xy=v.xy.round(3).tolist(), yaw=round(v.yaw, 4), surfaces=v.surfaces) for v in vps],
               holds=holds, walks=walks, fell_at=m.fell_at, t_end=round(m.t, 2), wall_s=round(time.time() - wall, 1))  # fmt: skip
    (a.out / "tour.json").write_text(json.dumps(rec, indent=1))
    print(f"t_end {m.t:.1f} s, wall {rec['wall_s']} s, fell {m.fell_at}", flush=True)
    return 0 if m.fell_at is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
