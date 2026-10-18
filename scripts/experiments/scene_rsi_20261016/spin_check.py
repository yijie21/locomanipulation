#!/usr/bin/env python3
"""TODO 27.4.6 (plan docs/handoff/scene-rsi-plan-20261016.md §10.2): free objects left alone on their surfaces, with and
without torsional friction. The r1_slide_then_grasp loop found the wallet turning about 2 deg/s on its shelf with
nothing touching it: the scene's object colliders had `condim` 3 (sliding friction only), so the torsional coefficient
in their `friction` (0.005) was never used and solver noise could spin a resting object.

Loads `<scene>/env_only.xml` (no robot) at the runner's timestep (5 ms), lets everything rest for `--seconds`, and
records per free object the yaw turned and the distance slid, once with the colliders' condim as built and once with
`--condim` on every free object's colliding geoms. Writes `<out>/spin_check.json`.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    $PY scripts/experiments/scene_rsi_20261016/spin_check.py --scene outputs/scene_discovery/adt_apartment_mars \\
        --out outputs/scene_rsi_20261016/physics
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

import mujoco
import numpy as np

DT = 0.005  # the ScaleBFM runner's physics step (backends/scalebfm/runtime.py)


def yaw(q) -> float:
    w, x, y, z = q
    return float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))


def free_bodies(m) -> list[tuple[str, int]]:
    """(name, qpos address) of every body with a free joint."""
    return [
        (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[j])), int(m.jnt_qposadr[j]))
        for j in range(m.njnt)
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    ]


def rest(m, seconds: float, settle: float = 1.0) -> dict:
    """Per free body: yaw turned (deg) and xy slid (m) between `settle` s and `settle + seconds` s."""
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bodies = free_bodies(m)
    for _ in range(int(settle / DT)):
        mujoco.mj_step(m, d)
    start = {n: (d.qpos[a : a + 3].copy(), yaw(d.qpos[a + 3 : a + 7])) for n, a in bodies}
    for _ in range(int(seconds / DT)):
        mujoco.mj_step(m, d)
    out = {}
    for n, a in bodies:
        p0, y0 = start[n]
        dy = (yaw(d.qpos[a + 3 : a + 7]) - y0 + 180.0) % 360.0 - 180.0
        out[n.removesuffix("_object")] = dict(
            yaw_deg=round(dy, 2), slid_m=round(float(np.linalg.norm(d.qpos[a : a + 2] - p0[:2])), 4),
            z_m=round(float(d.qpos[a + 2] - p0[2]), 4),
        )  # fmt: skip
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, required=True, help="the scene directory (env_only.xml)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--condim", type=int, default=4)
    a = ap.parse_args(argv)
    # env_only.xml names the free objects' textured visuals' materials without defining them; visuals do not collide
    xml = re.sub(r' material="[^"]*"', "", (a.scene / "env_only.xml").read_text())
    with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=a.scene) as fh:  # its mesh paths are absolute anyway
        fh.write(xml)
        fh.flush()
        m = mujoco.MjModel.from_xml_path(fh.name)
    m.opt.timestep = DT
    free = {int(m.jnt_bodyid[j]) for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}
    colliding = [
        g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in free and (m.geom_contype[g] or m.geom_conaffinity[g])
    ]
    built = sorted({int(m.geom_condim[g]) for g in colliding})
    runs = {f"condim {built}": rest(m, a.seconds)}
    m.geom_condim[colliding] = a.condim
    runs[f"condim {a.condim}"] = rest(m, a.seconds)
    a.out.mkdir(parents=True, exist_ok=True)
    report = dict(scene=str(a.scene), seconds=a.seconds, dt=DT, n_free=len(free), n_geoms=len(colliding), runs=runs)
    (a.out / "spin_check.json").write_text(json.dumps(report, indent=1))
    for k, r in runs.items():
        rate = sorted(((abs(v["yaw_deg"]) / a.seconds, n) for n, v in r.items()), reverse=True)
        slid = max(v["slid_m"] for v in r.values())
        print(f"{k}: top yaw rates (deg/s) {[(n, round(x, 3)) for x, n in rate[:6]]}; "
              f"> 0.1 deg/s: {sum(x > 0.1 for x, _ in rate)}/{len(rate)}; max slid {slid} m")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
