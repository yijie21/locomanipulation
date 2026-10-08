"""CuRobo warehouse study, stage 1c: does the expert's planned hand path clear the carton?

The ghost trace holds the expert's pre-feedback palm targets (pose of each wrist_yaw_link) for every tick of
the reach. Each hand's collision geoms (palm, Dex1 base, D405 mount, both fingers) are
placed at those poses and measured against the carton walls and the platform, against the box pose and the finger slides recorded
at the same tick. This is the plan itself: no IK, no tracking feedback, no physics.

    PYTHONPATH=src:scripts/experiments/curobo_warehouse_20261008 "$VENV_SIM_PY" \
        scripts/experiments/curobo_warehouse_20261008/ghost_path.py \
        outputs/curobo_warehouse_20261008/cases.json outputs/curobo_warehouse_20261008/ghost_path.json
"""

from __future__ import annotations

import argparse
import json
import os

import mujoco
import numpy as np
from cases import DIST_MAX, Surfaces, quat_wxyz_to_matrix
from hand_pose import SIDES, Hand, clearance

from locomani.tasks.crate.bimanual_carry import OFFSET

PHASES = ("reach", "align", "descend")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cases")
    ap.add_argument("out")
    args = ap.parse_args()

    data = json.load(open(args.cases))
    meta, cases = data["meta"], [c for c in data["cases"] if "start" in c]
    run = meta["run"]
    m = mujoco.MjModel.from_xml_path(meta["scene"])
    d = mujoco.MjData(m)
    box_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "nor_cal_box_free")]
    names = set(meta["obstacles"])
    obstacles = [
        (g, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g))
        for g in range(m.ngeom)
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) in names
    ]

    hands = [Hand(m, s) for s in SIDES]
    surfaces = Surfaces(m, [g for hand in hands for g, *_ in hand.parts])
    rows = []
    for c in cases:
        ghost = np.load(f"{run}/trace/{c['label']}_ghost.npz")
        trace = np.load(f"{run}/trace/{c['label']}_state_trace.npz")
        row = dict(label=c["label"], ok=c["ok"], why=c["why"], phases={})
        for phase in PHASES:
            span = c["phases"].get(phase)
            if not span:
                continue
            ticks = np.flatnonzero((ghost["seq"] >= span[0]) & (ghost["seq"] <= span[1]))
            worst = dict(dist=DIST_MAX, part=None, obstacle=None, hand=None, seq=None)
            for i in ticks:
                k = int(np.clip(np.searchsorted(trace["seq"], ghost["seq"][i]), 0, len(trace["seq"]) - 1))
                d.qpos[:] = trace["qpos"][k]
                mujoco.mj_kinematics(m, d)
                for h, hand in enumerate(hands):
                    R = quat_wxyz_to_matrix(ghost["quat"][i, h])
                    jaws = hand.jaws(trace["qpos"][k])  # measured at this tick
                    r = clearance(surfaces, d, hand, obstacles, ghost["pos"][i, h] + R @ OFFSET, R, jaws)
                    if r["dist"] < worst["dist"]:
                        worst = dict(
                            dist=r["dist"],
                            part=r["part"],
                            obstacle=r["obstacle"],
                            hand=SIDES[h],
                            seq=int(ghost["seq"][i]),
                        )
            row["phases"][phase] = dict(ticks=len(ticks), worst=worst)
        rows.append(row)
        w = min((p["worst"]["dist"] for p in row["phases"].values()), default=DIST_MAX)
        print(f"{c['label']} ok={c['ok']} {c['why']} ghost worst {w:+.4f}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(meta=dict(cases=args.cases), rows=rows), f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
