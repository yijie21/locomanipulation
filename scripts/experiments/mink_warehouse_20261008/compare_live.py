"""mink warehouse study, live pair: guard off vs guard mink on the same 20 seeds (TODO 20.5).

Per arm: outcome and failure family per round, and from the state traces the hands' measured clearance to the
carton and platform through align + descend (cases.Surfaces on the real hand hulls). For the mink arm also the
guard's own diagnostics (clearance of its hand boxes, how far it moved the targets). Paired by seed: concordant /
discordant outcomes and an exact two-sided sign (McNemar) test on the discordant pairs.

    PYTHONPATH=src:scripts/experiments/curobo_warehouse_20261008 "$VENV_SIM_PY" \
        scripts/experiments/mink_warehouse_20261008/compare_live.py \
        outputs/runs/warehouse_mars_guard_off_20261008 outputs/runs/warehouse_mars_guard_mink_20261008 \
        outputs/mink_warehouse_20261008/live/compare.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

import mujoco
import numpy as np
from cases import DIST_MAX, Surfaces

HAND = ("wrist_pitch_link", "wrist_yaw_link", "dex1_finger_1_link", "dex1_finger_2_link")
OBSTACLES = ("nor_cal_floor", "nor_cal_xp", "nor_cal_xm", "nor_cal_yp", "nor_cal_ym", "platform_col")


def load_arm(run, geometry):
    m, d, surfaces, hand, obstacles = geometry
    rows = {}
    for f in sorted(glob.glob(f"{run}/manifest_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            rows[r["label"]] = dict(ok=str(r["ok"]) == "True", why=r.get("why") or "", seed=int(r["seed"]))
    for label, row in rows.items():
        obs_path = f"{run}/observations/{label}/observations.jsonl"
        trace_path = f"{run}/trace/{label}_state_trace.npz"
        if not (os.path.exists(obs_path) and os.path.exists(trace_path)):
            continue
        obs = [json.loads(line) for line in open(obs_path)]
        seqs = {p: [o["state_seq"] for o in obs if o.get("phase") == p] for p in ("align", "descend")}
        row["phases"] = {p: len(s) for p, s in seqs.items()}
        guard = [o["low_reach"] for o in obs if (o.get("low_reach") or {}).get("guard_clearance") is not None]
        if guard:
            row["guard_clearance_min"] = float(min(g["guard_clearance"] for g in guard))
            row["guard_shift_median"] = float(np.median([max(g["guard_shift"]) for g in guard]))
            row["guard_shift_max"] = float(max(max(g["guard_shift"]) for g in guard))
        trace = np.load(trace_path)
        worst = DIST_MAX
        for phase in ("align", "descend"):
            for seq in seqs[phase]:
                i = int(np.clip(np.searchsorted(trace["seq"], seq), 0, len(trace["seq"]) - 1))
                d.qpos[:] = trace["qpos"][i]
                mujoco.mj_kinematics(m, d)
                for g in hand:
                    for o in obstacles:
                        worst = min(worst, surfaces.to_box(d, g, o))
        row["measured_clearance_min"] = float(worst)
    return rows


def sign_test(b, c):
    """Exact two-sided binomial p for b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("off")
    ap.add_argument("mink")
    ap.add_argument("out")
    args = ap.parse_args()

    scene = glob.glob(f"{args.off}/warehouse_box_*/scene_locomani.xml")[0]
    m = mujoco.MjModel.from_xml_path(scene)
    d = mujoco.MjData(m)
    name = lambda kind, i: mujoco.mj_id2name(m, kind, i) or ""  # noqa: E731
    hand = [
        g
        for g in range(m.ngeom)
        if (m.geom_contype[g] or m.geom_conaffinity[g])
        and name(mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]).split("_", 1)[-1] in HAND
    ]
    obstacles = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in OBSTACLES]
    geometry = (m, d, Surfaces(m, hand), hand, obstacles)
    arms = {k: load_arm(run, geometry) for k, run in (("off", args.off), ("mink", args.mink))}

    labels = sorted(set(arms["off"]) & set(arms["mink"]))
    both = sum(arms["off"][x]["ok"] and arms["mink"][x]["ok"] for x in labels)
    only_off = sum(arms["off"][x]["ok"] and not arms["mink"][x]["ok"] for x in labels)
    only_mink = sum(arms["mink"][x]["ok"] and not arms["off"][x]["ok"] for x in labels)
    summary = dict(
        paired=len(labels), both=both, only_off=only_off, only_mink=only_mink, p=sign_test(only_off, only_mink)
    )
    for k, rows in arms.items():
        why = {}
        for r in rows.values():
            why[r["why"]] = why.get(r["why"], 0) + 1
        clear = [r["measured_clearance_min"] for r in rows.values() if "measured_clearance_min" in r]
        summary[k] = dict(
            rounds=len(rows),
            ok=sum(r["ok"] for r in rows.values()),
            why=why,
            hands_into_walls=sum(c < -0.001 for c in clear),
            with_reach=len(clear),
        )
    with open(args.out, "w") as f:
        json.dump(dict(summary=summary, arms=arms), f, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"{'round':24s} {'off':28s} {'mink':28s}")
    for x in labels:
        cell = lambda r: (
            f"{'OK' if r['ok'] else r['why']:<17s} {1000 * r.get('measured_clearance_min', np.nan):+6.1f}mm"
        )  # noqa: E731
        print(f"{x:24s} {cell(arms['off'][x]):28s} {cell(arms['mink'][x]):28s}")


if __name__ == "__main__":
    main()
