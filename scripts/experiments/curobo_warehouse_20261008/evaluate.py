"""CuRobo warehouse study, stage 4: judge both planners' outputs with the same MuJoCo geometry.

For each method file (baseline_ik.json, curobo_plan.json) and each request x variant with a trajectory:
- reach: MuJoCo FK of the final joints against the goal palm poses (position error, attitude error);
- clearance: the smallest upper-body (torso, arms, hands with jaws as measured) distance to the carton walls and
  the platform over up to MAX_CHECKED evenly spaced waypoints and the final posture, with cases.Surfaces
  (sampled hull vs exact box);
- success = reach within 1 cm and 5 deg AND clearance >= -1 mm along the whole path.
The legs, pelvis and jaws stay as measured at the first reach frame; only the 17 planned joints change.

    PYTHONPATH=src:scripts/experiments/curobo_warehouse_20261008 "$VENV_SIM_PY" \
        scripts/experiments/curobo_warehouse_20261008/evaluate.py outputs/curobo_warehouse_20261008 \
        baseline_ik curobo_plan
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from baseline_ik import Checker
from scipy.spatial.transform import Rotation

from locomani.primitives.arm_ik import ArmIK

REACH_POS, REACH_ROT_DEG, CLEAR = 0.01, 5.0, -0.001
MAX_CHECKED = 30  # waypoints checked per trajectory (evenly spaced), plus the final posture


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir")
    ap.add_argument("methods", nargs="+")
    args = ap.parse_args()

    req = json.load(open(f"{args.dir}/requests.json"))
    by_label = {(r["label"], r["base"]): r for r in req["requests"]}
    index = np.asarray(req["active_index"])
    checker = Checker(req["scene"], [o["name"] for o in req["requests"][0]["obstacles"]])
    ik = ArmIK(req["scene"])
    summary = {}
    for method in args.methods:
        data = json.load(open(f"{args.dir}/{method}.json"))
        rows = []
        for res in data["results"]:
            r = by_label[(res["label"], res["base"])]
            q29 = np.asarray(r["qpos"])[checker.geo.hinge_qadr]
            row = dict(label=res["label"], base=res["base"], ok=res["ok"], why=res["why"], variants={})
            for name, entry in res["variants"].items():
                traj = entry.get("trajectory")
                final = entry.get("final")
                final = final["joints"] if isinstance(final, dict) else final
                if not traj or final is None:
                    row["variants"][name] = dict(planned=False, success=False, status=entry.get("status", ""))
                    continue
                q = q29.copy()
                q[index] = final
                errs = []
                for h, side in enumerate(("left", "right")):
                    p, R = ik.fk(q, side)
                    goal = r["goals"][name][h]
                    errs.append(
                        (
                            float(np.linalg.norm(p - np.asarray(goal["pos"]))),
                            float(
                                np.degrees(
                                    np.linalg.norm(Rotation.from_matrix(np.asarray(goal["rot"]) @ R.T).as_rotvec())
                                )
                            ),
                        )
                    )
                worst = dict(dist=1.0)
                step = max(1, int(np.ceil(len(traj) / MAX_CHECKED)))
                for wp in traj[::step] + [final]:
                    q = q29.copy()
                    q[index] = wp
                    w = checker.worst(r["qpos"], q)
                    if w["dist"] < worst["dist"]:
                        worst = w
                pos_err, rot_err = max(e[0] for e in errs), max(e[1] for e in errs)
                reached = pos_err < REACH_POS and rot_err < REACH_ROT_DEG
                clear = worst["dist"] >= CLEAR
                row["variants"][name] = dict(
                    planned=True,
                    pos_err=pos_err,
                    rot_err_deg=rot_err,
                    reached=reached,
                    clearance=worst,
                    clear=clear,
                    success=reached and clear,
                    duration=entry.get("duration"),
                    seconds=entry.get("seconds"),
                )
            rows.append(row)
        table = {}
        for base in dict.fromkeys(row["base"] for row in rows):
            for name in rows[0]["variants"]:
                vs = [row["variants"][name] for row in rows if row["base"] == base]
                table[f"{base}/{name}"] = dict(
                    n=len(vs),
                    planned=sum(v["planned"] for v in vs),
                    reached=sum(v.get("reached", False) for v in vs),
                    clear=sum(v.get("clear", False) for v in vs),
                    success=sum(v["success"] for v in vs),
                )
        summary[method] = table
        with open(f"{args.dir}/eval_{method}.json", "w") as f:
            json.dump(dict(method=data.get("method"), rows=rows, table=table), f)
        print(method)
        for key, t in table.items():
            print(
                f"  {key:22s} planned {t['planned']:3d}  reached {t['reached']:3d}  clear {t['clear']:3d}  success {t['success']:3d} / {t['n']}"
            )
    with open(f"{args.dir}/eval_summary.json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
