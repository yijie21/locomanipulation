"""The CuRobo reach route (visual.reach_route planned; user 2026-10-14: natural motion) against the same expert without it,
paired by start state on pools the route's settings were not chosen on: the acceptance pool (60) and the sealed set
(80). The route was developed on the diagnostic pool only (clearance sweep, fallback). Per round l2_measure's rules
(success = the expert's verdict; fell, dropped, ego p95); the question is non-inferiority: success, falls and drops no
worse, plus the approach motion (approach_motion.py's JSON). Writes route/eval.{json,md}.

    third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/scalebfm_warehouse_rsi_20261014/route_eval.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "scripts" / "experiments" / "controller_compare_20261013"), str(REPO / "src")]

from l2_measure import Torso, mcnemar_exact, round_facts  # noqa: E402

O = REPO / "outputs" / "scalebfm_warehouse_rsi_20261014"
X = REPO / "outputs" / "scalebfm_warehouse_rsi_20261014_hidden"
R = O / "route"
PAIRS = {
    "accept": (X / "gate" / "01_272307df4fcb" / "candidate", R / "eval_accept_route"),
    "sealed": (O / "sealed" / "final", R / "eval_sealed_route"),
}


def facts(d: Path, torso) -> dict:
    out = {}
    for npz in sorted(d.glob("bfm_*.npz")):
        f = round_facts(npz, "box", torso)
        meta = json.loads(npz.with_suffix(".json").read_text())
        f["route"] = meta.get("route") or {}
        out[npz.stem] = f
    return out


def main() -> int:
    torso = Torso("warehouse_box_45x32x33_mars")
    motion = json.loads((R / "approach_eval.json").read_text())
    report, lines = {}, []
    for pool, (straight_dir, route_dir) in PAIRS.items():
        a, b = facts(straight_dir, torso), facts(route_dir, torso)
        keys = sorted(set(a) & set(b))
        only_r = sum(b[k]["ok"] and not a[k]["ok"] for k in keys)
        only_s = sum(a[k]["ok"] and not b[k]["ok"] for k in keys)
        r = dict(
            n=len(keys),
            straight=sum(a[k]["ok"] for k in keys),
            route=sum(b[k]["ok"] for k in keys),
            only_route=only_r,
            only_straight=only_s,
            p=round(mcnemar_exact(only_r, only_s), 4),
            fell=(sum(a[k]["fell"] for k in keys), sum(b[k]["fell"] for k in keys)),
            dropped=(sum(a[k]["dropped"] for k in keys), sum(b[k]["dropped"] for k in keys)),
            ego_p95=(
                round(float(np.nanmedian([a[k]["ego_p95"] for k in keys])), 3),
                round(float(np.nanmedian([b[k]["ego_p95"] for k in keys])), 3),
            ),
            fallbacks=sum(bool(b[k]["route"].get("fallback")) for k in keys),
            start_in_collision=sum(bool(b[k]["route"].get("start_in_collision")) for k in keys),
            plan_s_median=float(np.median([b[k]["route"]["seconds"] for k in keys if b[k]["route"].get("seconds")])),
            why_straight=Counter(a[k]["why"] for k in keys if not a[k]["ok"]).most_common(),
            why_route=Counter(b[k]["why"] for k in keys if not b[k]["ok"]).most_common(),
            motion_straight=motion[str(straight_dir.relative_to(REPO))]["summary"],
            motion_route=motion[str(route_dir.relative_to(REPO))]["summary"],
        )
        report[pool] = r
        lines += [
            f"{pool}: n {r['n']}; success straight {r['straight']}, route {r['route']} (only route {only_r}, only "
            f"straight {only_s}, exact McNemar p {r['p']}); falls {r['fell'][0]} / {r['fell'][1]}; drops "
            f"{r['dropped'][0]} / {r['dropped'][1]}; ego p95 {r['ego_p95'][0]} / {r['ego_p95'][1]}; fallbacks "
            f"{r['fallbacks']}, start in collision {r['start_in_collision']}, plan {r['plan_s_median']:.1f} s median",
            f"{pool} motion straight: {json.dumps(r['motion_straight'])}",
            f"{pool} motion route:    {json.dumps(r['motion_route'])}",
            f"{pool} failures straight {r['why_straight']}; route {r['why_route']}",
        ]
    (R / "eval.json").write_text(json.dumps(report, indent=1))
    (R / "eval.md").write_text("\n".join(f"- {x}" for x in lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
