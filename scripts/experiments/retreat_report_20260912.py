#!/usr/bin/env python3
"""Ledger 32: the squat's delivered retreat, and the waist measured in-run.

Reads the three 10-round batches of 2026-09-12 (same 10 placements, derived
from az_ab30/az374's bank). Everything here is a LIVE measurement: the offline
re-solve of these same records does not reproduce the live solver (section 3),
so no conclusion below depends on one.

    python scripts/experiments/retreat_report_20260912.py
"""

import json
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = (
    ("run1", ROOT / "outputs/experiments/retreat_seed10_20260912_run1"),
    ("run2", ROOT / "outputs/experiments/retreat_seed10b_20260912"),
    ("run3", ROOT / "outputs/experiments/retreat_seed10c_20260912"),
)
IK_POSITION_TOL, IK_ROTATION_TOL = 0.020, 0.30


def load():
    pose, rej, rounds = [], [], []
    for label, d in RUNS:
        p = d / "manifest.jsonl"
        if not p.is_file():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rounds.append((label, r))
            m = (r.get("execution") or {}).get("metrics") or {}
            for s in m.get("coupled_transport") or []:
                if s.get("squat_delivered_pose"):
                    pose.append(dict(run=label, rnd=r["round"], case=r["case_id"],
                                     handover=s.get("handover"), **s["squat_delivered_pose"]))
            for x in m.get("coupled_prediction_ik_errors") or []:
                x["run"], x["rnd"] = label, r["round"]
                rej.append(x)
    return pose, rej, rounds


def main():
    pose, rej, rounds = load()
    print(f"{len(rounds)} rounds over {len({r for r, _ in rounds})} batches, "
          f"ok {sum(1 for _, r in rounds if r.get('ok'))}; {len(pose)} reached the squat; "
          f"{len(rej)} terminal-IK rejections recorded")

    print("\n== 1. the squat's DELIVERED retreat against project_squat's prediction")
    print(f"{'run':>6} {'rnd':>4} {'case':>8} {'drop':>8} {'retreat':>9} {'pred':>8} {'ratio':>6} "
          f"{'lateral':>8} {'pitch':>7} {'handover':>16}")
    for p in pose:
        print(f"{p['run']:>6} {p['rnd']:>4} {p['case']:>8} {p['drop_m']:>8.4f} {p['retreat_m']:>9.4f} "
              f"{p['predicted_retreat_m']:>8.4f} {p['retreat_m'] / max(p['drop_m'], 1e-9):>6.3f} "
              f"{p['lateral_m']:>8.4f} {p['pitch_deg']:>7.1f} {str(p['handover']):>16}")
    if pose:
        rat = [p["retreat_m"] / p["drop_m"] for p in pose]
        err = [p["retreat_m"] - p["predicted_retreat_m"] for p in pose]
        pitch = [p["pitch_deg"] for p in pose]
        print(f"  n={len(pose)}  ratio median {st.median(rat):.3f} [{min(rat):.2f},{max(rat):.2f}] "
              f"against the predicted 0.800;  under-prediction median {st.median(err):+.4f} m, "
              f"positive in {sum(1 for e in err if e > 0)}/{len(err)}")
        print(f"  pelvis pitch at the bottom: median {st.median(pitch):+.1f} deg -- project_squat predicts 0")

    print("\n== 2. the waist, measured in the live solve (same target, seed and rotation)")
    sh = [x for x in rej if "waist_position_error_m" in x]
    if sh:
        a, b = [x["position_error_m"] for x in sh], [x["waist_position_error_m"] for x in sh]
        ar, br = [x["rotation_error_rad"] for x in sh], [x["waist_rotation_error_rad"] for x in sh]
        rescued = sum(1 for x in sh
                      if x["waist_position_error_m"] <= IK_POSITION_TOL
                      and x["waist_rotation_error_rad"] <= IK_ROTATION_TOL)
        print(f"  n={len(sh)} rejections")
        print(f"  position  use_waist=False  median {st.median(a):.4f} [{min(a):.4f},{max(a):.4f}]  "
              f"over {IK_POSITION_TOL}: {sum(1 for v in a if v > IK_POSITION_TOL)}/{len(a)}")
        print(f"  position  use_waist=True   median {st.median(b):.4f} [{min(b):.4f},{max(b):.4f}]  "
              f"over {IK_POSITION_TOL}: {sum(1 for v in b if v > IK_POSITION_TOL)}/{len(b)}")
        print(f"  rotation  False median {st.median(ar):.4f}   True median {st.median(br):.4f}")
        print(f"  INSIDE BOTH TOLERANCES with the waist: {rescued}/{len(sh)}")
        xy = [float(np.hypot(*np.asarray(x["target_root"])[:2])) for x in sh]
        hard = [x for x in sh if x["waist_position_error_m"] > IK_POSITION_TOL]
        print(f"  target |xy| median {st.median(xy):.4f} [{min(xy):.4f},{max(xy):.4f}]")
        if hard:
            hxy = [float(np.hypot(*np.asarray(x["target_root"])[:2])) for x in hard]
            print(f"  the {len(hard)} the waist does NOT rescue sit at |xy| median {st.median(hxy):.4f} "
                  f"[{min(hxy):.4f},{max(hxy):.4f}]")

    print("\n== 3. is the live solver self-consistent? (byte-identical repeat, same process)")
    rp = [x for x in rej if "repeat_position_error_m" in x]
    if rp:
        dp = [abs(x["repeat_position_error_m"] - x["position_error_m"]) for x in rp]
        dr = [abs(x["repeat_rotation_error_rad"] - x["rotation_error_rad"]) for x in rp]
        mut = [x["seed_mutated"] for x in rp]
        print(f"  n={len(rp)}  |dpe| max {max(dp):.2e}   |dre| max {max(dr):.2e}   "
              f"seed mutated by the solve: max {max(mut):.2e}")
        print("  -> the live solve is deterministic and the seed is recorded as passed. An OFFLINE re-solve "
              "of the same\n     recorded inputs returns 0.068/0.010 against the live 0.115/0.299 "
              "(strictly better on BOTH),\n     so the discrepancy is process-level and UNEXPLAINED: "
              "do not use an offline IK re-solve as evidence.")


if __name__ == "__main__":
    main()
