#!/usr/bin/env python3
"""Ledger 32/33: the squat's delivered retreat and the waist, per checkpoint.

Four 10-round batches of 2026-09-12 on the SAME ten placements (derived from
`az_ab30_20260911/az374`'s bank): three on policy/sonic_v1_1 and one on
policy/release. Everything here is a LIVE measurement -- an offline re-solve of
these same records does not reproduce the live solver (section 3), so nothing
below depends on one.

The two checkpoints are summarised SEPARATELY and never pooled: pooling arms is
how ledger 31.2 produced a lateral AUC that reversed inside the paired rounds.

    python scripts/experiments/retreat_report_20260912.py
"""

import json
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "outputs/experiments"
GROUPS = (
    ("v1.1", ("retreat_seed10_20260912_run1", "retreat_seed10b_20260912", "retreat_seed10c_20260912")),
    ("release", ("retreat_seed10_release_20260912",)),
)
IK_POSITION_TOL, IK_ROTATION_TOL = 0.020, 0.30
MJCF_WAIST_PITCH_DEG = 29.79  # ArmIK deliberately allows 1.6x this


def load(dirs):
    pose, rej, rounds = [], [], []
    for d in dirs:
        p = E / d / "manifest.jsonl"
        if not p.is_file():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rounds.append(r)
            m = (r.get("execution") or {}).get("metrics") or {}
            for s in m.get("coupled_transport") or []:
                if s.get("squat_delivered_pose"):
                    pose.append(dict(rnd=r["round"], case=r["case_id"], **s["squat_delivered_pose"]))
            rej.extend(m.get("coupled_prediction_ik_errors") or [])
    return pose, rej, rounds


def retreat(label, pose):
    if not pose:
        print(f"  {label}: no squat reached")
        return
    rat = [p["retreat_m"] / max(p["drop_m"], 1e-9) for p in pose]
    err = [p["retreat_m"] - p["predicted_retreat_m"] for p in pose]
    print(
        f"  retreat median {st.median([p['retreat_m'] for p in pose]):.4f} m against the predicted "
        f"{pose[0]['predicted_retreat_m']:.4f};  ratio {st.median(rat):.3f} [{min(rat):.2f},{max(rat):.2f}] "
        f"against the constant 0.800"
    )
    print(
        f"  under-read median {st.median(err):+.4f} m, positive in {sum(1 for e in err if e > 0)}/{len(err)};  "
        f"pitch at the bottom {st.median([p['pitch_deg'] for p in pose]):+.1f} deg (the prediction says 0);  "
        f"lateral {st.median([p['lateral_m'] for p in pose]):+.4f} m"
    )


def waist(label, rej):
    sh = [x for x in rej if "waist_position_error_m" in x]
    if not sh:
        print("  no shadow solve in these batches (the field came later)")
        return
    a = [x["position_error_m"] for x in sh]
    b = [x["waist_position_error_m"] for x in sh]
    ar = [x["rotation_error_rad"] for x in sh]
    br = [x["waist_rotation_error_rad"] for x in sh]
    rescued = [
        x
        for x in sh
        if x["waist_position_error_m"] <= IK_POSITION_TOL and x["waist_rotation_error_rad"] <= IK_ROTATION_TOL
    ]
    xy = [float(np.hypot(*np.asarray(x["target_root"])[:2])) for x in sh]
    print(
        f"  n={len(sh)}  position {st.median(a):.4f} -> {st.median(b):.4f} m  "
        f"over {IK_POSITION_TOL}: {sum(1 for v in a if v > IK_POSITION_TOL)}/{len(a)} -> "
        f"{sum(1 for v in b if v > IK_POSITION_TOL)}/{len(b)}   rotation {st.median(ar):.4f} -> {st.median(br):.4f}"
    )
    print(f"  inside BOTH tolerances with the waist: {len(rescued)}/{len(sh)};  target |xy| median {st.median(xy):.4f}")
    hard = [x for x in sh if x["waist_position_error_m"] > IK_POSITION_TOL]
    if hard:
        h = [float(np.hypot(*np.asarray(x["target_root"])[:2])) for x in hard]
        print(f"  the {len(hard)} it does NOT rescue sit at |xy| median {st.median(h):.4f} [{min(h):.4f},{max(h):.4f}]")
    wq = [x["waist_q_rad"] for x in sh if "waist_q_rad" in x]
    if wq:
        pitch = [abs(float(np.degrees(w[2]))) for w in wq]
        over = sum(1 for v in pitch if v > MJCF_WAIST_PITCH_DEG)
        print(
            f"  solved waist pitch |deg| median {st.median(pitch):.1f} max {max(pitch):.1f} against the MJCF limit "
            f"{MJCF_WAIST_PITCH_DEG} -- OVER it: {over}/{len(pitch)}"
        )
    else:
        print("  waist angles NOT recorded in these batches -- the rescue count is an upper bound (ledger 32.6)")


def main():
    data = {label: load(dirs) for label, dirs in GROUPS}
    for label, (pose, rej, rounds) in data.items():
        ok = sum(1 for r in rounds if r.get("ok"))
        print(f"\n===== {label}: {len(rounds)} rounds, ok {ok}, {len(pose)} reached the squat, {len(rej)} rejections")
        print("-- 1. delivered retreat vs project_squat")
        retreat(label, pose)
        print("-- 2. the waist, measured in the live solve")
        waist(label, rej)

    print("\n===== paired per case (ledger 33.2)")
    rel = {r["case_id"]: bool(r.get("ok")) for r in data["release"][2]}
    v11 = {}
    for r in data["v1.1"][2]:
        v11.setdefault(r["case_id"], []).append(bool(r.get("ok")))
    for cid in sorted(rel):
        marks = "".join("o" if x else "x" for x in v11.get(cid, []))
        print(f"  {cid:>8}  release {'ok' if rel[cid] else ' x'}   v1.1 {marks}")
    only_rel = sum(1 for c in rel if rel[c] and not any(v11.get(c, [])))
    only_v11 = sum(1 for c in rel if not rel[c] and any(v11.get(c, [])))
    print(f"  discordant: release-only {only_rel}, v1.1-only {only_v11}")

    print("\n===== 3. is the live solver self-consistent? (byte-identical repeat, same process)")
    rp = [x for _, (_, rej, _) in data.items() for x in rej if "repeat_position_error_m" in x]
    if rp:
        dp = max(abs(x["repeat_position_error_m"] - x["position_error_m"]) for x in rp)
        dr = max(abs(x["repeat_rotation_error_rad"] - x["rotation_error_rad"]) for x in rp)
        print(
            f"  n={len(rp)}  |dpe| max {dp:.2e}  |dre| max {dr:.2e}  seed mutated max {max(x['seed_mutated'] for x in rp):.2e}"
        )
        print(
            "  -> live is deterministic and the seed is recorded as passed, yet an OFFLINE re-solve of the same\n"
            "     inputs returns 0.068/0.010 against the live 0.115/0.299 (better on BOTH). Process-level and\n"
            "     UNEXPLAINED: an offline IK re-solve is not evidence here (ledger 32.3)."
        )


if __name__ == "__main__":
    main()
