#!/usr/bin/env python3
"""Re-solve the recorded `coupled_terminal_ik` rejections with the LIVE seed.

DO NOT TRUST THIS SCRIPT'S NUMBERS -- kept only because ledger 32.3 cites it.
Its reproduction check FAILS: with the recorded target, seed and rotation target it
returns 0.068 / 0.010 where the live solver reported 0.115 / 0.299, strictly better on
both objectives. The live solve is deterministic (a byte-identical repeat in the same
process agrees to 0.00e+00) and the seed is unmutated, so the gap is process-level and
unexplained. The waist question is answered by the IN-RUN shadow solve instead
(coupled_grasp_policy.assess -> `waist_position_error_m`).

Ledger 31.5 item 1. `assess()` solves the terminal grasp with `use_waist=False`,
which pins the waist at the seed's value; an earlier re-solve from a DEFAULT-leg
seed produced a median position error of 0.011 m against the 0.040 the live
solve reported, so the target alone does not reproduce the difficulty and the
waist lead could not be tested. The run of 2026-09-12 records `seed_q` (29
joints), both pelvis frames, and the target in world, so the live solve can be
reproduced exactly -- and only then varied.

Three things this answers, in order, and the first one gates the other two:

  1. REPRODUCTION. Re-solving with the recorded seed must return the recorded
     `position_error_m`. If it does not, nothing below means anything.
  2. THE WAIST. The same target and seed with `use_waist=True`.
  3. THE RETREAT. `project_squat` predicts the pelvis retreats
     `0.8*(drop-0.015)` = 0.092 m; the measured retreat is larger (ledger 32),
     which moves the predicted frame and hence this target. Re-derive the
     target at the measured ratio and re-solve, because a prediction that is
     optimistic in depth makes the approved plan easier than the executed one.

    .venv_sim/bin/python scripts/experiments/resolve_rejections_20260912.py <run-dir>
"""

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

from locomani.primitives.arm_ik import ArmIK
from locomani.tasks.bottle.grasp_capture import IK_POSITION_TOL
from locomani.tasks.bottle.ready_grasp_policy import OFFSET, templates

PREDICTED_RATIO = 0.8
SHORTFALL = 0.015


def rot_of(quat):
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def pick_template(target_world, pelvis_pred, world_rot):
    """Which of the 9 FK templates the live solve used, by self-consistency.

    `rot = world_rot @ local_rot_i` and the target is `point - rot @ OFFSET`, so
    the template that produced a recorded target is the one whose own choice
    rule (nearest centre to `rel`) picks it back out.
    """
    rows = templates()
    hits = []
    for i, (_, _, local_rot) in enumerate(rows):
        rot = world_rot @ local_rot
        point = np.asarray(target_world) + rot @ OFFSET
        rel = world_rot.T @ (point - np.asarray(pelvis_pred))
        j = min(range(len(rows)), key=lambda k: np.linalg.norm(rows[k][1] - rel))
        if j == i:
            hits.append(i)
    return hits, rows


def main():
    run = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/experiments/retreat_seed10_20260912")
    recs = []
    for line in (run / "manifest.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        for d in (((r.get("execution") or {}).get("metrics") or {}).get("coupled_prediction_ik_errors") or []):
            if "seed_q" in d and "target_world" in d:
                d["round"] = r.get("round")
                recs.append(d)
    print(f"{len(recs)} rejection records with a live seed, from {run}")
    if not recs:
        return

    ik = ArmIK()
    rows = []
    for d in recs:
        world_rot = rot_of(d["pelvis_quat"])
        hits, tpl = pick_template(d["target_world"], d["pelvis_pred"], world_rot)
        if len(hits) != 1:
            rows.append(dict(round=d["round"], template_ambiguous=len(hits)))
            continue
        local_rot = tpl[hits[0]][2]
        seed = np.asarray(d["seed_q"], dtype=float)
        target_root = np.asarray(d["target_root"], dtype=float)
        _, pe0, re0 = ik.solve("right", target_root, local_rot, q_mj29=seed, use_waist=False, iters=80)
        _, pe1, re1 = ik.solve("right", target_root, local_rot, q_mj29=seed, use_waist=True, iters=80)
        # The same target under the MEASURED retreat: move the predicted pelvis
        # back by (ratio - 0.8) * delivered along the pelvis's own forward axis.
        drop = float(d["drop_m"])
        extra = (1.13 - PREDICTED_RATIO) * max(0.0, drop - SHORTFALL)
        fwd = world_rot[:, 0].copy()
        fwd[2] = 0.0
        fwd /= max(float(np.linalg.norm(fwd)), 1e-9)
        pelvis_true = np.asarray(d["pelvis_pred"]) - extra * fwd
        tgt_true = world_rot.T @ (np.asarray(d["target_world"]) - pelvis_true)
        _, pe2, _ = ik.solve("right", tgt_true, local_rot, q_mj29=seed, use_waist=False, iters=80)
        _, pe3, _ = ik.solve("right", tgt_true, local_rot, q_mj29=seed, use_waist=True, iters=80)
        rows.append(
            dict(
                round=d["round"],
                live_pe=float(d["position_error_m"]),
                live_re=float(d["rotation_error_rad"]),
                pe_nowaist=float(pe0),
                re_nowaist=float(re0),
                pe_waist=float(pe1),
                re_waist=float(re1),
                pe_true_nowaist=float(pe2),
                pe_true_waist=float(pe3),
                target_xy=float(np.hypot(*target_root[:2])),
                extra_m=extra,
            )
        )
    ok = [r for r in rows if "live_pe" in r]
    bad = len(rows) - len(ok)
    if bad:
        print(f"  {bad} records skipped: template not identifiable")

    def col(k):
        return [r[k] for r in ok]

    def line(label, a, b=None):
        s = f"{label:>30} median {st.median(a):.4f}  [{min(a):.4f},{max(a):.4f}]"
        if b is not None:
            s += f"   over tol({IK_POSITION_TOL}): {sum(1 for x in a if x > IK_POSITION_TOL)}/{len(a)}"
        return s

    print()
    print(line("live position error", col("live_pe"), 1))
    print(line("re-solved, use_waist=False", col("pe_nowaist"), 1))
    d = [abs(a - b) for a, b in zip(col("live_pe"), col("pe_nowaist"))]
    print(f"{'|reproduction error|':>30} median {st.median(d):.5f}  max {max(d):.5f}"
          f"   within 1 mm: {sum(1 for x in d if x < 0.001)}/{len(d)}")
    print()
    print(line("re-solved, use_waist=True", col("pe_waist"), 1))
    print(line("measured retreat, no waist", col("pe_true_nowaist"), 1))
    print(line("measured retreat, + waist", col("pe_true_waist"), 1))
    print()
    print(line("target |xy| as planned", col("target_xy")))
    out = run / "resolve_rejections.json"
    out.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
