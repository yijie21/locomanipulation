"""selfmodel_faildiag_20261126 revision 4 supplement 3: the fixed coffee table against the movable one, paired by
episode (measurement only).

movable  q1hit2/   (FD_HITLOG=2, bit-identical to question 1's rerun)
fixed    q1fix/    (FD_HITLOG=2 + FD_FIX_SUPPORT=coffeetable: armature 1e8 on the table's free joint)

Per episode: the table's largest move after the look (the fix held if < FIX_OK_CM), the first row where the two runs
differ and the first robot-table contact in the movable run, whether the plan is the same, and from fd_q1_moved.episode
(the corrected definition): moved before the hand's arrival, the onset's phase, the hitter (1 s window), the object's
displacement from its look pose at the arrival and at the episode's end, the outcome.

    python fd_fix_an.py --out <dir> [--fix-root <fd out>]  (pairs only the episodes the fixed root has)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import fd_q1_moved as M

FD = M.FD
TASKS = {v: k for k, v in M.TASKS.items()}
FIX_OK_CM = 0.1


def grp(b: str) -> str:
    if b == "none":
        return "no robot contact"
    if b.startswith("robot0_left_hand"):
        return "left hand"
    if b.startswith("robot0_right_hand"):
        return "right hand"
    if any(t in b for t in ("hip", "knee", "ankle")):
        return "left leg" if "left" in b else "right leg"
    if any(t in b for t in ("wrist", "elbow", "shoulder")):
        return "left arm" if "left" in b else "right arm"
    return "torso / pelvis" if b.startswith("robot0_") else b


def one(root: Path, tn: str, ep: int) -> dict:
    task = TASKS[tn]
    r = json.loads((root / "rerun/sonic/episodes" / task / "alone" / f"ep{ep:+05d}.json").read_text())
    z = np.load(root / "rerun/sonic/steplog" / task / "alone" / f"ep{ep:+05d}.npz")
    e = M.episode(r, z, True)
    C = {str(c): i for i, c in enumerate(z["cols"])}
    rows = z["rows"]
    obj = rows[:, [C["obj_px"], C["obj_py"], C["obj_pz"]]]
    fl = e["f_look"]
    sp = np.asarray(z["sup_pos"], float).reshape(len(rows), -1, 3)
    tab = np.nanmax(100 * np.linalg.norm(sp[fl:] - sp[fl][None], axis=2))
    SH = np.asarray(z["sup_hits"], float).reshape(-1, 4)
    SH = SH[SH[:, 0] >= fl]
    names = json.loads(str(z["names"]))["bodies"]
    e.update(reason=r.get("reason"), success=bool(r.get("success")), plan=r.get("plan"), rows=rows, n=len(rows),
             end_cm=float(100 * np.linalg.norm(obj[-1] - obj[fl])), table_max_cm=float(tab),
             table_touch_first=int(SH[0, 0]) if len(SH) else None,
             table_touch_bodies=dict(Counter(grp(names[int(b)]) for b in SH[:, 1]).most_common()),
             hitter=grp(next(iter(e.get("hit_bodies_1s") or {}), "none")) if e.get("moved") else None)  # fmt: skip
    return e


def numdiff(x, y) -> float:
    """The largest numeric difference between two plan records (inf if their structure or a non-number differs)."""
    if isinstance(x, dict) and isinstance(y, dict):
        return max([numdiff(x[k], y.get(k)) for k in x] + [0.0]) if x.keys() == y.keys() else float("inf")
    if isinstance(x, list) and isinstance(y, list):
        return max([numdiff(a, b) for a, b in zip(x, y)] + [0.0]) if len(x) == len(y) else float("inf")
    if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
        return abs(float(x) - float(y))
    return 0.0 if x == y else float("inf")


def med(v) -> float:
    return round(float(np.median(v)), 1) if len(v) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--fix-root", default=str(FD / "q1fix"))
    a = ap.parse_args()
    fr = Path(a.fix_root)
    lst = [x for x in json.load(open(FD / "q1hit2/an/q1_moved_hits.json")) if x.get("moved") and "support_hitters" in x]
    per = []
    for x in sorted(lst, key=lambda x: (x["task"], x["ep"])):
        if not (fr / "rerun/sonic/steplog" / TASKS[x["task"]] / "alone" / f"ep{x['ep']:+05d}.npz").exists():
            continue
        mv, fx = one(FD / "q1hit2", x["task"], x["ep"]), one(fr, x["task"], x["ep"])
        k = min(mv["n"], fx["n"])
        diff = np.nonzero(np.any(mv["rows"][:k] != fx["rows"][:k], axis=1))[0]
        C = {str(c): i for i, c in enumerate(np.load(FD / "q1hit2/rerun/sonic/steplog" / TASKS[x["task"]] / "alone" / f"ep{x['ep']:+05d}.npz")["cols"])}
        t0 = mv["table_touch_first"] if mv["table_touch_first"] is not None else k
        dd = lambda ks: float(np.abs(mv["rows"][:t0, [C[c] for c in ks]] - fx["rows"][:t0, [C[c] for c in ks]]).max())  # noqa: E731
        row = dict(task=x["task"], ep=x["ep"], first_diff=int(diff[0]) if len(diff) else (k if mv["n"] != fx["n"] else None),
                   pre_touch_obj_m=dd(("obj_px", "obj_py", "obj_pz")), pre_touch_pel_m=dd(("pel_px", "pel_py", "pel_pz")),
                   f_look=mv["f_look"], same_plan=json.dumps(mv["plan"], sort_keys=True, default=float) == json.dumps(fx["plan"], sort_keys=True, default=float),
                   plan_diff=numdiff(mv["plan"], fx["plan"]),
                   mv_table_touch_first=mv["table_touch_first"])  # fmt: skip
        for tag, e in (("mv", mv), ("fx", fx)):
            row.update({f"{tag}_{kk}": e.get(kk) for kk in ("moved", "phase", "hitter", "at_arr_cm", "at_arr_deg", "end_cm", "reason", "success", "table_max_cm", "table_touch_bodies", "onset", "f_arr")})
        per.append(row)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fix_pairs.json").write_text(json.dumps(per, indent=0, default=float))
    print(f"pairs {len(per)} (movable q1hit2 / fixed q1fix, SONIC alone; corrected 'moved' definition, > {M.MOVE_CM} cm or > {M.MOVE_DEG} deg before the hand's arrival)")
    for tn in ("dumbbell", "wcan"):
        S = [p for p in per if p["task"] == tn]
        if not S:
            continue
        bad = [p["ep"] for p in S if p["fx_table_max_cm"] >= FIX_OK_CM]
        print(f"\n{tn}: {len(S)} pairs")
        print(f"  the fix held (table's largest move after the look < {FIX_OK_CM} cm): {len(S) - len(bad)}/{len(S)}; fixed max p50/max {np.median([p['fx_table_max_cm'] for p in S]):.4f} / {max(p['fx_table_max_cm'] for p in S):.4f} cm; not held {bad}; movable max p50 {med([p['mv_table_max_cm'] for p in S])} cm")
        fd_rel = [p["first_diff"] - p["mv_table_touch_first"] for p in S if p["first_diff"] is not None and p["mv_table_touch_first"] is not None]
        print(f"  runs first differ: before the look {sum(p['first_diff'] is not None and p['first_diff'] < p['f_look'] for p in S)}; same plan {sum(p['same_plan'] for p in S)}/{len(S)} (largest numeric difference p50/max {np.median([p['plan_diff'] for p in S]):.1e} / {max(p['plan_diff'] for p in S):.1e}); "
              f"first differing row - first robot-table contact (movable), rows p10/50/90 {np.percentile(fd_rel, [10, 50, 90]).round(0).tolist() if fd_rel else None}; identical {sum(p['first_diff'] is None for p in S)}")  # fmt: skip
        print(f"  before the movable run's first robot-table contact, largest |fixed - movable|: object p50/max {np.median([p['pre_touch_obj_m'] for p in S]):.1e} / {max(p['pre_touch_obj_m'] for p in S):.1e} m, "
              f"pelvis {np.median([p['pre_touch_pel_m'] for p in S]):.1e} / {max(p['pre_touch_pel_m'] for p in S):.1e} m; first robot-table contact s after the look p50 {np.median([(p['mv_table_touch_first'] - p['f_look']) / M.FPS for p in S if p['mv_table_touch_first'] is not None]):.1f}")  # fmt: skip
        G = [p for p in S if p["ep"] not in bad]
        print(f"  pairs with the fix held: {len(G)}")
        print(f"  moved before the arrival: movable {sum(p['mv_moved'] for p in G)}/{len(G)}, fixed {sum(p['fx_moved'] for p in G)}/{len(G)}; "
              f"pairs (movable, fixed) {dict(Counter((p['mv_moved'], p['fx_moved']) for p in G))}")  # fmt: skip
        print(f"  fixed, moved: phase {dict(Counter(p['fx_phase'] for p in G if p['fx_moved']).most_common())}; hitter (1 s window) {dict(Counter(p['fx_hitter'] for p in G if p['fx_moved']).most_common())}")
        print(f"  fixed, robot parts touching the table after the look (rows) {dict(sum((Counter(p['fx_table_touch_bodies']) for p in G), Counter()).most_common())}")
        for kk, lab in (("at_arr_cm", "object off its look pose at the hand's arrival, cm"), ("at_arr_deg", "the same, deg"), ("end_cm", "object off its look pose at the episode's end, cm")):
            a1, a2 = np.array([p[f"mv_{kk}"] for p in G]), np.array([p[f"fx_{kk}"] for p in G])
            print(f"  {lab}: movable p25/50/75 {np.percentile(a1, [25, 50, 75]).round(1).tolist()}, fixed {np.percentile(a2, [25, 50, 75]).round(1).tolist()}; "
                  f"paired fixed - movable p50 {np.median(a2 - a1):.1f}; fixed smaller in {int((a2 < a1).sum())}/{len(G)}")  # fmt: skip
        print(f"  outcome: movable {dict(Counter(p['mv_reason'] for p in G).most_common())}; fixed {dict(Counter(p['fx_reason'] for p in G).most_common())}; success {sum(p['mv_success'] for p in G)} -> {sum(p['fx_success'] for p in G)}")
        print(f"  fixed, still within {M.MOVE_CM} cm / {M.MOVE_DEG} deg at the arrival: {sum(p['fx_at_arr_cm'] <= M.MOVE_CM and p['fx_at_arr_deg'] <= M.MOVE_DEG for p in G)}/{len(G)} (movable {sum(p['mv_at_arr_cm'] <= M.MOVE_CM and p['mv_at_arr_deg'] <= M.MOVE_DEG for p in G)})")


if __name__ == "__main__":
    main()
