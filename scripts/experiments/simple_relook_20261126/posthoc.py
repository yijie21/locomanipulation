"""simple_relook_20261126, analysis written after the formal run (not pre-registered; record only).

posthoc.py <relook bench> <none bench> [<look_only bench>]

1. The target's displacement split into the walk and the correction steps. The rows' `target_moved_true_cm` is measured
   from the start look (before the walk), not from the first look after it, so relook_eval's section 6 counts an episode
   whose target the walk moved too. Here: moved by the walk = the first look after the walk > 1 cm; moved by the steps =
   a later look differs from the one before it by > 1 cm (norms only, so a lower bound of the step's own push).
2. The correction steps as executed: the commanded step (look k, robot frame) against the true stance change from look k
   to look k + 1 (planned stance frame; the yaw error is a few degrees, so the frames nearly agree).
3. The reach after the steps: the pregrasp stage's hand error against the true stance error after the steps, relook vs
   none, per task.
"""

import glob
import json
import sys

import numpy as np

TASKS = ("XMovePickTeleop", "XMoveBendPickTeleop", "BendPickMP", "TabletopGraspMP")


def rows(bench):
    out = {}
    for t in TASKS:
        for line in open(f"{bench}/rows/A_{t}.jsonl"):
            r = json.loads(line)
            out[(t, r["ep"])] = r
    return out


R = rows(sys.argv[1])
N = rows(sys.argv[2])
L0 = rows(sys.argv[3]) if len(sys.argv) > 3 else None

print("1. the target's displacement: the walk vs the correction steps (truth, record only)")
for name, G in (("relook", R),) + ((("look_only", L0),) if L0 else ()):
    for t in TASKS + ("all",):
        walk, step = [], []
        for k, r in sorted(G.items()):
            if t != "all" and k[0] != t:
                continue
            tm = [x.get("target_moved_true_cm") for x in (r.get("relook") or {}).get("looks") or []]
            if tm and tm[0] is not None and tm[0] > 1.0:
                walk.append((k[1], tm[0]))
            inc = [tm[i] - tm[i - 1] for i in range(1, len(tm)) if tm[i] is not None and tm[i - 1] is not None]
            if any(abs(v) > 1.0 for v in inc):
                step.append((k[1], [round(v, 2) for v in inc], bool(r["success_45"]), bool(N[k]["success_45"])))
        print(f"  {name:9s} {t:20s} moved by the walk {len(walk)} {walk if walk and t != 'all' else ''}")
        print(f"  {'':9s} {'':20s} moved by the steps {len(step)}" + (f" (ep, cm per step, relook A ok, none A ok): {step}" if step and t != "all" else ""))

print("\n2. correction steps as executed (relook A): commanded vs the true stance change")
for t in TASKS + ("all",):
    v = []
    for k, r in sorted(R.items()):
        if t != "all" and k[0] != t:
            continue
        looks = (r.get("relook") or {}).get("looks") or []
        for i in range(len(looks) - 1):
            s = looks[i].get("step")
            if not s:
                continue
            c = 100 * np.asarray(s[:2], float)
            a = np.asarray(looks[i + 1]["stance_err_true_cm"], float) - np.asarray(looks[i]["stance_err_true_cm"], float)
            n = float(np.linalg.norm(c))
            u = c / n
            v.append((n, float(a @ u) / n, float(np.linalg.norm(a - (a @ u) * u)), float(np.linalg.norm(a - c))))
    v = np.asarray(v)
    print(f"  {t:20s} steps {len(v):3d}  commanded {np.median(v[:, 0]):5.1f} cm  along / commanded {np.median(v[:, 1]):.2f} "
          f"[p10 {np.percentile(v[:, 1], 10):.2f}, p90 {np.percentile(v[:, 1], 90):.2f}]  across {np.median(v[:, 2]):5.1f} cm  "
          f"|true - commanded| {np.median(v[:, 3]):5.1f} [p90 {np.percentile(v[:, 3], 90):5.1f}] cm")


def stance_after(r):
    looks = (r.get("relook") or {}).get("looks") or []
    return float(np.linalg.norm(looks[-1]["stance_err_true_cm"])) if looks else None


def pre_err(r):
    for s in r.get("stages") or []:
        if s.get("stage") == "pregrasp":
            return s.get("err_cm")
    return None


print("\n3. the reach after the steps: pregrasp hand error (cm), median [p90], and successes, by the true stance error after")
print("   the steps (relook; none has no looks, so its stance error is the relook's first look, the same walk)")
for t in TASKS + ("all",):
    for name, G in (("none", N), ("relook", R)):
        cells = {}
        for k, r in sorted(G.items()):
            if t != "all" and k[0] != t:
                continue
            rr = R[k]
            looks = (rr.get("relook") or {}).get("looks") or []
            if not looks:
                continue
            st = float(np.linalg.norm(looks[0]["stance_err_true_cm"])) if name == "none" else stance_after(rr)
            b = "<=10" if st <= 10 else ("10-20" if st <= 20 else ">20")
            cells.setdefault(b, []).append((pre_err(r), bool(r["success_45"])))
        out = []
        for b in ("<=10", "10-20", ">20"):
            c = cells.get(b, [])
            e = [x for x, _ in c if x is not None]
            if c:
                out.append(f"{b}: n {len(c):2d} ok {sum(s for _, s in c):2d} err {np.median(e):5.1f} [{np.percentile(e, 90):5.1f}]" if e else f"{b}: n {len(c)}")
        print(f"  {t:20s} {name:6s} " + " | ".join(out))
