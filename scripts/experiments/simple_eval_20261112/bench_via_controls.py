"""selfmodel_bench_20261123: the control arms of a bench_via.sh output against B and C (one controller).

    python3 -I bench_via_controls.py <bench_via.sh out dir>

Arms on the same planned paths: B sent as it is, C moved by the multi-waypoint self model, D every via point and the goal
6 cm back and 2 cm up (no model), E moved by the model file's constant correction. On the episodes where every arm ran
the path (it ends at the arm's own pregrasp within 1 cm before any move): per arm the 45 s and official successes, the
pregrasp and entry errors (median, cm), the share whose right arm penetrated a table or another object during the pregrasp
reach, how far the pregrasp goal was moved; then pairs, X against Y: the difference in 45 s successes, in that share, and the
paired pregrasp error, with intervals (episodes resampled, 2000 times, 95 %). Then the 120-episode totals.
"""

import json
import sys
from pathlib import Path

import numpy as np

D = Path(sys.argv[1])
TASKS = ("XMovePickTeleop", "XMoveBendPickTeleop", "BendPickMP", "TabletopGraspMP")
RIGHT = ("right_shoulder", "right_elbow", "right_wrist", "right_hand")
ARMS = [a for a in "BCDE" if (D / "rows" / f"{a}_{TASKS[0]}.jsonl").exists()]


def rows(arm: str) -> dict:
    return {(t, json.loads(x)["ep"]): json.loads(x) for t in TASKS for x in open(D / "rows" / f"{arm}_{t}.jsonl")}


def stage(r: dict, name: str) -> dict:
    return next((s for s in r["stages"] if s["stage"] == name), {})


def routed(r: dict) -> bool:
    a = r.get("avoid")
    return isinstance(a, dict) and a.get("goal_cm", 9) <= 1.0 and bool(stage(r, "pregrasp"))


def hit(r: dict) -> bool:
    return any(h[0] == "pregrasp" and h[1].startswith(RIGHT) and h[3] > 0 for h in r.get("hits", []))


def boot(n: int, f, k: int = 2000) -> tuple:
    rng = np.random.default_rng(1)
    v = np.array([f(rng.integers(n, size=n)) for _ in range(k)])
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


R = {a: rows(a) for a in ARMS}
keys = sorted(k for k in R["B"] if all(k in R[a] and routed(R[a][k]) for a in ARMS))
print(f"{D.name}: arms {' '.join(ARMS)}; episodes where every arm ran the planned path: {len(keys)} ("
      + ", ".join(f"{t} {sum(k[0] == t for k in keys)}" for t in TASKS) + ")")  # fmt: skip
S = {a: np.array([bool(R[a][k]["success_45"]) for k in keys]) for a in ARMS}
H = {a: np.array([hit(R[a][k]) for k in keys]) for a in ARMS}
P = {a: np.array([stage(R[a][k], "pregrasp")["err_cm"] for k in keys]) for a in ARMS}
print("arm  45 s  official  pregrasp cm  entry cm  arm contact  goal moved cm (median)")
for a in ARMS:
    rs = [R[a][k] for k in keys]
    ent = [stage(r, "entry").get("err_cm") for r in rs]
    mv = [
        stage(r, "pregrasp").get("comp_goal_cm", np.linalg.norm(stage(r, "pregrasp").get("fixed_shift_cm", [0.0])))
        for r in rs
    ]
    print(f" {a}   {int(S[a].sum()):4d}  {sum(bool(r['success_official']) for r in rs):8d}  {np.median(P[a]):11.2f}  "
          f"{np.median([x for x in ent if x is not None]):8.2f}  {H[a].mean():11.3f}  {np.median(mv):.2f}")  # fmt: skip
n = len(keys)
for x, y in (("C", "D"), ("C", "E"), ("D", "B"), ("E", "B"), ("C", "B")):
    if x not in ARMS or y not in ARMS:
        continue
    lo, hi = boot(n, lambda q: S[x][q].sum() - S[y][q].sum())
    l2, h2 = boot(n, lambda q: H[x][q].mean() - H[y][q].mean())
    l3, h3 = boot(n, lambda q: np.median(P[x][q] - P[y][q]))
    print(f"{x} against {y}: 45 s successes {int(S[x].sum())} and {int(S[y].sum())}, difference {int(S[x].sum()) - int(S[y].sum()):+d} "
          f"({lo:+.0f} to {hi:+.0f}), only {x} {int((S[x] & ~S[y]).sum())}, only {y} {int((~S[x] & S[y]).sum())}; arm contact "
          f"{H[x].mean():.3f} and {H[y].mean():.3f}, difference {H[x].mean() - H[y].mean():+.3f} ({l2:+.3f} to {h2:+.3f}); pregrasp error "
          f"paired difference median {np.median(P[x] - P[y]):+.2f} cm ({l3:+.2f} to {h3:+.2f})")  # fmt: skip
print("45 s successes over all 120 episodes: " + ", ".join(f"{a} {sum(bool(r['success_45']) for r in R[a].values())}" for a in ARMS)
      + "; falls: " + ", ".join(f"{a} {sum(r.get('fell') is not None for r in R[a].values())}" for a in ARMS))  # fmt: skip
print("by task, 45 s: " + "; ".join(f"{t} " + " ".join(f"{a} {sum(bool(r['success_45']) for (tt, _), r in R[a].items() if tt == t)}" for a in ARMS)
                                    for t in TASKS))  # fmt: skip
