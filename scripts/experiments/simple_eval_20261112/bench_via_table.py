"""selfmodel_bench_20261123: the tables of a bench_via.sh output (one controller).

    python3 -I bench_via_table.py <bench_via.sh out dir> [<training dataset dir of the controller's model>]

Arms: A a straight reach, B the planned path sent as it is, C the same path moved by the multi-waypoint self model.
Per task and arm: episodes, successes within the official limit and within 45 s, the palm's error at the pregrasp and at the
entry (median, cm), falls, targets knocked before the close. Then, on the episodes where B and C both ran a path that
ends at their own pregrasp (within 1 cm: the run reached the planned start), C against B paired: the pregrasp error, the
entry error, the share of episodes whose right arm (shoulder to hand) penetrated a table or another object between the
pregrasp command and the end of its settle, and over all 120 episodes the 45 s successes. Intervals: episodes resampled,
2000 times, 95 %.
With the training dataset (its table.npz): how far outside the model's training range each routed episode starts (the 29
joints at the command against the 0.5 to 99.5 % range of the training rows), and the model's expectation of the original
command's end against where arm B's hand ended.
"""

import json
import sys
from pathlib import Path

import numpy as np

D = Path(sys.argv[1])
TRAIN = Path(sys.argv[2]) if len(sys.argv) > 2 else None
TASKS = ("XMovePickTeleop", "XMoveBendPickTeleop", "BendPickMP", "TabletopGraspMP")
RIGHT = ("right_shoulder", "right_elbow", "right_wrist", "right_hand")


def rows(arm: str) -> dict:
    out = {}
    for t in TASKS:
        f = D / "rows" / f"{arm}_{t}.jsonl"
        for line in open(f) if f.exists() else []:
            r = json.loads(line)
            out[(t, r["ep"])] = r
    return out


def stage(r: dict, name: str) -> dict:
    return next((s for s in r["stages"] if s["stage"] == name), {})


def arm_hit(r: dict, name: str = "pregrasp") -> bool:
    return any(h[0] == name and h[1].startswith(RIGHT) and h[3] > 0 for h in r.get("hits", []))


def routed(r: dict) -> bool:
    a = r.get("avoid")
    return isinstance(a, dict) and "goal_cm" in a and a["goal_cm"] <= 1.0 and bool(stage(r, "pregrasp"))


def boot(n: int, f, k: int = 2000) -> tuple:
    rng = np.random.default_rng(1)
    v = np.array([f(rng.integers(n, size=n)) for _ in range(k)])
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


R = {arm: rows(arm) for arm in "ABC"}
print(f"{D.name}: rows A {len(R['A'])}, B {len(R['B'])}, C {len(R['C'])}")
print(f"{'task':22s} arm   n  paths  official  45 s   pregrasp cm   entry cm   falls  knocked  no grasp")
for t in TASKS:
    for arm in "ABC":
        rs = [r for (tt, _), r in sorted(R[arm].items()) if tt == t]
        if not rs:
            continue
        pre = [stage(r, "pregrasp").get("err_cm") for r in rs]
        ent = [stage(r, "entry").get("err_cm") for r in rs]
        med = lambda v: (
            float(np.median([x for x in v if x is not None])) if any(x is not None for x in v) else float("nan")
        )  # noqa: E731
        print(f"{t:22s}  {arm}  {len(rs):3d}  {sum(routed(r) for r in rs):5d}  {sum(bool(r.get('success_official')) for r in rs):8d}  "
              f"{sum(bool(r.get('success_45')) for r in rs):4d}   {med(pre):11.2f}   {med(ent):8.2f}   {sum(r.get('fell') is not None for r in rs):5d}  "
              f"{sum(bool(r.get('knocked')) for r in rs):7d}  {sum(r.get('why') == 'no grasp' for r in rs):8d}")  # fmt: skip

keys = sorted(k for k in R["B"] if k in R["C"])
plan = [
    k
    for k in keys
    if isinstance(R["B"][k].get("avoid"), dict)
    and R["B"][k]["avoid"].get("n_via") is not None
    and "goal_cm" in R["B"][k]["avoid"]
]
both = [k for k in keys if routed(R["B"][k]) and routed(R["C"][k])]
print(f"\nepisodes with a planned path: {len(plan)}; of them B and C both reached the planned start (goal within 1 cm): {len(both)}; "
      "by task: " + ", ".join(f"{t} {sum(k[0] == t for k in both)}" for t in TASKS))  # fmt: skip
if both:
    B, C = [R["B"][k] for k in both], [R["C"][k] for k in both]
    for name in ("pregrasp", "entry"):
        ok = [i for i in range(len(both)) if stage(B[i], name) and stage(C[i], name)]
        b = np.array([stage(B[i], name)["err_cm"] for i in ok])
        c = np.array([stage(C[i], name)["err_cm"] for i in ok])
        lo, hi = boot(len(ok), lambda q: np.median(c[q] - b[q]))
        print(f"  {name} error, {len(ok)} pairs: B {np.median(b):.2f} cm, C {np.median(c):.2f} cm; paired difference C - B median "
              f"{np.median(c - b):+.2f} cm ({lo:+.2f} to {hi:+.2f}); C worse in {int((c > b).sum())}, better in {int((c < b).sum())}")  # fmt: skip
    hb, hc = np.array([arm_hit(r) for r in B]), np.array([arm_hit(r) for r in C])
    lo, hi = boot(len(both), lambda q: hc[q].mean() - hb[q].mean())
    print(f"  right arm in penetration with a table or another object during the pregrasp reach: B {hb.mean():.3f}, C {hc.mean():.3f}; "
          f"difference {hc.mean() - hb.mean():+.3f} ({lo:+.3f} to {hi:+.3f}); only B {int((hb & ~hc).sum())}, only C {int((~hb & hc).sum())}")  # fmt: skip
    deep = lambda rs: [
        max([h[3] for h in r["hits"] if h[0] == "pregrasp" and h[1].startswith(RIGHT)], default=0.0) for r in rs
    ]  # noqa: E731
    print(f"  deepest such penetration, median over the episodes that have one: B {np.median([x for x in deep(B) if x > 0] or [0]):.2f} cm, "
          f"C {np.median([x for x in deep(C) if x > 0] or [0]):.2f} cm")  # fmt: skip
    for key in ("success_45", "success_official"):
        sb, sc = np.array([bool(r.get(key)) for r in B]), np.array([bool(r.get(key)) for r in C])
        print(
            f"  {key} on these episodes: B {int(sb.sum())}, C {int(sc.sum())}; only B {int((sb & ~sc).sum())}, only C {int((~sb & sc).sum())}"
        )
    if "comp_goal_cm" in stage(C[0], "pregrasp"):
        g = np.array([stage(r, "pregrasp").get("comp_goal_cm", np.nan) for r in C])
        v = np.array([stage(r, "pregrasp").get("comp_via_cm", np.nan) for r in C])
        top = np.array([stage(r, "pregrasp").get("comp_wrist_top", np.nan) for r in C])
        print(f"  the correction: goal moved median {np.nanmedian(g):.2f} cm, at the 8 cm limit in {np.mean(g >= 7.99):.2f} of the episodes; "
              f"largest via move median {np.nanmedian(v):.2f} cm, at the limit in {np.mean(v >= 7.99):.2f}; wanted wrist top median "
              f"{np.nanmedian(top):.3f} m, over 1.10 m in {np.mean(top > 1.10):.2f}")  # fmt: skip
        # the model's expectation of the original command's end against arm B's end (world; the stance's heading is the
        # start's within a few degrees, so the two frames are compared as they are -- a rough check)
        pred = np.array([stage(r, "pregrasp").get("comp_pred0", [np.nan] * 3) for r in C])
        act = np.array([stage(r, "pregrasp")["miss"] for r in B])
        e = np.linalg.norm(pred - act, axis=1)
        print(f"  the model's expected end offset of the original command against B's: error median {np.nanmedian(e):.2f} cm "
              f"(expected size {np.nanmedian(np.linalg.norm(pred, axis=1)):.2f} cm, B's {np.median(np.linalg.norm(act, axis=1)):.2f} cm)")  # fmt: skip
keys_all = sorted(k for k in R["B"] if k in R["C"])
if keys_all:
    sb = np.array([bool(R["B"][k].get("success_45")) for k in keys_all])
    sc = np.array([bool(R["C"][k].get("success_45")) for k in keys_all])
    lo, hi = boot(len(keys_all), lambda q: sc[q].sum() - sb[q].sum())
    print(f"\n45 s successes over {len(keys_all)} episodes: B {int(sb.sum())}, C {int(sc.sum())}; C - B {int(sc.sum()) - int(sb.sum()):+d} "
          f"({lo:+.0f} to {hi:+.0f}); only B {int((sb & ~sc).sum())}, only C {int((~sb & sc).sum())}; A {sum(bool(r.get('success_45')) for r in R['A'].values())}")  # fmt: skip
    fb, fc = (
        sum(R["B"][k].get("fell") is not None for k in keys_all),
        sum(R["C"][k].get("fell") is not None for k in keys_all),
    )
    print(f"falls: A {sum(r.get('fell') is not None for r in R['A'].values())}, B {fb}, C {fc}")
if TRAIN is not None and both:
    X = np.load(TRAIN / "table.npz")["x"][:, :29]
    lo_, hi_ = np.percentile(X, [0.5, 99.5], axis=0)
    out = {t: [] for t in TASKS}
    for k in both:
        s0 = np.array(stage(R["C"][k], "pregrasp")["S0"])
        out[k[0]].append(int(((s0 < lo_) | (s0 > hi_)).sum()))
    print("joints (of 29) outside the model's training range at the pregrasp command, routed episodes: "
          + "; ".join(f"{t} median {np.median(v):.0f} (none outside in {sum(x == 0 for x in v)} of {len(v)})" for t, v in out.items() if v))  # fmt: skip
    alle = {t: [] for t in TASKS}
    for k, r in R["A"].items():
        s = stage(r, "pregrasp")
        if s:
            s0 = np.array(s["S0"])
            alle[k[0]].append(int(((s0 < lo_) | (s0 > hi_)).sum()))
    print("the same over every episode of arm A: " + "; ".join(f"{t} median {np.median(v):.0f} (none outside in {sum(x == 0 for x in v)} of {len(v)})"
                                                               for t, v in alle.items() if v))  # fmt: skip
