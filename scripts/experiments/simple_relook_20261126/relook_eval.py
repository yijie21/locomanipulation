"""simple_relook_20261126 evaluation: the main metric with and without the second look, episode by episode.

    python -I relook_eval.py <controller> <relook bench dir> <relook sfull dir> <old bench dir> <old sfull dir> [old J arm]

[old J arm]: the old bench's rows for J (default J; JR where a tree has only JR, e.g. the development tree's bench).

Columns (as the main table): A; J (the relook run's JR rows, J with --film-ref, which equals J); S (SF on the episodes J
ran a plan on, else J's row); kin_rr (likewise). "old" is bench_rgbd_20261124 / sfull_20261125 (J's own rows there).
1. successes within 45 s and within the official limit, per task and column: relook vs old, only-relook / only-old, McNemar
   exact p over episodes, and a sign test over layouts (task and index: levels 0 and 1 share their layouts and level 2
   moves them 1-6 cm, so the 10 indices of a task are its layouts; a layout counts once, by the sign of its summed change)
2. localization (record only, against the simulator): the odometry's error and the first look's, the last look's
   (cm, deg), the stance error (planned against true) before the looks and after the last step, steps taken, the time the
   looks and steps add (s); per task
3. checks: every arm of an episode took the same looks and steps (the relook records equal, wall clock aside), and the
   same final belief; no-relook rows of the old runs carry no relook; failed or missing rows
4. failures of the relook arm by the true stance error after the steps (10 cm) and the remaining localization error (2 cm)
5. routes: episodes J ran a plan on, relook and old; both; only one
6. (added 11-26 before the formal run, the coordinator's request after LML's finding that SONIC knocks the object away
   in its correction steps) the target and the scene during the looks and steps, against the simulator, record only:
   per episode of the relook A rows, the target's displacement since the first look at each later look (so each step's
   share), episodes where it moved > 1 cm (listed, with the old A's success there), other bodies moved > 1 cm, and the
   bodies the robot touched in the relook phase (rows' hits)
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

CTRL = sys.argv[1]
RB, RS, OB, OS = (Path(x) for x in sys.argv[2:6])
TASKS = ("XMovePickTeleop", "XMoveBendPickTeleop", "BendPickMP", "TabletopGraspMP")
COLS = ("A", "J", "S", "kin_rr")


def rows(d: Path, arm: str) -> dict:
    out = {}
    for t in TASKS:
        f = d / "rows" / f"{arm}_{t}.jsonl"
        for line in open(f) if f.exists() else []:
            r = json.loads(line)
            out[(t, r["ep"])] = r
    return out


def column(bench: Path, sfull: Path, col: str, j_arm: str) -> dict:
    J = rows(bench, j_arm)
    if col == "A":
        return rows(bench, "A")
    if col == "J":
        return J
    sh = rows(sfull, "SF" if col == "S" else col)
    return {k: sh.get(k, r) for k, r in J.items()}


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def sign_layouts(new: dict, old: dict, key: str, keys) -> tuple:
    by = {}
    for k in keys:
        t, e = k
        by.setdefault((t, e % 10), 0)
        by[(t, e % 10)] += int(bool(new[k][key])) - int(bool(old[k][key]))
    up, down = sum(v > 0 for v in by.values()), sum(v < 0 for v in by.values())
    return up, down, mcnemar(up, down)


def routed(r: dict) -> bool:
    a = r.get("avoid")
    return isinstance(a, dict) and "goal_cm" in a and a["goal_cm"] <= 1.0


N = {c: column(RB, RS, c, "JR") for c in COLS}
O = {c: column(OB, OS, c, sys.argv[6] if len(sys.argv) > 6 else "J") for c in COLS}
print(f"== {CTRL}: relook {RB} / {RS}; old {OB} / {OS}")
print("1. successes, relook vs old (only relook / only old, McNemar p over episodes; layouts up/down, sign p)")
for key, lab in (("success_45", "45 s"), ("success_official", "official limit")):
    print(f"  {lab}")
    print(f"    {'task':22s}" + "".join(f"{c:>40s}" for c in COLS))
    for t in TASKS + ("all",):
        cells = []
        for c in COLS:
            ks = [k for k in O[c] if (t == "all" or k[0] == t) and k in N[c]]
            n_, o_ = sum(bool(N[c][k][key]) for k in ks), sum(bool(O[c][k][key]) for k in ks)
            b = sum(bool(N[c][k][key]) and not O[c][k][key] for k in ks)
            cc = sum(bool(O[c][k][key]) and not N[c][k][key] for k in ks)
            up, dn, ps = sign_layouts(N[c], O[c], key, ks)
            cells.append(f"{n_:3d} vs {o_:3d} ({b}/{cc} p {mcnemar(b, cc):.3f}; {up}/{dn} p {ps:.3f})")
        print(f"    {t:22s}" + "".join(f"{x:>40s}" for x in cells))
    missing = {c: len(set(O[c]) - set(N[c])) for c in COLS}
    print(f"    episodes missing in the relook run: {missing}")

print("\n2. localization and stance (record only; cm, deg): median [90th percentile] max")
A = N["A"]


def q(v):
    v = np.asarray([x for x in v if x is not None], float)
    return f"{np.median(v):6.2f} [{np.percentile(v, 90):6.2f}] {v.max():6.2f} (n {len(v)})" if len(v) else "-"


for t in TASKS + ("all",):
    rr = [r for k, r in A.items() if (t == "all" or k[0] == t) and (r.get("relook") or {}).get("looks")]
    L0 = [r["relook"]["looks"][0] for r in rr]
    LL = [r["relook"]["looks"][-1] for r in rr]
    st = [r["relook"]["steps"] for r in rr]
    print(f"  {t}  n {len(rr)}, steps taken {np.bincount(st, minlength=3).tolist() if st else []}")
    print(f"    odometry xy       {q([x['prior_err_true_cm'] for x in L0])}   yaw {q([abs(x['prior_err_true_deg']) for x in L0])}")
    print(f"    first look xy     {q([np.linalg.norm(x['est_err_true_cm']) for x in L0])}   yaw {q([abs(x['est_err_true_deg']) for x in L0])}")
    print(f"    last look xy      {q([np.linalg.norm(x['est_err_true_cm']) for x in LL])}   yaw {q([abs(x['est_err_true_deg']) for x in LL])}")
    print(f"    stance before     {q([np.linalg.norm(x['stance_err_true_cm']) for x in L0])}   yaw {q([abs(x['yaw_err_true_deg']) for x in L0])}")
    print(f"    stance after      {q([np.linalg.norm(x['stance_err_true_cm']) for x in LL])}   yaw {q([abs(x['yaw_err_true_deg']) for x in LL])}")
    print(f"    off (robot's own) {q([x['off_n_cm'] for x in L0])} -> {q([x['off_n_cm'] for x in LL])}")
    print(f"    time added, s     {q([r['relook']['t_end'] - r['relook']['looks'][0]['t'] for r in rr])}")
    print(f"    registration: inliers {q([x['inlier'] for x in L0 + LL])}, rms mm {q([x['rms_mm'] for x in L0 + LL])}, "
          f"sd y cm {q([x['sd_cm'][1] for x in L0 + LL])}")

print("\n3. checks")
strip = ("wall_s",)


def rec(r: dict):
    x = json.loads(json.dumps(r.get("relook") or {}))
    for k in strip:
        x.pop(k, None)
    return x, r.get("bel")


arms = {"A": rows(RB, "A"), "JR": rows(RB, "JR"), "SF": rows(RS, "SF"), "kin_rr": rows(RS, "kin_rr")}
for a_, R_ in arms.items():
    if a_ == "A":
        continue
    same = sum(rec(r) == rec(arms["A"][k]) for k, r in R_.items() if k in arms["A"])
    print(f"  {a_}: the relook record and final belief equal A's in {same}/{len(R_)} episodes")
olds = sum(bool(r.get("relook")) for c in COLS for r in O[c].values())
print(f"  old rows carrying a relook record: {olds}; relook A rows without one: {sum(not r.get('relook') for r in A.values())}")
print(f"  falls: relook {[k for k, r in A.items() if r.get('fell') is not None]}; "
      f"old {[k for k, r in O['A'].items() if r.get('fell') is not None]}")

print("\n4. failures of the relook arm (45 s), by the true stance error after the steps (> 10 cm) and the last look's "
      "error (> 2 cm): fail / episodes")
for c in COLS:
    out = []
    for t in TASKS + ("all",):
        cell = {}
        for k, r in N[c].items():
            if t != "all" and k[0] != t:
                continue
            L = ((A.get(k) or {}).get("relook") or {}).get("looks") or [{}]
            s_big = np.linalg.norm(L[-1].get("stance_err_true_cm", [0, 0])) > 10
            l_big = np.linalg.norm(L[-1].get("est_err_true_cm", [0, 0])) > 2
            g = ("stance>" if s_big else "stance<") + ("loc>" if l_big else "loc<")
            f_, n_ = cell.get(g, (0, 0))
            cell[g] = (f_ + (not r["success_45"]), n_ + 1)
        out.append(f"{t}: " + " ".join(f"{g} {v[0]}/{v[1]}" for g, v in sorted(cell.items())))
    print(f"  {c}: " + " | ".join(out))

print("\n5. routes (J ran a plan)")
nr = {k for k, r in N["J"].items() if routed(r)}
orr = {k for k, r in O["J"].items() if routed(r)}
print(f"  relook {len(nr)}, old {len(orr)}, both {len(nr & orr)}, only relook {sorted(nr - orr)}, only old {sorted(orr - nr)}")

print("\n6. the target and the scene during the looks and steps (truth, record only; cm since the first look)")
for t in TASKS + ("all",):
    rr = {k: r for k, r in A.items() if (t == "all" or k[0] == t) and (r.get("relook") or {}).get("looks")}
    moved, scene, touched = [], [], {}
    for k, r in sorted(rr.items()):
        L = r["relook"]["looks"]
        tm = [x.get("target_moved_true_cm") for x in L]
        if any(v is not None and v > 1.0 for v in tm):
            moved.append((k[1], [v for v in tm[1:]], bool(r["success_45"]), bool(O["A"].get(k, {}).get("success_45"))))
        sm = L[-1].get("scene_moved_true_cm")
        if sm and sm[0] > 1.0:
            scene.append((k[1], sm))
        for h in r.get("hits") or []:
            if h[0] == "relook":
                touched.setdefault(h[2], set()).add(k)
    stepped = sum(r["relook"]["steps"] > 0 for r in rr.values())
    print(f"  {t}  n {len(rr)}, stepped {stepped}; target moved > 1 cm in {len(moved)}"
          + (f" (ep, cm at each later look, relook A ok, old A ok): {moved}" if moved else ""))
    print(f"    other bodies moved > 1 cm: {len(scene)} {scene if scene else ''}")
    print(f"    touched in the relook phase (body: episodes): " + ", ".join(f"{b} {len(v)}" for b, v in sorted(touched.items())))
