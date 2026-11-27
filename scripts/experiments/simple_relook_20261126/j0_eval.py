"""simple_relook_20261126, the J0 control: J replanned from the same A films without the second look, against the old J.

    python -I j0_eval.py <j0 bench dir> <old bench dir> <old J arm (J or JR)>

Per task and in all: successes within 45 s and within the official limit, J0 against old, only-J0 / only-old, McNemar
exact p; the episodes J ran a plan on (as relook_eval.py), J0 and old, both, only one. The spread here is the planner's
own noise (cuRobo's replans are not reproducible), the yardstick for the J, S and kin_rr differences of the relook run."""

import json
import math
import sys
from pathlib import Path

TASKS = ("XMovePickTeleop", "XMoveBendPickTeleop", "BendPickMP", "TabletopGraspMP")


def rows(d: Path, arm: str) -> dict:
    out = {}
    for t in TASKS:
        f = d / "rows" / f"{arm}_{t}.jsonl"
        for line in open(f) if f.exists() else []:
            r = json.loads(line)
            out[(t, r["ep"])] = r
    return out


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def routed(r: dict) -> bool:
    a = r.get("avoid")
    return isinstance(a, dict) and "goal_cm" in a and a["goal_cm"] <= 1.0


new, old = rows(Path(sys.argv[1]), "JR"), rows(Path(sys.argv[2]), sys.argv[3])
print(f"J0 rows {len(new)}, old {sys.argv[3]} rows {len(old)}, common {len(set(new) & set(old))}")
for key, lab in (("success_45", "45 s"), ("success_official", "official limit")):
    print(f"  {lab}")
    for t in TASKS + ("all",):
        ks = [k for k in old if k in new and (t == "all" or k[0] == t)]
        n_, o_ = sum(bool(new[k][key]) for k in ks), sum(bool(old[k][key]) for k in ks)
        b = sum(bool(new[k][key]) and not old[k][key] for k in ks)
        c = sum(bool(old[k][key]) and not new[k][key] for k in ks)
        print(f"    {t:22s} n {len(ks):3d}  J0 {n_:3d}  old {o_:3d}  only J0 {b:2d}  only old {c:2d}  McNemar p {mcnemar(b, c):.3f}")
nr = {k for k, r in new.items() if routed(r)}
orr = {k for k, r in old.items() if routed(r)}
print(f"  routes: J0 {len(nr)}, old {len(orr)}, both {len(nr & orr)}, only J0 {sorted(nr - orr)}, only old {sorted(orr - nr)}")
