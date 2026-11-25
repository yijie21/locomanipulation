"""Summarise run_conditions.py output: success rates per condition and task, and per-start agreement between runs.

Agreement of two runs = the fraction of the 15 starts of a task (60 in all) on which both gave the same success. Within a
condition the two repeats set the noise floor; the cross-condition pairs are read against it.
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
TASKS = ("OpenOvenTeleop", "CloseDoorTeleop", "OpenFaucetTeleop", "OpenTrashCanTeleop")
runs: dict = {}
for d in sorted(root.glob("*_r*")):
    res = {}
    for t in TASKS:
        f = d / f"{t}.jsonl"
        if f.exists():
            for line in open(f):
                r = json.loads(line)
                res[r["episode_id"]] = (bool(r.get("success")), r.get("loss"), r.get("wall_s"))
    runs[d.name] = res
ids = sorted({i for r in runs.values() for i in r})
full = {k: r for k, r in runs.items() if len(r) == 60}
print("runs complete (60 starts):", sorted(full), "| partial:", {k: len(r) for k, r in runs.items() if len(r) != 60})
print("\nsuccess per run (of 15 per task):")
print(f"{'run':14}" + "".join(f"{t[:8]:>10}" for t in TASKS) + "     all   wall_s/start")
for k, r in runs.items():
    cells = [sum(r[i][0] for i in ids if i.startswith(t) and i in r) for t in TASKS]
    n = [sum(1 for i in ids if i.startswith(t) and i in r) for t in TASKS]
    w = np.mean([v[2] for v in r.values() if v[2]]) if r else float("nan")
    print(f"{k:14}" + "".join(f"{c:>6}/{m:<3}" for c, m in zip(cells, n)) + f" {sum(cells):>4}/{sum(n):<3} {w:8.1f}")


def agree(a, b):
    common = [i for i in ids if i in runs[a] and i in runs[b]]
    return (
        (sum(runs[a][i][0] == runs[b][i][0] for i in common) / len(common), len(common))
        if common
        else (float("nan"), 0)
    )


names = sorted(full)
print("\nper-start agreement (same success), complete runs:")
print(" " * 14 + "".join(f"{n:>13}" for n in names))
for a in names:
    print(f"{a:14}" + "".join(f"{agree(a, b)[0]:13.3f}" for b in names))
cond = lambda n: n.rsplit("_r", 1)[0]  # noqa: E731
within = [agree(a, b)[0] for a, b in itertools.combinations(names, 2) if cond(a) == cond(b)]
print(
    "\nwithin-condition repeat agreement:",
    {
        c: round(agree(f"{c}_r1", f"{c}_r2")[0], 3)
        for c in sorted({cond(n) for n in names})
        if f"{c}_r1" in full and f"{c}_r2" in full
    },
)
for c1, c2 in itertools.combinations(sorted({cond(n) for n in names}), 2):
    vals = [agree(a, b)[0] for a in names for b in names if cond(a) == c1 and cond(b) == c2]
    print(f"cross {c1} vs {c2}: mean agreement {np.mean(vals):.3f} over {len(vals)} run pairs")
print("\nfailure causes (loss) by condition, all runs:")
for k, r in runs.items():
    from collections import Counter

    print(f"{k:14}", dict(Counter(v[1] for v in r.values() if not v[0])))

# pooled success per condition (repeats pooled) and Fisher's exact test between conditions
from scipy.stats import fisher_exact

pool = {}
for k, r in runs.items():
    c = cond(k)
    s, n = pool.get(c, (0, 0))
    pool[c] = (s + sum(v[0] for v in r.values()), n + len(r))
print("\npooled success:", {c: f"{s}/{n} = {s / n:.3f}" for c, (s, n) in pool.items()})
for c1, c2 in itertools.combinations(sorted(pool), 2):
    (s1, n1), (s2, n2) = pool[c1], pool[c2]
    print(
        f"  {c1} vs {c2}: difference {s2 / n2 - s1 / n1:+.3f}, Fisher exact p = {fisher_exact([[s1, n1 - s1], [s2, n2 - s2]])[1]:.2f}"
    )
print("per task, pooled:")
for t in TASKS:
    cells = {}
    for k, r in runs.items():
        v = [x[0] for i, x in r.items() if i.startswith(t)]
        s, n = cells.get(cond(k), (0, 0))
        cells[cond(k)] = (s + sum(v), n + len(v))
    print(f"  {t:18}", {c: f"{s}/{n}" for c, (s, n) in cells.items()})
