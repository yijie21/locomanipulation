# notebook: selfmodel_simplesonicdata_20261126 -- REVISION 3, read only: the three runs (C: the first, allcpu 1 thread;
# G: nets on the GPU; T: allcpu 2 threads), each run's 4 x 2 table and the three runs merged, clustered by layout
"""    [SSD_MERGE_JSON=<out.json>] ssd_merge.py <label>=<run dir>[:<gate dir>] ...     (the first one is run C)
Per run: A -> R_D successes per task, 45 s and official; the four tasks' paired test (scmp_eval.paired: layout sign test,
McNemar, the interval by layout). Merged: every run's episode (task, ep) becomes (task, ep + 30 * i), so scmp_eval.paired's
layout (task, ep % 10) pools a layout's 3 levels x 3 runs: one sign per layout, 40 layouts. Each paired call gets its own
default_rng(0) (4000 resamples). Then revision 3's two reproduction rules on the runs after the first (45 s):
BendPick R_D - A >= +5 in every one; TabletopGrasp R_D < A in every one."""
import json
import os
import sys
from pathlib import Path

import numpy as np

CODE = os.environ.get("SSD_CODE", "/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplesonicdata_20261126/asrun/code2")
sys.path.insert(0, f"{CODE}/scripts/experiments/selfmodel_simplecmp_20261126")
import scmp_eval as SE  # noqa: E402

TASKS, SHORT, LIMITS = SE.TASKS, SE.SHORT, SE.LIMITS
runs = []
for a in sys.argv[1:]:
    lab, rest = a.split("=", 1)
    rd, _, gd = rest.partition(":")
    runs.append((lab, Path(rd), Path(gd) if gd else None))
R = {lab: {arm: SE.rows_of(rd / "rows", arm) for arm in ("A", "RD")} for lab, rd, _ in runs}
for lab, rd, gd in runs:
    n = {arm: len(R[lab][arm]) for arm in ("A", "RD")}
    assert n == {"A": 120, "RD": 120}, (lab, n)
    verdict = ""
    if gd and (gd / "gate.txt").exists():
        verdict = [x for x in (gd / "gate.txt").read_text().splitlines() if x.startswith("GATE")][-1]
    print(f"== run {lab}: {rd}/rows (A 120, RD 120); gate {gd}: {verdict}")
    print(f"  {'task':14s} " + "  ".join(f"{l:>14s}" for l, _ in LIMITS) + "   (A -> R_D, of 30)")
    for t in TASKS + ("all",):
        ks = [(tt, e) for tt in TASKS for e in range(30) if t in ("all", tt)]
        cells = []
        for _, col in LIMITS:
            a, b = (sum(bool(R[lab][arm][k].get(col)) for k in ks) for arm in ("A", "RD"))
            cells.append(f"{a:3d} -> {b:3d} ({b - a:+d})")
        print(f"  {SHORT.get(t, t):14s} " + "  ".join(f"{c:>14s}" for c in cells))
    for l, col in LIMITS:
        ks = [(t, e) for t in TASKS for e in range(30)]
        print(f"  paired, all four, {l:8s}: R_D vs A {SE.fmt_pair(SE.paired(R[lab]['A'], R[lab]['RD'], ks, col, np.random.default_rng(0)))}")
M = {arm: {} for arm in ("A", "RD")}
for i, (lab, _, _) in enumerate(runs):
    for arm in ("A", "RD"):
        for (t, e), r in R[lab][arm].items():
            M[arm][(t, e + 30 * i)] = r
out = {}
print(f"\n== merged: runs {', '.join(l for l, _, _ in runs)}; one sign per layout (task, index): 3 levels x {len(runs)} runs")
for l, col in LIMITS:
    for t in TASKS + ("all",):
        ks = [(tt, e) for tt in TASKS for e in range(30 * len(runs)) if t in ("all", tt)]
        o = SE.paired(M["A"], M["RD"], ks, col, np.random.default_rng(0))
        out[f"{l}/{t}"] = o
        print(f"  {l:8s} {SHORT.get(t, t):14s}: R_D vs A {SE.fmt_pair(o)}")
print("\n== revision 3's reproduction rules (45 s; the runs after the first)")
col = dict(LIMITS)["45 s"]
d = {lab: {t: sum(bool(R[lab]['RD'][(t, e)].get(col)) for e in range(30)) - sum(bool(R[lab]['A'][(t, e)].get(col)) for e in range(30)) for t in TASKS}
     for lab, _, _ in runs}  # fmt: skip
new = [lab for lab, _, _ in runs[1:]]
bp = all(d[lab]["BendPickMP"] >= 5 for lab in new)
tg = all(d[lab]["TabletopGraspMP"] < 0 for lab in new)
print(f"  BendPick R_D - A: " + ", ".join(f"{lab} {d[lab]['BendPickMP']:+d}" for lab, _, _ in runs) + f" -> {'REPRODUCED' if bp else 'NOT reproduced'} (every new run >= +5)")
print(f"  TabletopGrasp R_D - A: " + ", ".join(f"{lab} {d[lab]['TabletopGraspMP']:+d}" for lab, _, _ in runs) + f" -> {'REPRODUCED' if tg else 'NOT reproduced'} (every new run < 0)")
if os.environ.get("SSD_MERGE_JSON"):
    json.dump(dict(diff_45s=d, bendpick_reproduced=bp, tabletop_reproduced=tg, merged=out), open(os.environ["SSD_MERGE_JSON"], "w"), indent=1)
