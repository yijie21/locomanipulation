"""dev (heft_p4robust_20261126, after the formal run): why P4 fails on the new seeds. Per arm x cell (p4j, p4j4; straight
cells): crossings vs the line's jitter, the odometry's forward error at the read (belief - truth, + = the robot thinks it
is further ahead than it is), the final truth toe reach change and the own-FK one, and the re-issue count; the 4th
re-issue's asked forward / lateral (p4j4); the 0.05 m lateral failures; p2tj vs p2tj1 at 1.50 m (re-issues, fwd err, toe
reach). Units cm."""
import glob, json, sys
from collections import defaultdict
import numpy as np
P = sys.argv[1]
rows = {}
for fn in glob.glob(P + "/*/runs/*.jsonl"):
    for l in open(fn):
        r = json.loads(l); rows[(r["arm"], r["cid"], r["rep"])] = r
D = {0: .05, 1: .10, 2: .20, 3: .30, 4: .50, 5: 1.00, 13: 1.10, 14: 1.20, 15: 1.30, 16: 1.40, 6: 1.50}
def odo_fwd(r):
    return 100 * (r["final"]["odo"][0] - r["final"]["fwd"])
def f1(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return f"{np.median(v):+.1f} [{min(v):+.1f}, {max(v):+.1f}]" if v else "-"
print("arm   d     group      n  jitter cm          odo fwd err (belief-truth)  toe reach truth   own rN          re")
for arm in ("p4j", "p4j4"):
    for cid in (5, 13, 14, 15, 16, 6):
        rs = [r for (a, c, _), r in rows.items() if a == arm and c == cid and "final" in r]
        for g, sel in (("cross", [r for r in rs if r["cross_m"] > 0]), ("no cross", [r for r in rs if r["cross_m"] <= 0])):
            if not sel:
                continue
            print(f"{arm:5s} {D[cid]:4.2f} {g:9s} {len(sel):3d}  {f1([100 * r['jitter'] for r in sel]):18s} {f1([odo_fwd(r) for r in sel]):26s} "
                  f"{f1([100 * ((r['final']['front'] - r['final']['fwd']) - r['reach_true']) for r in sel]):16s} "
                  f"{f1([100 * (r['walks'][-1].get('reach_own_end', np.nan) - r['reach_own']) for r in sel]):16s} {np.mean([r['n_reissue'] for r in sel]):.1f}")
print("\np4j4 trials using the 4th re-issue: the 4th's asked fwd / lat (belief), plan kinds, and outcome")
k4 = defaultdict(int); out = []
for (a, c, rep), r in sorted(rows.items()):
    if a != "p4j4" or r.get("n_reissue", 0) < 4 or c not in D:
        continue
    w = [x for x in r["walks"] if x["kind"] in ("reissue", "fwd", "back")][3]
    lat_dom = abs(w.get("asked_lat", 0)) > abs(w.get("asked_fwd", 0))
    k4["lateral-dominant" if lat_dom else "forward-dominant"] += 1
    out.append((D[c], 100 * w.get("asked_fwd", np.nan), 100 * w.get("asked_lat", np.nan), (w.get("plan") or {}).get("kinds"), r["success"], r["cross_m"] > 0))
print(f"  n {len(out)}: {dict(k4)}")
by = defaultdict(list)
for d, af, al, kinds, s, cr in out:
    by[d].append((af, al, s, cr))
for d in sorted(by):
    v = by[d]
    print(f"  {d:4.2f}: n {len(v):2d} asked fwd {f1([x[0] for x in v])} lat {f1([x[1] for x in v])} success {sum(x[2] for x in v)} cross {sum(x[3] for x in v)}")
print("\n0.05 m lateral failures (p4j, p4j4): rep, jitter, first walk asked fwd, lat err, re-issues, step kinds per walk")
for arm in ("p4j", "p4j4"):
    for (a, c, rep), r in sorted(rows.items()):
        if a == arm and c == 0 and not r["success"]:
            print(f"  {arm} r{rep} jitter {100 * r['jitter']:+.1f} asked {100 * r['walks'][0]['asked_fwd']:+.1f} lat {100 * r['err_lat_m']:+.1f} re {r['n_reissue']} "
                  f"kinds {[ (w.get('plan') or {}).get('kinds') for w in r['walks']]} moved_lat {[round(100 * w.get('moved_lat', np.nan), 1) for w in r['walks']]}")
print("\np2tj vs p2tj1 by cell: re-issues, fwd err, truth toe reach change, gap (medians)")
for cid in (5, 13, 14, 15, 16, 6):
    for arm in ("p2tj", "p2tj1"):
        rs = [r for (a, c, _), r in rows.items() if a == arm and c == cid and "final" in r]
        print(f"  {D[cid]:4.2f} {arm:5s} re {np.mean([r['n_reissue'] for r in rs]):.1f} fwd err {f1([100 * r['err_fwd_m'] for r in rs]):18s} "
              f"reach {f1([100 * ((r['final']['front'] - r['final']['fwd']) - r['reach_true']) for r in rs]):18s} gap {f1([100 * r['gap_m'] for r in rs])}")
