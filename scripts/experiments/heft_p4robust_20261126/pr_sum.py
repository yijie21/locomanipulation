"""heft_p4robust_20261126 (the heft_reanchor2x2 review's follow-ups): tables, checks and the registered readings.

    python pr_sum.py --out DIR [--x2 DIR] [--json FILE]

Arms (heft_walkstop_20261126/hw_run.py), all HEFT on gait_heft unless noted, seeds rep 300+ (a new family):
  p4j4  P4 (odometry, own-FK toe aim), the line's jitter +-5 cm, up to 4 re-issues -- the main row
  p4j   the same with up to 3 re-issues (P4 as in the 2x2) -- the cap's paired reference
  p2tj / p2tj1  P2t (simulator pelvis; measurement only), jitter +-5 cm, stop when < 2 cm / < 1 cm remain
  lh* / lq* / lb*  pure lateral walks (program lat; HEFT / SONIC / ScaleBFM; n / s / w = from the stand / after a 3 cm
        side step to the right / after a 0.30 m walk ahead), +-2 / 3 / 4 cm, + = left
  p4 / p2 (reps 0-9, the 2x2's seeds) -- the default-path check against the 2x2's records (--x2)
Success is hw_run.trial's (heft_walkstop revision 1). Columns as rx_sum.py; units cm.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "heft_reanchor2x2_20261126"))
import rx_sum as RX  # noqa: E402

CELLS = RX.CELLS + ((1.10, 0.0), (1.20, 0.0), (1.30, 0.0), (1.40, 0.0),
                    (0.0, 0.02), (0.0, -0.02), (0.0, 0.03), (0.0, -0.03), (0.0, 0.04), (0.0, -0.04))  # fmt: skip
PLAN = ("p4j4", "p4j", "p2tj", "p2tj1")
STRAIGHT = (0, 1, 2, 3, 4, 5, 13, 14, 15, 16, 6)  # the main reading's cells: straight 0.05-1.50 m, in distance order
OFFSET = (7, 8, 9, 10)
P2T_CELLS = (5, 13, 14, 15, 16, 6)
LAT = tuple(f"l{c}{k}" for c in "hqb" for k in "nsw")
LAT_CIDS = tuple(range(17, 23))
CHECK = ("p4", "p2")


def plan_table(rows: dict, arms=PLAN) -> dict:
    by = defaultdict(list)
    for (arm, cid, rep), r in rows.items():
        by[(arm, cid)].append(r)
    res = {}
    print(f"{'arm':5s} {'d':>5s} {'lat':>5s} {'n':>3s} {'succ':>5s} {'cross':>5s} {'behind':>6s} {'trans':>5s} {'fall':>4s} {'s.2':>3s} {'s.5':>3s} "
          f"{'gap cm (mean ± sd [min, max])':>31s} {'fwd err':>6s} {'|lat|':>5s} {'re':>4s} {'cap':>3s} {'reach':>5s} {'odo':>5s} {'settle':>6s}  fail kinds")  # fmt: skip
    for arm in arms:
        for cid in STRAIGHT + OFFSET:
            rs = by.get((arm, cid))
            if not rs:
                continue
            st = RX.cell_stats(rs)
            ok = [r for r in rs if "gap_m" in r]
            d, lat = CELLS[cid]
            reach = [100 * ((r["final"]["front"] - r["final"]["fwd"]) - r["reach_true"]) for r in ok]
            cap = 4 if arm == "p4j4" else 3
            fk = defaultdict(int)
            for r in rs:
                if r.get("success"):
                    continue
                k = []
                if "error" in r:
                    k.append("error")
                elif r.get("fell") is not None:
                    k.append("fall")
                else:
                    if r.get("cross_m", -1) > 0:
                        k.append("cross")
                    if "gap_m" in r and r["gap_m"] > RX.GAP_OK[1]:
                        k.append("gap>5")
                    if "err_lat_m" in r and abs(r["err_lat_m"]) > RX.LAT_OK:
                        k.append("lat" + ("L" if r["err_lat_m"] > 0 else "R"))
                    if r.get("final", {}).get("how") != "settled":
                        k.append("unsettled")
                fk["+".join(k) or "other"] += 1
            st["fail"] = dict(fk)
            print(f"{arm:5s} {d:5.2f} {lat:+5.2f} {st['n']:3d} {st['succ']:2d}/{st['n']:<2d} {st['cross']:5d} {st['behind']:6d} {st['trans']:5d} {st['fall']:4d} "
                  f"{st['s2']:3d} {st['s5']:3d} {RX.ms([100 * r['gap_m'] for r in ok]):>31s} {RX.med([100 * r['err_fwd_m'] for r in ok]):>6s} "
                  f"{RX.med([abs(100 * r['err_lat_m']) for r in ok]):>5s} {np.mean([r['n_reissue'] for r in ok]) if ok else float('nan'):4.1f} "
                  f"{sum(r['n_reissue'] >= cap for r in ok):3d} {RX.med(reach):>5s} {RX.med([r['final']['odo_err_cm'] for r in ok]):>5s} "
                  f"{RX.med([r.get('settle_s') for r in ok]):>6s}  {' '.join(f'{k}:{v}' for k, v in sorted(fk.items()))}")  # fmt: skip
            res[(arm, cid)] = st
    return res


def passes(res: dict, arm: str, cid: int, n: int = 20) -> bool:
    st = res.get((arm, cid))
    return bool(st) and st["n"] == n and st["err"] == 0 and st["succ"] >= 18 and st["cross"] <= 1 and st["fall"] == 0


def readings(res: dict) -> dict:
    out = {}
    for arm in ("p4j4", "p4j"):
        bad = [CELLS[c][0] for c in STRAIGHT if not passes(res, arm, c)]
        out[arm] = dict(pass_all=not bad, failing_cells_m=bad)
    a, b = out["p4j4"]["pass_all"], out["p4j"]["pass_all"]
    if a:
        rd = "PASS: P4 (4 re-issues) meets the straight criterion on the new seeds -- the LML side of 'P4 into the task episodes' holds"
    elif b:
        rd = "CAP: P4 with 3 re-issues passes, with 4 it does not -- the 4th re-issue hurts; the candidate stays at 3"
    else:
        rd = "FAIL: P4 does not hold on the new seeds (with 3 or 4 re-issues) -- not into the task episodes from LML's side"
    out["reading_1"] = rd
    c1 = {c: res.get(("p2tj1", c), {}).get("cross", 0) - res.get(("p2tj", c), {}).get("cross", 0) for c in P2T_CELLS}
    tot = sum(c1.values())
    out["p2t_cross_diff_by_cell"] = {CELLS[c][0]: v for c, v in c1.items()}
    out["reading_2"] = ("STOP-DEPENDENT: with the stop at 1 cm, P2t crosses >= 6 more times over 1.00-1.50 m -- 'removing the odometry alone is "
                        "enough' leans on the 2 cm stop" if tot >= 6 else
                        "NOT STOP-DEPENDENT: < 6 more crossings with the stop at 1 cm")  # fmt: skip
    print(f"reading 1: {rd}")
    print(f"  p4j4 failing straight cells: {out['p4j4']['failing_cells_m']}; p4j: {out['p4j']['failing_cells_m']}")
    print(f"reading 2 (measurement): p2tj1 - p2tj crossings by cell {out['p2t_cross_diff_by_cell']}, total {tot}: {out['reading_2']}")
    return out


def lat_table(rows: dict) -> dict:
    by = defaultdict(list)
    for (arm, cid, rep), r in rows.items():
        by[(arm, cid)].append(r)
    res = {}
    print(f"{'arm':4s} {'ask':>5s} {'n':>3s} {'moved':>5s} {'ratio med [min, max]':>22s} {'fwd':>5s} {'lift':>4s} {'steps':>5s} kinds")
    for arm in LAT:
        for cid in LAT_CIDS:
            rs = by.get((arm, cid))
            if not rs:
                continue
            t = [r["lat_test"] for r in rs if r.get("lat_test")]
            ask = CELLS[cid][1]
            rat = [x["ratio"] for x in t if x.get("ratio") is not None]
            mv = sum(1 for x in t if np.sign(x["moved_lat"]) == np.sign(x["asked_lat"]) and abs(x["moved_lat"]) >= 0.5 * abs(x["asked_lat"]))
            kinds = defaultdict(int)
            for x in t:
                kinds["/".join(x.get("kinds") or ["-"])] += 1
            res[(arm, cid)] = dict(n=len(rs), n_test=len(t), moved=mv, ratio_med=float(np.median(rat)) if rat else None,
                                   fell=sum(r.get("fell") is not None for r in rs), err=sum("error" in r for r in rs))  # fmt: skip
            rr = f"{np.median(rat):+.2f} [{min(rat):+.2f}, {max(rat):+.2f}]" if rat else "-"
            print(f"{arm:4s} {100 * ask:+5.0f} {len(rs):3d} {mv:2d}/{len(t):<2d} {rr:>22s} {RX.med([100 * x['moved_fwd'] for x in t]):>5s} "
                  f"{RX.med([x['lift'] for x in t]):>4s} {RX.med([x['n_steps'] for x in t]):>5s} {dict(kinds)}  fell {res[(arm, cid)]['fell']} err {res[(arm, cid)]['err']}")  # fmt: skip
    print("asymmetry (left ratio median / right ratio median, |asks| 2-4 cm pooled):")
    asym = {}
    for arm in LAT:
        L = [r["lat_test"]["ratio"] for (a, c, _), r in rows.items() if a == arm and CELLS[c][1] > 0 and r.get("lat_test") and r["lat_test"]["ratio"] is not None]
        R = [r["lat_test"]["ratio"] for (a, c, _), r in rows.items() if a == arm and CELLS[c][1] < 0 and r.get("lat_test") and r["lat_test"]["ratio"] is not None]
        if L and R:
            asym[arm] = dict(left=float(np.median(L)), right=float(np.median(R)))
            print(f"  {arm}: left {np.median(L):+.2f} (n {len(L)}), right {np.median(R):+.2f} (n {len(R)})")
    return dict(cells={f"{a}_c{c}": v for (a, c), v in res.items()}, asym=asym)


def check_defaults(rows: dict, x2: dict) -> dict:
    keys = [k for k in rows if k[0] in CHECK]
    same = 0
    for k in keys:
        a, b = rows[k], x2.get(k)
        if b is None:
            continue
        fa = (a.get("gap_m"), a.get("cross_m"), a.get("steps"), [w.get("moved_fwd") for w in a["walks"]], [w.get("moved_lat") for w in a["walks"]])
        fb = (b.get("gap_m"), b.get("cross_m"), b.get("steps"), [w.get("moved_fwd") for w in b["walks"]], [w.get("moved_lat") for w in b["walks"]])
        same += fa == fb
    print(f"default path (p4 / p2 here == the 2x2's records: gap, cross, steps, every walk's moved_fwd / moved_lat): {same}/{len(keys)}")
    return dict(same=same, n=len(keys))


def first_walk_pairs(rows: dict) -> dict:
    out = {}
    for arm in ("p4j", "p2tj", "p2tj1"):
        ks = [k for k in rows if k[0] == arm]
        same = sum(1 for (_, c, rep) in ks if ("p4j4", c, rep) in rows
                   and rows[(arm, c, rep)]["walks"][0].get("moved_fwd") == rows[("p4j4", c, rep)]["walks"][0].get("moved_fwd")
                   and rows[(arm, c, rep)].get("jitter") == rows[("p4j4", c, rep)].get("jitter"))  # fmt: skip
        n = sum(1 for (_, c, rep) in ks if ("p4j4", c, rep) in rows)
        out[arm] = (same, n)
        print(f"first walk == p4j4's first walk (and the same line jitter): {arm} {same}/{n}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--x2", type=Path, default=None, help="the 2x2's output dir (for the default-path check)")
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args()
    rows = RX.load(a.out, arms=PLAN + LAT + CHECK)
    errs = [k for k, r in rows.items() if "error" in r]
    print(f"records {len(rows)}, errors {len(errs)}")
    for k in errs[:10]:
        print(f"  error {k}: {rows[k]['error']}")
    res = plan_table({k: r for k, r in rows.items() if k[0] in PLAN})
    rd = readings(res)
    pairs = first_walk_pairs(rows)
    lt = lat_table({k: r for k, r in rows.items() if k[0] in LAT})
    chk = check_defaults({k: r for k, r in rows.items() if k[0] in CHECK}, RX.load(a.x2, arms=CHECK)) if a.x2 else None
    if a.json:
        a.json.write_text(json.dumps(dict(plan={f"{k[0]}_c{k[1]}": v for k, v in res.items()}, readings=rd, pairs=pairs, lat=lt, check=chk),
                                     indent=1, default=str))  # fmt: skip


if __name__ == "__main__":
    main()
