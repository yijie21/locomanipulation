"""heft_reanchor2x2_20261126: the 2x2 on P2's re-issues -- re-anchor source (leg odometry | simulator pelvis, the latter a
measurement arm only) x aim (pelvis target fixed at the command | the robot's own FK toes re-derived at every settle) --
tables, pairing checks and the registered readings.

    python rx_sum.py --out DIR [--formal DIR] [--json FILE]

Arms (heft_walkstop_20261126/hw_run.py): p2 = (odo, pelvis) = heft_walkstop's P2; p2t = (truth, pelvis); p4 = (odo, toe);
p4t = (truth, toe). Success is hw_run.trial's (revision 1 of heft_walkstop: no crossing of the feet, gap 0-5 cm, |lat| <=
3 cm, settled, no fall). Units cm. Columns: behind = crossings whose final gap < 0, trans = crossings that end before the
line; s.2 / s.5 = success with a crossing tolerance of 0.2 / 0.5 cm (secondary); reach = final toe reach - start
(truth, (front - pelvis fwd) - reach_true); r1 / rN = the robot's own FK reach after the first / the last walk - start;
odo = |odometry - truth| at the read (median).
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CELLS = ((0.05, 0.0), (0.10, 0.0), (0.20, 0.0), (0.30, 0.0), (0.50, 0.0), (1.00, 0.0), (1.50, 0.0),
         (0.20, 0.20), (0.20, -0.20), (1.00, 0.20), (1.00, -0.20), (2.00, 0.0), (3.00, 0.0))  # fmt: skip
ARMS = ("p2", "p2t", "p4", "p4t")
LONG = (5, 6, 9, 10)  # the readings' cells: 1.00, 1.50, 1.00 +-0.20
STRAIGHT = (1, 2, 3, 4, 5, 6)  # heft_walkstop's criterion 1: straight, 0.10 <= d <= 1.50
GAP_OK, LAT_OK = (0.0, 0.05), 0.03


def load(out: Path, arms=ARMS) -> dict:
    seen = {}
    for fn in sorted(glob.glob(str(out / "*" / "runs" / "*.jsonl"))):
        for line in open(fn):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("arm") in arms:
                seen[(r["arm"], r["cid"], r["rep"])] = r
    return seen


def ok_tol(r: dict, tol: float) -> bool:
    if "gap_m" not in r or r.get("fell") is not None or r.get("timeout"):
        return False
    return (r["cross_m"] <= tol and GAP_OK[0] <= r["gap_m"] <= GAP_OK[1] and abs(r["err_lat_m"]) <= LAT_OK
            and r["final"]["how"] == "settled")  # fmt: skip


def fin(v) -> list:
    return [x for x in v if x is not None and np.isfinite(x)]


def ms(v) -> str:
    v = np.asarray(fin(v), float)
    return f"{v.mean():+.1f} ± {v.std():.1f} [{v.min():+.1f}, {v.max():+.1f}]" if len(v) else "-"


def med(v) -> str:
    v = fin(v)
    return f"{float(np.median(v)):+.1f}" if v else "-"


def cell_stats(rs: list[dict]) -> dict:
    ok = [r for r in rs if "error" not in r]
    cr = [r for r in ok if r.get("cross_m", -1) > 0]
    behind = sum(1 for r in cr if r.get("gap_m", 1) < 0)
    return dict(n=len(ok), err=len(rs) - len(ok), succ=sum(bool(r.get("success")) for r in ok), cross=len(cr), behind=behind,
                trans=len(cr) - behind, fall=sum(r.get("fell") is not None for r in ok),
                s2=sum(ok_tol(r, 0.002) for r in ok), s5=sum(ok_tol(r, 0.005) for r in ok))  # fmt: skip


def table(rows: dict) -> dict:
    by = defaultdict(list)
    for (arm, cid, rep), r in rows.items():
        by[(arm, cid)].append(r)
    res = {}
    print(f"{'arm':4s} {'d':>5s} {'lat':>5s} {'n':>3s} {'succ':>5s} {'cross':>5s} {'behind':>6s} {'trans':>5s} {'fall':>4s} {'s.2':>3s} {'s.5':>3s} "
          f"{'gap cm (mean ± sd [min, max])':>31s} {'fwd err':>6s} {'|lat|':>5s} {'re':>4s} {'reach':>5s} {'r1':>5s} {'rN':>5s} {'odo':>5s} {'settle':>6s}")  # fmt: skip
    for arm in ARMS:
        for cid in range(len(CELLS)):
            rs = by.get((arm, cid))
            if not rs:
                continue
            st = cell_stats(rs)
            ok = [r for r in rs if "gap_m" in r]
            d, lat = CELLS[cid]
            reach = [100 * ((r["final"]["front"] - r["final"]["fwd"]) - r["reach_true"]) for r in ok]
            r1 = [100 * (r["walks"][0].get("reach_own_end", np.nan) - r["reach_own"]) for r in ok]
            rn = [100 * (r["walks"][-1].get("reach_own_end", np.nan) - r["reach_own"]) for r in ok]
            print(f"{arm:4s} {d:5.2f} {lat:+5.2f} {st['n']:3d} {st['succ']:2d}/{st['n']:<2d} {st['cross']:5d} {st['behind']:6d} {st['trans']:5d} {st['fall']:4d} "
                  f"{st['s2']:3d} {st['s5']:3d} {ms([100 * r['gap_m'] for r in ok]):>31s} {med([100 * r['err_fwd_m'] for r in ok]):>6s} "
                  f"{med([abs(100 * r['err_lat_m']) for r in ok]):>5s} {np.mean([r['n_reissue'] for r in ok]) if ok else float('nan'):4.1f} {med(reach):>5s} {med(r1):>5s} "
                  f"{med(rn):>5s} {med([r['final']['odo_err_cm'] for r in ok]):>5s} {med([r.get('settle_s') for r in ok]):>6s}")  # fmt: skip
            res[(arm, cid)] = st
    return res


def passes(res: dict, arm: str, cid: int) -> bool:
    st = res.get((arm, cid))
    return bool(st) and st["n"] == 10 and st["succ"] >= 9 and st["cross"] <= 1 and st["fall"] == 0


def readings(res: dict) -> dict:
    pl = {a: all(passes(res, a, c) for c in LONG) for a in ARMS}
    if pl["p2t"] and pl["p4"]:
        rd = "E: either fix alone suffices (odometry and toe reach both carry the failure)"
    elif pl["p2t"]:
        rd = "A: odometry is the main cause (heft_walkstop's P2 reading overturned)"
    elif pl["p4"]:
        rd = "B: the toe reach is the main cause; P4 is the method candidate"
    elif pl["p4t"]:
        rd = "C: both are needed (P4 plus a better odometry)"
    else:
        rd = "D: neither explains it (transient crossings / the re-issue walk itself)"
    straight_p4 = all(passes(res, "p4", c) for c in STRAIGHT) and sum(res.get(("p4", c), {}).get("fall", 0) for c in range(len(CELLS))) == 0
    print(f"\nreading (cells {[CELLS[c] for c in LONG]}; pass = success >= 9/10, crossings <= 1, no fall):")
    for a in ARMS:
        print(f"  {a}: passes all four: {pl[a]}  ({', '.join(f'{res.get((a, c), {}).get('succ', '-')}/10' for c in LONG)})")
    print(f"  -> {rd}")
    print(f"P4 as a method (heft_walkstop criterion 1 on straight 0.10-1.50 m, no fall): {straight_p4}")
    return dict(passes_long=pl, reading=rd, p4_criterion1=straight_p4)


def pairing(rows: dict, formal: dict | None) -> dict:
    """(1) the first walk is the same in all four arms (moved_fwd, truth_end); (2) p2 here == heft_walkstop's formal P2."""
    same = tot = 0
    for (arm, cid, rep), r in rows.items():
        if arm == "p2" or "error" in r:
            continue
        b = rows.get(("p2", cid, rep))
        if b is None or "error" in b:
            continue
        tot += 1
        w, v = r["walks"][0], b["walks"][0]
        same += w.get("moved_fwd") == v.get("moved_fwd") and w.get("truth_end") == v.get("truth_end")
    print(f"\npairing: first walk == p2's first walk: {same}/{tot}")
    out = dict(first_walk_same=same, first_walk_n=tot)
    if formal is not None:
        s2 = t2 = 0
        for (arm, cid, rep), r in rows.items():
            f = formal.get(("p2", cid, rep))
            if arm != "p2" or f is None:
                continue
            t2 += 1
            s2 += r.get("gap_m") == f.get("gap_m") and [w.get("moved_fwd") for w in r["walks"]] == [w.get("moved_fwd") for w in f["walks"]]
        print(f"p2 here == heft_walkstop formal P2 (gap, every walk's moved_fwd): {s2}/{t2}")
        out.update(formal_same=s2, formal_n=t2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--formal", type=Path, default=None, help="heft_walkstop_20261126's output root (P2's formal records)")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    rows = load(a.out)
    errs = sum("error" in r for r in rows.values())
    print(f"records {len(rows)}, errors {errs}")
    res = table(rows)
    rd = readings(res)
    pr = pairing(rows, load(a.formal, ("p2",)) if a.formal else None)
    if a.json:
        a.json.write_text(json.dumps(dict(cells={f"{k[0]}_{k[1]}": v for k, v in res.items()}, **rd, **pr), indent=1))


if __name__ == "__main__":
    main()
