"""heft_walkstop_20261126 revisions 2 + 3: the HA free-space part's numbers beside hw_sum.py's table.

    python hw_ha_sum.py --out $O/ha [--json FILE]

Prints hw_sum.table for p0, p2, p4, q0, q2, then per arm x cell: trials with any lift-off, the first walk's lift-offs,
moved / asked (first walk, median), the signed pelvis lateral error (median, + = left; cm), the failure kinds (cross,
gap > 5, lat L / R, unsettled, fall, error), and the pairing (p2 / p4 first walk == p0's; q2 == q0's). Units cm.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hw_sum as HS  # noqa: E402

ARMS = ("p0", "p2", "p4", "q0", "q2")


def kinds(r: dict) -> str:
    if "error" in r:
        return "error"
    if r.get("fell") is not None:
        return "fall"
    k = []
    if r.get("cross_m", -1) > 0:
        k.append("cross")
    if "gap_m" in r and r["gap_m"] > 0.05:
        k.append("gap>5")
    if "err_lat_m" in r and abs(r["err_lat_m"]) > 0.03:
        k.append("latL" if r["err_lat_m"] > 0 else "latR")
    if (r.get("final") or {}).get("how") != "settled":
        k.append("unsettled")
    return "+".join(k) or "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rows = HS.load(a.out)
    arms = [x for x in ARMS if any(r["arm"] == x for r in rows)]
    print(f"{len(rows)} records, arms {arms}, errors {sum('error' in r for r in rows)}")
    res = HS.table(rows, arms)
    HS.pairing(rows)
    by = defaultdict(list)
    for r in rows:
        by[(r["arm"], r["cid"])].append(r)
    print("\narm   d     n  any-lift first-lift(med) moved/asked lat cm (signed med [min, max])  failures")
    extra = {}
    for arm in arms:
        for cid in range(len(HS.CELLS)):
            rs = sorted(by.get((arm, cid), []), key=lambda r: r["rep"])
            if not rs:
                continue
            ok = [r for r in rs if "error" not in r]
            anyl = sum(r.get("liftoffs", 0) > 0 for r in ok)
            fl = [(r.get("first") or {}).get("lift") for r in ok]
            rat = [((r.get("first") or {}).get("moved_fwd") or np.nan) / r["first"]["asked_fwd"] for r in ok if (r.get("first") or {}).get("asked_fwd")]
            lat = [100 * r["err_lat_m"] for r in ok if "err_lat_m" in r]
            fk = defaultdict(int)
            for r in rs:
                if not r.get("success"):
                    fk[kinds(r)] += 1
            lat_s = f"{np.median(lat):+.1f} [{min(lat):+.1f}, {max(lat):+.1f}]" if lat else "-"
            print(f"{arm:5s} {HS.CELLS[cid][0]:4.2f} {len(ok):3d} {anyl:4d}/{len(ok):<2d} {HS.med(fl, '{:.0f}'):>10s} {HS.med(rat, '{:.2f}'):>12s} "
                  f"{lat_s:>26s}  {' '.join(f'{k}:{v}' for k, v in sorted(fk.items()))}")  # fmt: skip
            extra[f"{arm}:{cid}"] = dict(any_lift=anyl, n=len(ok), lat=lat, fail=dict(fk), ratio=rat)
    idx = {(r["arm"], r["cid"], r["rep"]): r for r in rows if "error" not in r}
    same = n = 0
    for (arm, cid, rep), r in idx.items():
        if arm == "p4" and ("p0", cid, rep) in idx:
            w0, w4 = idx[("p0", cid, rep)]["walks"][0], r["walks"][0]
            n += 1
            same += w0.get("f_go") == w4.get("f_go") and w0.get("moved_fwd") == w4.get("moved_fwd")
    if n:
        print(f"pairing p4 first walk == p0: {same}/{n}")
    if a.json:
        Path(a.json).write_text(json.dumps(dict(table={f"{k[0]}:{k[1]}": v for k, v in res.items()}, extra=extra), default=float))


if __name__ == "__main__":
    main()
