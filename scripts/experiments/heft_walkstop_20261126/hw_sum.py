"""heft_walkstop_20261126: per arm x cell tables from the trial records, the pairing checks and the registered verdicts.

    python hw_sum.py [--out DIR] [--arms p0,p1a,...] [--json FILE]

Units: cm (gap: line - frontmost point at the read, + = before the line; cross: the largest frontmost point past the line
after the command, + = crossed; err_fwd / err_lat: pelvis - target), deg, s. Revision 1: the frontmost point is the feet's
(the toes); "xlow" counts crossings by the registered bfm_brake.frontmost_low (every robot geom with an AABB bottom below
0.37 m), shown for reference only. Success = no crossing, gap 0-5 cm, |lat| <= 3
cm, settled by the robot's own rule within 3 s of the last walk's end, no fall (hw_run.trial). Verdict (the entry's
criterion 1): an arm "solves HEFT's walking" when success >= 9/10 in every straight cell with d >= 0.10 m and no arm-wide
fall.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CELLS = ((0.05, 0.0), (0.10, 0.0), (0.20, 0.0), (0.30, 0.0), (0.50, 0.0), (1.00, 0.0), (1.50, 0.0),
         (0.20, 0.20), (0.20, -0.20), (1.00, 0.20), (1.00, -0.20))  # fmt: skip
STRAIGHT = range(7)
ORDER = ("p0", "p1a", "p1b", "p1c", "p2", "p3", "p0hg", "s0", "q0", "q2", "m050", "m035", "m025", "s2dev", "s3dev")


def load(out: Path) -> list[dict]:
    rows, seen = [], {}
    for fn in sorted(glob.glob(str(out / "*" / "runs" / "*.jsonl"))):
        for line in open(fn):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen[(r["arm"], r["cid"], r["rep"])] = r  # a rerun of a job keeps its last record
    rows = list(seen.values())
    return rows


def ms(v) -> str:
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(v):
        return "-"
    return f"{v.mean():+.1f} ± {v.std():.1f} [{v.min():+.1f}, {v.max():+.1f}]"


def med(v, f="{:.1f}") -> str:
    v = [x for x in v if x is not None and np.isfinite(x)]
    return f.format(float(np.median(v))) if v else "-"


def table(rows: list[dict], arms: list[str]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[(r["arm"], r["cid"])].append(r)
    res = {}
    hdr = (f"{'arm':5s} {'d':>5s} {'lat':>5s} {'n':>3s} {'succ':>5s} {'cross':>5s} {'xlow':>4s} {'fall':>4s} {'unset':>5s} {'gap cm (mean ± sd [min, max])':>34s} "
           f"{'|lat| cm':>8s} {'fwd err cm':>24s} {'|yaw|':>5s} {'lift':>4s} {'moved/asked':>11s} {'re':>4s} {'settle s':>8s} {'odo cm':>6s} {'err':>3s}")  # fmt: skip
    print(hdr)
    for arm in arms:
        if ARM_PROGRAM.get(arm) == "marker":
            continue
        for cid in range(len(CELLS)):
            rs = sorted(by.get((arm, cid), []), key=lambda r: r["rep"])
            if not rs:
                continue
            ok = [r for r in rs if "error" not in r]
            nerr = len(rs) - len(ok)
            fin = [r for r in ok if "gap_m" in r]
            succ = sum(bool(r.get("success")) for r in ok)
            cross = sum(r.get("cross_m", -1) > 0 for r in ok)
            xlow = sum(r.get("cross_low_m", -1) > 0 for r in ok)
            fall = sum(r.get("fell") is not None for r in ok)
            unset = sum((r.get("final") or {}).get("how") == "timeout" for r in fin)
            ratio = [((r.get("first") or {}).get("moved_fwd") or np.nan) / r["first"]["asked_fwd"] for r in ok
                     if (r.get("first") or {}).get("asked_fwd")]  # fmt: skip
            d, lat = CELLS[cid]
            res[(arm, cid)] = dict(n=len(ok), succ=succ, cross=cross, fall=fall, unset=unset, err=nerr,
                                   gap=[100 * r["gap_m"] for r in fin], ratio=ratio,
                                   lift=[r["liftoffs"] for r in ok], first_lift=[(r.get("first") or {}).get("lift") for r in ok])  # fmt: skip
            print(f"{arm:5s} {d:5.2f} {lat:+5.2f} {len(ok):3d} {succ:2d}/{len(ok):<2d} {cross:5d} {xlow:4d} {fall:4d} {unset:5d} {ms([100 * r['gap_m'] for r in fin]):>34s} "
                  f"{med([100 * abs(r['err_lat_m']) for r in fin]):>8s} {ms([100 * r['err_fwd_m'] for r in fin]):>24s} "
                  f"{med([abs(r['final']['yaw_deg']) for r in fin]):>5s} {med([r['liftoffs'] for r in ok], '{:.0f}'):>4s} {med(ratio, '{:.2f}'):>11s} "
                  f"{np.mean([r['n_reissue'] for r in ok]) if ok else 0:4.1f} {med([r.get('settle_s') for r in fin]):>8s} "
                  f"{med([r['final']['odo_err_cm'] for r in fin]):>6s} {nerr:3d}")  # fmt: skip
    return res


ARM_PROGRAM = {"m050": "marker", "m035": "marker", "m025": "marker"}


def markers(rows: list[dict], arms: list[str]) -> None:
    ms_ = [a for a in arms if ARM_PROGRAM.get(a) == "marker"]
    if not ms_:
        return
    print("\nmarker (braking carry past 1.5 m, truth; cm): arm, n, triggered, carry, pelvis past, toes past, |lat|, |yaw|, falls")
    for arm in ms_:
        rs = [r for r in rows if r["arm"] == arm and "error" not in r]
        if not rs:
            continue
        tr = [r for r in rs if r.get("carry_m") is not None]
        print(f"{arm:5s} {len(rs):3d} {len(tr):3d} carry {ms([100 * r['carry_m'] for r in tr])} pelvis {ms([100 * r['pelvis_past_m'] for r in tr])} "
              f"toes {ms([100 * r['toes_past_m'] for r in tr])} |lat| {med([100 * abs(r['final']['lat_pos']) for r in tr])} "
              f"|yaw| {med([abs(r['final']['yaw_deg']) for r in tr])} falls {sum(r.get('fell') is not None for r in rs)}")  # fmt: skip


def pairing(rows: list[dict]) -> None:
    """P2's first walk is P0's walk from the same start (same repeat, same cell): its settle row and truth moved should be
    identical (a deterministic sim); likewise q2 vs q0."""
    idx = {(r["arm"], r["cid"], r["rep"]): r for r in rows if "error" not in r}
    for a0, a2 in (("p0", "p2"), ("q0", "q2")):
        same = diff = 0
        for (arm, cid, rep), r2 in idx.items():
            if arm != a2 or (a0, cid, rep) not in idx:
                continue
            r0 = idx[(a0, cid, rep)]
            w0, w2 = (r0.get("walks") or [{}])[0], (r2.get("walks") or [{}])[0]
            if w0.get("f_go") == w2.get("f_go") and w0.get("moved_fwd") == w2.get("moved_fwd"):
                same += 1
            else:
                diff += 1
        if same + diff:
            print(f"pairing {a2} first walk == {a0}: {same}/{same + diff}")


def verdict(res: dict, arms: list[str]) -> None:
    print("\nverdict (criterion 1: success >= 9/10 in every straight cell d >= 0.10, no falls):")
    for arm in arms:
        if ARM_PROGRAM.get(arm) == "marker":
            continue
        cs = [res.get((arm, c)) for c in STRAIGHT if c >= 1]
        if any(c is None for c in cs):
            print(f"  {arm}: incomplete")
            continue
        falls = sum(res[(arm, c)]["fall"] for c in range(len(CELLS)) if (arm, c) in res)
        worst = min(c["succ"] / max(c["n"], 1) for c in cs)
        ok = all(c["succ"] >= 0.9 * c["n"] for c in cs) and falls == 0
        print(f"  {arm}: {'SOLVES' if ok else 'does not solve'} (worst straight cell d >= 0.10: {worst:.0%}; falls {falls}; "
              f"straight success {sum(c['succ'] for c in cs)}/{sum(c['n'] for c in cs)})")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/wuji-vefps-D/wuji-il/yijie/loco_outputs/heft_walkstop_20261126")
    ap.add_argument("--arms", default=",".join(ORDER))
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rows = load(Path(a.out))
    arms = [x for x in a.arms.split(",") if any(r["arm"] == x for r in rows)]
    print(f"{len(rows)} records, arms {arms}")
    res = table(rows, arms)
    markers(rows, arms)
    pairing(rows)
    verdict(res, arms)
    if a.json:
        Path(a.json).write_text(json.dumps({f"{k[0]}:{k[1]}": v for k, v in res.items()}, default=float))


if __name__ == "__main__":
    main()
