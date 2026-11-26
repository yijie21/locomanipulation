"""heft_walkstop_20261126: P3's gait and short-move rule from the formal one-shot records (P0, P1a-c), by the rule the
entry registered before the formal run. Writes p3_rule.json ({gait, dmin, ...}) for hw_run.py --p3.

    python hw_p3rule.py --out DIR [--write DIR/p3_rule.json]

d_step(s): the smallest straight run-up d such that at d and at every longer straight cell >= 9 of the cell's trials lift a
  foot during the first walk (faildiag's lift-off) and the median first-walk moved / asked is >= 0.7; none -> inf.
s*: among the settings with no fall in any cell and <= 2 crossings over all 11 cells, the least d_step; ties -> the most
  one-shot successes over the straight cells with d >= 0.10, then the slower setting (p0 < p1a < p1b < p1c). No setting
  qualifies -> p0.
dmin: d_step(s*) when it is 0.10 or 0.20 m (a walk shorter than that is split: a step back by dmin, then the walk to
  the target); 0.05 -> none (every walk is stepped); 0.30 or more / inf -> none (a step back that long is no longer an
  omnidirectional step: the plan would turn around).
P3 == P2 by construction when s* = p0 and dmin is none: then P3 is not run and P2's records stand for it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import hw_sum as S

SETTINGS = ("p0", "p1a", "p1b", "p1c")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/wuji-vefps-D/wuji-il/yijie/loco_outputs/heft_walkstop_20261126")
    ap.add_argument("--write", default="")
    a = ap.parse_args()
    rows = [r for r in S.load(Path(a.out)) if r["arm"] in SETTINGS and "error" not in r]
    info = {}
    for s in SETTINGS:
        rs = [r for r in rows if r["arm"] == s]
        if not rs:
            continue
        cells = {}
        for cid in S.STRAIGHT:
            c = [r for r in rs if r["cid"] == cid]
            lifts = sum(((r.get("first") or {}).get("lift") or 0) >= 1 for r in c)
            ratio = [((r.get("first") or {}).get("moved_fwd") or 0.0) / r["first"]["asked_fwd"] for r in c if (r.get("first") or {}).get("asked_fwd")]
            cells[cid] = dict(n=len(c), lifts=lifts, ratio=float(np.median(ratio)) if ratio else float("nan"),
                              ok=len(c) > 0 and lifts >= 9 and bool(ratio) and float(np.median(ratio)) >= 0.7)  # fmt: skip
        d_step = float("inf")
        for cid in S.STRAIGHT:
            if all(cells[k]["ok"] for k in S.STRAIGHT if k >= cid):
                d_step = S.CELLS[cid][0]
                break
        info[s] = dict(d_step=d_step, falls=sum(r.get("fell") is not None for r in rs), cross=sum(r.get("cross_m", -1) > 0 for r in rs),
                       succ=sum(bool(r.get("success")) for r in rs if r["cid"] in S.STRAIGHT and r["cid"] >= 1), n=len(rs), cells=cells)  # fmt: skip
        print(f"{s}: n {len(rs)} d_step {d_step} falls {info[s]['falls']} crossings {info[s]['cross']} straight success (d >= 0.10) {info[s]['succ']}; "
              + " ".join(f"{S.CELLS[k][0]:.2f}:{v['lifts']}/{v['n']},{v['ratio']:.2f}" for k, v in cells.items()))  # fmt: skip
    ok = [s for s in SETTINGS if s in info and info[s]["falls"] == 0 and info[s]["cross"] <= 2]
    star = min(ok, key=lambda s: (info[s]["d_step"], -info[s]["succ"], SETTINGS.index(s))) if ok else "p0"
    ds = info.get(star, {}).get("d_step", float("inf"))
    dmin = ds if ds in (0.10, 0.20) else None
    out = dict(gait=star, dmin=dmin, d_step=ds, qualified=ok, same_as_p2=star == "p0" and dmin is None, info=info)
    print(f"P3: gait {star} dmin {dmin} (d_step {ds}); qualified settings {ok}; P3 == P2 by construction: {out['same_as_p2']}")
    if a.write:
        Path(a.write).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
