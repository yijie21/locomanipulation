"""selfmodel_hacollect_20261126 (development, 2026-11-26): per-reset outcome rates of d3-style records from the reset logs
(w*_log.json), so HA records (smokes, xfer's) can be set against LocoManiLab's data3 for the same controller. Read only.

    python hc_rates.py DIR [DIR ...]      (each DIR holds data/w*_log.json)

Per record: resets, walk share, falls (height) / lost per reset (in place, after a walk); per slot: done / unsettled
(the 3 s wait ran out, the reach skipped) / not reached, for the first slot in place, the first slot after a walk and
the second slot; the wait before done slots (median s); data/ segments per reset (single-hand reaches done).
"""

from __future__ import annotations

import glob
import json
import sys

import numpy as np


def rates(d: str) -> None:
    R = []
    for f in sorted(glob.glob(f"{d}/data/w*_log.json")):
        R += json.load(open(f))
    if not R:
        print(f"== {d}: no logs")
        return
    walked = np.array([bool(r.get("walked")) for r in R])
    fell = np.array([r["fell"] is not None and r["lost"] is None for r in R])
    lost = np.array([r["lost"] is not None for r in R])
    print(f"== {d}: {len(R)} resets, walked {100 * walked.mean():.1f} %")
    for nm, m in (("in place", ~walked), ("after walk", walked)):
        if m.any():
            print(f"  {nm:10s} n {m.sum():6d}: fell {100 * fell[m].mean():5.1f} %, lost {100 * lost[m].mean():5.1f} %")
    G = {"first in place": [], "first after walk": [], "second": []}
    for r in R:
        for s in r["slots"]:
            g = "second" if s["slot"] > 0 else ("first after walk" if r.get("walked") else "first in place")
            G[g].append(s)
    for g, S in G.items():
        if not S:
            continue
        st = [s["status"] for s in S]
        wt = [s.get("wait_s", 0) for s in S if s["status"] == "done"]
        print(f"  slot {g:16s} n {len(S):6d}: done {100 * st.count('done') / len(S):5.1f} %, unsettled {100 * st.count('unsettled') / len(S):5.1f} %, "
              f"not reached {100 * st.count('not reached') / len(S):5.1f} %; wait before done median {np.median(wt) if wt else float('nan'):.2f} s")
    single = sum(s["status"] == "done" and s["mode"] in ("right", "left") for r in R for s in r["slots"])
    print(f"  single-hand reaches done per reset {single / len(R):.2f}; wall s/reset {np.mean([r.get('wall_s', np.nan) for r in R]):.1f}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        rates(d)
