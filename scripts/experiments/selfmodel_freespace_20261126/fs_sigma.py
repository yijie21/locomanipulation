"""selfmodel_freespace_20261126: the smoke's sigma, valid-pair rates, per-episode times and the episode counts the entry's
rule gives (read-only).

    FS_OUT=<smoke out> python fs_sigma.py --backends scalebfm,heft,sonic [--procs 24] [--json F]

sigma (cm): within each smoke cell, the valid pairs' R - A differences minus that cell's median, pooled over the cells,
1.4826 x MAD (the entry's "每格的局数"). Valid pairs per cell needed n = ceil(12.3 sigma^2), clipped to [16, 40];
episodes run per cell = n / the smoke's valid-pair rate (pairs valid / pairs run), rounded up, at most 60 -- applied to
HEFT by the entry, reported for every controller. Time: per-episode wall seconds by group (median and mean) and the
formal run's process-hours for 216 cells.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fs_load as FL  # noqa: E402

GROUPS = {"scalebfm": ("alone", "selfmodel", "reachable"), "heft": ("alone", "reachable"), "sonic": ("alone", "reachable")}
N_LO, N_HI, N_RUN_MAX = 16, 40, 60


def valid(r) -> bool:
    if r is None or "error" in r or r.get("err_cm") is None or r.get("fell") is not None:
        return False
    w = r.get("reach") or {}
    return w.get("label_kind") == 0 and bool(w.get("settled_end"))


def main(out: Path, backends: list[str], procs: int, js: Path | None) -> None:
    res = {}
    for be in backends:
        E, dup = FL.runs(out / be)
        cells = sorted({k[0] for k in E})
        eps = sorted({k[1] for k in E})
        dev, n_pair, n_valid, per_cell = [], 0, 0, {}
        for c in cells:
            d = []
            for e in eps:
                ra, rr = E.get((c, e, "alone")), E.get((c, e, "reachable"))
                if ra is None or rr is None:
                    continue
                n_pair += 1
                if valid(ra) and valid(rr):
                    n_valid += 1
                    d.append(rr["err_cm"] - ra["err_cm"])
            if d:
                d = np.array(d)
                dev += (d - np.median(d)).tolist()
                per_cell[c] = dict(n=len(d), med=float(np.median(d)), a_med=float(np.median([E[(c, e, "alone")]["err_cm"] for e in eps if valid(E.get((c, e, "alone")))])))
        dev = np.array(dev)
        sig = float(1.4826 * np.median(np.abs(dev - np.median(dev)))) if len(dev) else float("nan")
        n_need = int(min(N_HI, max(N_LO, math.ceil(12.3 * sig * sig))))
        rate = n_valid / n_pair if n_pair else float("nan")
        n_run = int(min(N_RUN_MAX, math.ceil(n_need / rate))) if rate > 0 else N_RUN_MAX
        wall = {m: [r["wall_s"] for k, r in E.items() if k[2] == m and "wall_s" in r] for m in GROUPS[be]}
        fails = {m: dict(fell=sum(r.get("fell") is not None and r.get("lost") is None for k, r in E.items() if k[2] == m),
                         lost=sum(r.get("lost") is not None for k, r in E.items() if k[2] == m),
                         unsettled=sum(r.get("err_cm") is not None and r.get("fell") is None and not valid(r) for k, r in E.items() if k[2] == m),
                         errors=sum("error" in r for k, r in E.items() if k[2] == m), n=sum(k[2] == m for k in E)) for m in GROUPS[be]}  # fmt: skip
        mean_ep = sum(float(np.mean(v)) for v in wall.values() if v)  # one (cell, ep): every group once
        res[be] = dict(records=len(E), duplicates=len(dup), cells=len(cells), eps=len(eps), pairs=n_pair, valid=n_valid, valid_rate=rate,
                       sigma_cm=sig, n_need=n_need, n_run_if_rate=n_run, resolvable_at_40_cm=3.5 * sig / math.sqrt(40),
                       wall={m: dict(n=len(v), median=float(np.median(v)) if v else None, mean=float(np.mean(v)) if v else None) for m, v in wall.items()},
                       fails=fails, sec_per_cell_ep=mean_ep, per_cell=per_cell)  # fmt: skip
        print(f"== {be}: {len(E)} records ({len(dup)} duplicates), {len(cells)} cells x {len(eps)} episode ids; pairs {n_pair}, valid {n_valid} "
              f"({100 * rate:.1f} %); sigma {sig:.3f} cm -> n = ceil(12.3 sigma^2) clipped = {n_need} valid pairs; episodes per cell at this "
              f"rate {n_run}; resolvable at 40: {3.5 * sig / math.sqrt(40):.2f} cm")  # fmt: skip
        print("   wall s per episode: " + "; ".join(f"{m} median {v['median']:.1f} mean {v['mean']:.1f} (n {v['n']})" for m, v in res[be]["wall"].items() if v["n"]))
        print("   fails: " + "; ".join(f"{m} {v}" for m, v in fails.items()))
        print("   cells: " + "; ".join(f"{c}: n {v['n']} A {v['a_med']:.1f} R-A {v['med']:+.2f}" for c, v in per_cell.items()))
    print("\n== formal E2a estimate (216 cells): episodes and process-hours, the entry's rule (HEFT runs n / rate; others n, plus the "
          "rate's loss shown)")  # fmt: skip
    tot_h = 0.0
    for be, v in res.items():
        n_ep = v["n_run_if_rate"] if be == "heft" else v["n_need"]
        h = 216 * n_ep * v["sec_per_cell_ep"] / 3600
        tot_h += h
        print(f"  {be}: {n_ep} episode ids per cell x {len(GROUPS[be])} groups = {216 * n_ep * len(GROUPS[be])} episodes, {h:.1f} process-hours "
              f"({h / procs:.1f} h on {procs} processes)")  # fmt: skip
    print(f"  total {tot_h:.1f} process-hours: {tot_h / procs:.1f} h on {procs} processes")
    if js:
        js.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="scalebfm,heft,sonic")
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    main(Path(os.environ["FS_OUT"]), a.backends.split(","), a.procs, a.json)
