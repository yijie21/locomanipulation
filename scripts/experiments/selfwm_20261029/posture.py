"""Item 1 of the improvement list: choose the pelvis height before the reach.

`sweep` runs method B (and R) from every candidate pelvis height on the grip-probe cells where B fails standing (6 and
8 cm boxes on the 0.65 m table, 8 cm on the 0.90 m table) and on a control cell (6 cm, 0.75 m): the upper bound a
perfect chooser could reach. Results to outputs/selfwm_20261029/posture/sweep/<cell>_<method>_<pel>.json.

    $PY scripts/experiments/selfwm_20261029/posture.py sweep --procs 8 --gpus 0,1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import env as E
import eval2 as EV2

OUT = E.OUT / "posture"
PELS = [None, 0.74, 0.70, 0.66]
CELLS = ["side3", "side6", "side8", "side4"]  # 6 cm / 0.65 m, 8 cm / 0.65 m, 8 cm / 0.90 m, control 6 cm / 0.75 m
# the wider test (no single height can be right everywhere): 6 cm boxes on six tables, one more candidate height
PELS_WIDE = [None, 0.74, 0.70, 0.66, 0.62]
TOPS_WIDE = [0.50, 0.55, 0.65, 0.75, 0.90, 1.00]


def wide_cells(methods="B", conds=("none", "yaw3"), with_pel=True):
    out = []
    for ti, top in enumerate(TOPS_WIDE):
        for m in methods:
            for cond in conds:
                for place in range(len(EV2.PLACES)):
                    base = dict(kind="side", box=4, method=m, cond=cond, place=place, top=top)
                    for pel in PELS_WIDE if with_pel else [None]:
                        tag = ("stand" if pel is None else f"{pel:.2f}") if with_pel else "x"
                        c = dict(base, id=f"wide{ti}_{m}_{cond}_{place}_p{tag}")
                        if with_pel:
                            c["pel"] = pel
                        out.append(c)
    return out


def sweep_cells(methods="BR", conds=("none", "yaw3")):
    out = []
    for cell in CELLS:
        for m in methods:
            for cond in conds:
                for place in range(len(EV2.PLACES)):
                    for pel in PELS:
                        base = next(x for x in EV2.cells() if x["id"] == f"{cell}_{m}_{cond}_{place}")
                        tag = "stand" if pel is None else f"{pel:.2f}"
                        out.append(dict(base, id=f"{base['id']}_p{tag}", pel=pel))
    return out


def worker(ids, wide=False):
    r = EV2.Runner2()
    by = {c["id"]: c for c in (wide_cells() if wide else sweep_cells())}
    d = OUT / ("sweep_wide" if wide else "sweep")
    for i in ids:
        f = d / f"{i}.json"
        if f.exists():
            continue
        t0 = time.time()
        res = r.run(by[i])
        res["wall_s"] = round(time.time() - t0, 1)
        f.write_text(json.dumps(res))
        print(i, res.get("lifted"), res.get("knocked_cm"), flush=True)


def run(procs, gpus, wide=False):
    (OUT / ("sweep_wide" if wide else "sweep")).mkdir(parents=True, exist_ok=True)
    ids = [c["id"] for c in (wide_cells() if wide else sweep_cells())]
    cards = gpus.split(",")
    jobs = []
    for w in range(procs):
        env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=cards[w % len(cards)])
        log = open(OUT / f"sweep_w{w}.log", "w")
        jobs.append(subprocess.Popen([str(E.SIM_PY), __file__, "worker", "--ids", ",".join(ids[w::procs])] + (["--wide"] if wide else []), stdout=log,
                                     stderr=subprocess.STDOUT, env=env, cwd=str(E.REPO)))  # fmt: skip
    print("exit codes", [j.wait() for j in jobs])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["sweep", "worker"])
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--ids", default="")
    ap.add_argument("--wide", action="store_true")
    a = ap.parse_args(argv)
    if a.what == "sweep":
        run(a.procs, a.gpus, a.wide)
    else:
        worker([x for x in a.ids.split(",") if x], a.wide)
    return 0


if __name__ == "__main__":
    sys.exit(main())
