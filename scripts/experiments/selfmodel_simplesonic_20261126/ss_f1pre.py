"""selfmodel_simplesonic_20261126, F1-0 (read only, CPU, no simulation): how many of the main metric's A reach starts
fall inside a self model's training range.

    LOCO_REPO=<repo> REACHWM_GOAL_REF=1 ... (the model's switches) python ss_f1pre.py --train <data spec> --rows <dir> \
        --label <name> [--n-train <train.json's n_train>] [--json <out>]

A start is S0 of a stage row (pregrasp, entry): the 29 joints at the command, simple_bench's rec.S[-1][:29]
(env.self_state, the data's S order). The range: the spec's train split (data.splits, as train.py and sd_range take it),
each joint at the segment's start row (so + k0) within its 0.5 / 99.5 % (sx_eval's rule = sd_range's joints part).
In range = all 29 joints inside. Shares per task and stage, both stages, all four tasks; 95 % intervals from 1000
resamples of the layouts (task, ep % 10), seed 0. The val split's share is printed as the model's own reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ["LOCO_REPO"])
EXP = REPO / "scripts" / "experiments"
sys.path[:0] = [str(EXP / "reachwm_20261030"), str(EXP / "selfwm_20261029"), str(EXP / "selfmodel_data3_20261125"),
                str(EXP / "locomanilab_20261117")]  # fmt: skip
import data as DT  # noqa: E402
from lml_selfmodel import SCALEBFM_ORDER as ORDER  # noqa: E402

TASKS = ("XMovePickTeleop", "XMoveBendPickTeleop", "BendPickMP", "TabletopGraspMP")
STAGES = ("pregrasp", "entry")
NB = 1000


def starts(D: dict, idx: np.ndarray) -> np.ndarray:
    r = np.asarray(D["so"])[idx] + np.asarray(D["k0"])[idx].astype(int)
    return np.asarray(D["S"])[r, :29].astype(np.float64)


def boot(lay: np.ndarray, inr: np.ndarray, rng) -> tuple[float, float]:
    u = np.unique(lay)
    by = {g: np.where(lay == g)[0] for g in u}
    v = []
    for _ in range(NB):
        s = np.concatenate([by[g] for g in rng.choice(u, len(u))])
        v.append(inr[s].mean())
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    sw = {k: os.environ.get(k, "") for k in ("REACHWM_GOAL_REF", "REACHWM_APPROACH", "REACHWM_QFIX", "REACHWM_FDVEL",
                                             "REACHWM_TRAIN_HELD", "REACHWM_SUBSET")}  # fmt: skip
    D = DT.load(a.train)
    sp = DT.splits(D)
    tr, va = sp["train"], sp["val"]
    qt = starts(D, tr)
    lo, hi = np.percentile(qt, 0.5, axis=0), np.percentile(qt, 99.5, axis=0)
    qv = starts(D, va)
    inv = (((qv < lo) | (qv > hi)).sum(1) == 0).mean()
    print(f"== {a.label}: train {a.train}; switches {sw}")
    print(f"  train split {len(tr)} segments" + (f" (train.json n_train {a.n_train}: {'equal' if a.n_train == len(tr) else 'DIFFERS'})"
          if a.n_train is not None else "") + f"; val {len(va)}, val starts in range {100 * inv:.1f} %")  # fmt: skip
    recs = []
    for t in TASKS:
        f = a.rows / f"A_{t}.jsonl"
        md5 = hashlib.md5(f.read_bytes()).hexdigest()
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        print(f"  rows {f.name}: {len(rows)} episodes, md5 {md5}")
        for r in rows:
            for s in r["stages"]:
                if s["stage"] in STAGES and s.get("S0") is not None:
                    recs.append((t, int(r["ep"]), s["stage"], np.asarray(s["S0"], np.float64)))
    Q = np.stack([x[3] for x in recs])
    out = (Q < lo) | (Q > hi)
    nout = out.sum(1)
    inr = nout == 0
    task = np.array([x[0] for x in recs])
    stage = np.array([x[2] for x in recs])
    lay = np.array([f"{x[0]}:{x[1] % 10}" for x in recs])
    res = dict(label=a.label, train=a.train, switches=sw, n_train=int(len(tr)), val_in=float(inv), cells={})
    print("  in range (all 29 joints inside the train split's 0.5 / 99.5 %): share [95 %, layouts resampled]; joints out "
          "median; most often out (% of starts)")  # fmt: skip
    for t in TASKS + ("all",):
        for g in STAGES + ("both",):
            m = (task == t if t != "all" else np.ones(len(recs), bool)) & (stage == g if g != "both" else np.ones(len(recs), bool))
            if not m.any():
                continue
            ci = boot(lay[m], inr[m], rng)
            fr = out[m].mean(0)
            worst = [(ORDER[j], round(100 * float(fr[j]), 1)) for j in np.argsort(-fr)[:5] if fr[j] > 0]
            res["cells"][f"{t}/{g}"] = dict(n=int(m.sum()), share=float(inr[m].mean()), ci=ci, out_median=float(np.median(nout[m])),
                                            worst=worst)  # fmt: skip
            print(f"    {t:20s} {g:8s} n {int(m.sum()):3d}  {100 * inr[m].mean():5.1f} % [{100 * ci[0]:5.1f}, {100 * ci[1]:5.1f}]  "
                  f"out median {np.median(nout[m]):.0f}  {worst}")  # fmt: skip
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
