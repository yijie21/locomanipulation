"""selfmodel_fdvel_20261126's LocoManiLab cost: F (REACHWM_FDVEL=1, seeds 0, 1, 2) against T3 (train3's E3_ensemble frac
1.0, seed 0) on one controller's data3 split, read-only (data and models).

    cd $A/repo && REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 CUDA_VISIBLE_DEVICES=1 $VPY .../fd_eval_lml.py \
        --data <data3>/<backend> --t3 <data3>/<backend>/models_s0 --f <out>/<backend>/models_s0 ... --split val [--json F]

Segments as d3_eval: the split's (data.splits), kept when they did not fall or get lost and the end label is settled.
T3 sees x0 as recorded (data.FDVEL off), F the same record loaded with data.FDVEL on (data.fdvel_x0); everything else is
the same segment list. Per cell: the model's median prediction error (predicted final hand point to the real one, cm);
F / T3 = F's median over T3's median, per seed and for the mean of the seeds' medians; 95 % intervals from 1000 paired
resamples of the cell's resets (the same segments for F and T3); the seeds' spread (max / min of their medians - 1);
model / direct (direct: the hand left at the command, data.py err_cm). Cells: d3_eval's (start, hand, height, yaw,
roll, pitch, pelvis, duration) plus first reach (chain 0) / chained (chain >= 1) and their crossing with the start.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path.cwd()
assert os.environ.get("REACHWM_FDVEL", "0") == "0", "run with REACHWM_FDVEL unset: T3 and F are loaded here"
for k in ("REACHWM_GOAL_REF", "REACHWM_APPROACH", "REACHWM_QFIX"):
    assert os.environ.get(k) == "1", f"{k}=1 is train3's switch"
sys.path[:0] = [str(REPO / "scripts/experiments/reachwm_20261030"), str(REPO / "scripts/experiments/selfwm_20261029")]
import data as DT  # noqa: E402
import train as TR  # noqa: E402

M = TR.M
NB = 1000


def errs(models: Path, D: dict, idx: np.ndarray, dev: str) -> np.ndarray:
    m = TR.load_model("E3_ensemble", 1.0, D, dev, root=models)
    GD = M.tensors(D, dev)
    return M.pos_error_cm(m, GD, torch.tensor(idx, device=dev)).cpu().numpy().astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="selfmodel_data3_20261125/<backend> (data/all.npz)")
    ap.add_argument("--t3", type=Path, required=True)
    ap.add_argument("--f", type=Path, nargs="+", required=True, help="F's models_s<seed> directories")
    ap.add_argument("--split", default="val")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    f = str(a.data / "data" / "all.npz")
    DT.FDVEL = False
    D = DT.load(f)
    DT.FDVEL = True
    DF = DT.load(f)
    DT.FDVEL = False
    sp = DT.splits(D)[a.split]
    assert np.array_equal(sp, DT.splits(DF)[a.split])
    keep = (np.asarray(D["fell"])[sp] == 0) & (np.asarray(D["settled_end"])[sp] > 0)
    idx = sp[keep]
    eT = errs(a.t3, D, idx, dev)
    eF = [errs(p, DF, idx, dev) for p in a.f]
    g = lambda k: np.asarray(D[k])[idx]  # noqa: E731
    pos_d = g("err_cm").astype(np.float64)
    rs = g("reset")
    print(f"{a.data.name} {a.split}: {len(idx)} segments from {len(np.unique(rs))} resets (d3_eval's kept segments); "
          f"T3 {a.t3}/E3_ensemble_f1; F {[str(p) for p in a.f]}")  # fmt: skip

    def groups_of(sel: np.ndarray) -> list[np.ndarray]:
        r = rs[sel]
        o = np.argsort(r, kind="stable")
        return np.split(np.flatnonzero(sel)[o], np.flatnonzero(np.diff(r[o])) + 1)

    def boot(G, stat):
        n = len(G)
        v = [stat(np.concatenate([G[j] for j in rng.integers(0, n, n)])) for _ in range(NB)]
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    t, pz, ch, wk = g("target"), g("pelvis_z"), g("chain"), g("walked")
    dyaw, pitch, roll = np.abs(np.degrees(g("dyaw"))), np.degrees(g("pitch")), np.abs(np.degrees(g("roll")))
    cells = [
        ("all", "all", np.ones(len(idx), bool)),
        ("reach", "first reach", ch == 0), ("reach", "chained", ch >= 1),
        ("start", "in place", wk == 0), ("start", "walk first", wk == 1),
        ("start x reach", "in place, first", (wk == 0) & (ch == 0)), ("start x reach", "walk first, first", (wk == 1) & (ch == 0)),
        ("start x reach", "in place, chained", (wk == 0) & (ch >= 1)), ("start x reach", "walk first, chained", (wk == 1) & (ch >= 1)),
        ("hand", "right", g("side") > 0), ("hand", "left", g("side") < 0),
        ("height", "z < 0.3", t[:, 2] < 0.3), ("height", "0.3-0.6", (t[:, 2] >= 0.3) & (t[:, 2] < 0.6)),
        ("height", "0.6-0.9", (t[:, 2] >= 0.6) & (t[:, 2] < 0.9)), ("height", "0.9-1.2", (t[:, 2] >= 0.9) & (t[:, 2] < 1.2)),
        ("height", "z >= 1.2", t[:, 2] >= 1.2),
        ("yaw", "|yaw| <= 60", dyaw <= 60), ("yaw", "|yaw| > 60", dyaw > 60),
        ("roll", "|roll| <= 40", roll <= 40), ("roll", "|roll| > 40", roll > 40),
        ("pitch", "pitch < 0", pitch < 0), ("pitch", "pitch >= 0", pitch >= 0),
        ("pelvis", "standing", pz < 0), ("pelvis", "crouched", pz >= 0),
        ("duration", "reach <= 2.5 s", g("reach_s") <= 2.5), ("duration", "reach > 2.5 s", g("reach_s") > 2.5),
    ]  # fmt: skip
    K = len(eF)
    out = dict(controller=a.data.name, split=a.split, n=int(len(idx)), n_resets=int(len(np.unique(rs))), t3=str(a.t3),
               f=[str(p) for p in a.f], cells=[])  # fmt: skip
    print(f"  per cell: n, resets | T3 cm | F cm per seed | F / T3 per seed [95 %, paired reset resamples] | "
          f"F (mean of {K} seeds) / T3 [95 %] | seeds' spread | model / direct T3, F mean")
    for dim, lab, sel in cells:
        if sel.sum() == 0:
            continue
        G = groups_of(sel)
        mT = float(np.median(eT[sel]))
        mF = [float(np.median(e[sel])) for e in eF]
        rF = [m / mT for m in mF]
        ciF = [boot(G, lambda s, e=e: np.median(e[s]) / np.median(eT[s])) for e in eF]
        mean_r = float(np.mean(mF) / mT)
        ci_mean = boot(G, lambda s: np.mean([np.median(e[s]) for e in eF]) / np.median(eT[s]))
        spread = max(mF) / min(mF) - 1
        md = float(np.median(pos_d[sel]))
        c = dict(dim=dim, cell=lab, n=int(sel.sum()), n_resets=len(G), t3=mT, f=mF, f_over_t3=rF, f_over_t3_ci=ciF,
                 f_mean_over_t3=mean_r, f_mean_over_t3_ci=ci_mean, seed_spread=spread, direct=md,
                 t3_over_direct=mT / md, f_mean_over_direct=float(np.mean(mF)) / md)  # fmt: skip
        out["cells"].append(c)
        fr = ", ".join(f"{r:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]" for r, ci in zip(rF, ciF))
        print(f"    {dim + ': ' + lab:30s} {c['n']:5d} {c['n_resets']:5d} | {mT:6.3f} | {' '.join(f'{m:6.3f}' for m in mF)} | "
              f"{fr} | {mean_r:.3f} [{ci_mean[0]:.3f}, {ci_mean[1]:.3f}] | {100 * spread:4.1f} % | "
              f"{c['t3_over_direct']:.3f}, {c['f_mean_over_direct']:.3f}")  # fmt: skip
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
