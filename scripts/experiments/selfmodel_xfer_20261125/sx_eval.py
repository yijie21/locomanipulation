"""selfmodel_xfer_20261125's metrics: a LocoManiLab self model (train3's E3_ensemble frac 1.0, or train3b's pick) scored
on one combination's record (sx_collect.py; one-hand segments, data/all.npz), read-only.

    cd $REPO && REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 CUDA_VISIBLE_DEVICES= \
        $VPY .../sx_eval.py --lml $D3/<scalebfm|sonic> --xfer $X/<combo> [--models DIR] [--json F]

Segments, as d3_eval: kept when they did not fall or get lost and the end label is settled (settled_end); first reach =
chain 0, chained = chain >= 1, all = both. The model is rebuilt on the LocoManiLab record (train.load_model: its
normalisation comes from the state dict) and predicts the xfer segments from their own features (data.py, unchanged).
  primary     the median prediction error (predicted final hand point to the real one, cm) over the LocoManiLab val median
              of the same model and split (d3_eval's val segments, recomputed here; walk-first resets included as in the
              given val numbers). 95 % interval: 1000 resamples of the xfer resets (val fixed); the interval with val
              resampled too is printed beside it.
  secondary   direct error (the hand left at the command: target to the real hand point, data.py err_cm) median over
              LocoManiLab's in-place kept segments (all splits; in place = the xfer setting), by split; the angle between
              the mean direct error vectors (heading frame of the start, the left hand's y mirrored); the share of
              starts inside the training range (all 29 joints of S[k0] within the train split's 0.5 / 99.5 % per joint,
              hg_run.train_range's rule on train3's training segments) and the prediction error inside / outside it;
              the ensemble's member disagreement (|std over the 5 members' predicted position|, cm) median, xfer and val.
Also printed: the model / direct ratio on the xfer segments (does the LocoManiLab correction help here).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(os.environ.get("SX_REPO", "/wuji-vefps-D/wuji-il/yijie/ws/code/loco"))
for k in ("REACHWM_GOAL_REF", "REACHWM_APPROACH", "REACHWM_QFIX"):
    assert os.environ.get(k) == "1", f"{k}=1 is train3's switch"
sys.path[:0] = [str(REPO / "scripts/experiments/reachwm_20261030"), str(REPO / "scripts/experiments/selfwm_20261029")]
import data as DT  # noqa: E402
import train as TR  # noqa: E402

M = TR.M
NB = 1000
SPLITS = (("first reach", lambda c: c == 0), ("chained", lambda c: c >= 1), ("all", lambda c: c >= 0))


def groups_of(reset: np.ndarray) -> list[np.ndarray]:
    o = np.argsort(reset, kind="stable")
    return np.split(o, np.flatnonzero(np.diff(reset[o])) + 1)


def boot(G: list[np.ndarray], stat, rng) -> tuple[float, float]:
    n = len(G)
    v = [stat(np.concatenate([G[j] for j in rng.integers(0, n, n)])) for _ in range(NB)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def boot2(Ga, Gb, stat, rng) -> tuple[float, float]:
    """Both groups resampled independently: stat(ia, ib)."""
    na, nb = len(Ga), len(Gb)
    v = [stat(np.concatenate([Ga[j] for j in rng.integers(0, na, na)]), np.concatenate([Gb[j] for j in rng.integers(0, nb, nb)]))
         for _ in range(NB)]  # fmt: skip
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def angle(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))))


def kept(D: dict, idx: np.ndarray | None = None, ended: bool = False) -> np.ndarray:
    """d3_eval's segments: not fallen / lost and settled at the end label; ended: not fallen / lost only (the end label
    of an unsettled segment is the hand 3 s after the reference end plus the delay, d3_collect's timeout)."""
    idx = np.arange(len(D["n_steps"])) if idx is None else idx
    ok = np.asarray(D["fell"])[idx] == 0
    return idx[ok if ended else ok & (np.asarray(D["settled_end"])[idx] > 0)]


def predict(m, D: dict, idx: np.ndarray, dev: str) -> dict:
    GD = M.tensors(D, dev)
    ti = torch.tensor(idx, device=dev)
    p = M.predict_batched(m, GD, ti)
    err = (100 * (p["pos"] - GD["pos_f"][ti]).norm(dim=-1)).cpu().numpy()
    sd = (100 * p["pos_std"].norm(dim=-1)).cpu().numpy()
    return dict(err=err, sd=sd)


def evec(D: dict, idx: np.ndarray) -> np.ndarray:
    """Direct error vectors (heading frame of the start), the left hand's y mirrored."""
    e = (np.asarray(D["pos_f"])[idx] - np.asarray(D["target"])[idx]).astype(float)
    e[:, 1] *= np.sign(np.asarray(D["side"])[idx])
    return e


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lml", type=Path, required=True, help="selfmodel_data3_20261125/<backend>: data/all.npz + models_s0")
    ap.add_argument("--xfer", type=Path, required=True, help="the combination's record (sx_collect.py --out)")
    ap.add_argument("--models", type=Path, default=None, help="default: <lml>/models_s0")
    ap.add_argument("--model", default="E3_ensemble")
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--keep", choices=["settled", "ended"], default="settled",
                    help="settled: d3_eval's segments (the registered metric); ended: the xfer segments that did not fall, "
                         "settled at the end label or not (a supplement; LocoManiLab's side stays d3_eval's)")
    a = ap.parse_args()
    dev = "cpu"
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    rng = np.random.default_rng(0)
    models = a.models or a.lml / "models_s0"
    DL = DT.load(str(a.lml / "data" / "all.npz"))
    DX = DT.load(str(a.xfer / "data" / "all.npz"))
    m = TR.load_model(a.model, a.frac, DL, dev, root=models)
    run = json.loads((a.xfer / "run.json").read_text())
    sp = DT.splits(DL)
    vi = kept(DL, sp["val"])
    xi = kept(DX, ended=a.keep == "ended")
    nx_all = len(DX["n_steps"])
    PV, PX = predict(m, DL, vi, dev), predict(m, DX, xi, dev)
    cv, cx = np.asarray(DL["chain"])[vi], np.asarray(DX["chain"])[xi]
    rv, rx = np.asarray(DL["reset"])[vi], np.asarray(DX["reset"])[xi]
    dx = np.asarray(DX["err_cm"])[xi].astype(float)
    li = kept(DL)
    li = li[np.asarray(DL["walked"])[li] == 0]  # LocoManiLab's in-place kept segments, every split
    dl, cl, rl = np.asarray(DL["err_cm"])[li].astype(float), np.asarray(DL["chain"])[li], np.asarray(DL["reset"])[li]
    out = dict(lml=str(a.lml), xfer=str(a.xfer), model=f"{models}/{a.model}_f{a.frac:g}", run={k: run[k] for k in ("controller", "planner", "stance", "delay_f")},
               n_xfer_segments=int(nx_all), n_xfer_kept=int(len(xi)), n_xfer_resets=int(len(np.unique(rx))),
               xfer_fell=int((np.asarray(DX["fell"]) != 0).sum()), xfer_unsettled_end=int(((np.asarray(DX["fell"]) == 0) & (np.asarray(DX["settled_end"]) == 0)).sum()),
               n_val_kept=int(len(vi)), keep=a.keep, splits={})  # fmt: skip
    print(f"xfer {run['controller']} ({run['planner']} planner, {run['stance']} stance, delay {run['delay_f']} frames): {a.xfer}; "
          f"xfer segments kept: {'not fallen and settled at the end label (d3_eval)' if a.keep == 'settled' else 'not fallen, settled or not (supplement)'}")
    print(f"  model {out['model']} (built on {a.lml}/data/all.npz)")
    print(f"  xfer one-hand segments {nx_all}: kept {len(xi)} from {out['n_xfer_resets']} resets ({out['xfer_fell']} fell / lost, "
          f"{out['xfer_unsettled_end']} not settled at the end label); LocoManiLab val kept {len(vi)}")  # fmt: skip
    # primary
    print("  primary: prediction error median (cm), xfer / LocoManiLab val [95 % interval, xfer resets resampled; val too]:")
    for name, f in SPLITS:
        sv, sx = f(cv), f(cx)
        ev, ex = PV["err"][sv], PX["err"][sx]
        if len(ex) == 0:
            print(f"    {name:12s}: no kept xfer segments (val {np.median(ev):.3f} cm, n {len(ev)})")
            out["splits"][name] = dict(n=0, val=float(np.median(ev)), n_val=int(len(ev)))
            continue
        Gx, Gv = groups_of(rx[sx]), groups_of(rv[sv])
        mv = float(np.median(ev))
        r = float(np.median(ex)) / mv
        ci = boot(Gx, lambda s: np.median(ex[s]) / mv, rng)
        ci2 = boot2(Gx, Gv, lambda s, t: np.median(ex[s]) / np.median(ev[t]), rng)
        ci_x = boot(Gx, lambda s: np.median(ex[s]), rng)
        o = dict(n=int(len(ex)), n_resets=len(Gx), xfer=float(np.median(ex)), xfer_ci=ci_x, val=mv, n_val=int(len(ev)), ratio=r,
                 ratio_ci=ci, ratio_ci_val_resampled=ci2)  # fmt: skip
        # secondary: direct error vs LocoManiLab in place
        dsel = f(cl)
        dxs, dls = dx[sx], dl[dsel]
        Gl = groups_of(rl[dsel])
        o["direct_xfer"], o["direct_lml_inplace"] = float(np.median(dxs)), float(np.median(dls))
        o["direct_ratio"] = o["direct_xfer"] / o["direct_lml_inplace"]
        o["direct_ratio_ci"] = boot2(Gx, Gl, lambda s, t: np.median(dxs[s]) / np.median(dls[t]), rng)
        vx, vl = evec(DX, xi[sx]), evec(DL, li[dsel])
        mvx, mvl = vx.mean(0), vl.mean(0)
        o["mean_err_xfer_cm"], o["mean_err_lml_cm"] = (100 * mvx).tolist(), (100 * mvl).tolist()
        o["angle_deg"] = angle(mvx, mvl)
        o["angle_ci"] = boot(Gx, lambda s: angle(vx[s].mean(0), mvl), rng)
        o["model_over_direct_xfer"] = float(np.median(ex) / np.median(dxs))
        out["splits"][name] = o
        print(f"    {name:12s}: n {len(ex):4d} ({len(Gx)} resets) xfer {o['xfer']:.3f} cm [{ci_x[0]:.3f}, {ci_x[1]:.3f}] / val "
              f"{mv:.3f} cm (n {len(ev)}) = {r:.3f}x [{ci[0]:.3f}, {ci[1]:.3f}] (val resampled too [{ci2[0]:.3f}, {ci2[1]:.3f}])")  # fmt: skip
    print("  secondary: direct error median (cm), xfer / LocoManiLab in place (kept, every split) [95 %]; mean direct error "
          "vector (heading frame, cm; left y mirrored) and the angle between them [95 %, xfer resampled]; model / direct on xfer:")  # fmt: skip
    for name, _ in SPLITS:
        o = out["splits"][name]
        if not o.get("n"):
            continue
        ci, aci = o["direct_ratio_ci"], o["angle_ci"]
        print(f"    {name:12s}: direct {o['direct_xfer']:.2f} / {o['direct_lml_inplace']:.2f} = {o['direct_ratio']:.3f}x [{ci[0]:.3f}, "
              f"{ci[1]:.3f}]; mean vector xfer {np.round(o['mean_err_xfer_cm'], 2).tolist()} (|{np.linalg.norm(o['mean_err_xfer_cm']):.2f}|), "
              f"LML {np.round(o['mean_err_lml_cm'], 2).tolist()} (|{np.linalg.norm(o['mean_err_lml_cm']):.2f}|), angle {o['angle_deg']:.1f} deg "
              f"[{aci[0]:.1f}, {aci[1]:.1f}]; model / direct {o['model_over_direct_xfer']:.3f}")  # fmt: skip
    # training range (hg_run's rule on train3's training segments)
    tr = sp["train"]
    k0 = lambda D, idx: np.asarray(D["so"])[idx] + np.asarray(D["k0"])[idx].astype(int)  # noqa: E731
    qtr = np.asarray(DL["S"])[k0(DL, tr), :29]
    lo, hi = np.percentile(qtr, 0.5, axis=0), np.percentile(qtr, 99.5, axis=0)
    qx, qv = np.asarray(DX["S"])[k0(DX, xi), :29], np.asarray(DL["S"])[k0(DL, vi), :29]
    outx = ((qx < lo) | (qx > hi)).sum(1)
    outv = ((qv < lo) | (qv > hi)).sum(1)
    inx = outx == 0
    worst = np.argsort(-((qx < lo) | (qx > hi)).mean(0))[:5]
    sys.path.insert(0, str(REPO / "scripts/experiments/locomanilab_20261117"))
    from lml_selfmodel import SCALEBFM_ORDER as ORDER  # noqa: E402  (S's joint order)

    out["in_range"] = dict(xfer=float(inx.mean()), val=float((outv == 0).mean()), xfer_median_joints_out=float(np.median(outx)),
                           err_in=float(np.median(PX["err"][inx])) if inx.any() else None,
                           err_out=float(np.median(PX["err"][~inx])) if (~inx).any() else None,
                           worst=[(ORDER[j], float(((qx[:, j] < lo[j]) | (qx[:, j] > hi[j])).mean())) for j in worst])  # fmt: skip
    ci = boot(groups_of(rx), lambda s: inx[s].mean(), rng)
    ir = out["in_range"]
    print(f"  training range (29 joints at the command row inside train3's 0.5 / 99.5 %): xfer {100 * ir['xfer']:.1f} % "
          f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}] (val {100 * ir['val']:.1f} %), joints out median {ir['xfer_median_joints_out']:.0f}; "
          f"prediction error inside {ir['err_in'] if ir['err_in'] is None else round(ir['err_in'], 3)} cm (n {int(inx.sum())}), outside "
          f"{ir['err_out'] if ir['err_out'] is None else round(ir['err_out'], 3)} cm (n {int((~inx).sum())}); most often out: "
          f"{[(n, round(100 * v, 1)) for n, v in ir['worst']]} (%)")  # fmt: skip
    # ensemble disagreement
    ci = boot(groups_of(rx), lambda s: np.median(PX["sd"][s]), rng)
    out["disagreement"] = dict(xfer=float(np.median(PX["sd"])), xfer_ci=ci, val=float(np.median(PV["sd"])),
                               ratio=float(np.median(PX["sd"]) / np.median(PV["sd"])))  # fmt: skip
    dg = out["disagreement"]
    print(f"  ensemble disagreement (|member std| of the predicted hand point, median cm): xfer {dg['xfer']:.3f} [{ci[0]:.3f}, "
          f"{ci[1]:.3f}], val {dg['val']:.3f}, ratio {dg['ratio']:.2f}x")
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
