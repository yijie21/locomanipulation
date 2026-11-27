"""selfmodel_hacollect_20261126 (development, 2026-11-26): start-state diversity of an HA record against LocoManiLab's
data3 (and data3b) for the same controller. Read only; CPU.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python   (cwd = the 4090's repo)
    REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1 $PY hc_start.py --ha DIR --lml D3DIR [--lml-b D3BDIR]

Per group (first reach in place / first reach after a walk / later reaches), at the command row k0 of each segment:
  distinct starts (joint vectors rounded to 1e-4 rad), per-joint std (median and max over the 29 joints), pelvis height
  mean / std, tilt (gravity's x / y in the pelvis frame, deg) std, the 0.1 s difference speed of the joints (median |.|);
and the share of HA's command rows whose joints lie outside LML's own 0.5-99.5 % range of the same group, per joint
(median over joints) and for any joint. Then data.py's x0 (REACHWM_FDVEL=1, the features the HA models will train on):
the share of HA segments with any x0 input beyond 6 SD of LML's training split (fdvel's OOD measure), by block.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("SX_REPO", "/wuji-vefps-D/wuji-il/yijie/ws/code/loco"))
# the reader: selfmodel_fdvel_20261126's frozen data.py (174071cb, REACHWM_FDVEL); the 4090 repo's copy predates it
FD_REPO = Path(os.environ.get("HC_FD_REPO", "/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_fdvel_20261126/asrun/repo"))


def rows_at(z, sel, back=0):
    n = z["n_steps"].astype(int)
    so = np.r_[0, np.cumsum(n)[:-1]]
    return so[sel] + z["k0"].astype(int)[sel] - back


def groups(z):
    ch, wk = z["chain"].astype(int), z["walked"].astype(int)
    return {"first in place": (ch == 0) & (wk == 0), "first after walk": (ch == 0) & (wk == 1), "later": ch > 0}


def stats(z, sel):
    r = rows_at(z, np.where(sel)[0])
    r1 = rows_at(z, np.where(sel)[0], 1)
    S = z["S"]
    q, g, hz = S[r, :29].astype(float), S[r, 58:61].astype(float), S[r, 64].astype(float)
    v = (q - S[r1, :29].astype(float)) / 0.1
    return dict(n=len(r), uniq=len(np.unique(np.round(q, 4), axis=0)), q=q, qs=q.std(0) if len(r) > 1 else np.zeros(29),
                hz=hz, tilt=np.degrees(g[:, :2]), v=v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ha", type=Path, nargs="+", required=True)
    ap.add_argument("--lml", type=Path, required=True)
    ap.add_argument("--lml-b", type=Path, default=None)
    ap.add_argument("--no-x0", action="store_true")
    ap.add_argument("--x0-only", action="store_true")
    a = ap.parse_args()
    lml = dict(np.load(a.lml / "data" / "all.npz"))
    for hd in ([] if a.x0_only else a.ha):
        z = dict(np.load(hd / "data" / "all.npz"))
        print(f"== {hd} ({len(z['n_steps'])} segments, {len(np.unique(z['reset']))} resets) vs {a.lml}")
        gl, gh = groups(lml), groups(z)
        for k in gl:
            if not gh[k].any():
                print(f"  {k}: no HA segments")
                continue
            L_, H_ = stats(lml, gl[k]), stats(z, gh[k])
            lo, hi = np.percentile(L_["q"], 0.5, axis=0), np.percentile(L_["q"], 99.5, axis=0)
            out = (H_["q"] < lo) | (H_["q"] > hi)
            for nm, s in (("LML", L_), ("HA ", H_)):
                print(f"  {k:16s} {nm} n {s['n']:6d} distinct {s['uniq']:6d} | joint std median {np.median(s['qs']):.4f} max "
                      f"{s['qs'].max():.4f} rad | pelvis z {s['hz'].mean():.3f} +- {100 * s['hz'].std():.2f} cm | tilt std "
                      f"{np.round(s['tilt'].std(0), 2).tolist()} deg | joint speed (0.1 s diff) median {np.median(np.abs(s['v'])):.4f}, "
                      f"p90 {np.percentile(np.abs(s['v']), 90):.4f} rad/s")
            print(f"  {k:16s} HA rows outside LML's 0.5-99.5 % per joint: median over joints {100 * np.median(out.mean(0)):.1f} %, "
                  f"any joint {100 * out.any(1).mean():.1f} %; joints most often outside (S index: %) "
                  f"{[(int(j), round(100 * float(out[:, j].mean()), 1)) for j in np.argsort(-out.mean(0))[:5]]}")
            d = (H_["q"].mean(0) - L_["q"].mean(0)) / np.maximum(L_["qs"], 1e-4)
            dr = np.abs(H_["q"].mean(0) - L_["q"].mean(0))
            print(f"  {k:16s} HA mean - LML mean in LML sd: median |.| {np.median(np.abs(d)):.2f}, max {np.abs(d).max():.2f} (joint {int(np.argmax(np.abs(d)))}); "
                  f"in rad median {np.median(dr):.4f}, max {dr.max():.4f} (joint {int(np.argmax(dr))}); pelvis z {100 * (H_['hz'].mean() - L_['hz'].mean()):+.2f} cm")
    if a.no_x0:
        return
    sys.path[:0] = [str(FD_REPO / "scripts/experiments/reachwm_20261030"), str(FD_REPO / "src")]
    import data as DT

    assert DT.FDVEL, "run with REACHWM_FDVEL=1 (and GOAL_REF / APPROACH / QFIX as T3 / F)"
    specs = [str(a.lml / "data" / "all.npz")] + ([str(a.lml_b / "data" / "all.npz")] if a.lml_b else [])
    DL = DT.load(specs)
    tr = DT.splits(DL)["train"]
    X = DL["x0"][tr].astype(np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    vel = np.zeros(X.shape[1], bool)
    vel[DT.X0_DQ_COLS] = True
    vel[DT.X0_ANGVEL_COLS] = True
    for hd in a.ha:
        DH = DT.load([str(hd / "data" / "all.npz")])
        Z = np.abs((DH["x0"].astype(np.float64) - mu) / sd)
        fin = np.isfinite(DH["x0"]).all()
        bad = Z > 6
        g = groups(DH)
        print(f"== x0 (FDVEL) {hd}: {len(Z)} segments, {Z.shape[1]} inputs, finite {fin}; LML train {len(X)} segments ({len(specs)} sets)")
        print(f"  any input beyond 6 SD: {100 * bad.any(1).mean():.1f} % (velocity block {100 * bad[:, vel].any(1).mean():.1f} %, "
              f"other x0 {100 * bad[:, ~vel].any(1).mean():.1f} %); by group "
              f"{ {k: round(100 * float(bad[m].any(1).mean()), 1) for k, m in g.items() if m.any()} }")
        top = np.argsort(-bad.mean(0))[:6]
        print(f"  inputs most often beyond: {[(int(c), round(100 * float(bad[:, c].mean()), 1)) for c in top]}")


if __name__ == "__main__":
    main()
