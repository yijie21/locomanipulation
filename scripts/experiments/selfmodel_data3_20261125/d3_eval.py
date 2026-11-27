"""selfmodel_train3_20261125's metrics: the trained end models (reachwm_20261030 train.py E3_ensemble, seed 0) on one
controller's selfmodel_data3 one-hand segments, overall and split; read-only (data and models).

    cd $REPO && REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_ROOT=$O CUDA_VISIBLE_DEVICES=1 \
        $VPY .../d3_eval.py $O $O/models_s0 --split val [--json $O/eval_val.json]

Segments: the split's (data.py's splits: crc32 of the reset id; val / test_iid), kept when they did not fall or get lost
and the end label is settled (settled_end, the collector's settle rule).
Per cell, medians over its segments:
  position (cm)  model: the predicted final hand point to the real one (M.pos_error_cm); direct: the hand left at the
                 command (B0_command: target to the real hand point, data.py err_cm); ratio = model / direct.
  attitude (deg) geodesic angle between the predicted final wrist attitude (the model's rot6, Gram-Schmidt) and the real
                 one (data.py rot_f); direct: the commanded rot6 instead of the prediction. E3 predicts rot6 as a
                 learned residual rotation on the command (models.EndNet "residual", trained by pose_loss).
  95 % intervals from 1000 resamples of the cell's resets (all of a reset's segments together).
Cells: in place / walk first, right / left hand, target height z < 0.3 / 0.3-0.6 / 0.6-0.9 / 0.9-1.2 / >= 1.2 m,
|wrist yaw| <= 60 / > 60 deg, |roll| <= 40 / > 40, pitch < 0 / >= 0, standing / crouched (a pelvis height asked), and
reach duration <= 2.5 / > 2.5 s (the entry asks it of HEFT; printed for every controller). Learning curve: each
training fraction's overall medians on the same segments, and frac 1.0 / 0.3 with its interval.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path.cwd()
sys.path[:0] = [str(REPO / "scripts/experiments/reachwm_20261030"), str(REPO / "scripts/experiments/selfwm_20261029")]
import data as DT  # noqa: E402
import train as TR  # noqa: E402

M = TR.M
NB = 1000


def boot(groups: list[np.ndarray], stat, rng) -> tuple[float, float]:
    """95 % interval of stat(indices) over resamples of the groups (resets)."""
    n = len(groups)
    vals = [stat(np.concatenate([groups[j] for j in rng.integers(0, n, n)])) for _ in range(NB)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("models", type=Path)
    ap.add_argument("--split", default="val")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)
    D = DT.load(str(a.out / "data" / "all.npz"))
    GD = M.tensors(D, dev)
    sp = DT.splits(D)[a.split]
    keep = (np.asarray(D["fell"])[sp] == 0) & (np.asarray(D["settled_end"])[sp] > 0)
    idx = sp[keep]
    ti = torch.tensor(idx, device=dev)
    g = lambda k: np.asarray(D[k])[idx]  # noqa: E731
    rot_f = M.gs6(GD["rot_f"][ti])
    pos_d = g("err_cm").astype(float)
    att_d = M.rot_angle_deg(M.gs6(GD["rot6"][ti]), rot_f).cpu().numpy()
    pos_m, att_m = {}, {}
    for frac in DT.FRACS:
        if not (TR.model_dir("E3_ensemble", frac, a.models) / "model.pt").exists():
            continue
        m = TR.load_model("E3_ensemble", frac, D, dev, root=a.models)
        p = M.predict_batched(m, GD, ti)
        pos_m[frac] = (100 * (p["pos"] - GD["pos_f"][ti]).norm(dim=-1)).cpu().numpy()
        att_m[frac] = M.rot_angle_deg(M.gs6(p["rot6"]), rot_f).cpu().numpy()
    rs = g("reset")
    out = dict(controller=a.out.name, split=a.split, n_split=int(len(sp)), n=int(len(idx)), n_resets=int(len(np.unique(rs))),
               dropped_fell=int((np.asarray(D["fell"])[sp] != 0).sum()), dropped_unsettled=int((~keep).sum() - (np.asarray(D["fell"])[sp] != 0).sum()),
               att_direct_check=float(np.median(g("att_err_deg"))), curve={}, cells=[])  # fmt: skip
    print(f"{a.out.name} {a.split}: {len(idx)} segments from {out['n_resets']} resets (of {len(sp)}: {out['dropped_fell']} fell / lost, "
          f"{out['dropped_unsettled']} not settled at the label); direct attitude check: recorded att_err_deg median "
          f"{out['att_direct_check']:.2f} vs recomputed {np.median(att_d):.2f} deg")  # fmt: skip

    def groups_of(sel: np.ndarray) -> list[np.ndarray]:
        r = rs[sel]
        o = np.argsort(r, kind="stable")
        cut = np.flatnonzero(np.diff(r[o])) + 1
        return np.split(np.flatnonzero(sel)[o], cut)

    allsel = np.ones(len(idx), bool)
    G = groups_of(allsel)
    print("  learning curve (all kept segments):")
    for frac in pos_m:
        pm, am = pos_m[frac], att_m[frac]
        c = dict(pos=float(np.median(pm)), pos_ci=boot(G, lambda s: np.median(pm[s]), rng),
                 att=float(np.median(am)), att_ci=boot(G, lambda s: np.median(am[s]), rng))  # fmt: skip
        out["curve"][f"{frac:g}"] = c
        print(f"    frac {frac:g}: position {c['pos']:.3f} cm [{c['pos_ci'][0]:.3f}, {c['pos_ci'][1]:.3f}], "
              f"attitude {c['att']:.2f} deg [{c['att_ci'][0]:.2f}, {c['att_ci'][1]:.2f}]")  # fmt: skip
    if 0.3 in pos_m and 1.0 in pos_m:
        a3, a1 = pos_m[0.3], pos_m[1.0]
        r = float(np.median(a1) / np.median(a3))
        ci = boot(G, lambda s: np.median(a1[s]) / np.median(a3[s]), rng)
        out["curve"]["ratio_1_over_0.3"] = dict(pos=r, pos_ci=ci)
        print(f"    frac 1.0 / 0.3 position: {r:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    if 1.0 not in pos_m:
        print("  no frac 1.0 model: no cells")
        return
    pm, am = pos_m[1.0], att_m[1.0]
    t, pz = g("target"), g("pelvis_z")
    dyaw, pitch, roll = np.abs(np.degrees(g("dyaw"))), np.degrees(g("pitch")), np.abs(np.degrees(g("roll")))
    cells = [
        ("all", "all", allsel),
        ("start", "in place", g("walked") == 0), ("start", "walk first", g("walked") == 1),
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
    tot = float(np.median(pm))
    print(f"  cells (frac 1.0 model; x all = the cell's model median over the overall {tot:.3f} cm):")
    print(f"    {'cell':22s} {'n':>6s} {'resets':>6s} | {'model cm':>22s} {'x all':>5s} | {'direct cm':>21s} | {'ratio':>19s} | "
          f"{'model deg':>20s} | {'direct deg':>20s}")  # fmt: skip
    for dim, lab, sel in cells:
        if sel.sum() == 0:
            continue
        Gs = groups_of(sel)
        pmc, pdc, amc, adc = pm[sel], pos_d[sel], am[sel], att_d[sel]
        loc = {k: v for k, v in zip(("pm", "pd", "am", "ad"), (pm, pos_d, am, att_d))}
        c = dict(dim=dim, cell=lab, n=int(sel.sum()), n_resets=len(Gs),
                 pos=float(np.median(pmc)), pos_ci=boot(Gs, lambda s: np.median(loc["pm"][s]), rng),
                 direct=float(np.median(pdc)), direct_ci=boot(Gs, lambda s: np.median(loc["pd"][s]), rng),
                 ratio=float(np.median(pmc) / np.median(pdc)), ratio_ci=boot(Gs, lambda s: np.median(loc["pm"][s]) / np.median(loc["pd"][s]), rng),
                 att=float(np.median(amc)), att_ci=boot(Gs, lambda s: np.median(loc["am"][s]), rng),
                 att_direct=float(np.median(adc)), att_direct_ci=boot(Gs, lambda s: np.median(loc["ad"][s]), rng),
                 x_all=float(np.median(pmc) / tot))  # fmt: skip
        out["cells"].append(c)
        f2 = lambda v, ci, p=3: f"{v:.{p}f} [{ci[0]:.{p}f}, {ci[1]:.{p}f}]"  # noqa: E731
        print(f"    {dim + ': ' + lab:22s} {c['n']:6d} {c['n_resets']:6d} | {f2(c['pos'], c['pos_ci']):>22s} {c['x_all']:5.2f} | "
              f"{f2(c['direct'], c['direct_ci'], 2):>21s} | {f2(c['ratio'], c['ratio_ci']):>19s} | "
              f"{f2(c['att'], c['att_ci'], 2):>20s} | {f2(c['att_direct'], c['att_direct_ci'], 2):>20s}")  # fmt: skip
    w = [c for c in out["cells"] if c["dim"] == "start"]
    if len(w) == 2:
        r = w[1]["pos"] / w[0]["pos"]
        print(f"  walk first / in place (model position): {r:.3f}")
        out["walk_over_inplace"] = r
    h = [c for c in out["cells"] if c["dim"] == "hand"]
    if len(h) == 2:
        print(f"  left / right (model position): {h[1]['pos'] / h[0]['pos']:.3f}")
        out["left_over_right"] = h[1]["pos"] / h[0]["pos"]
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
