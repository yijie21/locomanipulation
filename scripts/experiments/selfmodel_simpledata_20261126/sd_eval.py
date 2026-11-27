"""selfmodel_simpledata_20261126, evaluation 1: model D (seed 0, frac 1.0) and its learning-curve twin D_f0.3 on SIMPLE's own
held-out data (data.py's splits of this experiment's data3 and data3b), read-only.

    cd <repo>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1 LOCO_REPO=<repo> \
        $PY sd_eval.py --models <out>/models_D_s0 --data3 <all.npz> --data3b <all.npz> [--json F]

Segments: data.py's split (val, test_iid, band_x, band_z, band_yaw, band_pel, edge; crc32 of the reset id), kept when they
did not fall or get lost and the end label is settled (train3b's t3b_eval.kept). Groups: all, first (chain 0) and
chained (chain >= 1), as t3b_eval. Cells: data3 all, and on first reaches in place / walk first (task walk) / walk first
(d3 walk); data3b all and by kind (entry, pre-grasp, relative).
Per cell: position error (cm, the predicted final hand point to the real one; direct: data.py err_cm, the hand left at
the command) median and p90; attitude (deg) median; the learning curve D / D_f0.3 (ratio of medians). 95 % intervals:
1000 resamples of the cell's resets, every statistic on the same resamples (so the ratio is paired).
The registration's numbers: "SIMPLE val" = data3 val, all reaches, D's position median; the curve = the same cell's
D / D_f0.3 (test_iid's printed beside it).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(os.environ.get("LOCO_REPO", "/wuji-vefps-D/wuji-il/yijie/ws/code/loco"))
sys.path[:0] = [str(REPO / "scripts/experiments/reachwm_20261030"), str(REPO / "scripts/experiments/selfwm_20261029")]
import data as DT  # noqa: E402
import train as TR  # noqa: E402

for k in ("REACHWM_GOAL_REF", "REACHWM_APPROACH", "REACHWM_QFIX", "REACHWM_FDVEL"):
    assert os.environ.get(k) == "1", f"{k}=1 is D's switch here"
M = TR.M
NB = 1000
SPLITS = ("val", "test_iid", "band_x", "band_z", "band_yaw", "band_pel", "edge")
KIND = {"pre": 1, "entry": 2, "rel": 3}
A3 = ("all", "first", "chained")


def kept(D: dict, sp: np.ndarray) -> np.ndarray:  # t3b_eval.kept
    se = np.asarray(D["settled_end"]) > 0 if "settled_end" in D else np.ones(len(D["n_steps"]), bool)
    return sp[(np.asarray(D["fell"])[sp] == 0) & se[sp]]


def predict(models, D: dict, idx: np.ndarray, dev) -> tuple[dict, dict]:  # t3b_eval.predict, direct as "direct"
    GD = M.tensors(D, dev)
    ti = torch.tensor(idx, device=dev)
    rot_f = M.gs6(GD["rot_f"][ti])
    pos = {"direct": np.asarray(D["err_cm"])[idx].astype(float)}
    att = {"direct": M.rot_angle_deg(M.gs6(GD["rot6"][ti]), rot_f).cpu().numpy()}
    for name, root, frac in models:
        m = TR.load_model("E3_ensemble", frac, D, dev, root=root)
        p = M.predict_batched(m, GD, ti)
        pos[name] = (100 * (p["pos"] - GD["pos_f"][ti]).norm(dim=-1)).cpu().numpy()
        att[name] = M.rot_angle_deg(M.gs6(p["rot6"]), rot_f).cpu().numpy()
        del m
    return pos, att


def stats(pos: dict, att: dict, s: np.ndarray) -> dict:
    o = {}
    for k in pos:
        o[k], o["p90_" + k], o["att_" + k] = float(np.median(pos[k][s])), float(np.percentile(pos[k][s], 90)), float(np.median(att[k][s]))
    if "D" in pos and "D_f0.3" in pos:
        o["curve"] = o["D"] / o["D_f0.3"]
    return o


def cell(pos, att, rs: np.ndarray, sel: np.ndarray, rng) -> dict:
    idx = np.flatnonzero(sel)
    r = rs[idx]
    o_ = np.argsort(r, kind="stable")
    groups = np.split(idx[o_], np.flatnonzero(np.diff(r[o_])) + 1)
    point = stats(pos, att, idx)
    boots = [stats(pos, att, np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])) for _ in range(NB)]
    ci = {k: (float(np.percentile([b[k] for b in boots], 2.5)), float(np.percentile([b[k] for b in boots], 97.5))) for k in point}
    return dict(n=int(len(idx)), n_resets=len(groups), point=point, ci=ci)


def section(title, D, idx, cells, models, dev, rng) -> dict:
    pos, att = predict(models, D, idx, dev)
    rs = np.asarray(D["reset"])[idx]
    ch = np.asarray(D["chain"])[idx]
    grp = {"all": np.ones(len(idx), bool), "first": ch == 0, "chained": ch >= 1}
    out = dict(title=title, n=int(len(idx)), n_resets=int(len(np.unique(rs))), cells=[])
    print(f"\n== {title}: {len(idx)} kept segments from {out['n_resets']} resets")
    names = list(pos)
    for g in A3:
        for dim, lab, sel, gs in cells:
            if g not in gs:
                continue
            s = sel & grp[g]
            if s.sum() == 0:
                print(f"  {g} | {dim}: {lab}: no segments")
                continue
            c = cell(pos, att, rs, s, rng)
            c.update(group=g, dim=dim, cell=lab)
            out["cells"].append(c)
            p, ci = c["point"], c["ci"]
            f = lambda k, d=2: f"{p[k]:.{d}f} [{ci[k][0]:.{d}f}, {ci[k][1]:.{d}f}]"  # noqa: E731
            print(f"  {g} | {dim}: {lab}  n {c['n']} ({c['n_resets']} resets)")
            print("    position cm, median: " + " | ".join(f"{k} " + f(k) for k in names))
            print("    position cm, p90:    " + " | ".join(f"{k} " + f("p90_" + k) for k in names))
            print("    attitude deg, median: " + " | ".join(f"{k} " + f("att_" + k) for k in names))
            if "curve" in p:
                print(f"    learning curve D / D_f0.3: {f('curve', 3)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True, help="<out>/models_D_s0 (E3_ensemble_f1 and _f0.3 inside)")
    ap.add_argument("--data3", type=Path, required=True)
    ap.add_argument("--data3b", type=Path, required=True)
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    rng = np.random.default_rng(0)
    models = [(n, a.models, f) for n, f in (("D", 1.0), ("D_f0.3", 0.3)) if (TR.model_dir("E3_ensemble", f, a.models) / "model.pt").exists()]
    print(f"models {[(n, str(TR.model_dir('E3_ensemble', f, r))) for n, r, f in models]}; device {dev}; FDVEL {DT.FDVEL}")
    out = dict(models=[(n, str(r), f) for n, r, f in models], sections=[], headline={})
    for name, path in (("data3", a.data3), ("data3b", a.data3b)):
        D = DT.load(str(path))
        sp = DT.splits(D)
        for split in a.splits.split(","):
            idx = kept(D, sp[split])
            if len(idx) == 0:
                print(f"\n== {name} {split}: no kept segments")
                continue
            g = lambda k: np.asarray(D[k])[idx]  # noqa: E731
            cells = [("all", "all", np.ones(len(idx), bool), A3)]
            if name == "data3":
                wt = g("walk_task") if "walk_task" in D else np.zeros(len(idx))
                cells += [("start", "in place", g("walked") == 0, ("first",)),
                          ("start", "walk first, task walk", (g("walked") == 1) & (wt == 1), ("first",)),
                          ("start", "walk first, d3 walk", (g("walked") == 1) & (wt == 0), ("first",))]  # fmt: skip
            else:
                kind = g("kind")
                cells += [("kind", "entry", kind == KIND["entry"], A3), ("kind", "pre-grasp", kind == KIND["pre"], A3),
                          ("kind", "relative", kind == KIND["rel"], A3)]  # fmt: skip
            sec = section(f"{name} {split}", D, idx, cells, models, dev, rng)
            out["sections"].append(sec)
            if split in ("val", "test_iid"):
                c = next(c for c in sec["cells"] if c["group"] == "all" and c["cell"] == "all")
                out["headline"][f"{name}_{split}"] = dict(D=c["point"].get("D"), D_ci=c["ci"].get("D"), curve=c["point"].get("curve"),
                                                          curve_ci=c["ci"].get("curve"), n=c["n"], n_resets=c["n_resets"])  # fmt: skip
        del D
    h = out["headline"].get("data3_val", {})
    t = out["headline"].get("data3_test_iid", {})
    if h.get("D") is not None:
        print(f"\nSIMPLE val (data3 val, all, kept): D {h['D']:.3f} cm [{h['D_ci'][0]:.3f}, {h['D_ci'][1]:.3f}] (n {h['n']}, "
              f"{h['n_resets']} resets); learning curve D / D_f0.3 {h.get('curve') or float('nan'):.3f} "
              f"{h.get('curve_ci')}; test_iid: D {t.get('D')}, curve {t.get('curve')}")  # fmt: skip
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
