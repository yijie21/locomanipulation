"""Old vs retrained reach world models on held-out reaches (2026-11-02, wide wrist commands in the training data).

old = models_side/ (plain) or models_goalref_side/ (plan-aware), trained on data_ref + data_side; new = models_wrist/ or
models_goalref_wrist/, trained on those plus data_wrist (gen_reach.py REACH_WRIST=1: the short-step reaches with the
wrist command turned a further 0-90 deg about a random axis, roll included). Both are scored on the test split (reset
hash bucket 9, outside the held-out bands) of each dataset: final jaw position error and ensemble std as in
side_eval.py, plus the final wrist rotation: the angle between the predicted and the delivered rotation, the members'
mean angle to their mean rotation (spread) and how well it ranks the angle error, against "the hand ends at the
commanded rotation". The wide-wrist reaches also by how far the command was turned. REACHWM_GOAL_REF picks the pair.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 $PY scripts/experiments/reachwm_20261030/wrist_eval.py
    REACHWM_GOAL_REF=1 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 $PY scripts/experiments/reachwm_20261030/wrist_eval.py
Writes outputs/reachwm_20261030/wrist_eval/{plain,ref}.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as DT  # noqa: E402
import models as M  # noqa: E402
import train as TR  # noqa: E402

REF, SIDE, WRIST = (DT.OUT / f"data_{k}" / "all.npz" for k in ("ref", "side", "wrist"))
SIDE_ID0, WRIST_ID0 = 100000, 200000  # gen_reach.py REACH_ID0 of data_side and data_wrist
OUT = DT.OUT / "wrist_eval"


def sets(D: dict) -> dict[str, np.ndarray]:
    sp = DT.splits(D)
    test = np.zeros(len(D["n_steps"]), bool)
    test[sp["test_iid"]] = True
    test &= D["fell"] == 0
    side = (D["reset"] >= SIDE_ID0) & (D["reset"] < WRIST_ID0)
    wrist = D["reset"] >= WRIST_ID0
    t = D["target"] * np.c_[np.ones(len(test)), D["side"], np.ones(len(test))]  # y mirrored to the right hand
    core = np.all((t >= [0.15, -0.40, 0.15]) & (t <= [0.70, 0.15, 1.35]), axis=1)
    a = np.degrees(np.linalg.norm(D["wrist_dev"], axis=1))
    out = {"old data, all": test & ~side & ~wrist, "short step, all": test & side, "short step, core targets": test & side & core,
           "wide wrist, all": test & wrist, "wide wrist, core targets": test & wrist & core,
           "wide wrist, turned < 30 deg": test & wrist & (a < 30), "wide wrist, turned 30-60 deg": test & wrist & (a >= 30) & (a < 60),
           "wide wrist, turned >= 60 deg": test & wrist & (a >= 60)}  # fmt: skip
    return {k: np.where(v)[0] for k, v in out.items()}


def score(model, GD, idx: np.ndarray) -> dict:
    ti = torch.as_tensor(idx, device=GD["x0"].device)
    p = M.predict_batched(model, GD, ti)
    err = (100 * (p["pos"] - GD["pos_f"][ti]).norm(dim=-1)).cpu().numpy()
    std = (100 * p["pos_std"].norm(dim=-1)).cpu().numpy()
    Rf, Rp = M.gs6(GD["rot_f"][ti]), M.gs6(p["rot6"])
    ang = M.rot_angle_deg(Rp, Rf).cpu().numpy()
    cmd = M.rot_angle_deg(M.gs6(GD["rot6"][ti]), Rf).cpu().numpy()
    Rk = torch.stack([M.gs6(M.predict_batched(mm, GD, ti)["rot6"]) for mm in model.members])
    spread = torch.stack([M.rot_angle_deg(Rk[k], Rp) for k in range(len(Rk))]).mean(0).cpu().numpy()
    return dict(n=len(idx), median=float(np.median(err)), p90=float(np.percentile(err, 90)), gt2=float(np.mean(err > 2)),
                std=float(np.median(std)), rho=float(spearmanr(std, err).correlation), att=float(np.median(ang)),
                att_p90=float(np.percentile(ang, 90)), att_cmd=float(np.median(cmd)), spread=float(np.median(spread)),
                att_rho=float(spearmanr(spread, ang).correlation))  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrist", default=str(WRIST), help="data_wrist file or glob (smoke test: a few chunks)")
    ap.add_argument("--new-root", default=None, help="default models_wrist/ or models_goalref_wrist/")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    kind = "ref" if DT.GOAL_REF else "plain"
    roots = dict(old=DT.OUT / ("models_goalref_side" if DT.GOAL_REF else "models_side"),
                 new=Path(a.new_root or DT.OUT / ("models_goalref_wrist" if DT.GOAL_REF else "models_wrist")))  # fmt: skip
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D0 = DT.load(f"{REF},{SIDE}")  # the old models' normalization comes from their own training split
    models = dict(old=TR.load_model("E3_ensemble", 1.0, D0, dev, roots["old"]))
    del D0
    D = DT.load(f"{REF},{SIDE},{a.wrist}")
    W = DT.load(a.wrist)  # the per-segment wrist turns (only data_wrist has them)
    wrist = D["reset"] >= WRIST_ID0
    assert np.array_equal(D["reset"][wrist], W["reset"])
    D["wrist_dev"] = np.zeros((len(wrist), 3), np.float32)
    D["wrist_dev"][wrist] = W["wrist_dev"]
    del W
    models["new"] = TR.load_model("E3_ensemble", 1.0, D, dev, roots["new"])
    GD = M.tensors(D, dev)
    res = {nm: {k: score(models[k], GD, idx) for k in models} for nm, idx in sets(D).items()}
    L = [f"{kind} E3 ensemble, test split; old = {roots['old'].name}, new = {roots['new'].name} ({len(D['n_steps'])} segments, "
         f"{int(wrist.sum())} with a wide wrist command)", "",
         "| reaches | n | position median cm old / new | p90 old / new | >2 cm old / new | std/error Spearman old / new | "
         "wrist median deg old / new | wrist p90 old / new | hand at the command deg | wrist spread median old / new | "
         "spread/error Spearman old / new |", "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for nm, r in res.items():
        o, n = r["old"], r["new"]
        L.append(f"| {nm} | {o['n']} | {o['median']:.2f} / {n['median']:.2f} | {o['p90']:.2f} / {n['p90']:.2f} | "
                 f"{o['gt2']:.2f} / {n['gt2']:.2f} | {o['rho']:.2f} / {n['rho']:.2f} | {o['att']:.1f} / {n['att']:.1f} | "
                 f"{o['att_p90']:.1f} / {n['att_p90']:.1f} | {o['att_cmd']:.1f} | {o['spread']:.1f} / {n['spread']:.1f} | "
                 f"{o['att_rho']:.2f} / {n['att_rho']:.2f} |")  # fmt: skip
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / f"{kind}.json", "w"), indent=1)
    (out / f"{kind}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
