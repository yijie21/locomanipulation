"""Old vs retrained reach world models on held-out reaches (2026-11-01, short steps in the training data).

old = models/ (plain) or models_goalref/ (plan-aware), trained on data_ref's 40k reaches; new = models_side/ or
models_goalref_side/, trained on data_ref + data_side (gen_reach.py REACH_SIDESTEP=1: reaches that start right after a
short heading-kept step). Both are scored on the test split (reset hash bucket 9, outside the held-out bands) of each
dataset: final jaw position error, ensemble std, and how well the std ranks the error (Spearman); the short-step reaches
also by step size. REACHWM_GOAL_REF picks the pair.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 $PY scripts/experiments/reachwm_20261030/side_eval.py
    REACHWM_GOAL_REF=1 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 $PY scripts/experiments/reachwm_20261030/side_eval.py
Writes outputs/reachwm_20261030/side_eval/{plain,ref}.{json,md}.
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

REF, SIDE = DT.OUT / "data_ref" / "all.npz", DT.OUT / "data_side" / "all.npz"
SIDE_ID0 = 100000  # gen_reach.py REACH_ID0 of data_side
OUT = DT.OUT / "side_eval"


def sets(D: dict) -> dict[str, np.ndarray]:
    sp = DT.splits(D)
    test = np.zeros(len(D["n_steps"]), bool)
    test[sp["test_iid"]] = True
    test &= D["fell"] == 0
    side = D["reset"] >= SIDE_ID0
    t = D["target"] * np.c_[np.ones(len(test)), D["side"], np.ones(len(test))]  # y mirrored to the right hand
    core = np.all((t >= [0.15, -0.40, 0.15]) & (t <= [0.70, 0.15, 1.35]), axis=1)
    still = (D["walked"] == 0) & (D["chain"] == 0)
    out = {"old data, all": test & ~side, "old data, first reach from the reset stand": test & ~side & still,
           "old data, after a 0.3-1.2 m walk": test & ~side & (D["walked"] == 1) & (D["chain"] == 0),
           "short step, all": test & side, "short step, core targets": test & side & core}  # fmt: skip
    if "sidestep" in D:
        a = 100 * np.linalg.norm(D["sidestep"], axis=1)
        out.update({"short step, |step| < 10 cm": test & side & (a < 10),
                    "short step, |step| 10-20 cm": test & side & (a >= 10) & (a < 20),
                    "short step, |step| >= 20 cm": test & side & (a >= 20),
                    "short step, two steps": test & side & (D["n_sidesteps"] == 2)})  # fmt: skip
    return {k: np.where(v)[0] for k, v in out.items()}


def score(model, GD, idx: np.ndarray) -> dict:
    p = M.predict_batched(model, GD, torch.as_tensor(idx, device=GD["x0"].device))
    err = (100 * (p["pos"] - GD["pos_f"][idx]).norm(dim=-1)).cpu().numpy()
    std = (100 * p["pos_std"].norm(dim=-1)).cpu().numpy()
    return dict(n=len(idx), median=float(np.median(err)), p90=float(np.percentile(err, 90)), gt2=float(np.mean(err > 2)),
                std=float(np.median(std)), rho=float(spearmanr(std, err).correlation))  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default=str(SIDE), help="data_side file or glob (smoke test: a few chunks)")
    ap.add_argument("--new-root", default=None, help="default models_side/ or models_goalref_side/")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    kind = "ref" if DT.GOAL_REF else "plain"
    roots = dict(old=DT.OUT / ("models_goalref" if DT.GOAL_REF else "models"),
                 new=Path(a.new_root or DT.OUT / ("models_goalref_side" if DT.GOAL_REF else "models_side")))  # fmt: skip
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D0 = DT.load(str(REF))  # the old models' normalization comes from their own training split
    models = dict(old=TR.load_model("E3_ensemble", 1.0, D0, dev, roots["old"]))
    del D0
    D = DT.load(f"{REF},{a.side}")
    S = DT.load(a.side)  # the per-segment step sizes (only data_side has them)
    side = D["reset"] >= SIDE_ID0
    assert np.array_equal(D["reset"][side], S["reset"])
    D["sidestep"] = np.zeros((len(side), 2), np.float32)
    D["n_sidesteps"] = np.zeros(len(side), int)
    D["sidestep"][side], D["n_sidesteps"][side] = S["sidestep"], S["n_sidesteps"]
    del S
    models["new"] = TR.load_model("E3_ensemble", 1.0, D, dev, roots["new"])
    GD = M.tensors(D, dev)
    res = {nm: {k: score(models[k], GD, idx) for k in models} for nm, idx in sets(D).items()}
    L = [f"{kind} E3 ensemble, test split; old = {roots['old'].name}, new = {roots['new'].name} ({len(D['n_steps'])} segments, "
         f"{int(side.sum())} with a short step)", "",
         "| reaches | n | median cm old / new | p90 old / new | >2 cm old / new | std median old / new | std/error Spearman old / new |",
         "|---|---|---|---|---|---|---|"]  # fmt: skip
    for nm, r in res.items():
        o, n = r["old"], r["new"]
        L.append(f"| {nm} | {o['n']} | {o['median']:.2f} / {n['median']:.2f} | {o['p90']:.2f} / {n['p90']:.2f} | "
                 f"{o['gt2']:.2f} / {n['gt2']:.2f} | {o['std']:.2f} / {n['std']:.2f} | {o['rho']:.2f} / {n['rho']:.2f} |")  # fmt: skip
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / f"{kind}.json", "w"), indent=1)
    (out / f"{kind}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
