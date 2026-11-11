"""Learning curve of the wide-wrist data (2026-11-02, after wrist_invert.py's no-servo rules).

Open loop, the inversion's picks miss > 2 cm far more often the further the wrist command is turned (6-8 % up to 30 deg,
19 % at 30-60, 46 % at 60-90) while the model predicted ~0: the plan-aware model's own position error grows with the
turn. Does more wide-wrist data shrink it? The plan-aware E3 ensemble retrained with a quarter (4500) and a half (9000)
of data_wrist, next to none (models_goalref_side) and all of it (models_goalref_wrist), everything else equal (data_ref
+ data_side, train.py's fixed budget). All four are scored on the full data's test split (reset hash bucket 9, never
trained on), with wrist_eval.py's sets and metrics. "far" (2026-11-02) adds data_wrist2, 18k reaches turned 45-90 deg
only (gen_reach.py REACH_WRIST_MIN=45), on top of all of data_wrist; it is scored on the same test split, and on
data_wrist2's own test split in lc_far.md.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    REACHWM_GOAL_REF=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 $PY scripts/experiments/reachwm_20261030/lc_eval.py
Writes outputs/reachwm_20261030/wrist_eval/lc.{json,md}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as DT  # noqa: E402
import models as M  # noqa: E402
import train as TR  # noqa: E402
import wrist_eval as WE  # noqa: E402

W = DT.OUT / "data_wrist"
# wide-wrist data each model trained on (data_wrist chunks are w<worker>_<chunk>.npz, 500 reaches each), and its root
CURVE = {"none": ("", "models_goalref_side"), "quarter": (f"{W}/w?_000.npz,{W}/w[012]_001.npz", "models_goalref_wrist_q"),
         "half": (f"{W}/w?_00[012].npz", "models_goalref_wrist_h"), "full": (str(WE.WRIST), "models_goalref_wrist"),
         "far": (f"{WE.WRIST},{DT.OUT / 'data_wrist2' / 'all.npz'}", "models_goalref_wrist2")}  # fmt: skip
WRIST2_ID0 = 300000


def main() -> int:
    assert DT.GOAL_REF
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = f"{WE.REF},{WE.SIDE}"
    models, n_wrist = {}, {}
    for nm, (spec, root) in CURVE.items():
        D = DT.load(f"{base},{spec}" if spec else base)  # each model's normalization comes from its own training split
        n_wrist[nm] = int((D["reset"] >= WE.WRIST_ID0).sum())
        models[nm] = TR.load_model("E3_ensemble", 1.0, D, dev, DT.OUT / root)
        del D
    D = DT.load(f"{base},{WE.WRIST}")
    Wd = DT.load(str(WE.WRIST))
    wrist = D["reset"] >= WE.WRIST_ID0
    assert np.array_equal(D["reset"][wrist], Wd["reset"])
    D["wrist_dev"] = np.zeros((len(wrist), 3), np.float32)
    D["wrist_dev"][wrist] = Wd["wrist_dev"]
    del Wd
    GD = M.tensors(D, dev)
    res = {s: {nm: WE.score(models[nm], GD, idx) for nm in models} for s, idx in WE.sets(D).items()}
    del D, GD
    far = {}
    if "far" in models:  # data_wrist2's own test split: far-turn reaches only, by turn
        F = DT.load(f"{base},{CURVE['far'][0]}")
        Fw = DT.load(str(DT.OUT / "data_wrist2" / "all.npz"))
        k2 = F["reset"] >= WRIST2_ID0
        assert np.array_equal(F["reset"][k2], Fw["reset"])
        sp = DT.splits(F)
        test = np.zeros(len(k2), bool)
        test[sp["test_iid"]] = True
        test &= (F["fell"] == 0) & k2
        a = np.zeros(len(k2))
        a[k2] = np.degrees(np.linalg.norm(Fw["wrist_dev"], axis=1))
        GF = M.tensors(F, dev)
        sets2 = {"far turns, all": test, "far turns, 45-60 deg": test & (a < 60), "far turns, 60-75 deg": test & (a >= 60) & (a < 75),
                 "far turns, 75-90 deg": test & (a >= 75)}  # fmt: skip
        far = {s: {nm: WE.score(models[nm], GF, np.where(m)[0]) for nm in ("full", "far")} for s, m in sets2.items()}
    names = list(models)
    L = ["plan-aware E3 ensemble, test split of the full data; wide-wrist reaches trained on: "
         + ", ".join(f"{nm} {n_wrist[nm]}" for nm in names), ""]  # fmt: skip
    for key, title in (("median", "position median cm"), ("p90", "position p90 cm"), ("gt2", "position > 2 cm share"),
                       ("att", "wrist median deg"), ("std", "position std median cm")):  # fmt: skip
        L += [f"| {title} | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
        for s, r in res.items():
            L.append(f"| {s} | {r[names[0]]['n']} | " + " | ".join(f"{r[nm][key]:.2f}" for nm in names) + " |")
        L.append("")
    out = WE.OUT
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(n_wrist=n_wrist, res=res, far=far), open(out / "lc.json", "w"), indent=1)
    (out / "lc.md").write_text("\n".join(L) + "\n")
    if far:
        L2 = ["data_wrist2 test split (turned 45-90 deg); full = models_goalref_wrist, far = models_goalref_wrist2", "",
              "| reaches | n | position median full / far | p90 full / far | >2 cm full / far | wrist median full / far |",
              "|---|---|---|---|---|---|"]  # fmt: skip
        for s, r in far.items():
            o, n = r["full"], r["far"]
            L2.append(f"| {s} | {o['n']} | {o['median']:.2f} / {n['median']:.2f} | {o['p90']:.2f} / {n['p90']:.2f} | "
                      f"{o['gt2']:.2f} / {n['gt2']:.2f} | {o['att']:.1f} / {n['att']:.1f} |")  # fmt: skip
        (out / "lc_far.md").write_text("\n".join(L2) + "\n")
        L += [""] + L2
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
