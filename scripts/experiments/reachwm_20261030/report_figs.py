"""Figure data for the report rewrite (2026-11-02): where the controller puts the hand when told a position, and how
much of that the world model predicts, on held-out reaches.

data_ref's test split (reset hash bucket 9, not fallen), the far-turn plan-aware ensemble (invert/wrist_invert2/
wm_ref.pt, pickled with its normalization). Per reach: the commanded jaw centre, the delivered one and the predicted one
(heading frame, left-hand reaches mirrored to the right). Writes media/figs.json: a fixed random sample of 400 reaches
for the arrow plots, the quantiles of both errors over the whole split, and every reach's offset of the delivered
position from the command and from the prediction (cm), for the target-board plots.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    REACHWM_GOAL_REF=1 NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=4 $PY scripts/experiments/reachwm_20261030/report_figs.py
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

MED = DT.OUT / "media"
MODEL = DT.OUT / "invert" / "wrist_invert2" / "wm_ref.pt"


def main() -> int:
    assert DT.GOAL_REF
    D = DT.load(str(DT.OUT / "data_ref" / "all.npz"))
    sp = DT.splits(D)
    idx = np.array([i for i in sp["test_iid"] if D["fell"][i] == 0])
    model = torch.load(MODEL, map_location="cpu", weights_only=False).eval()
    GD = M.tensors(D, "cpu")
    p = M.predict_batched(model, GD, torch.as_tensor(idx))
    pred = p["pos"].numpy().astype(np.float64)
    std = p["pos_std"].norm(dim=-1).numpy() * 100
    tgt = D["target"][idx].astype(np.float64)
    got = D["pos_f"][idx].astype(np.float64)
    mir = np.c_[np.ones(len(idx)), D["side"][idx], np.ones(len(idx))]  # y mirrored to the right hand
    tgt, got, pred = tgt * mir, got * mir, pred * mir
    e_cmd = 100 * np.linalg.norm(got - tgt, axis=1)
    e_wm = 100 * np.linalg.norm(got - pred, axis=1)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(idx), size=min(400, len(idx)), replace=False)
    q = np.linspace(0, 100, 101)
    out = dict(n=int(len(idx)), median_cmd=float(np.median(e_cmd)), median_wm=float(np.median(e_wm)),
               p90_cmd=float(np.percentile(e_cmd, 90)), p90_wm=float(np.percentile(e_wm, 90)),
               gt2_cmd=float(np.mean(e_cmd > 2)), gt2_wm=float(np.mean(e_wm > 2)),
               q_cmd=np.percentile(e_cmd, q).tolist(), q_wm=np.percentile(e_wm, q).tolist(),
               mean_offset_cm=(100 * (got - tgt).mean(0)).tolist(),
               off_cmd=(100 * (got - tgt)).round(2).tolist(), off_wm=(100 * (got - pred)).round(2).tolist(),
               sample=dict(target=tgt[pick].round(4).tolist(), got=got[pick].round(4).tolist(), pred=pred[pick].round(4).tolist(),
                           std=std[pick].round(2).tolist(), pelvis_z=D["pelvis_z"][idx][pick].round(3).tolist()))  # fmt: skip
    MED.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(MED / "figs.json", "w"))
    print(
        f"{len(idx)} test reaches: command miss median {out['median_cmd']:.2f} cm p90 {out['p90_cmd']:.2f}; "
        f"model miss median {out['median_wm']:.2f} p90 {out['p90_wm']:.2f}; mean offset {np.round(out['mean_offset_cm'], 2)} cm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
