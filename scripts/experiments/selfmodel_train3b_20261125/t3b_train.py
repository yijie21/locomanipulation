"""selfmodel_train3b_20261125: reachwm_20261030 train.py's main for E3_ensemble, with two changes and nothing else:
  1. --frac takes any fraction (0.6 for M_f0.6); data.splits nests the training fractions by the reset's frac rank,
     as it does for 0.1 / 0.3.
  2. the padded per-step tensors (models.tensors(step=True): S, O, tc as (N, L, 65)) are not built. E3 is an end model:
     EndNet / EndEnsemble read per-segment tensors only (models.sub drops S, O, tc for them), and D's ~1M segments
     padded to the longest one would not fit a 24 GB card.
The training itself is train.train unchanged (data.load, the splits, the seeds, sampling, schedule, val, best state).
Checked against `train.py E3_ensemble` on M's data for a short run: identical model.pt and log (t3b_equiv.py).

    cd <repo>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 \
        $VPY t3b_train.py E3_ensemble --frac 0.6 --seed 0 --data <spec> --out-root <dir> [--iters N]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

REPO = Path(os.environ.get("LOCO_REPO", "/wuji-vefps-D/wuji-il/yijie/ws/code/loco"))
sys.path.insert(0, str(REPO / "scripts" / "experiments" / "reachwm_20261030"))
import data as DT  # noqa: E402
import train as TR  # noqa: E402

M = TR.M


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["E3_ensemble"])
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--data", required=True, help="data.load spec (comma-separated files)")
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args(argv)
    assert 0 < a.frac <= 1
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = DT.load(a.data)
    GD = M.tensors(D, dev, step=False)
    print(f"{len(D['n_steps'])} segments, splits { {k: len(v) for k, v in DT.splits(D).items()} }", flush=True)
    TR.train(a.what, D, GD, a.frac, a.seed, a.iters, None, dev, Path(a.out_root), None, 1e-3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
