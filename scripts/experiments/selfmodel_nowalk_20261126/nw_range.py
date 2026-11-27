"""selfmodel_nowalk_20261126: simpledata's evaluation 4 (sd_range.py, frozen, unchanged) run against one arm's training
set and model: data.load is wrapped so that the training spec (--train) gets the arm's dropped resets out of the
training split (bucket 10, nw_split's splits.npz) and, for N / C2, every loaded set gets x0's walk-flag column (204)
zeroed (as the arm was trained and is evaluated). sd_range's own ranges, neighbour threshold, table check and output
then describe the arm's training set and the arm's model.

    cd <repo>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1 LOCO_REPO=<repo> CUDA_VISIBLE_DEVICES= \
        $PY nw_range.py --arm N --splits <splits.npz> --sd-code <simpledata asrun/repo/.../selfmodel_simpledata_20261126> -- \
        --train <D's spec> --data3 <data3 all.npz> --models <out>/models_N_s0 --real main=...,task=... --json <F>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("LOCO_REPO", "/wuji-vefps-D/wuji-il/yijie/ws/code/loco"))
sys.path[:0] = [str(REPO / "scripts/experiments/reachwm_20261030"), str(REPO / "scripts/experiments/selfwm_20261029")]
import data as DT  # noqa: E402

WALKED_COL = 204


def main() -> None:
    argv = sys.argv[1:]
    k = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["D", "N", "C", "C2"])
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--sd-code", type=Path, required=True)
    a = ap.parse_args(argv[:k])
    rest = argv[k + 1 :]
    train_spec = rest[rest.index("--train") + 1]
    Z = np.load(a.splits)
    drop, zero = Z[f"{a.arm}_drop"].astype(np.int64), bool(Z[f"{a.arm}_zero"])
    n_train = int(Z[f"{a.arm}_n_train"])
    orig = DT.load

    def load(spec=None):
        D = orig(spec)
        assert (np.asarray(D["x0"])[:, WALKED_COL] == np.asarray(D["walked"]).astype(np.float32)).all()
        if spec == train_spec:
            D["bucket"] = np.where(np.isin(np.asarray(D["reset"]).astype(np.int64), drop), 10, D["bucket"])
            assert len(DT.splits(D)["train"]) == n_train
            print(f"nw_range arm {a.arm}: training split {n_train} segments ({len(drop)} resets dropped)", flush=True)
        if zero:
            D["x0"][:, WALKED_COL] = 0.0
        return D

    DT.load = load
    sys.path.insert(0, str(a.sd_code))
    import sd_range as SR  # noqa: E402

    assert SR.DT is DT
    sys.argv = ["sd_range.py"] + rest
    print(f"nw_range: arm {a.arm}, zero walk flag {zero}; sd_range {SR.__file__}", flush=True)
    SR.main()


if __name__ == "__main__":
    main()
