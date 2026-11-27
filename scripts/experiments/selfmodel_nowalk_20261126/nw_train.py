"""selfmodel_nowalk_20261126: train one arm (N, C, C2) with train3b's frozen t3b_train.py, unchanged, on the arm's view of
D's data: data.load is wrapped so that, after loading, the arm's dropped resets (nw_split.py's splits.npz) get bucket 10
(out of train / val / test_iid) and, for N and C2, x0's walk-flag column (204) is zeroed. Everything else -- the
splits, seeds, bootstrap, schedule, val pick, the saved model -- is t3b_train / train.train's.

    cd <repo>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1 LOCO_REPO=<repo> CUDA_VISIBLE_DEVICES=<g> \
        $PY nw_train.py --arm N --splits <out>/splits/splits.npz --t3b <train3b asrun dir> -- \
        E3_ensemble --seed 0 --frac 1 --data <D's spec> --out-root <out>/models_N_s0 [--iters N]

Before training it asserts the arm's training and val counts equal splits.npz's, writes <out-root>/arm.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("LOCO_REPO", "/wuji-vefps-D/wuji-il/yijie/ws/code/loco"))
sys.path.insert(0, str(REPO / "scripts" / "experiments" / "reachwm_20261030"))
import data as DT  # noqa: E402  (the module t3b_train and train import: the same object)

WALKED_COL, APPROACH_COLS = 204, slice(274, 278)


def main() -> None:
    argv = sys.argv[1:]
    k = argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["N", "C", "C2"])
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--t3b", type=Path, required=True, help="train3b's asrun dir holding t3b_train.py")
    a = ap.parse_args(argv[:k])
    rest = argv[k + 1 :]
    Z = np.load(a.splits)
    drop, zero = Z[f"{a.arm}_drop"].astype(np.int64), bool(Z[f"{a.arm}_zero"])
    n_train, n_val = int(Z[f"{a.arm}_n_train"]), int(Z[f"{a.arm}_n_val"])
    orig = DT.load
    seen = []

    def load(spec=None):
        D = orig(spec)
        assert (np.asarray(D["x0"])[:, WALKED_COL] == np.asarray(D["walked"]).astype(np.float32)).all()
        assert (np.asarray(D["x0"])[:, APPROACH_COLS] == 0).all()
        D["bucket"] = np.where(np.isin(np.asarray(D["reset"]).astype(np.int64), drop), 10, D["bucket"])
        if zero:
            D["x0"][:, WALKED_COL] = 0.0
        sp = DT.splits(D)
        v = sp["val"][np.asarray(D["fell"])[sp["val"]] == 0]
        assert len(sp["train"]) == n_train and len(v) == n_val, (len(sp["train"]), n_train, len(v), n_val)
        wt = int(np.asarray(D["walked"])[sp["train"]].sum())
        if a.arm == "N":
            assert wt == 0, f"N: {wt} walked training segments"
        x = np.asarray(D["x0"])[sp["train"]]
        info = dict(arm=a.arm, splits=str(a.splits), splits_md5=hashlib.md5(a.splits.read_bytes()).hexdigest()[:8],
                    n_drop_resets=int(len(drop)), zero_walk_flag=zero, n_train=n_train, n_val=n_val, train_walked_segments=wt,
                    train_x0_walked_col_mean=float(x[:, WALKED_COL].mean()), argv=rest)  # fmt: skip
        seen.append(info)
        print(f"nw_train arm {a.arm}: {json.dumps(info)}", flush=True)
        return D

    DT.load = load
    sys.path.insert(0, str(a.t3b))
    import t3b_train  # noqa: E402

    assert t3b_train.DT is DT, "t3b_train must see the wrapped data module"
    out_root = Path(rest[rest.index("--out-root") + 1])
    out_root.mkdir(parents=True, exist_ok=True)
    rc = t3b_train.main(rest)
    assert len(seen) == 1, f"data.load called {len(seen)} times"
    (out_root / "arm.json").write_text(json.dumps(seen[0], indent=1))
    sys.exit(rc)


if __name__ == "__main__":
    main()
