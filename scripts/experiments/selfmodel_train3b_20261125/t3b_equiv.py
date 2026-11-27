"""selfmodel_train3b_20261125: t3b_train.py against reachwm_20261030 train.py on the same data, seed and iterations:
the saved states must be equal tensor for tensor and the logs (iterations, losses, val errors) equal. Read-only on the
two model directories; exits 1 on a difference.

    $VPY t3b_equiv.py <train.py out-root> <t3b_train.py out-root> [frac]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


def main() -> int:
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    frac = sys.argv[3] if len(sys.argv) > 3 else "1"
    da, db = a / f"E3_ensemble_f{frac}", b / f"E3_ensemble_f{frac}"
    sa, sb = torch.load(da / "model.pt", map_location="cpu"), torch.load(db / "model.pt", map_location="cpu")
    keys = sorted(sa["state"])
    same_keys = keys == sorted(sb["state"])
    diff = [k for k in keys if not torch.equal(sa["state"][k], sb["state"].get(k, torch.empty(0)))]
    ja, jb = json.load(open(da / "train.json")), json.load(open(db / "train.json"))
    strip = lambda log: [{k: v for k, v in r.items() if k != "s"} for r in log]  # noqa: E731
    same_log = strip(ja["log"]) == strip(jb["log"])
    same_meta = all(ja[k] == jb[k] for k in ("n_train", "n_val", "best_val_median_cm", "iters", "batch", "seed", "frac"))
    print(f"{len(keys)} state tensors; same keys {same_keys}; differing tensors {len(diff)} {diff[:5]}; n_train {sa['n_train']} / "
          f"{sb['n_train']}; logs equal {same_log}; n_train / n_val / best val / iters equal {same_meta}")  # fmt: skip
    for r1, r2 in zip(ja["log"], jb["log"]):
        print(f"  it {r1['it']}: train {r1.get('train')} / {r2.get('train')}, val {r1['val_median_cm']} / {r2['val_median_cm']}")
    ok = same_keys and not diff and same_log and same_meta and sa["n_train"] == sb["n_train"]
    print("EQUIV", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
