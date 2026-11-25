"""bigym_selfmodel_20261124: the test bucket's numbers via_point.py eval does not print: the palm's settled-point error and
the drops of the constant (train) against the reference and of the model against the constant (train), with 95 %
intervals resampling stances (via_point.report's bootstrap: 2000 draws, seed 0, a stance's reaches together), for
everything and by task and by the stance's first / second reach; and on how many paths each first predictor is
better. Either settled-point label (bigym_data.py build, with or without --end early): the model's own datasets. The predictions are via_point.offsets' (reference 0,
constant: the training paths' mean offset, model: the ensemble); errors via_point.errors (via_model's definition).

    $SIMPLE/.venv/bin/python scripts/experiments/bigym_20261124/bigym_test.py <via_point model dir> [--bucket 9]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "selfmodel_via_20261120"))
import via_model as VM  # noqa: E402
import via_point as VP  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--bucket", type=int, default=9)
    a = ap.parse_args()
    import torch

    M = torch.load(Path(a.model) / "via_point.pt", map_location="cpu", weights_only=False)
    dirs = VP.model_dirs(M, Path(a.model))
    T = VP.load(dirs)
    rows = np.where((~T["sector"]) & (T["bucket"] == a.bucket))[0]
    off = VP.offsets(M, T, rows)
    names = (("reference", "reference"), ("constant", "constant_train"), ("direction", "model"))
    res = {}
    for key, name in names:
        e, eh = VP.errors(dirs, T, off[key], rows)
        ps = np.linalg.norm((T["off"][rows, VM.Q] - off[key][:, VM.Q])[:, T["palm"]].mean(1), axis=1)
        res[name] = dict(e=e, eh=eh, ps=ps)
    subsets = [("all", np.ones(len(rows), bool))]
    for tid, tn in ((0, "reach"), (1, "drawer")):
        subsets.append((tn, T["task"][rows] == tid))
    for k in (0, 1):
        subsets.append((f"reach {k + 1} of the stance", T["k"][rows] == k))
    for sname, m in subsets:
        idx = np.where(m)[0]
        ids = T["route"][rows][idx]
        groups = [idx[ids == g] for g in np.unique(ids)]
        print(f"bucket {a.bucket}, {sname}: {len(idx)} paths, {len(groups)} stances")
        for _, name in names:
            v = res[name]
            print(f"  {name:15s} swept-region error median {100 * np.median(v['e'][idx]):5.2f} p90 "
                  f"{100 * np.percentile(v['e'][idx], 90):5.2f} cm | hand {100 * np.median(v['eh'][idx]):5.2f} cm | palm "
                  f"settled point median {100 * np.median(v['ps'][idx]):5.2f} p90 {100 * np.percentile(v['ps'][idx], 90):5.2f} cm")  # fmt: skip
        rng = np.random.default_rng(0)
        draws = [np.concatenate([groups[g] for g in rng.integers(len(groups), size=len(groups))]) for _ in range(2000)]

        def drop(x, y, field):
            b = [1 - np.median(res[x][field][j]) / np.median(res[y][field][j]) for j in draws]
            lo, hi = np.percentile(b, [2.5, 97.5])
            return f"{1 - np.median(res[x][field][idx]) / np.median(res[y][field][idx]):.2f} ({lo:.2f} to {hi:.2f})"

        pairs = (("model", "reference"), ("constant_train", "reference"), ("model", "constant_train"))
        for field, label in (("e", "swept-region error"), ("ps", "palm settled point")):
            print(f"  drops of the {label} median (95 % interval, resampling stances): "
                  + "; ".join(f"{x} vs {y} {drop(x, y, field)}" for x, y in pairs))  # fmt: skip
        for field, label in (("e", "swept-region error"), ("ps", "palm settled point")):
            print(f"  better on ({label}): " + "; ".join(
                f"{x} than {y} {int((res[x][field][idx] < res[y][field][idx]).sum())}/{len(idx)}" for x, y in pairs))  # fmt: skip


if __name__ == "__main__":
    main()
