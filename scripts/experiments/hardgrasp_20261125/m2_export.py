"""selfmodel_hardgrasp_20261125 (BiGym): M2 (bigym_endpoint_20261125's model, outputs/bigym_20261124/selfmodel/model_early,
trained on selfmodel/data_early) exported for numpy, and the training paths' joint band.

Runs in a venv with torch (SIMPLE's). Writes
  <out>/m2_weights.npz  the point model M["point"]: per member the Linear layers' weights and biases in order
                        (W{member}_{layer}, b{member}_{layer}), n_members, n_layers, mu_x, sd_x, mu_y, sd_y, din, depth,
                        width, dropout (eval: no-op), and the md5 of via_point.pt
  <out>/band.npz        the training paths' (bucket < 8) S0 0.5 / 99.5 % percentiles, 29 body joints in the films'
                        order (bigym_data.joint_range's band), jnt names
and checks numpy against via_point.predict on 50 random rows of the training table (max |diff| must be < 1e-5 m).

    $SIMPLE/.venv/bin/python scripts/experiments/hardgrasp_20261125/m2_export.py \
        outputs/bigym_20261124/selfmodel/model_early outputs/bigym_20261124/selfmodel/data_early \
        outputs/hardgrasp_20261125/bigym/m2
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "selfmodel_via_20261120"))


def main() -> None:
    import torch
    import via_point as VP

    import m2 as M2mod

    model_dir, data_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    pt = model_dir / "via_point.pt"
    md5 = hashlib.md5(pt.read_bytes()).hexdigest()
    M = torch.load(pt, map_location="cpu", weights_only=False)
    P = M["point"]
    save = dict(n_members=len(P["members"]), mu_x=np.asarray(P["mu_x"], np.float32), sd_x=np.asarray(P["sd_x"], np.float32),
                mu_y=np.asarray(P["mu_y"], np.float32), sd_y=np.float32(P["sd_y"]), din=int(P["din"]),
                depth=int(P["depth"]), width=int(P["width"]), dropout=float(P["dropout"]), pt_md5=md5)  # fmt: skip
    n_layers = None
    for i, m in enumerate(P["members"]):
        st = m["state"]
        idx = sorted({int(k.split(".")[0]) for k in st})  # the Sequential's Linear layers (SiLU / Dropout have no params)
        n_layers = len(idx)
        for j, k in enumerate(idx):
            save[f"W{i}_{j}"] = st[f"{k}.weight"].numpy().astype(np.float32)
            save[f"b{i}_{j}"] = st[f"{k}.bias"].numpy().astype(np.float32)
    save["n_layers"] = n_layers
    np.savez(out / "m2_weights.npz", **save)

    # the band: bigym_data.joint_range's, the training paths only (bucket < 8)
    T = np.load(data_dir / "table.npz")
    S0 = T["S0"][T["bucket"] < 8]
    lo, hi = np.percentile(S0, [0.5, 99.5], axis=0)
    jnt = np.array([str(x) for x in np.load(data_dir / "arm.npz")["jnt"]])
    np.savez(out / "band.npz", lo=lo, hi=hi, jnt=jnt, n_paths=len(S0))
    print(f"exported {save['n_members']} members x {n_layers} Linear layers (din {save['din']}, width {save['width']}, "
          f"depth {save['depth']}) from {pt} (md5 {md5}); band over {len(S0)} training paths, {len(jnt)} joints")  # fmt: skip

    # the check: torch's via_point.predict against the numpy forward on 50 random training rows
    TT = VP.load([data_dir])
    rows = np.random.default_rng(0).choice(len(TT["seg"]), 50, replace=False)
    y_t = VP.predict(P, TT, rows)
    mm = M2mod.M2(out / "m2_weights.npz", out / "band.npz")
    y_n = mm.forward(VP.features(TT, rows)).reshape(len(rows), VP.Q + 1, 24, 3)
    d = float(np.abs(y_t - y_n).max())
    print(f"numpy vs torch on 50 training rows: max |diff| {d:.3e} m ({'OK' if d < 1e-5 else 'FAIL'})")
    if d >= 1e-5:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
