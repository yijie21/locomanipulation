"""selfmodel_xfer_20261125, post-hoc diagnostic (not a registered metric), read-only: how much of an xfer record's
prediction error comes from the joint-velocity block of the start features.

HumanoidArena's joint_vel reads a large velocity on some joints while their positions stay put (scalebfm_ha: S joints
17 / 18 = left / right ankle roll in S's SCALEBFM_ORDER, +1.70 / -1.90 rad/s at the first command, positions changing
1e-4 rad over 0.4 s; LocoManiLab |dq| there ~0.002). The model's x0 holds S at the command row and the two rows before
it, dq = S[29:58] of each.
Variants, on the xfer record and (as a control) on LocoManiLab val, same model, same kept segments as sx_eval.py:
  recorded       x0 as recorded (reproduces sx_eval.py's primary)
  dq zero <J>    dq of joints J set to 0 in the three slices ("all" = every joint)
  dq from q      every dq replaced by the backward difference of the recorded q at 10 Hz
  sx_dq.py --lml <data3>/<ctrl> --xfer <out> [--joints 17,18] [--keep settled|ended]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sx_eval as E  # noqa: E402

NS, Q, DQ = 65, slice(0, 29), slice(29, 58)


def rows_k0(D: dict, idx: np.ndarray) -> np.ndarray:
    n = np.asarray(D["n_steps"]).astype(int)
    so = np.r_[0, np.cumsum(n)[:-1]]
    return so[idx] + np.asarray(D["k0"]).astype(int)[idx]


def variant(D: dict, idx: np.ndarray, how: str, joints: list[int]) -> dict:
    D2 = dict(D)
    x0 = np.array(D["x0"], np.float32, copy=True)
    S = D["S"]
    r = rows_k0(D, np.arange(len(D["n_steps"])))
    assert np.asarray(D["k0"]).min() >= 3, "the backward difference of the third slice needs k0 >= 3"
    for s in range(3):  # x0's S[k0], S[k0-1], S[k0-2]
        c = s * NS + DQ.start
        if how == "zero":
            for j in joints:
                x0[:, c + j] = 0.0
        elif how == "fd":
            x0[:, c : c + 29] = ((S[r - s, Q].astype(np.float64) - S[r - s - 1, Q]) / 0.1).astype(np.float32)
    D2["x0"] = x0
    return D2


def med(a: np.ndarray) -> float:
    return float(np.median(a))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lml", type=Path, required=True)
    ap.add_argument("--xfer", type=Path, required=True)
    ap.add_argument("--joints", default="17,18")
    ap.add_argument("--keep", choices=["settled", "ended"], default="settled")
    a = ap.parse_args()
    dev = "cpu"
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    DL = E.DT.load(str(a.lml / "data" / "all.npz"))
    DX = E.DT.load(str(a.xfer / "data" / "all.npz"))
    m = E.TR.load_model("E3_ensemble", 1.0, DL, dev, root=a.lml / "models_s0")
    vi = E.kept(DL, E.DT.splits(DL)["val"])
    xi = E.kept(DX, ended=a.keep == "ended")
    J = [int(j) for j in a.joints.split(",")]
    rv, rx = rows_k0(DL, vi), rows_k0(DX, xi)
    print(f"{a.xfer.name}: kept {len(xi)} xfer segments ({a.keep}), {len(vi)} LocoManiLab val; model {a.lml}/models_s0/E3_ensemble_f1")
    print("  |dq| at the command row, median / p90 (rad/s), xfer vs LocoManiLab val, and |backward difference of q| xfer:")
    for j in sorted(set(J) | {10, 19}):
        dx, dl = np.abs(np.asarray(DX["S"])[rx, 29 + j]), np.abs(np.asarray(DL["S"])[rv, 29 + j])
        fx = np.abs((np.asarray(DX["S"])[rx, j] - np.asarray(DX["S"])[rx - 1, j]) / 0.1)
        print(f"    joint {j:2d}: xfer {np.median(dx):.3f} / {np.percentile(dx, 90):.3f}; val {np.median(dl):.3f} / {np.percentile(dl, 90):.3f}; "
              f"xfer from q {np.median(fx):.3f} / {np.percentile(fx, 90):.3f}")  # fmt: skip
    ch_x, ch_v = np.asarray(DX["chain"])[xi], np.asarray(DL["chain"])[vi]
    print("  prediction error median (cm): xfer / val = ratio; first reach, chained, all; ensemble disagreement median xfer / val")
    for name, how, js in [("recorded", "none", []), (f"dq zero {a.joints}", "zero", J), ("dq zero all", "zero", list(range(29))),
                          ("dq from q", "fd", [])]:  # fmt: skip
        PX = E.predict(m, variant(DX, xi, how, js), xi, dev)
        PV = E.predict(m, variant(DL, vi, how, js), vi, dev)
        parts = []
        for lab, sx, sv in [("first", ch_x == 0, ch_v == 0), ("chained", ch_x > 0, ch_v > 0), ("all", ch_x >= 0, ch_v >= 0)]:
            ex, ev = med(PX["err"][sx]), med(PV["err"][sv])
            parts.append(f"{lab} {ex:.2f} / {ev:.2f} = {ex / ev:.2f}x")
        print(f"    {name:16s}: " + "; ".join(parts) + f"; disagreement {med(PX['sd']):.2f} / {med(PV['sd']):.2f}")


if __name__ == "__main__":
    main()
