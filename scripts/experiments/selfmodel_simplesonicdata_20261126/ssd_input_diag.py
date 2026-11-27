# notebook: selfmodel_simplesonicdata_20261126 -- read-only, CPU: which of D's input columns (x0, u) put a set of
# recorded reaches outside the model's training distribution (revision 2's diagnosis of the GATE FAIL)
"""For the reaches of one --correct record directory (<corr>/data/*.npz, data3's layout; the same arrays sh_err.py and
the gate read), build x0 and u with data.py (the switches of the environment: run with the model's REACHWM_*), and z-score
every column with the model's own normalisation (each ensemble member's nx / nu: mean and std of its training split, std
floored at 1e-3 by wm.Norm). Prints the columns by median |z|, the model's prediction error and ensemble spread as
recorded, and the same with one column replaced (--zero <col>, a diagnosis only: not a gate number).

    ssd_input_diag.py --models <dir with E3_ensemble_f1> --corr <corr dir> --label <name> [--top 15] [--zero 204]"""
import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--models", type=Path, required=True)
ap.add_argument("--corr", type=Path, required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--top", type=int, default=15)
ap.add_argument("--zero", type=int, action="append", default=[], help="x0 column set to 0 for a second prediction (diagnosis)")
g = ap.parse_args()
R = Path(os.environ["SX_REPO"])
sys.path[:0] = [str(R / "scripts" / "experiments" / "reachwm_20261030"), str(R / "src")]
import torch  # noqa: E402

import data as DT  # noqa: E402
import models as M  # noqa: E402
import train as TR  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
try:
    import lml_selfmodel as LS  # noqa: E402  (reachwm_20261030: S's joint order)

    JN = [n.replace("_joint", "") for n in LS.SCALEBFM_ORDER]
except Exception:  # noqa: BLE001
    JN = [f"j{i}" for i in range(29)]
assert len(JN) == 29


def names() -> tuple[list[str], list[str]]:
    srow = [f"q:{j}" for j in JN] + [f"dq:{j}" for j in JN] + ["grav_x", "grav_y", "grav_z", "w_x", "w_y", "w_z", "pelvis_z"]
    x = [f"S[k0{s}].{n}" for s in ("", "-1", "-2") for n in srow]
    x += [f"jaw0_{c}" for c in "xyz"] + [f"rot0_{i}" for i in range(6)] + ["walked", "chain"]
    if DT.X0_REF:
        x += [f"x0ref_{i}" for i in range(6)]
    if DT.GOAL_REF:
        x += [f"ref_q:{j}" for j in JN] + [f"ref_q-start:{j}" for j in JN] + [f"ref_pel_{c}" for c in "xyz"]
        x += [f"ref_c0_{c}" for c in "xyz"] + [f"ref_c1_{c}" for c in "xyz"] + ["ref_s"]
    if DT.APPROACH:
        x += ["walk_fwd", "walk_left", "walk_sin", "walk_cos-1"]
    u = ["side", "target_x", "target_y", "target_z"] + [f"rot6_{i}" for i in range(6)] + ["stand", "pelvis_z"]
    assert len(x) == DT.X0_DIM and len(u) == DT.U_DIM, (len(x), DT.X0_DIM, len(u), DT.U_DIM)
    return x, u


fs = sorted(glob.glob(f"{g.corr}/data/*.npz"))
D = DT.load(",".join(fs))
m = TR.load_model("E3_ensemble", 1.0, D, "cpu", root=g.models)
mem = [mod for _, mod in m.named_modules() if hasattr(mod, "nx") and hasattr(mod, "nu")]
xn, un = names()
keep = np.asarray(D["fell"]) == 0
x0, u = np.asarray(D["x0"], np.float64)[keep], np.asarray(D["u"], np.float64)[keep]
print(f"== {g.label}: {g.corr} ({len(fs)} files, {int(keep.sum())} reaches kept of {len(keep)}); model {g.models}/E3_ensemble_f1, "
      f"{len(mem)} members; switches GOAL_REF {DT.GOAL_REF} APPROACH {DT.APPROACH} QFIX {DT.QFIX} FDVEL {DT.FDVEL}")
zs = []
for k, mb in enumerate(mem):
    mx, sx = mb.nx.mu.numpy().astype(np.float64), mb.nx.sd.numpy().astype(np.float64)
    mu_, su = mb.nu.mu.numpy().astype(np.float64), mb.nu.sd.numpy().astype(np.float64)
    zs.append((np.abs((x0 - mx) / sx), np.abs((u - mu_) / su), mx, sx))
zx, zu, mx, sx = zs[0]
print(f"  member 0's normalisation; the 'walked' column (x0 {xn.index('walked')}): training mean / std per member "
      + "; ".join(f"{z[2][xn.index('walked')]:.4f} / {z[3][xn.index('walked')]:.4f}" for z in zs)
      + f"; this set's values {sorted(set(np.round(x0[:, xn.index('walked')], 3).tolist()))}")
cols = [(f"x0[{i}] {xn[i]}", zx[:, i]) for i in range(len(xn))] + [(f"u[{i}] {un[i]}", zu[:, i]) for i in range(len(un))]
cols.sort(key=lambda c: -np.median(c[1]))
print(f"  columns by median |z| (member 0), top {g.top}; |z| > 4 share")
for nm, z in cols[: g.top]:
    print(f"    {nm:34s} median |z| {np.median(z):9.2f}  max {z.max():9.2f}  |z| > 4: {100 * (z > 4).mean():5.1f} %")
big = [nm for nm, z in cols if np.median(z) > 4]
print(f"  columns with median |z| > 4: {len(big)} of {len(cols)}")


def run(Dz: dict, tag: str) -> None:
    GD = M.tensors(Dz, "cpu")
    o = M.predict_batched(m, GD, torch.arange(len(Dz["n_steps"])))
    pos, sd = o["pos"].numpy().astype(np.float64)[keep], o["pos_std"].numpy().astype(np.float64)[keep]
    e = 100 * np.linalg.norm(pos - np.asarray(Dz["pos_f"], np.float64)[keep], axis=1)
    s = 100 * np.linalg.norm(sd, axis=1)
    dr = np.asarray(Dz["err_cm"], np.float64)[keep]
    print(f"  {tag}: model error median {np.median(e):.3f} cm, direct miss median {np.median(dr):.3f} cm, model / direct "
          f"{np.median(e) / np.median(dr):.3f}; ensemble spread |std| median {np.median(s):.2f} cm (p10 {np.percentile(s, 10):.2f}, "
          f"p90 {np.percentile(s, 90):.2f})")


run(D, "as recorded")
if g.zero:
    Dz = dict(D)
    x = np.asarray(D["x0"]).copy()
    x[:, g.zero] = 0.0
    Dz["x0"] = x
    run(Dz, f"x0 columns {g.zero} ({', '.join(xn[c] for c in g.zero)}) set to 0 -- a diagnosis, not a gate number")
