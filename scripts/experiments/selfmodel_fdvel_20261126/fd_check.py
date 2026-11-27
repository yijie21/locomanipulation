"""selfmodel_fdvel_20261126's consistency checks for data.py's REACHWM_FDVEL switch, read-only (data only).

    cd $A/repo && REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 $VPY .../fd_check.py --old <train3 asrun data.py> \
        --lml <data3>/{scalebfm,heft,sonic} --xfer <xfer>/<record>...

1. Switch off = before: every record loaded with this repo's data.py (FDVEL off) and with train3's frozen data.py
   (e0cabb83); x0, u and every other derived array must be bit-identical (max |difference| 0).
2. Switch on: only x0's 96 velocity columns change (data.X0_DQ_COLS, X0_ANGVEL_COLS); the 182 others stay bit-identical;
   the joint-velocity columns equal sx_dq.py's "dq from q" (the backward difference of q at 10 Hz).
3. Differences against the readings: on every record, the difference features against the recorded velocities
   (joint velocities S 29:58, pelvis angular velocity S 61:64), at x0's rows (S[k0], S[k0-1], S[k0-2] of every segment)
   and at every row that has a row before it in its segment (the reach itself included): pooled Pearson correlation,
   per-joint / per-axis correlation (median, minimum), median of difference - reading, median |difference - reading|,
   median |reading| and |difference|. LocoManiLab's (data3) are the check; the xfer records are printed for reference.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path.cwd()
assert os.environ.get("REACHWM_FDVEL", "0") == "0", "run with REACHWM_FDVEL unset: the check toggles data.FDVEL itself"
for k in ("REACHWM_GOAL_REF", "REACHWM_APPROACH", "REACHWM_QFIX"):
    assert os.environ.get(k) == "1", f"{k}=1 is train3's switch"
sys.path[:0] = [str(REPO / "scripts/experiments/reachwm_20261030"), str(REPO / "scripts/experiments/selfwm_20261029")]
import data as DT  # noqa: E402

NS = DT.S_DIM


def load_old(path: Path):
    spec = importlib.util.spec_from_file_location("data_train3", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def equal(a, b) -> tuple[bool, float]:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False, float("inf")
    if a.dtype.kind in "fc":
        same = np.array_equal(a, b, equal_nan=True) and np.array_equal(np.signbit(a), np.signbit(b))
        return same, float(np.nanmax(np.abs(a.astype(np.float64) - b))) if a.size else 0.0
    return bool(np.array_equal(a, b)), 0.0 if np.array_equal(a, b) else float("inf")


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def stats(fd: np.ndarray, rd: np.ndarray) -> str:
    """fd, rd (N, k): difference features and readings."""
    pc = [corr(fd[:, j], rd[:, j]) for j in range(fd.shape[1])]
    d = fd - rd
    j_min = int(np.nanargmin(pc))
    return (f"pooled r {corr(fd, rd):.3f}, per-column r median {np.nanmedian(pc):.3f} min {pc[j_min]:.3f} (col {j_min}); "
            f"median(diff - reading) {np.median(d):+.5f}, median |diff - reading| {np.median(np.abs(d)):.5f}, "
            f"median |reading| {np.median(np.abs(rd)):.5f}, median |diff| {np.median(np.abs(fd)):.5f}")  # fmt: skip


def all_rows(D: dict) -> np.ndarray:
    """Global rows with a row before them in the same segment."""
    n = D["n_steps"].astype(int)
    so = np.r_[0, np.cumsum(n)[:-1]]
    first = np.zeros(int(n.sum()), bool)
    first[so] = True
    return np.flatnonzero(~first)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=Path, required=True, help="train3's frozen data.py (asrun/train/data.py)")
    ap.add_argument("--lml", type=Path, nargs="+", required=True)
    ap.add_argument("--xfer", type=Path, nargs="*", default=[])
    a = ap.parse_args()
    old = load_old(a.old)
    print(f"data.py {DT.__file__}; train3's {a.old}; X0_DIM {DT.X0_DIM}; velocity columns: joint {len(DT.X0_DQ_COLS)} "
          f"{DT.X0_DQ_COLS[[0, 28, 29, 57, 58, 86]].tolist()} (first/last of each row), angular {DT.X0_ANGVEL_COLS.tolist()}; "
          f"ROW_S {DT.ROW_S}")  # fmt: skip
    vel = np.r_[DT.X0_DQ_COLS, DT.X0_ANGVEL_COLS]
    other = np.setdiff1d(np.arange(DT.X0_DIM), vel)
    ok_all = True
    for rec in [(p, "lml") for p in a.lml] + [(p, "xfer") for p in a.xfer]:
        p, kind = rec
        f = str(p / "data" / "all.npz")
        DT.FDVEL = False
        Dn = DT.load(f)
        Do = old.load(f)
        DT.FDVEL = True
        Df = DT.load(f)
        DT.FDVEL = False
        # 1. off = before
        keys = sorted(k for k in Do if k != "files")
        bad = []
        for k in keys:
            same, md = equal(Dn[k], Do[k])
            if not same:
                bad.append((k, md))
        same_x0, md_x0 = equal(Dn["x0"], Do["x0"])
        extra = sorted(set(Dn) - set(Do))
        ok1 = not bad and same_x0 and set(Dn) >= set(Do)
        # 2. on: only the velocity columns change; dq = sx_dq's "dq from q"
        same_other, md_other = equal(Df["x0"][:, other], Do["x0"][:, other])
        keys_on = [k for k in keys if k != "x0" and not equal(Df[k], Do[k])[0]]
        n = Do["n_steps"].astype(int)
        r0 = np.r_[0, np.cumsum(n)[:-1]] + Do["k0"].astype(int)
        S = Do["S"]
        sxdq = np.concatenate([((S[r0 - s, DT.Q].astype(np.float64) - S[r0 - s - 1, DT.Q]) / 0.1).astype(np.float32)
                               for s in range(3)], 1)  # fmt: skip
        same_sxdq, md_sxdq = equal(Df["x0"][:, DT.X0_DQ_COLS], sxdq)
        changed = int((Df["x0"][:, vel] != Do["x0"][:, vel]).any(0).sum())
        ok2 = same_other and not keys_on and same_sxdq
        ok_all &= ok1 and ok2
        print(f"\n{p.name} ({kind}; {len(n)} segments, {int(n.sum())} rows, k0 min {int(Do['k0'].min())} max {int(Do['k0'].max())}):")
        print(f"  1. FDVEL off vs train3's data.py: x0 {Dn['x0'].shape} bit-identical {same_x0} (max |diff| {md_x0:.1f}); "
              f"other derived arrays {len(keys) - 1} compared, differing: {bad if bad else 'none'}; keys only in new: {extra} "
              f"-> {'PASS' if ok1 else 'FAIL'}")  # fmt: skip
        print(f"  2. FDVEL on: non-velocity x0 columns ({len(other)}) bit-identical {same_other} (max |diff| {md_other:.1f}); "
              f"velocity columns changed {changed} of {len(vel)}; other derived arrays changed: {keys_on if keys_on else 'none'}; "
              f"joint columns = sx_dq's dq from q: {same_sxdq} (max |diff| {md_sxdq:.2e}) -> {'PASS' if ok2 else 'FAIL'}")  # fmt: skip
        # 3. differences vs readings
        fdq, rdq = Df["x0"][:, DT.X0_DQ_COLS].astype(np.float64), Do["x0"][:, DT.X0_DQ_COLS].astype(np.float64)
        fw, rw = Df["x0"][:, DT.X0_ANGVEL_COLS].astype(np.float64), Do["x0"][:, DT.X0_ANGVEL_COLS].astype(np.float64)
        rows = all_rows(Do)
        dq_all, w_all = DT.fd_rows(S, Do["W"], rows)
        print("  3. difference features vs readings")
        print(f"     x0 rows, joint velocity (rad/s): {stats(fdq.reshape(-1, 29), rdq.reshape(-1, 29))}")
        print(f"     x0 rows, pelvis angular velocity (rad/s): {stats(fw.reshape(-1, 3), rw.reshape(-1, 3))}")
        print(f"     all rows ({len(rows)}), joint velocity: {stats(dq_all, S[rows, DT.DQ].astype(np.float64))}")
        print(f"     all rows, pelvis angular velocity: {stats(w_all, S[rows, DT.ANGVEL].astype(np.float64))}")
        pa = [corr(w_all[:, i], S[rows, 61 + i].astype(np.float64)) for i in range(3)]
        print(f"     all rows, angular velocity per axis r (x, y, z): {[round(v, 3) for v in pa]}")
        sys.stdout.flush()
    print(f"\nFDVEL CHECK {'PASSED' if ok_all else 'FAILED'} (items 1 and 2 on every record; item 3 is reported, not gated)")


if __name__ == "__main__":
    main()
