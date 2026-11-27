"""selfmodel_simpledata_20261126 (revision 2): sd_mid.py's mid starts against the frozen t3b_mid.py's, on the same input.

    cd <repo>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 $PY sd_mid_check.py --old <t3b_mid out> --new <sd_mid out>
        --data3 <all.npz> --data3b <all.npz> [--n 3000]

Both outputs are built beforehand from the same --data3 / --data3b (t3b_mid.py from train3b's asrun, sd_mid.py).
  1. REACHWM_FDVEL off (this process): data.load of each mid file, old and new: the starts in the same order (reset,
     chain, side), and x0, u and the labels (t3b_mid's X_ARRAYS: x0, u, pos_f, rot_f, settled, fell, err_cm, target,
     rot6, jaw0, rot0, pel_f, q_f) byte for byte equal; the new starts all have k0 = 3 and one row more than the old.
  2. REACHWM_FDVEL on, on the new starts: data.fdvel_x0 (the function data.derive calls under the switch) against the
     source segments' own rows: for --n sampled starts, the 87 joint-velocity columns of S[k0], S[k0-1], S[k0-2] equal
     (q[g-s] - q[g-s-1]) / 0.1 s computed here from the source rows (s = 0, 1, 2; g the start row, from
     mid_<name>_src.npz), the 9 angular-velocity columns equal data.fd_rows on the source arrays at the same rows; the
     other 182 columns are the FDVEL-off x0's. Max |difference| printed (float32 rounding allowed: 1e-5).
  3. A child process with REACHWM_FDVEL=1 loads the new mid file through data.load (k0 >= 3 assertion) and its x0 is
     byte for byte the fdvel_x0 of step 2. The child runs with $SD_PY when set (the 4090b launcher,
     "/usr/bin/python3.10 -S .../py4090b.py"), else this interpreter.
Prints "MID FDVEL CHECK PASS" or "... FAIL"; exit 1 on a failure.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("LOCO_REPO", "/wuji-vefps-D/wuji-il/yijie/ws/code/loco"))
sys.path.insert(0, str(REPO / "scripts" / "experiments" / "reachwm_20261030"))
import data as DT  # noqa: E402

X_ARRAYS = ("x0", "u", "pos_f", "rot_f", "settled", "fell", "err_cm", "target", "rot6", "jaw0", "rot0", "pel_f", "q_f")
DT_ = 0.1


def child(path: str, dump: str) -> None:
    D = DT.load(path)
    np.save(dump, D["x0"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=Path)
    ap.add_argument("--new", type=Path)
    ap.add_argument("--data3", type=Path)
    ap.add_argument("--data3b", type=Path)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--child", nargs=2, default=None)
    a = ap.parse_args()
    if a.child:
        child(*a.child)
        return 0
    assert not DT.FDVEL, "the parent runs with REACHWM_FDVEL off"
    ok = True
    rng = np.random.default_rng(0)
    for name, srcp in (("data3", a.data3), ("data3b", a.data3b)):
        po, pn = a.old / f"mid_{name}.npz", a.new / f"mid_{name}.npz"
        zo, zn = dict(np.load(po)), dict(np.load(pn))
        Do, Dn = DT.load(str(po)), DT.load(str(pn))
        same_order = all(np.array_equal(np.asarray(Do[k]), np.asarray(Dn[k])) for k in ("reset", "chain", "side"))
        eq = {k: bool(np.asarray(Do[k]).dtype == np.asarray(Dn[k]).dtype and np.asarray(Do[k]).tobytes() == np.asarray(Dn[k]).tobytes())
              for k in X_ARRAYS}  # fmt: skip
        k0_new, k0_old = sorted(set(zn["k0"].tolist())), sorted(set(zo["k0"].tolist()))
        rows_plus1 = bool(np.array_equal(zn["n_steps"], zo["n_steps"] + 1))
        print(f"{name}: {len(zn['n_steps'])} new / {len(zo['n_steps'])} old starts; same order {same_order}; k0 new {k0_new} old {k0_old}; "
              f"n_steps new = old + 1: {rows_plus1}")  # fmt: skip
        print(f"  FDVEL off, byte equal: " + ", ".join(f"{k} {v}" for k, v in eq.items()))
        ok &= same_order and all(eq.values()) and k0_new == [3] and rows_plus1
        # 2. FDVEL on: fdvel_x0 against the source rows
        x_fd = DT.fdvel_x0(Dn, Dn["x0"])
        cols = np.zeros(x_fd.shape[1], bool)
        for s in range(3):
            cols[s * DT.S_DIM + DT.DQ.start : s * DT.S_DIM + DT.DQ.stop] = True
            cols[s * DT.S_DIM + DT.ANGVEL.start : s * DT.S_DIM + DT.ANGVEL.stop] = True
        other_same = bool(np.array_equal(x_fd[:, ~cols], Dn["x0"][:, ~cols]))
        z = dict(np.load(srcp))
        sr = dict(np.load(a.new / f"mid_{name}_src.npz"))
        assert len(sr["g"]) == len(zn["n_steps"]), "src file and mid file differ in length"
        js = np.sort(rng.choice(len(sr["g"]), size=min(a.n, len(sr["g"])), replace=False))
        g = sr["g"][js].astype(int)
        S, W = z["S"], z["W"]
        e_dq, e_w = 0.0, 0.0
        for s in range(3):
            dq_ref = (S[g - s, :29].astype(np.float64) - S[g - s - 1, :29].astype(np.float64)) / DT_
            got = x_fd[js, s * DT.S_DIM + DT.DQ.start : s * DT.S_DIM + DT.DQ.stop].astype(np.float64)
            e_dq = max(e_dq, float(np.abs(got - dq_ref).max()))
            _, w_ref = DT.fd_rows(S, W, g - s)
            got_w = x_fd[js, s * DT.S_DIM + DT.ANGVEL.start : s * DT.S_DIM + DT.ANGVEL.stop].astype(np.float64)
            e_w = max(e_w, float(np.abs(got_w - np.asarray(w_ref, np.float64)).max()))
        print(f"  FDVEL on, {len(js)} sampled starts vs the source rows: joint velocity max |d| {e_dq:.2e} rad/s, angular "
              f"velocity max |d| {e_w:.2e} rad/s; {int(cols.sum())} columns replaced, the other {int((~cols).sum())} equal to "
              f"FDVEL off: {other_same}")  # fmt: skip
        ok &= e_dq < 1e-5 and e_w < 1e-5 and other_same and int(cols.sum()) == 96
        # 3. a child with REACHWM_FDVEL=1 reads the file through data.load
        dump = str(a.new / f".x0_fdvel_{name}.npy")
        env = dict(os.environ, REACHWM_FDVEL="1")
        py = shlex.split(os.environ["SD_PY"]) if os.environ.get("SD_PY") else [sys.executable]
        r = subprocess.run(py + [__file__, "--child", str(pn), dump], env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  child with REACHWM_FDVEL=1 failed: {r.stderr[-800:]}")
            ok = False
            continue
        xc = np.load(dump)
        os.remove(dump)
        same_child = bool(xc.dtype == x_fd.dtype and xc.tobytes() == x_fd.tobytes())
        print(f"  child with REACHWM_FDVEL=1: data.load ran (k0 >= 3), its x0 byte equal to fdvel_x0 here: {same_child}")
        ok &= same_child
    print("MID FDVEL CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
