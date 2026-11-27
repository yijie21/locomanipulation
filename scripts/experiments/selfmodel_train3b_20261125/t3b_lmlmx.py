"""selfmodel_train3b_20261125, lmlreal revision 1 (coordinator, after 3cfb0ee6): model MX's input for
selfmodel_lmlreal_20261125's recorded episodes of one controller -- lmlreal's data/all.npz with C[k0, :3] replaced by
this reach's reference start on chained reaches (chain >= 1) only; data.py unchanged (REACHWM_X0_REF reads C[k0, :3]).

lr_run writes d3_collect's C rows (C = Rp^T (ref_hand - p), ref_hand = fk(reference) - shift, the previous plan's shift
d_prev on the command row k0) and, like d3_collect, holds the previous plan's last pose after every reach and sets the
shift to that plan's delta (walks happen only before the first reach). On a chained reach the reference start is
therefore, as in t3b_mxdata.py, X = Cw - (ref_pel_prev - W[k0, :2]) (xy; Cw = C[k0] in the world, ref_pel_prev = the
previous reach's plan end root, same episode = reset, chain - 1), C'[k0, :3] = Rp_k0^T (X - p_k0).
First reaches (chain 0) keep the stored C[k0]: their start needs this reach's d_this and d_prev, which lmlreal does
not record (the walk maps the stance with the believed pose; the fix walk and the in-place start reference are not
recorded), so MX is not evaluated on them (t3b_lmleval.py leaves MX empty wherever a chain-0 reach is in the cell).
Checks (any failure: exit 1, nothing written): every chained reach finds its previous reach; when that reach used the
same hand its label row and this command row hold the same reference in the world (< 0.1 cm: the previous plan's last
pose is held); every other array and every other row (chain-0 command rows included) byte-equal to the source;
data.load of the output with REACHWM_X0_REF=1 gives x0's reference features from C'[k0] and the other x0 columns
unchanged.

    cd <repo>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_X0_REF=1 $PY t3b_lmlmx.py --backend heft \
        --out <dir>     -> <dir>/mx_lmlreal.npz, mx_lmlreal.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("LOCO_REPO", "/wuji-vefps-D/wuji-il/yijie/ws/code/loco"))
sys.path.insert(0, str(REPO / "scripts" / "experiments" / "reachwm_20261030"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data as DT  # noqa: E402
from t3b_cref import pelvis_rot  # noqa: E402

ROOT = Path("/wuji-vefps-D/wuji-il/yijie/loco_outputs")
B_REF = 206  # X0_REF's six columns follow the base x0 (data.py)


def build(src: Path, out: Path) -> tuple[dict, bool]:
    z = dict(np.load(src))
    S, W, C = z["S"], z["W"].astype(np.float64), z["C"].astype(np.float64)
    n = z["n_steps"].astype(int)
    N = len(n)
    so = np.r_[0, np.cumsum(n)[:-1]]
    g0, gl = so + z["k0"].astype(int), so + n - 1
    R0, Rl = pelvis_rot(S, W, g0), pelvis_rot(S, W, gl)
    cw = np.einsum("nij,nj->ni", R0, C[g0, :3]) + W[g0, :3]
    cl = np.einsum("nij,nj->ni", Rl, C[gl, :3]) + W[gl, :3]
    ch, sd, rs = z["chain"].astype(int), z["side"].astype(int), z["reset"].astype(int)
    ic = np.flatnonzero(ch >= 1)
    key = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(rs, ch))}
    assert len(key) == N, "duplicate (reset, chain) in the source"
    pj = np.array([key.get((int(rs[i]), int(ch[i]) - 1), -1) for i in ic], int)
    found = pj >= 0
    X = cw.copy()
    X[ic[found], :2] -= z["ref_pel"][pj[found], :2].astype(np.float64) - W[g0[ic[found]], :2]
    same = found & (sd[ic] == sd[np.maximum(pj, 0)])
    moved = 100 * np.linalg.norm(cw[ic[same]] - cl[pj[same]], axis=1)
    right = sd > 0
    meas = np.where(right[:, None], W[g0, 4:7], W[g0, 7:10])
    q = lambda v: dict(median=float(np.median(v)), p90=float(np.percentile(v, 90))) if len(v) else None  # noqa: E731
    info = dict(source=str(src), segments=N, chain_counts=np.bincount(ch).tolist(), reach_no_counts=np.bincount(z["reach_no"].astype(int)).tolist(),
                first_kept_stored=int((ch == 0).sum()), chained=len(ic), chained_previous_found=int(found.sum()),
                chained_same_hand_previous=int(same.sum()),
                chained_reference_moved_since_previous_label_cm_max=float(moved.max()) if len(moved) else 0.0,
                chained_ref_minus_meas_cm_stored=q(100 * np.linalg.norm(cw[ic] - meas[ic], axis=1)),
                chained_ref_minus_meas_cm_mx=q(100 * np.linalg.norm(X[ic] - meas[ic], axis=1)),
                chained_correction_cm=q(100 * np.linalg.norm((X[ic] - cw[ic])[:, :2], axis=1)))  # fmt: skip
    ok = bool(found.all()) and (moved.max() if len(moved) else 0.0) < 0.1
    if not ok:
        return info, False
    Cn = z["C"].copy()
    Cn[g0[ic], :3] = np.einsum("nji,nj->ni", R0[ic], X[ic] - W[g0[ic], :3]).astype(Cn.dtype)
    zo = dict(z)
    zo["C"] = Cn
    rows = np.ones(len(Cn), bool)
    rows[g0[ic]] = False
    eq = all(np.array_equal(zo[k], z[k]) for k in z if k != "C") and np.array_equal(Cn[rows], z["C"][rows]) \
        and np.array_equal(Cn[g0, 3], z["C"][g0, 3])  # fmt: skip
    info["other_arrays_and_rows_byte_equal"] = bool(eq)
    ok &= eq
    tmp = out / ".mx_lmlreal.tmp.npz"
    np.savez(tmp, **zo)
    if DT.X0_REF:
        Dm, Ds = DT.load(str(tmp)), DT.load(str(src))
        e = float(np.abs(Dm["x0"][:, B_REF : B_REF + 3] - Cn[g0, :3].astype(np.float32)).max())
        e0 = float(np.abs(Dm["x0"][ch == 0] - Ds["x0"][ch == 0]).max()) if (ch == 0).any() else 0.0
        oc = np.array_equal(np.delete(Dm["x0"], np.s_[B_REF : B_REF + 6], 1), np.delete(Ds["x0"], np.s_[B_REF : B_REF + 6], 1))
        info.update(x0_reference_from_new_C_max_abs=e, x0_chain0_rows_unchanged_max_abs=e0, x0_other_columns_equal=bool(oc))
        ok &= e < 1e-6 and e0 == 0.0 and oc
    else:
        ok = False
        info["error"] = "run with REACHWM_X0_REF=1 (the data.py check needs it)"
    if ok:
        os.replace(tmp, out / "mx_lmlreal.npz")
    else:
        tmp.unlink()
    return info, bool(ok)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["scalebfm", "heft", "sonic"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--real", type=Path, default=None, help="lmlreal's data/all.npz of this controller")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    src = a.real or ROOT / "selfmodel_lmlreal_20261125" / a.backend / "data" / "all.npz"
    info, ok = build(src, a.out)
    print(json.dumps(info), flush=True)
    json.dump(info, open(a.out / "mx_lmlreal.json", "w"), indent=1)
    print("LML MX DATA", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
