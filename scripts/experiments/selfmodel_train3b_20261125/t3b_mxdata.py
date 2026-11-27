"""selfmodel_train3b_20261125 revision 2 follow-up: model MX's input data, one controller -- data3's and data3b's
data/all.npz with only C[k0, :3] replaced by this reach's reference start (data.py unchanged; REACHWM_X0_REF reads
C[k0, :3]).

d3_collect writes C = Rp^T (ref_hand - p) with ref_hand = fk(reference frame) - shift, the shift being the plan's xy
offset between the reference root and the measured root (delta). On the command row k0 it still subtracts the previous
plan's shift (d_prev); this reach's plan starts from the reference pose held at k0 (shifted[0] = ref_pose) mapped with
its own shift d_this = (held reference root - measured root at k0, xy). This reach's reference start is therefore
X = fk(held pose) - d_this = Cw - (d_this - d_prev), Cw = C[k0] in the world, and d_this - d_prev = (held reference root
- d_prev) - measured root, the held root mapped with the old shift:
  first reach, in place  the start stance, root at the origin, d_prev = 0:            X = Cw + W[k0, :2]
  first reach, walk first the walk's goal stance; the walk plan starts at reference yaw 0 and moves the reference by
                         walk_cmd; the robot moved walk_real (its walk-start frame):  X = Cw - (walk_cmd - Rz(y0) walk_real)
  chained (chain >= 1)   the previous plan's last pose, root = its ref_pel (+ d_prev): X = Cw - (ref_pel_prev - W[k0, :2])
(xy only; z unchanged), then C'[k0, :3] = Rp_k0^T (X - p_k0) with data.py's X0_REF pelvis rotation. The previous reach
of a chained one is looked up by (reset, chain - 1) in data/, data_both/ and data_detour/.
Checks (any failure: exit 1, nothing written): every chained reach finds its previous reach, and when that reach used
the same hand its label row is byte-equal to this reach's command row (S, W, C), or else the held reference has not
moved between them (< 0.1 cm in the world); first in place: the stored
reference is one world point per hand (spread < 0.1 cm); walk first: X equals the goal stance's hand placed at the
robot's root with the commanded turn (median < 0.05 cm, p99 < 1 cm); every other array and every other row is
byte-equal to the source; data.load of the output with REACHWM_X0_REF=1 gives x0's reference features from C'[k0] and
the same splits as the source.

    cd <repo>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_X0_REF=1 $PY t3b_mxdata.py --backend heft \
        --out <dir>     -> <dir>/mx_data3.npz, mx_data3b.npz, mx.json
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


def build(name: str, d: Path, out: Path) -> tuple[dict, bool]:
    z = dict(np.load(d / "data" / "all.npz"))
    S, W, C = z["S"], z["W"].astype(np.float64), z["C"].astype(np.float64)
    n = z["n_steps"].astype(int)
    N = len(n)
    so = np.r_[0, np.cumsum(n)[:-1]]
    g0 = so + z["k0"].astype(int)
    R0 = pelvis_rot(S, W, g0)
    cw = np.einsum("nij,nj->ni", R0, C[g0, :3]) + W[g0, :3]
    ch, wk, sd = z["chain"].astype(int), z["walked"] > 0, z["side"]
    ip, iw, ic = np.flatnonzero((ch == 0) & ~wk), np.flatnonzero((ch == 0) & wk), np.flatnonzero(ch >= 1)
    X = cw.copy()
    X[ip, :2] += W[g0[ip], :2]
    y0 = W[g0[iw], 3] - z["walk_real"][iw, 2].astype(np.float64)
    c, s_ = np.cos(y0), np.sin(y0)
    wr = z["walk_real"][iw].astype(np.float64)
    X[iw, :2] -= z["walk_cmd"][iw, :2].astype(np.float64) - np.c_[c * wr[:, 0] - s_ * wr[:, 1], s_ * wr[:, 0] + c * wr[:, 1]]
    prev, last, srcs = {}, {}, []
    for sub in ("data", "data_both", "data_detour"):
        p = d / sub / "all.npz"
        if p.exists():
            zz = dict(np.load(p))
            srcs.append(sub)
            nl = zz["n_steps"].astype(int)
            gl = np.cumsum(nl) - 1
            Rl = pelvis_rot(zz["S"], zz["W"].astype(np.float64), gl)
            wl = np.einsum("nij,nj->ni", Rl, zz["C"][gl, :3].astype(np.float64)) + zz["W"][gl, :3]
            for j, (r, c_, sd_, rp) in enumerate(zip(zz["reset"], zz["chain"], zz["side"], zz["ref_pel"])):
                prev[(int(r), int(c_))] = rp[:2].astype(np.float64)
                # the previous reach's label row of the same hand: raw S / W / C bytes and the reference in the world
                last[(int(r), int(c_), int(sd_))] = (zz["S"][gl[j]].tobytes() + zz["W"][gl[j]].tobytes() + zz["C"][gl[j]].tobytes(), wl[j])
    keys = [(int(z["reset"][i]), int(ch[i]) - 1) for i in ic]
    found = np.array([k in prev for k in keys], bool)
    pr = np.array([prev.get(k, (np.nan, np.nan)) for k in keys]).reshape(-1, 2)
    X[ic, :2] -= pr - W[g0[ic], :2]
    # chained: is the command row the previous reach's label row (same hand)? if not, the held reference must not move
    same_row, moved = 0, []
    for i in ic:
        k = (int(z["reset"][i]), int(ch[i]) - 1, int(sd[i]))
        if k in last:
            b, wl_ = last[k]
            if b == z["S"][g0[i]].tobytes() + z["W"][g0[i]].tobytes() + z["C"][g0[i]].tobytes():
                same_row += 1
            else:
                moved.append(100 * float(np.linalg.norm(cw[i] - wl_)))
    # checks
    info = dict(name=name, source=str(d / "data" / "all.npz"), lookup=srcs, segments=N, first_in_place=len(ip),
                first_walk=len(iw), chained=len(ic), chained_previous_found=int(found.sum()))  # fmt: skip
    ok = bool(found.all())
    info.update(chained_same_hand_previous=same_row + len(moved), chained_command_row_is_previous_label_row=same_row,
                chained_other_rows=len(moved),
                chained_other_rows_reference_moved_cm_max=float(max(moved)) if moved else 0.0)  # fmt: skip
    ok &= (max(moved) if moved else 0.0) < 0.1
    spread = {}
    for s in (1, -1):
        j = ip[sd[ip] == s]
        spread[int(s)] = float(100 * cw[j].std(0).max()) if len(j) else 0.0
    info["first_in_place_stored_reference_spread_cm"] = spread
    ok &= all(v < 0.1 for v in spread.values())
    res = []
    for s in (1, -1):
        jp, jw = ip[sd[ip] == s], iw[sd[iw] == s]
        if len(jw) == 0:
            continue
        off = cw[jp, :2].mean(0) if len(jp) else None
        assert off is not None, "walk-first segments without in-place ones to take the stance hand from"
        g = z["walk_cmd"][jw, 2].astype(np.float64)
        cg, sg = np.cos(g), np.sin(g)
        pred = W[g0[jw], :2] + np.c_[cg * off[0] - sg * off[1], sg * off[0] + cg * off[1]]
        res.append(100 * np.linalg.norm(X[jw, :2] - pred, axis=1))
    if res:
        r = np.concatenate(res)
        info["walk_first_vs_goal_stance_cm"] = dict(median=float(np.median(r)), p99=float(np.percentile(r, 99)), max=float(r.max()))
        ok &= np.median(r) < 0.05 and np.percentile(r, 99) < 1.0
    right = sd > 0
    meas = np.where(right[:, None], W[g0, 4:7], W[g0, 7:10])
    q = lambda v: dict(median=float(np.median(v)), p90=float(np.percentile(v, 90))) if len(v) else None  # noqa: E731
    for lab, ii in (("first_in_place", ip), ("first_walk", iw), ("chained", ic)):
        info[f"{lab}_ref_minus_meas_cm_stored"] = q(100 * np.linalg.norm(cw[ii] - meas[ii], axis=1))
        info[f"{lab}_ref_minus_meas_cm_mx"] = q(100 * np.linalg.norm(X[ii] - meas[ii], axis=1))
        info[f"{lab}_correction_cm"] = q(100 * np.linalg.norm((X[ii] - cw[ii])[:, :2], axis=1))
    if not ok:
        return info, False
    zo = dict(z)
    Cn = z["C"].copy()
    Cn[g0, :3] = np.einsum("nji,nj->ni", R0, X - W[g0, :3]).astype(Cn.dtype)
    zo["C"] = Cn
    rows = np.ones(len(Cn), bool)
    rows[g0] = False
    same = all(np.array_equal(zo[k], z[k]) for k in z if k != "C") and np.array_equal(Cn[rows], z["C"][rows]) \
        and np.array_equal(Cn[g0, 3], z["C"][g0, 3])  # fmt: skip
    info["other_arrays_and_rows_byte_equal"] = bool(same)
    ok &= same
    path = out / f"mx_{name}.npz"
    tmp = out / f".mx_{name}.tmp.npz"
    np.savez(tmp, **zo)
    # data.py's view (REACHWM_X0_REF=1): the reference features come from C'[k0]; the splits are the source's
    if DT.X0_REF:
        Dm, Ds = DT.load(str(tmp)), DT.load(str(d / "data" / "all.npz"))
        b = 206  # X0_REF's six columns follow the base x0 (data.py)
        ref = Dm["x0"][:, b : b + 3].astype(np.float64)
        e = float(np.abs(ref - Cn[g0, :3].astype(np.float32)).max())
        sp_m, sp_s = DT.splits(Dm), DT.splits(Ds)
        same_sp = all(np.array_equal(sp_m[k], sp_s[k]) for k in sp_s)
        same_x0 = np.array_equal(np.delete(Dm["x0"], np.s_[b : b + 6], 1), np.delete(Ds["x0"], np.s_[b : b + 6], 1))
        info.update(x0_reference_from_new_C_max_abs=e, splits_equal_to_source=bool(same_sp), x0_other_columns_equal=bool(same_x0))
        ok &= e < 1e-6 and same_sp and same_x0
    else:
        ok = False
        info["error"] = "run with REACHWM_X0_REF=1 (the data.py check needs it)"
    if ok:
        os.replace(tmp, path)
    else:
        tmp.unlink()
    return info, bool(ok)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["scalebfm", "heft", "sonic"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data3", type=Path, default=None, help="data3's output dir of this controller")
    ap.add_argument("--data3b", type=Path, default=None, help="data3b's output dir of this controller")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    src = dict(data3=a.data3 or ROOT / "selfmodel_data3_20261125" / a.backend,
               data3b=a.data3b or ROOT / "selfmodel_data3b_20261125" / a.backend)  # fmt: skip
    infos, ok = [], True
    for name, d in src.items():
        info, good = build(name, d, a.out)
        print(json.dumps(info), flush=True)
        infos.append(info)
        ok &= good
    json.dump(infos, open(a.out / "mx.json", "w"), indent=1)
    print("MX DATA", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
