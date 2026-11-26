"""selfmodel_hadata_20261126 revision 2b: one collection's training and evaluation input -- <dir>/data/all.npz (the
one-hand segments, sx_collect merge) without the segments that touched the scene, revision 1's rule (xfer's):
touch_n > 1 N and touch_z >= 0.2 m (the largest non-foot contact force from the command to the end label, and the height
of the body that took it). Written as <out> in the same layout, so data.py reads it unchanged.

    cd <repo>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1 LOCO_REPO=<repo> \
        $PY hd_touch.py <collection dir> <out npz> [data|data_both]

The third argument picks the merged file: data (default; the training input) or data_both (the two-hand segments, one
per hand, evaluation only: gate (b)'s two-hand row).

Arrays by their first dimension: per segment (N = len(n_steps)) masked; per row (sum of n_steps: S, W, J, C, HQ, KP)
the kept segments' rows; per reference row (sum of n_ref: RT) the kept segments' reference rows; anything else (or a
length that fits two of these) fails the run.
Checks, any failure exits 1:
  1. counts: kept + touching = N; every key present in the output; per-row / per-reference totals match the kept
     segments' n_steps / n_ref;
  2. data.load of the output against data.load of all.npz restricted to the kept segments (the same flags, so the same
     reader as the training): every derived per-segment array (x0, u, pos_f, rot_f, settled, fell, err_cm, target, rot6,
     jaw0, rot0, pel_f, q_f, bucket, frac_rank, band_* and edge) byte-equal;
  3. data.splits: each kept segment in the same split as in all.npz (the split is a function of the reset id).
Prints the counts (touching segments by data3b kind, chain, walked, fallen) and "TOUCH FILTER PASS" / "... FAIL".
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("LOCO_REPO", "/wuji-vefps-D/wuji-il/yijie/ws/code/loco"))
sys.path.insert(0, str(REPO / "scripts" / "experiments" / "reachwm_20261030"))
import data as DT  # noqa: E402

TOUCH_N, TOUCH_Z = 1.0, 0.2  # N, m (revision 1)
DERIVED = ("x0", "u", "pos_f", "rot_f", "settled", "fell", "err_cm", "target", "rot6", "jaw0", "rot0", "pel_f", "q_f",
           "bucket", "frac_rank", "band_x", "band_z", "band_yaw", "band_pel", "edge")  # fmt: skip


def touching(z: dict) -> np.ndarray:
    return (np.asarray(z["touch_n"], float) > TOUCH_N) & (np.asarray(z["touch_z"], float) >= TOUCH_Z)


def split_of(D: dict) -> np.ndarray:
    lab = np.full(len(D["n_steps"]), "", dtype=object)
    for k, idx in DT.splits(D).items():
        for i in idx:
            lab[i] = lab[i] + k + ";"
    return lab


def main() -> int:
    sub = sys.argv[3] if len(sys.argv) > 3 else "data"
    assert sub in ("data", "data_both"), sub
    src, out = Path(sys.argv[1]) / sub / "all.npz", Path(sys.argv[2])
    z = dict(np.load(src))
    n = z["n_steps"].astype(int)
    N, rows = len(n), int(n.sum())
    nref = z["n_ref"].astype(int)
    refs = int(nref.sum())
    assert len({N, rows, refs}) == 3, f"ambiguous lengths: segments {N}, rows {rows}, reference rows {refs}"
    bad = touching(z)
    keep = ~bad
    rmask = np.repeat(keep, n)
    fmask = np.repeat(keep, nref)
    o, ok = {}, True
    for k, v in z.items():
        v = np.asarray(v)
        if v.ndim == 0:
            print(f"FAIL {k}: a scalar array")
            ok = False
            continue
        L = v.shape[0]
        o[k] = v[keep] if L == N else v[rmask] if L == rows else v[fmask] if L == refs else None
        if o[k] is None:
            print(f"FAIL {k}: first dimension {L} is none of {N} / {rows} / {refs}")
            ok = False
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.stem}.tmp.npz")
    np.savez_compressed(tmp, **{k: v for k, v in o.items() if v is not None})
    os.replace(tmp, out)
    # 1. counts
    nk = int(keep.sum())
    ok &= nk + int(bad.sum()) == N and set(o) == set(z)
    ok &= len(o["S"]) == int(n[keep].sum()) and len(o["RT"]) == int(nref[keep].sum())
    print(f"{src}: {N} segments, touching {int(bad.sum())} ({100 * bad.mean():.1f} %), kept {nk}; rows {rows} -> {len(o['S'])}")
    for key in ("kind", "chain", "walked", "fell"):
        if key in z:
            vals = np.asarray(z[key])
            print(f"  touching by {key}: " + ", ".join(f"{u}: {int((bad & (vals == u)).sum())}/{int((vals == u).sum())}" for u in np.unique(vals)))
    # 2. the reader
    A, B = DT.load(str(src)), DT.load(str(out))
    for k in DERIVED:
        a, b = np.asarray(A[k])[keep], np.asarray(B[k])
        same = a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
        if not same:
            print(f"FAIL derived {k}: differs from all.npz's kept segments")
            ok = False
    print(f"  derived arrays ({len(DERIVED)}) byte-equal to all.npz's kept segments: {all(np.asarray(A[k])[keep].tobytes() == np.asarray(B[k]).tobytes() for k in DERIVED)}")
    # 3. the splits
    sa, sb = split_of(A)[keep], split_of(B)
    same = bool((sa == sb).all())
    print(f"  splits of the kept segments unchanged: {same}; kept splits {{{', '.join(f'{k}: {len(v)}' for k, v in DT.splits(B).items())}}}")
    ok &= same
    print("TOUCH FILTER", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
