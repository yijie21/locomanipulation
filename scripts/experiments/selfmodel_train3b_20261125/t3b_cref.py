"""selfmodel_train3b_20261125 revision 0: are the C rows of data3 / data3b the per-row reference hand point that
data.py's REACHWM_X0_REF reads (C[k0, :3], "the jaw centre the motor commands at k0, pelvis frame"), so model MX can be
trained? d3_collect.py writes C = (Rp^T (ref_hand - p), time since the command), ref_hand = the hand point (wrist +
R HAND_OFF) of the controller's kinematic reference frame at that row, moved back by the plan's xy shift (delta) into
the measured world; W's hands are the same hand point measured. Read-only checks on one data/all.npz (one-hand
segments), every row's pelvis rotation rebuilt as data.py's X0_REF does (gravity in S, yaw in W):
  1. time column: C[k0, 3] = 0, -1 before the command, +0.1 s per row after it.
  2. the reference at the end label is the plan's final IK pose, whose hand point lies ik_err_cm from the command:
     | |C[last, :3] - command (pelvis frame)| - ik_err_cm | on settled segments whose label falls after the plan ends.
  3. at the command (k0), |ref - measured| small on first reaches from the standing hold; on chained reaches (chain >= 1)
     whose previous reach used the same hand, ref - measured follows that reach's miss: the hand stays where it landed,
     the reference stays at the plan's end, so ref - meas ~ (IK hand - landing). Correlation of |ref - meas| at k0
     with the previous segment's |IK hand - landing| at its label (C[last] - meas[last], same quantity one reach back).
Prints the numbers and PASS / FAIL (thresholds below); exits 1 on FAIL.

    cd <repo>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 $PY t3b_cref.py <all.npz> [<all.npz> ...]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("LOCO_REPO", "/wuji-vefps-D/wuji-il/yijie/ws/code/loco"))
sys.path.insert(0, str(REPO / "scripts" / "experiments" / "reachwm_20261030"))
import data as DT  # noqa: E402

TOL_TIME = 1e-4  # s
TOL_END_CM = 0.5  # end-label reference vs the IK hand, median
MIN_CORR = 0.8  # chained reaches, same hand: |ref - meas| at k0 vs the previous reach's |ref - meas| at its label


def pelvis_rot(S: np.ndarray, W: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """data.py X0_REF's pelvis rotation (Rz(yaw) Ry(pitch) Rx(roll) from S's gravity and W's yaw) at the given rows."""
    g = S[rows][:, DT.GRAV].astype(np.float64)
    th, ph = np.arcsin(np.clip(g[:, 0], -1, 1)), np.arctan2(-g[:, 1], -g[:, 2])
    yw = W[rows, 3].astype(np.float64)
    cz, sz, cy, sy, cx, sx = np.cos(yw), np.sin(yw), np.cos(th), np.sin(th), np.cos(ph), np.sin(ph)
    return np.stack([np.stack([cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx], -1),
                     np.stack([sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx], -1),
                     np.stack([-sy, cy * sx, cy * cx], -1)], -2)  # fmt: skip


def check(path: str) -> bool:
    D = DT.load(path)
    S, W, C = D["S"], D["W"].astype(np.float64), D["C"].astype(np.float64)
    n = D["n_steps"].astype(int)
    N = len(n)
    so = np.r_[0, np.cumsum(n)[:-1]]
    k0 = D["k0"].astype(int)
    g0, gl = so + k0, so + n - 1
    right = D["side"] > 0
    seg = np.repeat(np.arange(N), n)
    pos = np.arange(len(seg)) - so[seg]

    def meas(rows, segs):
        hw = np.where(right[segs, None], W[rows, 4:7], W[rows, 7:10])
        return np.einsum("nji,nj->ni", pelvis_rot(S, W, rows), hw - W[rows, :3])

    print(f"\n{path}: {N} segments, {len(seg)} rows, chain counts {np.bincount(D['chain'].astype(int)).tolist()}")
    # 1. time column
    want = np.where(pos < k0[seg], -1.0, (pos - k0[seg]) * 0.1)
    dt = np.abs(C[:, 3] - want)
    ok1 = bool(dt.max() < TOL_TIME)
    print(f"1. C[:, 3]: max |C - expected| {dt.max():.2e} s (-1 before the command, 0 at k0, +0.1 s per row) -> {ok1}")
    # 2. end label: reference = the plan's final IK pose (labels after the plan ends: time since command >= reach_s)
    st = (D["fell"] == 0) & (D["settled_end"] > 0) if "settled_end" in D else (D["fell"] == 0)
    after = C[gl, 3] >= D["reach_s"] - 1e-6
    sel = np.flatnonzero(st & after)
    st0, tg = D["start"].astype(np.float64), D["target"].astype(np.float64)
    c, s_ = np.cos(st0[:, 3]), np.sin(st0[:, 3])
    tw = np.c_[st0[:, 0] + c * tg[:, 0] - s_ * tg[:, 1], st0[:, 1] + s_ * tg[:, 0] + c * tg[:, 1], tg[:, 2]]
    Rl = pelvis_rot(S, W, gl[sel])
    tp = np.einsum("nji,nj->ni", Rl, tw[sel] - W[gl[sel], :3])
    d_end = 100 * np.linalg.norm(C[gl[sel], :3] - tp, axis=1)
    gap = np.abs(d_end - D["ik_err_cm"][sel])
    ok2 = bool(np.median(gap) < TOL_END_CM)
    print(f"2. end label, {len(sel)} settled segments labelled after the plan ends ({(st & ~after).sum()} before, left out): "
          f"|C_last - command| median {np.median(d_end):.2f} cm, ik_err_cm median {np.median(D['ik_err_cm'][sel]):.2f}; "
          f"| |C_last - command| - ik_err | median {np.median(gap):.3f} p90 {np.percentile(gap, 90):.3f} cm -> {ok2}")  # fmt: skip
    # 3. at the command
    r0 = 100 * np.linalg.norm(C[g0, :3] - meas(g0, np.arange(N)), axis=1)
    rl = 100 * np.linalg.norm(C[gl, :3] - meas(gl, np.arange(N)), axis=1)
    ch = D["chain"].astype(int)
    first, chained = ch == 0, ch >= 1
    q = lambda v: f"median {np.median(v):.2f} p90 {np.percentile(v, 90):.2f} cm (n {len(v)})"  # noqa: E731
    wk = D["walked"] > 0
    print(f"3. |ref - meas| at k0: first reaches {q(r0[first])} (in place {q(r0[first & ~wk])}, walk first "
          f"{q(r0[first & wk])}); chained {q(r0[chained])}; at the end label (all settled) {q(rl[st])}")  # fmt: skip
    print("   (a chained reach's k0 row is the previous reach's label row when the next command follows the label at once: "
          "check 3's pairing then compares one row recorded twice)")
    key = {(int(r), int(c_), int(sd)): i for i, (r, c_, sd) in enumerate(zip(D["reset"], ch, D["side"]))}
    pair = [(i, key[(int(D["reset"][i]), int(ch[i]) - 1, int(D["side"][i]))]) for i in np.flatnonzero(chained)
            if (int(D["reset"][i]), int(ch[i]) - 1, int(D["side"][i])) in key]  # fmt: skip
    pair = np.array([(i, j) for i, j in pair if st[j]], int).reshape(-1, 2)
    if len(pair) > 2:
        a, b = r0[pair[:, 0]], rl[pair[:, 1]]
        cor = float(np.corrcoef(a, b)[0, 1])
        # the signed vectors in the world: ref - meas at this k0 vs at the previous label
        va = np.einsum("nij,nj->ni", pelvis_rot(S, W, g0[pair[:, 0]]), C[g0[pair[:, 0]], :3] - meas(g0[pair[:, 0]], pair[:, 0]))
        vb = np.einsum("nij,nj->ni", pelvis_rot(S, W, gl[pair[:, 1]]), C[gl[pair[:, 1]], :3] - meas(gl[pair[:, 1]], pair[:, 1]))
        dv = 100 * np.linalg.norm(va - vb, axis=1)
        ok3 = cor > MIN_CORR
        print(f"   chained reaches after a settled reach of the same hand: {len(pair)}; corr(|ref - meas| at k0, the previous "
              f"reach's at its label) {cor:.3f}; |difference of the two world vectors| {q(dv)}; previous {q(b)} -> {ok3}")  # fmt: skip
    else:
        ok3 = True
        print(f"   chained reaches after a settled reach of the same hand: {len(pair)} (too few to correlate, not judged)")
    ok = ok1 and ok2 and ok3
    print(f"C ROW CHECK {Path(path).parent.parent.name}/{Path(path).parent.name}: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    ok = all([check(p) for p in sys.argv[1:]])
    print("C ROWS", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
