"""selfmodel_simplecollect_20261126: the start state of the first reach of each reset, in records of data3's layout --
how much it varies from reset to reset (LocoManiLab's data3 against SIMPLE's records). Read only; numpy only.

    python sc_start.py NAME=DIR[,DIR...] [NAME=DIR ...] [--max-files N] [--json OUT]

DIR is a record directory (its data/all.npz, or data/w*.npz chunks when there is no all.npz; data_both/ and data_detour/
are read too, one row per two-hand reach). A "first reach" is a segment with chain 0 (the reset's first commanded reach); a reset's two-hand
first reach counts once. Groups: in place (walked 0) and walk-first (walked 1). At the command row S[k0]:
  joints     29 angles (S 0:29): per-joint std across resets (deg); the deviation from the group's per-joint median
             (deg, max over the 29 and RMS over the 29), its median / p90
  dq         the 0.1 s backward difference of the joints (data.py REACHWM_FDVEL, rad/s, norm over the 29): median / p90
  tilt       pelvis tilt from S's gravity (deg): median / std
  height     S[64] (m): median / std
  hand       the acting hand point relative to the pelvis, heading frame (W): std per axis (cm)
  unique     distinct start states (joints rounded to 1e-4 rad) / first reaches
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

Q, GRAV, H = slice(0, 29), slice(58, 61), 64


def load(d: Path, max_files: int = 0, keys=("n_steps", "k0", "chain", "walked", "reset", "side", "hands", "fell")) -> dict | None:
    """max_files > 0: at most that many chunk files per sub-directory (the first chunk of each shard first), not all.npz."""
    out = []
    for sub in ("data", "data_both", "data_detour"):
        p = d / sub / "all.npz"
        chunks = sorted((d / sub).glob("w[0-9]*_[0-9]*.npz"), key=lambda f: (f.stem.split("_")[1], f.stem))
        ps = chunks[:max_files] if max_files and chunks else [p] if p.exists() else chunks
        for f in ps:
            z = np.load(f)
            n = z["n_steps"].astype(int)
            so = np.r_[0, np.cumsum(n)[:-1]]
            r0 = so + z["k0"].astype(int)
            S, W = z["S"], z["W"]
            seg = {k: z[k] for k in keys if k in z.files}
            seg["S0"], seg["S1"], seg["W0"] = S[r0].astype(np.float64), S[r0 - 1].astype(np.float64), W[r0].astype(np.float64)
            seg["both"] = np.full(len(n), sub == "data_both")
            out.append(seg)
    if not out:
        return None
    return {k: np.concatenate([o[k] for o in out]) for k in out[0] if all(k in o for o in out)}


def stats(D: dict, m: np.ndarray) -> dict:
    S0, S1, W0 = D["S0"][m], D["S1"][m], D["W0"][m]
    q = S0[:, Q]
    med = np.median(q, axis=0)
    dev = np.degrees(np.abs(q - med))
    dq = np.linalg.norm((S0[:, Q] - S1[:, Q]) / 0.1, axis=1)
    g = S0[:, GRAV] / np.linalg.norm(S0[:, GRAV], axis=1, keepdims=True)
    tilt = np.degrees(np.arccos(np.clip(-g[:, 2], -1, 1)))
    yaw = W0[:, 3]
    right = D["side"][m] > 0
    hp = np.where(right[:, None], W0[:, 4:7], W0[:, 7:10]) - W0[:, :3]
    c, s = np.cos(yaw), np.sin(yaw)
    hh = np.c_[c * hp[:, 0] + s * hp[:, 1], -s * hp[:, 0] + c * hp[:, 1], hp[:, 2]]
    hh[:, 1] *= np.where(right, 1.0, -1.0)  # left mirrored
    uniq = len(np.unique(np.round(q, 4), axis=0))
    pq = lambda v: [round(float(np.median(v)), 4), round(float(np.percentile(v, 90)), 4)]  # noqa: E731
    return dict(
        n=int(m.sum()), unique=uniq,
        joint_std_deg=np.round(np.degrees(q.std(0)), 3).tolist(),
        joint_std_deg_median=round(float(np.median(np.degrees(q.std(0)))), 3),
        dev_max_deg=pq(dev.max(1)), dev_rms_deg=pq(np.sqrt((dev**2).mean(1))),
        dq_fd_norm=pq(dq), dq_read_norm=pq(np.linalg.norm(S0[:, 29:58], axis=1)),
        tilt_deg=[round(float(np.median(tilt)), 3), round(float(tilt.std()), 3)],
        height_m=[round(float(np.median(S0[:, H])), 4), round(float(S0[:, H].std()), 4)],
        hand_std_cm=np.round(100 * hh.std(0), 2).tolist(),
        q_median=np.round(med, 4).tolist(),
    )  # fmt: skip


def first_mask(D: dict) -> np.ndarray:
    m = D["chain"] == 0
    if "hands" in D:  # a two-hand reach is two segments (one per hand): keep the right one
        m &= ~((D["hands"] == 2) & (D["side"] < 0))
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="NAME=DIR[,DIR...]")
    ap.add_argument("--json", default=None)
    ap.add_argument("--max-files", type=int, default=0, help="read at most this many chunk files per sub-directory")
    a = ap.parse_args()
    res = {}
    for spec in a.specs:
        name, dirs = spec.split("=", 1)
        parts = [load(Path(d), a.max_files) for d in dirs.split(",")]
        parts = [p for p in parts if p is not None]
        if not parts:
            print(f"{name}: no records", file=sys.stderr)
            continue
        D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if all(k in p for p in parts)}
        res[name] = {}
        for g, wk in (("in place", 0), ("walk first", 1)):
            m = first_mask(D) & (D["walked"] == wk)
            if m.sum() < 2:
                continue
            st = stats(D, m)
            res[name][g] = st
            print(f"{name:24s} {g:10s} n {st['n']:6d} unique {st['unique']:6d} | joint std median {st['joint_std_deg_median']:.2f} deg "
                  f"| dev from median: max-joint {st['dev_max_deg'][0]:.2f} / p90 {st['dev_max_deg'][1]:.2f} deg, rms "
                  f"{st['dev_rms_deg'][0]:.2f} / {st['dev_rms_deg'][1]:.2f} | fd dq norm {st['dq_fd_norm'][0]:.3f} / "
                  f"{st['dq_fd_norm'][1]:.3f} rad/s | tilt {st['tilt_deg'][0]:.2f} (std {st['tilt_deg'][1]:.2f}) deg | z "
                  f"{st['height_m'][0]:.3f} (std {100 * st['height_m'][1]:.2f} cm) | hand std {st['hand_std_cm']} cm")  # fmt: skip
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    names = list(res)
    if len(names) >= 2:
        print("per-joint std (deg) of the in-place first reach, joint by joint (S order):")
        for nm in names:
            if "in place" in res[nm]:
                print(f"  {nm:24s} " + " ".join(f"{v:5.2f}" for v in res[nm]["in place"]["joint_std_deg"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
