"""selfmodel_relabel_20261125: the training sets of models A and C from ScaleBFM's grasp-room data (data_graspr_odo,
data_graspr_wrist_odo of simple_selfmodel_20261119), in gen_reach.py's npz layout so reachwm_20261030/data.py reads them
unchanged. B trains on the two all.npz as they are.

Each reset of these datasets is a walk, then three reaches in this order: the waypoint (chain 0), the pre-grasp (chain 1)
and the slow entry (chain 1, 0.08 m/s, its reference from the midpoint). The entry is told apart by its place in the reset.

    C_<name>.npz    the original segments without the entries that data.splits puts in train (val, test_iid, the bands
                    and the edge set keep every segment, so the three models see the same ones there)
    mid_<name>.npz  mid-trajectory starts cut from the train split's waypoint and pre-grasp segments (fallen ones excluded):
                    a new start every 0.2 s from 0.3 s after the command to 0.3 s before the segment's last row (rows are
                    0.1 s apart). A start at row k keeps rows k-2 .. last, k0 = 2, so x0's S[k0], S[k0-1], S[k0-2] are
                    the true states there and the label (last row) is the original one. start = the pelvis (x, y, z, yaw)
                    of row k (W); target and rot6 = the original world command in that heading frame; ref_q and ref_pel
                    (world) unchanged, so data.py re-expresses the plan's end pose and recomputes "plan joints minus start
                    joints" from S[k]; ref_s = ref_s minus the time since the command at row k (C[:, 3]), floored at 0.
                    Every other array is copied from the source segment. Starts that data.splits would not put in train
                    after the frame change (band_x or edge on the new target) are dropped and counted.
    midmove_<name>.npz  exploratory, not in the registration: the starts of mid_<name>.npz at which the reference still
                    runs (time since the command < ref_s); model Am trains on C plus these.

    cd <repo>; REACHWM_GOAL_REF=1 $PY mid_data.py --out <dir> [--max-resets N]   # N: smoke test, also writes B_<name>.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("LOCO_REPO", "/wuji-vefps-D/wuji-il/yijie/ws/code/loco"))
sys.path.insert(0, str(REPO / "scripts" / "experiments" / "reachwm_20261030"))
import data as DT  # noqa: E402

SRC = Path("/wuji-vefps-D/wuji-il/yijie/loco_outputs/simple_selfmodel_20261119")
NAMES = ("graspr_odo", "graspr_wrist_odo")
FIRST, EVERY, TAIL = 3, 2, 3  # in rows of 0.1 s: 0.3 s after the command, every 0.2 s, until 0.3 s before the last row
PER_STEP = ("S", "W", "J", "C")
ENTRY = 2  # place of the entry in its reset


def src_path(name: str) -> Path:
    return SRC / f"data_{name}" / "all.npz"


def ordinal(reset: np.ndarray) -> np.ndarray:
    """Place of each segment within its reset; a reset's segments are contiguous (checked)."""
    new = np.r_[True, reset[1:] != reset[:-1]]
    assert new.sum() == len(np.unique(reset)), "a reset's segments are not contiguous"
    i = np.arange(len(reset))
    return i - np.maximum.accumulate(np.where(new, i, 0))


def row_index(a: np.ndarray, length: np.ndarray) -> np.ndarray:
    """Concatenated ranges a[j] .. a[j] + length[j] - 1."""
    off = np.repeat(np.cumsum(length) - length, length)
    return np.arange(int(length.sum())) - off + np.repeat(a, length)


def take(z: dict, seg: np.ndarray, a: np.ndarray, length: np.ndarray) -> dict:
    """Segments seg (per-segment arrays), their per-step rows a .. a + length - 1."""
    rows = row_index(a, length)
    out = {k: (v[rows] if k in PER_STEP else v[seg]) for k, v in z.items()}
    out["n_steps"] = length.astype(z["n_steps"].dtype)
    return out


def rz_t(yaw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rz(yaw)^T v for rows v (N, 2)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]], -1)


def rz(yaw: np.ndarray, v: np.ndarray) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], -1)


def build(name: str, out: Path, max_resets: int | None) -> dict:
    t0 = time.time()
    z = dict(np.load(src_path(name)))
    N = len(z["n_steps"])
    assert all(len(v) == (int(z["n_steps"].sum()) if k in PER_STEP else N) for k, v in z.items())
    n = z["n_steps"].astype(int)
    so = np.r_[0, np.cumsum(n)[:-1]]
    if max_resets:  # smoke test: the first resets only
        keep = np.isin(z["reset"], np.unique(z["reset"])[:max_resets])
        z = take(z, np.where(keep)[0], so[keep], n[keep])
        N, n = len(z["n_steps"]), z["n_steps"].astype(int)
        so = np.r_[0, np.cumsum(n)[:-1]]
        np.savez(out / f"B_{name}.npz", **z)  # the smoke test's B
    D = dict(z)
    DT.derive(D)
    sp = DT.splits(D)
    train = np.zeros(N, bool)
    train[sp["train"]] = True
    held = D["band_x"] | D["band_z"] | D["band_yaw"] | D["band_pel"] | D["edge"]
    o = ordinal(z["reset"])
    k0 = z["k0"].astype(int)
    assert (k0 == 3).all(), "expected k0 = 3 (PRE_S 0.3 s at 0.1 s rows)"
    tr_reset = D["bucket"] < 8
    info = dict(name=name, segments=N, resets=int(len(np.unique(z["reset"]))),
                per_place={int(p): int((o == p).sum()) for p in np.unique(o)},
                train_split=int(train.sum()), val=len(sp["val"]), test_iid=len(sp["test_iid"]))  # fmt: skip

    # C: drop the train split's entries
    drop = train & (o == ENTRY)
    keepc = ~drop
    zc = take(z, np.where(keepc)[0], so[keepc], n[keepc])
    info.update(C_dropped_entries=int(drop.sum()), C_segments=int(keepc.sum()),
                train_reset_entries_held_kept=int((tr_reset & (o == ENTRY) & held).sum()),
                train_reset_entries_fell=int((drop & z["fell"].astype(bool)).sum()))  # fmt: skip
    np.savez(out / f"C_{name}.npz", **zc)

    # A's extra: mid-trajectory starts
    cand = np.where(train & (o < ENTRY) & ~z["fell"].astype(bool))[0]
    info.update(mid_source_segments=int(len(cand)),
                mid_source_excluded_held=int((tr_reset & (o < ENTRY) & held).sum()),
                mid_source_excluded_fell=int((train & (o < ENTRY) & z["fell"].astype(bool)).sum()))  # fmt: skip
    src, kl = [], []
    for i in cand:
        ks = np.arange(k0[i] + FIRST, n[i] - 1 - TAIL + 1, EVERY)
        src.append(np.full(len(ks), i))
        kl.append(ks)
    src, kl = np.concatenate(src), np.concatenate(kl)
    g = so[src] + kl  # global row of the new start
    zm = take(z, src, g - 2, n[src] - kl + 2)
    zm["k0"] = np.full(len(src), 2, z["k0"].dtype)
    st0 = z["start"][src].astype(np.float64)
    st1 = z["W"][g, :4].astype(np.float64)
    zm["start"] = st1.astype(z["start"].dtype)
    tg = z["target"][src].astype(np.float64)
    tw = st0[:, :2] + rz(st0[:, 3], tg[:, :2])
    zm["target"] = np.c_[rz_t(st1[:, 3], tw - st1[:, :2]), tg[:, 2]].astype(z["target"].dtype)
    r6 = z["rot6"][src].astype(np.float64)
    dy = st0[:, 3] - st1[:, 3]
    c0 = np.c_[rz(dy, r6[:, 0:2]), r6[:, 2]]
    c1 = np.c_[rz(dy, r6[:, 3:5]), r6[:, 5]]
    zm["rot6"] = np.c_[c0, c1].astype(z["rot6"].dtype)
    tc = z["C"][g, 3].astype(np.float64)
    assert (tc > 0).all()
    zm["ref_s"] = np.maximum(z["ref_s"][src].astype(np.float64) - tc, 0.0).astype(z["ref_s"].dtype)

    # checks against the source segments, then keep only the starts data.splits puts in train
    Dm = dict(zm)
    DT.derive(Dm)
    e = np.abs(Dm["err_cm"] - D["err_cm"][src])
    ang_ok = np.abs(np.linalg.norm(Dm["rot6"][:, :3], axis=1) - 1).max()
    checks = dict(
        err_cm_invariant_max_abs=float(e.max()),  # |delivered - command| does not depend on the frame
        x0_S_equal=bool(np.array_equal(Dm["x0"][:, :65], z["S"][g])),
        x0_S_prev_equal=bool(np.array_equal(Dm["x0"][:, 65:130], z["S"][g - 1])),
        x0_S_prev2_equal=bool(np.array_equal(Dm["x0"][:, 130:195], z["S"][g - 2])),
        settled_equal=bool(np.array_equal(Dm["settled"], D["settled"][src])),
        fell_equal=bool(np.array_equal(Dm["fell"], D["fell"][src])),
        rot6_unit_max_dev=float(ang_ok),
    )  # fmt: skip
    spm = DT.splits(Dm)
    intr = np.zeros(len(src), bool)
    intr[spm["train"]] = True
    heldm = Dm["band_x"] | Dm["band_z"] | Dm["band_yaw"] | Dm["band_pel"] | Dm["edge"]
    info.update(
        mid_starts=int(len(src)),
        mid_dropped_not_train=int((~intr).sum()),
        mid_dropped_band_x=int((~intr & Dm["band_x"]).sum()),
        mid_dropped_edge=int((~intr & Dm["edge"]).sum()),
        mid_dropped_other_held=int((~intr & heldm & ~Dm["band_x"] & ~Dm["edge"]).sum()),
        mid_kept=int(intr.sum()),
        mid_kept_from_waypoint=int((intr & (o[src] == 0)).sum()),
        mid_kept_from_pregrasp=int((intr & (o[src] == 1)).sum()),
        mid_kept_ref_s_zero=int((intr & (zm["ref_s"] == 0)).sum()),
        mid_kept_rows=int(zm["n_steps"][intr].sum()),
        mid_kept_start_jaw_speed_cm_s_median=None,
        checks=checks,
    )
    # how fast the acting jaw moves at the kept starts (row k-1 -> k), for the record
    W = z["W"]
    right = z["side"][src] > 0
    jp = lambda r: np.where(right[:, None], W[r, 4:7], W[r, 7:10])  # noqa: E731
    v = 100 * np.linalg.norm(jp(g) - jp(g - 1), axis=1) / 0.1
    info["mid_kept_start_jaw_speed_cm_s_median"] = float(np.median(v[intr]))
    info["mid_kept_start_jaw_speed_cm_s_p90"] = float(np.percentile(v[intr], 90))
    info["mid_kept_start_still_frac"] = float((v[intr] < 2).mean())
    rem = 100 * np.linalg.norm(Dm["pos_f"] - Dm["jaw0"], axis=1)
    info["mid_kept_remaining_cm_median"] = float(np.median(rem[intr]))
    som = np.r_[0, np.cumsum(zm["n_steps"].astype(int))[:-1]]

    def save(path, sel):
        ki = np.where(sel)[0]
        zk = {k: (v_ if k in PER_STEP else v_[ki]) for k, v_ in zm.items()}
        rows = row_index(som[ki], zm["n_steps"][ki].astype(int))  # rows of the kept starts only
        for k in PER_STEP:
            zk[k] = zm[k][rows]
        np.savez(path, **zk)

    save(out / f"mid_{name}.npz", intr)
    # exploratory (not registered): the kept starts while the reference still runs (time since the command < ref_s)
    run = intr & (z["ref_s"][src].astype(np.float64) - tc > 0)
    info.update(midmove_kept=int(run.sum()), midmove_rows=int(zm["n_steps"][run].sum()),
                midmove_start_jaw_speed_cm_s_median=float(np.median(v[run])),
                midmove_remaining_cm_median=float(np.median(rem[run])))  # fmt: skip
    save(out / f"midmove_{name}.npz", run)
    info["seconds"] = round(time.time() - t0, 1)
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-resets", type=int, default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    infos = []
    for name in NAMES:
        info = build(name, out, a.max_resets)
        print(json.dumps(info), flush=True)
        infos.append(info)
    json.dump(infos, open(out / "mid_data.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
