"""selfmodel_train3b_20261125: model D's extra training data for one controller -- mid-trajectory starts cut from the
train split of its one-hand segments in selfmodel_data3_20261125 and selfmodel_data3b_20261125 (data/all.npz), in their
npz layout so reachwm_20261030/data.py reads them unchanged.

The cut is selfmodel_relabel_20261125's (mid_data.py, frozen there): a new start every 0.2 s from 0.3 s after the
command to 0.3 s before the segment's last row (rows 0.1 s apart, k0 = 3); a start at row k has k0 = 2 and the rows
k-2, k-1, k as its S[k0-2..k0], so x0 holds the true states there, and the label is the source's last row. start = the
pelvis (x, y, z, yaw) of row k (W); target and rot6 = the source's world command in that heading frame; ref_q and
ref_pel (world) unchanged, so data.py re-expresses the plan's end pose and recomputes "plan joints minus start joints"
from S[k]; ref_s = ref_s minus the time since the command at row k (C[:, 3]), floored at 0. Every other per-segment
array is copied from the source. Starts that data.splits would not put in train after the frame change (band_x or the
edge set on the new target; the other bands and the reset bucket do not change) are dropped and counted.

Differences from mid_data.py, all from the data3 layout or from what D is:
  - sources: every one-hand segment data.splits puts in train, fallen or lost ones excluded (data3's reaches, data3b's
    pre-grasps, entries and relative reaches). relabel's model A left the entries out because it was the no-entry model;
    D is M's data plus the starts.
  - per-step arrays copied: S, W, J, C. HQ and KP (per step) are not read by data.py and are dropped; RT (n_ref rows,
    not n_steps) is dropped. data.load keeps only the arrays every file has, so D's data3 / data3b segments lose them too.
  - storage: a start keeps the rows k-2, k-1, k and the source's last 6 rows (all of k-2 .. last when they overlap)
    instead of every row k-2 .. last. data.py reads S / W / J only at k0-2 .. k0 and at the last row, the jaw over the
    last 6 rows (settled) and, for the step models' odometry and padding only, the rows between; E3 trains on
    per-segment arrays. For a sample of starts the full form is built too and its derived x0, u and labels are compared
    byte for byte (checks.compact_*); any difference fails the run.

    cd <repo>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 $PY t3b_mid.py --backend heft --out <dir>
        [--data3 <all.npz>] [--data3b <all.npz>]    (default: the formal collections' data/all.npz)
writes <out>/mid_data3.npz, <out>/mid_data3b.npz and <out>/mid.json; exits 1 if a check fails.
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

ROOT = Path("/wuji-vefps-D/wuji-il/yijie/loco_outputs")
FIRST, EVERY, TAIL = 3, 2, 3  # in rows of 0.1 s: 0.3 s after the command, every 0.2 s, until 0.3 s before the last row
PER_STEP = ("S", "W", "J", "C")
DROP_STEP = ("HQ", "KP")  # per step, not read by data.py
DROP_REF = ("RT",)  # n_ref rows
LAST = 6  # rows the settled flag reads (data.derive: the jaw speed over the last 5 intervals)
N_FULL_CHECK = 3000
X_ARRAYS = ("x0", "u", "pos_f", "rot_f", "settled", "fell", "err_cm", "target", "rot6", "jaw0", "rot0", "pel_f", "q_f")


def row_index(a: np.ndarray, length: np.ndarray) -> np.ndarray:
    """Concatenated ranges a[j] .. a[j] + length[j] - 1."""
    off = np.repeat(np.cumsum(length) - length, length)
    return np.arange(int(length.sum())) - off + np.repeat(a, length)


def rz_t(yaw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rz(yaw)^T v for rows v (N, 2)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]], -1)


def rz(yaw: np.ndarray, v: np.ndarray) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], -1)


def derived(z: dict) -> dict:
    """data.load's view of an npz's arrays: its REACHWM_APPROACH defaults (no walk: zeros; target_b = target, so a mid
    start's bands are judged on its own target), then data.derive."""
    d = dict(z)
    if DT.APPROACH:
        d.setdefault("walk", np.zeros((len(d["n_steps"]), 3), np.float32))
        d.setdefault("target_b", d["target"])
    DT.derive(d)
    return d


def starts(z: dict, src: np.ndarray, kl: np.ndarray, rows: np.ndarray, length: np.ndarray) -> dict:
    """The mid starts (source segment src, start row kl within it) with the given per-step rows (global) and lengths."""
    n = z["n_steps"].astype(int)
    so = np.r_[0, np.cumsum(n)[:-1]]
    g = so[src] + kl
    zm = {k: v[rows] for k, v in z.items() if k in PER_STEP}
    zm.update({k: v[src] for k, v in z.items() if k not in PER_STEP + DROP_STEP + DROP_REF})
    zm["n_steps"] = length.astype(z["n_steps"].dtype)
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
    return zm


def build(name: str, path: Path, out: Path, seed: int) -> tuple[dict, bool]:
    t0 = time.time()
    z = dict(np.load(path))
    N = len(z["n_steps"])
    n = z["n_steps"].astype(int)
    nr = int(z["n_ref"].sum()) if "n_ref" in z else -1
    for k, v in z.items():
        want = int(n.sum()) if k in PER_STEP + DROP_STEP else (nr if k in DROP_REF else N)
        assert len(v) == want, f"{name}: {k} has {len(v)} rows, expected {want}"
    assert "walk" not in z and "target_b" not in z, "a walk / target_b array would be copied unchanged into the starts"
    so = np.r_[0, np.cumsum(n)[:-1]]
    D = derived(z)
    sp = DT.splits(D)
    train = np.zeros(N, bool)
    train[sp["train"]] = True
    held = D["band_x"] | D["band_z"] | D["band_yaw"] | D["band_pel"] | D["edge"]
    k0 = z["k0"].astype(int)
    assert (k0 == 3).all(), "expected k0 = 3 (PRE_S 0.3 s at 0.1 s rows)"
    fell = z["fell"].astype(bool)
    tr_reset = D["bucket"] < 8
    info = dict(name=name, source=str(path), segments=N, resets=int(len(np.unique(z["reset"]))), train_split=int(train.sum()),
                val=len(sp["val"]), test_iid=len(sp["test_iid"]))  # fmt: skip
    cand = np.where(train & ~fell)[0]
    info.update(mid_source_segments=int(len(cand)), mid_source_excluded_fell=int((train & fell).sum()),
                mid_source_excluded_held=int((tr_reset & held).sum()))  # fmt: skip
    if "kind" in z:
        info["mid_source_by_kind"] = {int(k): int((z["kind"][cand] == k).sum()) for k in np.unique(z["kind"][cand])}
    src, kl = [], []
    for i in cand:
        ks = np.arange(k0[i] + FIRST, n[i] - 1 - TAIL + 1, EVERY)
        src.append(np.full(len(ks), i))
        kl.append(ks)
    src, kl = np.concatenate(src), np.concatenate(kl)
    g = so[src] + kl
    gl = so[src] + n[src] - 1
    # compact rows: k-2, k-1, k, then the last LAST rows (from k+1 when they overlap)
    t1 = np.maximum(g + 1, gl - (LAST - 1))
    length = 3 + (gl - t1 + 1)
    off = np.repeat(np.cumsum(length) - length, length)
    pos = np.arange(int(length.sum())) - off
    gj, tj = np.repeat(g, length), np.repeat(t1, length)
    rows = np.where(pos < 3, gj - 2 + pos, tj + pos - 3)
    zm = starts(z, src, kl, rows, length)

    # checks against the source segments, then keep only the starts data.splits puts in train
    Dm = derived(zm)
    e = np.abs(Dm["err_cm"] - D["err_cm"][src])
    checks = dict(
        err_cm_invariant_max_abs=float(e.max()),  # |delivered - command| does not depend on the frame
        x0_S_equal=bool(np.array_equal(Dm["x0"][:, :65], z["S"][g])),
        x0_S_prev_equal=bool(np.array_equal(Dm["x0"][:, 65:130], z["S"][g - 1])),
        x0_S_prev2_equal=bool(np.array_equal(Dm["x0"][:, 130:195], z["S"][g - 2])),
        settled_equal=bool(np.array_equal(Dm["settled"], D["settled"][src])),
        fell_equal=bool(np.array_equal(Dm["fell"], D["fell"][src])),
        rot6_unit_max_dev=float(np.abs(np.linalg.norm(Dm["rot6"][:, :3], axis=1) - 1).max()),
    )  # fmt: skip
    # the compact rows against the full form (every row k-2 .. last), on a sample
    rng = np.random.default_rng(seed)
    js = np.sort(rng.choice(len(src), size=min(N_FULL_CHECK, len(src)), replace=False))
    lf = gl[js] - g[js] + 3
    zf = starts(z, src[js], kl[js], row_index(g[js] - 2, lf), lf)
    Df = derived(zf)
    sel = {k: v[js] for k, v in zm.items() if k not in PER_STEP}
    so_c = np.r_[0, np.cumsum(length)[:-1]]
    rc = row_index(so_c[js], length[js])
    for k in PER_STEP:
        sel[k] = zm[k][rc]
    sel = derived(sel)
    for k in X_ARRAYS:
        checks[f"compact_{k}_equal"] = bool(np.array_equal(np.asarray(sel[k]), np.asarray(Df[k])))
    checks["compact_checked"] = int(len(js))
    checks["compact_rows_full_form_kept"] = int((length == gl - g + 3).sum())
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
        mid_kept_ref_s_zero=int((intr & (zm["ref_s"] == 0)).sum()),
        mid_kept_rows=int(zm["n_steps"][intr].sum()),
        mid_kept_rows_full_form=int((gl - g + 3)[intr].sum()),
        checks=checks,
    )
    if "kind" in z:
        info["mid_kept_by_kind"] = {int(k): int((z["kind"][src][intr] == k).sum()) for k in np.unique(z["kind"][src])}
    W = z["W"]
    right = z["side"][src] > 0
    jp = lambda r: np.where(right[:, None], W[r, 4:7], W[r, 7:10])  # noqa: E731
    v = 100 * np.linalg.norm(jp(g) - jp(g - 1), axis=1) / 0.1
    rem = 100 * np.linalg.norm(Dm["pos_f"] - Dm["jaw0"], axis=1)
    info.update(mid_kept_start_jaw_speed_cm_s_median=float(np.median(v[intr])),
                mid_kept_start_jaw_speed_cm_s_p90=float(np.percentile(v[intr], 90)),
                mid_kept_start_still_frac=float((v[intr] < 2).mean()),
                mid_kept_remaining_cm_median=float(np.median(rem[intr])))  # fmt: skip
    ki = np.where(intr)[0]
    zk = {k: v_[ki] for k, v_ in zm.items() if k not in PER_STEP}
    rk = row_index(so_c[ki], length[ki])
    for k in PER_STEP:
        zk[k] = zm[k][rk]
    tmp = out / f".mid_{name}.tmp.npz"
    np.savez(tmp, **zk)
    os.replace(tmp, out / f"mid_{name}.npz")
    info["seconds"] = round(time.time() - t0, 1)
    ok = (checks["err_cm_invariant_max_abs"] < 1e-3 and checks["x0_S_equal"] and checks["x0_S_prev_equal"]
          and checks["x0_S_prev2_equal"] and checks["settled_equal"] and checks["fell_equal"]
          and checks["rot6_unit_max_dev"] < 1e-5 and all(checks[f"compact_{k}_equal"] for k in X_ARRAYS))  # fmt: skip
    info["checks_ok"] = bool(ok)
    return info, ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["scalebfm", "heft", "sonic"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data3", type=Path, default=None)
    ap.add_argument("--data3b", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    src = dict(data3=a.data3 or ROOT / "selfmodel_data3_20261125" / a.backend / "data" / "all.npz",
               data3b=a.data3b or ROOT / "selfmodel_data3b_20261125" / a.backend / "data" / "all.npz")  # fmt: skip
    infos, ok = [], True
    for name, path in src.items():
        info, good = build(name, path, a.out, a.seed)
        print(json.dumps(info), flush=True)
        infos.append(info)
        ok &= good
    json.dump(infos, open(a.out / "mid.json", "w"), indent=1)
    print("MID CHECKS", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
