"""selfmodel_nowalk_20261126 phase 2 (LocoManiLab, one controller): nw_split.py's arms N, C, C2 built from train3b's D
data of that controller (selfmodel_data3_20261125/<be>, selfmodel_data3b_20261125/<be>, train3b's two mid files; D's
switches: REACHWM_GOAL_REF / APPROACH / QFIX = 1, FDVEL off); read-only, writes <out>/splits.npz, splits.json.

    cd <repo>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 LOCO_REPO=<repo> CUDA_VISIBLE_DEVICES= \
        $PY nw2_split.py --backend <scalebfm|sonic|heft> --out <dir>

Differences from nw_split.py (everything else is the same code): the data paths; D's training count from train3b's
logs/train_D_s0_f1.log; train3b's t3b_mid wrote no mid_*_src.npz, so a mid segment is checked by its reset id (one of
its source file's) and its walked flag against the logs; the input check reads lmlreal's records of the controller (not
fallen / lost, settled end: train3b's lmlreal metric), walked and in-place segments apart.
Walk-first resets: the collection logs (data3/data, data3b/data: w*_log.json, one entry per reset), `walked`. Checked
against every segment's walked flag (data3, data3b and the mid files).
Arms (each a set of resets taken out of the training process: their segments get bucket 10, so data.splits leaves
them out of train, val and test_iid; bands / edge are never trained on):
  N   every walk-first reset (all buckets): N trains on in-place resets only and picks its checkpoint on the in-place
      val (bucket 8, not walked).
  C   the walk-first resets of bucket 8 (the same in-place val as N) plus whole training resets (bucket < 8, any type)
      dropped in the order of md5("nowalk_c:<reset>"), the smallest digests kept first, until the kept training
      segments reach N's count (the first prefix with at least as many); |C - N| / N <= 1 % is asserted.
  C2  C's resets.
Zeroed inputs (nw_train / nw_eval, arms N and C2): x0 column 204 (data.py's `walked`, checked equal to the flag); the
four REACHWM_APPROACH columns 274-277 are zero in every file here (no file has a `walk` array, data.load fills zeros;
checked).
Input check before any training (no model): the 39 start quantities of sd_range (joints, pelvis, hands at S[k0]) of the
real main-label settled segments inside each arm's training p0.5 / p99.5, and the x0 columns whose training standard
deviation is below Norm's floor (1e-3) with the largest |z| a real segment gets under that arm's normalisation.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("LOCO_REPO", "/wuji-vefps-D/wuji-il/yijie/ws/code/loco"))
EXP = REPO / "scripts" / "experiments"
sys.path[:0] = [str(EXP / "reachwm_20261030"), str(EXP / "selfwm_20261029")]
import data as DT  # noqa: E402

for k in ("REACHWM_GOAL_REF", "REACHWM_APPROACH", "REACHWM_QFIX"):
    assert os.environ.get(k) == "1", f"{k}=1 is train3b D's switch"
assert os.environ.get("REACHWM_FDVEL", "0") != "1" and os.environ.get("REACHWM_X0_REF", "0") != "1", "train3b's D: FDVEL / X0_REF off"
assert DT.X0_DIM == 278, DT.X0_DIM
WALKED_COL, CHAIN_COL, APPROACH_COLS = 204, 205, slice(274, 278)
TOL = 0.01
L = Path("/wuji-vefps-D/wuji-il/yijie/loco_outputs")


def md5_key(r: int) -> int:
    return int(hashlib.md5(f"nowalk_c:{int(r)}".encode()).hexdigest(), 16)


def log_walked(d: Path) -> dict[int, bool]:
    out = {}
    for f in sorted(glob.glob(str(d / "w*_log.json"))):
        for e in json.load(open(f)):
            assert e["reset"] not in out, f"reset {e['reset']} twice in the logs"
            assert bool(e["walk_asked"]) == bool(e["walked"]), e["reset"]
            out[int(e["reset"])] = bool(e["walked"])
    return out


def quantities(D: dict, idx: np.ndarray) -> dict:
    """sd_range.quantities (simpledata, 1276163a), verbatim: the 39 range quantities at the start row."""
    r = np.asarray(D["so"])[idx] + np.asarray(D["k0"])[idx].astype(int)
    S, W = np.asarray(D["S"]), np.asarray(D["W"])
    st = np.asarray(D["start"])[idx].astype(np.float64)
    yaw, org = st[:, 3], np.c_[st[:, :2], np.zeros(len(idx))]
    hands = [DT.hv(W[r, a:b].astype(np.float64) - org, yaw) for a, b in ((4, 7), (7, 10))]
    return dict(joints=S[r, 0:29].astype(np.float64), pelvis=np.c_[S[r, 64], S[r, 58:61]].astype(np.float64), hands=np.concatenate(hands, 1))


def apply(D: dict, drop: np.ndarray, zero: bool) -> None:
    """An arm's view of D (in place): dropped resets' segments to bucket 10, the walk flag column zeroed."""
    D["bucket"] = np.where(np.isin(np.asarray(D["reset"]).astype(np.int64), drop), 10, D["bucket"])
    if zero:
        D["x0"][:, WALKED_COL] = 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["scalebfm", "sonic", "heft"])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    be = a.backend
    SR = argparse.Namespace(quantities=quantities)  # sd_range's, copied above (sd_range asserts REACHWM_FDVEL=1 on import)

    a.out.mkdir(parents=True, exist_ok=True)
    T = L / "selfmodel_train3b_20261125" / be
    files = [L / f"selfmodel_data3_20261125/{be}/data/all.npz", L / f"selfmodel_data3b_20261125/{be}/data/all.npz",
             T / "data/mid_data3.npz", T / "data/mid_data3b.npz"]
    tl = (T / "logs/train_D_s0_f1.log").read_text()
    D_N_TRAIN = int(tl.split("'train': ")[1].split(",")[0])  # train3b's D: "<n> segments, splits {'train': <n>, ..."
    names = ["data3", "data3b", "mid_data3", "mid_data3b"]
    spec = ",".join(map(str, files))
    md5s = {n: hashlib.md5(open(p, "rb").read()).hexdigest()[:8] for n, p in zip(names, files)}
    print(f"spec {spec}\n  md5 {md5s}")
    lw = {**log_walked(files[0].parent), **log_walked(files[1].parent)}
    wf_ids = np.array(sorted(r for r, w in lw.items() if w), np.int64)
    print(f"logs: {len(lw)} resets, walk-first {len(wf_ids)}")
    lens = [len(np.load(p)["n_steps"]) for p in files]
    fid = np.repeat(np.arange(4), lens)
    D = DT.load(spec)
    N = len(D["n_steps"])
    assert N == sum(lens), (N, lens)
    reset = np.asarray(D["reset"]).astype(np.int64)
    walked = np.asarray(D["walked"]).astype(bool)
    chain = np.asarray(D["chain"]).astype(int)
    x0 = D["x0"]
    fails = []

    def check(ok: bool, what: str) -> None:
        print(f"  [{'ok' if ok else 'FAIL'}] {what}")
        if not ok:
            fails.append(what)

    print("checks:")
    check(all(int(r) in lw for r in np.unique(reset)), "every segment's reset is in the collection logs")
    lwa = np.array([lw[int(r)] for r in reset])
    check(bool((lwa == walked).all()), f"segment walked == the log's walked, all {N} segments (data3, data3b, mid)")
    for k, nm in ((2, "mid_data3"), (3, "mid_data3b")):
        m0 = np.flatnonzero(fid == k)
        src_r = set(np.unique(reset[fid == k - 2]).tolist())
        check(set(np.unique(reset[m0]).tolist()) <= src_r, f"{nm}: every reset id is one of its source file's")
    check(bool((x0[:, WALKED_COL] == walked.astype(np.float32)).all()), "x0 column 204 == walked")
    check(bool((x0[:, CHAIN_COL] == chain.astype(np.float32)).all()), "x0 column 205 == chain")
    check(bool((x0[:, APPROACH_COLS] == 0).all()), "x0 columns 274-277 (REACHWM_APPROACH) zero everywhere")
    sp = DT.splits(D)
    tr = sp["train"]
    check(len(tr) == D_N_TRAIN, f"D's training split {len(tr)} == {D_N_TRAIN} (train3b logs/train_D_s0_f1.log)")
    bucket0 = np.asarray(D["bucket"]).copy()
    per_reset_bucket = {}
    for r, b in zip(reset, bucket0):
        assert per_reset_bucket.setdefault(int(r), int(b)) == int(b)
    is_wf = np.isin(reset, wf_ids)
    # N
    drop_N = wf_ids
    # C: the in-place val as N, then whole training resets in md5 order
    tr_resets, tr_counts = np.unique(reset[tr], return_counts=True)
    n_N = int((~is_wf[tr]).sum())
    order = np.argsort([md5_key(r) for r in tr_resets], kind="stable")
    cum = np.cumsum(tr_counts[order])
    K = int(np.searchsorted(cum, n_N)) + 1  # the first prefix with cum >= n_N
    keep_C = np.sort(tr_resets[order[:K]])
    val_wf = np.array(sorted(r for r in wf_ids if per_reset_bucket.get(int(r), -1) == 8), np.int64)
    drop_C = np.union1d(np.setdiff1d(tr_resets, keep_C), val_wf)
    arms = {"D": (np.zeros(0, np.int64), False), "N": (drop_N, True), "C": (drop_C, False), "C2": (drop_C, True)}
    info = dict(backend=be, spec=spec, md5=md5s, n_segments=N, walk_first_resets=int(len(wf_ids)), log_resets=len(lw), arms={})
    print("\narms (training split after the arm's drop; val = data.splits' val, fell excluded as train.train does):")
    for arm, (drop, zero) in arms.items():
        b = np.where(np.isin(reset, drop), 10, bucket0)
        D["bucket"] = b
        s = DT.splits(D)
        t, v = s["train"], s["val"]
        v = v[np.asarray(D["fell"])[v] == 0]
        rs = np.unique(reset[t])
        wf_seg, wf_res = int(is_wf[t].sum()), int(np.isin(rs, wf_ids).sum())
        per_file = {names[k]: int((fid[t] == k).sum()) for k in range(4)}
        wt = np.asarray(D["walk_task"])[t] if "walk_task" in D else np.zeros(len(t), int)  # LML: every walk is a d3 walk
        o = dict(n_train=int(len(t)), train_resets=int(len(rs)), train_walk_first_segments=wf_seg, train_walk_first_resets=wf_res,
                 train_walk_first_task=int((is_wf[t] & (wt == 1)).sum()), train_walk_first_d3=int((is_wf[t] & (wt == 0)).sum()),
                 train_first=int((chain[t] == 0).sum()), train_chained=int((chain[t] >= 1).sum()), per_file=per_file,
                 n_val=int(len(v)), val_resets=int(len(np.unique(reset[v]))), val_walk_first=int(is_wf[v].sum()),
                 n_test_iid=int(len(s["test_iid"])), n_drop_resets=int(len(drop)), zero_walk_flag=zero)  # fmt: skip
        info["arms"][arm] = o
        print(f"  {arm:2s}: train {o['n_train']} segments ({o['train_resets']} resets; first {o['train_first']}, chained "
              f"{o['train_chained']}), walk-first {wf_seg} segments / {wf_res} resets (task {o['train_walk_first_task']}, d3 "
              f"{o['train_walk_first_d3']}); per file {per_file}; val {o['n_val']} ({o['val_resets']} resets, walk-first "
              f"{o['val_walk_first']}); test_iid {o['n_test_iid']}; drop {len(drop)} resets; zero walk flag {zero}")  # fmt: skip
    D["bucket"] = bucket0
    nN, nC = info["arms"]["N"]["n_train"], info["arms"]["C"]["n_train"]
    check(nN == n_N, f"N's training split {nN} == D's training segments of in-place resets {n_N}")
    check(info["arms"]["N"]["train_walk_first_segments"] == 0 and info["arms"]["N"]["train_walk_first_resets"] == 0,
          "N: 0 walk-first segments / resets in training")  # fmt: skip
    check(int(walked[DT.splits({**D, "bucket": np.where(np.isin(reset, drop_N), 10, bucket0)})["train"]].sum()) == 0,
          "N: walked flag 0 on every training segment")  # fmt: skip
    check(info["arms"]["N"]["val_walk_first"] == 0 and info["arms"]["C"]["val_walk_first"] == 0, "N, C: val (checkpoint pick) in place only")
    check(info["arms"]["N"]["n_val"] == info["arms"]["C"]["n_val"], "N and C pick on the same val segments")
    check(abs(nC - nN) / nN <= TOL, f"C's training split {nC} vs N's {nN}: {100 * (nC - nN) / nN:+.3f} % (tolerance 1 %)")
    info["C_md5_kept_resets"] = int(K)
    # input check (no model): real main settled starts against each arm's training range
    rp = L / f"selfmodel_lmlreal_20261125/{be}/data/all.npz"
    DR = DT.load(str(rp))
    ri = np.flatnonzero((np.asarray(DR["fell"]) == 0) & (np.asarray(DR["settled_end"]) > 0))
    qr = SR.quantities(DR, ri)
    wr = np.asarray(DR["walked"])[ri] > 0
    chr_ = np.where(wr, 0, 1)  # the input check's "first" / "second" columns here: walked / in place
    xr = np.asarray(DR["x0"])[ri].astype(np.float64)
    check(bool((xr[:, APPROACH_COLS] == 0).all()), "lmlreal: x0 columns 274-277 zero")
    check(bool((xr[:, WALKED_COL] == np.asarray(DR["walked"])[ri]).all()), "lmlreal: x0 column 204 == walked")
    print(f"\ninput check, lmlreal {be} settled ({len(ri)} segments; walked {int(wr.sum())}, in place {int((~wr).sum())}): share inside "
          "the arm's training p0.5 / p99.5 (sd_range's 39 quantities; below 'first' = walked, 'second' = in place), and x0 columns under Norm's floor")
    info["input_check"] = {}
    for arm in ("D", "N", "C"):
        drop, zero = arms[arm]
        t = DT.splits({**D, "bucket": np.where(np.isin(reset, drop), 10, bucket0)})["train"]
        qt = SR.quantities(D, t)
        res = {}
        allin = np.ones(len(ri), bool)
        for g in qt:
            lo, hi = np.percentile(qt[g], 0.5, axis=0), np.percentile(qt[g], 99.5, axis=0)
            ins = ((qr[g] >= lo) & (qr[g] <= hi)).all(1)
            allin &= ins
            res[g] = dict(all=float(ins.mean()), first=float(ins[chr_ == 0].mean()), second=float(ins[chr_ == 1].mean()))
            if g == "joints":
                out_j = ((qr[g] < lo) | (qr[g] > hi)).mean(0)
                res["joints_out_top"] = [(int(j), float(out_j[j])) for j in np.argsort(-out_j)[:6]]
        res["range39"] = dict(all=float(allin.mean()), first=float(allin[chr_ == 0].mean()), second=float(allin[chr_ == 1].mean()))
        X = x0[t].astype(np.float64)
        if zero:
            X[:, WALKED_COL] = 0
        mu, sd = X.mean(0), X.std(0)
        xq = xr.copy()
        if zero:
            xq[:, WALKED_COL] = 0
        z = np.abs((xq - mu) / np.maximum(sd, 1e-3))
        low = np.flatnonzero(sd < 1e-3)
        res["low_sd_cols"] = [(int(c), float(sd[c]), float(z[:, c].max())) for c in low]
        res["max_abs_z"] = dict(all=float(z.max()), p99=float(np.percentile(z.max(1), 99)), median_rowmax=float(np.median(z.max(1))))
        res["max_abs_z_top_cols"] = [(int(c), float(z[:, c].max())) for c in np.argsort(-z.max(0))[:6]]
        info["input_check"][arm] = res
        f = lambda d: f"{100 * d['all']:.1f} % (first {100 * d['first']:.1f}, second {100 * d['second']:.1f})"  # noqa: E731
        print(f"  {arm:2s}: joints {f(res['joints'])}; pelvis {f(res['pelvis'])}; hands {f(res['hands'])}; all 39 {f(res['range39'])}")
        print(f"      joints most often outside (S index, share): {res['joints_out_top']}")
        print(f"      x0 columns with training sd < 1e-3 (col, sd, max |z| on real): {res['low_sd_cols']}")
        print(f"      |z| on real under this arm's normalisation: max {res['max_abs_z']['all']:.1f}, p99 of row max "
              f"{res['max_abs_z']['p99']:.1f}, median row max {res['max_abs_z']['median_rowmax']:.1f}; top columns {res['max_abs_z_top_cols']}")  # fmt: skip
    np.savez(a.out / "splits.npz", **{f"{arm}_drop": d for arm, (d, _) in arms.items()},
             **{f"{arm}_n_train": np.int64(info["arms"][arm]["n_train"]) for arm in arms},
             **{f"{arm}_n_val": np.int64(info["arms"][arm]["n_val"]) for arm in arms},
             **{f"{arm}_zero": np.bool_(z) for arm, (_, z) in arms.items()}, walk_first=wf_ids)  # fmt: skip
    info["fails"] = fails
    (a.out / "splits.json").write_text(json.dumps(info, indent=1))
    print(f"\n{'SPLIT CHECK PASS' if not fails else 'SPLIT CHECK FAIL: ' + '; '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
