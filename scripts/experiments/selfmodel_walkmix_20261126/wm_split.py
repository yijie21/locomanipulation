"""selfmodel_walkmix_20261126: the training sets of the middle arms Ma (in place + task-walk resets), Mb (in place + walks
asked <= 0.6 m, both kinds), their shared random control Cm, and Dh (D's data as is), built from model D's data spec
(selfmodel_simpledata_20261126: data3, data3b, the two mid files). nowalk's nw_split.py with only the arms changed;
read-only on the data, writes <out>/splits.npz, splits.json and prints the checks.

    cd <repo>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1 LOCO_REPO=<repo> CUDA_VISIBLE_DEVICES= \
        $PY wm_split.py --sd <selfmodel_simpledata_20261126 dir> --out <dir> [--real <real/main/data/all.npz>]

Walk-first resets, their kind (task / d3) and the asked walk (planar distance of walk.asked, m): the collection logs
(data3/data, data3b/data: w*_log.json, one entry per reset). Checked against every segment's walked and walk_task.
Arms (each a set of resets taken out of the training process: their segments get bucket 10, so data.splits leaves them
out of train, val and test_iid). Every arm drops the walk-first resets of bucket 8, so all pick their checkpoint on
nowalk's in-place val (the same segments as nowalk's N and C):
  Ma  + every d3-walk reset (all buckets): trains on in-place and task-walk resets.
  Mb  + every walk-first reset whose asked walk is > 0.6 m (all buckets, both kinds).
  Cm  + whole training resets (bucket < 8, any type) dropped in the order of md5("walkmix_cm:<reset>"), the smallest
      digests kept first; the kept prefix is the one whose training segments are closest to (Ma + Mb) / 2;
      |Cm - target| / target <= 0.1 % is asserted.
  Dh  nothing else: D's training set, picked on the in-place val (D itself picked on the whole val).
No arm zeroes the walk flag (x0 column 204 stays data.py's `walked`).
Input check before any training (no model): as nw_split, the 39 start quantities of sd_range of the real main-label
settled segments inside each arm's training p0.5 / p99.5, and x0's low-sd columns.
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

for k in ("REACHWM_GOAL_REF", "REACHWM_APPROACH", "REACHWM_QFIX", "REACHWM_FDVEL"):
    assert os.environ.get(k) == "1", f"{k}=1 is D's switch here"
assert DT.X0_DIM == 278, DT.X0_DIM
WALKED_COL, CHAIN_COL, APPROACH_COLS = 204, 205, slice(274, 278)
TOL = 0.001
WALK_MAX = 0.6  # m, Mb
D_N_TRAIN = 809804  # selfmodel_simpledata_20261126 train/run.log: D's training split


def md5_key(r: int) -> int:
    return int(hashlib.md5(f"walkmix_cm:{int(r)}".encode()).hexdigest(), 16)


def log_walked(d: Path) -> dict[int, tuple[bool, str, float]]:
    """reset -> (walked, kind 'task' / 'd3' / '', planar distance of walk.asked in m)."""
    out = {}
    for f in sorted(glob.glob(str(d / "w*_log.json"))):
        for e in json.load(open(f)):
            assert e["reset"] not in out, f"reset {e['reset']} twice in the logs"
            assert bool(e["walk_asked"]) == bool(e["walked"]), e["reset"]
            w = e["walk"] if e["walked"] else {}
            kind = w.get("kind", "") if e["walked"] else ""
            assert (kind in ("task", "d3")) == bool(e["walked"]), (e["reset"], kind)
            ask = w.get("asked") if e["walked"] else None
            out[int(e["reset"])] = (bool(e["walked"]), kind, float(np.hypot(ask[0], ask[1])) if ask else 0.0)
    return out


def apply(D: dict, drop: np.ndarray, zero: bool) -> None:
    """An arm's view of D (in place): dropped resets' segments to bucket 10, the walk flag column zeroed."""
    D["bucket"] = np.where(np.isin(np.asarray(D["reset"]).astype(np.int64), drop), 10, D["bucket"])
    if zero:
        D["x0"][:, WALKED_COL] = 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sd", type=Path, required=True, help="selfmodel_simpledata_20261126's output dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--real", type=Path, default=None, help="default <sd>/real/main/data/all.npz")
    ap.add_argument("--nowalk", type=Path, required=True, help="nowalk's splits.npz (its N / C: the same val asserted)")
    a = ap.parse_args()
    # simpledata's own scripts live in asrun/repo only (repo_b holds the shared code, identical data.py)
    sys.path.insert(0, str(a.sd / "asrun/repo/scripts/experiments/selfmodel_simpledata_20261126"))
    import sd_range as SR  # quantities(): the 39 start quantities

    assert SR.DT is DT

    a.out.mkdir(parents=True, exist_ok=True)
    files = [a.sd / "data3/data/all.npz", a.sd / "data3b/data/all.npz", a.sd / "train/data/mid_data3.npz", a.sd / "train/data/mid_data3b.npz"]
    names = ["data3", "data3b", "mid_data3", "mid_data3b"]
    spec = ",".join(map(str, files))
    md5s = {n: hashlib.md5(open(p, "rb").read()).hexdigest()[:8] for n, p in zip(names, files)}
    print(f"spec {spec}\n  md5 {md5s}")
    lw = {**log_walked(a.sd / "data3/data"), **log_walked(a.sd / "data3b/data")}
    wf_ids = np.array(sorted(r for r, w in lw.items() if w[0]), np.int64)
    d3_ids = np.array(sorted(r for r, w in lw.items() if w[0] and w[1] == "d3"), np.int64)
    far_ids = np.array(sorted(r for r, w in lw.items() if w[0] and w[2] > WALK_MAX), np.int64)
    print(f"logs: {len(lw)} resets, walk-first {len(wf_ids)} (task {len(wf_ids) - len(d3_ids)}, d3 {len(d3_ids)}; asked > {WALK_MAX} m {len(far_ids)})")
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
    lwa = np.array([lw[int(r)][0] for r in reset])
    check(bool((lwa == walked).all()), f"segment walked == the log's walked, all {N} segments (data3, data3b, mid)")
    lwt = np.array([lw[int(r)][1] == "task" for r in reset])
    check(bool((lwt == (np.asarray(D["walk_task"]) > 0)).all()), f"segment walk_task == the log's kind 'task', all {N} segments")
    for k, nm in ((2, "mid_data3"), (3, "mid_data3b")):
        src = np.load(a.sd / f"train/data/{nm}_src.npz")["src"]
        s0 = int(np.flatnonzero(fid == k - 2)[0])
        m0 = np.flatnonzero(fid == k)
        check(len(src) == len(m0), f"{nm}_src rows {len(src)} == {nm} segments {len(m0)}")
        check(bool((reset[s0 + src] == reset[m0]).all() and (walked[s0 + src] == walked[m0]).all()),
              f"{nm}: reset and walked equal to the source segment's (mid_*_src.npz)")  # fmt: skip
    check(bool((x0[:, WALKED_COL] == walked.astype(np.float32)).all()), "x0 column 204 == walked")
    check(bool((x0[:, CHAIN_COL] == chain.astype(np.float32)).all()), "x0 column 205 == chain")
    check(bool((x0[:, APPROACH_COLS] == 0).all()), "x0 columns 274-277 (REACHWM_APPROACH) zero everywhere")
    sp = DT.splits(D)
    tr = sp["train"]
    check(len(tr) == D_N_TRAIN, f"D's training split {len(tr)} == {D_N_TRAIN} (simpledata run.log)")
    bucket0 = np.asarray(D["bucket"]).copy()
    per_reset_bucket = {}
    for r, b in zip(reset, bucket0):
        assert per_reset_bucket.setdefault(int(r), int(b)) == int(b)
    is_wf = np.isin(reset, wf_ids)
    tr_resets, tr_counts = np.unique(reset[tr], return_counts=True)
    val_wf = np.array(sorted(r for r in wf_ids if per_reset_bucket.get(int(r), -1) == 8), np.int64)
    drop_Ma = np.union1d(d3_ids, val_wf)
    drop_Mb = np.union1d(far_ids, val_wf)
    cnt = lambda drop: int((~np.isin(reset[tr], drop)).sum())  # noqa: E731  training segments kept
    n_Ma, n_Mb = cnt(drop_Ma), cnt(drop_Mb)
    target = (n_Ma + n_Mb) / 2
    order = np.argsort([md5_key(r) for r in tr_resets], kind="stable")
    cum = np.cumsum(tr_counts[order])
    K = int(np.argmin(np.abs(cum - target))) + 1  # the prefix closest to the target
    keep_Cm = np.sort(tr_resets[order[:K]])
    drop_Cm = np.union1d(np.setdiff1d(tr_resets, keep_Cm), val_wf)
    arms = {"Dh": (val_wf, False), "Ma": (drop_Ma, False), "Mb": (drop_Mb, False), "Cm": (drop_Cm, False)}
    info = dict(spec=spec, md5=md5s, n_segments=N, walk_first_resets=int(len(wf_ids)), log_resets=len(lw), arms={})
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
        wt = np.asarray(D["walk_task"])[t]
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
    A_ = info["arms"]
    check(A_["Dh"]["n_train"] == D_N_TRAIN, f"Dh's training split {A_['Dh']['n_train']} == D's {D_N_TRAIN}")
    check(A_["Ma"]["n_train"] == n_Ma and A_["Ma"]["train_walk_first_d3"] == 0, f"Ma: {n_Ma} training segments, 0 from d3-walk resets")
    tm = DT.splits({**D, "bucket": np.where(np.isin(reset, drop_Mb), 10, bucket0)})["train"]
    far_t = int(np.isin(reset[tm], far_ids).sum())
    check(A_["Mb"]["n_train"] == n_Mb and far_t == 0, f"Mb: {n_Mb} training segments, 0 from walks asked > {WALK_MAX} m")
    vs = {arm: set(np.unique(reset[DT.splits({**D, "bucket": np.where(np.isin(reset, arms[arm][0]), 10, bucket0)})["val"]]).tolist()) for arm in arms}
    check(all(A_[arm]["val_walk_first"] == 0 for arm in arms), "every arm: val (checkpoint pick) in place only")
    check(len({frozenset(v) for v in vs.values()}) == 1 and len({A_[arm]["n_val"] for arm in arms}) == 1, "every arm picks on the same val resets and segments")
    ZN = np.load(a.nowalk)
    vN = set(np.unique(reset[DT.splits({**D, "bucket": np.where(np.isin(reset, ZN["N_drop"]), 10, bucket0)})["val"]]).tolist())
    check(vN == vs["Ma"] and int(ZN["N_n_val"]) == A_["Ma"]["n_val"] == int(ZN["C_n_val"]),
          f"the val is nowalk's N / C val: {A_['Ma']['n_val']} segments == {int(ZN['N_n_val'])}")  # fmt: skip
    nCm = A_["Cm"]["n_train"]
    check(abs(nCm - target) / target <= TOL, f"Cm's training split {nCm} vs target (Ma + Mb) / 2 = {target:.1f}: {100 * (nCm - target) / target:+.4f} % (tolerance 0.1 %)")
    print(f"  sizes: Ma {n_Ma} ({100 * (n_Ma - nCm) / nCm:+.3f} % vs Cm), Mb {n_Mb} ({100 * (n_Mb - nCm) / nCm:+.3f} % vs Cm), Cm {nCm}, Dh {A_['Dh']['n_train']}")
    info["Cm_md5_kept_resets"], info["Cm_target"] = int(K), float(target)
    # input check (no model): real main settled starts against each arm's training range
    rp = a.real or a.sd / "real/main/data/all.npz"
    DR = DT.load(str(rp))
    ri = np.flatnonzero((np.asarray(DR["fell"]) == 0) & (np.asarray(DR["settled_end"]) > 0))
    qr = SR.quantities(DR, ri)
    chr_ = np.asarray(DR["chain"])[ri]
    xr = np.asarray(DR["x0"])[ri].astype(np.float64)
    check(bool((xr[:, APPROACH_COLS] == 0).all()), "real main: x0 columns 274-277 zero")
    check(bool((xr[:, WALKED_COL] == 1).all()), "real main settled: walked flag 1 on every segment")
    print(f"\ninput check, real main settled ({len(ri)} segments; first {int((chr_ == 0).sum())}, second {int((chr_ == 1).sum())}): "
          "share inside the arm's training p0.5 / p99.5 (sd_range's 39 quantities), and x0 columns under Norm's floor")
    info["input_check"] = {}
    for arm in ("Dh", "Ma", "Mb", "Cm"):
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
    np.savez(a.out / "splits.npz", **{f"{arm}_drop": d for arm, (d, _) in arms.items()}, d3_walk=d3_ids, walk_far=far_ids,
             **{f"{arm}_n_train": np.int64(info["arms"][arm]["n_train"]) for arm in arms},
             **{f"{arm}_n_val": np.int64(info["arms"][arm]["n_val"]) for arm in arms},
             **{f"{arm}_zero": np.bool_(z) for arm, (_, z) in arms.items()}, walk_first=wf_ids)  # fmt: skip
    info["fails"] = fails
    (a.out / "splits.json").write_text(json.dumps(info, indent=1))
    print(f"\n{'SPLIT CHECK PASS' if not fails else 'SPLIT CHECK FAIL: ' + '; '.join(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
