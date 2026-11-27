"""selfmodel_nowalk_20261126 revision 3 (descriptive only, not a criterion): do the walk-first training starts of a
LocoManiLab controller match the starts of its real task episodes' first reach after a walk? The collection walks every
controller with hg_walk; the task episodes walk HEFT with gait_heft (ScaleBFM and SONIC with hg_walk). Read-only.

    cd <repo>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 LOCO_REPO=<repo> CUDA_VISIBLE_DEVICES= \
        $PY nw2_walkfit.py --backend <be> --eval <lml/run/<be>/eval> --out <dir>

Real segments: lmlreal's records of the controller, kept as nw2_eval keeps them (not fallen / lost, settled end), the
first reach (reach_no 1), walked or in place (the record's walked flag); each also restricted to chain 0 (no via point
before it).
1. errors   from nw2_eval's per_segment.npz (the same numbers as its tables): per model the median [95 %: 4000
            resamples of the episodes]; N/C = mean over seeds of N_k/C_k (seed-paired, as nw2_eval), per seed, C/D; and
            the paired share of segments with N_k's error above C_k's (mean over seeds).
2. starts   sd_range's logic on four reference sets from D's training split (data.splits' train of train3b's D spec):
            all; walk-first (walked); walk-first first reaches (walked, chain 0); in place (walked 0 = N's training set).
            range      each of the 39 start quantities (sd_range.quantities, copied in nw2_split) inside the reference
                       set's 0.5 / 99.5 %; per group (joints, pelvis, hands) and all 39.
            neighbour  the distance in standardized x0 (mean / sd of the whole training split; constant columns and the
                       walk-flag column 204 dropped, so the flag itself does not count) to the nearest segment of the
                       reference set, inside when not above the reference set's own nearest-neighbour p99 (10 000 of
                       its segments drawn with seed 0, each to its nearest segment of another reset).
            nearer     share of real segments whose nearest walk-first training segment is nearer than the nearest
                       in-place one, beside the base rate (walk-first share of the two sets' sizes).
            Shares with 95 % intervals from 1000 resamples of the episodes.
Writes <out>/walkfit.txt (this print) and walkfit.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(os.environ.get("LOCO_REPO", "/wuji-vefps-D/wuji-il/yijie/ws/code/loco"))
sys.path[:0] = [str(REPO / "scripts/experiments/reachwm_20261030"), str(REPO / "scripts/experiments/selfwm_20261029")]
import data as DT  # noqa: E402

for k in ("REACHWM_GOAL_REF", "REACHWM_APPROACH", "REACHWM_QFIX"):
    assert os.environ.get(k) == "1", f"{k}=1 is train3b D's switch"
assert not DT.FDVEL and not DT.X0_REF, "train3b's D: FDVEL / X0_REF off"
L = Path("/wuji-vefps-D/wuji-il/yijie/loco_outputs")
WALKED_COL = 204
SEEDS = (0, 1, 2)
NB_ERR, NB_SHARE, N_INTERNAL = 4000, 1000, 10_000


def quantities(D: dict, idx: np.ndarray) -> dict:
    """sd_range.quantities (simpledata, 1276163a), verbatim: the 39 range quantities at the start row."""
    r = np.asarray(D["so"])[idx] + np.asarray(D["k0"])[idx].astype(int)
    S, W = np.asarray(D["S"]), np.asarray(D["W"])
    st = np.asarray(D["start"])[idx].astype(np.float64)
    yaw, org = st[:, 3], np.c_[st[:, :2], np.zeros(len(idx))]
    hands = [DT.hv(W[r, a:b].astype(np.float64) - org, yaw) for a, b in ((4, 7), (7, 10))]
    return dict(joints=S[r, 0:29].astype(np.float64), pelvis=np.c_[S[r, 64], S[r, 58:61]].astype(np.float64), hands=np.concatenate(hands, 1))


def nn_dist(Q: torch.Tensor, qr: torch.Tensor, T: torch.Tensor, tr: torch.Tensor, chunk_q=2048, chunk_t=262_144) -> np.ndarray:
    """sd_range.nn_dist, verbatim: per query row the distance to the nearest row of T from another reset."""
    out = []
    tn = (T * T).sum(1)
    for i in range(0, len(Q), chunk_q):
        q, rq = Q[i : i + chunk_q], qr[i : i + chunk_q]
        qn = (q * q).sum(1)
        best = torch.full((len(q),), float("inf"), device=Q.device)
        for j in range(0, len(T), chunk_t):
            t = T[j : j + chunk_t]
            d2 = qn[:, None] + tn[None, j : j + chunk_t] - 2 * q @ t.T
            d2 = d2.masked_fill(rq[:, None] == tr[None, j : j + chunk_t], float("inf"))
            best = torch.minimum(best, d2.min(1).values)
        out.append(best.clamp_min(0).sqrt().cpu().numpy())
    return np.concatenate(out)


def groups(keys: np.ndarray) -> list[np.ndarray]:
    o = np.argsort(keys, kind="stable")
    return np.split(o, np.flatnonzero(np.diff(keys[o])) + 1)


def resample(G, rng):
    return np.concatenate([G[j] for j in rng.integers(0, len(G), len(G))])


def err_cell(title, E, rs, sel, rng, out) -> None:
    idx = np.flatnonzero(sel)
    if len(idx) == 0:
        print(f"  {title}: no segments")
        return
    G = [idx[g] for g in groups(rs[idx])]
    names = list(E)
    ks = [k for k in SEEDS if f"N{k}" in E and f"C{k}" in E]
    kd = [k for k in SEEDS if f"C{k}" in E and f"D{k}" in E]

    def st(s):
        o = {n: float(np.median(E[n][s])) for n in names}
        for k in ks:
            o[f"N{k}/C{k}"] = o[f"N{k}"] / o[f"C{k}"]
            o[f"P(N{k}>C{k})"] = float(np.mean(E[f"N{k}"][s] > E[f"C{k}"][s]))
        o["N/C mean"] = float(np.mean([o[f"N{k}/C{k}"] for k in ks]))
        o["P(N>C) mean"] = float(np.mean([o[f"P(N{k}>C{k})"] for k in ks]))
        if kd:
            o["C/D mean"] = float(np.mean([o[f"C{k}"] / o[f"D{k}"] for k in kd]))
        return o

    pt = st(idx)
    bs = [st(resample(G, rng)) for _ in range(NB_ERR)]
    ci = {k: (float(np.percentile([b[k] for b in bs], 2.5)), float(np.percentile([b[k] for b in bs], 97.5))) for k in pt}
    out[title] = dict(n=int(len(idx)), episodes=len(G), point=pt, ci=ci)
    f = lambda k, d=3: f"{pt[k]:.{d}f} [{ci[k][0]:.{d}f}, {ci[k][1]:.{d}f}]"  # noqa: E731
    print(f"  {title}: n {len(idx)} ({len(G)} episodes)")
    print("    median cm: " + " | ".join(f"{n} {f(n, 2)}" for n in names))
    print(f"    N/C mean over seeds {f('N/C mean')}; per seed " + ", ".join(f"N{k}/C{k} {f(f'N{k}/C{k}')}" for k in ks))
    print(f"    share N_k error > C_k error, mean over seeds {f('P(N>C) mean')}; per seed " + ", ".join(f"{pt[f'P(N{k}>C{k})']:.3f}" for k in ks))
    if kd:
        print(f"    C/D mean over seeds {f('C/D mean')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["scalebfm", "sonic", "heft"])
    ap.add_argument("--eval", type=Path, required=True, help="nw2_eval's --out of this controller (per_segment.npz)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    be = a.backend
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    rng = np.random.default_rng(0)
    a.out.mkdir(parents=True, exist_ok=True)
    res = dict(backend=be, errors={}, starts={})
    # real first reaches
    DR = DT.load(str(L / f"selfmodel_lmlreal_20261125/{be}/data/all.npz"))
    Z = dict(np.load(a.eval / "per_segment.npz"))
    idx = Z["real__idx"]
    kept = np.flatnonzero((np.asarray(DR["fell"]) == 0) & (np.asarray(DR["settled_end"]) > 0))
    assert np.array_equal(idx, kept), "per_segment.npz's real segments are not the kept ones"
    rs = np.asarray(DR["reset"]).astype(np.int64)[idx]
    assert np.array_equal(rs, Z["real__reset"])
    rn, wk, ch = np.asarray(DR["reach_no"])[idx], np.asarray(DR["walked"])[idx] > 0, np.asarray(DR["chain"])[idx]
    assert np.array_equal(rn, Z["real__reach_no"]) and np.array_equal(wk, Z["real__walked"])
    E = {k.split("__err_")[1]: v for k, v in Z.items() if k.startswith("real__err_")}
    E = {n: E[n] for n in sorted(E, key=lambda n: ("DNC".index(n[0]), n))}
    print(f"{be}: lmlreal kept {len(idx)} segments; models {list(E)}; eval {a.eval}")
    cells = [("第一下, walked", (rn == 1) & wk), ("第一下, in place", (rn == 1) & ~wk),
             ("第一下 chain 0, walked", (rn == 1) & (ch == 0) & wk), ("第一下 chain 0, in place", (rn == 1) & (ch == 0) & ~wk)]
    print("\n== 1. errors, real first reaches (reach_no 1; kept: not fallen, settled end)")
    for t, s in cells:
        err_cell(t, E, rs, s, rng, res["errors"])
    # reference sets from D's training split
    T3 = L / "selfmodel_train3b_20261125" / be
    spec = ",".join(str(p) for p in (L / f"selfmodel_data3_20261125/{be}/data/all.npz", L / f"selfmodel_data3b_20261125/{be}/data/all.npz",
                                     T3 / "data/mid_data3.npz", T3 / "data/mid_data3b.npz"))  # fmt: skip
    D = DT.load(spec)
    tr = DT.splits(D)["train"]
    twk, tch = np.asarray(D["walked"])[tr] > 0, np.asarray(D["chain"])[tr]
    refs = {"all": tr, "walk-first": tr[twk], "walk-first chain 0": tr[twk & (tch == 0)], "in place (N's set)": tr[~twk]}
    X = np.asarray(D["x0"])[tr].astype(np.float64)
    mu, sd = X.mean(0), X.std(0)
    keep = sd > 1e-6
    keep[WALKED_COL] = False
    print(f"\n== 2. starts: D's training split {len(tr)} segments ({spec}); reference sets "
          + ", ".join(f"{k} {len(v)}" for k, v in refs.items()) + f"; neighbour space {int(keep.sum())} x0 columns (204 dropped)")  # fmt: skip
    pos = np.full(len(D["n_steps"]), -1, np.int64)
    pos[tr] = np.arange(len(tr))
    Tall = torch.tensor(((X - mu) / np.where(keep, sd, 1))[:, keep], dtype=torch.float32)
    del X
    rall = torch.tensor(np.asarray(D["reset"])[tr].astype(np.int64))
    qsel = {"第一下, walked": (rn == 1) & wk, "第一下, in place": (rn == 1) & ~wk}
    qi = {k: idx[v] for k, v in qsel.items()}
    qq = {k: quantities(DR, v) for k, v in qi.items()}
    xr = {k: torch.tensor(((np.asarray(DR["x0"])[v].astype(np.float64) - mu) / np.where(keep, sd, 1))[:, keep], dtype=torch.float32) for k, v in qi.items()}
    qr = {k: torch.tensor(np.asarray(DR["reset"])[v].astype(np.int64)) for k, v in qi.items()}
    qg = {k: [g for g in groups(np.asarray(DR["reset"])[v].astype(np.int64))] for k, v in qi.items()}
    dist = {}
    for rname, ridx in refs.items():
        q_t = quantities(D, ridx)
        lim = {g: (np.percentile(v, 0.5, axis=0), np.percentile(v, 99.5, axis=0)) for g, v in q_t.items()}
        rows = torch.from_numpy(pos[ridx])
        assert (rows >= 0).all()
        T, trr = Tall[rows], rall[rows]
        pick = np.sort(np.random.default_rng(0).choice(len(ridx), size=min(N_INTERNAL, len(ridx)), replace=False))
        thr = float(np.percentile(nn_dist(T[pick], trr[pick], T, trr), 99))
        print(f"  reference {rname}: {len(ridx)} segments, {len(np.unique(trr.numpy()))} resets; neighbour p99 {thr:.3f}")
        res["starts"][rname] = dict(n=int(len(ridx)), nn_p99=thr, sets={})
        for qname in qsel:
            if len(qi[qname]) == 0:
                continue
            q = qq[qname]
            inr = {g: ((q[g] >= lim[g][0]) & (q[g] <= lim[g][1])).all(1) for g in q}
            inr["range (39)"] = inr["joints"] & inr["pelvis"] & inr["hands"]
            d = nn_dist(xr[qname], qr[qname], T, trr)
            dist[(rname, qname)] = d
            inr["neighbour"] = d <= thr
            G = qg[qname]
            o = {}
            for k, v in inr.items():
                bs = [float(v[resample(G, rng)].mean()) for _ in range(NB_SHARE)]
                o[k] = (float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
            out_j = np.argsort(-(~((q["joints"] >= lim["joints"][0]) & (q["joints"] <= lim["joints"][1]))).mean(0))[:4]
            jo = [(int(j), float((~((q["joints"][:, j] >= lim["joints"][0][j]) & (q["joints"][:, j] <= lim["joints"][1][j]))).mean())) for j in out_j]
            res["starts"][rname]["sets"][qname] = dict(n=int(len(qi[qname])), shares=o, joints_outside=jo, nn_median=float(np.median(d)))
            print(f"    {qname} (n {len(qi[qname])}, {len(G)} episodes): " + "; ".join(f"{k} {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]" for k, v in o.items())
                  + f"; nn median {np.median(d):.3f}; joints most often outside (S index, share) {[(j, round(s, 3)) for j, s in jo]}")  # fmt: skip
    nb, nc = len(refs["walk-first"]), len(refs["in place (N's set)"])
    print(f"  nearer: nearest walk-first training segment nearer than the nearest in-place one; base rate {nb / (nb + nc):.3f} (walk-first {nb}, in place {nc})")
    res["nearer_base_rate"] = nb / (nb + nc)
    for qname in qsel:
        if ("walk-first", qname) not in dist:
            continue
        v = dist[("walk-first", qname)] < dist[("in place (N's set)", qname)]
        G = qg[qname]
        bs = [float(v[resample(G, rng)].mean()) for _ in range(NB_SHARE)]
        res.setdefault("nearer", {})[qname] = (float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
        print(f"    {qname}: {v.mean():.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]")
    (a.out / "walkfit.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
