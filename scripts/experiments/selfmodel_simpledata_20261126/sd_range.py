"""selfmodel_simpledata_20261126, evaluation 4: are the real SIMPLE segments' starts inside model D's training data, and
does the model's error differ inside and outside; read-only.

    cd <repo>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1 LOCO_REPO=<repo> CUDA_VISIBLE_DEVICES=<g> \
        $PY sd_range.py --train <D's data spec> --data3 <all.npz> --models <out>/models_D_s0 \
        --real main=<real>/main/data/all.npz,task=<real>/task/data/all.npz [--json F]

The training set: D's training split (data.load of --train, the spec D was trained on: data3, data3b and the mid starts;
data.splits' train, frac 1.0). For every real segment (and, as the reference, data3's val segments) at its start row
(S[k0], the command row):
  range      each of 39 quantities inside the training set's 0.5 / 99.5 % (per quantity): joints (the 29 joint angles,
             S[k0, 0:29]), pelvis (height S[k0, 64] and the gravity direction in the pelvis frame S[k0, 58:61]), hands
             (both hand points, the data's W columns 4:7 / 7:10, in the start's heading frame from the pelvis xy, z as
             is). Reported per group and all together.
  neighbour  the distance in the standardized x0 space (the model's input with REACHWM_FDVEL=1: per column the training
             set's mean and standard deviation, constant columns dropped) to the nearest training segment, against the
             training set's own nearest-neighbour distance (10 000 training segments drawn with seed 0, each to the
             nearest training segment of another reset, so a trajectory's own mid starts do not count): inside when not
             above that distribution's 99th percentile.
  table      the command target (cmd_target: heading frame from the pelvis, z as is) inside the reach table's box for
             its hand and pelvis height (d3_collect.in_table on the table the collection used, --table).
  in range   all three. The error (D, frac 1.0: the predicted final hand point to the real one, cm) is reported inside /
             outside, for every check, with the ratio to SIMPLE val (data3 val, all, kept: the same model's median);
             groups: in place / walk first (data's walked), and the first / second reach (chain 0 / 1).
Real segments are kept as evaluation 3 keeps them: the main label not fallen and settled at the end (primary), the task
label not fallen (primary), each also with the other rule. Shares: 95 % intervals from 1000 resamples of the episodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(os.environ.get("LOCO_REPO", "/wuji-vefps-D/wuji-il/yijie/ws/code/loco"))
EXP = REPO / "scripts" / "experiments"
sys.path[:0] = [str(EXP / "reachwm_20261030"), str(EXP / "selfwm_20261029"), str(EXP / "selfmodel_data3_20261125"),
                str(EXP / "locomanilab_20261117")]  # fmt: skip
import data as DT  # noqa: E402
import train as TR  # noqa: E402

for k in ("REACHWM_GOAL_REF", "REACHWM_APPROACH", "REACHWM_QFIX", "REACHWM_FDVEL"):
    assert os.environ.get(k) == "1", f"{k}=1 is D's switch here"
M = TR.M
NB = 1000
N_INTERNAL = 10_000
PRIMARY = {"main": "settled", "task": "ended"}


def kept(D: dict, idx: np.ndarray, rule: str) -> np.ndarray:
    ok = np.asarray(D["fell"])[idx] == 0
    if rule == "settled":
        ok &= np.asarray(D["settled_end"])[idx] > 0
    return idx[ok]


def quantities(D: dict, idx: np.ndarray) -> dict:
    """The 39 range quantities at the start row (see the module docstring)."""
    r = np.asarray(D["so"])[idx] + np.asarray(D["k0"])[idx].astype(int)
    S, W = np.asarray(D["S"]), np.asarray(D["W"])
    st = np.asarray(D["start"])[idx].astype(np.float64)
    yaw, org = st[:, 3], np.c_[st[:, :2], np.zeros(len(idx))]
    hands = [DT.hv(W[r, a:b].astype(np.float64) - org, yaw) for a, b in ((4, 7), (7, 10))]
    return dict(joints=S[r, 0:29].astype(np.float64), pelvis=np.c_[S[r, 64], S[r, 58:61]].astype(np.float64), hands=np.concatenate(hands, 1))


def nn_dist(Q: torch.Tensor, qr: torch.Tensor, T: torch.Tensor, tr: torch.Tensor, chunk_q=2048, chunk_t=262_144) -> np.ndarray:
    """Per query row the Euclidean distance to the nearest row of T from another reset (qr / tr: reset ids)."""
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


def groups_of(reset: np.ndarray) -> list[np.ndarray]:
    o = np.argsort(reset, kind="stable")
    return np.split(o, np.flatnonzero(np.diff(reset[o])) + 1)


def boot(G, stat, rng) -> tuple[float, float]:
    n = len(G)
    v = [stat(np.concatenate([G[j] for j in rng.integers(0, n, n)])) for _ in range(NB)]
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="D's data spec (comma-separated: data3, data3b, mid files)")
    ap.add_argument("--data3", required=True, help="data3's all.npz (the SIMPLE val reference)")
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--real", required=True, help="label=path,... (main=..., task=...)")
    ap.add_argument("--table", type=Path, default=None, help="default d3_collect.TABLE")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    import d3_collect as D3

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    rng = np.random.default_rng(0)
    tpath = a.table or D3.TABLE
    table = json.loads(Path(tpath).read_text())
    print(f"table {tpath} md5 {hashlib.md5(Path(tpath).read_bytes()).hexdigest()}; device {dev}")
    # the training set
    DTr = DT.load(a.train)
    tr = DT.splits(DTr)["train"]
    qt = quantities(DTr, tr)
    lim = {g: (np.percentile(v, 0.5, axis=0), np.percentile(v, 99.5, axis=0)) for g, v in qt.items()}
    X = np.asarray(DTr["x0"])[tr].astype(np.float64)
    mu, sd = X.mean(0), X.std(0)
    keep_c = sd > 1e-6
    T = torch.tensor(((X - mu) / np.where(keep_c, sd, 1))[:, keep_c], dtype=torch.float32, device=dev)
    rtr = np.asarray(DTr["reset"])[tr].astype(np.int64)
    trr = torch.tensor(rtr, device=dev)
    pick = np.sort(rng.choice(len(tr), size=min(N_INTERNAL, len(tr)), replace=False))
    d_int = nn_dist(T[pick], trr[pick], T, trr)
    thr = float(np.percentile(d_int, 99))
    print(f"training set: {len(tr)} segments ({len(np.unique(rtr))} resets) from {a.train}; x0 {X.shape[1]} columns, "
          f"{int(keep_c.sum())} not constant; internal nearest-neighbour distance (other reset) median {np.median(d_int):.3f}, "
          f"p99 {thr:.3f} ({len(pick)} drawn)")  # fmt: skip
    del X, DTr
    out = dict(train=a.train, n_train=int(len(tr)), nn_p99=thr, nn_internal_median=float(np.median(d_int)), table=str(tpath), sets={})

    def assess(D: dict, idx: np.ndarray) -> dict:
        q = quantities(D, idx)
        inr = {g: ((q[g] >= lim[g][0]) & (q[g] <= lim[g][1])).all(1) for g in q}
        x = np.asarray(D["x0"])[idx].astype(np.float64)
        Q = torch.tensor(((x - mu) / np.where(keep_c, sd, 1))[:, keep_c], dtype=torch.float32, device=dev)
        rq = torch.tensor(np.asarray(D["reset"])[idx].astype(np.int64), device=dev)
        d = nn_dist(Q, rq, T, trr)
        side, pz, ct = np.asarray(D["side"])[idx], np.asarray(D["pelvis_z"])[idx], np.asarray(D["cmd_target"])[idx]
        tab = np.array([D3.in_table(table["rows"]["right" if s > 0 else "left"][D3.pz_key(None if p < 0 else float(p))], t)
                        for s, p, t in zip(side, pz, ct)])  # fmt: skip
        checks = dict(joints=inr["joints"], pelvis=inr["pelvis"], hands=inr["hands"], neighbour=d <= thr, table=tab)
        checks["range (39)"] = inr["joints"] & inr["pelvis"] & inr["hands"]
        checks["all three"] = checks["range (39)"] & checks["neighbour"] & checks["table"]
        return dict(checks=checks, nn=d)

    def errors(D: dict, idx: np.ndarray) -> np.ndarray:
        GD = M.tensors(D, dev)
        ti = torch.tensor(idx, device=dev)
        m = TR.load_model("E3_ensemble", 1.0, D, dev, root=a.models)
        p = M.predict_batched(m, GD, ti)
        return (100 * (p["pos"] - GD["pos_f"][ti]).norm(dim=-1)).cpu().numpy()

    DV = DT.load(a.data3)
    vi = kept(DV, DT.splits(DV)["val"], "settled")
    ev = errors(DV, vi)
    val_med = float(np.median(ev))
    av = assess(DV, vi)
    print(f"SIMPLE val (data3 val, all, kept settled): {len(vi)} segments, D median {val_med:.3f} cm")
    print("  reference, val inside: " + ", ".join(f"{k} {100 * v.mean():.1f} %" for k, v in av["checks"].items())
          + f"; nearest-neighbour distance median {np.median(av['nn']):.3f}")  # fmt: skip
    out["val"] = dict(n=int(len(vi)), median=val_med, inside={k: float(v.mean()) for k, v in av["checks"].items()})
    del DV
    for item in a.real.split(","):
        lab, path = item.split("=", 1)
        D = DT.load(path)
        for rule in ("settled", "ended"):
            idx = kept(D, np.arange(len(D["n_steps"])), rule)
            tag = f"{lab} label, kept {rule}" + (" (primary)" if PRIMARY.get(lab) == rule else "")
            if len(idx) == 0:
                print(f"\n== {tag}: no segments")
                continue
            e = errors(D, idx)
            A = assess(D, idx)
            rs = np.asarray(D["reset"])[idx]
            G = groups_of(rs)
            walked, chain = np.asarray(D["walked"])[idx] > 0, np.asarray(D["chain"])[idx]
            print(f"\n== {tag}: {len(idx)} segments from {len(G)} episodes ({path}); in place {int((~walked).sum())}, walk first "
                  f"{int(walked.sum())}; D median {np.median(e):.3f} cm = {np.median(e) / val_med:.3f}x SIMPLE val; nearest-neighbour "
                  f"distance median {np.median(A['nn']):.3f} (training p99 {thr:.3f})")  # fmt: skip
            res = dict(n=int(len(idx)), n_episodes=len(G), median=float(np.median(e)), ratio=float(np.median(e) / val_med), checks={})
            for k, v in A["checks"].items():
                ci = boot(G, lambda s: v[s].mean(), rng)
                o = dict(inside=float(v.mean()), inside_ci=ci, n_in=int(v.sum()), n_out=int((~v).sum()))
                for nm, sel in (("in", v), ("out", ~v)):
                    o[f"err_{nm}"] = float(np.median(e[sel])) if sel.any() else None
                    o[f"ratio_{nm}"] = o[f"err_{nm}"] / val_med if sel.any() else None
                for gname, gsel in (("in place", ~walked), ("walk first", walked), ("first reach", chain == 0), ("second reach", chain == 1)):
                    if gsel.any():
                        o[gname] = dict(n=int(gsel.sum()), inside=float(v[gsel].mean()),
                                        err_in=float(np.median(e[gsel & v])) if (gsel & v).any() else None,
                                        err_out=float(np.median(e[gsel & ~v])) if (gsel & ~v).any() else None)  # fmt: skip
                res["checks"][k] = o
                fm = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
                gs = "; ".join(f"{gn} n {o[gn]['n']} inside {100 * o[gn]['inside']:.1f} % (err in {fm(o[gn]['err_in'])} / out "
                               f"{fm(o[gn]['err_out'])})" for gn in ("in place", "walk first", "first reach", "second reach") if gn in o)  # fmt: skip
                print(f"  {k:11s}: inside {100 * o['inside']:.1f} % [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}] ({o['n_in']} / {o['n_out']} out); "
                      f"error inside {fm(o['err_in'])} cm ({fm(o['ratio_in'])}x), outside {fm(o['err_out'])} cm ({fm(o['ratio_out'])}x)")  # fmt: skip
                print(f"               {gs}")
            out["sets"][f"{lab}_{rule}"] = res
        del D
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
