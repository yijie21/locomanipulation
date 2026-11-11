"""Generalised self world model (step 1): frozen ScaleBFM + the body over the standing reach envelope, both hands,
three wrist attitudes, postures and post-walk starts (gen_data2.py). Kinematics are exact (fk.py), not learned: the
loss compares jaw centres and the head camera computed from predicted and true joints.

  dynamics  f([s_{k-1}, a_{k-1}, s_k, a_k]) -> (s_{k+1} - s_k, odometry_k), a = (side, target, lead, rotation 6)
  held out  10 % of episodes at random, and every episode whose anchor lies 0.40-0.47 m ahead of the pelvis (a
            forward slice the model never sees)

    CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/selfwm_20261029/wm2.py train
    CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/selfwm_20261029/wm2.py eval
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import env as E
import fk as FKM
import numpy as np
import torch
import torch.nn as nn
import wm as WMM

DATA = E.OUT / "data2" / "all.npz"
WM2 = E.OUT / "wm2"
A_DIM = 13
ROLL = 8
SLICE = (0.40, 0.47)  # held-out forward band of the anchor, heading frame, m


def load_data(path=DATA):
    z = dict(np.load(path))
    ns, na = z["n_states"].astype(int), z["n_actions"].astype(int)
    so = np.r_[0, np.cumsum(ns)[:-1]]
    ao = np.r_[0, np.cumsum(na)[:-1]]
    W = z["W"]
    O = np.zeros((len(z["A"]), 4), np.float32)
    for e in range(len(ns)):
        for k in range(na[e]):
            O[ao[e] + k] = E.odometry(W[so[e] + k], W[so[e] + k + 1])
    # anchor ahead of the pelvis at the episode start, heading frame
    fwd = np.array([E.rot_z(float(W[so[e], 3])).T @ (z["anchor"][e] - W[so[e], :3]) for e in range(len(ns))])[:, 0]
    rng = np.random.default_rng(0)
    sliced = (fwd >= SLICE[0]) & (fwd < SLICE[1])
    rand = (~sliced) & (rng.random(len(ns)) < 0.10)
    split = np.where(sliced, 2, np.where(rand, 1, 0))  # 0 train, 1 random held out, 2 held-out slice
    return dict(z=z, S=z["S"], A=z["A"], O=O, so=so, ao=ao, ns=ns, na=na, split=split, fwd=fwd)


def starts(D, which, H):
    """(state index, action index) of every window start with a full H-step future, in episodes of split `which`."""
    s_idx, a_idx = [], []
    for e in np.where(np.isin(D["split"], which))[0]:
        n = D["na"][e]
        if n < H + 1:
            continue
        k = np.arange(1, n - H + 1)
        s_idx.append(D["so"][e] + k)
        a_idx.append(D["ao"][e] + k)
    return np.concatenate(s_idx), np.concatenate(a_idx)


class SelfWM2(nn.Module):
    def __init__(self, S, A, D_s, O, fk: FKM.FK, width=1024, depth=4):
        super().__init__()
        self.ns, self.na, self.nd, self.no = WMM.Norm(S), WMM.Norm(A), WMM.Norm(D_s), WMM.Norm(O)
        self.f = WMM.mlp(2 * (E.S_DIM + A_DIM), E.S_DIM + 4, width=width, depth=depth)
        self.fk = fk

    def step(self, s_prev, a_prev, s, a):
        x = torch.cat([self.ns.fwd(s_prev), self.na.fwd(a_prev), self.ns.fwd(s), self.na.fwd(a)], -1)
        y = self.f(x)
        return s + self.nd.inv(y[..., : E.S_DIM]), self.no.inv(y[..., E.S_DIM :])

    def rollout(self, s_prev, a_prev, s0, actions):
        ss, oo = [], []
        sp, ap, s = s_prev, a_prev, s0
        for h in range(actions.shape[1]):
            s_next, o = self.step(sp, ap, s, actions[:, h])
            ss.append(s_next)
            oo.append(o)
            sp, ap, s = s, actions[:, h], s_next
        return torch.stack(ss, 1), torch.stack(oo, 1)

    def hands(self, q, side):
        """Acting jaw centre and head camera position, pelvis frame: q (..., 29), side (...) in {+1, -1}."""
        sh = q.shape[:-1]
        out = self.fk(q.reshape(-1, 29))
        right, left, cam = out["jaw_right"][0], out["jaw_left"][0], out["head_camera"][0]
        jaw = torch.where(side.reshape(-1, 1) > 0, right, left)
        return jaw.reshape(*sh, 3), cam.reshape(*sh, 3)


def build(D, dev):
    import json as _json

    sys.path[:0] = [str(E.REPO / "src")]
    from locomani.backends.scalebfm import runtime as R
    from locomani.backends.scalebfm.config import ScaleBfmBackendConfig

    meta = _json.load(open(R.META))
    sim = R.Sim(
        meta, backend=ScaleBfmBackendConfig(scene_dir=str(E.REPO / "outputs" / "scene" / "walk_mars"), object_name="")
    )
    fk = FKM.FK(sim.m, sim.qadr, device=dev)
    tr = np.where(D["split"] == 0)[0]
    s_rows = np.concatenate([np.arange(D["so"][e], D["so"][e] + D["ns"][e]) for e in tr])
    a_rows = np.concatenate([np.arange(D["ao"][e], D["ao"][e] + D["na"][e]) for e in tr])
    d_rows = np.concatenate([D["so"][e] + np.arange(D["na"][e]) for e in tr])
    Ds = D["S"][d_rows + 1] - D["S"][d_rows]
    return SelfWM2(D["S"][s_rows], D["A"][a_rows], Ds, D["O"][a_rows], fk).to(dev)


def batch(T, s_idx, a_idx, H):
    S, A, O = T
    ar = torch.arange(H, device=S.device)
    return (
        S[s_idx - 1],
        A[a_idx - 1],
        S[s_idx],
        A[a_idx[:, None] + ar],
        S[s_idx[:, None] + 1 + ar],
        O[a_idx[:, None] + ar],
    )


def train(iters: int, B: int, dev: str, data=DATA, out=WM2) -> None:
    torch.manual_seed(0)
    D = load_data(data)
    model = build(D, dev)
    T = tuple(torch.tensor(x, device=dev) for x in (D["S"], D["A"], D["O"]))
    si, ai = (torch.tensor(x, device=dev) for x in starts(D, [0], ROLL))
    vi, va = (torch.tensor(x[:8192], device=dev) for x in starts(D, [1], ROLL))
    opt = torch.optim.AdamW(model.f.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)

    def loss_of(s_idx, a_idx):
        sp, ap, s0, acts, sts, odo = batch(T, s_idx, a_idx, ROLL)
        ps, po = model.rollout(sp, ap, s0, acts)
        ls = ((model.ns.fwd(ps) - model.ns.fwd(sts)) ** 2).mean()
        lo = ((model.no.fwd(po) - model.no.fwd(odo)) ** 2).mean()
        side = acts[..., 0]
        pj, pc = model.hands(ps[..., :29], side)
        tj, tc = model.hands(sts[..., :29], side)
        lk = (((pj - tj) / 0.01) ** 2).mean() + (((pc - tc) / 0.01) ** 2).mean()  # in cm^2
        return ls + lo + 0.1 * lk, ls, lo, lk

    log, t0 = [], time.time()
    for it in range(iters):
        j = torch.randint(0, len(si), (B,), device=dev)
        l, ls, lo, lk = loss_of(si[j], ai[j])
        opt.zero_grad()
        l.backward()
        nn.utils.clip_grad_norm_(model.f.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 2000 == 0 or it == iters - 1:
            with torch.no_grad():
                v = [float(x) for x in loss_of(vi, va)]
            row = dict(
                it=it,
                train=float(l.detach()),
                val=v[0],
                val_s=v[1],
                val_o=v[2],
                val_kin_cm2=v[3],
                s=round(time.time() - t0, 1),
            )
            log.append(row)
            print(json.dumps(row), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    json.dump(dict(iters=iters, batch=B, episodes=int(len(D["ns"])), split=np.bincount(D["split"]).tolist(), log=log),
              open(out / "train.json", "w"), indent=1)  # fmt: skip


def load(dev: str, data=DATA, out=WM2):
    D = load_data(data)
    model = build(D, dev)
    model.load_state_dict(torch.load(out / "model.pt", map_location=dev))
    model.eval()
    return model, D


def evaluate(dev: str) -> None:
    model, D = load(dev)
    T = tuple(torch.tensor(x, device=dev) for x in (D["S"], D["A"], D["O"]))
    z = D["z"]
    res = {}
    for name, which in (("random_held_out", [1]), ("held_out_slice", [2])):
        res[name] = {}
        for H in (10, 20, 30):
            s_idx, a_idx = starts(D, which, H)
            ep = np.searchsorted(D["so"], s_idx, side="right") - 1
            errs = {k: [] for k in ("wm", "command", "static", "camera")}
            for c0 in range(0, len(s_idx), 4096):
                si = torch.tensor(s_idx[c0 : c0 + 4096], device=dev)
                ai = torch.tensor(a_idx[c0 : c0 + 4096], device=dev)
                with torch.no_grad():
                    sp, ap, s0, acts, sts, _ = batch(T, si, ai, H)
                    ps, _ = model.rollout(sp, ap, s0, acts)
                    side = acts[:, -1, 0]
                    pj, pc = model.hands(ps[:, -1, :29], side)
                    tj, tc = model.hands(sts[:, -1, :29], side)
                    j0, _ = model.hands(s0[:, :29], side)
                errs["wm"].append((pj - tj).norm(dim=-1).cpu().numpy())
                errs["camera"].append((pc - tc).norm(dim=-1).cpu().numpy())
                errs["command"].append((acts[:, -1, 1:4] - tj).norm(dim=-1).cpu().numpy())
                errs["static"].append((j0 - tj).norm(dim=-1).cpu().numpy())
            errs = {k: np.concatenate(v) for k, v in errs.items()}
            row = {k: dict(median_cm=round(100 * float(np.median(v)), 2), p90_cm=round(100 * float(np.percentile(v, 90)), 2))
                   for k, v in errs.items()}  # fmt: skip
            row["n"] = int(len(s_idx))
            # break the 1 s case down by side, attitude, walked
            if H == 10:
                for key, vals in (("side", z["side"]), ("att", z["att"]), ("walked", z["walked"])):
                    for v in np.unique(vals):
                        m = vals[ep] == v
                        row[f"wm_{key}={int(v)}_median_cm"] = round(100 * float(np.median(errs["wm"][m])), 2)
            res[name][f"{H / 10:.0f}s"] = row
    json.dump(res, open(WM2 / "eval.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["train", "eval"])
    ap.add_argument("--iters", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out", default=str(WM2))
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from pathlib import Path

    train(a.iters, a.batch, dev, Path(a.data), Path(a.out)) if a.what == "train" else evaluate(dev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
