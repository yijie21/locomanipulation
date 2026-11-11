"""Self world model of the minimal proof: what frozen ScaleBFM + the body do under a jaw-centre command, learned from
gen_data.py's episodes. Only self quantities go in or come out (env.py's docstring lists them).

  dynamics  f([s_{k-1}, a_{k-1}, s_k, a_k]) -> (s_{k+1} - s_k, odometry_k)        MLP, multi-step rollout loss
  kinematics g(q_k) -> jaw centre and head camera in the pelvis frame             MLP (stands in for exact FK here)

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/selfwm_20261029/wm.py train
    CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/selfwm_20261029/wm.py eval

Writes outputs/selfwm_20261029/wm/{model.pt, train.json, eval.json}. Held out: the last 15 % of episode ids.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import env as E
import numpy as np
import torch
import torch.nn as nn

WM = E.OUT / "wm"
HOLD = 0.15
ROLL = 8  # training rollout length (10 Hz steps)


def episodes(path=E.OUT / "data" / "all.npz") -> list[dict]:
    z = dict(np.load(path))  # each key once: NpzFile decompresses a key on every access
    out, o, oa = [], 0, 0
    for i in range(len(z["id"])):
        ns, na = int(z["n_states"][i]), int(z["n_actions"][i])
        S, K, W = (z[k][o : o + ns].copy() for k in ("S", "K", "W"))
        A = z["A"][oa : oa + na].copy()
        o, oa = o + ns, oa + na
        if na < ROLL + 2 or len(S) != na + 1:
            continue
        O = np.array([E.odometry(W[k], W[k + 1]) for k in range(na)], dtype=np.float32)
        out.append(dict(id=int(z["id"][i]), S=S, K=K, W=W, A=A, O=O))
    return out


def split(eps: list[dict]):
    ids = sorted(e["id"] for e in eps)
    cut = ids[int(len(ids) * (1 - HOLD))]
    return [e for e in eps if e["id"] < cut], [e for e in eps if e["id"] >= cut]


def mlp(i: int, o: int, width: int = 512, depth: int = 4) -> nn.Sequential:
    layers, d = [], i
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    return nn.Sequential(*layers, nn.Linear(d, o))


class Norm(nn.Module):
    def __init__(self, x: np.ndarray):
        super().__init__()
        self.register_buffer("mu", torch.tensor(x.mean(0), dtype=torch.float32))
        self.register_buffer("sd", torch.tensor(np.maximum(x.std(0), 1e-3), dtype=torch.float32))

    def fwd(self, x):
        return (x - self.mu) / self.sd

    def inv(self, x):
        return x * self.sd + self.mu


class SelfWM(nn.Module):
    def __init__(self, train: list[dict]):
        super().__init__()
        S = np.concatenate([e["S"] for e in train])
        A = np.concatenate([e["A"] for e in train])
        D = np.concatenate([e["S"][1:] - e["S"][:-1] for e in train])
        O = np.concatenate([e["O"] for e in train])
        K = np.concatenate([e["K"] for e in train])
        self.ns, self.na, self.nd, self.no, self.nk = Norm(S), Norm(A), Norm(D), Norm(O), Norm(K)
        self.nq = Norm(S[:, :29])
        self.f = mlp(2 * (E.S_DIM + E.A_DIM), E.S_DIM + E.ODO_DIM)
        self.g = mlp(29, E.KIN_DIM, width=256, depth=3)

    def step(self, s_prev, a_prev, s, a):
        x = torch.cat([self.ns.fwd(s_prev), self.na.fwd(a_prev), self.ns.fwd(s), self.na.fwd(a)], -1)
        y = self.f(x)
        return s + self.nd.inv(y[..., : E.S_DIM]), self.no.inv(y[..., E.S_DIM :])

    def kin(self, s):
        return self.nk.inv(self.g(self.nq.fwd(s[..., :29])))

    def rollout(self, s_prev, a_prev, s0, actions):
        """Open loop over actions (B, H, 6) from history (s_prev, a_prev, s0): states (B, H, 65), odometry (B, H, 4)."""
        ss, oo = [], []
        sp, ap, s = s_prev, a_prev, s0
        for h in range(actions.shape[1]):
            a = actions[:, h]
            s_next, o = self.step(sp, ap, s, a)
            ss.append(s_next)
            oo.append(o)
            sp, ap, s = s, a, s_next
        return torch.stack(ss, 1), torch.stack(oo, 1)


def windows(eps: list[dict], H: int):
    """(s_prev, a_prev, s0, actions[H], states[H], odo[H], kin[H]) for every start with a full window."""
    cols = [[] for _ in range(7)]
    for e in eps:
        S, A, O, K = e["S"], e["A"], e["O"], e["K"]
        for k in range(0, len(A) - H + 1):
            kp = max(k - 1, 0)
            for c, v in zip(
                cols, (S[kp], A[kp], S[k], A[k : k + H], S[k + 1 : k + 1 + H], O[k : k + H], K[k + 1 : k + 1 + H])
            ):
                c.append(v)
    return [torch.tensor(np.array(c)) for c in cols]


def train(iters: int, batch: int, dev: str) -> None:
    torch.manual_seed(0)
    eps = episodes()
    tr, te = split(eps)
    model = SelfWM(tr).to(dev)
    W = [w.to(dev) for w in windows(tr, ROLL)]
    Wv = [w.to(dev) for w in windows(te, ROLL)]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    allS = torch.tensor(np.concatenate([e["S"] for e in tr]), device=dev)
    allK = torch.tensor(np.concatenate([e["K"] for e in tr]), device=dev)

    def loss_of(w, idx):
        sp, ap, s0, acts, sts, odo, kin = (x[idx] for x in w)
        ps, po = model.rollout(sp, ap, s0, acts)
        ls = ((model.ns.fwd(ps) - model.ns.fwd(sts)) ** 2).mean()
        lo = ((model.no.fwd(po) - model.no.fwd(odo)) ** 2).mean()
        lk = ((model.nk.fwd(model.kin(ps)) - model.nk.fwd(kin)) ** 2).mean()
        return ls + lo + lk, ls, lo, lk

    t0, log = time.time(), []
    n = len(W[0])
    for it in range(iters):
        idx = torch.randint(0, n, (batch,), device=dev)
        l, ls, lo, lk = loss_of(W, idx)
        j = torch.randint(0, len(allS), (batch,), device=dev)
        lg = ((model.nk.fwd(model.kin(allS[j])) - model.nk.fwd(allK[j])) ** 2).mean()
        opt.zero_grad()
        (l + lg).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 1000 == 0 or it == iters - 1:
            with torch.no_grad():
                vi = torch.arange(min(len(Wv[0]), 4096), device=dev)
                vl = [float(x) for x in loss_of(Wv, vi)]
            row = dict(it=it, train=float(l), val=vl[0], val_s=vl[1], val_o=vl[2], val_k=vl[3], kin=float(lg),
                       s=round(time.time() - t0, 1))  # fmt: skip
            log.append(row)
            print(json.dumps(row), flush=True)
    WM.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), WM / "model.pt")
    json.dump(dict(iters=iters, batch=batch, train_eps=len(tr), test_eps=len(te), windows=n, log=log),
              open(WM / "train.json", "w"), indent=1)  # fmt: skip


def load(dev: str):
    eps = episodes()
    tr, te = split(eps)
    model = SelfWM(tr).to(dev)
    model.load_state_dict(torch.load(WM / "model.pt", map_location=dev))
    model.eval()
    return model, tr, te


def tilt_rot(g: np.ndarray) -> np.ndarray:
    """The yaw-free rotation R_t with R_t^T (0, 0, -1) = g (gravity seen in the pelvis frame)."""
    a, b = g / np.linalg.norm(g), np.array([0.0, 0.0, -1.0])
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def evaluate(dev: str) -> None:
    """Open-loop prediction on the held-out episodes from every start with a full horizon, against two references:
    'command' (the jaw is where it was told to go) and 'static' (the jaw stays put)."""
    model, tr, te = load(dev)
    res = {}
    for H in (10, 20, 30):
        sp, ap, s0, acts, kin_true, w0, w_end, a_last, k0 = [], [], [], [], [], [], [], [], []
        for e in te:
            S, A, K, W = e["S"], e["A"], e["K"], e["W"]
            for k in range(1, len(A) - H + 1):
                sp.append(S[k - 1]), ap.append(A[k - 1]), s0.append(S[k]), acts.append(A[k : k + H])
                kin_true.append(K[k + H, :3]), w0.append(W[k]), w_end.append(W[k + H, 4:7])
                a_last.append(A[k + H - 1, :3]), k0.append(K[k, :3])
        t = lambda x: torch.tensor(np.array(x), device=dev)  # noqa: E731
        with torch.no_grad():
            ps, po = model.rollout(t(sp), t(ap), t(s0), t(acts))
            kin = model.kin(ps[:, -1]).cpu().numpy()
        ps, po = ps.cpu().numpy(), po.cpu().numpy()
        kin_true, w0, w_end = np.array(kin_true), np.array(w0, dtype=float), np.array(w_end)
        wm_world = []
        for i in range(len(w0)):
            p, yaw = w0[i, :3].copy(), float(w0[i, 3])
            for h in range(H):
                p = p + E.rot_z(yaw) @ po[i, h, :3]
                yaw = yaw + float(po[i, h, 3])
            Rp = E.rot_z(yaw) @ tilt_rot(ps[i, -1, 58:61].astype(float))
            wm_world.append(np.linalg.norm(p + Rp @ kin[i, :3] - w_end[i]))
        errs = dict(
            wm_pel=np.linalg.norm(kin[:, :3] - kin_true, axis=1),
            wm_world=np.array(wm_world),
            command=np.linalg.norm(np.array(a_last) - kin_true, axis=1),
            static=np.linalg.norm(np.array(k0) - kin_true, axis=1),
        )
        res[f"{H / 10:.0f}s"] = {
            k: dict(
                median_cm=round(100 * float(np.median(v)), 2),
                p90_cm=round(100 * float(np.percentile(v, 90)), 2),
                n=len(v),
            )  # fmt: skip
            for k, v in errs.items()
        }
    json.dump(res, open(WM / "eval.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["train", "eval"])
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.what == "train":
        train(a.iters, a.batch, dev)
    else:
        evaluate(dev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
