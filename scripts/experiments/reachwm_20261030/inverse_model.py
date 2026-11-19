"""The direct inverse model: the self model's same-data baseline (2026-11-19, simple_bench_20261119).

From the state at a reach's start and the palm position wanted at its end, it gives the command. It is trained by
hindsight on the self model's own training split: every recorded reach says which command (target, heading frame)
brought the palm to where it ended (pos_f), from that state.

Inputs:
  - the 206 state features of x0 (not the plan features, which are functions of the command);
  - the wanted final palm;
  - the wrist command's rot6;
  - the hand;
  - standing, or the pelvis height;
  - whether the reach is a slow entry (the third reach of a REACH_GRASP_ROOM reset).
Output: the command's offset from the wanted palm (target - pos_f).

The model is an ensemble of MLPs. Segments where the robot fell are left out. The best state on validation is kept.

    REACHWM_GOAL_REF=1 $PY scripts/experiments/reachwm_20261030/inverse_model.py --data <spec> --out <dir>/inverse.pt
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as DT  # noqa: E402

GRASPR_ID0 = 1300000  # data_graspr_odo's reset ids
X_STATE = 206


def entry_flag(D: dict) -> np.ndarray:
    r = D["reset"]
    pos = np.zeros(len(r), int)
    for j in range(1, len(r)):
        pos[j] = pos[j - 1] + 1 if r[j] == r[j - 1] else 0
    return ((r >= GRASPR_ID0) & (r < GRASPR_ID0 + 100000) & (pos == 2)).astype(np.float32)


def features(x0s, goal, rot6, sd, stand, pz, entry) -> np.ndarray:
    """Rows of the model's input; every argument has one row per reach."""
    col = lambda v: np.asarray(v, np.float32).reshape(-1, 1)  # noqa: E731
    return np.c_[np.asarray(x0s, np.float32), np.asarray(goal, np.float32), np.asarray(rot6, np.float32), col(sd),
                 col(stand), col(pz), col(entry)].astype(np.float32)  # fmt: skip


class Net(nn.Module):
    def __init__(self, din: int, dout: int, width: int, depth: int):
        super().__init__()
        layers, d = [], din
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        self.f = nn.Sequential(*layers, nn.Linear(d, dout))
        for k, n in (("mu_x", din), ("sd_x", din), ("mu_y", dout), ("sd_y", dout)):
            self.register_buffer(k, torch.zeros(n) if k.startswith("mu") else torch.ones(n))

    def forward(self, x):
        return self.mu_y + self.sd_y * self.f((x - self.mu_x) / self.sd_x)


def load(path: str) -> list:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    nets = []
    for sd_ in ck["members"]:
        n = Net(ck["din"], ck["dout"], ck["width"], ck["depth"])
        n.load_state_dict(sd_)
        nets.append(n.eval())
    return nets


def predict(nets: list, x: np.ndarray) -> np.ndarray:
    """The members' mean offset (m, heading frame) for the input rows."""
    with torch.no_grad():
        t = torch.as_tensor(x, dtype=torch.float32)
        return torch.stack([n(t) for n in nets]).mean(0).numpy().astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--members", type=int, default=3)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = DT.load(a.data)
    u = D["u"]
    X = features(D["x0"][:, :X_STATE], D["pos_f"], D["rot6"], u[:, 0], u[:, 10], u[:, 11], entry_flag(D))
    Y = (D["target"] - D["pos_f"]).astype(np.float32)
    sp = DT.splits(D)
    ok = ~D["fell"].astype(bool)
    tr, va = sp["train"][ok[sp["train"]]], sp["val"][ok[sp["val"]]]
    print(f"{len(D['n_steps'])} segments, train {len(tr)} val {len(va)}, input {X.shape[1]}", flush=True)
    mu_x, sd_x = X[tr].mean(0), X[tr].std(0) + 1e-6
    mu_y, sd_y = Y[tr].mean(0), Y[tr].std(0) + 1e-6
    Xt, Yt = torch.as_tensor(X, device=dev), torch.as_tensor(Y, device=dev)
    tri, vai = torch.as_tensor(tr, device=dev), torch.as_tensor(va, device=dev)
    members = []
    for s in range(a.members):
        torch.manual_seed(s)
        net = Net(X.shape[1], 3, a.width, a.depth).to(dev)
        for k, v in (("mu_x", mu_x), ("sd_x", sd_x), ("mu_y", mu_y), ("sd_y", sd_y)):
            getattr(net, k).copy_(torch.as_tensor(v, device=dev))
        opt = torch.optim.Adam(net.parameters(), lr=a.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
        best, t0 = (np.inf, None), time.time()
        for it in range(1, a.iters + 1):
            i = tri[torch.randint(len(tri), (a.batch,), device=dev)]
            loss = (((net(Xt[i]) - Yt[i]) / net.sd_y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if it % 2000 == 0:
                with torch.no_grad():
                    e = 100 * (net(Xt[vai]) - Yt[vai]).norm(dim=1)
                med = float(e.median())
                if med < best[0]:
                    best = (med, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
                print(
                    f"member {s} iter {it} val median {med:.3f} cm (best {best[0]:.3f}) {time.time() - t0:.0f}s",
                    flush=True,
                )
        members.append(best[1])
    nets = []
    for sd_ in members:
        n = Net(X.shape[1], 3, a.width, a.depth)
        n.load_state_dict(sd_)
        nets.append(n.eval())
    e = 100 * np.linalg.norm(predict(nets, X[va]) - Y[va], axis=1)
    e0 = 100 * np.linalg.norm(Y[va], axis=1)
    print(f"ensemble val: command error median {np.median(e):.3f} cm (p90 {np.percentile(e, 90):.3f}); "
          f"the offset itself median {np.median(e0):.3f} cm", flush=True)  # fmt: skip
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(members=members, din=X.shape[1], dout=3, width=a.width, depth=a.depth, data=a.data), a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
