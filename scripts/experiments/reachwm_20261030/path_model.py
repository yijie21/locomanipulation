"""The jaw on its way in (2026-11-07): how far off the straight line it is when it comes FRONT_M short of where the reach ends.

The self model of wm_ref.pt says where a reach ends. The grasp's pushes happen earlier: the arm swings across as it starts
to extend (the torso does not move), by about 1.5 cm some 6 cm in, settling over the next 12 cm, so on a short entry the
fingertips reach the box at the peak. This model learns that swing from the entries of data_grasp (REACH_GRASP=1): the same
inputs as wm_ref.pt, the same held-out resets, label = jaw at FRONT_M before the end minus the straight line from the
jaw's start to its end, heading frame, in cm. Those ran at 0.15 m/s from the measured jaw. In grasps run that way it tracks
the measured swing (correlation 0.68, 96 runs); at 0.08 m/s from the half-way start only where the members agree, so
grasp_exp.py uses it there alone, moving the via point against 100 % of it (g100 h35, picked on the short and calib sets).

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    REACHWM_GOAL_REF=1 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 $PY scripts/experiments/reachwm_20261030/path_model.py train
    # on the entries of the fixed planner (2026-11-08; held-out R2 0.53 and 0.25 across and up; used with g100 and no h)
    PATH_DATA=data_grasp2 PATH_SAVE=outputs/reachwm_20261030/path/path_swing2.pt REACHWM_GOAL_REF=1 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 \
        $PY scripts/experiments/reachwm_20261030/path_model.py train
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as DT  # noqa: E402

FRONT_M = 0.08  # the fingertips meet the box's front with the jaw centre about this far before the grasp point
MIN_RUN_M = 0.10  # entries that start closer to their end than this have no such point
MEMBERS, ITERS, BATCH = 5, 6000, 512
CLIP = 8.0  # scaled inputs beyond this are outside anything the data showed
# PATH_DATA / PATH_SAVE (2026-11-08): another entry dataset (data_grasp2: the entries as the grasp runs them) and its own file
SAVE = Path(os.environ.get("PATH_SAVE", DT.OUT / "path" / "path_swing.pt"))


def labels(D: dict) -> tuple[np.ndarray, np.ndarray]:
    """Entries with a point FRONT_M before their end: their index and the jaw's offset from the straight line there, cm."""
    n, so = D["n_steps"].astype(int), D["so"]
    W, side, start = D["W"].astype(np.float64), D["side"], D["start"].astype(np.float64)
    idx, lab = [], []
    for i in np.where((D["chain"] == 1) & (D["fell"] == 0))[0]:
        k0, last = so[i] + int(D["k0"][i]), so[i] + n[i] - 1
        jaw = W[k0 : last + 1, 4:7] if side[i] > 0 else W[k0 : last + 1, 7:10]
        p = DT.hv(jaw - np.r_[start[i, :2], 0.0], np.full(len(jaw), start[i, 3]))
        a, b = p[0], p[-1]
        if b[0] - a[0] < MIN_RUN_M:
            continue
        hit = np.where(p[:, 0] >= b[0] - FRONT_M)[0]
        if not len(hit) or hit[0] == 0:
            continue
        r = hit[0]
        u = (b[0] - FRONT_M - p[r - 1, 0]) / max(p[r, 0] - p[r - 1, 0], 1e-6)
        at = p[r - 1] + u * (p[r] - p[r - 1])
        line = a + (b[0] - FRONT_M - a[0]) / (b[0] - a[0]) * (b - a)
        idx.append(i)
        lab.append(100 * (at - line)[1:])
    return np.array(idx), np.array(lab, np.float32)


class Member(nn.Module):
    def __init__(self, nin: int, width: int = 256, depth: int = 3, drop: float = 0.1):
        super().__init__()
        L, d = [], nin
        for _ in range(depth):
            L += [nn.Linear(d, width), nn.SiLU(), nn.Dropout(drop)]
            d = width
        self.f = nn.Sequential(*L, nn.Linear(d, 2))

    def forward(self, x):
        return self.f(x)


class PathModel(nn.Module):
    """Input [x0, u] as wm_ref.pt takes them; output the swing across (y) and up (z), cm, and the members' spread."""

    def __init__(self, members, mu, sd):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)

    def predict(self, x0: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        o = torch.stack([m(self.scaled(x0, u)) for m in self.members])
        return o.mean(0), o.std(0, unbiased=False)

    def scaled(self, x0: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return ((torch.cat([x0, u], -1) - self.mu) / self.sd).clamp(-CLIP, CLIP)


def load(path: str | Path = SAVE) -> PathModel:
    d = torch.load(path, map_location="cpu", weights_only=True)
    pm = PathModel([Member(d["nin"]) for _ in range(MEMBERS)], torch.zeros(d["nin"]), torch.ones(d["nin"]))
    pm.load_state_dict(d["state"])
    return pm.eval()


def report(name: str, y: np.ndarray, p: np.ndarray) -> dict:
    e = y - p
    r2 = 1 - (e**2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)
    out = dict(n=len(y), raw_sd=y.std(0).round(3).tolist(), resid_sd=e.std(0).round(3).tolist(), r2=r2.round(3).tolist(),
               over_1p5_raw=round(float((np.abs(y[:, 0]) > 1.5).mean() * 100), 1),
               over_1p5_after=round(float((np.abs(e[:, 0]) > 1.5).mean() * 100), 1))  # fmt: skip
    print(name, json.dumps(out), flush=True)
    return out


def train(dev: str = "cuda") -> None:
    D = DT.load(str(DT.OUT / os.environ.get("PATH_DATA", "data_grasp") / "all.npz"))
    idx, lab = labels(D)
    bucket = D["bucket"][idx]
    print(f"{len(D['n_steps'])} segments, {len(idx)} entries with a point {FRONT_M} m before the end", flush=True)
    X = np.c_[D["x0"][idx], D["u"][idx]].astype(np.float32)
    tr, va, te = np.where(bucket < 8)[0], np.where(bucket == 8)[0], np.where(bucket == 9)[0]
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd < 1e-3] = 1.0  # constant in the data (level jaw): live values differ a little and must not blow up
    T = lambda a: torch.as_tensor(a, device=dev)  # noqa: E731
    Xn, Y = T((X - mu) / sd), T(lab)
    members, g = [], torch.Generator(device="cpu").manual_seed(0)
    for k in range(MEMBERS):
        torch.manual_seed(k)
        m = Member(X.shape[1]).to(dev)
        boot = T(tr[torch.randint(len(tr), (len(tr),), generator=g).numpy()])
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS)
        best, best_sd = 1e9, None
        for it in range(ITERS):
            m.train()
            j = boot[torch.randint(len(boot), (BATCH,), device=dev)]
            loss = ((m(Xn[j]) - Y[j]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            if it % 200 == 199:
                m.eval()
                with torch.no_grad():
                    v = float(((m(Xn[T(va)]) - Y[T(va)]) ** 2).mean())
                if v < best:
                    best, best_sd = v, {a: b.clone() for a, b in m.state_dict().items()}
        m.load_state_dict(best_sd)
        members.append(m.eval())
        print(f"member {k} best val mse {best:.3f}", flush=True)
    pm = PathModel(members, T(mu), T(sd)).eval()
    with torch.no_grad():
        P = pm.predict(T(D["x0"][idx].astype(np.float32)), T(D["u"][idx].astype(np.float32)))[0].cpu().numpy()
    print("constant predictor would give R2 0; tr", (report("train", lab[tr], P[tr])["r2"]), flush=True)
    rep = dict(val=report("val", lab[va], P[va]), test=report("test", lab[te], P[te]))
    # the placements that matter: shelf-height reaches on the right hand, short entries
    t = D["target"][idx]
    sel = (t[:, 2] > 0.85) & (t[:, 2] < 1.05) & (D["side"][idx] > 0)
    rep["test_mid_right"] = report("test z 0.85-1.05 right", lab[te][sel[te]], P[te][sel[te]])
    sel = (t[:, 2] > 0.85) & (t[:, 2] < 1.05) & (D["side"][idx] < 0)
    rep["test_mid_left"] = report("test z 0.85-1.05 left", lab[te][sel[te]], P[te][sel[te]])
    print(
        "mean swing on test, right / left, y cm",
        lab[te][D["side"][idx][te] > 0, 0].mean().round(2),
        lab[te][D["side"][idx][te] < 0, 0].mean().round(2),
    )
    SAVE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=pm.to("cpu").state_dict(), nin=X.shape[1]), SAVE)
    json.dump(rep, open(SAVE.with_suffix(".json"), "w"), indent=1)
    print("saved", SAVE, flush=True)


if __name__ == "__main__":
    train()
