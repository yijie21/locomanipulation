"""Nonlinear (2-layer MLP) version of probe_visual_grounding_20260923.py on its cached DINOv2 features; rounds held out.

third_party/Isaac-GR00T/.venv/bin/python scripts/experiments/probe_visual_grounding_mlp_20260923.py [--extra-only]
"""

import json
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts/experiments")
from pathlib import Path

import probe_visual_grounding_20260923 as pv

run = Path("outputs/runs/box_room_20260922")
rounds = sorted(p for p in (run / "observations").iterdir() if p.is_dir())
data = [pv.read_round(d) for d in rounds]
imgs = list(np.load("outputs/diagnostics/visual_grounding_20260923/box_room/features.npz").values())
P, I, H, Y, ph, g = [], [], [], [], [], []
for k, (rows, img) in enumerate(zip(data, imgs)):
    for t in range(len(rows) - pv.AHEAD):
        r = rows[t]
        P.append(r["prop"])
        I.append(img[t])
        H.append(r["hidden"])
        Y.append(rows[t + pv.AHEAD]["ref"][pv.ARM] - r["ref"][pv.ARM])
        ph.append(pv.PHASES.index(r["phase"]))
        g.append(k)
P, I, H, Y = map(np.asarray, (P, I, H, Y))
ph = np.array(ph)
g = np.array(g)


def fit(X):
    pred = np.zeros_like(Y)
    for te in pv.grouped_folds(g):
        tr = ~te
        Xtr, Xte = pv.standardise(X[tr], X[te])
        mu, sd = Y[tr].mean(0), Y[tr].std(0) + 1e-6
        torch.manual_seed(0)
        m = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], 256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, Y.shape[1]),
        )
        o = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2)
        xt = torch.tensor(Xtr, dtype=torch.float32)
        yt = torch.tensor((Y[tr] - mu) / sd, dtype=torch.float32)
        for ep in range(300):
            m.train()
            idx = torch.randperm(len(xt))
            for i in range(0, len(xt), 256):
                b = idx[i : i + 256]
                l = ((m(xt[b]) - yt[b]) ** 2).mean()
                o.zero_grad()
                l.backward()
                o.step()
        m.eval()
        pred[te] = m(torch.tensor(Xte, dtype=torch.float32)).detach().numpy() * sd + mu
    return pred


out = {}
for name, X in (
    {"P": P, "P+I": np.c_[P, I], "P+I+H": np.c_[P, I, H]} if "--extra-only" not in sys.argv else {}
).items():
    p = fit(X)
    out[name] = {"all": pv.r2(Y, p), **{pv.PHASES[k]: pv.r2(Y[ph == k], p[ph == k]) for k in (0, 1, 3)}}
    print(name, {k: round(v, 3) for k, v in out[name].items()}, flush=True)
json.dump(out, open("outputs/diagnostics/visual_grounding_20260923/box_room/mlp_report.json", "w"), indent=1)

# The expert's own reach state: the hand targets it is servoing to and its closure progress.
X = []
for rows_d in rounds:
    obs = [json.loads(l) for l in (rows_d / "observations.jsonl").read_text().splitlines()]
    for o in obs[: -pv.AHEAD]:
        tr = np.asarray(o.get("tool_reference") or np.full((2, 3), np.nan), float).ravel()
        X.append(np.r_[tr, o.get("closure") or 0.0])
E = np.asarray(X)
for name, Xs in {"P+I+E": np.c_[P, I, E], "P+E": np.c_[P, E]}.items():
    p = fit(Xs)
    out[name] = {"all": pv.r2(Y, p), **{pv.PHASES[k]: pv.r2(Y[ph == k], p[ph == k]) for k in (0, 1, 3)}}
    print(name, {k: round(v, 3) for k, v in out[name].items()}, flush=True)
json.dump(out, open("outputs/diagnostics/visual_grounding_20260923/box_room/mlp_report.json", "w"), indent=1)
