"""Train the reach world-model candidates (models.py) on gen_reach.py's data. Same budget for every model: AdamW lr 1e-3,
weight decay 1e-5, cosine schedule, grad clip 1.0; end-state models 30k iterations x batch 1024, step models 40k x 512
windows; val final-position error every 2k iterations, the best state is kept. Fixed seeds.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 $PY scripts/experiments/reachwm_20261030/train.py E2_residual --frac 0.3
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 $PY scripts/experiments/reachwm_20261030/train.py all
    ... train.py all --iters 300 --out-root /tmp/x     # tiny smoke test

Writes outputs/reachwm_20261030/models/<name>_f<frac>/{model.pt, train.json}. `all` = every model at frac 1.0, plus
E2_residual, E3_ensemble, S1_mlp (and the cheap B1_knn) at 0.1 and 0.3.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as DT  # noqa: E402
import models as M  # noqa: E402

MODELS = DT.OUT / "models"
EVAL_EVERY = 2000
VAL_CAP_STEP = 1024  # val segments for step models (a val pass is a full rollout)


def model_dir(name: str, frac: float, root: Path = MODELS) -> Path:
    return root / f"{name}_f{frac:g}"


def train(name, D, GD, frac=1.0, seed=0, iters=None, batch=None, dev="cuda", root=MODELS, init=None, lr=1e-3) -> dict:
    """init (2026-11-04, the student controller): a model.pt to start from, normalization included (its state replaces
    the one built from D); its val error is the iteration-0 entry and the state to beat."""
    sp = DT.splits(D, frac)
    step = name in M.STEP_NAMES
    iters = iters or (40000 if step else 30000)
    batch = batch or (512 if step else 1024)
    torch.manual_seed(seed)
    tr = torch.tensor(sp["train"], device=dev)
    va = sp["val"][D["fell"][sp["val"]] == 0]
    va = torch.tensor(va[:VAL_CAP_STEP] if step else va, device=dev)
    model = M.build(name, D, sp["train"], dev, seed)
    if init:
        model.load_state_dict(torch.load(init, map_location=dev)["state"])
    out = model_dir(name, frac, root)
    out.mkdir(parents=True, exist_ok=True)
    log, best, best_sd, t0 = [], float("inf"), None, time.time()

    def val() -> float:
        model.eval()
        e = float(M.pos_error_cm(model, GD, va).median())
        model.train()
        return e

    n_train = len(sp["train"])
    if name == "B1_knn":
        model.fit(GD, tr)
        best, best_sd = val(), {k: v.clone() for k, v in model.state_dict().items()}
        n_train = int((GD["fell"][tr] == 0).sum())
        log.append(dict(it=0, val_median_cm=best))
    elif name != "B0_command":
        if init:
            best, best_sd = val(), {k: v.detach().clone() for k, v in model.state_dict().items()}
            log.append(dict(it=0, val_median_cm=best, init=str(init)))
            print(name, frac, json.dumps(log[-1]), flush=True)
        model.prepare(GD, tr, seed)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        model.train()
        for it in range(iters):
            l = model.sample_loss(GD, batch)
            opt.zero_grad()
            l.backward()
            for g in model.groups():
                nn.utils.clip_grad_norm_(g, 1.0)
            opt.step()
            sched.step()
            if (it + 1) % EVAL_EVERY == 0 or it == iters - 1:
                v = val()
                row = dict(it=it + 1, train=float(l.detach()), val_median_cm=v, s=round(time.time() - t0, 1))
                log.append(row)
                print(name, frac, json.dumps(row), flush=True)
                if v < best:
                    best, best_sd = v, {k: x.detach().clone() for k, x in model.state_dict().items()}
        model.load_state_dict(best_sd)
    else:
        best, best_sd = val(), model.state_dict()
        log.append(dict(it=0, val_median_cm=best))
    torch.save(dict(state=best_sd, n_train=n_train), out / "model.pt")
    info = dict(name=name, frac=frac, seed=seed, iters=iters if name not in ("B0_command", "B1_knn") else 0, batch=batch,
                n_train=int(len(sp["train"])), n_val=int(len(va)), best_val_median_cm=best, log=log,
                data_files=len(D["files"]), n_segments=int(len(D["n_steps"])), seconds=round(time.time() - t0, 1),
                init=str(init) if init else None, lr=lr)  # fmt: skip
    json.dump(info, open(out / "train.json", "w"), indent=1)
    print(f"{name} frac {frac}: best val median {best:.3f} cm ({info['seconds']} s) -> {out}", flush=True)
    return info


def load_model(name, frac, D, dev, root=MODELS):
    """Rebuild a trained model (normalization from the same splits) and load its weights."""
    ck = torch.load(model_dir(name, frac, root) / "model.pt", map_location=dev)
    m = M.build(name, D, DT.splits(D, frac)["train"], dev, n_train=ck["n_train"])
    m.load_state_dict(ck["state"])
    return m.eval()


def jobs(what: str, frac: float):
    if what != "all":
        return [(what, frac)]
    js = [(n, 1.0) for n in M.NAMES]
    js += [(n, f) for f in (0.1, 0.3) for n in ("B1_knn", "E2_residual", "E3_ensemble", "S1_mlp")]
    return js


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["all"] + M.NAMES)
    ap.add_argument("--frac", type=float, default=1.0, choices=list(DT.FRACS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--data", default=None, help="glob of chunk files (default: all.npz, else every chunk)")
    ap.add_argument("--out-root", default=str(MODELS))
    ap.add_argument("--init", default=None, help="model.pt to fine-tune from (normalization included)")
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = DT.load(a.data)
    GD = M.tensors(D, dev, step=True)
    print(f"{len(D['n_steps'])} segments, splits { {k: len(v) for k, v in DT.splits(D).items()} }", flush=True)
    for name, frac in jobs(a.what, a.frac):
        train(name, D, GD, frac, a.seed, a.iters, a.batch, dev, Path(a.out_root), a.init, a.lr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
