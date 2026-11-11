"""Walk model (2026-11-03, after approach_eval v2): from the state before the walk and the walk command, predict where the
open walk (gen_reach.open_walk, stop correction off) really stops, off the command. Labels: walk_err of data_approach and
data_approach0 = (forward, left, turn) the pelvis ended off the commanded walk, start frame, measured after the post-walk
hold (truth, training only). approach_eval.py's post_wk then believes it stands at the commanded walk plus this prediction.
2026-11-09, BFM_ROOT=odo models (WALK_SUFFIX=_odo): validation 1.38 full, 2.20 cmd, 0.60 post, 0.53 seq cm; 300-case result in approach_eval.py.

  inputs   the 206 pre-walk start features (data.py x0: 3 frames of S, the acting jaw pose, walked, chain) and the walk
           (forward, left, sin turn, cos turn - 1, has turn: any turn adds a turn-in-place that moves the feet)
  output   (forward cm, left cm, turn deg); 5 MLPs (4 x 512, SiLU) on bootstrap resamples, mean and spread
  cmd      the same ensemble on the walk features only (how much is the command's bias, how much the start state)
  post     (2026-11-03, the user: "走完再喂一遍腿的读数") also the body after the walk: the 3 frames of S at the end of the
           post-walk hold (the reach's start), the robot's own readings
  seq      also the readings during the walk, from the sighting to the end of that hold: legs and waist (15 joints,
           position and velocity), gravity, angular velocity and pelvis height per frame, resampled to SEQ_T points, and
           the walk's duration; a learned leg odometry
S's joints are in the controller's order (legs and waist interleaved): LEGS picks them by name.

Split by reset (data.hash_bucket): buckets 0-7 train, 8 val, 9 test. Fallen segments out.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/reachwm_20261030/walk_model.py t     # train full + cmd, pickle
    $PY scripts/experiments/reachwm_20261030/walk_model.py o                            # offline tables
Writes outputs/reachwm_20261030/approach/walk_model.pt, walk_model_cmd.pt, walk_offline.{json,md}, logs/train_walk.log.
"""

from __future__ import annotations

import os

os.environ.setdefault("REACHWM_GOAL_REF", "1")
os.environ.setdefault("REACHWM_APPROACH", "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "selfwm_20261029"))
sys.path.insert(0, str(HERE))
import data as DT  # noqa: E402
import wm as WMM  # noqa: E402

OUT = DT.OUT / "approach"
SPEC = ",".join(
    str(DT.OUT / f"data_{k}" / "all.npz") for k in os.environ.get("WALK_DATA", "approach,approach0").split(",")
)
_SFX = os.environ.get(
    "WALK_SUFFIX", ""
)  # another set of models (2026-11-09: BFM_ROOT=odo, WALK_DATA=approach_odo,approach0_odo, WALK_SUFFIX=_odo)
WALK_MODEL, WALK_MODEL_CMD = OUT / f"walk_model{_SFX}.pt", OUT / f"walk_model_cmd{_SFX}.pt"
WALK_MODEL_POST, WALK_MODEL_SEQ = OUT / f"walk_model_post{_SFX}.pt", OUT / f"walk_model_seq{_SFX}.pt"
ENS, ITERS, BATCH, EVAL_EVERY = 5, 20000, 512, 1000
N_BASE, N_WALK, N_POST = 206, 5, 195
SEQ_T = 32
_JN = json.load(open(DT.E.REPO / "assets" / "scalebfm" / "model_22200_tensorrt_metadata.json"))["joint_names"]
LEGS = [i for i, n in enumerate(_JN) if any(k in n for k in ("hip", "knee", "ankle", "waist"))]
SEQ_COLS = (
    LEGS + [29 + i for i in LEGS] + list(range(58, 65))
)  # q, dq of legs and waist; gravity, angular velocity, height
N_SEQ = SEQ_T * len(SEQ_COLS) + 1


def walk_feats(walk: np.ndarray) -> np.ndarray:
    w = np.asarray(walk, np.float64).reshape(-1, 3)
    return np.c_[DT.walk_features(w), (w[:, 2] != 0.0).astype(np.float64)].astype(np.float32)


def features(x0b: np.ndarray, walk: np.ndarray, post=None, seq=None) -> np.ndarray:
    """(N, 211) the pre-walk start features and the walk; with post (N, 195) and seq (N, N_SEQ) appended when given
    (zeros otherwise: the pre-walk models never read those columns)."""
    x = np.c_[np.asarray(x0b, np.float32).reshape(-1, N_BASE), walk_feats(walk)]
    n = len(x)
    post = np.zeros((n, N_POST), np.float32) if post is None else np.asarray(post, np.float32).reshape(n, N_POST)
    seq = np.zeros((n, N_SEQ), np.float32) if seq is None else np.asarray(seq, np.float32).reshape(n, N_SEQ)
    return np.c_[x, post, seq].astype(np.float32)


def post_feats(S_tail: np.ndarray) -> np.ndarray:
    """The body after the walk: the last 3 rows of S, newest first (as data.py's x0)."""
    S_tail = np.asarray(S_tail, np.float32)
    return np.r_[S_tail[-1], S_tail[-2], S_tail[-3]].astype(np.float32)


def seq_feats(S_walk: np.ndarray) -> np.ndarray:
    """Rows of S from the sighting to the end of the post-walk hold (10 Hz), resampled to SEQ_T, and the duration (s)."""
    S_walk = np.asarray(S_walk, np.float64)[:, SEQ_COLS]
    n = len(S_walk)
    t = np.linspace(0.0, n - 1.0, SEQ_T)
    r = np.stack([np.interp(t, np.arange(n), S_walk[:, j]) for j in range(S_walk.shape[1])], -1)
    return np.r_[r.reshape(-1), 0.1 * (n - 1)].astype(np.float32)


COLS = {"full": list(range(N_BASE + N_WALK)), "cmd": list(range(N_BASE, N_BASE + N_WALK)),
        "post": list(range(N_BASE + N_WALK + N_POST)), "seq": list(range(N_BASE + N_WALK + N_POST + N_SEQ))}  # fmt: skip


def labels(walk_err: np.ndarray) -> np.ndarray:
    e = np.asarray(walk_err, np.float64)
    return np.c_[100 * e[:, :2], np.degrees(e[:, 2])].astype(np.float32)


class WalkEnsemble(nn.Module):
    """5 MLPs on normalized inputs, each predicting the normalized (forward cm, left cm, turn deg). cols picks the input
    columns (all 211, or the walk's 5 for the command-only model)."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, cols: list[int], seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.cols = list(cols)
        self.nx, self.ny = WMM.Norm(X[:, self.cols]), WMM.Norm(Y)
        self.members = nn.ModuleList(WMM.mlp(len(self.cols), 3, width=512, depth=4) for _ in range(ENS))

    def raw(self, x: torch.Tensor) -> torch.Tensor:
        z = self.nx.fwd(x[:, self.cols])
        return torch.stack([m(z) for m in self.members])  # (ENS, B, 3) normalized

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and spread across the members, (B, 3) each, in cm, cm, deg."""
        y = self.ny.inv(self.raw(x))
        return y.mean(0), y.std(0, unbiased=False)

    def predict_np(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """x: (211,) or (N, 211) features() rows."""
        x = np.asarray(x, np.float32)
        with torch.no_grad():
            m, s = self.predict(torch.as_tensor(x.reshape(-1, x.shape[-1])))
        return m.numpy(), s.numpy()


def load() -> dict:
    D = DT.load(SPEC)
    keep = D["fell"] == 0
    so, k0, kr = D["so"], D["k0"].astype(int), D["k_reach"].astype(int)
    S = D["S"]
    post = np.stack([post_feats(S[so[i] + kr[i] - 2 : so[i] + kr[i] + 1]) for i in range(len(so))])
    seq = np.stack([seq_feats(S[so[i] + k0[i] : so[i] + kr[i] + 1]) for i in range(len(so))])
    X = features(D["x0"][:, :N_BASE], D["walk"], post, seq)
    Y = labels(D["walk_err"])
    b = D["bucket"]
    return dict(X=X, Y=Y, walk=D["walk"].astype(np.float32), reset=D["reset"], walked=D["walked"],
                train=np.where(keep & (b < 8))[0], val=np.where(keep & (b == 8))[0], test=np.where(keep & (b == 9))[0])  # fmt: skip


def pos_resid(pred: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.hypot(pred[:, 0] - Y[:, 0], pred[:, 1] - Y[:, 1])


def train_one(S: dict, cols: list[int], name: str, dev: str) -> WalkEnsemble:
    X, Y, tr, va = S["X"], S["Y"], S["train"], S["val"]
    model = WalkEnsemble(X[tr], Y[tr], cols).to(dev)
    g = torch.Generator().manual_seed(0)
    boots = [torch.as_tensor(tr)[torch.randint(len(tr), (len(tr),), generator=g)].to(dev) for _ in range(ENS)]
    XT, YT = torch.as_tensor(X, device=dev), torch.as_tensor(Y, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS)
    vx = XT[torch.as_tensor(va, device=dev)]
    best, best_sd, log, t0 = float("inf"), None, [], time.time()
    for it in range(ITERS):
        loss = 0.0
        for k, m in enumerate(model.members):
            j = boots[k][torch.randint(len(boots[k]), (BATCH,), device=dev)]
            z = m(model.nx.fwd(XT[j][:, model.cols]))
            loss = loss + F.smooth_l1_loss(z, model.ny.fwd(YT[j]), beta=1.0)
        opt.zero_grad()
        loss.backward()
        for m in model.members:
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        if (it + 1) % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                pm, _ = model.predict(vx)
            v = float(np.median(pos_resid(pm.cpu().numpy(), Y[va])))
            model.train()
            log.append(dict(it=it + 1, loss=float(loss.detach()), val_median_cm=v, s=round(time.time() - t0, 1)))
            print(name, json.dumps(log[-1]), flush=True)
            if v < best:
                best, best_sd = v, {k: x.detach().clone() for k, x in model.state_dict().items()}
    model.load_state_dict(best_sd)
    model.log, model.best = log, best
    return model.eval().cpu()


POST_ONLY = os.environ.get("WALK_POST_ONLY", "0") == "1"  # retrain only the post and seq models


def train(dev: str) -> None:
    S = load()
    print(f"walk segments: train {len(S['train'])}, val {len(S['val'])}, test {len(S['test'])}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, path in (("post", WALK_MODEL_POST), ("seq", WALK_MODEL_SEQ)) if POST_ONLY else (
            ("full", WALK_MODEL), ("cmd", WALK_MODEL_CMD), ("post", WALK_MODEL_POST), ("seq", WALK_MODEL_SEQ)):  # fmt: skip
        m = train_one(S, COLS[name], name, dev)
        torch.save(m, path)
        print(f"{name}: best val {m.best:.3f} cm -> {path}", flush=True)


def offline() -> None:
    S = load()
    full = torch.load(WALK_MODEL, weights_only=False).eval()
    cmd = torch.load(WALK_MODEL_CMD, weights_only=False).eval()
    te = S["test"]
    X, Y, w = S["X"][te], S["Y"][te], S["walk"][te]
    if X.shape[1] != max(max(c) for c in COLS.values()) + 1:
        raise SystemExit("feature width changed")
    with torch.no_grad():
        pf, sf = (a.numpy() for a in full.predict(torch.as_tensor(X)))
        pc, _ = (a.numpy() for a in cmd.predict(torch.as_tensor(X)))
        extra = {}
        for nm, path in (("post", WALK_MODEL_POST), ("seq", WALK_MODEL_SEQ)):
            if path.exists():
                mm = torch.load(path, weights_only=False).eval()
                extra[nm] = tuple(a.numpy() for a in mm.predict(torch.as_tensor(X)))
    zero = np.zeros_like(pf)
    sets = {"all test walks": np.ones(len(te), bool), "no turn": w[:, 2] == 0,
            "no turn, nearly straight (|left| < 5 cm)": (w[:, 2] == 0) & (np.abs(w[:, 1]) < 0.05),
            "with a turn": w[:, 2] != 0}  # fmt: skip
    out, L = {}, ["# Walk model, offline (test split of data_approach + data_approach0)", "",
                  "Stop error left after the prediction: |predicted - real| of the pelvis (forward, left), cm; turn in deg.", "",
                  "| walks | n | dead reckoning (predict 0) | command only | pre-walk state + command | + body after the walk | + readings during the walk | turn: dead reckoning / pre-walk / during, deg |",
                  "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for nm, msk in sets.items():
        r0, rc, rf = (pos_resid(p[msk], Y[msk]) for p in (zero, pc, pf))
        t0, tf = np.abs(Y[msk, 2]), np.abs(pf[msk, 2] - Y[msk, 2])
        out[nm] = dict(n=int(msk.sum()), zero=[float(np.median(r0)), float(np.percentile(r0, 90))],
                       cmd=[float(np.median(rc)), float(np.percentile(rc, 90))], full=[float(np.median(rf)), float(np.percentile(rf, 90))],
                       turn=[float(np.median(t0)), float(np.median(tf))])  # fmt: skip
        cells = []
        for k in ("post", "seq"):
            if k in extra:
                rk = pos_resid(extra[k][0][msk], Y[msk])
                out[nm][k] = [float(np.median(rk)), float(np.percentile(rk, 90))]
                cells.append(f"{np.median(rk):.2f} (p90 {np.percentile(rk, 90):.2f})")
            else:
                cells.append("")
        ts = np.median(np.abs(extra["seq"][0][msk, 2] - Y[msk, 2])) if "seq" in extra else float("nan")
        L.append(f"| {nm} | {int(msk.sum())} | {np.median(r0):.2f} (p90 {np.percentile(r0, 90):.2f}) | {np.median(rc):.2f} "
                 f"(p90 {np.percentile(rc, 90):.2f}) | {np.median(rf):.2f} (p90 {np.percentile(rf, 90):.2f}) | {cells[0]} | {cells[1]} | "
                 f"{np.median(t0):.2f} / {np.median(tf):.2f} / {ts:.2f} |")  # fmt: skip
    rf = pos_resid(pf, Y)
    sp = np.hypot(sf[:, 0], sf[:, 1])
    rho = float(np.corrcoef(np.argsort(np.argsort(sp)), np.argsort(np.argsort(rf)))[0, 1])
    out["spread_vs_error_spearman"] = rho
    L += ["", f"Spread of the 5 networks against the error left (Spearman): {rho:.2f}.",
          f"Best val medians: full {full.best:.3f} cm, cmd {cmd.best:.3f} cm."]  # fmt: skip
    json.dump(out, open(OUT / "walk_offline.json", "w"), indent=1)
    (OUT / "walk_offline.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    import walk_model as WK  # the pickled models must name walk_model.WalkEnsemble, not __main__'s

    what = sys.argv[1] if len(sys.argv) > 1 else "t"
    if what == "t":
        WK.train("cuda" if torch.cuda.is_available() else "cpu")
    elif what == "o":
        WK.offline()
    else:
        raise SystemExit("phase: t (train) or o (offline)")
