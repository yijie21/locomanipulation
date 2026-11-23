"""selfmodel_via_20261120, second round: the arm's offset from the reference sweep predicted point by point along the path.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python; V=scripts/experiments/selfmodel_via_20261120
    $PY $V/via_point.py train <model dir> <dataset dir> [<dataset dir> ...] [--dropout 0.1] [--wd 1e-4] [--seed 0]
    $PY $V/via_point.py dev   <model dir>                       # validation paths only: the stop line
    $PY $V/via_point.py eval  <model dir> [<replay dataset dir> ...]   # test sets: only once the model is fixed

The first round's model (via_model.py) took a whole path at once and gained least where it matters: on long paths with many
via points its drop against the reference fell from 0.41 to 0.13. Here one row is ONE POINT of one path -- the Q = 24
moments of the reference plan and the settled end, via_model.py's table -- and one network serves every point, so what it
learns on short paths is what it uses on long ones.

Inputs of a row
  path    the state at the command and the two ticks before, the acting jaw's position and attitude, the goal and its
          attitude, the number of via points, the speed, the pelvis height asked, the hand, the plan's duration
  point   progress along the plan (0..1), seconds since the command and to the plan's end, whether it is the settled end,
          the reference palm's path length so far and still to go, the reference positions of 5 spheres (shoulder, upper
          arm, elbow, wrist, palm) at this moment, their displacements to the three moments before and after, and the
          palm's second sphere and a fingertip relative to the palm (the hand's attitude)
Output: the 24 spheres' offsets at that point (72 numbers), real minus reference, in metres on one common scale, so the
loss is the offsets' squared error. 5 members, each on a bootstrap draw of the training PATHS; a member keeps the state
with the lowest squared offset error on the early-stopping half of the validation paths, checked every 200 iterations.

Everything else is via_model.py's: the swept-region error, the constant predictor, a radius model per predictor trained
on out-of-bag errors. The validation bucket is split in two by reset: one half stops the training, the other calibrates
the radii. Splits, sector hold-out and replay sets as there. Several datasets are joined row-wise.

`train` and `dev` read training and validation paths only. `dev` prints the second round's stop line: validation paths,
standing, the reference palm travelling more than 1 m -- the drop of the arm's median swept-region error against the
reference (docs/handoff/notebook-20261120/selfmodel_via_20261120.md, "第二轮").
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import via_model as VM  # noqa: E402

Q = VM.Q
G_COLS = np.r_[
    0:213, 373:379
]  # via_model.build's x without the via points, their attitudes and mask, and the reference
T_REF, N_KEY, LONG_M = 378, len(VM.KEY_LINKS), 1.0


def load(dirs: list[Path]) -> dict:
    """The datasets' tables joined; src: which of `dirs` a row is from."""
    Ts = []
    for k, D in enumerate(dirs):
        T = VM.load_table(D) if (D / "table.npz").exists() else VM.build(D)
        T["src"] = np.full(len(T["seg"]), k)
        Ts.append(T)
    out = {k: (Ts[0][k] if k == "palm" else np.concatenate([T[k] for T in Ts])) for k in Ts[0]}
    # md5, not data.py's crc32: a crc's lowest bit is a parity of the id's bits whatever the salt, so crc % 2 was the
    # same for every reset of an even bucket and the "half" was all or nothing
    out["cal"] = np.array([int(hashlib.md5(f"cal{int(r)}".encode()).hexdigest(), 16) % 2 for r in out["reset"]])
    z = [np.load(D / "arm.npz") for D in dirs]
    link = [str(x) for x in z[0]["link"]]
    out["key"] = np.array([link.index(x) for x in VM.KEY_LINKS])
    out["hand2"] = np.array([link.index("hand_palm_link") + 1, link.index("hand_index_1_link")])
    palm = out["ref"][:, :, out["key"][-1]]
    out["length"] = np.linalg.norm(np.diff(palm, axis=1), axis=2).sum(1)
    return out


def errors(dirs: list[Path], T: dict, off: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """via_model.errors for the rows `rows` of the joined table (off: their offsets)."""
    e, eh = np.zeros(len(rows)), np.zeros(len(rows))
    for k, D in enumerate(dirs):
        m = T["src"][rows] == k
        if m.any():
            sub = {key: (T[key] if key == "palm" else T[key][rows[m]]) for key in ("seg", "a", "b", "ref", "palm")}
            e[m], eh[m] = VM.errors(D, sub, off[m])
    return e, eh


def features(T: dict, rows: np.ndarray) -> np.ndarray:
    """(len(rows) * (Q + 1), d) inputs, path by path, the points of a path in order."""
    n = len(rows)
    g = T["x"][rows][:, G_COLS]
    ref = T["ref"][rows].astype(np.float32)  # (n, Q, 24, 3)
    dur = T["x"][rows, T_REF]
    key = ref[:, :, T["key"]]  # (n, Q, 5, 3)
    step = np.linalg.norm(np.diff(key[:, :, -1], axis=1), axis=2)
    done = np.concatenate([np.zeros((n, 1), np.float32), np.cumsum(step, 1)], 1)  # (n, Q)
    q = np.r_[np.arange(Q), Q - 1]  # the settled end sits at the plan's last moment
    u = (q / (Q - 1)).astype(np.float32)
    f = [np.broadcast_to(u, (n, Q + 1)), u[None] * dur[:, None], (1 - u[None]) * dur[:, None],
         np.broadcast_to((np.arange(Q + 1) == Q).astype(np.float32), (n, Q + 1)), done[:, q], done[:, -1:] - done[:, q]]  # fmt: skip
    loc = [np.stack(f, 2), key[:, q].reshape(n, Q + 1, -1)]
    for d in (-3, -2, -1, 1, 2, 3):
        loc.append((key[:, np.clip(q + d, 0, Q - 1)] - key[:, q]).reshape(n, Q + 1, -1))
    loc.append((ref[:, q][:, :, T["hand2"]] - key[:, q, -1:]).reshape(n, Q + 1, -1))
    loc = np.concatenate(loc, 2)
    return np.concatenate([np.repeat(g[:, None], Q + 1, 1), loc], 2).reshape(n * (Q + 1), -1).astype(np.float32)


def net(din: int, width: int, depth: int, dropout: float):
    import torch.nn as nn

    layers, d = [], din
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()] + ([nn.Dropout(dropout)] if dropout > 0 else [])
        d = width
    return nn.Sequential(*layers, nn.Linear(d, 72))


def fit(T, tr, va, a: dict, log) -> dict:
    import torch

    dev = a["dev"]
    Xtr, Xva = features(T, tr), features(T, va)
    Ytr = T["off"][tr].reshape(len(tr) * (Q + 1), 72).astype(np.float32)
    Yva = T["off"][va].reshape(len(va) * (Q + 1), 72).astype(np.float32)
    mu_x, sd_x = Xtr.mean(0), Xtr.std(0)
    sd_x = np.where(sd_x < 1e-3, 1.0, sd_x)
    mu_y, sd_y = Ytr.mean(0), float(Ytr.std())  # ONE scale for all 72 outputs
    Xt, Yt = torch.as_tensor((Xtr - mu_x) / sd_x, device=dev), torch.as_tensor((Ytr - mu_y) / sd_y, device=dev)
    Xv, Yv = torch.as_tensor((Xva - mu_x) / sd_x, device=dev), torch.as_tensor(Yva, device=dev)
    mu, base = torch.as_tensor(mu_y, device=dev), float(1e4 * (Yva**2).mean())
    log(f"rows: train {len(Xtr)}, early-stopping validation {len(Xva)}; input {Xtr.shape[1]}; validation squared offset "
        f"{base:.2f} cm2 a coordinate with no model")  # fmt: skip
    members, t0 = [], time.time()
    pts = torch.arange(Q + 1, device=dev)
    for s in range(a["members"]):
        torch.manual_seed(a["seed"] * 100 + s)
        g = torch.Generator().manual_seed(a["seed"] * 100 + 1000 + s)
        bag = torch.randint(len(tr), (len(tr),), generator=g)
        boot = (bag.to(dev)[:, None] * (Q + 1) + pts[None]).reshape(-1)
        f = net(Xtr.shape[1], a["width"], a["depth"], a["dropout"]).to(dev)
        opt = torch.optim.AdamW(f.parameters(), lr=1e-3, weight_decay=a["wd"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a["iters"])
        best, curve = (np.inf, None, 0), []
        for it in range(1, a["iters"] + 1):
            i = boot[torch.randint(len(boot), (4096,), device=dev)]
            loss = ((f(Xt[i]) - Yt[i]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if it % 200 == 0:
                f.eval()
                with torch.no_grad():
                    v = float(1e4 * ((mu + sd_y * f(Xv) - Yv) ** 2).mean())
                f.train()
                curve.append(v)
                if v < best[0]:
                    best = (v, {k: p.detach().cpu().clone() for k, p in f.state_dict().items()}, it)
        log(f"member {s}: validation squared offset {best[0]:.3f} cm2 at iteration {best[2]} (first check {curve[0]:.3f}, "
            f"last {curve[-1]:.3f}; {time.time() - t0:.0f}s)")  # fmt: skip
        members.append(dict(state=best[1], bag=np.unique(tr[bag.numpy()]), val=best[0]))
    return dict(members=members, mu_x=mu_x, sd_x=sd_x, mu_y=mu_y, sd_y=sd_y, din=Xtr.shape[1],
                **{k: a[k] for k in ("width", "depth", "dropout")})  # fmt: skip


def predict(P: dict, T: dict, rows: np.ndarray, oob: bool = False) -> np.ndarray:
    """(len(rows), Q + 1, 24, 3) predicted offsets; oob: each row by the members that did not draw its path (nan if all did)."""
    import torch

    X = torch.as_tensor((features(T, rows) - P["mu_x"]) / P["sd_x"])
    out, cnt = 0.0, np.zeros(len(rows))
    for m in P["members"]:
        f = net(P["din"], P["width"], P["depth"], P["dropout"])
        f.load_state_dict(m["state"])
        f.eval()
        with torch.no_grad():
            y = (P["mu_y"] + P["sd_y"] * f(X).numpy()).reshape(len(rows), Q + 1, 24, 3)
        use = (~np.isin(rows, m["bag"])).astype(float) if oob else np.ones(len(rows))
        out, cnt = out + use[:, None, None, None] * y, cnt + use
    return np.where(cnt[:, None, None, None] > 0, out / np.maximum(cnt, 1)[:, None, None, None], np.nan)


def offsets(M: dict, T: dict, rows: np.ndarray, oob: bool = False) -> dict:
    const = np.where((T["side"][rows] > 0)[:, None, None, None], M["const_right"][None], M["const_left"][None])
    return dict(reference=np.zeros((len(rows), Q + 1, 24, 3)), constant=const.astype(float),
                direction=predict(M["point"], T, rows, oob))  # fmt: skip


def train(out: Path, dirs: list[Path], a: dict) -> None:
    import torch

    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    T = load(dirs)
    free = ~T["sector"]
    tr = np.where(free & (T["bucket"] < 8))[0]
    va, ca = (
        np.where(free & (T["bucket"] == 8) & (T["cal"] == 0))[0],
        np.where(free & (T["bucket"] == 8) & (T["cal"] == 1))[0],
    )
    log(f"datasets {[str(d) for d in dirs]}; settings {a}")
    log(f"paths {len(T['seg'])}: train {len(tr)}, validation for early stopping {len(va)}, for calibration {len(ca)}, "
        f"test {int((free & (T['bucket'] == 9)).sum())}, sector hold-out {int(T['sector'].sum())}")  # fmt: skip
    M = dict(dirs=[str(d) for d in dirs], settings=a)
    for name, m in (("const_right", T["side"][tr] > 0), ("const_left", T["side"][tr] < 0)):
        M[name] = T["off"][tr][m].mean(0) if m.any() else np.zeros(T["off"].shape[1:])
    M["point"] = fit(T, tr, va, a, log)
    log(
        f"ensemble's members: mean validation squared offset {np.mean([m['val'] for m in M['point']['members']]):.3f} cm2"
    )
    off = offsets(M, T, tr, oob=True)
    has = ~np.isnan(off["direction"]).any((1, 2, 3))
    log(f"direction: {int(has.sum())} of {len(tr)} training paths have an out-of-bag prediction")
    M["radius"], M["scale"], M["fixed"] = {}, {}, {}
    both = np.r_[tr[has], va, ca]
    for name in ("reference", "constant", "direction"):
        o = np.concatenate([off[name][has], offsets(M, T, np.r_[va, ca])[name]])
        e, _ = errors(dirs, T, o, both)
        E = np.full(len(T["seg"]), np.nan)
        E[both] = e
        M["radius"][name] = VM.fit(T["x"], np.log(np.maximum(E, 1e-4))[:, None], tr[has], va, 3, 256, a["iters"] // 2,
                                   a["dev"], f"radius[{name}]", log)  # fmt: skip
        r = np.exp(VM.predict(M["radius"][name], T["x"][ca])[:, 0])
        M["scale"][name], M["fixed"][name] = (
            float(np.quantile(E[ca] / r, VM.LEVEL)),
            float(np.quantile(E[ca], VM.LEVEL)),
        )
        log(f"radius[{name}]: scale {M['scale'][name]:.3f}; calibration paths' error median {100 * np.median(E[ca]):.2f} cm, "
            f"{VM.LEVEL:.0%} level {100 * M['fixed'][name]:.2f} cm")  # fmt: skip
    torch.save(M, out / "via_point.pt")
    log(f"saved {out / 'via_point.pt'}")


def report(name: str, dirs: list[Path], T: dict, M: dict, rows: np.ndarray, log) -> dict | None:
    if len(rows) < 5:
        log(f"{name}: {len(rows)} paths, too few")
        return None
    off = offsets(M, T, rows)
    log(f"{name}: {len(rows)} paths")
    res = {}
    for p in ("reference", "constant", "direction"):
        e, eh = errors(dirs, T, off[p], rows)
        r = M["scale"][p] * np.exp(VM.predict(M["radius"][p], T["x"][rows])[:, 0])
        res[p] = dict(e=e, eh=eh)
        log(f"  {p:10s} swept-region error: arm median {100 * np.median(e):5.2f} p90 {100 * np.percentile(e, 90):5.2f} cm | hand "
            f"median {100 * np.median(eh):5.2f} cm | margin (median radius) {100 * np.median(r):5.2f} cm, inside {np.mean(e <= r):.2f}"
            f" | fixed radius {100 * M['fixed'][p]:5.2f} cm, inside {np.mean(e <= M['fixed'][p]):.2f}")  # fmt: skip
    b, c, d = (res[p] for p in ("reference", "constant", "direction"))
    rng, ids = np.random.default_rng(0), T["route"][rows]
    groups = (
        [np.where(ids == g)[0] for g in np.unique(ids)]
        if (ids >= 0).all()
        else [np.array([j]) for j in range(len(rows))]
    )
    boots = []
    for _ in range(2000):
        j = np.concatenate([groups[k] for k in rng.integers(len(groups), size=len(groups))])
        boots.append(1 - np.median(d["e"][j]) / np.median(b["e"][j]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    log(f"  drop of the arm's median against the reference: {1 - np.median(d['e']) / np.median(b['e']):.2f} (95 % interval "
        f"{lo:.2f} to {hi:.2f}, resampling {'routes' if (ids >= 0).all() else 'paths'}); hand "
        f"{1 - np.median(d['eh']) / np.median(b['eh']):.2f}; constant: arm {1 - np.median(c['e']) / np.median(b['e']):.2f}")  # fmt: skip
    log(f"  fixed radius: direction / reference {M['fixed']['direction'] / M['fixed']['reference']:.2f} (set on the calibration "
        f"paths); direction wins on {np.mean(d['e'] < b['e']):.2f} of the paths")  # fmt: skip
    return res


def model_dirs(M: dict, model_dir: Path) -> list[Path]:
    """The model's training datasets. The file stores the paths they had on the machine that trained it; one that is not
    there is looked for by its name beside the model's own directory (a copy of the outputs on another machine)."""
    return [d if d.exists() else Path(model_dir).resolve().parent / d.name for d in (Path(x) for x in M["dirs"])]


def opened(out: Path, name: str):
    import torch

    M = torch.load(out / "via_point.pt", map_location="cpu", weights_only=False)
    dirs = model_dirs(M, out)
    logf = open(out / name, "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")

    return M, dirs, load(dirs), log


def dev(out: Path) -> None:
    M, dirs, T, log = opened(out, "dev.log")
    v = (~T["sector"]) & (T["bucket"] == 8)
    stand = T["pelvis_z"] < 0
    report("validation, all", dirs, T, M, np.where(v)[0], log)
    report("validation, standing", dirs, T, M, np.where(v & stand)[0], log)
    report(f"validation, standing, palm path over {LONG_M:.1f} m (THE STOP LINE: drop of the arm's median >= 0.25)", dirs, T, M,
           np.where(v & stand & (T["length"] > LONG_M))[0], log)  # fmt: skip
    report("validation, standing, palm path under 0.6 m", dirs, T, M, np.where(v & stand & (T["length"] < 0.6))[0], log)
    report(
        "validation, standing, no walk before the reach", dirs, T, M, np.where(v & stand & (T["walked"] == 0))[0], log
    )
    report("validation, crouching", dirs, T, M, np.where(v & ~stand)[0], log)


def evaluate(out: Path, replays: list[Path]) -> None:
    M, dirs, T, log = opened(out, "eval.log")
    t = (~T["sector"]) & (T["bucket"] == 9)
    stand = T["pelvis_z"] < 0
    report("test, ordinary hold-out", dirs, T, M, np.where(t)[0], log)
    report("test, sector hold-out", dirs, T, M, np.where(T["sector"])[0], log)
    report("test, ordinary hold-out, standing", dirs, T, M, np.where(t & stand)[0], log)
    report(f"test, ordinary hold-out, standing, palm path over {LONG_M:.1f} m", dirs, T, M,
           np.where(t & stand & (T["length"] > LONG_M))[0], log)  # fmt: skip
    for R in replays:
        TR = load([R])
        report(f"real paths replayed, {R.name}", [R], TR, M, np.arange(len(TR["seg"])), log)
        report(f"real paths replayed, {R.name}, walked first", [R], TR, M, np.where(TR["walked"] > 0)[0], log)


def main(argv=None) -> int:
    a = sys.argv[1:] if argv is None else argv
    opt = dict(dropout=0.1, wd=1e-4, seed=0, width=512, depth=4, members=5, iters=30000, dev="cuda")
    pos = []
    while a:
        x = a.pop(0)
        if x.startswith("--"):
            opt[x[2:]] = type(opt[x[2:]])(a.pop(0))
        else:
            pos.append(x)
    if pos[0] == "train":
        train(Path(pos[1]), [Path(p) for p in pos[2:]], opt)
    elif pos[0] == "dev":
        dev(Path(pos[1]))
    elif pos[0] == "eval":
        evaluate(Path(pos[1]), [Path(p) for p in pos[2:]])
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
