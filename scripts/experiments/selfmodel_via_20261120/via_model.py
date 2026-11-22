"""selfmodel_via_20261120: where will the whole arm really sweep when the frozen controller is sent a path of via points?
The path-aware model ("direction"), and the two opponents it has to beat.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python; V=scripts/experiments/selfmodel_via_20261120
    $PY $V/via_model.py build <dataset dir>                      # all.npz + arm.npz -> table.npz
    $PY $V/via_model.py train <dataset dir> <model dir>          # -> <model dir>/via_model.pt, train.log
    $PY $V/via_model.py eval  <model dir> <dataset dir> [<replay dataset dir>]

One row per path (chain 0 of a REACH_VIA / REACH_VIA_REPLAY dataset, no fall). Everything is in the heading frame of the
command (x ahead, y left, z up), not mirrored; the hand is an input.

Sweeps. The reference plan's sweep is each of the acting arm's 24 collision spheres at Q = 24 equal steps of the plan's
duration (arm.npz XR between the reference poses' times RT). The real sweep at the same moments after the command is the
recorded one (arm.npz X at 10 Hz), plus one last point: the mean of the recording's last 0.5 s, where the arm has settled.
offset = real - reference at these Q + 1 points (the last against the plan's end). A predicted sweep is the reference plus
predicted offsets, a polyline of Q + 1 points per sphere.

Error of a predicted sweep for one path ("swept-region error", the plan entry's main measure): the largest distance of any
recorded tick of any sphere, from the command to the end of the recording, to that sphere's predicted polyline. Lag along
the path costs nothing; being beside the path does. "hand" is the same over the palm's 3 spheres only.

Predictors
  reference   offsets 0: the arm goes where the plan says (the baseline)
  constant    one mean offset profile per hand, from the training paths: a correction that does not look at the path
  direction   an ensemble of 5 MLPs (bootstrap) from the start state and the command to the offsets, as coefficients of
              the training offsets' principal components (99.5 % of their variance)
  inputs      the state at the command and the two ticks before (S, 3 x 65), the acting jaw's position and attitude, the
              goal and its attitude, the via points and attitudes (16 padded, with their count), the speed, the pelvis
              height asked, the hand, the plan's duration and the reference sweep of 5 spheres (shoulder, upper arm,
              elbow, wrist, palm)
Radius. For each predictor a second small ensemble predicts the log of its own error from the same inputs: a radius to keep
clear around the predicted sweep. It is trained on errors the predictor has not fitted: a training path's direction
prediction comes from the members whose bootstrap draw left it out. "scalar radius" is that model on the reference: the learned tube, which cannot move the
sweep, only widen it. Every radius model is scaled by one factor so that 95 % of the calibration paths' errors fall
inside, and the margin reported is the median scaled radius over the test paths (with the share of test errors inside).

Splits by reset (data.py's hash): buckets 0-7 train, 8 validation and calibration, 9 test. Sector hold-out: paths whose
largest bend points inwards-down (phi 202.5-247.5 deg, via_gen's convention) are in no training or calibration set and are
a test set of their own. A replay dataset is a test set too: nothing of it is trained or calibrated on.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reachwm_20261030"))
Q, TAIL, VIA_MAX = 24, 5, 16
KEY_LINKS = ("shoulder_pitch_link", "shoulder_yaw_link", "elbow_link", "wrist_pitch_link", "hand_palm_link")
SECTOR = (202.5, 247.5)
LEVEL = 0.95


def hash_bucket(reset: np.ndarray, salt: str, mod: int) -> np.ndarray:
    import data as DT

    return DT.hash_bucket(reset, salt, mod)


def r6m(r) -> np.ndarray:
    a, b = np.asarray(r[:3], float), np.asarray(r[3:6], float)
    return np.c_[a, b, np.cross(a, b)]


def interp(T, X, t) -> np.ndarray:
    """X (n, ...) given at increasing times T (n,), linearly at times t (m,) -> (m, ...); held beyond the ends."""
    t = np.clip(t, T[0], T[-1])
    j = np.clip(np.searchsorted(T, t, side="right") - 1, 0, len(T) - 2)
    w = ((t - T[j]) / np.maximum(T[j + 1] - T[j], 1e-9)).reshape(-1, *([1] * (X.ndim - 1)))
    return X[j] + w * (X[j + 1] - X[j])


def sweep_err(Xa, P) -> np.ndarray:
    """Recorded ticks Xa (T, S, 3) against the polylines P (M, S, 3), one per sphere -> (S,) each sphere's largest
    distance from its polyline over the ticks."""
    A_, AB = P[:-1], P[1:] - P[:-1]
    d = Xa[:, None] - A_[None]
    u = np.clip((d * AB[None]).sum(-1) / np.maximum((AB * AB).sum(-1), 1e-12)[None], 0, 1)
    return np.linalg.norm(d - u[..., None] * AB[None], axis=-1).min(1).max(0)


def build(D: Path) -> dict:
    """<D>/all.npz + arm.npz -> <D>/table.npz, one row per path."""
    z, A = (
        dict(np.load(D / "all.npz")),
        dict(np.load(D / "arm.npz")),
    )  # every array read once: an NpzFile decompresses the whole array again at each z[...]
    S, W, J = z["S"], z["W"], z["J"]
    n, k0s = z["n_steps"].astype(int), z["k0"].astype(int)
    so, ro = np.r_[0, np.cumsum(n)[:-1]], np.r_[0, np.cumsum(z["n_ref"])[:-1]].astype(int)
    link = [str(x) for x in A["link"]]
    key = [link.index(x) for x in KEY_LINKS]
    u = np.linspace(0.0, 1.0, Q)
    rows = dict(x=[], ref=[], off=[], seg=[], a=[], b=[])
    for i in np.where((z["chain"] == 0) & (z["fell"] == 0))[0]:
        a, b = so[i] + k0s[i], so[i] + n[i]
        RT = z["RT"][ro[i] : ro[i] + int(z["n_ref"][i])].astype(float)
        XR = A["XR"][ro[i] : ro[i] + int(z["n_ref"][i])].astype(float)
        if b - a < TAIL + 2 or len(RT) < 2 or RT[-1] <= 0:
            continue
        ref = interp(RT, XR, u * RT[-1])
        Xa = A["X"][a:b].astype(float)
        act = interp(0.1 * np.arange(b - a), Xa, u * RT[-1])
        off = np.concatenate([act - ref, (Xa[-TAIL:].mean(0) - ref[-1])[None]])
        sd = 1.0 if z["side"][i] > 0 else -1.0
        p0, yaw = z["start"][i][:3].astype(float), float(z["start"][i][3])
        Rh = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
        jw = (W[a, 4:7] if sd > 0 else W[a, 7:10]).astype(float)
        jaw = np.r_[(jw[:2] - p0[:2]) @ Rh[:2, :2], jw[2]]
        R0 = Rh.T @ r6m(J[a])
        nv, pz = int(z["n_via"][i]), float(z["pelvis_z"][i])
        mask = (np.arange(VIA_MAX) < nv).astype(float)
        rows["x"].append(np.r_[
            S[a], S[a - 1], S[a - 2], jaw, R0[:, 0], R0[:, 1], z["target"][i], z["rot6"][i], z["via_p"][i].ravel(),
            z["via_r6"][i].ravel(), mask, nv / VIA_MAX, z["via_speed"][i], float(pz > 0), max(pz, 0.0), sd, RT[-1],
            ref[::3, key].ravel(),
        ].astype(np.float32))  # fmt: skip
        rows["ref"].append(ref.astype(np.float32))
        rows["off"].append(off.astype(np.float32))
        rows["seg"].append(i)
        rows["a"].append(a)
        rows["b"].append(b)
    seg = np.array(rows["seg"])
    g = z["via_gen"][seg]
    phi = np.full(len(seg), -1.0)
    for k in range(
        3
    ):  # the bend furthest off the line: columns 6, 10, 14 are the bends' offsets, 5, 9, 13 their angles
        m = (g[:, 0] > k) & (g[:, 6 + 4 * k] >= g[:, 6:15:4].max(1) - 1e-9)
        phi[m] = g[m, 5 + 4 * k]
    T = dict(
        x=np.array(rows["x"]), ref=np.array(rows["ref"]), off=np.array(rows["off"]), seg=seg, a=np.array(rows["a"]),
        b=np.array(rows["b"]), reset=z["reset"][seg], side=z["side"][seg], pelvis_z=z["pelvis_z"][seg], bends=g[:, 0],
        route=z["via_route"][seg], walked=z["walked"][seg], phi=phi, sector=(phi >= SECTOR[0]) & (phi < SECTOR[1]),
        bucket=hash_bucket(z["reset"][seg], "reach", 10), palm=np.array([x == "hand_palm_link" for x in link]),
    )  # fmt: skip
    np.savez_compressed(D / "table.npz", **T)
    print(f"{D}: {len(seg)} paths of {int((z['chain'] == 0).sum())} (falls and too-short ones left out), input {T['x'].shape[1]}, "
          f"sector hold-out {int(T['sector'].sum())}, buckets train/val/test "
          f"{int((T['bucket'] < 8).sum())}/{int((T['bucket'] == 8).sum())}/{int((T['bucket'] == 9).sum())}")  # fmt: skip
    return T


_ARM: dict = {}


def errors(D: Path, T: dict, off: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The swept-region error of the sweeps `reference + off` for every row of T: (all spheres, palm) in m."""
    if D not in _ARM:
        _ARM[D] = np.load(D / "arm.npz")["X"]
    X = _ARM[D]
    e, eh = np.zeros(len(T["seg"])), np.zeros(len(T["seg"]))
    for j in range(len(T["seg"])):
        P = np.concatenate([T["ref"][j], T["ref"][j][-1:]]).astype(float) + off[j]
        d = sweep_err(X[T["a"][j] : T["b"][j]].astype(float), P)
        e[j], eh[j] = d.max(), d[T["palm"]].max()
    return e, eh


def fit(X, Y, tr, va, members: int, width: int, iters: int, dev: str, tag: str, log) -> list:
    """An ensemble of MLPs X -> Y on bootstrap draws of the rows tr; each member keeps its best state on the rows va."""
    import torch
    import torch.nn as nn

    mu_x, sd_x = X[tr].mean(0), X[tr].std(0)
    sd_x = np.where(sd_x < 1e-3, 1.0, sd_x)  # a constant input is not scaled (inverse_model.py's -311 m)
    mu_y, sd_y = Y[tr].mean(0), Y[tr].std(0) + 1e-6
    Xt = torch.as_tensor((X - mu_x) / sd_x, dtype=torch.float32, device=dev)
    Yt = torch.as_tensor((Y - mu_y) / sd_y, dtype=torch.float32, device=dev)
    vai = torch.as_tensor(va, device=dev)
    nets, t0 = [], time.time()
    for s in range(members):
        torch.manual_seed(s)
        g = torch.Generator(device="cpu").manual_seed(1000 + s)
        boot = torch.as_tensor(tr, device=dev)[torch.randint(len(tr), (len(tr),), generator=g).to(dev)]
        net = nn.Sequential(nn.Linear(X.shape[1], width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, Y.shape[1])).to(dev)  # fmt: skip
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        best = (np.inf, None)
        for it in range(1, iters + 1):
            i = boot[torch.randint(len(boot), (512,), device=dev)]
            loss = ((net(Xt[i]) - Yt[i]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if it % 500 == 0:
                with torch.no_grad():
                    v = float(((net(Xt[vai]) - Yt[vai]) ** 2).mean())
                if v < best[0]:
                    best = (v, {k: p.detach().cpu().clone() for k, p in net.state_dict().items()}, it)
        log(f"{tag} member {s}: best validation loss {best[0]:.4f} at iteration {best[2]} ({time.time() - t0:.0f}s)")
        nets.append((best[1], np.unique(boot.cpu().numpy())))
    return [dict(state=n, bag=bag, mu_x=mu_x, sd_x=sd_x, mu_y=mu_y, sd_y=sd_y, width=width) for n, bag in nets]


def predict(nets: list, X: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
    """The ensemble's mean prediction for the rows X. With `rows` (the rows' indices in the table the members were trained
    on): each row is predicted only by the members whose bootstrap draw left it out (out of bag); a row every member drew
    comes back as nan."""
    import torch
    import torch.nn as nn

    out, cnt = 0.0, np.zeros(len(X))
    for m in nets:
        w, dout = m["width"], len(m["mu_y"])
        net = nn.Sequential(nn.Linear(X.shape[1], w), nn.SiLU(), nn.Linear(w, w), nn.SiLU(), nn.Linear(w, w), nn.SiLU(),
                            nn.Linear(w, dout))  # fmt: skip
        net.load_state_dict(m["state"])
        with torch.no_grad():
            y = net(torch.as_tensor((X - m["mu_x"]) / m["sd_x"], dtype=torch.float32)).numpy()
        use = np.ones(len(X)) if rows is None else (~np.isin(rows, m["bag"])).astype(float)
        out, cnt = out + use[:, None] * (m["mu_y"] + m["sd_y"] * y), cnt + use
    return np.where(cnt[:, None] > 0, out / np.maximum(cnt, 1)[:, None], np.nan)


def load_table(D: Path) -> dict:
    return dict(np.load(D / "table.npz"))


def offsets(M: dict, T: dict, oob: np.ndarray | None = None) -> dict:
    """The three predictors' offsets for the rows of T. oob: these rows of T (table indices the model was trained on) get the
    direction model's out-of-bag prediction, nan where there is none."""
    n = len(T["seg"])
    const = np.where((T["side"] > 0)[:, None, None, None], M["const_right"][None], M["const_left"][None])
    coef = predict(M["direction"], T["x"])
    if oob is not None:
        coef[oob] = predict(M["direction"], T["x"][oob], oob)
    return dict(reference=np.zeros_like(T["off"], dtype=float), constant=const.astype(float),
                direction=(M["pca_mean"] + coef @ M["pca_basis"]).reshape(n, Q + 1, -1, 3))  # fmt: skip


def train(D: Path, out: Path, iters: int = 20000, dev: str = "cuda") -> None:
    import torch

    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    T = load_table(D) if (D / "table.npz").exists() else build(D)
    free = ~T["sector"]
    tr, va = np.where(free & (T["bucket"] < 8))[0], np.where(free & (T["bucket"] == 8))[0]
    log(f"paths {len(T['seg'])}: train {len(tr)}, validation {len(va)}, test {int((free & (T['bucket'] == 9)).sum())}, "
        f"sector hold-out {int(T['sector'].sum())}; input {T['x'].shape[1]}")  # fmt: skip
    Y = T["off"].reshape(len(T["seg"]), -1).astype(float)
    mean = Y[tr].mean(0)
    _, sv, Vt = np.linalg.svd(Y[tr] - mean, full_matrices=False)
    k = int(np.searchsorted(np.cumsum(sv**2) / (sv**2).sum(), 0.995) + 1)
    basis = Vt[:k]
    log(f"offsets: {Y.shape[1]} numbers a path, {k} principal components keep 99.5 % of the training variance")
    M = dict(pca_mean=mean, pca_basis=basis, q=Q, data=str(D))
    for name, m in (("const_right", T["side"][tr] > 0), ("const_left", T["side"][tr] < 0)):
        M[name] = T["off"][tr][m].mean(0) if m.any() else np.zeros(T["off"].shape[1:])
    M["direction"] = fit(T["x"], (Y - mean) @ basis.T, tr, va, 5, 512, iters, dev, "direction", log)
    off = offsets(M, T, tr)  # the training rows out of bag: the radius is not learned from errors the model has fitted
    has = ~np.isnan(off["direction"][tr]).any((1, 2, 3))
    log(f"direction: {int(has.sum())} of {len(tr)} training paths have an out-of-bag prediction")
    off["direction"] = np.nan_to_num(off["direction"])
    M["radius"], M["scale"] = {}, {}
    for name in ("reference", "constant", "direction"):
        e, _ = errors(D, T, off[name])
        M["radius"][name] = fit(
            T["x"], np.log(np.maximum(e, 1e-4))[:, None], tr[has], va, 3, 256, iters // 2, dev, f"radius[{name}]", log
        )
        r = np.exp(predict(M["radius"][name], T["x"][va])[:, 0])
        M["scale"][name] = float(np.quantile(e[va] / r, LEVEL))
        M[f"fixed_{name}"] = float(np.quantile(e[va], LEVEL))
        log(f"radius[{name}]: scale {M['scale'][name]:.3f} puts {LEVEL:.0%} of the validation errors inside; "
            f"validation error median {100 * np.median(e[va]):.2f} cm, its {LEVEL:.0%} level {100 * M[f'fixed_{name}']:.2f} cm")  # fmt: skip
    torch.save(M, out / "via_model.pt")
    log(f"saved {out / 'via_model.pt'}")


def report(name: str, D: Path, T: dict, M: dict, rows: np.ndarray, log) -> None:
    if not len(rows):
        log(f"{name}: no paths")
        return
    Tr = {k: (v[rows] if k != "palm" else v) for k, v in T.items()}
    off = offsets(M, Tr)
    log(f"{name}: {len(rows)} paths")
    res = {}
    for p in ("reference", "constant", "direction"):
        e, eh = errors(D, Tr, off[p])
        r = M["scale"][p] * np.exp(predict(M["radius"][p], Tr["x"])[:, 0])
        res[p] = dict(e=e, eh=eh, r=r)
        log(f"  {p:10s} swept-region error: arm median {100 * np.median(e):5.2f} p90 {100 * np.percentile(e, 90):5.2f} cm | "
            f"hand median {100 * np.median(eh):5.2f} p90 {100 * np.percentile(eh, 90):5.2f} cm | margin (median radius) "
            f"{100 * np.median(r):5.2f} cm, errors inside {np.mean(e <= r):.2f} | fixed radius {100 * M[f'fixed_{p}']:5.2f} cm, "
            f"inside {np.mean(e <= M[f'fixed_{p}']):.2f}")  # fmt: skip
    b, c, d = (res[p] for p in ("reference", "constant", "direction"))
    log(f"  drop of the median against the reference: arm {1 - np.median(d['e']) / np.median(b['e']):.2f}, hand "
        f"{1 - np.median(d['eh']) / np.median(b['eh']):.2f} (constant: arm {1 - np.median(c['e']) / np.median(b['e']):.2f}, "
        f"hand {1 - np.median(c['eh']) / np.median(b['eh']):.2f})")  # fmt: skip
    log(f"  margins: direction / scalar radius {np.median(d['r']) / np.median(b['r']):.2f}, direction / constant "
        f"{np.median(d['r']) / np.median(c['r']):.2f}")  # fmt: skip


def evaluate(out: Path, D: Path, replay: Path | None) -> None:
    import torch

    M = torch.load(out / "via_model.pt", map_location="cpu", weights_only=False)
    logf = open(out / "eval.log", "w")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")

    T = load_table(D)
    free = ~T["sector"]
    report("validation (calibrated on these)", D, T, M, np.where(free & (T["bucket"] == 8))[0], log)
    report("test, ordinary hold-out", D, T, M, np.where(free & (T["bucket"] == 9))[0], log)
    report("test, sector hold-out (inwards-down bends, never trained on)", D, T, M, np.where(T["sector"])[0], log)
    report(
        "test, ordinary hold-out, standing", D, T, M, np.where(free & (T["bucket"] == 9) & (T["pelvis_z"] < 0))[0], log
    )
    report(
        "test, ordinary hold-out, crouching", D, T, M, np.where(free & (T["bucket"] == 9) & (T["pelvis_z"] > 0))[0], log
    )
    if replay is not None:
        R = load_table(replay) if (replay / "table.npz").exists() else build(replay)
        report("real paths replayed (never trained or calibrated on)", replay, R, M, np.arange(len(R["seg"])), log)


def main(argv=None) -> int:
    a = sys.argv[1:] if argv is None else argv
    if a[0] == "build":
        build(Path(a[1]))
    elif a[0] == "train":
        train(Path(a[1]), Path(a[2]), *(int(x) for x in a[3:4]))
    elif a[0] == "eval":
        evaluate(Path(a[1]), Path(a[2]), Path(a[3]) if len(a) > 3 else None)
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
