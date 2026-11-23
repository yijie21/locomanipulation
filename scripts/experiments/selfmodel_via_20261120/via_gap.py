"""selfmodel_via_20261120, fourth step, part one: from open-floor replays, how narrow a gap can each way of acting pass
safely?

    $PY via_gap.py <out dir> <model dir> <direct replay dir> <const replay dir> <comp replay dir> \
        [--base <baseline model dir>] [--cal <compensated replay dir on other routes>] [--seen <route indices, a,b,...>]

A gap of half-width c: obstacles stand c away, on every side, from the path the planner wants each sphere to take (the
original command's reference sweep). A reach TOUCHES when any sphere of the part that has to pass leaves its wanted path by
more than c (via_model.py's swept-region error over those spheres).
  through   wrist, palm and fingers, 17 spheres -- the main set: the forearm's end is in the gap with the hand
  hand      palm and fingers, 15 spheres
  arm       all 24
Five ways of acting. Each decides before the command whether to go, and what to send:
  small     go when c >= 3 cm; the original command
  fixed     go when c >= one margin for all reaches, the 95 % level of the direct command's error on the calibration paths
  radius    go when c >= this reach's predicted radius (a model from the same inputs, scaled to hold 95 % of the
            calibration errors): the learned tube
  const     go when c >= its own fixed margin (95 % level of what is left after the mean offset on the calibration paths);
            every via point moved by the training paths' mean offset (REACH_VIA_COMP_MODE=const)
  ours      go when the sweep predicted for the command to be sent stays within c minus this reach's residual radius of the
            wanted one; the via points moved against the point model's prediction (REACH_VIA_COMP)
Margins, radius models and their scales come from the model's own training paths only (train: fit; one half of the
validation bucket: early stopping; the other half: the 95 % levels). No replayed route that is scored sets any of them.
--base: the margins and radius models of small / fixed / radius / const come from this model's training paths instead (a
model trained on random paths alone), so those four ways are scored as before when <model dir> has also seen replays.
--cal: the scale of ours' residual radius is set on this run instead -- a compensated replay, by <model dir>, of routes that
are neither trained on nor scored (the development routes): the 95 % level of what its reaches left of the sweep predicted
for the command sent, over the radius model's prediction. A model that trained on replays has other passes of its training
routes in its own calibration half, which would understate the residual on new routes.
--seen: routes of the scored set that were looked at during development; the detour table is printed once more without them.

Scored on the replays (the same routes from the same states in all three runs): a reach is a success when it goes and does
not touch, a collision when it goes and touches, an abstention when it does not go. The headline is the narrowest gap passed
safely: the smallest c with success >= 90 % and collisions <= 5 %. Primary set: the detour routes (every kind but
FreshUnder). Intervals: 95 %, resampling routes.
-> <out dir>/gap.log (what is printed), gap.json (the numbers), radius.pt (the radius models).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import via_model as VM  # noqa: E402
import via_point as VP  # noqa: E402

OUT, MD, DD, DK, DC = (Path(x) for x in sys.argv[1:6])
OUT.mkdir(parents=True, exist_ok=True)
Q, LEVEL, SMALL = VM.Q, VM.LEVEL, 0.03
GRID = np.arange(0.02, 0.1501, 0.0025)
WAYS = ("small", "fixed", "radius", "const", "ours")
logf = open(OUT / "gap.log", "w")


def log(s=""):
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


def per_sphere(D: Path, T: dict, rows, want, off) -> np.ndarray:
    """(len(rows), 24): each sphere's largest distance, over the recorded ticks of the path, from the polyline want + off
    (want: (n, Q, 24, 3) sweeps; off: (n, Q + 1, 24, 3) offsets or None)."""
    if D not in VM._ARM:
        VM._ARM[D] = np.load(D / "arm.npz")["X"]
    X, out = VM._ARM[D], np.zeros((len(rows), 24))
    for n, j in enumerate(rows):
        P = np.concatenate([want[n], want[n][-1:]]).astype(float)
        if off is not None:
            P = P + off[n]
        out[n] = VM.sweep_err(X[T["a"][j] : T["b"][j]].astype(float), P)
    return out


ARGS = sys.argv[6:]
BASE = Path(ARGS[ARGS.index("--base") + 1]) if "--base" in ARGS else None
CALRUN = Path(ARGS[ARGS.index("--cal") + 1]) if "--cal" in ARGS else None
SEEN = [int(x) for x in ARGS[ARGS.index("--seen") + 1].split(",")] if "--seen" in ARGS else []
M = torch.load(MD / "via_point.pt", map_location="cpu", weights_only=False)
link = [str(x) for x in np.load(VP.model_dirs(M, MD)[0] / "arm.npz")["link"]]
SETS = dict(
    through=np.array([x == "wrist_pitch_link" or x.startswith("hand_") for x in link]),
    hand=np.array([x.startswith("hand_") for x in link]),
    arm=np.ones(len(link), bool),
)
log("spheres: " + ", ".join(f"{k} {int(v.sum())}" for k, v in SETS.items()))


def calibrate(Mx: dict, which: tuple, tag: str, where: Path) -> dict:
    """set -> the fixed margins, radius models and scales of the offsets `which` (reference, constant, direction), from
    the training paths of the model Mx."""
    dirs = VP.model_dirs(Mx, where)
    T = VP.load(dirs)
    free = ~T["sector"]
    tr = np.where(free & (T["bucket"] < 8))[0]
    va, ca = (
        np.where(free & (T["bucket"] == 8) & (T["cal"] == 0))[0],
        np.where(free & (T["bucket"] == 8) & (T["cal"] == 1))[0],
    )
    off_tr = VP.offsets(Mx, T, tr, oob=True)
    has = ~np.isnan(off_tr["direction"]).any((1, 2, 3))
    rows = np.r_[tr[has], va, ca]
    off_all = {p: np.concatenate([off_tr[p][has], VP.offsets(Mx, T, np.r_[va, ca])[p]]) for p in which}
    E = {}
    for p in which:
        e = np.full((len(T["seg"]), 24), np.nan)
        for k, D in enumerate(dirs):
            m = T["src"][rows] == k
            e[rows[m]] = per_sphere(D, T, rows[m], T["ref"][rows[m]], off_all[p][m])
        E[p] = e
    log(
        f"{tag}: {len(dirs)} datasets, fit on {int(has.sum())} paths, early stopping on {len(va)}, 95 % levels on {len(ca)}"
    )
    out = {}
    for name, mask in SETS.items():
        c = dict()
        for p in which:
            e = E[p][:, mask].max(1)
            c[f"fixed_{p}"] = float(np.quantile(e[ca], LEVEL))
            if p != "constant":
                nets = VM.fit(T["x"], np.log(np.maximum(e, 1e-4))[:, None], tr[has], va, 3, 256, 10000, "cuda",
                              f"radius[{name}, {p}]", log)  # fmt: skip
                c[f"nets_{p}"] = nets
                c[f"scale_{p}"] = float(np.quantile(e[ca] / np.exp(VM.predict(nets, T["x"][ca])[:, 0]), LEVEL))
        out[name] = c
    return out


# ---- margins and radius models, from training paths only
if BASE is None:
    CAL = calibrate(M, ("reference", "constant", "direction"), "margins", MD)
else:
    Mb = torch.load(BASE / "via_point.pt", map_location="cpu", weights_only=False)
    CAL = calibrate(Mb, ("reference", "constant"), f"margins of small / fixed / radius / const ({BASE.name})", BASE)
    for name, c in calibrate(M, ("direction",), f"ours' residual radius ({MD.name})", MD).items():
        CAL[name].update(c)
if CALRUN is not None:  # ours' scale from a compensated replay of routes neither trained on nor scored
    Tx, zx = VP.load([CALRUN]), dict(np.load(CALRUN / "all.npz"))
    rx = np.arange(len(Tx["seg"]))
    res_x = per_sphere(CALRUN, Tx, rx, zx["comp_ref"][Tx["seg"]].astype(float), zx["comp_off"][Tx["seg"]].astype(float))
    for name, mask in SETS.items():
        c = CAL[name]
        r = np.exp(VM.predict(c["nets_direction"], Tx["x"])[:, 0])
        new = float(np.quantile(res_x[:, mask].max(1) / r, LEVEL))
        log(f"{name}: ours' residual scale {c['scale_direction']:.2f} on the training side -> {new:.2f} on {len(rx)} reaches of "
            f"{CALRUN.name} (their residual: median {100 * np.median(res_x[:, mask].max(1)):.2f} cm, 95 % {100 * np.quantile(res_x[:, mask].max(1), LEVEL):.2f} cm)")  # fmt: skip
        c["scale_direction"] = new
for name, c in CAL.items():
    log(f"{name}: fixed margin {100 * c['fixed_reference']:.2f} cm (direct), {100 * c['fixed_constant']:.2f} cm (after the mean offset), "
        f"{100 * c['fixed_direction']:.2f} cm (after the point model); radius scales {c['scale_reference']:.2f} / {c['scale_direction']:.2f}")  # fmt: skip
torch.save(CAL, OUT / "radius.pt")

# ---- the three replays, paired by reset
Td, Tk, Tc = VP.load([DD]), VP.load([DK]), VP.load([DC])
names = json.load(open(DD / "routes.json"))["names"]
key = lambda t: t["reset"] % 100000  # noqa: E731  (REACH_ID0 is a multiple of 100000: the reset's index in its run)
common = np.intersect1d(np.intersect1d(key(Td), key(Tk)), key(Tc))
ix = [np.array([np.where(key(t) == k)[0][0] for k in common]) for t in (Td, Tk, Tc)]
i_d, i_k, i_c = ix
zk, zc = dict(np.load(DK / "all.npz")), dict(np.load(DC / "all.npz"))
want = zc["comp_ref0"][Tc["seg"][i_c]].astype(float)
log(f"replays: {len(common)} triples of {len(names) * 4}; wanted sweep differs between the runs by at most "
    f"{1000 * max(np.abs(want - Td['ref'][i_d]).max(), np.abs(want - zk['comp_ref0'][Tk['seg'][i_k]]).max()):.2f} mm")  # fmt: skip
ACT = dict(direct=per_sphere(DD, Td, i_d, want, None), const=per_sphere(DK, Tk, i_k, want, None),
           ours=per_sphere(DC, Tc, i_c, want, None))  # fmt: skip
sent = zc["comp_ref"][Tc["seg"][i_c]].astype(float), zc["comp_off"][Tc["seg"][i_c]].astype(float)
pred = np.zeros((len(common), 24))  # the predicted sweep of the command sent, against the wanted sweep
for n in range(len(common)):
    P = np.concatenate([sent[0][n], sent[0][n][-1:]]) + sent[1][n]
    pred[n] = VM.sweep_err(P, np.concatenate([want[n], want[n][-1:]]))
route = Td["route"][i_d]
kind = np.array([names[r].split("_")[0] for r in route])


def outcomes(name: str, rows: np.ndarray) -> dict:
    """way -> (go (n, len(GRID)), touch (n, len(GRID))) for the replays `rows`."""
    mask, c = SETS[name], CAL[name]
    e = {k: v[rows][:, mask].max(1) for k, v in ACT.items()}
    r_ref = c["scale_reference"] * np.exp(VM.predict(c["nets_reference"], Td["x"][i_d[rows]])[:, 0])
    r_res = c["scale_direction"] * np.exp(VM.predict(c["nets_direction"], Tc["x"][i_c[rows]])[:, 0])
    g = GRID[None]
    return dict(
        small=(np.broadcast_to(g >= SMALL, (len(rows), len(GRID))), e["direct"][:, None] > g),
        fixed=(np.broadcast_to(g >= c["fixed_reference"], (len(rows), len(GRID))), e["direct"][:, None] > g),
        radius=(g >= r_ref[:, None], e["direct"][:, None] > g),
        const=(np.broadcast_to(g >= c["fixed_constant"], (len(rows), len(GRID))), e["const"][:, None] > g),
        ours=(pred[rows][:, mask].max(1)[:, None] + r_res[:, None] <= g, e["ours"][:, None] > g),
    )


def narrowest(go, touch) -> float:
    ok = ((go & ~touch).mean(0) >= 0.90) & ((go & touch).mean(0) <= 0.05)
    return float(GRID[np.argmax(ok)]) if ok.any() else float("nan")


def table(name: str, tag: str, sel: np.ndarray, res: dict) -> None:
    rows = np.where(sel)[0]
    O = outcomes(name, rows)
    log(f"\n{tag}, spheres '{name}': {len(np.unique(route[rows]))} routes, {len(rows)} reaches")
    e = {k: v[rows][:, SETS[name]].max(1) for k, v in ACT.items()}
    log("  error against the wanted sweep, median / 90 % / 95 % cm: " + "; ".join(
        f"{k} {100 * np.median(v):.2f} / {100 * np.percentile(v, 90):.2f} / {100 * np.percentile(v, 95):.2f}" for k, v in e.items()))  # fmt: skip
    log("  half-width cm     " + "".join(f"{100 * c:6.0f}" for c in GRID[::4]))
    for w in WAYS:
        go, touch = O[w]
        log(f"  {w:7s} success  " + "".join(f"{x:6.2f}" for x in (go & ~touch).mean(0)[::4]))
        log(f"  {'':7s} collide  " + "".join(f"{x:6.2f}" for x in (go & touch).mean(0)[::4]))
    star = {w: narrowest(*O[w]) for w in WAYS}
    rng, groups = np.random.default_rng(0), [np.where(route[rows] == r)[0] for r in np.unique(route[rows])]
    bs = {w: [] for w in WAYS}
    ratio = []
    for _ in range(1000):
        j = np.concatenate([groups[k] for k in rng.integers(len(groups), size=len(groups))])
        s = {w: narrowest(O[w][0][j], O[w][1][j]) for w in WAYS}
        for w in WAYS:
            bs[w].append(s[w])
        ratio.append(s["ours"] / np.nanmin([s[w] for w in WAYS if w != "ours"]))
    best = min((w for w in WAYS if w != "ours"), key=lambda w: (np.isnan(star[w]), star[w]))
    log("  narrowest gap passed safely (success >= 0.90, collisions <= 0.05), half-width cm, with its 95 % interval:")
    for w in WAYS:
        lo, hi = np.nanpercentile(bs[w], [2.5, 97.5])
        log(f"    {w:7s} {100 * star[w]:6.2f}  ({100 * lo:.2f} to {100 * hi:.2f})")
    lo, hi = np.nanpercentile(ratio, [2.5, 97.5])
    log(f"  ours / best other way ({best}): {star['ours'] / star[best]:.2f} ({lo:.2f} to {hi:.2f})")
    k5 = int(np.argmin(np.abs(GRID - 0.05)))
    log("  at a half-width of 5 cm, success / collision / abstention: " + "; ".join(
        f"{w} {(O[w][0] & ~O[w][1]).mean(0)[k5]:.2f} / {(O[w][0] & O[w][1]).mean(0)[k5]:.2f} / {(~O[w][0]).mean(0)[k5]:.2f}" for w in WAYS))  # fmt: skip
    res[f"{tag} | {name}"] = dict(narrowest_cm={w: 100 * star[w] for w in WAYS}, ratio=float(star["ours"] / star[best]),
                                 ratio_ci=[float(lo), float(hi)], best_other=best, reaches=len(rows))  # fmt: skip


res: dict = {}
detour = kind != "FreshUnder"
for name in SETS:
    table(name, "detour routes", detour, res)
if SEEN:
    table(
        "through",
        f"detour routes without the {len(SEEN)} looked at during development",
        detour & ~np.isin(route, SEEN),
        res,
    )
table("through", "near-straight routes (under a board)", ~detour, res)
table("through", "all routes", np.ones(len(kind), bool), res)
json.dump(res, open(OUT / "gap.json", "w"), indent=1)
