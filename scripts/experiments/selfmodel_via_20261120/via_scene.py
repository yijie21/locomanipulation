"""selfmodel_via_20261120: would the recorded sweeps have touched the boxes each route was planned around?

    $PY via_scene.py <model dir> <routes dir> <base replay dir> [<replay dir> ...]

The replays ran on the open floor. Here every route's boxes (<routes dir>/prep/prep_<NAME>.npz: centre and half sizes in
the heading frame of the stand, the frame arm.npz is written in) are put back. Per reach and per sphere the smallest
clearance over a sweep is the distance to the nearest box minus the sphere's radius, negative inside a box. Four sweeps:
  wanted   the original command's reference plan (comp_ref0; a direct run's own reference): our whole-body plan for the
           planner's hand route, not the planner's joint path
  plan     the reference plan of the command that was sent
  model    the sweep <model dir> predicts for the command that was sent (the plan plus the predicted offsets)
  actual   the recorded 10 Hz ticks
A reach TOUCHES when any of its 24 spheres has an actual clearance below 0. Detour routes only (every kind but FreshUnder).
Printed per run and per group of spheres: the median clearance and the share of reaches touching, for each sweep; how the
model's sweep foresees the touches; and per later run, against the base run (the direct one, or another compensated run)
pair by pair (same reset index), both shares and their difference with a 95 % interval from resampling routes.
The CLEARANCE LOST by a reach is the wanted plan's smallest clearance minus the actual one, over all 24 spheres: how much
of the room the plan left to the boxes the execution used up (negative when the arm stayed further away than planned). Its
95 % level is the extra margin a planner would have to leave for 95 % of the reaches not to come closer than planned for.
Printed per run with a 95 % interval (routes resampled), and per later run as a ratio to the base run's 95 % level; the
ratio is printed once more for every box displaced by 1 cm along +x, -x, +y, -y, +z, -z (how much it leans on the frame).
A reach ARRIVES when the palm's settled end (via_comp_eval.py's: the first palm sphere over the last 0.5 s) is within 2 cm
of the wanted plan's end, Motor.reach's own tolerance. Printed per run: the share arriving, and the share that arrives and
touches nothing; per later run the difference of the latter to the base run.
-> <replay dir>/scene_check.npz: reset index, route, the four (reaches, 24) clearance arrays, end (m), and the smallest
wanted and actual clearance under the six displacements (reaches, 6).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import via_model as VM  # noqa: E402
import via_point as VP  # noqa: E402

SWEEPS = ("wanted", "plan", "model", "actual")
SHIFTS = {
    f"{'xyz'[k]}{'+-'[q]}1": np.eye(6)[k] * (0.01, -0.01)[q] for k in range(3) for q in range(2)
}  # box centres, m
ARRIVE = 0.02
GROUPS = {
    "shoulder": lambda n: "shoulder" in n,
    "elbow": lambda n: "elbow" in n,
    "wrist": lambda n: "wrist" in n,
    "palm": lambda n: "palm" in n,
    "fingers": lambda n: n.startswith("hand_") and "palm" not in n,
    "all 24": lambda n: True,
}


def clearance(P: np.ndarray, boxes: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """P (..., spheres, 3), boxes (n, 6: centre, half sizes) -> per sphere, the smallest over the leading axes and the
    boxes of the signed distance to the box minus the sphere's radius (m)."""
    P = P.reshape(-1, P.shape[-2], 3)
    best = np.full(P.shape[1], np.inf)
    for b in boxes:
        q = np.abs(P - b[:3]) - b[3:]
        d = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)
        best = np.minimum(best, d.min(0))
    return best - radius


def check(M, R: Path, D: Path) -> dict:
    T, z, X = VP.load([D]), dict(np.load(D / "all.npz")), dict(np.load(D / "arm.npz"))
    names = json.load(open(D / "routes.json"))["names"]
    link, radius = [str(x) for x in X["link"]], X["radius"].astype(float)
    rows = np.arange(len(T["seg"]))
    rows = rows[[not names[r].startswith("FreshUnder") for r in T["route"]]]
    off = VP.offsets(M, T, rows)["direction"]
    C, palm, end, sw, sa = {k: [] for k in SWEEPS}, int(np.where(T["palm"])[0][0]), [], [], []
    for n, j in enumerate(rows):
        boxes = np.load(R / "prep" / f"prep_{names[T['route'][j]]}.npz")["boxes"]
        ticks, ref = X["X"][T["a"][j] : T["b"][j]].astype(float), T["ref"][j].astype(float)
        want = z["comp_ref0"][T["seg"][j]].astype(float) if "comp_ref0" in z else ref
        C["wanted"].append(clearance(want, boxes, radius))
        C["plan"].append(clearance(ref, boxes, radius))
        C["model"].append(clearance(np.concatenate([ref, ref[-1:]]) + off[n], boxes, radius))
        C["actual"].append(clearance(ticks, boxes, radius))
        end.append(float(np.linalg.norm(ticks[-VM.TAIL :, palm].mean(0) - want[-1, palm])))
        sw.append([clearance(want, boxes + d, radius).min() for d in SHIFTS.values()])
        sa.append([clearance(ticks, boxes + d, radius).min() for d in SHIFTS.values()])
    C = {k: np.array(v) for k, v in C.items()}
    route = np.asarray(T["route"])[rows]
    print(f"{D.parent.name}/{D.name}: {len(rows)} reaches on {len(set(route.tolist()))} detour routes")
    print("  median clearance cm / share touching" + "".join(f"{k:>18s}" for k in SWEEPS))
    for g, f in GROUPS.items():
        i = [k for k, n in enumerate(link) if f(n)]
        print(f"  {g:36s}" + "".join(f"{100 * np.median(C[k][:, i].min(1)):9.2f} / {np.mean(C[k][:, i].min(1) < 0):.3f}" for k in SWEEPS))  # fmt: skip
    hit, lo = C["actual"].min(1) < 0, C["model"].min(1)
    for m in (0.0, 0.01, 0.02):
        w = lo < m
        print(f"  model's sweep within {100 * m:.0f} cm of a box as the warning: {w.sum()} warned, {np.sum(hit & ~w)} of {hit.sum()} touches "
              f"not warned, {np.sum(~hit & w)} warned without a touch")  # fmt: skip
    if hit.any():
        deep = -C["actual"].min(1)[hit]
        by = ", ".join(f"{g} {np.sum(C['actual'][hit][:, [k for k, n in enumerate(link) if f(n)]].min(1) < 0)}" for g, f in list(GROUPS.items())[:-1])  # fmt: skip
        print(f"  depth of the touches: median {100 * np.median(deep):.2f} cm, largest {100 * deep.max():.2f} cm; reaches touching with: {by}")  # fmt: skip
    out = dict(reset=np.asarray(z["reset"])[T["seg"][rows]] % 100000, route=route, end=np.array(end), shift_wanted=np.array(sw),
               shift_actual=np.array(sa), **C)  # fmt: skip
    print(f"  palm's settled end within {100 * ARRIVE:.0f} cm of the wanted end: {np.mean(out['end'] <= ARRIVE):.3f} of the reaches; arrives and "
          f"touches nothing: {np.mean(good(out)):.3f}")  # fmt: skip
    np.savez(D / "scene_check.npz", **out)
    loss, rng = lost(out), np.random.default_rng(1)
    g = [np.where(route == r)[0] for r in np.unique(route)]
    bs = [
        np.percentile(loss[np.concatenate([g[q] for q in rng.integers(len(g), size=len(g))])], 95) for _ in range(2000)
    ]
    print(f"  clearance lost (wanted minus actual, all 24): median {100 * np.median(loss):.2f} cm, 90 % level {100 * np.percentile(loss, 90):.2f}, "
          f"95 % level {100 * np.percentile(loss, 95):.2f} ({100 * np.percentile(bs, 2.5):.2f} to {100 * np.percentile(bs, 97.5):.2f}), "
          f"largest {100 * loss.max():.2f}; reaches that lost more than 2 cm: {np.mean(loss > 0.02):.3f}")  # fmt: skip
    return out


def good(o: dict) -> np.ndarray:
    """Per reach: arrives (the palm's settled end within ARRIVE of the wanted end) and touches nothing."""
    return (o["end"] <= ARRIVE) & (o["actual"].min(1) >= 0)


def lost(o: dict) -> np.ndarray:
    """Per reach, the wanted plan's smallest clearance minus the actual one (m), over all 24 spheres."""
    return o["wanted"].min(1) - o["actual"].min(1)


def main(argv=None) -> int:
    a = sys.argv[1:] if argv is None else argv
    M = torch.load(Path(a[0]) / "via_point.pt", map_location="cpu", weights_only=False)
    R = Path(a[1])
    base = check(M, R, Path(a[2]))
    rng = np.random.default_rng(0)
    for d in a[3:]:
        o = check(M, R, Path(d))
        _, i, j = np.intersect1d(base["reset"], o["reset"], return_indices=True)
        a_hit, b_hit, route = base["actual"][i].min(1) < 0, o["actual"][j].min(1) < 0, base["route"][i]
        g = [np.where(route == r)[0] for r in np.unique(route)]
        bs = []
        for _ in range(2000):
            k = np.concatenate([g[q] for q in rng.integers(len(g), size=len(g))])
            bs.append(b_hit[k].mean() - a_hit[k].mean())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  against {Path(a[2]).name}, {len(i)} pairs on {len(g)} routes: touching {a_hit.sum()} ({a_hit.mean():.3f}) there, {b_hit.sum()} "
              f"({b_hit.mean():.3f}) this run; difference {b_hit.mean() - a_hit.mean():+.3f} ({lo:+.3f} to {hi:+.3f}); touching in both "
              f"{np.sum(a_hit & b_hit)}, only there {np.sum(a_hit & ~b_hit)}, only this run {np.sum(~a_hit & b_hit)}")  # fmt: skip
        la, lb, bs = lost(base)[i], lost(o)[j], []
        for _ in range(2000):
            k = np.concatenate([g[q] for q in rng.integers(len(g), size=len(g))])
            bs.append(np.percentile(lb[k], 95) / np.percentile(la[k], 95))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  95 % level of the clearance lost: {100 * np.percentile(la, 95):.2f} cm there, {100 * np.percentile(lb, 95):.2f} cm this run; "
              f"ratio {np.percentile(lb, 95) / np.percentile(la, 95):.2f} ({lo:.2f} to {hi:.2f})")  # fmt: skip
        ga, gb, bs = good(base)[i], good(o)[j], []
        for _ in range(2000):
            k = np.concatenate([g[q] for q in rng.integers(len(g), size=len(g))])
            bs.append(gb[k].mean() - ga[k].mean())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  arrives and touches nothing: {ga.mean():.3f} there, {gb.mean():.3f} this run; difference {gb.mean() - ga.mean():+.3f} ({lo:+.3f} to {hi:+.3f})")  # fmt: skip
        r = []
        for q, name in enumerate(SHIFTS):
            la = base["shift_wanted"][i, q] - base["shift_actual"][i, q]
            lb = o["shift_wanted"][j, q] - o["shift_actual"][j, q]
            r.append(f"{name} {np.percentile(lb, 95) / np.percentile(la, 95):.2f} ({np.mean(base['shift_actual'][i, q] < 0):.3f} / {np.mean(o['shift_actual'][j, q] < 0):.3f})")  # fmt: skip
        print(
            "  boxes displaced by 1 cm, ratio of the 95 % levels of the clearance lost (touching there / this run): "
            + ", ".join(r)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
