"""selfmodel_via_20261120, fourth step, part two (the plan entry's "第四步第二部分"): the narrow-gap test with real obstacles.

    $PY gap_real.py calib <out dir> <routes dir> <direct> <const> <comp> <radius.pt>
    $PY gap_real.py score <out dir> <routes dir> <calib dir> <direct obst> <const obst> <comp obst> \
        <direct open> <const open> <comp open> [--part1 <part one's direct replay dir>]

Every run is gap_replay.sh's: one reset per (slot, pass), slot = (base scene s, level f, margin m) of <routes dir>/slots.json;
direct / const / comp: the planned command sent as is / moved by the training paths' mean offset / compensated (position and
attitude, via_point_onpol2_20261123); obst: the path's real obstacles in the simulation, open: the open floor. Within one
routes dir every run of a (s, pass) starts from the same state, whatever f, m and the way of sending (gen_reach.py,
REACH_VIA_SLOTS), so each rule below -- try the margins from small to large and go with the first that passes -- is scored by
reading, for the margin it picks, the outcome of the run that sent that command.

The quantities a rule looks at before the command come from the open-floor runs (the same command from the same state; the
obstacle runs are checked to hold the same, bit for bit). Spheres: arm_spheres.py's 24; "through" the wrist, palm and fingers'
17. Distances to the real obstacles (the path's prep boxes): a polyline is filled in with 10 points a segment.
  wanted sweep   the planned command's reference sweep (direct: the table's ref; const / comp: comp_ref0)
  predicted      the compensated command's reference sweep plus the model's offsets (comp_ref + comp_off, at run time)
calib (development scenes, open floor): the error of the actual sweep against the wanted one (through, the largest over the
spheres and the path), its 95 % level for the direct command (-> the calibrated fixed margin, rounded up to the margin grid)
and for the constant correction (-> const's margin); the radius model of part one (radius.pt, through, reference) rescaled to
hold 95 % of the direct errors; the error of the actual sweep against the predicted one for the compensated command, its 95 %
level (-> mu) and part one's residual radius model (through, direction) rescaled likewise (-> rule A); the 95 % level of the
compensated command's error against the wanted sweep (-> the ablation's fixed margin). -> <out>/calib.json, calib.log.
score (test scenes): per (s, f, pass) and way, the margin picked (or none: an abstention) and the outcome of that reach in the
obstacle run: success (went, no wrist / palm / finger penetration, no fall), collision (went, penetration), abstention, fall.
  fixed m      every m of the grid: go when the plan exists, direct
  fixed_cal    fixed at the calibrated margin
  radius       first m with clearance(wanted, through) >= the radius model's r for this command, direct
  const        first m with clearance(wanted, through) >= const's margin, constant correction
  ours         first m with clearance(predicted, through) >= mu, compensated
  rule_a       first m with clearance(predicted, through) >= the residual radius for this command, compensated (secondary)
  comp_fixed   fixed at the ablation's margin, compensated (ablation, not an opponent)
The narrowest gap passed safely f*: the smallest level with success >= 0.90 and collisions <= 0.05; intervals by resampling base
scenes. The best opponent: the smallest f* of fixed_cal, the best fixed m after the fact (the m with the smallest f* on these
very scenes, chosen again in every resample), radius and const. Also: the arm's (24 spheres) collisions, falls, the open-floor
spheres' verdict (actual sweep within the real obstacles) against MuJoCo's contact, and the fixed margins' collision rate
against part one's at the same half-width (--part1: its direct replays, the detour routes).
-> <out>/gap_real.log, gap_real.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import via_model as VM  # noqa: E402
import via_point as VP  # noqa: E402

WHAT, OUT, RT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
SL = json.load(open(RT / "slots.json"))
NSLOT, LEVELS, MARGINS = len(SL["names"]), SL["levels"], SL["margins"]
KEY = np.array(SL["key"])  # (slots, 3): s, f, m
LEVEL = VM.LEVEL
logf = open(OUT / f"gap_real_{WHAT}.log", "w")


def log(s=""):
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


def dense(P: np.ndarray, k: int = 10) -> np.ndarray:
    """A polyline (M, S, 3) with k - 1 points put in each segment -> (k (M - 1) + 1, S, 3)."""
    u = np.linspace(0, 1, k, endpoint=False)[:, None, None, None]
    mid = (P[:-1][None] + u * (P[1:] - P[:-1])[None]).transpose(1, 0, 2, 3).reshape(-1, *P.shape[1:])
    return np.concatenate([mid, P[-1:]])


_BOX: dict = {}


def boxes(name: str) -> list:
    """The path's real obstacles in its heading frame (as via_gen.load_routes places them): [(centre, half, axes)]."""
    if name not in _BOX:
        pr = np.load(RT / "prep" / f"prep_{name}.npz")
        o, Rp = pr["pelvis_p"], pr["pelvis_R"]
        yaw = math.atan2(Rp[1, 0], Rp[0, 0])
        c, s = math.cos(yaw), math.sin(yaw)
        Rh = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        B = pr["boxes"]
        cen = np.c_[(B[:, :2] - o[:2]) @ Rh[:2, :2], B[:, 2]]
        _BOX[name] = [(cen[k], B[k, 3:], Rh.T) for k in range(len(B))]
    return _BOX[name]


def clearance(X: np.ndarray, rad: np.ndarray, bx: list) -> float:
    """X (..., S, 3) sphere centres, rad (S,) -> the smallest distance to the boxes (negative: inside)."""
    best = np.inf
    for c, h, Rb in bx:
        q = np.abs((X - c) @ Rb) - h
        best = min(best, float((np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0) - rad).min()))
    return best


def run(D: Path, with_table: bool = True) -> dict:
    """One gap run, one entry per segment: s, f, m, pass, slot, fell, touch (n, 3) ...; and per segment that did not fall
    (row >= 0), its table row's features, wanted sweep, errors and clearances."""
    z = dict(np.load(D / "all.npz"))
    names = json.load(open(D / "routes.json"))["names"]
    i = z["reset"] % 100000
    slot, pas = i % NSLOT, i // NSLOT
    route = [names[r] for r in z["via_route"]]
    assert all(SL["names"][sl] == r for sl, r in zip(slot, route)), f"{D}: a reset ran another path than its slot's"
    R = dict(D=D, slot=slot, pas=pas, s=KEY[slot, 0], f=KEY[slot, 1], m=KEY[slot, 2], name=np.array(route),
             fell=z["fell"].astype(bool), n=len(slot))  # fmt: skip
    if "gap_touch" in z:
        R.update(
            touch=z["gap_touch"].astype(bool), t=z["gap_t"], pen=z["gap_pen"], force=z["gap_force"], t0=z["gap_t0"]
        )
    if not with_table:
        return R
    T = VP.load([D])
    A = np.load(D / "arm.npz")
    X, rad, link = A["X"], A["radius"], [str(x) for x in A["link"]]
    thr = np.array([x == "wrist_pitch_link" or x.startswith("hand_") for x in link])
    row = np.full(R["n"], -1)
    row[T["seg"]] = np.arange(len(T["seg"]))
    comp = "comp_ref0" in z
    n = R["n"]
    R.update(row=row, x=np.full((n, T["x"].shape[1]), np.nan, np.float32), thr=thr, rad=rad,
             e_want=np.full((n, 24), np.nan), c_want=np.full((n, 2), np.nan), c_act=np.full((n, 2), np.nan),
             want=np.full((n, VM.Q, 24, 3), np.nan, np.float32))  # fmt: skip
    if comp:
        R.update(e_pred=np.full((n, 24), np.nan), c_pred=np.full((n, 2), np.nan),
                 pred=np.full((n, VM.Q + 1, 24, 3), np.nan, np.float32))  # fmt: skip
    for k in np.where(row >= 0)[0]:
        j = row[k]
        bx = boxes(R["name"][k])
        want = (z["comp_ref0"][k] if comp else T["ref"][j]).astype(float)
        Pw = np.concatenate([want, want[-1:]])
        Xa = X[T["a"][j] : T["b"][j]].astype(float)
        R["x"][k], R["want"][k] = T["x"][j], want
        R["e_want"][k] = VM.sweep_err(Xa, Pw)
        Dw = dense(Pw)
        R["c_want"][k] = clearance(Dw[:, thr], rad[thr], bx), clearance(Dw, rad, bx)
        R["c_act"][k] = clearance(Xa[:, thr], rad[thr], bx), clearance(Xa, rad, bx)
        if comp:
            cr = z["comp_ref"][k].astype(float)
            Pp = np.concatenate([cr, cr[-1:]]) + z["comp_off"][k].astype(float)
            R["pred"][k] = Pp
            R["e_pred"][k] = VM.sweep_err(Xa, Pp)
            Dp = dense(Pp)
            R["c_pred"][k] = clearance(Dp[:, thr], rad[thr], bx), clearance(Dp, rad, bx)
    log(f"{D.name}: {n} reaches, fell {int(R['fell'].sum())}, in the table {int((row >= 0).sum())}"
        + (f", touched (through / arm / any) {R['touch'].sum(0).tolist()}" if "touch" in R else ""))  # fmt: skip
    return R


def up_to_grid(v: float) -> float:
    g = [m for m in MARGINS if m / 100 >= v - 1e-9]
    return float(g[0]) if g else float("nan")


def q95(v) -> float:
    v = np.asarray(v, float)
    return float(np.quantile(v[~np.isnan(v)], LEVEL))


if WHAT == "calib":
    import torch

    DD, DK, DC, RAD = (Path(x) for x in sys.argv[4:8])
    Rd, Rk, Rc = run(DD), run(DK), run(DC)
    thr = Rd["thr"]
    ed, ek, ec = (R["e_want"][:, thr].max(1) for R in (Rd, Rk, Rc))
    res = Rc["e_pred"][:, thr].max(1)
    CAL = torch.load(RAD, map_location="cpu", weights_only=False)["through"]
    ok_d, ok_c = ~np.isnan(ed), ~np.isnan(res)
    r_ref = np.exp(VM.predict(CAL["nets_reference"], Rd["x"][ok_d])[:, 0])
    r_res = np.exp(VM.predict(CAL["nets_direction"], Rc["x"][ok_c])[:, 0])
    out = dict(
        fixed_cal_cm=up_to_grid(q95(ed)), direct_q95_cm=100 * q95(ed), const_cm=100 * q95(ek), mu_cm=100 * q95(res),
        comp_fixed_cm=up_to_grid(q95(ec)), comp_q95_cm=100 * q95(ec), scale_radius=q95(ed[ok_d] / r_ref),
        scale_rule_a=q95(res[ok_c] / r_res), part_one_scales=[CAL["scale_reference"], CAL["scale_direction"]],
        reaches=dict(direct=int(ok_d.sum()), const=int((~np.isnan(ek)).sum()), comp=int(ok_c.sum())), radius_pt=str(RAD),
    )  # fmt: skip
    for tag, e in (("direct", ed), ("const", ek), ("comp", ec), ("comp against the predicted sweep", res)):
        e = e[~np.isnan(e)]
        log(f"through, {tag}: {len(e)} reaches, median {100 * np.median(e):.2f} cm, 90 % {100 * np.quantile(e, 0.9):.2f}, "
            f"95 % {100 * np.quantile(e, LEVEL):.2f}")  # fmt: skip
    log(json.dumps(out, indent=1))
    json.dump(out, open(OUT / "calib.json", "w"), indent=1)
    sys.exit(0)

# ---------------------------------------------------------------------------------------------------- score
import torch  # noqa: E402

CD = Path(sys.argv[4])
C = json.load(open(CD / "calib.json"))
NETS = torch.load(C["radius_pt"], map_location="cpu", weights_only=False)["through"]
ARGS = sys.argv[11:]
PART1 = Path(ARGS[ARGS.index("--part1") + 1]) if "--part1" in ARGS else None
DO = dict(direct=Path(sys.argv[5]), const=Path(sys.argv[6]), comp=Path(sys.argv[7]))
DF = dict(direct=Path(sys.argv[8]), const=Path(sys.argv[9]), comp=Path(sys.argv[10]))
O = {w: run(D, with_table=False) for w, D in DO.items()}
F = {w: run(D) for w, D in DF.items()}
log(f"calibration ({CD}): " + json.dumps(C))


def index(R: dict) -> dict:
    return {(int(s), int(f), int(m), int(p)): k for k, (s, f, m, p) in enumerate(zip(R["s"], R["f"], R["m"], R["pas"]))}


IO, IF = {w: index(R) for w, R in O.items()}, {w: index(R) for w, R in F.items()}
S_ALL = sorted({int(s) for s in KEY[:, 0]})
CASES = [(s, f, p) for s in S_ALL for f in LEVELS for p in range(4)]
planned = {tuple(int(v) for v in KEY[k]) for k, nm in enumerate(SL["names"]) if nm is not None}

# the obstacle runs ran the same commands from the same states as the open-floor ones: check on the start (the first
# recorded tick is not kept per reach here, so check the wanted sweep of the compensated / constant runs and the sent ones)
for w in ("const", "comp"):
    zo, zf = np.load(DO[w] / "all.npz"), np.load(DF[w] / "all.npz")
    a = {k: i for k, i in IO[w].items()}
    both = [(a[k], IF[w][k]) for k in a if k in IF[w]]
    d = max(float(np.abs(zo["comp_ref"][i] - zf["comp_ref"][j]).max()) for i, j in both) if both else float("nan")
    log(
        f"check, {w}: the command sent, obstacle run against open floor, {len(both)} reaches: largest difference {1000 * d:.4f} mm"
    )


def pre(w: str, key: tuple):
    """The open-floor run's entry of (s, f, m, pass) for way of sending w, or None."""
    k = IF[w].get(key)
    return None if k is None or F[w]["row"][k] < 0 else k


def outcome(w: str, key: tuple):
    """(went, touched through, touched arm, fell) of the obstacle run's reach; a planned reach missing from the run (a
    skipped reset) counts as a fall."""
    k = IO[w].get(key)
    if k is None:
        return True, False, False, True
    R = O[w]
    return True, bool(R["touch"][k, 0]), bool(R["touch"][k, 1]), bool(R["fell"][k])


def radius_r(w: str, k: int, nets_key: str, scale: float) -> float:
    return scale * float(np.exp(VM.predict(NETS[nets_key], F[w]["x"][k : k + 1])[:, 0])[0])


def decide(way: str, s: int, f: int, p: int):
    """-> (margin picked or None, way of sending)."""
    if way.startswith("fixed_m"):
        m = int(way[7:])
        return (m if (s, f, m) in planned else None), "direct"
    if way in ("fixed_cal", "comp_fixed"):
        mm = C["fixed_cal_cm"] if way == "fixed_cal" else C["comp_fixed_cm"]
        if math.isnan(mm):
            return None, "direct"
        m = int(mm)
        return (m if (s, f, m) in planned else None), ("direct" if way == "fixed_cal" else "comp")
    send = dict(radius="direct", const="const", ours="comp", rule_a="comp")[way]
    for m in MARGINS:
        if (s, f, m) not in planned:
            continue
        k = pre(send, (s, f, m, p))
        if k is None:  # the open-floor reach fell: decide on nothing, count as going
            return m, send
        R = F[send]
        if way == "radius":
            ok = R["c_want"][k, 0] >= radius_r(send, k, "nets_reference", C["scale_radius"])
        elif way == "const":
            ok = R["c_want"][k, 0] >= C["const_cm"] / 100
        elif way == "ours":
            ok = R["c_pred"][k, 0] >= C["mu_cm"] / 100
        else:
            ok = R["c_pred"][k, 0] >= radius_r(send, k, "nets_direction", C["scale_rule_a"])
        if ok:
            return m, send
    return None, send


WAYS = [f"fixed_m{m}" for m in MARGINS] + ["fixed_cal", "radius", "const", "ours", "rule_a", "comp_fixed"]
RES = {}  # way -> arrays over CASES: go, touch, touch_arm, fell, m
for way in WAYS:
    go, tc, ta, fl, mm = [], [], [], [], []
    for s, f, p in CASES:
        m, send = decide(way, s, f, p)
        if m is None:
            go.append(False), tc.append(False), ta.append(False), fl.append(False), mm.append(np.nan)
            continue
        g, t1, t2, fe = outcome(send, (s, f, m, p))
        go.append(g), tc.append(t1), ta.append(t2), fl.append(fe), mm.append(m)
    RES[way] = {k: np.array(v) for k, v in zip(("go", "touch", "arm", "fell", "m"), (go, tc, ta, fl, mm))}
CS = np.array(CASES)


def rates(r: dict, sel: np.ndarray):
    succ = r["go"][sel] & ~r["touch"][sel] & ~r["fell"][sel]
    return (
        succ.mean(),
        (r["go"][sel] & r["touch"][sel]).mean(),
        (~r["go"][sel]).mean(),
        (r["go"][sel] & r["fell"][sel]).mean(),
    )


def fstar(r: dict, rows: np.ndarray) -> float:
    for f in LEVELS:
        sel = rows[CS[rows, 1] == f]
        if not len(sel):
            continue
        su, co, _, _ = rates(r, sel)
        if su >= 0.90 and co <= 0.05:
            return float(f)
    return float("nan")


ALL = np.arange(len(CASES))
OPP = ["fixed_cal", "radius", "const"]
FIX = [f"fixed_m{m}" for m in MARGINS]


def best_opp(st: dict):
    """(name, f*) of the best opponent; the best fixed margin chosen on the scored set itself."""
    bf = min(FIX, key=lambda w: (math.isnan(st[w]), st[w]))
    cand = OPP + [bf]
    b = min(cand, key=lambda w: (math.isnan(st[w]), st[w]))
    return b, st[b], bf


log(f"\ntest scenes: {len(S_ALL)} base scenes x {len(LEVELS)} levels x 4 passes = {len(CASES)} cases; "
    f"planned (s, f, m): {len(planned)} of {NSLOT}")  # fmt: skip
log("per level, success / collision / abstention (falls in brackets when any):")
log("  way          " + "".join(f"{f:>16d}" for f in LEVELS))
for way in WAYS:
    cells = []
    for f in LEVELS:
        su, co, ab, fa = rates(RES[way], ALL[CS[:, 1] == f])
        cells.append(f"{su:.2f}/{co:.2f}/{ab:.2f}" + (f"[{fa:.2f}]" if fa > 0 else ""))
    log(f"  {way:12s} " + "".join(f"{c:>16s}" for c in cells))
STAR = {w: fstar(RES[w], ALL) for w in WAYS}
b, bstar, bf = best_opp(STAR)
groups = [np.where(CS[:, 0] == s)[0] for s in S_ALL]
rng = np.random.default_rng(0)
BS, RAT = {w: [] for w in WAYS}, []
for _ in range(1000):
    rows = np.concatenate([groups[k] for k in rng.integers(len(groups), size=len(groups))])
    st = {w: fstar(RES[w], rows) for w in WAYS}
    for w in WAYS:
        BS[w].append(st[w])
    RAT.append(st["ours"] / best_opp(st)[1])
log(
    "\nnarrowest gap passed safely f* (cm over the hand's thickness; success >= 0.90, collisions <= 0.05), 95 % interval:"
)
for w in WAYS:
    v = np.array(BS[w])
    lo, hi = np.nanpercentile(v, [2.5, 97.5]) if (~np.isnan(v)).any() else (np.nan, np.nan)
    log(f"  {w:12s} {STAR[w]:6.1f}  ({lo:.1f} to {hi:.1f}); not reached in {np.isnan(v).mean():.2f} of the resamples")
RAT = np.array(RAT)
ratio = STAR["ours"] / bstar
lo, hi = np.nanpercentile(RAT, [2.5, 97.5]) if (~np.isnan(RAT)).any() else (np.nan, np.nan)
log(f"  best opponent: {b} (best fixed after the fact: {bf}); ours / best: {ratio:.2f} ({lo:.2f} to {hi:.2f}); "
    f"undefined in {np.isnan(RAT).mean():.2f} of the resamples")  # fmt: skip
if math.isnan(bstar):
    win = [f for f in LEVELS if (lambda r: r[0] >= 0.9 and r[1] <= 0.05)(rates(RES["ours"], ALL[CS[:, 1] == f]))
           and all(not (lambda r: r[0] >= 0.9 and r[1] <= 0.05)(rates(RES[w], ALL[CS[:, 1] == f])) for w in OPP + FIX)]  # fmt: skip
    log(f"  no opponent reaches the bar; levels where ours does and none of them does: {win}")
log("\nmargin picked (median over the cases that went), arm collisions (24 spheres) and falls among them:")
for w in WAYS:
    r = RES[w]
    g = r["go"]
    log(f"  {w:12s} went {g.mean():.2f}, margin {np.nanmedian(r['m']) if g.any() else float('nan'):.1f} cm, "
        f"arm collisions {(g & r['arm']).mean():.2f}, falls {int((g & r['fell']).sum())}")  # fmt: skip

# ---- part one's verdict on the open floor against MuJoCo's contact, every reach that ran in both
log("\nopen-floor spheres (actual sweep within the real obstacles) against MuJoCo's contact, same reach:")
agree = {}
for w in DO:
    rows = [
        (IO[w][k], IF[w][k]) for k in IO[w] if k in IF[w] and F[w]["row"][IF[w][k]] >= 0 and not O[w]["fell"][IO[w][k]]
    ]
    for j, part in ((0, "through"), (1, "arm")):
        mj = np.array([O[w]["touch"][a, j] for a, _ in rows])
        sp = np.array([F[w]["c_act"][b_, j] < 0 for _, b_ in rows])
        agree[(w, part)] = float((mj == sp).mean())
        log(f"  {w:6s} {part:7s} {len(rows)} reaches: agree {(mj == sp).mean():.3f}; MuJoCo touch {mj.mean():.3f}, spheres "
            f"inside {sp.mean():.3f}; touch but not inside {(mj & ~sp).mean():.3f}, inside but no touch {(sp & ~mj).mean():.3f}")  # fmt: skip

out = dict(star=STAR, best=b, best_fixed=bf, ratio=ratio, ratio_ci=[float(lo), float(hi)], agree={f"{a}|{b_}": v for (a, b_), v in agree.items()},
           rates={w: {str(f): rates(RES[w], ALL[CS[:, 1] == f]) for f in LEVELS} for w in WAYS}, calib=C)  # fmt: skip

# ---- the fixed margins against part one at the same half-width
if PART1 is not None:
    T1 = VP.load([PART1])
    A1 = np.load(PART1 / "arm.npz")
    link1 = [str(x) for x in A1["link"]]
    thr1 = np.array([x == "wrist_pitch_link" or x.startswith("hand_") for x in link1])
    nm1 = json.load(open(PART1 / "routes.json"))["names"]
    det = np.array([nm1[r].split("_")[0] != "FreshUnder" for r in T1["route"]])
    e1 = np.array([VM.sweep_err(A1["X"][T1["a"][j] : T1["b"][j]].astype(float),
                                np.concatenate([T1["ref"][j], T1["ref"][j][-1:]]).astype(float))[thr1].max()
                   for j in np.where(det)[0]])  # fmt: skip
    log(f"\nfixed margin m: collision rate here (among the reaches that went) against part one's at half-width c = m "
        f"(direct, {len(e1)} detour reaches of {PART1.name}):")  # fmt: skip
    for m in MARGINS:
        r = RES[f"fixed_m{m}"]
        g = r["go"]
        log(f"  m {m:2d} cm: here {(r['touch'][g]).mean() if g.any() else float('nan'):.2f} ({int(g.sum())} reaches); "
            f"part one {(e1 > m / 100).mean():.2f}")  # fmt: skip
json.dump(out, open(OUT / "gap_real.json", "w"), indent=1, default=float)
