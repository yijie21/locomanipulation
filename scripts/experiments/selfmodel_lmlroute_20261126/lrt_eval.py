"""selfmodel_lmlroute_20261126's metrics (the entry's 度量 / 预测 / 判据), one controller; read-only (episode records, segs,
the planner's answers).

    $VPY lrt_eval.py --backend scalebfm [--json OUT.json]          (numpy + scipy)

Groups (revision 1: all four run in this round, on the looks with the simulator's mask): A = alone, R = reachable, C =
route, CR = route_reachable, from $LC_OUT (default loco_outputs/selfmodel_lmlroute_20261126); $LC_AR moves A and R
elsewhere (default: the same directory). Episodes 30..119 paired by (task, ep); LC_EVAL_EPS=a:b overrides (smoke only).
Simulator truth only scores.

E1 the decision set: lmlcmp3's rule (lc3_eval C1) on A: per task the share of episodes failed before the groups split
   (perception, the plan after the look, no look) > 50 % -> excluded; A >= 80 % -> ceiling; the pre-split failures the
   same in A, R, C, CR (checked). Revision 1: per task the pre-split failures by kind (perception / the plan after the
   look / no look) and the looks' mask source.
E2 per task: successes A R C CR, discordant pairs of C v A, CR v R, CR v C, R v A (revision 1) (McNemar exact p).
E3 pooled, decision set and all 7 tasks: P1 = C v A, P2 = CR v R (paired McNemar exact; Holm over the two: the smaller p
   < 0.025, the other < 0.05); CR v C; R v A (revision 1: LML's main table on the same looks; not in Holm); close-time hand to the planned grasp point, paired differences. Sensitivity: the same
   without episodes 30..39 (40..119), P1 and P2 with Holm.
E4 where the first furniture contact was (lmlroute dev contact_split.py's rule): the window's first contact (frame, body)
   -- before the first reach command ('before': 伸手前), else the right hand point at that frame (the segment's nearest 10 Hz
   W row) against the pre-grasp and the grasp targets the robot meant (p_cmd + Rh(yaw_cmd) goal_h of the k = 0 / k = 1
   reaches; the look's grasp point when the entry never went): both > 10 cm -> 'way' (路上), else 'near' (目标附近).
   M1: decision set, episodes after the split whose first contact is 'way', C v A and CR v R paired (both / x only / y
   only, McNemar), the change in %; apple apart (revision 1: salt apart too); 'near' alongside. Also (descriptive, C / CR only): episodes with any
   contact row on the way (lmlroute.contacts, every row).
E5 failure reasons (lc3_eval's order: pregrasp_touch first), hit_furniture split into way / near / before.
E6 the routes: per task (episodes after the split) the answers' status, the share routed, the gate's clearance, via points,
   path length, the route command's time and IK error; planning failures.
E7 M2 (routed episodes): the executed hand against the route (segment 0's 10 Hz W rows to the dense route polyline: the
   largest and the median distance), the executed arm's least clearance to the look's voxels over the route segment
   (arm.npz X spheres, segment 0's rows) against the plan's own (plan_clear_kept_cm); at an episode's first contact (any
   segment): the executed arm's clearance to the look voxels at the nearest 10 Hz row (<= 1 cm: the look saw what was hit
   -- execution went into it; > 3 cm: it did not -- the map).
E8 the criteria rows and the predictions (revision 1: the predictions rewritten for the simulator's mask).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path("/wuji-vefps-D/wuji-il/yijie/loco_outputs")
OUT = Path(os.environ.get("LC_OUT", str(ROOT / "selfmodel_lmlroute_20261126")))
AR = Path(os.environ.get("LC_AR", str(OUT)))  # revision 1: A and R run in this round too
TASKS = ("IThorBathroomLiftCandleFromFloor", "IThorKitchenApproachTableAndLiftApple", "IThorKitchenApproachTableAndLiftBowl",
         "IThorKitchenApproachTableAndLiftSaltShaker", "IThorKitchenTakeStatueFromCabinet",
         "IThorLivingRoomApproachTableAndLiftDumbbell", "IThorLivingRoomApproachTableAndLiftWateringCan")  # fmt: skip
SHORT = dict(zip(TASKS, ("candle", "apple", "bowl", "salt", "statue_cab", "dumbbell", "wcan")))
NAME = {"A": "alone", "R": "reachable", "C": "route", "CR": "route_reachable"}
SRC = {"A": AR, "R": AR, "C": OUT, "CR": OUT}
G = ("A", "R", "C", "CR")
COMPARE = (("C", "A"), ("CR", "R"), ("CR", "C"), ("R", "A"))  # revision 1: R v A (LML's main table, the same looks) as a secondary row
REASONS = ("ok", "perception", "pregrasp_touch", "stance_or_ik", "fell_walking", "hit_furniture", "missed", "dropped", "other")
APPLE = "IThorKitchenApproachTableAndLiftApple"
SALT = "IThorKitchenApproachTableAndLiftSaltShaker"  # revision 1: in the decision set once the simulator's mask finds it
_lo, _hi = (int(v) for v in os.environ.get("LC_EVAL_EPS", "30:120").split(":"))
EPS = range(_lo, _hi)
NEP = len(EPS)
WAY_CM = 10.0
VOX = 0.02


# -- statistics (lc3_eval's) ------------------------------------------------------------------------------------------
def mcnemar(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def sign_p(d: np.ndarray) -> float:
    d = d[d != 0]
    return mcnemar(int((d < 0).sum()), int((d > 0).sum())) if len(d) else 1.0


def median_ci(d: np.ndarray, conf: float = 0.95) -> tuple[float, float]:
    d = np.sort(d)
    n = len(d)
    if n < 6:
        return float("nan"), float("nan")
    from scipy.stats import binom

    k = int(binom.ppf((1 - conf) / 2, n, 0.5))
    while k > 0 and binom.cdf(k - 1, n, 0.5) > (1 - conf) / 2:
        k -= 1
    k = max(k, 1)
    return float(d[k - 1]), float(d[n - k])


def holm2(p1: float, p2: float, alpha: float = 0.05) -> tuple[bool, bool, float, float]:
    order = sorted([(p1, 0), (p2, 1)])
    rej = [False, False]
    adj = [0.0, 0.0]
    (pa, ia), (pb, ib) = order
    adj[ia] = min(1.0, 2 * pa)
    adj[ib] = min(1.0, max(adj[ia], pb))
    if pa < alpha / 2:
        rej[ia] = True
        rej[ib] = pb < alpha
    return rej[0], rej[1], adj[0], adj[1]


def med(v) -> float:
    return float(np.median(v)) if len(v) else float("nan")


# -- records ------------------------------------------------------------------------------------------------------------
def load(be: str) -> dict:
    rs = {}
    for g in G:
        for p in glob.glob(str(SRC[g] / be / "episodes" / "*" / NAME[g] / "ep*.json")):
            r = json.loads(Path(p).read_text())
            if r["ep"] in EPS:
                rs[(r["task"], g, r["ep"])] = r
    return rs


def presplit(r: dict) -> bool:
    p = r.get("plan") or {}
    return p.get("fail") in ("perception", "reach") or ("error" in r and not r.get("plan"))


def pre_kind(r: dict) -> str | None:
    p = r.get("plan") or {}
    if p.get("fail") == "perception":
        return "perception"
    if p.get("fail") == "reach":
        return "plan after look"
    if "error" in r and not r.get("plan"):
        return "no look"
    return None


def touched_pre(r: dict) -> bool | None:
    d = next((d for d in r.get("d3", []) if d["reach_no"] == 1), None)
    return None if d is None else d["obj_rows"] > 0


def reason3(r: dict) -> str:
    if r.get("reason") == "ok":
        return "ok"
    if not presplit(r) and touched_pre(r):
        return "pregrasp_touch"
    return r.get("reason", "other")


def close(r: dict) -> float:
    v = r.get("close_dist_cm")
    return float("nan") if v is None else float(v)


def tgt_w(w: dict) -> np.ndarray:
    p, y = np.asarray(w["p_cmd"], float), float(w["yaw_cmd"])
    c, s = np.cos(y), np.sin(y)
    g = np.asarray(w["goal_h"], float)
    return np.array([p[0] + c * g[0] - s * g[1], p[1] + s * g[0] + c * g[1], g[2]])


def seg_rows(r: dict, z) -> list[np.ndarray]:
    """Per segment, the frame of each 10 Hz row (lr_run's grid: TICKS = 5 frames, k0 = the command row)."""
    return [r["d3"][s]["f_cmd"] + 5 * (np.arange(int(z["n_steps"][s])) - int(z["k0"][s])) for s in range(len(z["n_steps"]))]


def row_at(r: dict, z, fr: int) -> int | None:
    """contact_split's row for frame fr: inside a segment's rows (+-2 frames) the nearest; between segments the last row
    of the segment before."""
    off = np.r_[0, np.cumsum(z["n_steps"])]
    best = None
    for s, frames in enumerate(seg_rows(r, z)):
        if frames[0] - 2 <= fr <= frames[-1] + 2:
            return int(off[s] + int(np.argmin(np.abs(frames - fr))))
        if frames[0] <= fr:
            best = int(off[s + 1] - 1)
    return best


_Z: dict = {}


def segz(be: str, g: str, t: str, e: int, arm: bool = False):
    key = (be, g, t, e, arm)
    if key not in _Z:
        p = SRC[g] / be / "segs" / t / NAME[g] / ("arm" if arm else "") / f"ep{e:+05d}.npz"
        _Z[key] = dict(np.load(p)) if p.exists() else None
    return _Z[key]


def where(be: str, g: str, r: dict) -> str | None:
    """The first contact: None (none), 'before', 'way', 'near', 'unknown' (no hand row)."""
    fc = r.get("first_contact")
    if fc is None or presplit(r) or not r.get("reaches"):
        return None
    fr = int(fc[0])
    rs = r["reaches"]
    if not any(w["f_cmd"] <= fr for w in rs):
        return "before"
    z = segz(be, g, r["task"], r["ep"])
    if z is None:
        return "unknown"
    i = row_at(r, z, fr)
    if i is None:
        return "unknown"
    hand = z["W"][i, 4:7].astype(float)
    tg = {int(w["k"]): tgt_w(w) for w in rs if "p_cmd" in w}
    ds = [100 * float(np.linalg.norm(hand - tg[k])) for k in (0, 1) if k in tg]
    if 1 not in tg and (r.get("plan") or {}).get("goal") is not None:
        ds.append(100 * float(np.linalg.norm(hand - np.asarray(r["plan"]["goal"], float))))
    if not ds:
        return "unknown"
    return "way" if min(ds) > WAY_CM else "near"


def any_way_row(r: dict) -> bool:
    """C / CR: any contact row of the window with the hand > 10 cm from both targets (lmlroute.contacts)."""
    rs = r.get("reaches", [])
    tg = {int(w["k"]): tgt_w(w) for w in rs if "p_cmd" in w}
    if 1 not in tg and (r.get("plan") or {}).get("goal") is not None:
        tg[1] = np.asarray(r["plan"]["goal"], float)
    out = False
    for c in (r.get("lmlroute") or {}).get("contacts", []):
        f, h = int(c[0]), np.asarray(c[2:5], float)
        if not any(w["f_cmd"] <= f for w in rs):
            continue
        ds = [100 * float(np.linalg.norm(h - tg[k])) for k in (0, 1) if k in tg]
        out |= bool(ds) and min(ds) > WAY_CM
    return out


def box_sd(X, c, h):
    dd = np.abs(X - c) - h
    return np.linalg.norm(np.maximum(dd, 0), axis=-1) + np.minimum(dd.max(-1), 0)


def clearance(C, r, tree, V) -> float:
    """Least sphere-surface to voxel-cube distance (cm); lrt_plan's."""
    if tree is None:
        return float("inf")
    d, j = tree.query(C, k=min(8, len(V)))
    j = np.atleast_2d(j).reshape(len(C), -1)
    return 100 * min(float((box_sd(C[i], V[j[i]], 0.5 * VOX) - r[i]).min()) for i in range(len(C)))


def answer(be: str, t: str, e: int):
    p = OUT / "routes" / "routes" / be / t / f"ep{e:+05d}.npz"
    if not p.exists():
        return None, {}
    z = np.load(p, allow_pickle=False)
    return json.loads(str(z["meta"])), {k: z[k] for k in z.files if k != "meta"}


def poly_dist(P: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Distance of each point of P to the polyline H."""
    a, b = H[:-1], H[1:]
    ab = b - a
    L2 = np.maximum((ab**2).sum(1), 1e-12)
    out = []
    for p in P:
        u = np.clip(((p - a) * ab).sum(1) / L2, 0, 1)
        out.append(float(np.linalg.norm(a + u[:, None] * ab - p, axis=1).min()))
    return np.array(out)


def main(be: str, js: str | None) -> None:
    rs = load(be)
    res: dict = dict(backend=be, eps=[EPS.start, EPS.stop])
    n_by = Counter(g for (_, g, _) in rs)
    get = lambda t, g, e: rs.get((t, g, e))  # noqa: E731
    have = lambda t, e: all(get(t, g, e) is not None for g in G)  # noqa: E731
    EP = {t: [e for e in EPS if have(t, e)] for t in TASKS}
    miss = [(SHORT[t], g, e) for t in TASKS for g in G for e in EPS if (t, g, e) not in rs]
    print(f"== {be}: records {dict(n_by)} (A, R from {AR}; C, CR from {OUT}); episodes {EPS.start}..{EPS.stop - 1}; missing {len(miss)} "
          f"{miss[:6]}; worker errors {sum('error' in r for r in rs.values())}")  # fmt: skip

    # E1
    print(f"E1. decision set (lmlcmp3's rule on A): pre-split > 50 % -> out; A >= 80 % ({int(np.ceil(0.8 * NEP))}/{NEP}) -> ceiling")
    per = {}
    for t in TASKS:
        sets = {g: tuple(presplit(get(t, g, e)) for e in EP[t]) for g in G}
        same = len(set(sets.values())) == 1
        pre = sets["A"]
        share = sum(pre) / max(1, len(pre))
        nA = sum(bool(get(t, "A", e)["success"]) for e in EP[t])
        excl, top = share > 0.5 or not pre, nA >= 0.8 * NEP
        per[t] = dict(share=share, excl=excl, top=top, same=same, n_pre=sum(pre), n=len(pre), A=nA)
        flag = f"EXCLUDED (pre-split {100 * share:.0f} %)" if excl else f"AT CEILING (A {nA})" if top else "decision set"
        kinds = Counter(pre_kind(get(t, "A", e)) for e in EP[t])
        src = Counter((get(t, "A", e).get("look") or {}).get("mask_source", "sam3" if get(t, "A", e).get("look") else "no look") for e in EP[t])
        per[t].update(perception=kinds["perception"], plan_after_look=kinds["plan after look"], no_look=kinds["no look"], mask=dict(src))
        print(f"  {SHORT[t]:10s} pre-split {sum(pre)}/{len(pre)} ({100 * share:.0f} %): perception {kinds['perception']}, plan after the look "
              f"{kinds['plan after look']}, no look {kinds['no look']}; the same in A R C CR {same} | A {nA}/{len(pre)} | {flag} | masks {dict(src)}")  # fmt: skip
    DEC = [t for t in TASKS if not per[t]["excl"] and not per[t]["top"]]
    res["decision_set"] = [SHORT[t] for t in DEC]
    res["e1"] = {SHORT[t]: v for t, v in per.items()}
    print(f"  decision set: {[SHORT[t] for t in DEC]}; pre-split the same everywhere: {all(v['same'] for v in per.values())}")

    # E2
    print("E2. per task: successes and discordant pairs (y-only / x-only, McNemar exact p)")
    res["per_task"] = {}
    for t in TASKS:
        n = {g: sum(bool(get(t, g, e)["success"]) for e in EP[t]) for g in G}
        parts = []
        for x, y in COMPARE:
            b = sum(bool(get(t, y, e)["success"]) and not get(t, x, e)["success"] for e in EP[t])
            c = sum(bool(get(t, x, e)["success"]) and not get(t, y, e)["success"] for e in EP[t])
            parts.append(f"{x}v{y} {y}-only {b} {x}-only {c} p {mcnemar(b, c):.3g}")
        res["per_task"][SHORT[t]] = n
        print(f"  {SHORT[t]:10s} n {len(EP[t])} " + " ".join(f"{g} {n[g]:2d}" for g in G) + " | " + " | ".join(parts))

    # E3
    print("E3. pooled")
    res["pooled"] = {}

    def pooled(name, ts, eps_ok=lambda e: True, quiet=False):
        out = {}
        for x, y in COMPARE:
            P = [(get(t, x, e), get(t, y, e)) for t in ts for e in EP[t] if eps_ok(e)]
            nx, ny = sum(bool(a["success"]) for a, _ in P), sum(bool(b["success"]) for _, b in P)
            b = sum(bool(bb["success"]) and not aa["success"] for aa, bb in P)
            c = sum(bool(aa["success"]) and not bb["success"] for aa, bb in P)
            p = mcnemar(b, c)
            dpp = 100 * (nx - ny) / max(1, len(P))
            ent = dict(nx=nx, ny=ny, b=b, c=c, p=p, dpp=dpp, n=len(P))
            if not quiet:
                print(f"    {x} vs {y}: {x} {nx} ({100 * nx / max(1, len(P)):.1f} %) {y} {ny} ({100 * ny / max(1, len(P)):.1f} %) diff {dpp:+.1f} pp "
                      f"({nx - ny:+d}) | {y}-only {b} {x}-only {c} McNemar exact p {p:.3g}")  # fmt: skip
                d = np.array([close(aa) - close(bb) for aa, bb in P if not np.isnan(close(aa)) and not np.isnan(close(bb))])
                if len(d):
                    lo, hi = median_ci(d)
                    print(f"       close-time hand to the grasp point, paired {x} - {y}: median {np.median(d):+.2f} cm [{lo:+.2f}, {hi:+.2f}], n {len(d)}, "
                          f"{x} closer {int((d < 0).sum())}, {y} closer {int((d > 0).sum())}, sign p {sign_p(d):.3g}")  # fmt: skip
                    ent.update(close_med=float(np.median(d)), close_ci=[lo, hi], close_n=len(d), sign_p=sign_p(d))
            out[f"{x}v{y}"] = ent
        r1, r2, a1, a2 = holm2(out["CvA"]["p"], out["CRvR"]["p"])
        out["holm"] = dict(rej1=r1, rej2=r2, adj1=a1, adj2=a2)
        print(f"    [{name}] Holm over P1 (C v A) and P2 (CR v R): raw p {out['CvA']['p']:.3g} / {out['CRvR']['p']:.3g}, adjusted {a1:.3g} / {a2:.3g}; "
              f"P1 (C > A and significant) {out['CvA']['nx'] > out['CvA']['ny'] and r1}, P2 (CR > R and significant) {out['CRvR']['nx'] > out['CRvR']['ny'] and r2}")  # fmt: skip
        return out

    for name, ts in (("decision set", DEC), ("all 7 tasks", list(TASKS))):
        print(f"  {name} ({', '.join(SHORT[t] for t in ts)}): {sum(len(EP[t]) for t in ts)} pairs")
        res["pooled"][name] = pooled(name, ts)
    print("  sensitivity: without episodes 30..39 (40..119)")
    sens = pooled("decision set, 40..119", DEC, lambda e: e >= 40, quiet=True)
    for k in ("CvA", "CRvR"):
        v = sens[k]
        print(f"    {k}: {v['nx']} vs {v['ny']} of {v['n']} ({v['dpp']:+.1f} pp), discordant {v['b']} / {v['c']}, p {v['p']:.3g}")
    res["pooled"]["sensitivity 40..119"] = sens

    # E4
    print("E4. the first furniture contact: before the first reach / on the way (hand > 10 cm from both targets) / near a target")
    W = {(t, g, e): where(be, g, get(t, g, e)) for t in TASKS for g in G for e in EP[t]}
    res["contacts"] = {}
    for scope, ts in (("decision set", DEC), ("apple", [APPLE]), ("salt", [SALT]), ("all 7 tasks", list(TASKS))):
        for g in G:
            cnt = Counter(W[(t, g, e)] for t in ts for e in EP[t] if not presplit(get(t, g, e)))
            res["contacts"][f"{scope}|{g}"] = {str(k): v for k, v in cnt.items()}
            print(f"  {scope:12s} {g:2s} way {cnt['way']:3d} near {cnt['near']:3d} before {cnt['before']:3d} unknown {cnt['unknown']} none {cnt[None]}")
    res["m1"] = {}
    for scope, ts in (("decision set", DEC), ("apple", [APPLE]), ("salt", [SALT])):
        for x, y in (("C", "A"), ("CR", "R")):
            for kind in ("way", "near"):
                P = [(W[(t, x, e)] == kind, W[(t, y, e)] == kind) for t in ts for e in EP[t] if not presplit(get(t, x, e))]
                nx, ny = sum(a for a, _ in P), sum(b for _, b in P)
                both = sum(a and b for a, b in P)
                xo, yo = sum(a and not b for a, b in P), sum(b and not a for a, b in P)
                chg = 100 * (nx - ny) / ny if ny else float("nan")
                res["m1"][f"{scope}|{x}v{y}|{kind}"] = dict(nx=nx, ny=ny, both=both, x_only=xo, y_only=yo, change_pct=chg, p=mcnemar(xo, yo))
                print(f"  M1 [{scope}] {kind:4s} {x} {nx} vs {y} {ny} ({chg:+.0f} %): both {both}, {x} only {xo}, {y} only {yo}, McNemar p {mcnemar(xo, yo):.3g}")
    for g in ("C", "CR"):
        n = sum(any_way_row(get(t, g, e)) for t in DEC for e in EP[t] if not presplit(get(t, g, e)))
        print(f"  (descriptive) {g}: decision-set episodes with any contact row on the way {n}")

    # E5
    print("E5. failure reasons (pregrasp_touch first; hit_furniture by where the first contact was)")
    res["reasons"] = {}
    for name, ts in (("decision set", DEC), ("all 7 tasks", list(TASKS))):
        for g in G:
            cnt = Counter()
            for t in ts:
                for e in EP[t]:
                    r = get(t, g, e)
                    k = reason3(r)
                    if k == "hit_furniture":
                        k = f"hit_furniture_{W[(t, g, e)]}"
                    cnt[k] += 1
            res["reasons"][f"{name}|{g}"] = dict(cnt)
            keys = [k for k in REASONS if k != "hit_furniture"] + ["hit_furniture_way", "hit_furniture_near", "hit_furniture_before", "hit_furniture_unknown", "hit_furniture_None"]
            print(f"  {name:12s} {g:2s} " + " ".join(f"{k} {cnt.get(k, 0)}" for k in keys if cnt.get(k, 0) or k in REASONS))

    # E6
    print("E6. the routes (episodes after the split): answers, routed, gate clearance, via points, path, route command")
    res["routes"] = {}
    for t in TASKS:
        st, on, gates, nv, path, dur, ike, ikm, tpl = Counter(), 0, [], [], [], [], [], [], []
        n = 0
        for e in EP[t]:
            r = get(t, "C", e)
            if presplit(r):
                continue
            n += 1
            rt = (r.get("lmlroute") or {}).get("route") or {}
            st[rt.get("status")] += 1
            if rt.get("gate_clear_cm") is not None:
                gates.append(rt["gate_clear_cm"])
            if rt.get("t_plan_s") is not None:
                tpl.append(rt["t_plan_s"])
            if (r.get("lmlroute") or {}).get("route_on"):
                on += 1
                w = r["reaches"][0]
                nv.append(w.get("n_via"))
                path.append(rt.get("path_cm"))
                d0 = r["d3"][0]
                dur.append((d0["f_end"] - d0["f_cmd"]) / 50)
                ike.append(w.get("ik_err_cm"))
                ikm.append(w.get("ik_err_max_cm"))
        g_ = np.array(gates, float)
        res["routes"][SHORT[t]] = dict(n=n, routed=on, status=dict(st), gate_med=med(g_), n_via_med=med(nv), path_med=med(path), dur_med=med(dur),
                                       ik_med=med(ike), ik_max_med=med(ikm), t_plan_med=med(tpl))  # fmt: skip
        print(f"  {SHORT[t]:10s} n {n:2d} routed {on:2d} ({100 * on / max(1, n):.0f} %) | {dict(st)} | gate median {med(g_):.2f} cm (< 5: {int((g_ < 5).sum())}) | "
              f"via {med(nv):.0f} path {med(path):.0f} cm time {med(dur):.1f} s IK end {med(ike):.2f} / max {med(ikm):.2f} cm | plan {med(tpl):.1f} s")  # fmt: skip
    allst = Counter()
    for t in DEC:
        for e in EP[t]:
            r = get(t, "C", e)
            if not presplit(r):
                allst[((r.get("lmlroute") or {}).get("route") or {}).get("status")] += 1
    nplan = sum(v for k, v in allst.items() if k not in ("GATE_CLEAR", None))
    nfail = sum(v for k, v in allst.items() if k in ("START_IK_FAIL", "PLAN_FAIL", "ERROR", "NO_INNER_VIA"))
    res["plan_fail"] = dict(planned=nplan, failed=nfail, status=dict(allst))
    print(f"  decision set: {dict(allst)}; planning failures (start IK, plan, error, no inner via) {nfail}/{nplan}")

    # E7
    print("E7. M2 (routed episodes): executed hand vs the route, executed arm vs the look's voxels")
    m2 = dict(dev_max=[], dev_med=[], arm_clear=[], plan_clear=[], contact_clear=[], contact_seen=Counter())
    for t in TASKS:
        for e in EP[t]:
            for g in ("C", "CR"):
                r = get(t, g, e)
                if not (r.get("lmlroute") or {}).get("route_on"):
                    continue
                meta, arr = answer(be, t, e)
                z, za = segz(be, g, t, e), segz(be, g, t, e, True)
                if meta is None or z is None or za is None or "H" not in arr:
                    continue
                k0, n0 = int(z["k0"][0]), int(z["n_steps"][0])
                hand = z["W"][k0:n0, 4:7].astype(float)
                if g == "C":
                    dv = poly_dist(hand, arr["H"].astype(float))
                    m2["dev_max"].append(100 * float(dv.max()))
                    m2["dev_med"].append(100 * float(np.median(dv)))
                V = arr.get("V")
                if V is None or not len(V):
                    continue
                V = V.astype(float)
                W0 = z["W"][k0].astype(float)
                Rs = np.array([[np.cos(W0[3]), -np.sin(W0[3])], [np.sin(W0[3]), np.cos(W0[3])]])
                Vs = np.c_[(V[:, :2] - W0[:2]) @ Rs, V[:, 2]]
                tree = cKDTree(Vs)
                rad = za["radius"].astype(float)
                X = za["X"][k0:n0].astype(float)
                if g == "C":
                    m2["arm_clear"].append(min(clearance(x, rad, tree, Vs) for x in X))
                    if meta.get("plan_clear_kept_cm") is not None:
                        m2["plan_clear"].append(meta["plan_clear_kept_cm"])
                fc = r.get("first_contact")
                if fc is not None:
                    i = row_at(r, z, int(fc[0]))
                    if i is not None:
                        off = np.r_[0, np.cumsum(z["n_steps"])]
                        s = int(np.searchsorted(off, i, side="right") - 1)
                        Ws = z["W"][off[s] + int(z["k0"][s])].astype(float)
                        Rq = np.array([[np.cos(Ws[3]), -np.sin(Ws[3])], [np.sin(Ws[3]), np.cos(Ws[3])]])
                        Vq = np.c_[(V[:, :2] - Ws[:2]) @ Rq, V[:, 2]]
                        cc = clearance(za["X"][i].astype(float), rad, cKDTree(Vq), Vq)
                        m2["contact_clear"].append(cc)
                        m2["contact_seen"][f"{g} " + ("seen (<= 1 cm)" if cc <= 1 else "unseen (> 3 cm)" if cc > 3 else "1-3 cm")] += 1
    res["m2"] = dict(dev_max_med=med(m2["dev_max"]), dev_med_med=med(m2["dev_med"]), arm_clear_med=med(m2["arm_clear"]),
                     plan_clear_med=med(m2["plan_clear"]), contact_clear_med=med(m2["contact_clear"]), contact_seen=dict(m2["contact_seen"]),
                     n=len(m2["dev_max"]))  # fmt: skip
    ac = np.array(m2["arm_clear"])
    print(f"  C, {len(m2['dev_max'])} routed episodes: executed hand to the route, largest per episode median {med(m2['dev_max']):.1f} cm, "
          f"median per episode median {med(m2['dev_med']):.1f} cm | executed arm's least clearance to the voxels over the route median "
          f"{med(ac):.1f} cm (< 0: {int((ac < 0).sum())}) vs the plan's {med(m2['plan_clear']):.1f} cm")  # fmt: skip
    print(f"  at the first contact (C and CR, routed): arm clearance to the look's voxels median {med(m2['contact_clear']):.1f} cm; {dict(m2['contact_seen'])}")

    # E8
    print("E8. criteria rows and predictions")
    pd_ = res["pooled"]["decision set"]
    P1, P2, H = pd_["CvA"], pd_["CRvR"], pd_["holm"]
    p1 = P1["nx"] > P1["ny"] and H["rej1"]
    p2 = P2["nx"] > P2["ny"] and H["rej2"]
    m1w = res["m1"]["decision set|CvA|way"]
    red = -m1w["change_pct"]
    harm = P1["nx"] < P1["ny"] and P1["p"] < 0.05
    crc = pd_["CRvC"]
    rows = {"P1 and M1 >= 25 % fewer": p1 and red >= 25, "not P1 but M1 >= 25 % fewer": (not p1) and red >= 25,
            "M1 < 10 % fewer": red < 10, "C - A < 0 and p < 0.05": harm, "P2": p2, "not P2 but P1": p1 and not p2,
            "CR - C <= 0": crc["nx"] - crc["ny"] <= 0}  # fmt: skip
    for k, v in rows.items():
        print(f"  row '{k}': {v}")
    print(f"  P1 C - A {P1['dpp']:+.1f} pp (raw p {P1['p']:.3g}, Holm {H['adj1']:.3g}); P2 CR - R {P2['dpp']:+.1f} pp (raw p {P2['p']:.3g}, Holm "
          f"{H['adj2']:.3g}); M1 way C v A {m1w['nx']} vs {m1w['ny']} ({-red:+.0f} %); CR - C {crc['dpp']:+.1f} pp")  # fmt: skip
    inr = lambda v, lo, hi: lo <= v <= hi  # noqa: E731
    pt = res["per_task"]
    # revision 1 (the user's decision of 2026-11-26: the simulator's mask; A R C CR all new): the predictions rewritten
    pre_pred = dict(candle=(11, 12), apple=(13, 17), bowl=(53, 58), salt=(0, 2), statue_cab=(10, 13), dumbbell=(0, 1), wcan=(0, 1))
    e1 = res["e1"]
    print("  pre-split per task (predicted, every controller): " + "; ".join(
        f"{k} {e1[k]['n_pre']} ({v[0]}-{v[1]}: {inr(e1[k]['n_pre'], *v)})" for k, v in pre_pred.items()))  # fmt: skip
    dec_pred = {"scalebfm": ["candle", "apple", "salt", "statue_cab", "dumbbell", "wcan"], "heft": ["candle", "apple", "salt", "statue_cab", "dumbbell", "wcan"],
                "sonic": ["apple", "salt", "statue_cab", "dumbbell", "wcan"]}  # fmt: skip
    print(f"  decision set {res['decision_set']} (predicted {dec_pred.get(be)}: {res['decision_set'] == dec_pred.get(be)})")
    rate = lambda g: 100 * sum(pt[SHORT[t]][g] for t in DEC) / max(1, sum(len(EP[t]) for t in DEC))  # noqa: E731
    if be == "scalebfm":
        m1r = res["m1"]["decision set|CRvR|way"]
        m1a, m1s = res["m1"]["apple|CvA|way"], res["m1"]["salt|CvA|way"]
        nr = res["m1"]["decision set|CvA|near"]
        print(f"  A {rate('A'):.1f} % (11-20: {inr(rate('A'), 11, 20)}); R {rate('R'):.1f} % (29-41: {inr(rate('R'), 29, 41)}); C {rate('C'):.1f} % "
              f"(17-30: {inr(rate('C'), 17, 30)}); CR {rate('CR'):.1f} % (33-47: {inr(rate('CR'), 33, 47)})")  # fmt: skip
        rva = pd_["RvA"]
        print(f"  C - A {P1['dpp']:+.1f} (+2..+14: {inr(P1['dpp'], 2, 14)}); CR - R {P2['dpp']:+.1f} (-1..+12: {inr(P2['dpp'], -1, 12)}); CR - C "
              f"{crc['dpp']:+.1f} (+10..+24: {inr(crc['dpp'], 10, 24)}); R - A {rva['dpp']:+.1f} (+12..+28: {inr(rva['dpp'], 12, 28)}, p {rva['p']:.3g})")  # fmt: skip
        print(f"  M1 C v A {m1w['change_pct']:+.0f} % (-25..-55: {inr(-m1w['change_pct'], 25, 55)}); apple {m1a['change_pct']:+.0f} % (-30..-65: "
              f"{inr(-m1a['change_pct'], 30, 65)}); salt {m1s['change_pct']:+.0f} % (-30..-65: {inr(-m1s['change_pct'], 30, 65)}); CR v R "
              f"{m1r['change_pct']:+.0f} % (-20..-50: {inr(-m1r['change_pct'], 20, 50)}); near C - A {nr['nx'] - nr['ny']:+d} (within +-15: "
              f"{abs(nr['nx'] - nr['ny']) <= 15})")  # fmt: skip
        pr = dict(candle=dict(A=(17, 35), R=(66, 79)), apple=dict(A=(8, 21), R=(11, 26), C=(20, 41), CR=(23, 44)),
                  salt=dict(A=(4, 18), R=(28, 52), C=(15, 45), CR=(35, 65)), statue_cab=dict(A=(0, 1), R=(0, 1), C=(0, 3), CR=(0, 3)),
                  dumbbell=dict(A=(17, 33), R=(21, 37), C=(20, 36), CR=(24, 40)), wcan=dict(A=(2, 13), R=(17, 35), C=(4, 19), CR=(16, 38)))  # fmt: skip
        parts = [f"{k} " + " ".join(f"{g} {pt[k][g]} ({inr(pt[k][g], *v[g])})" for g in ("A", "R", "C", "CR") if g in v) for k, v in pr.items()]
        parts.append(f"candle C = A {pt['candle']['C']} / {pt['candle']['A']}, CR = R {pt['candle']['CR']} / {pt['candle']['R']}")
        print("  per task: " + "; ".join(parts))
        rr = res["routes"]
        print("  routed share: " + "; ".join(f"{k} {100 * rr[k]['routed'] / max(1, rr[k]['n']):.0f} %" for k in ("apple", "salt", "statue_cab", "dumbbell", "wcan", "candle"))
              + f" (predicted apple / salt / statue >= 90, dumbbell 80-90, wcan 70-80, candle ~0); planning failures {nfail}/{nplan} (<= 3 %: "
              f"{nfail <= 0.03 * max(1, nplan)})")  # fmt: skip
    elif be == "sonic":
        print(f"  candle A {pt['candle']['A']} (70-79, at the ceiling >= 72: {pt['candle']['A'] >= 72}); A {rate('A'):.1f} % (3-10: {inr(rate('A'), 3, 10)}); "
              f"R {rate('R'):.1f} % (6-14: {inr(rate('R'), 6, 14)}); C - A {P1['dpp']:+.1f} (-2..+4: {inr(P1['dpp'], -2, 4)}); CR - R {P2['dpp']:+.1f} "
              f"(-2..+4: {inr(P2['dpp'], -2, 4)})")  # fmt: skip
    else:
        print(f"  candle A {pt['candle']['A']} (38-60) R {pt['candle']['R']} (58-76); A {rate('A'):.1f} % (8-14: {inr(rate('A'), 8, 14)}); R {rate('R'):.1f} % "
              f"(10-17: {inr(rate('R'), 10, 17)}); C - A {P1['dpp']:+.1f} (-1..+4: {inr(P1['dpp'], -1, 4)}); CR - R {P2['dpp']:+.1f} (-1..+4: "
              f"{inr(P2['dpp'], -1, 4)})")  # fmt: skip
    res["criteria"] = rows
    if js:
        Path(js).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    main(a.backend, a.json)
