"""selfmodel_lmlcmp2_20261126's metrics (the entry's 度量 / 预测 / 判据), one controller; read-only (episode records only).

    $VPY lc2_eval.py --backend scalebfm [--json OUT.json]          (numpy + scipy)

Groups: A = alone (direct), S = selfmodel (lmlcmp's fixed point), G = guard (S's choice, then the 10 cm cap and the 3 cm
fallback), R = reachable (S's fixed point with every candidate shrunk back to IK error <= 2 cm). Jittered episodes 0..29
paired by (task, ep); the official unjittered start (-1) listed apart.

C0 consistency with lmlcmp: A and S against lmlcmp's alone / selfmodel records, every (task, ep) of -1..29: success,
   reason, close_dist_cm (equal = |d| < 1e-6 cm), the number of reaches; differences listed.
C1 the decision set (the entry's new rule): per task the share of episodes that failed before the groups split --
   the look's perception failure (plan.fail "perception"), the plan made after the look (plan.fail "reach"), or an
   episode whose look could not run (worker error, no plan) -- the same in every group by construction (checked);
   > 50 % -> excluded. Ceiling unchanged: A >= 80 % -> out. The rest is the decision set; all 7 tasks reported too.
C2 per task: successes of A / S / G / R; G-A, G-S, R-S discordant pairs and McNemar exact p.
C3 pooled (decision set, all 7 tasks), pairs G vs A (primary; "win" = >= 10 pp and McNemar p < 0.05 on the decision
   set), G vs S, R vs S, S vs A (lmlcmp's comparison, for reference): successes, difference (pp), discordant pairs,
   McNemar exact p. Close-time hand to the planned grasp point (close_dist_cm), paired G - A, G - S, R - S: median,
   distribution-free 95 % interval, how many closer, sign test, Wilcoxon.
C4 failure reasons per group (hardgrasp's order), all 7 tasks and the decision set; episodes with an executed reach IK
   error > 3 cm per group.
C5 overshoot on the far targets (watering can, dumbbell), first reach (pre-grasp, reach_no 1) of every group: the hand
   at the reach's landing row (act_h) minus the goal, along the horizontal direction from the pelvis to the goal in the
   command's heading frame; > 3 cm = past the goal. Share per task and both tasks; also among reaches with IK > 3 cm.
C6 triggers (jittered episodes; the official start apart): G -- the cap, the fallback to the goal (command changed),
   choices that were the goal already with D's miss > 3 cm; per reach kind and per task. R -- inverted reaches with a
   shrunk candidate, shrunk candidates, executed command shrunk, shrunk all the way to the goal (alpha 0).
C7 the criteria rows, evaluated as fixed in the revision before the formal run (G vs S rows on all 7 tasks, the unit of
   the prediction table; decision-set numbers printed alongside).
C8 the official unjittered start (-1) per group.
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

ROOT = Path("/wuji-vefps-D/wuji-il/yijie/loco_outputs")
OUT = Path(os.environ.get("LC_OUT", str(ROOT / "selfmodel_lmlcmp2_20261126")))
LMLCMP = ROOT / "selfmodel_lmlcmp_20261126"
TASKS = ("IThorBathroomLiftCandleFromFloor", "IThorKitchenApproachTableAndLiftApple", "IThorKitchenApproachTableAndLiftBowl",
         "IThorKitchenApproachTableAndLiftSaltShaker", "IThorKitchenTakeStatueFromCabinet",
         "IThorLivingRoomApproachTableAndLiftDumbbell", "IThorLivingRoomApproachTableAndLiftWateringCan")  # lc2_run.TASKS
SHORT = dict(zip(TASKS, ("candle", "apple", "bowl", "salt", "statue_cab", "dumbbell", "wcan")))
GROUPS = {"A": "alone", "S": "selfmodel", "G": "guard", "R": "reachable"}
REASONS = ("ok", "perception", "stance_or_ik", "fell_walking", "hit_furniture", "missed", "dropped", "other")
FAR = ("IThorLivingRoomApproachTableAndLiftWateringCan", "IThorLivingRoomApproachTableAndLiftDumbbell")
EPS = range(0, 30)


# -- statistics (hg_analyze's / lc_eval's) ----------------------------------------------------------------------------
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


def wilcoxon_p(d: np.ndarray) -> float:
    from scipy.stats import wilcoxon

    d = d[d != 0]
    return float(wilcoxon(d).pvalue) if len(d) >= 6 else float("nan")


# -- records ------------------------------------------------------------------------------------------------------------
def load(base: Path, be: str) -> dict:
    rs = {}
    for p in glob.glob(str(base / be / "episodes" / "*" / "*" / "ep*.json")):
        r = json.loads(Path(p).read_text())
        rs[(r["task"], r["method"], r["ep"])] = r
    return rs


def presplit(r: dict) -> bool:
    """Failed before the groups split: the look's perception, the plan after the look, or no look at all."""
    p = r.get("plan") or {}
    return p.get("fail") in ("perception", "reach") or ("error" in r and not r.get("plan"))


def close(r: dict) -> float:
    v = r.get("close_dist_cm")
    return float("nan") if v is None else float(v)


def ik_over(r: dict) -> bool:
    return any(w.get("ik_err_cm", 0) > 3.0 for w in r.get("reaches", []))


def overshoot_cm(w: dict) -> float:
    if "act_h" not in w:
        return float("nan")
    g, a = np.asarray(w["goal_h"], float), np.asarray(w["act_h"], float)
    u = g[:2] / max(np.linalg.norm(g[:2]), 1e-9)
    return 100 * float((a[:2] - g[:2]) @ u)


def main(be: str, js: str | None) -> None:
    rs = load(OUT, be)
    old = load(LMLCMP, be)
    res: dict = dict(backend=be)
    n_by = Counter(m for (_, m, _) in rs)
    print(f"== {be}: {len(rs)} episode records under {OUT / be / 'episodes'} {dict(n_by)}; worker errors "
          f"{sum('error' in r for r in rs.values())} ({sorted({(SHORT[t], e) for (t, m, e), r in rs.items() if 'error' in r})})")  # fmt: skip
    get = lambda t, g, e: rs.get((t, GROUPS[g], e))  # noqa: E731

    # C0
    print("C0. A and S against lmlcmp's records, episodes -1..29 of the 7 tasks")
    c0 = {}
    for g in ("A", "S"):
        n = ns = nr = nc = nn = 0
        dmax, diffs = 0.0, []
        for t in TASKS:
            for e in range(-1, 30):
                a, b = get(t, g, e), old.get((t, GROUPS[g], e))
                if a is None or b is None:
                    diffs.append((SHORT[t], e, "missing", a is None, b is None))
                    continue
                n += 1
                ok_s = bool(a.get("success")) == bool(b.get("success"))
                ok_r = a.get("reason") == b.get("reason")
                ca, cb = close(a), close(b)
                ok_c = (np.isnan(ca) and np.isnan(cb)) or (not np.isnan(ca) and not np.isnan(cb) and abs(ca - cb) < 1e-6)
                if not np.isnan(ca) and not np.isnan(cb):
                    dmax = max(dmax, abs(ca - cb))
                ok_n = len(a.get("reaches", [])) == len(b.get("reaches", []))
                ns += ok_s
                nr += ok_r
                nc += ok_c
                nn += ok_n
                if not (ok_s and ok_r and ok_c and ok_n):
                    diffs.append((SHORT[t], e, a.get("success"), b.get("success"), a.get("reason"), b.get("reason"), round(ca, 4), round(cb, 4)))
        c0[g] = dict(n=n, success_same=ns, reason_same=nr, close_same=nc, reaches_same=nn, close_max_d=dmax, diffs=diffs)
        print(f"  {g} vs lmlcmp {GROUPS[g]}: {n} episodes; success same {ns}, reason same {nr}, close_dist same (|d| < 1e-6 cm) {nc} "
              f"(max |d| {dmax:.2e} cm), reach count same {nn}; differing {len(diffs)} {diffs[:8]}")  # fmt: skip
    res["consistency"] = c0
    consistent = all(v["n"] == 217 and not v["diffs"] for v in c0.values())

    # C1
    print("C1. decision set: failures before the groups split (perception, the plan after the look, no look) > 50 % -> out; A >= 80 % -> ceiling")
    per = {}
    for t in TASKS:
        sets = {g: tuple(presplit(get(t, g, e)) for e in EPS if get(t, g, e) is not None) for g in GROUPS}
        same = len({v for v in sets.values()}) == 1 and all(len(v) == 30 for v in sets.values())
        pre = sets["A"]
        share = sum(pre) / max(1, len(pre))
        nA = sum(bool(get(t, "A", e)["success"]) for e in EPS)
        excl, top = share > 0.5, nA >= 0.8 * 30
        per[t] = dict(share=share, excl=excl, top=top, same=same)
        flag = f"EXCLUDED (pre-split {100 * share:.0f} %)" if excl else f"AT CEILING (A {nA}/30)" if top else "decision set"
        kinds = Counter(("perception" if (get(t, "A", e).get("plan") or {}).get("fail") == "perception" else
                         "plan after look" if (get(t, "A", e).get("plan") or {}).get("fail") == "reach" else "no look")
                        for e in EPS if presplit(get(t, "A", e)))  # fmt: skip
        print(f"  {SHORT[t]:10s} pre-split failures {sum(pre)}/30 ({100 * share:.0f} %) {dict(kinds)}; same in all four groups {same} | A {nA}/30 | {flag}")
    DEC = [t for t in TASKS if not per[t]["excl"] and not per[t]["top"]]
    res["decision_set"] = [SHORT[t] for t in DEC]
    print(f"  decision set: {[SHORT[t] for t in DEC]}")

    # C2
    print("C2. per task (jittered episodes 0..29): successes and discordant pairs (x-only / y-only, McNemar exact p)")
    res["per_task"] = {}
    for t in TASKS:
        n = {g: sum(bool(get(t, g, e)["success"]) for e in EPS) for g in GROUPS}
        parts = []
        for x, y in (("G", "A"), ("G", "S"), ("R", "S")):
            b = sum(get(t, y, e)["success"] and not get(t, x, e)["success"] for e in EPS)
            c = sum(get(t, x, e)["success"] and not get(t, y, e)["success"] for e in EPS)
            parts.append(f"{x}v{y} {y}-only {b} {x}-only {c} p {mcnemar(b, c):.3g}")
        res["per_task"][SHORT[t]] = n
        print(f"  {SHORT[t]:10s} A {n['A']:2d} S {n['S']:2d} G {n['G']:2d} R {n['R']:2d} | " + " | ".join(parts))

    # C3
    print("C3. pooled")
    res["pooled"] = {}
    for name, ts in (("decision set", DEC), ("all 7 tasks", list(TASKS))):
        print(f"  {name} ({', '.join(SHORT[t] for t in ts)}): {30 * len(ts)} pairs")
        for x, y in (("G", "A"), ("G", "S"), ("R", "S"), ("S", "A")):
            P = [(get(t, x, e), get(t, y, e)) for t in ts for e in EPS]
            nx, ny = sum(bool(a["success"]) for a, _ in P), sum(bool(b["success"]) for _, b in P)
            b = sum(bb["success"] and not aa["success"] for aa, bb in P)
            c = sum(aa["success"] and not bb["success"] for aa, bb in P)
            p = mcnemar(b, c)
            dpp = 100 * (nx - ny) / max(1, len(P))
            win = dpp >= 10 and p < 0.05
            line = f"    {x} vs {y}: success {x} {nx} ({100 * nx / len(P):.1f} %) {y} {ny} ({100 * ny / len(P):.1f} %) diff {dpp:+.1f} pp | {y}-only {b} {x}-only {c} McNemar exact p {p:.3g}"
            if (x, y) == ("G", "A"):
                line += f" | WIN {win}" if name == "decision set" else ""
            print(line)
            ent = dict(nx=nx, ny=ny, b=b, c=c, p=p, dpp=dpp, win=win, n=len(P))
            if (x, y) in (("G", "A"), ("G", "S"), ("R", "S")):
                d = np.array([close(aa) - close(bb) for aa, bb in P if not np.isnan(close(aa)) and not np.isnan(close(bb))])
                if len(d):
                    lo, hi = median_ci(d)
                    md, sp, wp = float(np.median(d)), sign_p(d), wilcoxon_p(d)
                    print(f"       close-time hand to the grasp point, paired {x} - {y}: median {md:+.2f} cm, 95 % CI [{lo:+.2f}, {hi:+.2f}], n {len(d)}, "
                          f"{x} closer in {int((d < 0).sum())}, sign p {sp:.3g}, Wilcoxon p {wp:.3g}; medians {x} "
                          f"{np.nanmedian([close(aa) for aa, _ in P]):.2f} {y} {np.nanmedian([close(bb) for _, bb in P]):.2f} cm")  # fmt: skip
                    ent.update(close_med=md, close_ci=[lo, hi], close_n=len(d), sign_p=sp, wilcoxon_p=wp)
            res["pooled"][f"{name}|{x}v{y}"] = ent

    # C4
    print("C4. failure reasons (hardgrasp's order), jittered episodes; episodes with an executed reach IK error > 3 cm")
    res["reasons"] = {}
    for name, ts in (("all 7 tasks", list(TASKS)), ("decision set", DEC)):
        for g in GROUPS:
            cnt = Counter(get(t, g, e)["reason"] for t in ts for e in EPS)
            ik3 = sum(ik_over(get(t, g, e)) for t in ts for e in EPS)
            res["reasons"][f"{name}|{g}"] = dict(cnt, ik_over3=ik3)
            print(f"  {name:12s} {g} " + " ".join(f"{k} {cnt.get(k, 0)}" for k in REASONS) + f" | reach IK > 3 cm: {ik3}")

    # C5
    print("C5. overshoot past the goal > 3 cm, first reach (pre-grasp) on the far targets (wcan, dumbbell)")
    res["overshoot"] = {}
    for g in GROUPS:
        parts, allv, allik = [], [], []
        for t in FAR:
            ov, ovik = [], []
            for e in EPS:
                w = next((w for w in get(t, g, e).get("reaches", []) if w.get("reach_no") == 1), None)
                if w is None:
                    continue
                o = overshoot_cm(w)
                if np.isnan(o):
                    continue
                ov.append(o)
                if w.get("ik_err_cm", 0) > 3.0:
                    ovik.append(o)
            ov, ovik = np.array(ov), np.array(ovik)
            allv += ov.tolist()
            allik += ovik.tolist()
            parts.append(f"{SHORT[t]} {int((ov > 3).sum())}/{len(ov)} (median {np.median(ov) if len(ov) else float('nan'):+.1f} cm; IK > 3 cm: {int((ovik > 3).sum())}/{len(ovik)})")
        a, ai = np.array(allv), np.array(allik)
        sh = float((a > 3).mean()) if len(a) else float("nan")
        res["overshoot"][g] = dict(n=len(a), over3=int((a > 3).sum()), share=sh, n_ik=len(ai), over3_ik=int((ai > 3).sum()))
        print(f"  {g}: both {int((a > 3).sum())}/{len(a)} ({100 * sh:.0f} %) | " + " | ".join(parts))

    # C6
    print("C6. triggers, jittered episodes (official start apart)")
    res["triggers"] = {}
    for scope, eps in (("jittered", list(EPS)), ("official", [-1])):
        gk = Counter()
        gt = {t: Counter() for t in TASKS}
        rk = Counter()
        rt = {t: Counter() for t in TASKS}
        for t in TASKS:
            for e in eps:
                for w in (get(t, "G", e) or {}).get("reaches", []):
                    gd = w.get("guard")
                    if gd is None:
                        continue
                    kind = "pre-grasp" if w["reach_no"] == 1 else "entry"
                    for key, v in (("inverted", True), ("capped", gd["capped"]), ("fallback", gd["fallback"]),
                                   ("goal already, miss > 3 cm", bool(gd.get("would_fall_back")) and not gd["fallback"])):  # fmt: skip
                        if v:
                            gk[(kind, key)] += 1
                            gt[t][key] += 1
                for w in (get(t, "R", e) or {}).get("reaches", []):
                    if "shrink" not in w:
                        continue
                    kind = "pre-grasp" if w["reach_no"] == 1 else "entry"
                    shs = w["shrink"]
                    n_sh = sum(s["shrunk"] for s in shs)
                    bi = w.get("inv_best", 0)
                    ex = next((s for s in shs if s["it"] == bi), None)
                    for key, v in (("inverted", 1), ("with a shrunk candidate", int(n_sh > 0)), ("shrunk candidates", n_sh),
                                   ("executed command shrunk", int(ex is not None and ex["shrunk"])),
                                   ("shrunk to the goal (alpha 0)", sum(s["shrunk"] and s["alpha"] == 0 for s in shs))):  # fmt: skip
                        rk[(kind, key)] += v
                        rt[t][key] += v
        tot = lambda C, key: sum(v for (kd, k), v in C.items() if k == key)  # noqa: E731
        print(f"  [{scope}] G: inverted {tot(gk, 'inverted')}, capped {tot(gk, 'capped')} (pre-grasp {gk[('pre-grasp', 'capped')]}, entry {gk[('entry', 'capped')]}), "
              f"fallback to the goal {tot(gk, 'fallback')} (pre-grasp {gk[('pre-grasp', 'fallback')]}, entry {gk[('entry', 'fallback')]}), "
              f"choice already the goal with D's miss > 3 cm {tot(gk, 'goal already, miss > 3 cm')}")  # fmt: skip
        if scope == "jittered":
            print("     by task: " + "; ".join(f"{SHORT[t]} cap {gt[t]['capped']} fb {gt[t]['fallback']} (of {gt[t]['inverted']})" for t in TASKS))
        print(f"  [{scope}] R: inverted {tot(rk, 'inverted')}, with a shrunk candidate {tot(rk, 'with a shrunk candidate')} (pre-grasp "
              f"{rk[('pre-grasp', 'with a shrunk candidate')]}, entry {rk[('entry', 'with a shrunk candidate')]}), shrunk candidates "
              f"{tot(rk, 'shrunk candidates')}, executed command shrunk {tot(rk, 'executed command shrunk')}, shrunk to the goal "
              f"{tot(rk, 'shrunk to the goal (alpha 0)')}")  # fmt: skip
        if scope == "jittered":
            print("     by task: " + "; ".join(f"{SHORT[t]} {rt[t]['with a shrunk candidate']}/{rt[t]['inverted']} (executed {rt[t]['executed command shrunk']})" for t in TASKS))
        res["triggers"][scope] = dict(G={f"{a}|{b}": v for (a, b), v in gk.items()}, R={f"{a}|{b}": v for (a, b), v in rk.items()},
                                      G_task={SHORT[t]: dict(gt[t]) for t in TASKS}, R_task={SHORT[t]: dict(rt[t]) for t in TASKS})  # fmt: skip

    # C7
    print("C7. criteria rows (G vs S rows on all 7 tasks; decision set in brackets)")
    P = res["pooled"]
    ga, gs7, gsd, rs7 = P["decision set|GvA"], P["all 7 tasks|GvS"], P["decision set|GvS"], P["all 7 tasks|RvS"]
    g_s, g_sd = gs7["nx"] - gs7["ny"], gsd["nx"] - gsd["ny"]
    wcan_g = res["per_task"]["wcan"]["G"]
    print(f"  G vs A wins on the decision set: {ga['win']} ({ga['dpp']:+.1f} pp, p {ga['p']:.3g})")
    print(f"  G - S = {g_s:+d} [{g_sd:+d}]; wcan G {wcan_g}/30 | 'G >= S + 4 and wcan G >= 6': {g_s >= 4 and wcan_g >= 6} | "
          f"'|G - S| <= 3': {abs(g_s) <= 3} | 'G <= S - 4': {g_s <= -4}")  # fmt: skip
    r4 = res["reasons"]
    ik_r, ik_s, ik_a = r4["all 7 tasks|R"]["ik_over3"], r4["all 7 tasks|S"]["ik_over3"], r4["all 7 tasks|A"]["ik_over3"]
    st_r, st_s, st_a = (r4[f"all 7 tasks|{g}"].get("stance_or_ik", 0) for g in ("R", "S", "A"))
    ov_r = res["overshoot"]["R"]["share"]
    cleared = ik_r <= ik_a
    print(f"  R: reach IK > 3 cm episodes R {ik_r} S {ik_s} A {ik_a} (cleared = R <= A: {cleared}); stance_or_ik R {st_r} S {st_s} A {st_a}; "
          f"overshoot > 3 cm (wcan + dumbbell first reach) {100 * ov_r:.0f} %; R - S = {rs7['nx'] - rs7['ny']:+d} | row "
          f"'cleared, overshoot >= 50 %, R - S < 3': {cleared and ov_r >= 0.5 and rs7['nx'] - rs7['ny'] < 3}")  # fmt: skip
    gc = P["decision set|GvA"]
    if "close_med" in gc:
        lo, hi = gc["close_ci"]
        print(f"  close G - A (decision set): median {gc['close_med']:+.2f} cm [{lo:+.2f}, {hi:+.2f}], sign p {gc['sign_p']:.3g} | "
              f"'< -1 cm and p < 0.05': {gc['close_med'] < -1 and gc['sign_p'] < 0.05} | 'interval contains 0': {lo <= 0 <= hi}"
              f"{'  (the two HEFT rows)' if be == 'heft' else ''}")  # fmt: skip
    print(f"  A and S episode by episode = lmlcmp: {consistent}")
    res["criteria"] = dict(win=ga["win"], g_minus_s=g_s, g_minus_s_dec=g_sd, wcan_g=wcan_g, r_cleared=cleared, r_overshoot=ov_r,
                           r_minus_s=rs7["nx"] - rs7["ny"], consistent=consistent)  # fmt: skip

    # C8
    print("C8. official unjittered start (ep -1)")
    for t in TASKS:
        parts = []
        for g in GROUPS:
            r = get(t, g, -1)
            parts.append(f"{g} {'ok' if r and r['success'] else 'fail'} ({r.get('reason') if r else None}, close {close(r) if r else float('nan'):.1f})")
        print(f"  {SHORT[t]:10s} " + " | ".join(parts))
    if js:
        Path(js).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    main(a.backend, a.json)
