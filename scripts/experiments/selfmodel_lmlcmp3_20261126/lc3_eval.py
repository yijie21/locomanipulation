"""selfmodel_lmlcmp3_20261126's metrics (the entry's 度量 / 预测 / 判据), one controller; read-only (episode records only).
lmlcmp2's lc2_eval.py (ae881c04) rewritten for this round: its statistics, its decision-set rule, its overshoot measure.

    $VPY lc3_eval.py --backend scalebfm [--json OUT.json]          (numpy + scipy)

Groups: A = alone (direct), S = selfmodel (D's fixed point), R = reachable (S's fixed point, every candidate shrunk back to
IK error <= 2 cm), E = entryinv (pre-grasp direct, entry by S's fixed point). Run: ScaleBFM A S R E, SONIC A S R, HEFT A R.
Episodes: the new jittered seeds 30..119 (90 per task), paired by (task, ep). LC_EVAL_EPS=a:b overrides (smoke only).

C1 the decision set (lmlcmp2's rule on the new seeds): per task the share of episodes that failed before the groups split
   -- the look's perception (plan.fail "perception"), the plan made after the look (plan.fail "reach"), no look (a worker
   error without a plan) -- the same in every group (checked); > 50 % -> excluded; A >= 80 % (72 / 90) -> ceiling, out.
C2 per task: successes per group and the discordant pairs of the controller's comparisons (McNemar exact p).
C3 pooled, decision set and all 7 tasks: successes, difference (pp), discordant pairs, McNemar exact p; ScaleBFM's co-primary
   P1 = R vs A and P2 = R vs S with Holm over the two (the smaller p < 0.025, then the other < 0.05; adjusted p printed);
   secondary: ScaleBFM E vs A, E vs S, R vs E; SONIC R vs A, R vs S; HEFT R vs A; S vs A for reference. Close-time hand to
   the planned grasp point (close_dist_cm), paired differences of the same pairs: median, distribution-free 95 % interval,
   how many closer, sign test, Wilcoxon.
C4 failure reasons. The new item "pregrasp_touch" (预抓取时碰到物体): a failed episode, not failed before the split, whose
   pre-grasp segment (the d3 record with reach_no 1) has obj_rows > 0 -- the right hand touched the target object on a
   50 Hz row from the pre-grasp command to its landing label (lr_run's obj_rows: hg_run obj_contacts, simulator truth,
   scoring only). It is checked before hardgrasp's IK / furniture / grasp labels; every other episode keeps hardgrasp's
   label. Table per group (all 7 tasks, decision set), the relabelled episodes' old labels, and hardgrasp's table as it was.
   Also per group: episodes whose pre-grasp touched the object (any outcome) and episodes with an executed reach IK > 3 cm.
C5 overshoot on the far targets (watering can, dumbbell), first reach (pre-grasp, reach_no 1): the hand at the landing row
   (act_h) minus the goal, along the horizontal pelvis-to-goal direction of the command's heading frame; > 3 cm = past.
C6 corrections: the executed |command - goal| (cm), pre-grasp and entry apart, per group and task (median, = 0, > 10 cm);
   R's shrink triggers (lmlcmp2's C6 counts), pre-grasp and entry apart.
C7 mechanism checks (ScaleBFM). M1: reaches without a furniture contact (the segment's furn_rows = 0, not fallen), pre-grasp
   and entry: |landing (act_h) - D's prediction for the executed command| by the command's IK error (<= 2, 2-3, > 3 cm), per
   group; the criterion number is S's > 3 cm median (R's alongside). With the new record: the reference's end (ref_end_h)
   against the command and the landing for the same reaches (descriptive). M2: R - E and E - S successes on the decision
   set. M3: the watering can -- share of episodes with a pre-grasp segment whose pre-grasp touched the can, S and R (A, E
   alongside), and the success rate of the touched episodes (per group and pooled).
C8 the criteria rows and the predictions (ScaleBFM's rows; SONIC / HEFT predicted ranges).
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
OUT = Path(os.environ.get("LC_OUT", str(ROOT / "selfmodel_lmlcmp3_20261126")))
TASKS = ("IThorBathroomLiftCandleFromFloor", "IThorKitchenApproachTableAndLiftApple", "IThorKitchenApproachTableAndLiftBowl",
         "IThorKitchenApproachTableAndLiftSaltShaker", "IThorKitchenTakeStatueFromCabinet",
         "IThorLivingRoomApproachTableAndLiftDumbbell", "IThorLivingRoomApproachTableAndLiftWateringCan")  # lc3_run.TASKS
SHORT = dict(zip(TASKS, ("candle", "apple", "bowl", "salt", "statue_cab", "dumbbell", "wcan")))
NAME = {"A": "alone", "S": "selfmodel", "R": "reachable", "E": "entryinv"}
RUN = {"scalebfm": "ASRE", "sonic": "ASR", "heft": "AR"}
COMPARE = {"scalebfm": (("R", "A"), ("R", "S"), ("E", "A"), ("E", "S"), ("R", "E"), ("S", "A")),
           "sonic": (("R", "A"), ("R", "S"), ("S", "A")), "heft": (("R", "A"),)}  # fmt: skip
REASONS_OLD = ("ok", "perception", "stance_or_ik", "fell_walking", "hit_furniture", "missed", "dropped", "other")
REASONS = ("ok", "perception", "pregrasp_touch", "stance_or_ik", "fell_walking", "hit_furniture", "missed", "dropped", "other")
FAR = ("IThorLivingRoomApproachTableAndLiftWateringCan", "IThorLivingRoomApproachTableAndLiftDumbbell")
WCAN = "IThorLivingRoomApproachTableAndLiftWateringCan"
_lo, _hi = (int(v) for v in os.environ.get("LC_EVAL_EPS", "30:120").split(":"))
EPS = range(_lo, _hi)
NEP = len(EPS)


# -- statistics (lc2_eval's) ------------------------------------------------------------------------------------------
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


def holm2(p1: float, p2: float, alpha: float = 0.05) -> tuple[bool, bool, float, float]:
    """Holm over two p values: (reject 1, reject 2, adjusted p1, adjusted p2)."""
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
def load(base: Path, be: str) -> dict:
    rs = {}
    for p in glob.glob(str(base / be / "episodes" / "*" / "*" / "ep*.json")):
        r = json.loads(Path(p).read_text())
        rs[(r["task"], r["method"], r["ep"])] = r
    return rs


def presplit(r: dict) -> bool:
    """Failed before the groups split: the look's perception, the plan after the look, or no look at all (lc2_eval's)."""
    p = r.get("plan") or {}
    return p.get("fail") in ("perception", "reach") or ("error" in r and not r.get("plan"))


def seg(r: dict, rn: int) -> dict | None:
    return next((d for d in r.get("d3", []) if d["reach_no"] == rn), None)


def touched_pre(r: dict) -> bool | None:
    """The pre-grasp segment's hand-object contact rows > 0; None without a pre-grasp segment."""
    d = seg(r, 1)
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


def ik_over(r: dict) -> bool:
    return any(w.get("ik_err_cm", 0) > 3.0 for w in r.get("reaches", []))


def overshoot_cm(w: dict) -> float:
    if "act_h" not in w:
        return float("nan")
    g, a = np.asarray(w["goal_h"], float), np.asarray(w["act_h"], float)
    u = g[:2] / max(np.linalg.norm(g[:2]), 1e-9)
    return 100 * float((a[:2] - g[:2]) @ u)


def corr_cm(w: dict) -> float:
    return 100 * float(np.linalg.norm(np.asarray(w["cmd_h"], float) - np.asarray(w["goal_h"], float)))


def main(be: str, js: str | None) -> None:
    rs = load(OUT, be)
    G = RUN[be]
    res: dict = dict(backend=be, groups=G, eps=[EPS.start, EPS.stop])
    n_by = Counter(m for (_, m, _) in rs)
    miss = [(SHORT[t], g, e) for t in TASKS for g in G for e in EPS if (t, NAME[g], e) not in rs]
    print(f"== {be}: {len(rs)} episode records under {OUT / be / 'episodes'} {dict(n_by)}; episodes {EPS.start}..{EPS.stop - 1}; groups "
          f"{' '.join(G)}; missing {len(miss)} {miss[:6]}; worker errors {sum('error' in r for r in rs.values())} "
          f"({sorted({(SHORT[t], e) for (t, m, e), r in rs.items() if 'error' in r})[:12]})")  # fmt: skip
    get = lambda t, g, e: rs.get((t, NAME[g], e))  # noqa: E731
    have = lambda t, e: all(get(t, g, e) is not None for g in G)  # noqa: E731
    EP = {t: [e for e in EPS if have(t, e)] for t in TASKS}

    # C1
    print(f"C1. decision set: failures before the groups split (perception, the plan after the look, no look) > 50 % -> out; "
          f"A >= 80 % ({int(np.ceil(0.8 * NEP))}/{NEP}) -> ceiling")  # fmt: skip
    per = {}
    for t in TASKS:
        sets = {g: tuple(presplit(get(t, g, e)) for e in EP[t]) for g in G}
        same = len(set(sets.values())) == 1
        pre = sets["A"]
        share = sum(pre) / max(1, len(pre))
        nA = sum(bool(get(t, "A", e)["success"]) for e in EP[t])
        excl, top = share > 0.5 or not pre, nA >= 0.8 * NEP  # a task without episodes cannot be in the set
        per[t] = dict(share=share, excl=excl, top=top, same=same, n_pre=sum(pre), n=len(pre), A=nA)
        flag = f"EXCLUDED (pre-split {100 * share:.0f} %)" if excl else f"AT CEILING (A {nA}/{len(pre)})" if top else "decision set"
        kinds = Counter(("perception" if (get(t, "A", e).get("plan") or {}).get("fail") == "perception" else
                         "plan after look" if (get(t, "A", e).get("plan") or {}).get("fail") == "reach" else "no look")
                        for e in EP[t] if presplit(get(t, "A", e)))  # fmt: skip
        print(f"  {SHORT[t]:10s} pre-split failures {sum(pre)}/{len(pre)} ({100 * share:.0f} %) {dict(kinds)}; same in all groups {same} | "
              f"A {nA}/{len(pre)} | {flag}")  # fmt: skip
    DEC = [t for t in TASKS if not per[t]["excl"] and not per[t]["top"]]
    res["decision_set"] = [SHORT[t] for t in DEC]
    res["c1"] = {SHORT[t]: v for t, v in per.items()}
    print(f"  decision set: {[SHORT[t] for t in DEC]}; pre-split the same in every group for every task: {all(v['same'] for v in per.values())}")

    # C2
    print(f"C2. per task (episodes {EPS.start}..{EPS.stop - 1}): successes and discordant pairs (x-only / y-only, McNemar exact p)")
    res["per_task"] = {}
    for t in TASKS:
        n = {g: sum(bool(get(t, g, e)["success"]) for e in EP[t]) for g in G}
        parts = []
        for x, y in COMPARE[be]:
            b = sum(bool(get(t, y, e)["success"]) and not get(t, x, e)["success"] for e in EP[t])
            c = sum(bool(get(t, x, e)["success"]) and not get(t, y, e)["success"] for e in EP[t])
            parts.append(f"{x}v{y} {y}-only {b} {x}-only {c} p {mcnemar(b, c):.3g}")
        res["per_task"][SHORT[t]] = n
        print(f"  {SHORT[t]:10s} n {len(EP[t])} " + " ".join(f"{g} {n[g]:2d}" for g in G) + " | " + " | ".join(parts))

    # C3
    print("C3. pooled")
    res["pooled"] = {}
    for name, ts in (("decision set", DEC), ("all 7 tasks", list(TASKS))):
        print(f"  {name} ({', '.join(SHORT[t] for t in ts)}): {sum(len(EP[t]) for t in ts)} pairs")
        for x, y in COMPARE[be]:
            P = [(get(t, x, e), get(t, y, e)) for t in ts for e in EP[t]]
            nx, ny = sum(bool(a["success"]) for a, _ in P), sum(bool(b["success"]) for _, b in P)
            b = sum(bool(bb["success"]) and not aa["success"] for aa, bb in P)
            c = sum(bool(aa["success"]) and not bb["success"] for aa, bb in P)
            p = mcnemar(b, c)
            dpp = 100 * (nx - ny) / max(1, len(P))
            print(f"    {x} vs {y}: success {x} {nx} ({100 * nx / max(1, len(P)):.1f} %) {y} {ny} ({100 * ny / max(1, len(P)):.1f} %) diff "
                  f"{dpp:+.1f} pp ({nx - ny:+d}) | {y}-only {b} {x}-only {c} McNemar exact p {p:.3g}")  # fmt: skip
            ent = dict(nx=nx, ny=ny, b=b, c=c, p=p, dpp=dpp, n=len(P))
            d = np.array([close(aa) - close(bb) for aa, bb in P if not np.isnan(close(aa)) and not np.isnan(close(bb))])
            if len(d):
                lo, hi = median_ci(d)
                md, sp, wp = float(np.median(d)), sign_p(d), wilcoxon_p(d)
                print(f"       close-time hand to the grasp point, paired {x} - {y}: median {md:+.2f} cm, 95 % CI [{lo:+.2f}, {hi:+.2f}], n {len(d)}, "
                      f"{x} closer in {int((d < 0).sum())}, {y} closer in {int((d > 0).sum())}, sign p {sp:.3g}, Wilcoxon p {wp:.3g}; medians {x} "
                      f"{np.nanmedian([close(aa) for aa, _ in P]):.2f} {y} {np.nanmedian([close(bb) for _, bb in P]):.2f} cm")  # fmt: skip
                ent.update(close_med=md, close_ci=[lo, hi], close_n=len(d), sign_p=sp, wilcoxon_p=wp)
            res["pooled"][f"{name}|{x}v{y}"] = ent
        if be == "scalebfm":
            P1, P2 = res["pooled"][f"{name}|RvA"], res["pooled"][f"{name}|RvS"]
            r1, r2, a1, a2 = holm2(P1["p"], P2["p"])
            res["pooled"][f"{name}|holm"] = dict(rej1=r1, rej2=r2, adj1=a1, adj2=a2)
            print(f"    Holm over P1 (R vs A) and P2 (R vs S): raw p {P1['p']:.3g} / {P2['p']:.3g}, adjusted {a1:.3g} / {a2:.3g}, "
                  f"significant {r1} / {r2}" + (" | P1 (>= 10 pp and significant): " + str(P1["dpp"] >= 10 and r1) +
                                                 f", P2 (R > S and significant): {P2['nx'] > P2['ny'] and r2}" if name == "decision set" else ""))  # fmt: skip

    # C4
    print("C4. failure reasons: the new item pregrasp_touch (the pre-grasp segment's hand-object contact rows > 0, checked before "
          "hardgrasp's IK / furniture / grasp labels); then hardgrasp's table as it was")  # fmt: skip
    res["reasons"] = {}
    for name, ts in (("all 7 tasks", list(TASKS)), ("decision set", DEC)):
        for g in G:
            rr = [get(t, g, e) for t in ts for e in EP[t]]
            cnt = Counter(reason3(r) for r in rr)
            moved = Counter(r["reason"] for r in rr if reason3(r) == "pregrasp_touch")
            tch = sum(bool(touched_pre(r)) for r in rr)
            npre = sum(touched_pre(r) is not None for r in rr)
            ik3 = sum(ik_over(r) for r in rr)
            res["reasons"][f"{name}|{g}"] = dict(cnt, relabelled_from=dict(moved), touched_pre=tch, with_pre=npre, ik_over3=ik3)
            print(f"  {name:12s} {g} " + " ".join(f"{k} {cnt.get(k, 0)}" for k in REASONS) +
                  f" | pregrasp_touch was {dict(moved)} | pre-grasp touched the object (any outcome) {tch}/{npre} | reach IK > 3 cm: {ik3}")  # fmt: skip
    for name, ts in (("all 7 tasks", list(TASKS)), ("decision set", DEC)):
        for g in G:
            cnt = Counter(get(t, g, e)["reason"] for t in ts for e in EP[t])
            res["reasons"][f"{name}|{g}|old"] = dict(cnt)
            print(f"  [hardgrasp's order] {name:12s} {g} " + " ".join(f"{k} {cnt.get(k, 0)}" for k in REASONS_OLD))

    # C5
    print("C5. overshoot past the goal > 3 cm, first reach (pre-grasp) on the far targets (wcan, dumbbell)")
    res["overshoot"] = {}
    for g in G:
        parts, allv, allik = [], [], []
        for t in FAR:
            ov, ovik = [], []
            for e in EP[t]:
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
            parts.append(f"{SHORT[t]} {int((ov > 3).sum())}/{len(ov)} (median {med(ov):+.1f} cm; IK > 3 cm: {int((ovik > 3).sum())}/{len(ovik)})")
        a, ai = np.array(allv), np.array(allik)
        sh = float((a > 3).mean()) if len(a) else float("nan")
        res["overshoot"][g] = dict(n=len(a), over3=int((a > 3).sum()), share=sh, n_ik=len(ai), over3_ik=int((ai > 3).sum()))
        print(f"  {g}: both {int((a > 3).sum())}/{len(a)} ({100 * sh:.0f} %) | " + " | ".join(parts))

    # C6
    print("C6. executed corrections |command - goal| (cm), pre-grasp (reach_no 1) and entry (reach_no 2) apart: median / = 0 / > 10 cm / n")
    res["corrections"] = {}
    for g in [g for g in G if g != "A"]:
        for rn, kind in ((1, "pre-grasp"), (2, "entry")):
            parts = []
            allc = []
            for t in TASKS:
                cs = [corr_cm(w) for e in EP[t] for w in get(t, g, e).get("reaches", []) if w.get("reach_no") == rn and "cmd_h" in w]
                allc += cs
                cs = np.array(cs)
                parts.append(f"{SHORT[t]} {med(cs):.1f}/{int((cs == 0).sum())}/{int((cs > 10).sum())}/{len(cs)}")
                res["corrections"][f"{g}|{kind}|{SHORT[t]}"] = dict(median=med(cs), zero=int((cs == 0).sum()), over10=int((cs > 10).sum()), n=len(cs))
            ac = np.array(allc)
            res["corrections"][f"{g}|{kind}|all"] = dict(median=med(ac), zero=int((ac == 0).sum()), over10=int((ac > 10).sum()), n=len(ac))
            print(f"  {g} {kind:9s} all {med(ac):.1f}/{int((ac == 0).sum())}/{int((ac > 10).sum())}/{len(ac)} | " + " ".join(parts))
    if "R" in G:
        rk, rt = Counter(), {t: Counter() for t in TASKS}
        for t in TASKS:
            for e in EP[t]:
                for w in get(t, "R", e).get("reaches", []):
                    if "shrink" not in w:
                        continue
                    kind = "pre-grasp" if w["reach_no"] == 1 else "entry"
                    shs = w["shrink"]
                    n_sh = sum(s["shrunk"] for s in shs)
                    ex = next((s for s in shs if s["it"] == w.get("inv_best", 0)), None)
                    for key, v in (("inverted", 1), ("with a shrunk candidate", int(n_sh > 0)), ("shrunk candidates", n_sh),
                                   ("executed command shrunk", int(ex is not None and ex["shrunk"])),
                                   ("shrunk to the goal (alpha 0)", sum(s["shrunk"] and s["alpha"] == 0 for s in shs))):  # fmt: skip
                        rk[(kind, key)] += v
                        rt[t][(kind, key)] += v
        keys = ("inverted", "with a shrunk candidate", "executed command shrunk", "shrunk candidates", "shrunk to the goal (alpha 0)")
        for kind in ("pre-grasp", "entry"):
            print(f"  R shrink, {kind}: " + ", ".join(f"{k} {rk[(kind, k)]}" for k in keys) + " | by task (with a shrunk candidate / executed "
                  "shrunk / inverted): " + "; ".join(f"{SHORT[t]} {rt[t][(kind, 'with a shrunk candidate')]}/{rt[t][(kind, 'executed command shrunk')]}/"
                                                       f"{rt[t][(kind, 'inverted')]}" for t in TASKS))  # fmt: skip
        res["shrink"] = {f"{a}|{b}": v for (a, b), v in rk.items()}

    # C7 (ScaleBFM's mechanism checks; printed for every controller, the criteria are ScaleBFM's)
    print("C7. mechanism checks")
    bins = (("IK <= 2", 0.0, 2.0), ("IK 2-3", 2.0, 3.0), ("IK > 3", 3.0, np.inf))
    m1 = {}
    for scope, ts in (("all 7 tasks", list(TASKS)), ("decision set", DEC)):
        for g in G:
            acc = {b[0]: dict(pe=[], ra=[], rc=[], ac=[]) for b in bins}
            for t in ts:
                for e in EP[t]:
                    r = get(t, g, e)
                    for w in r.get("reaches", []):
                        if w.get("reach_no") not in (1, 2) or "act_h" not in w or "pred_h" not in w:
                            continue
                        d = seg(r, w["reach_no"])
                        if d is None or d["furn_rows"] > 0 or d.get("seg_fell"):
                            continue
                        ik = w["ik_err_cm"]
                        bn = next(b[0] for b in bins if b[1] < ik <= b[2] or (b[1] == 0.0 and ik <= b[2]))
                        a_, D_, c_ = np.asarray(w["act_h"]), np.asarray(w["pred_h"]["D"]), np.asarray(w["cmd_h"])
                        acc[bn]["pe"].append(100 * float(np.linalg.norm(a_ - D_)))
                        acc[bn]["ac"].append(100 * float(np.linalg.norm(a_ - c_)))
                        if "ref_end_h" in w:
                            re_ = np.asarray(w["ref_end_h"])
                            acc[bn]["ra"].append(100 * float(np.linalg.norm(a_ - re_)))
                            acc[bn]["rc"].append(100 * float(np.linalg.norm(re_ - c_)))
            m1[f"{scope}|{g}"] = {b: dict(n=len(v["pe"]), pred_err_med=med(v["pe"]), land_vs_ref_med=med(v["ra"]), ref_vs_cmd_med=med(v["rc"]),
                                          land_vs_cmd_med=med(v["ac"])) for b, v in acc.items()}  # fmt: skip
            print(f"  M1 [{scope}] {g}: |landing - D| median (n) " + "; ".join(f"{b} {med(v['pe']):.2f} cm ({len(v['pe'])})" for b, v in acc.items())
                  + " | IK > 3: |landing - ref end| " + f"{med(acc['IK > 3']['ra']):.2f}, |ref end - command| {med(acc['IK > 3']['rc']):.2f}, "
                  f"|landing - command| {med(acc['IK > 3']['ac']):.2f} cm")  # fmt: skip
    res["m1"] = m1
    pd_ = res["pooled"]
    if all(f"decision set|{x}v{y}" in pd_ for x, y in (("R", "E"), ("E", "S"))):
        re_, es_ = pd_["decision set|RvE"], pd_["decision set|EvS"]
        res["m2"] = dict(r_minus_e=re_["nx"] - re_["ny"], e_minus_s=es_["nx"] - es_["ny"])
        print(f"  M2 (decision set): R - E = {re_['nx'] - re_['ny']:+d} (R {re_['nx']}, E {re_['ny']}, p {re_['p']:.3g}); E - S = "
              f"{es_['nx'] - es_['ny']:+d} (E {es_['nx']}, S {es_['ny']}, p {es_['p']:.3g})")  # fmt: skip
    m3 = {}
    parts = []
    tpool = []
    for g in G:
        rr = [get(WCAN, g, e) for e in EP[WCAN]]
        wp = [r for r in rr if touched_pre(r) is not None]
        tt = [r for r in wp if touched_pre(r)]
        tpool += tt
        ns = sum(bool(r["success"]) for r in tt)
        m3[g] = dict(with_pre=len(wp), touched=len(tt), share=len(tt) / max(1, len(wp)), touched_success=ns)
        parts.append(f"{g} touched {len(tt)}/{len(wp)} ({100 * len(tt) / max(1, len(wp)):.0f} %), of them ok {ns}")
    res["m3"] = dict(m3, pooled_touched=len(tpool), pooled_touched_success=sum(bool(r["success"]) for r in tpool))
    print(f"  M3 (wcan, episodes with a pre-grasp segment): " + "; ".join(parts) +
          f" | all groups: touched {len(tpool)}, ok {sum(bool(r['success']) for r in tpool)} "
          f"({100 * sum(bool(r['success']) for r in tpool) / max(1, len(tpool)):.0f} %)")  # fmt: skip

    # C8
    print("C8. criteria rows and predictions (decision set unless said)")
    dec5 = [SHORT[t] for t in DEC] == ["candle", "apple", "statue_cab", "dumbbell", "wcan"]
    print(f"  decision set = candle, apple, statue_cab, dumbbell, wcan (predicted, ~80 %): {dec5}; bowl / salt pre-split > 50 %: "
          f"{per[TASKS[2]]['excl']} / {per[TASKS[3]]['excl']}")  # fmt: skip
    crit: dict = dict(decision_set_as_predicted=dec5)
    rate = lambda g: 100 * sum(bool(get(t, g, e)["success"]) for t in DEC for e in EP[t]) / max(1, sum(len(EP[t]) for t in DEC))  # noqa: E731
    inr = lambda v, lo, hi: lo <= v <= hi  # noqa: E731
    if be == "scalebfm":
        P1, P2, H = pd_["decision set|RvA"], pd_["decision set|RvS"], pd_["decision set|holm"]
        p1 = P1["dpp"] >= 10 and H["rej1"]
        p2 = P2["nx"] > P2["ny"] and H["rej2"]
        row3 = (not p1) and P1["dpp"] >= 5 and P1["p"] < 0.05
        row4 = P1["dpp"] < 5 or P1["p"] >= 0.05
        wc = res["per_task"]["wcan"]
        r_s_dec = P2["nx"] - P2["ny"]
        overturn2 = r_s_dec <= 0 or wc["R"] - wc["S"] <= 2
        s_ik3 = m1["all 7 tasks|S"]["IK > 3"]
        m1_row = s_ik3["n"] > 0 and s_ik3["pred_err_med"] <= 5
        r_e, e_s = res["m2"]["r_minus_e"], res["m2"]["e_minus_s"]
        print(f"  P1 {p1} (R - A {P1['dpp']:+.1f} pp, raw p {P1['p']:.3g}, Holm-adjusted {H['adj1']:.3g}); P2 {p2} (R - S {P2['dpp']:+.1f} pp / "
              f"{r_s_dec:+d}, raw p {P2['p']:.3g}, Holm-adjusted {H['adj2']:.3g})")  # fmt: skip
        print(f"  row 'P1 and P2': {p1 and p2} | 'only P1': {p1 and not p2} | 'not P1, R - A >= 5 pp and p < 0.05': {row3} | "
              f"'R - A < 5 pp or p >= 0.05': {row4}")  # fmt: skip
        print(f"  row 'decision set R - S <= 0, or wcan R - S <= 2': {overturn2} (decision set {r_s_dec:+d}; wcan R {wc['R']} S {wc['S']}, {wc['R'] - wc['S']:+d})")
        print(f"  row 'M1: S's out-of-reach commands' median prediction error <= 5 cm': {m1_row} (S IK > 3 cm: {s_ik3['pred_err_med']:.2f} cm, n {s_ik3['n']}; "
              f"R IK > 3 cm: {m1['all 7 tasks|R']['IK > 3']['pred_err_med']:.2f} cm, n {m1['all 7 tasks|R']['IK > 3']['n']})")  # fmt: skip
        print(f"  row 'M2: R - E >= 4': {r_e >= 4} | 'M2: |R - E| <= 3 and E - S >= 4': {abs(r_e) <= 3 and e_s >= 4} (R - E {r_e:+d}, E - S {e_s:+d})")
        rates = {g: rate(g) for g in G}
        pr = dict(A=(16, 24), S=(23, 31), R=(26, 35), E=(23, 33))
        print("  predicted success rates: " + "; ".join(f"{g} {rates[g]:.1f} % ({pr[g][0]}-{pr[g][1]}: {inr(rates[g], *pr[g])})" for g in G))
        print(f"  predicted R - A +6..+13 pp: {P1['dpp']:+.1f} ({inr(P1['dpp'], 6, 13)}); R - S +1..+6 pp: {P2['dpp']:+.1f} ({inr(P2['dpp'], 1, 6)})")
        pw = dict(A=(6, 14), S=(3, 12), R=(14, 28), E=(10, 24))
        print("  predicted wcan successes: " + "; ".join(f"{g} {wc[g]} ({pw[g][0]}-{pw[g][1]}: {inr(wc[g], *pw[g])})" for g in G))
        m3s, m3r = res["m3"]["S"], res["m3"]["R"]
        pt = res["m3"]["pooled_touched"]
        tsr = 100 * res["m3"]["pooled_touched_success"] / max(1, pt)
        print(f"  predicted M1: S IK > 3 median 12-25 cm: {s_ik3['pred_err_med']:.2f} ({inr(s_ik3['pred_err_med'], 12, 25)}); R IK > 3 <= 3 cm: "
              f"{m1['all 7 tasks|R']['IK > 3']['pred_err_med']:.2f} | M2: R - E -2..+6: {r_e:+d} ({inr(r_e, -2, 6)}), E - S 0..+8: {e_s:+d} ({inr(e_s, 0, 8)}) | "
              f"M3: S touched 40-65 %: {100 * m3s['share']:.0f} ({inr(100 * m3s['share'], 40, 65)}), R 3-15 %: {100 * m3r['share']:.0f} "
              f"({inr(100 * m3r['share'], 3, 15)}), touched success <= 10 %: {tsr:.0f} % ({tsr <= 10})")  # fmt: skip
        crit.update(p1=p1, p2=p2, row3=row3, row4=row4, overturn_r_better_s=overturn2, m1_row=m1_row, m2_r_e=r_e, m2_e_s=e_s, rates=rates)
    elif be == "sonic":
        P1, P2 = pd_["decision set|RvA"], pd_["decision set|RvS"]
        print(f"  predicted R - A +3..+9 pp: {P1['dpp']:+.1f} ({inr(P1['dpp'], 3, 9)}); R - S -2..+3 pp: {P2['dpp']:+.1f} ({inr(P2['dpp'], -2, 3)})")
        crit.update(r_a=P1["dpp"], r_s=P2["dpp"])
    else:
        P1 = pd_["decision set|RvA"]
        print(f"  predicted R - A 0..+4 pp: {P1['dpp']:+.1f} ({inr(P1['dpp'], 0, 4)})")
        crit.update(r_a=P1["dpp"])
    res["criteria"] = crit
    if js:
        Path(js).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    main(a.backend, a.json)
