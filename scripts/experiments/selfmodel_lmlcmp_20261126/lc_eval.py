"""selfmodel_lmlcmp_20261126's metrics (the entry's 度量 / 预测 / 判据), one controller; read-only (records, data3, models).

    cd $W && REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 $VPY lc_eval.py --backend scalebfm [--json OUT.json]

A. Episodes (episodes/<task>/<method>/ep*.json), jittered episodes 0..29 paired by (task, ep); the official unjittered
   start (-1) listed apart.
   A1 per task: pairs, successes (the task's own judgement), A-only / S-only, McNemar exact p, close-time hand to target
      (R close_dist_cm: the hand point to the planned grasp point when the close goes) paired S - A median; exclusion
      (hardgrasp's): perception + stance_or_ik over both groups > 50 % -> excluded; direct >= 80 % -> at ceiling; both
      out of the decision set.
   A2 pooled on the decision set and on all 7 tasks: successes, difference (pp), discordant pairs, McNemar exact p; the
      close-time paired difference: median, distribution-free 95 % interval, sign test, Wilcoxon; the criteria's "win"
      (>= 10 pp and p < 0.05 on the decision set) and "hand clearly more accurate" (median < -1 cm and sign p < 0.05).
   A3 failure reasons per group (hardgrasp's order); stance_or_ik S - A (criterion: >= 10 -> over-correction).
   A4 settle: per command kind (pre-grasp = 第一下, entry = 第二下, via, close, lift): the own rule held when it went (vs
      3 s timeout), the wait (median, p90), data3's truth rule at the command row; the two rules' agreement at the
      command row and over every step of every wait (both quiet or both not); the lag of the own rule behind the truth
      rule's first quiet step.
   A5 walking: lost / fell during a walk, per group (HEFT's registered count: lmlreal's 61 direct episodes).
   A6 the inversion (selfmodel group): D's predicted miss at its chosen command, the correction |command - goal|, the
      iterate that won, fallbacks, candidates that could not be planned.
B. Offline, the d3 records (data/all.npz, data.py's derive; T3 reads chain clipped to 1): kept = not fell / lost and a
   settled end label. Errors: each model's predicted final hand point to the settled one (heading frame of the command
   row), median and p90; x val = median / the model's data3 val median (T3 3.08 / 3.99 / 4.25 registered; M0, D0 from
   train3b's eval_val "data3 val" row; recomputed here as a check). Primary group for the model metrics: the direct
   group (its commands do not depend on any model; train3b's real-episode D/M 1.02 was on direct records); the
   selfmodel group and both together reported next to it. Cells: 第一下 (reach_no 1) / 第二下 (2), each all / settled
   start (data3's truth rule at the command row, lmlreal's cut) / not settled; settled start by the own rule as well;
   contact with the scene during the reach (furn_rows / win_rows: none / light < 50 % / stuck >= 50 %); start in data3's
   training band (0.5-99.5 % of the frac-1.0 training split's command rows): the 29 joints, the acting hand's speed
   (|W hand point k0 - k0-1| / 0.1 s), both. D/M (ratio of medians), 95 % intervals from 1000 resamples of episodes.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np
import torch

REPO = Path(os.environ.get("LOCO_REPO", "/wuji-vefps-D/wuji-il/yijie/ws/code/loco"))
sys.path[:0] = [str(REPO / "scripts/experiments/reachwm_20261030"), str(REPO / "scripts/experiments/selfwm_20261029")]
for k in ("REACHWM_GOAL_REF", "REACHWM_APPROACH", "REACHWM_QFIX"):
    assert os.environ.get(k) == "1", f"{k}=1 must be set before data.py is imported"
for k in ("REACHWM_X0_REF", "REACHWM_FDVEL"):
    assert os.environ.get(k, "0") == "0", f"{k} must be off"
import data as DT  # noqa: E402
import train as TR  # noqa: E402

M = TR.M
NB = 1000
ROOT = Path("/wuji-vefps-D/wuji-il/yijie/loco_outputs")
OUT = Path(os.environ.get("LC_OUT", str(ROOT / "selfmodel_lmlcmp_20261126")))
TASKS = ("IThorBathroomLiftCandleFromFloor", "IThorKitchenApproachTableAndLiftApple", "IThorKitchenApproachTableAndLiftBowl",
         "IThorKitchenApproachTableAndLiftSaltShaker", "IThorKitchenTakeStatueFromCabinet",
         "IThorLivingRoomApproachTableAndLiftDumbbell", "IThorLivingRoomApproachTableAndLiftWateringCan")  # lc_run.TASKS
SHORT = dict(zip(TASKS, ("candle", "apple", "bowl", "salt", "statue_cab", "dumbbell", "wcan")))
METHODS = ("alone", "selfmodel")
REASONS = ("ok", "perception", "stance_or_ik", "fell_walking", "hit_furniture", "missed", "dropped", "other")
MODELS = ("T3", "M", "D")
VAL_REF = {"scalebfm": (3.08, 2.64, 2.25), "heft": (3.99, 3.51, 2.94), "sonic": (4.25, 3.72, 3.06)}  # T3 (entry), M0, D0 (train3b eval_val)
CHAIN_COL, T3_CHAIN_MAX = 205, 1
LMLREAL_HEFT_LOST = 61


def roots(be: str) -> dict:
    return {"T3": ROOT / "selfmodel_data3_20261125" / be / "models_s0", "M": ROOT / "selfmodel_train3b_20261125" / be / "models_M_s0",
            "D": ROOT / "selfmodel_train3b_20261125" / be / "models_D_s0"}  # fmt: skip


# -- statistics (hg_analyze's) ------------------------------------------------------------------------------------------
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


def q(a, p) -> float:
    a = np.asarray(a, float)
    return float(np.percentile(a, p)) if len(a) else float("nan")


# -- A. episodes ----------------------------------------------------------------------------------------------------
def load_eps(be: str) -> dict:
    rs = {}
    for p in glob.glob(str(OUT / be / "episodes" / "*" / "*" / "ep*.json")):
        r = json.loads(Path(p).read_text())
        rs[(r["task"], r["method"], r["ep"])] = r
    return rs


def reach(r: dict, rn: int) -> dict:
    return next((w for w in r.get("reaches", []) if w.get("reach_no") == rn), {})


def part_a(be: str, res: dict) -> None:
    rs = load_eps(be)
    eps = range(0, 30)
    print(f"== A. {be}: {len(rs)} episode records under {OUT / be / 'episodes'}; worker errors "
          f"{sum('error' in r for r in rs.values())} {[k for k, r in rs.items() if 'error' in r][:4]}")  # fmt: skip
    per = {}
    print("A1. per task (A = alone / direct, S = selfmodel; pairs = jittered episodes 0..29 present in both groups)")
    for t in TASKS:
        E = [e for e in eps if (t, "alone", e) in rs and (t, "selfmodel", e) in rs]
        A = [rs[(t, "alone", e)] for e in E]
        S = [rs[(t, "selfmodel", e)] for e in E]
        b = sum(a["success"] and not s["success"] for a, s in zip(A, S))
        c = sum(s["success"] and not a["success"] for a, s in zip(A, S))
        nA, nS = sum(g["success"] for g in A), sum(g["success"] for g in S)
        bad = sum(g["reason"] in ("perception", "stance_or_ik") for g in A + S) / max(1, 2 * len(E))
        top = nA >= 0.8 * len(E)
        excl = bad > 0.5
        d = np.array([s["close_dist_cm"] - a["close_dist_cm"] for a, s in zip(A, S) if "close_dist_cm" in a and "close_dist_cm" in s])
        per[t] = dict(E=E, A=A, S=S, b=b, c=c, nA=nA, nS=nS, d=d, excl=excl, top=top, bad=bad)
        ci = median_ci(d)
        flag = ("EXCLUDED (perception+stance %.0f %%)" % (100 * bad)) if excl else ("AT CEILING (direct %d/%d)" % (nA, len(E))) if top else "decision set"
        print(f"  {SHORT[t]:10s} pairs {len(E):2d} | success A {nA:2d} S {nS:2d} | A-only {b:2d} S-only {c:2d} McNemar p {mcnemar(b, c):.3g} | "
              f"close S-A median {np.median(d) if len(d) else float('nan'):+.2f} cm [{ci[0]:+.2f}, {ci[1]:+.2f}] n {len(d)} | "
              f"perception+stance {100 * bad:.0f} % | {flag}")  # fmt: skip
        res.setdefault("per_task", {})[SHORT[t]] = dict(pairs=len(E), nA=nA, nS=nS, b=b, c=c, p=mcnemar(b, c), excl=excl, top=top,
                                                         bad=bad, close_med=float(np.median(d)) if len(d) else None, n_close=len(d))  # fmt: skip
    print("A2. pooled")
    for name, ts in (("decision set", [t for t in TASKS if not per[t]["excl"] and not per[t]["top"]]), ("all 7 tasks", list(TASKS))):
        n = sum(len(per[t]["E"]) for t in ts)
        nA, nS = sum(per[t]["nA"] for t in ts), sum(per[t]["nS"] for t in ts)
        b, c = sum(per[t]["b"] for t in ts), sum(per[t]["c"] for t in ts)
        d = np.concatenate([per[t]["d"] for t in ts]) if ts else np.zeros(0)
        ci = median_ci(d)
        pp = 100 * (nS - nA) / max(n, 1)
        p = mcnemar(b, c)
        win = pp >= 10 and p < 0.05
        sp = sign_p(d)
        acc = len(d) > 0 and float(np.median(d)) < -1 and sp < 0.05
        print(f"  {name} ({', '.join(SHORT[t] for t in ts)}): pairs {n} | success A {nA} ({100 * nA / max(n, 1):.1f} %) S {nS} "
              f"({100 * nS / max(n, 1):.1f} %) diff {pp:+.1f} pp | A-only {b} S-only {c} McNemar exact p {p:.3g} | WIN {win}")  # fmt: skip
        print(f"     close-time hand to target, paired S - A: median {np.median(d) if len(d) else float('nan'):+.2f} cm, 95 % CI "
              f"[{ci[0]:+.2f}, {ci[1]:+.2f}], n {len(d)}, S closer in {int((d < 0).sum())}, sign p {sp:.3g}, Wilcoxon p {wilcoxon_p(d):.3g} | "
              f"clearly more accurate (median < -1 cm, sign p < 0.05) {acc}; medians A {q([r['close_dist_cm'] for t in ts for r in per[t]['A'] if 'close_dist_cm' in r], 50):.2f} "
              f"S {q([r['close_dist_cm'] for t in ts for r in per[t]['S'] if 'close_dist_cm' in r], 50):.2f} cm")  # fmt: skip
        res.setdefault("pooled", {})[name] = dict(tasks=[SHORT[t] for t in ts], pairs=n, nA=nA, nS=nS, diff_pp=pp, b=b, c=c, p=p, win=win,
                                                  close_med=float(np.median(d)) if len(d) else None, close_ci=ci, close_n=len(d),
                                                  sign_p=sp, wilcoxon_p=wilcoxon_p(d), more_accurate=acc)  # fmt: skip
    print("A3. failure reasons (hardgrasp's order), jittered episodes")
    for name, ts in (("all 7 tasks", list(TASKS)), ("decision set", [t for t in TASKS if not per[t]["excl"] and not per[t]["top"]])):
        for m, key in (("alone", "A"), ("selfmodel", "S")):
            cnt = Counter(g["reason"] for t in ts for g in per[t][key])
            print(f"  {name:12s} {m:9s} " + " ".join(f"{k} {cnt.get(k, 0)}" for k in REASONS))
            res.setdefault("reasons", {}).setdefault(name, {})[m] = dict(cnt)
    si = {m: sum(g["reason"] == "stance_or_ik" for t in TASKS for g in per[t][key]) for m, key in (("alone", "A"), ("selfmodel", "S"))}
    why = {m: Counter(str(g.get("why", ""))[:40] for t in TASKS for g in per[t][key] if g["reason"] == "stance_or_ik") for m, key in (("alone", "A"), ("selfmodel", "S"))}
    print(f"  stance_or_ik: A {si['alone']} S {si['selfmodel']} (S - A {si['selfmodel'] - si['alone']:+d}; criterion >= +10 -> over-correction "
          f"{si['selfmodel'] - si['alone'] >= 10}); why: A {dict(why['alone'])} S {dict(why['selfmodel'])}")  # fmt: skip
    ik_big = {m: sum(any(w.get("ik_err_cm", 0) > 3 for w in g.get("reaches", [])) for t in TASKS for g in per[t][key]) for m, key in (("alone", "A"), ("selfmodel", "S"))}
    print(f"     episodes with a reach IK error > 3 cm: A {ik_big['alone']} S {ik_big['selfmodel']}")
    res["stance_or_ik"] = dict(si, why={m: dict(v) for m, v in why.items()}, ik_big=ik_big)
    # A4 settle
    print("A4. settle waits (jittered episodes, both groups; own rule = hands in the pelvis frame < 2 cm/s and legs < 0.05 rad/s for 0.5 s)")
    kinds = ((1, "pre-grasp (第一下)"), (2, "entry (第二下)"), (0, "via"), (-1, "close"), (-2, "lift"))
    allw = {m: [w for t in TASKS for g in per[t]["A" if m == "alone" else "S"] for w in g.get("waits", [])] for m in METHODS}
    for m in METHODS + ("both",):
        W = allw["alone"] + allw["selfmodel"] if m == "both" else allw[m]
        for rn, nm in kinds:
            ws = [w for w in W if w.get("reach_no") == rn]
            if not ws:
                continue
            own = np.array([w["prop_settled"] for w in ws])
            tru = np.array([w["truth_quiet"] for w in ws])
            wt = np.array([w["wait_s"] for w in ws])
            st = sum(sum(w["steps"].values()) for w in ws)
            ag = sum(w["steps"]["both"] + w["steps"]["neither"] for w in ws)
            lag = np.array([w["wait_s"] - w["truth_first_s"] for w in ws if w["truth_first_s"] >= 0 and w["prop_settled"]])
            print(f"  {m:9s} {nm:18s} n {len(ws):3d} | own rule held {own.sum()}/{len(ws)} ({100 * own.mean():.1f} %), 3 s timeout {int((~own).sum())} | "
                  f"wait median {np.median(wt):.2f} s p90 {q(wt, 90):.2f} s | truth rule at the command {tru.sum()}/{len(ws)} ({100 * tru.mean():.1f} %) | "
                  f"rules agree at the command {100 * np.mean(own == tru):.1f} %, over the wait steps {100 * ag / max(st, 1):.1f} % ({st} steps) | "
                  f"own rule after the truth rule's first quiet step: median {q(lag, 50):+.2f} s p90 {q(lag, 90):+.2f} s")  # fmt: skip
            res.setdefault("settle", {}).setdefault(m, {})[nm] = dict(n=len(ws), own=int(own.sum()), truth=int(tru.sum()), wait_med=float(np.median(wt)),
                                                                     wait_p90=q(wt, 90), agree_cmd=float(np.mean(own == tru)), agree_steps=ag / max(st, 1))  # fmt: skip
        Wc = [w for w in W if w.get("reach_no") in (0, 1, 2, -1, -2)]
        own = np.array([w["prop_settled"] for w in Wc])
        tru = np.array([w["truth_quiet"] for w in Wc])
        st = sum(sum(w["steps"].values()) for w in Wc)
        ag = sum(w["steps"]["both"] + w["steps"]["neither"] for w in Wc)
        both = Counter(c for w in Wc for c, v in w["steps"].items() for _ in range(v))
        tv = np.array([w["truth_v_max_cms"] for w in Wc if w["prop_settled"] and not w["truth_quiet"]])
        print(f"  {m:9s} every command n {len(Wc)}: agree at the command {100 * np.mean(own == tru):.1f} % (own & truth {int((own & tru).sum())}, own only "
              f"{int((own & ~tru).sum())}, truth only {int((~own & tru).sum())}, neither {int((~own & ~tru).sum())}); over the wait steps "
              f"{100 * ag / max(st, 1):.1f} % ({dict(both)}); own only: data3's truth speed (max over the last 0.5 s) median {q(tv, 50):.2f} "
              f"p90 {q(tv, 90):.2f} max {q(tv, 100):.2f} cm/s (its limit 2)")  # fmt: skip
        res.setdefault("settle", {}).setdefault(m, {})["every"] = dict(n=len(Wc), agree_cmd=float(np.mean(own == tru)), agree_steps=ag / max(st, 1),
                                                                 own_only_truth_v=[q(tv, 50), q(tv, 90), q(tv, 100)])  # fmt: skip
    # A5 walking
    print("A5. walking (jittered episodes)")
    for m, key in (("alone", "A"), ("selfmodel", "S")):
        G = [g for t in TASKS for g in per[t][key]]
        lost = sum(str(g.get("why", "")).startswith("lost during walk") for g in G)
        fellw = sum(str(g.get("why", "")).startswith("fell during walk") for g in G)
        walked = sum(g.get("n_walks", 0) > 0 for g in G)
        fix = sum(g.get("n_walks", 0) > 1 for g in G)
        arr = [g["arrive"]["pos_err_cm"] for g in G if "arrive" in g and g.get("n_walks", 0) > 0]
        print(f"  {m:9s} walked {walked}, with a correction walk {fix}, lost during a walk {lost}, fell during a walk {fellw}; arrival error median "
              f"{q(arr, 50):.1f} cm p90 {q(arr, 90):.1f} cm" + (f" (HEFT registered: lmlreal direct {LMLREAL_HEFT_LOST} -> <= 20)" if be == "heft" else ""))  # fmt: skip
        res.setdefault("walk", {})[m] = dict(walked=walked, fix=fix, lost=lost, fell=fellw, arrive_med=q(arr, 50))
    # A6 the inversion
    print("A6. the inversion (selfmodel group, pre-grasp and entry)")
    for rn, nm in ((1, "pre-grasp"), (2, "entry")):
        ws = [reach(g, rn) for t in TASKS for g in per[t]["S"]]
        ws = [w for w in ws if "inv" in w]
        pm = [w["pred_miss_cm"] for w in ws if "pred_miss_cm" in w]
        pm0 = [w["pred_miss0_cm"] for w in ws if "pred_miss0_cm" in w]
        cc = [w["corr_cm"] for w in ws if "corr_cm" in w]
        fb = sum(bool(w.get("model_fallback")) for w in ws)
        cand_err = sum(any(e.get("err") for e in w["inv"]) for w in ws)
        best = Counter(w.get("inv_best") for w in ws if "inv_best" in w)
        print(f"  {nm:9s} n {len(ws)} | D's predicted miss: at the goal median {q(pm0, 50):.2f} cm, at the chosen command median {q(pm, 50):.2f} p90 "
              f"{q(pm, 90):.2f} max {q(pm, 100):.2f} cm, > 1 cm {sum(v > 1 for v in pm)} | correction median {q(cc, 50):.2f} p90 {q(cc, 90):.2f} max "
              f"{q(cc, 100):.2f} cm | winning iterate {dict(sorted(best.items()))} | fallbacks {fb}, with a candidate not planned {cand_err}")  # fmt: skip
        res.setdefault("inversion", {})[nm] = dict(n=len(ws), pred_miss_med=q(pm, 50), pred_miss_p90=q(pm, 90), over1=sum(v > 1 for v in pm),
                                                   corr_med=q(cc, 50), corr_p90=q(cc, 90), fallbacks=fb, cand_err=cand_err)  # fmt: skip
    print("A7. official unjittered start (ep -1)")
    for t in TASKS:
        line = []
        for m in METHODS:
            r = rs.get((t, m, -1))
            line.append(f"{m}: missing" if r is None else f"{m}: {'success' if r['success'] else 'fail'} ({r['reason']}, close "
                        f"{r.get('close_dist_cm', float('nan')):.1f} cm, rise {r.get('rise_max_cm', 0):.1f} cm)")  # fmt: skip
        print(f"  {SHORT[t]:10s} " + " | ".join(line))
        res.setdefault("official", {})[SHORT[t]] = {m: (rs[(t, m, -1)]["success"] if (t, m, -1) in rs else None) for m in METHODS}


# -- B. offline -----------------------------------------------------------------------------------------------------
def predict(models: dict, D: dict, idx: np.ndarray) -> dict:
    GD = M.tensors(D, "cpu")
    out = {}
    for nm, m in models.items():
        G = GD
        if nm == "T3":
            G = dict(GD)
            G["x0"] = GD["x0"].clone()
            G["x0"][:, CHAIN_COL] = G["x0"][:, CHAIN_COL].clamp(max=T3_CHAIN_MAX)
        p = M.predict_batched(m, G, torch.tensor(idx))
        out[nm] = (100 * (p["pos"] - GD["pos_f"][idx]).norm(dim=-1)).numpy().astype(float)
    return out


def hand_speed(D: dict) -> np.ndarray:
    """The acting hand point's speed at the command row: |W[k0] - W[k0 - 1]| / 0.1 s (W cols 4:7 right, 7:10 left)."""
    n = D["n_steps"].astype(int)
    s0 = np.r_[0, np.cumsum(n)[:-1]] + D["k0"].astype(int)
    W = D["W"]
    col = np.where(np.asarray(D["side"]) > 0, 4, 7)
    a = np.stack([W[s0, c : c + 3] for c in (4, 7)], 1)[np.arange(len(n)), (col == 7).astype(int)]
    b = np.stack([W[s0 - 1, c : c + 3] for c in (4, 7)], 1)[np.arange(len(n)), (col == 7).astype(int)]
    return np.linalg.norm(a.astype(float) - b.astype(float), axis=1) / 0.1


def part_b(be: str, res: dict) -> None:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    rng = np.random.default_rng(0)
    # data3: the training band, the val check
    Dt = DT.load(str(ROOT / "selfmodel_data3_20261125" / be / "data" / "all.npz"))
    models = {nm: TR.load_model("E3_ensemble", 1.0, Dt, "cpu", root=r) for nm, r in roots(be).items()}
    sp = DT.splits(Dt, 1.0)
    va = sp["val"][(Dt["fell"][sp["val"]] == 0) & (Dt["settled_end"][sp["val"]] > 0)]
    pv = predict(models, Dt, va)
    tr = sp["train"]
    qtr = Dt["q_0"][tr].astype(float)
    qlo, qhi = np.percentile(qtr, 0.5, axis=0), np.percentile(qtr, 99.5, axis=0)
    vtr = hand_speed(Dt)[tr]
    vlo, vhi = np.percentile(vtr, 0.5), np.percentile(vtr, 99.5)
    val = dict(zip(MODELS, VAL_REF[be]))
    print(f"\n== B. {be} offline. data3 val check ({len(va)} kept val segments): " + ", ".join(
        f"{nm} {np.median(pv[nm]):.3f} cm (reference {val[nm]:.2f})" for nm in MODELS)
          + f"; training band: {len(tr)} command rows; hand speed band [{100 * vlo:.2f}, {100 * vhi:.2f}] cm/s")  # fmt: skip
    res["val_check"] = {nm: float(np.median(pv[nm])) for nm in MODELS}
    res["val_ref"] = val
    del Dt
    # the records
    p = OUT / be / "data" / "all.npz"
    D = dict(np.load(p))
    D.setdefault("walk", np.zeros((len(D["n_steps"]), 3), np.float32))
    D.setdefault("target_b", D["target"])
    D["files"] = [str(p)]
    DT.derive(D)
    N = len(D["n_steps"])
    err = predict(models, D, np.arange(N))
    fell, setl = D["fell"].astype(bool), D["settled_end"].astype(bool)
    kept = ~fell & setl
    rn, mid = D["reach_no"].astype(int), D["method_id"].astype(int)
    official = D["ep"].astype(int) < 0
    pre, own = D["pre_settled"].astype(bool), D["prop_settled"].astype(bool)
    share = np.where(D["win_rows"] > 0, D["furn_rows"] / np.maximum(D["win_rows"], 1), 0.0)
    cont = np.where(D["furn_rows"] == 0, 0, np.where(share < 0.5, 1, 2))
    q0 = D["q_0"].astype(float)
    jin = ~((q0 < qlo) | (q0 > qhi)).any(1)
    hv = hand_speed(D)
    vin = (hv >= vlo) & (hv <= vhi)
    epk = D["task_id"].astype(int) * 100_000 + mid * 10_000 + D["ep"].astype(int)
    dire = D["err_cm"].astype(float)
    base = ~official
    print(f"records {p}: {N} segments (official ep -1 left out: {int(official.sum())}); fell / lost {int((fell & base).sum())}; label not settled "
          f"{int((~fell & ~setl & base).sum())} (kinds {dict(Counter(D['label_kind'][~fell & ~setl & base].astype(int).tolist()))}); kept "
          f"{int((kept & base).sum())}; by method kept {dict(Counter(mid[kept & base].tolist()))}")  # fmt: skip

    def groups(sel):
        k = epk[sel]
        o = np.argsort(k, kind="stable")
        return np.split(np.flatnonzero(sel)[o], np.flatnonzero(np.diff(k[o])) + 1)

    def boot(sel, stats):
        G = groups(sel)
        n = len(G)
        if n < 2:
            return [(float("nan"), float("nan"))] * len(stats)
        v = []
        for _ in range(NB):
            s = np.concatenate([G[j] for j in rng.integers(0, n, n)])
            v.append([f(s) for f in stats])
        v = np.array(v, float)
        return [(q(c[np.isfinite(c)], 2.5), q(c[np.isfinite(c)], 97.5)) for c in v.T]

    def cell(name, sel, key):
        s = np.flatnonzero(sel)
        if len(s) == 0:
            print(f"  {name:44s} n 0")
            return
        meds = {nm: float(np.median(err[nm][s])) for nm in MODELS}
        st = [lambda s_, nm=nm: float(np.median(err[nm][s_])) / val[nm] for nm in MODELS] + [lambda s_: float(np.median(err["D"][s_]) / np.median(err["M"][s_]))]
        ci = boot(sel, st)
        dm = meds["D"] / meds["M"]
        print(f"  {name:44s} n {len(s):4d} | " + " | ".join(f"{nm} {meds[nm]:.2f} cm (p90 {q(err[nm][s], 90):.2f}) x val {meds[nm] / val[nm]:.2f} "
                                                           f"[{ci[i][0]:.2f}, {ci[i][1]:.2f}]" for i, nm in enumerate(MODELS))
              + f" | D/M {dm:.3f} [{ci[3][0]:.3f}, {ci[3][1]:.3f}] | command target to landing (data.py err_cm) {np.median(dire[s]):.2f} cm")  # fmt: skip
        res.setdefault("cells", {})[key] = dict(n=len(s), med=meds, xval={nm: meds[nm] / val[nm] for nm in MODELS}, xval_ci={nm: ci[i] for i, nm in enumerate(MODELS)},
                                               dm=dm, dm_ci=ci[3], direct=float(np.median(dire[s])))  # fmt: skip

    for gname, gsel in (("direct (primary)", mid == 0), ("selfmodel", mid == 1), ("both groups", np.ones(N, bool))):
        print(f"B1. kept segments, {gname}: model prediction error (predicted vs settled hand point); x val with 95 % episode-resample intervals")
        k_ = kept & base & gsel
        for r_, rname in ((1, "第一下 pre-grasp"), (2, "第二下 entry")):
            cell(f"{rname}: all", k_ & (rn == r_), f"{gname}|{r_}|all")
            cell(f"{rname}: settled start (truth rule)", k_ & (rn == r_) & pre, f"{gname}|{r_}|settled")
            cell(f"{rname}: not settled (truth rule)", k_ & (rn == r_) & ~pre, f"{gname}|{r_}|unsettled")
            cell(f"{rname}: settled start (own rule)", k_ & (rn == r_) & own, f"{gname}|{r_}|own_settled")
            for cv, cn in ((0, "no scene contact"), (1, "light contact < 50 %"), (2, "stuck >= 50 %")):
                cell(f"{rname}: {cn}", k_ & (rn == r_) & (cont == cv), f"{gname}|{r_}|contact{cv}")
            cell(f"{rname}: start in band (joints + hand speed)", k_ & (rn == r_) & jin & vin, f"{gname}|{r_}|inband")
        cell("via point (reach_no 0)", k_ & (rn == 0), f"{gname}|0|all")
    print("B2. start state in data3's training band (every segment that was commanded, jittered episodes)")
    for gname, gsel in (("direct", mid == 0), ("selfmodel", mid == 1)):
        for r_, rname in ((1, "第一下"), (2, "第二下")):
            s = base & gsel & (rn == r_)
            n = int(s.sum())
            if n == 0:
                continue
            print(f"  {gname:9s} {rname}: n {n} | joints in band {int((jin & s).sum())} ({100 * (jin & s).sum() / n:.1f} %) | hand speed in band "
                  f"{int((vin & s).sum())} ({100 * (vin & s).sum() / n:.1f} %; above {int(((hv > vhi) & s).sum())}, below {int(((hv < vlo) & s).sum())}) | both "
                  f"{int((jin & vin & s).sum())} ({100 * (jin & vin & s).sum() / n:.1f} %) | truth-settled start {int((pre & s).sum())} ({100 * (pre & s).sum() / n:.1f} %) | "
                  f"own-rule settled start {int((own & s).sum())} | hand speed median {100 * np.median(hv[s]):.2f} cm/s | scene contact: none "
                  f"{int(((cont == 0) & s).sum())}, light {int(((cont == 1) & s).sum())}, stuck {int(((cont == 2) & s).sum())}")  # fmt: skip
            res.setdefault("band", {})[f"{gname}|{r_}"] = dict(n=n, joints=int((jin & s).sum()), speed=int((vin & s).sum()), both=int((jin & vin & s).sum()),
                                                              truth_settled=int((pre & s).sum()), own_settled=int((own & s).sum()))  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=list(VAL_REF))
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--skip-offline", action="store_true")
    a = ap.parse_args()
    res: dict = dict(backend=a.backend)
    part_a(a.backend, res)
    if not a.skip_offline:
        part_b(a.backend, res)
    if a.json:
        a.json.write_text(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o), indent=1))
