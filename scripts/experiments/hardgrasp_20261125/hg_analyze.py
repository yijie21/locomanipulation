"""selfmodel_hardgrasp_20261125, LocoManiLab: analysis of the formal episodes (read-only; written after the freeze, it
changes nothing that ran). Formal episodes 0..29 per task (our jitter), the official unjittered start (-1) listed apart.

    python hg_analyze.py --out $O [--eps 0:30]
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np

REASONS = ("ok", "perception", "stance_or_ik", "fell_walking", "hit_furniture", "missed", "dropped", "other")
VAL_CM = 1.88  # hostdata self model's validation error


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def sign_p(d: np.ndarray) -> float:
    d = d[d != 0]
    n, k = len(d), int((d < 0).sum())
    return mcnemar(k, n - k) if n else 1.0


def median_ci(d: np.ndarray, conf: float = 0.95) -> tuple[float, float]:
    """Distribution-free CI of the median of the paired differences (order statistics, binomial)."""
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


def load(out: Path) -> dict:
    rs = {}
    for p in glob.glob(str(out / "episodes" / "*" / "*" / "ep*.json")):
        r = json.loads(Path(p).read_text())
        rs[(r["task"], r["method"], r["ep"])] = r
    return rs


def reach(r: dict, k: int) -> dict:
    return next((w for w in r.get("reaches", []) if w.get("k") == k), {})


def main(out: Path, lo: int, hi: int) -> None:
    rs = load(out)
    tasks = sorted({k[0] for k in rs if lo <= k[2] < hi})
    print(f"formal episodes {lo}..{hi - 1}; {sum(1 for k in rs if lo <= k[2] < hi)} records\n")
    rows, pooled, pooled_x = [], dict(b=0, c=0, nA=0, nS=0, n=0, d=[]), dict(b=0, c=0, nA=0, nS=0, n=0, d=[])
    print("== per task (A = alone, S = selfmodel; success = the task's own lift judgement, take-statue: lift >= 10 cm 1 s)")
    for t in tasks:
        eps = [e for e in range(lo, hi) if (t, "alone", e) in rs and (t, "selfmodel", e) in rs]
        A = [rs[(t, "alone", e)] for e in eps]
        S = [rs[(t, "selfmodel", e)] for e in eps]
        b = sum(a["success"] and not s["success"] for a, s in zip(A, S))
        c = sum(s["success"] and not a["success"] for a, s in zip(A, S))
        d = np.array([s["close_dist_cm"] - a["close_dist_cm"] for a, s in zip(A, S) if "close_dist_cm" in a and "close_dist_cm" in s])
        bad = sum(g["reason"] in ("perception", "stance_or_ik") for g in A + S) / max(1, 2 * len(eps))
        nA, nS = sum(g["success"] for g in A), sum(g["success"] for g in S)
        top = nA >= 0.8 * len(eps)
        excl = bad > 0.5
        for P, use in ((pooled, True), (pooled_x, not excl)):
            if use:
                P["b"] += b
                P["c"] += c
                P["nA"] += nA
                P["nS"] += nS
                P["n"] += len(eps)
                P["d"] += d.tolist()
        cdA = [g["close_dist_cm"] for g in A if "close_dist_cm" in g]
        cdS = [g["close_dist_cm"] for g in S if "close_dist_cm" in g]
        ci = median_ci(d)
        print(f"{t}  pairs {len(eps)}  {'EXCLUDED (perception+stance %.0f%%)' % (100 * bad) if excl else 'kept (perception+stance %.0f%%)' % (100 * bad)}{'  AT CEILING' if top else ''}")
        print(f"  success A {nA}/{len(eps)}  S {nS}/{len(eps)}  | A-only {b}  S-only {c}  McNemar p {mcnemar(b, c):.3g}")
        print(f"  close-dist median A {np.median(cdA) if cdA else float('nan'):.1f} cm (n {len(cdA)})  S {np.median(cdS) if cdS else float('nan'):.1f} cm (n {len(cdS)})"
              f"  | paired S-A median {np.median(d) if len(d) else float('nan'):+.1f} cm, 95% CI [{ci[0]:+.1f}, {ci[1]:+.1f}], n {len(d)}, sign p {sign_p(d):.3g}")  # fmt: skip
        for name, G in (("A", A), ("S", S)):
            cnt = Counter(g["reason"] for g in G)
            rin0 = [reach(g, 0).get("q_in_range") for g in G if "q_in_range" in reach(g, 0)]
            rin1 = [reach(g, 1).get("q_in_range") for g in G if "q_in_range" in reach(g, 1)]
            qo = [reach(g, 0).get("q_out") for g in G if "q_out" in reach(g, 0)]
            pe0 = [reach(g, 0)["pred_err_cm"] for g in G if "pred_err_cm" in reach(g, 0)]
            pe1 = [reach(g, 1)["pred_err_cm"] for g in G if "pred_err_cm" in reach(g, 1)]
            e1 = [reach(g, 1)["err_cm"] for g in G if "err_cm" in reach(g, 1)]
            pm = [reach(g, 1)["pred_miss_cm"] for g in G if "pred_miss_cm" in reach(g, 1)]
            print(f"  {name} reasons " + " ".join(f"{k} {cnt.get(k, 0)}" for k in REASONS)
                  + f" | start in range: pre-grasp cmd {sum(rin0)}/{len(rin0)}, grasp cmd {sum(rin1)}/{len(rin1)}; joints out at pre-grasp median {np.median(qo) if qo else float('nan'):.0f}"
                  + f" | pred err (pred vs actual landing) pre-grasp {np.median(pe0) if pe0 else float('nan'):.1f} cm, grasp {np.median(pe1) if pe1 else float('nan'):.1f} cm (n {len(pe1)}; val {VAL_CM})"
                  + f" | grasp-reach miss {np.median(e1) if e1 else float('nan'):.1f} cm"
                  + (f" | model's own predicted miss {np.median(pm):.2f} cm" if pm else ""))  # fmt: skip
        rows.append((t, len(eps), nA, nS, b, c))
    for name, P in (("ALL kept tasks", pooled), ("after the exclusion rule", pooled_x)):
        d = np.array(P["d"])
        ci = median_ci(d)
        n = max(P["n"], 1)
        print(f"\n== pooled, {name}: pairs {P['n']}  success A {P['nA']} ({100 * P['nA'] / n:.1f}%)  S {P['nS']} ({100 * P['nS'] / n:.1f}%)  "
              f"diff {100 * (P['nS'] - P['nA']) / n:+.1f} pp  | A-only {P['b']}  S-only {P['c']}  McNemar exact p {mcnemar(P['b'], P['c']):.3g}")  # fmt: skip
        print(f"   close-dist paired S-A median {np.median(d) if len(d) else float('nan'):+.2f} cm, 95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}], n {len(d)}, "
              f"sign p {sign_p(d):.3g}, Wilcoxon p {wilcoxon_p(d):.3g}")  # fmt: skip
    print("\n== pooled over all tasks, per group (formal jittered episodes)")
    for m in ("alone", "selfmodel"):
        G = [r for k, r in rs.items() if k[1] == m and lo <= k[2] < hi]
        cnt = Counter(g["reason"] for g in G)
        print(f"  {m}: n {len(G)}  reasons " + " ".join(f"{k} {cnt.get(k, 0)}" for k in REASONS))
        for kk, nm in ((0, "pre-grasp reach"), (1, "grasp reach")):
            W_ = [reach(g, kk) for g in G if "q_in_range" in reach(g, kk)]
            qin = [w["q_in_range"] for w in W_]
            qo = np.array([w["q_out"] for w in W_])
            pe = np.array([w["pred_err_cm"] for w in W_ if "pred_err_cm" in w])
            er = np.array([w["err_cm"] for w in W_ if "err_cm" in w])
            pm = np.array([w["pred_miss_cm"] for w in W_ if "pred_miss_cm" in w])
            q = lambda a, p: float(np.percentile(a, p)) if len(a) else float("nan")  # noqa: E731
            print(f"    {nm}: n {len(W_)}  start in range {sum(qin)}/{len(qin)}  joints out (of 29) median {q(qo, 50):.0f} max {q(qo, 100):.0f}"
                  f"  | pred err median {q(pe, 50):.2f} cm [IQR {q(pe, 25):.2f}-{q(pe, 75):.2f}] n {len(pe)} (val {VAL_CM})"
                  f"  | in-range only: {np.median([w['pred_err_cm'] for w in W_ if w['q_in_range'] and 'pred_err_cm' in w]) if any(w['q_in_range'] and 'pred_err_cm' in w for w in W_) else float('nan'):.2f} cm"
                  f"  | reach miss median {q(er, 50):.2f} cm" + (f"  | model's own predicted miss median {q(pm, 50):.2f} cm" if len(pm) else ""))  # fmt: skip
    print("\n== official unjittered start (ep -1)")
    for t in tasks:
        line = []
        for m in ("alone", "selfmodel"):
            r = rs.get((t, m, -1))
            if r is None:
                line.append(f"{m}: missing")
                continue
            line.append(f"{m}: {'success' if r['success'] else 'fail'} ({r['reason']}, close {r.get('close_dist_cm', float('nan')):.1f} cm, rise {r.get('rise_max_cm', 0):.1f} cm)")
        print(f"  {t}: " + " | ".join(line))
    # errors / jitter
    errs = [k for k, r in rs.items() if lo <= k[2] < hi and "error" in r]
    jit = [k for k, r in rs.items() if lo <= k[2] < hi and not (r.get("jitter") or {}).get("jitter_ok", True)]
    dev = [r.get("look_state_dev_mm", 0) for k, r in rs.items() if lo <= k[2] < hi and "look_state_dev_mm" in r]
    print(f"\nworker errors {len(errs)} {errs[:5]}; jitter fell back to unjittered {len(jit)}; look-state deviation max {max(dev) if dev else float('nan'):.3f} mm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/hardgrasp_20261125/locomanilab"))
    ap.add_argument("--eps", default="0:30")
    a = ap.parse_args()
    lo, hi = (int(v) for v in a.eps.split(":"))
    main(a.out, lo, hi)
