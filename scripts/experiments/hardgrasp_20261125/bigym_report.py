"""selfmodel_hardgrasp_20261125, BiGym part: the report's tables from the formal rows (post-run analysis only; written
after the freeze, reads the rows, decides nothing in the run). Adds to bigym_hg.py analyze: the distribution-free
interval of the paired median hand difference (the sign test inverted), the pairs where S actually moved its goal, the
failure table side by side, and the dropped episodes' contacts.

    python bigym_report.py "<formal>/rows/*.jsonl"
"""

from __future__ import annotations

import glob
import math
import sys

import json


def binom_cdf(k: int, n: int) -> float:
    return sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n


def sign_p(d) -> float:
    d = [x for x in d if x != 0]
    n, k = len(d), sum(x < 0 for x in d)
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * binom_cdf(min(k, n - k), n))


def mcnemar(b: int, c: int) -> float:
    return 1.0 if b + c == 0 else min(1.0, 2.0 * binom_cdf(min(b, c), b + c))


def median(v):
    v = sorted(v)
    n = len(v)
    return None if n == 0 else (v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2]))


def median_ci(d, level=0.95):
    """Order-statistic interval for the median of d (the sign test inverted): [d_(k), d_(n-k+1)] with the largest k
    such that P(Binom(n, 1/2) <= k-1) <= (1-level)/2; its exact coverage. None when n is too small."""
    d, n = sorted(d), len(d)
    a = (1 - level) / 2
    k = 0
    while k + 1 <= n and binom_cdf(k, n) <= a:  # P(X <= k) <= a -> the (k+1)-th order statistic is a lower bound
        k += 1
    if k == 0:
        return None
    return d[k - 1], d[n - k], 1 - 2 * binom_cdf(k - 1, n)


def hand_block(title, pairs) -> None:
    hp = [(d["hand_target_cm"], s["hand_target_cm"], d["seed"]) for d, s in pairs
          if d.get("hand_target_cm") is not None and s.get("hand_target_cm") is not None]  # fmt: skip
    print(f"  {title}: {len(hp)} pairs")
    if not hp:
        return
    dd = [y - x for x, y, _ in hp]
    ci = median_ci(dd)
    cis = f"[{ci[0]:+.2f}, {ci[1]:+.2f}] (exact coverage {100 * ci[2]:.1f} %)" if ci else "n too small for 95 %"
    print(f"    D median {median([x for x, _, _ in hp]):.2f} cm, S median {median([y for _, y, _ in hp]):.2f} cm;"
          f" paired S - D median {median(dd):+.2f} cm, 95 % interval {cis}; S nearer {sum(x < 0 for x in dd)},"
          f" farther {sum(x > 0 for x in dd)}, equal {sum(x == 0 for x in dd)}; sign test p = {sign_p(dd):.4g}")  # fmt: skip


def shifted(s) -> bool:
    sc = (s.get("reach1") or {}).get("scale")
    return sc is not None and sc > 0


def main() -> None:
    rows = []
    for pat in sys.argv[1:]:
        for fn in sorted(glob.glob(pat)):
            rows += [json.loads(x) for x in open(fn)]
    errs = [r for r in rows if r.get("error")]
    ok = [r for r in rows if not r.get("error")]
    R = {(r["seed"], r["group"]): r for r in ok}
    seeds = sorted({s for s, _ in R})
    pairs = [(R[(s, "D")], R[(s, "S")]) for s in seeds if (s, "D") in R and (s, "S") in R]
    print(f"rows {len(rows)} (error rows {len(errs)}), complete pairs {len(pairs)}")
    for r in errs:
        print(f"  error row: seed {r['seed']} {r['group']}: {r['error'][:160]}")
    n = len(pairs)
    kd, ks = sum(d["success"] for d, _ in pairs), sum(s["success"] for _, s in pairs)
    b = sum(d["success"] and not s["success"] for d, s in pairs)
    c = sum(s["success"] and not d["success"] for d, s in pairs)
    print("\n[1] grasp-and-lift success")
    print(f"  D {kd}/{n}, S {ks}/{n}; discordant D-only {b}, S-only {c}; McNemar exact p = {mcnemar(b, c):.4g}")
    print("\n[2] hand (Dex1 pinch centre) to the grasp target at the close, cm")
    hand_block("all pairs with both closes", pairs)
    sp = [(d, s) for d, s in pairs if shifted(s)]
    hand_block("pairs where S moved its goal (reach 1 scale 1, 2/3 or 1/3)", sp)
    ns = [(d, s) for d, s in pairs if s.get("reach1") and not shifted(s)]
    hand_block("pairs where S ran D's route (D had no plan, or no scale planned)", ns)
    sc = {}
    for _, s in pairs:
        if s.get("reach1"):
            k = str((s["reach1"]).get("scale"))
            sc[k] = sc.get(k, 0) + 1
    print(f"    S's reach-1 scale counts: {sc}")
    print("\n[3] failure classes (registration order)")
    order = ("success", "perception", "reach", "fell", "furniture", "missed", "dropped", "other")
    cnt = {g: {} for g in "DS"}
    for d, s in pairs:
        for g, r in (("D", d), ("S", s)):
            f = r.get("fail") or "success"
            cnt[g][f] = cnt[g].get(f, 0) + 1
    print("  class        D    S")
    for f in order:
        print(f"  {f:<11} {cnt['D'].get(f, 0):>3}  {cnt['S'].get(f, 0):>3}")
    whys = {}
    for d, s in pairs:
        for g, r in (("D", d), ("S", s)):
            if r.get("why"):
                whys[(g, r["why"])] = whys.get((g, r["why"]), 0) + 1
    for (g, w), k in sorted(whys.items()):
        print(f"    {g} {w}: {k}")
    print("\n[4] start state against M2's training band (29 joints, 0.5-99.5 %), at reach 1's command, D episodes")
    rg = [d["range"] for d, _ in pairs if d.get("range")]
    if rg:
        o = [x["n_out29"] for x in rg]
        o7 = [x["n_out_arm7"] for x in rg]
        print(f"  {len(rg)} episodes: any joint out {sum(x > 0 for x in o)}, out-of-band joints median {median(o)}, max"
              f" {max(o)}; right arm (7) any out {sum(x > 0 for x in o7)}, median {median(o7)}")  # fmt: skip
        names = {}
        for x in rg:
            for nm in x["names"]:
                names[nm] = names.get(nm, 0) + 1
        print(f"  joints out (episodes): {dict(sorted(names.items(), key=lambda t: -t[1]))}")
        rg2 = [d["range2"] for d, _ in pairs if d.get("range2")]
        if rg2:
            print(f"  at reach 2's command: out-of-band joints median {median([x['n_out29'] for x in rg2])}, right arm"
                  f" median {median([x['n_out_arm7'] for x in rg2])} of 7 ({len(rg2)} episodes)")  # fmt: skip
    print("\n[5] M2's prediction error, reach 1 (|predicted - actual settled palm offset|, cm; validation 0.1-0.4)")
    for g, k in (("D", 0), ("S", 1)):
        e = sorted(p[k]["reach1"]["pred_err_cm"] for p in pairs if p[k].get("reach1"))
        lab = [math.sqrt(sum(v * v for v in p[k]["reach1"]["label_head_cm"])) for p in pairs if p[k].get("reach1")]
        if e:
            p90 = e[min(len(e) - 1, int(math.ceil(0.9 * len(e))) - 1)]
            print(f"  {g}: n {len(e)}, median {median(e):.2f}, p90 {p90:.2f}, max {e[-1]:.2f}; the actual offset itself"
                  f" median {median(lab):.2f}")  # fmt: skip
    e2 = [d["reach2"]["pred_err_cm"] for d, _ in pairs if d.get("reach2")]
    if e2:
        print(f"  reach 2 (recorded, not used; out of band): D median {median(e2):.1f}")
    print("\n[6] per pair")
    for d, s in pairs:
        def one(r):
            r1 = r.get("reach1") or {}
            return (f"{'OK' if r['success'] else r.get('fail')}{'(' + r['why'][:18] + ')' if r.get('why') else ''}"
                    f" hand {r.get('hand_target_cm')} mv {r.get('obj_moved_before_close_mm')} held"
                    f" {r.get('held_after_close')}/{r.get('held_end')} rise {r.get('rise_end_cm')} sc {r1.get('scale')}"
                    f" line {r1.get('fallback_line', False)}")  # fmt: skip
        print(f"  {d['seed']}  D: {one(d)}  |  S: {one(s)}")
    print("\n[7] dropped episodes: the mug's contacts after the lift")
    for d, s in pairs:
        for g, r in (("D", d), ("S", s)):
            if r.get("fail") == "dropped":
                tc = r.get("target_contacts_after_lift") or []
                grip = [x for x in tc if "g1" in x[1].lower() or "dex" in x[1].lower() or "right" in x[1].lower()]
                print(f"  {r['seed']} {g}: hand rise {r.get('hand_rise_cm')} cm, mug rise {r.get('rise_end_cm')} cm,"
                      f" gripper contacts {len(grip)}, contacts {tc}")  # fmt: skip


if __name__ == "__main__":
    main()
