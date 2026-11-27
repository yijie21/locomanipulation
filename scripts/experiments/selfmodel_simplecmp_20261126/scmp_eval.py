"""selfmodel_simplecmp_20261126: the registered evaluation; read-only.

    python3 scmp_eval.py --run <run dir> --main <main metric bench dir> --range <scmp_range.py json> [--json F]

Arms: A and J (the main metric's rows, reused: AL / JL must equal them), AL (dryR), R_D (rows RD_), S_D (SD_), JL, J_D (JD_),
J0. Episodes: the four pick tasks x 0-29, ep = 10 level + index; a layout is (task, index) (levels 0-2 of one index are the
same layout: 40 layouts). TabletopGrasp's blocked layouts (A's pregrasp stopped by the table's edge: indices 0, 1, 2, 4,
fixed from A's rows before this entry) are reported apart.

Printed: the consistency checks; the pre-split failures per task and class; the decision set per time limit; per task and
on the decision set, success at 45 s and SIMPLE's official limit for every arm; the paired tests (the layout sign test:
per layout the successes over its three levels, + / - where they differ, exact two-sided binomial; the episode McNemar
exact test beside it; the success difference with a 95 % interval from 4000 resamples of the layouts, seed 0); the
mechanism (each stage's palm miss, D's prediction error of the sent command at the stage's end, the blocked layouts'
pregrasps that cleared the edge); F1 (D on the sent corrected commands: error at the task label against the two
denominators, the in-range share, the IK > 2 cm share against AL's); S_D's pregrasps with IK > 3 cm; R_D's shrink; the
route arm's plans.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

TASKS = ("XMovePickTeleop", "XMoveBendPickTeleop", "BendPickMP", "TabletopGraspMP")
SHORT = dict(zip(TASKS, ("XMovePick", "XMoveBendPick", "BendPick", "TabletopGrasp")))
BLOCKED = {0, 1, 2, 4}  # TabletopGrasp indices (all three levels): A's 12 failures, the hand held under the table's edge
VAL, TIID = 1.886, 0.993
LIMITS = (("45 s", "success_45"), ("official", "success_official"))
NB = 4000
SKIP_ROW = {"wall_s", "correct_reset", "film"}


# -- rows -------------------------------------------------------------------------------------------------------------
def rows_of(d: Path, arm: str) -> dict:
    out = {}
    for f in sorted(glob.glob(str(d / f"{arm}_*.jsonl"))):
        if "__" in Path(f).name or not Path(f).name[len(arm) + 1].isupper():
            continue
        for line in open(f):
            r = json.loads(line)
            out[(r["task"], int(r["ep"]))] = r
    return out


def clean(r: dict) -> dict:
    r = {k: v for k, v in r.items() if k not in SKIP_ROW}
    r["stages"] = [{k: v for k, v in s.items() if k != "correct"} for s in r.get("stages", [])]
    r.get("bench", {}).get("options", {}).pop("sonic_vr_frame", None)  # recorded by the current code only (its default)
    return r


def stage(r: dict, name: str) -> dict:
    s = [x for x in r.get("stages", []) if x["stage"] == name]
    return s[0] if s else {}


def pre_split(r: dict) -> str | None:
    """A failure before the arms' first different command (the pregrasp): the class, else None."""
    why = str(r.get("why") or "")
    if stage(r, "pregrasp"):
        return None
    if "look" in why and ("no" in why or "missing" in why):
        return "no look"
    if "plan" in why or "ik" in why.lower():
        return "plan after the look"
    if r.get("fell") is not None:
        return "fell before the pregrasp"
    return "perception"


# -- statistics -------------------------------------------------------------------------------------------------------
def binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2**n)


def paired(X: dict, Y: dict, keys: list, col: str, rng) -> dict:
    """Y against X on the episodes keys: counts, layout sign test, episode McNemar, the difference's interval (pp)."""
    keys = [k for k in keys if k in X and k in Y]
    if not keys:
        return dict(n=0)
    x = {k: bool(X[k].get(col)) for k in keys}
    y = {k: bool(Y[k].get(col)) for k in keys}
    lay = defaultdict(list)
    for k in keys:
        lay[(k[0], k[1] % 10)].append(k)
    L = list(lay)
    d = {lk: sum(y[k] for k in lay[lk]) - sum(x[k] for k in lay[lk]) for lk in L}
    plus, minus = sum(v > 0 for v in d.values()), sum(v < 0 for v in d.values())
    b, c = sum(y[k] and not x[k] for k in keys), sum(x[k] and not y[k] for k in keys)
    n_l = np.array([len(lay[lk]) for lk in L])
    dv = np.array([d[lk] for lk in L], float)
    bs = []
    for _ in range(NB):
        j = rng.integers(0, len(L), len(L))
        bs.append(100 * dv[j].sum() / n_l[j].sum())
    return dict(n=len(keys), n_layouts=len(L), x=sum(x.values()), y=sum(y.values()), diff_pp=100 * (sum(y.values()) - sum(x.values())) / len(keys),
                ci_pp=(float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))), layouts_plus=plus, layouts_minus=minus,
                p_sign=binom_two_sided(min(plus, minus), plus + minus), mc_b=b, mc_c=c, p_mcnemar=binom_two_sided(min(b, c), b + c))  # fmt: skip


def fmt_pair(o: dict) -> str:
    if not o.get("n"):
        return "no pairs"
    return (f"{o['y']} / {o['n']} vs {o['x']} / {o['n']} ({o['diff_pp']:+.1f} pp [{o['ci_pp'][0]:+.1f}, {o['ci_pp'][1]:+.1f}] by layout); "
            f"layouts + {o['layouts_plus']} / - {o['layouts_minus']} of {o['n_layouts']}, sign test p {o['p_sign']:.4g}; "
            f"episodes + {o['mc_b']} / - {o['mc_c']}, McNemar p {o['p_mcnemar']:.4g}")  # fmt: skip


def med(v) -> float:
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else float("nan")


def boot_med(vals: list, lays: list, rng) -> tuple[float, float]:
    """95 % interval of the median, the layouts resampled."""
    g = defaultdict(list)
    for v, lk in zip(vals, lays):
        g[lk].append(v)
    L = list(g)
    if not L:
        return (float("nan"), float("nan"))
    bs = []
    for _ in range(NB // 4):
        j = rng.integers(0, len(L), len(L))
        bs.append(np.median(np.concatenate([g[L[i]] for i in j])))
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# -- the --correct record -----------------------------------------------------------------------------------------------
def corr_logs(run: Path, arm: str) -> dict:
    out = {}
    for f in sorted(glob.glob(str(run / f"corr_{arm}" / "log" / "*.jsonl"))):
        for line in open(f):
            g = json.loads(line)
            out[(g["task"], 10 * int(g["level"]) + int(g["index"]))] = g
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--range", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    g = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = {arm: rows_of(g.main / "rows", arm) for arm in ("A", "J")}
    for arm in ("AL", "RD", "SD", "JL", "JD", "J0", "JDR"):
        rows[arm] = rows_of(g.run / "rows", arm)
    logs = {arm: corr_logs(g.run, arm) for arm in ("AL", "RD", "SD", "JL", "JD", "J0", "JDR")}
    eps = [(t, e) for t in TASKS for e in range(30)]
    out: dict = {}
    print("== rows: " + ", ".join(f"{a} {len(r)}" for a, r in rows.items()))

    # 1. consistency
    print("\n== 1. consistency (every row and stage key but wall_s, --correct's fields and the option sonic_vr_frame)")
    for a, b in (("A", "AL"), ("J", "JL"), ("RD", "SD")):
        ks = [k for k in rows[a] if k in rows[b]]
        same = [k for k in ks if clean(rows[a][k]) == clean(rows[b][k])]
        print(f"  {a} = {b}: {len(same)} / {len(ks)} episodes identical" + ("" if len(same) == len(ks) else
              f"; differ: {sorted(set(ks) - set(same))[:12]}"))  # fmt: skip
        out[f"same_{a}_{b}"] = [len(same), len(ks)]
    # R_D and S_D sent the same commands where R's shrink never moved a candidate
    nsh = [k for k, lg in logs["RD"].items() if not any(sh["shrunk"] for s in lg["stages_correct"] for sh in (s.get("shrink") or []))]
    print(f"  R_D episodes without a shrunk candidate (S_D's commands are then R_D's): {len(nsh)} / {len(logs['RD'])}")

    # 2. pre-split failures
    print("\n== 2. failures before the arms split (A's rows), per task and class")
    excl_pre = set()
    for t in TASKS:
        cls = defaultdict(int)
        for e in range(30):
            r = rows["A"].get((t, e))
            c = "no row" if r is None else pre_split(r)
            if c:
                cls[c] += 1
        n = sum(cls.values())
        if n > 15:
            excl_pre.add(t)
        print(f"  {SHORT[t]:14s}: {n} ({dict(cls) if cls else 'none'})" + (" -> excluded (over half)" if n > 15 else ""))

    # 3. per task tables
    print("\n== 3. success per task (TabletopGrasp also without / within the blocked layouts)")
    arms_s = ("A", "AL", "RD", "SD")
    for lab, col in LIMITS:
        print(f"  -- {lab}")
        for t in TASKS:
            parts = [f"{a} {sum(bool(rows[a].get((t, e), {}).get(col)) for e in range(30))}" for a in arms_s]
            line = f"  {SHORT[t]:14s}: " + ", ".join(parts) + " (of 30)"
            if t == "TabletopGraspMP":
                parts_j = [f"{a} {sum(bool(rows[a].get((t, e), {}).get(col)) for e in range(30) if (t, e) in rows[a])}" for a in ("J", "JL", "JD", "J0", "JDR")]
                line += "; route " + ", ".join(parts_j)
                for nm, sel in (("free", lambda e: e % 10 not in BLOCKED), ("blocked", lambda e: e % 10 in BLOCKED)):
                    line += f"\n  {'':14s}  {nm}: " + ", ".join(f"{a} {sum(bool(rows[a].get((t, e), {}).get(col)) for e in range(30) if sel(e))}"
                                                            for a in ("A", "RD", "SD", "J", "JD", "J0")) + f" (of {sum(sel(e) for e in range(30))})"  # fmt: skip
            print(line)

    # 4. decision set
    dsets = {}
    print("\n== 4. decision set (lmlcmp3's rule: pre-split failures over half out; A >= 80 % 'ceiling' out)")
    for lab, col in LIMITS:
        keep = []
        for t in TASKS:
            sa = sum(bool(rows["A"].get((t, e), {}).get(col)) for e in range(30))
            why = "pre-split" if t in excl_pre else ("ceiling" if sa >= 24 else "")
            if not why:
                keep.append(t)
            print(f"  {lab:8s} {SHORT[t]:14s}: A {sa} / 30" + (f" -> out ({why})" if why else " -> in"))
        dsets[lab] = keep
    out["decision_set"] = dsets

    # 5. paired tests
    print("\n== 5. paired tests")
    res = {}
    for lab, col in LIMITS:
        ds = [k for k in eps if k[0] in dsets[lab]]
        for nm, a, b in (("P1 R_D vs A", "A", "RD"), ("S_D vs A", "A", "SD"), ("R_D vs S_D", "SD", "RD")):
            o = paired(rows[a], rows[b], ds, col, rng)
            res[f"{nm} | {lab} | decision set"] = o
            print(f"  {nm:12s} {lab:8s} decision set {[SHORT[t] for t in dsets[lab]]}: {fmt_pair(o)}")
        for t in TASKS:
            o = paired(rows["A"], rows["RD"], [k for k in eps if k[0] == t], col, rng)
            res[f"R_D vs A | {lab} | {t}"] = o
            print(f"  {'R_D vs A':12s} {lab:8s} {SHORT[t]:14s}: {fmt_pair(o)}")
        o = paired(rows["A"], rows["RD"], eps, col, rng)
        res[f"R_D vs A | {lab} | all"] = o
        print(f"  {'R_D vs A':12s} {lab:8s} {'all four':14s}: {fmt_pair(o)}")
        tg = [k for k in eps if k[0] == "TabletopGraspMP"]
        for nm, sel in (("free", lambda k: k[1] % 10 not in BLOCKED), ("blocked", lambda k: k[1] % 10 in BLOCKED)):
            o = paired(rows["A"], rows["RD"], [k for k in tg if sel(k)], col, rng)
            res[f"R_D vs A | {lab} | TabletopGrasp {nm}"] = o
            print(f"  {'R_D vs A':12s} {lab:8s} TabletopGrasp {nm}: {fmt_pair(o)}")
        routed = [k for k in tg if k in rows["JD"]]
        for nm, a, b in (("route J_D vs J0", "J0", "JD"), ("J_D vs J", "J", "JD"), ("J0 vs J", "J", "J0"),
                         ("J_DR vs J0", "J0", "JDR"), ("J_DR vs J_D", "JD", "JDR")):
            o = paired(rows[a], rows[b], routed, col, rng)
            res[f"{nm} | {lab}"] = o
            print(f"  {nm:15s} {lab:8s} routed TabletopGrasp: {fmt_pair(o)}")
    out["tests"] = res

    # 6. mechanism
    print("\n== 6. mechanism: each stage's palm miss (cm, median; the episodes all three arms ran), D's prediction error of the sent"
          " command at the stage's end (palm, cm, median)")
    groups = [(SHORT[t], (lambda t_: lambda k: k[0] == t_)(t)) for t in TASKS] + [
        ("TabletopGrasp free", lambda k: k[0] == "TabletopGraspMP" and k[1] % 10 not in BLOCKED),
        ("TabletopGrasp blocked", lambda k: k[0] == "TabletopGraspMP" and k[1] % 10 in BLOCKED)]  # fmt: skip
    for nm, sel in groups:
        ks = [k for k in eps if sel(k) and all(k in rows[a] for a in ("A", "RD", "SD"))]
        for st in ("pregrasp", "entry"):
            line = f"  {nm:21s} {st:8s} (n {len(ks)}): miss " + ", ".join(f"{a} {med([stage(rows[a][k], st).get('err_cm') for k in ks]):.2f}"
                                                                         for a in ("A", "RD", "SD"))  # fmt: skip
            line += " | D's error " + ", ".join(
                f"{a} {med([s.get('pred_err_palm_cm') for k, lg in logs[a].items() if sel(k) for s in lg['stages_correct'] if s['stage'] == st]):.2f}"
                for a in ("AL", "RD", "SD"))  # fmt: skip
            print(line)
    clr = [k for k in rows["RD"] if k[0] == "TabletopGraspMP" and k[1] % 10 in BLOCKED and (stage(rows["RD"][k], "pregrasp").get("err_cm") or 99) < 5]
    print(f"  TabletopGrasp blocked: R_D pregrasps within 5 cm (cleared the edge) {len(clr)} / 12 {sorted(e for _, e in clr)}; "
          f"A {sum((stage(rows['A'][k], 'pregrasp').get('err_cm') or 99) < 5 for k in rows['A'] if k[0] == 'TabletopGraspMP' and k[1] % 10 in BLOCKED)} / 12")  # fmt: skip

    # 7. F1
    print("\n== 7. F1: is D still trustworthy on the commands R_D / S_D sent (AL: A's commands, the reference)")
    rg = json.loads(g.range.read_text())["sets"] if g.range else {}
    f1 = {}
    for arm in ("AL", "RD", "SD", "JL", "JD", "JDR"):
        R = [x for x in rg.get(arm, []) if x["kept"]]
        if not R:
            continue
        for nm, sel in [(SHORT[t], (lambda t_: lambda x: x["task"] == t_)(t)) for t in TASKS] + [
            ("TabletopGrasp free", lambda x: x["task"] == "TabletopGraspMP" and x["index"] not in BLOCKED),
            ("decision set 45 s", lambda x: x["task"] in dsets["45 s"]), ("all", lambda x: True)]:  # fmt: skip
            S = [x for x in R if sel(x)]
            if not S:
                continue
            e = [x["err_cm"] for x in S]
            ci = boot_med(e, [(x["task"], x["index"]) for x in S], rng)
            inr = float(np.mean([x["in_range"] for x in S]))
            k_ = (arm, nm)
            lgk = logs.get(arm, {})
            ik = [s.get("ik13_cm") for kk, lg in lgk.items() if sel(dict(task=kk[0], index=kk[1] % 10))
                  for s in lg["stages_correct"] if s.get("ik13_cm") is not None]  # fmt: skip
            ik2 = float(np.mean([v > 2.0 for v in ik])) if ik else float("nan")
            f1[k_] = dict(n=len(S), med=float(np.median(e)), ci=ci, ratio_val=float(np.median(e)) / VAL, ratio_tiid=float(np.median(e)) / TIID,
                          in_range=inr, ik2=ik2)  # fmt: skip
            print(f"  {arm:3s} {nm:18s}: n {len(S):3d}, D median {np.median(e):.3f} cm [{ci[0]:.3f}, {ci[1]:.3f}] = {np.median(e) / VAL:.2f}x SIMPLE val "
                  f"[{ci[0] / VAL:.2f}, {ci[1] / VAL:.2f}], {np.median(e) / TIID:.2f}x test_iid task-walk; in range {100 * inr:.1f} %; "
                  f"IK > 2 cm {100 * ik2:.1f} %")  # fmt: skip
    for arm in ("RD", "SD"):
        for nm in [SHORT[t] for t in TASKS] + ["decision set 45 s", "all"]:
            o, r0 = f1.get((arm, nm)), f1.get(("AL", nm))
            if not o:
                continue
            bad = [w for w, c in (("ratio > 1.3", o["ratio_val"] > 1.3), ("in range < 80 %", o["in_range"] < 0.8),
                                  ("IK > 2 cm share up >= 10 pp", r0 is not None and o["ik2"] - r0["ik2"] >= 0.10)) if c]  # fmt: skip
            print(f"  F1 {arm} {nm:18s}: {'NOT trustworthy (' + ', '.join(bad) + ')' if bad else 'trustworthy'}")
    out["f1"] = {f"{a}|{n}": v for (a, n), v in f1.items()}

    # 8. S's pregrasps with IK > 3 cm; R's shrink
    print("\n== 8. the commands")
    for arm in ("RD", "SD"):
        for st in ("pregrasp", "entry"):
            S = [s for lg in logs[arm].values() for s in lg["stages_correct"] if s["stage"] == st]
            ik = [s.get("ik13_cm") for s in S if s.get("ik13_cm") is not None]
            sh = [x for s in S for x in (s.get("shrink") or [])]
            print(f"  {arm} {st:8s}: n {len(S)}, sent IK > 3 cm {sum(v > 3 for v in ik)} ({100 * np.mean([v > 3 for v in ik]) if ik else float('nan'):.1f} %), "
                  f"> 2 cm {sum(v > 2 for v in ik)}; correction median {med([s.get('corr_cm') for s in S]):.2f} cm, max {max([s.get('corr_cm') or 0 for s in S] or [0]):.2f}; "
                  f"shrunk candidates {sum(x['shrunk'] for x in sh)} of {len(sh)}; fallbacks {sum(bool(s.get('model_fallback')) for s in S)}; "
                  f"plan_same {sum(s.get('plan_same') is True for s in S)}; think median {med([s.get('think_s') for s in S]):.1f} s, max {max([s.get('think_s') or 0 for s in S] or [0]):.1f}")  # fmt: skip

    # 9. the route
    pk = g.run / "P" / "pick.jsonl"
    if pk.exists():
        P = [json.loads(x) for x in open(pk)]
        for arm in ("JD", "J0"):
            Q = [x for x in P if x["arm"] == arm]
            sc = defaultdict(int)
            for x in Q:
                sc[x["scale"]] += 1
            print(f"  route {arm}: {len(Q)} episodes, plans by scale {dict(sc)}, same as J's plan {sum(bool(x.get('same_as_J')) for x in Q)}")
        sh = [json.loads(x) for x in open(g.run / "P" / "shifts.jsonl")]
        print(f"  route shifts (D's -(predicted palm - goal) of J's path): median {med([x['shift_cm'] for x in sh]):.2f} cm, max {max([x['shift_cm'] for x in sh] or [0]):.2f}")
        for a in ("J", "JL", "J0", "JD", "JDR"):
            print(f"  {a:3s} pregrasp / entry miss median {med([stage(rows[a][k], 'pregrasp').get('err_cm') for k in rows[a] if k in rows['JD']]):.2f}"
                  f" / {med([stage(rows[a][k], 'entry').get('err_cm') for k in rows[a] if k in rows['JD']]):.2f} cm (routed)"
                  + (f"; D's error at the pregrasp's end {med([s.get('pred_err_palm_cm') for k, lg in logs[a].items() if k in rows['JD'] for s in lg['stages_correct'] if s['stage'] == 'pregrasp']):.2f}"
                     if a in logs else ""))  # fmt: skip
    if g.json:
        g.json.write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
