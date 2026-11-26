"""selfmodel_simplesonicdata_20261126: the registered evaluation; read only (selfmodel_simpleheft_20261126's sh_eval.py
with its revision 1, for D_S: the gate files are ssd_gate.sh's, F1(b) is sd_range's three checks from ssd_f1.sh's
range.json, the look is compared with the main metric's SONIC A, GPU nets, as a description).

    python3 ssd_eval.py --run <run dir> --main <the main metric's SONIC bench dir> [--gate <gate dir>] [--f1 <f1 dir>]
        [--json F]

The statistics and the decision set are selfmodel_simplecmp_20261126's (scmp_eval.py, imported): a layout is (task,
index), 40 layouts; the layout sign test (per layout the successes over its three levels, + / - where the two arms
differ, exact two-sided binomial), the episode McNemar test beside it, the success difference's 95 % interval from 4000
resamples of the layouts (seed 0); a task is out of a limit's decision set when over half its episodes fail before the
arms split (pre-split) or when A >= 24 / 30 there (ceiling).

Arms (<run>/rows): A (direct), AL (dryR: must equal A), RD (R_D; absent before phase R). Printed: 1 the consistency (AL =
A; the look: cloud, score, seen_cm of every A row against the main metric's SONIC A row of the episode, a description: those
ran the nets on the GPU); 2 the
pre-split failures; 3 success per task and limit; 4 the decision set; 5 the paired tests (P1 R_D vs A at 45 s on the
decision set, the official limit, per task, all four; revision 1: the four tasks together as the main comparison and the
gate-passed tasks together beside it, each task with its gate verdict from --gate, ssd_gate.sh's rule); 6 the mechanism (each stage's palm miss for A and R_D, D_S's
error of the sent command, the reaches whose wrist or hand touched a table, AL's model / direct from --gate's
err_D_S.json); 7 F1 on R_D per task from --f1 (sh_err.py's err_RD.json, scmp_range.py's range.json, ss_f1pre.py's
f1pre_RD.json; (a) ratio > 1.3, (b) sd_range's three checks' share < 80 %, F1-0 beside, (c) the sent commands' IK > 2 cm
share >= AL's + 10 pp); 8 the commands (correction, shrink, plan_same, think time).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "selfmodel_simplecmp_20261126"))
import scmp_eval as SE  # noqa: E402

TASKS, SHORT, LIMITS = SE.TASKS, SE.SHORT, SE.LIMITS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--gate", type=Path, default=None)
    ap.add_argument("--f1", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    g = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = {arm: SE.rows_of(g.run / "rows", arm) for arm in ("A", "AL", "RD")}
    rows["S"] = SE.rows_of(g.main / "rows", "A")  # the main metric's SONIC A (GPU nets): the look only, a description
    logs = {arm: SE.corr_logs(g.run, arm) for arm in ("AL", "RD")}
    eps = [(t, e) for t in TASKS for e in range(30)]
    have_rd = len(rows["RD"]) > 0
    out: dict = {}
    print("== rows: " + ", ".join(f"{a} {len(r)}" for a, r in rows.items()))

    print("\n== 1. consistency")
    ks = [k for k in rows["A"] if k in rows["AL"]]
    same = [k for k in ks if SE.clean(rows["A"][k]) == SE.clean(rows["AL"][k])]
    print(f"  A = AL: {len(same)} / {len(ks)} episodes identical (every row and stage key but wall_s, --correct's fields and "
          f"the option sonic_vr_frame)" + ("" if len(same) == len(ks) else f"; differ: {sorted(set(ks) - set(same))[:12]}"))  # fmt: skip
    out["same_A_AL"] = [len(same), len(ks)]
    look = [k for k in rows["A"] if k in rows["S"] and all(rows["A"][k].get(c) == rows["S"][k].get(c) for c in ("cloud", "score", "seen_cm"))]
    print(f"  the look (cloud, score, seen_cm) = the main metric's SONIC A (GPU nets; a description): {len(look)} / {sum(k in rows['S'] for k in rows['A'])}")
    out["same_look"] = [len(look), sum(k in rows["S"] for k in rows["A"])]
    for arm in ("AL", "RD"):
        S = [s for lg in logs[arm].values() for s in lg["stages_correct"]]
        if S:
            print(f"  {arm}: {len(logs[arm])} episodes logged, {len(S)} stages, plan_same {sum(s.get('plan_same') is True for s in S)}")

    print("\n== 2. failures before the arms split (A's rows), per task and class")
    excl_pre = set()
    for t in TASKS:
        cls = defaultdict(int)
        for e in range(30):
            r = rows["A"].get((t, e))
            c = "no row" if r is None else SE.pre_split(r)
            if c:
                cls[c] += 1
        n = sum(cls.values())
        if n > 15:
            excl_pre.add(t)
        print(f"  {SHORT[t]:14s}: {n} ({dict(cls) if cls else 'none'})" + (" -> excluded (over half)" if n > 15 else ""))

    print("\n== 3. success per task")
    for lab, col in LIMITS:
        print(f"  -- {lab}")
        for t in TASKS + ("all",):
            sel = [k for k in eps if t == "all" or k[0] == t]
            print(f"  {SHORT.get(t, t):14s}: " + ", ".join(f"{a} {sum(bool(rows[a].get(k, {}).get(col)) for k in sel)}" for a in ("A", "AL", "RD"))
                  + f" (of {len(sel)})")  # fmt: skip

    dsets = {}
    print("\n== 4. decision set (pre-split failures over half out; A >= 24 / 30 'ceiling' out)")
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

    gate = {}
    if g.gate and (g.gate / "f1pre_D_S.json").exists() and (g.gate / "err_D_S.json").exists():
        f0 = json.loads((g.gate / "f1pre_D_S.json").read_text())["cells"]
        fa = json.loads((g.gate / "err_D_S.json").read_text())["sets"]["AL"]["rows"]
        for t in TASKS:  # ssd_gate.sh's verdict
            a_, b_ = f0.get(f"{t}/both"), fa.get(f"{t}/both")
            gate[t] = bool(a_ and b_ and a_["share"] >= 0.80 and b_["ratio"] <= 1.3)
        print("\n== gate (ssd_gate.sh's rule): " + ", ".join(f"{SHORT[t]} {'PASS' if gate[t] else 'FAIL'}" for t in TASKS))
    out["gate"] = gate

    if have_rd:
        print("\n== 5. paired tests (R_D vs A); revision 1: the four tasks together is the main comparison, the gate-passed "
              "tasks together beside it, every task with its gate")  # fmt: skip
        res = {}
        for lab, col in LIMITS:
            o = SE.paired(rows["A"], rows["RD"], eps, col, rng)
            res[f"main | {lab} | all four"] = o
            print(f"  main R_D vs A {lab:8s} all four tasks: {SE.fmt_pair(o)}")
            gp = [t for t in TASKS if gate.get(t)]
            if gp:
                o = SE.paired(rows["A"], rows["RD"], [k for k in eps if k[0] in gp], col, rng)
                res[f"gate-passed | {lab}"] = o
                print(f"       R_D vs A {lab:8s} gate-passed {[SHORT[t] for t in gp]}: {SE.fmt_pair(o)}")
            else:
                print(f"       R_D vs A {lab:8s} gate-passed: no task passed")
        for lab, col in LIMITS:
            ds = [k for k in eps if k[0] in dsets[lab]]
            o = SE.paired(rows["A"], rows["RD"], ds, col, rng)
            res[f"P1 | {lab} | decision set"] = o
            print(f"  {'P1' if lab == '45 s' else '  '} R_D vs A {lab:8s} decision set {[SHORT[t] for t in dsets[lab]]}: {SE.fmt_pair(o)}")
            for t in TASKS:
                o = SE.paired(rows["A"], rows["RD"], [k for k in eps if k[0] == t], col, rng)
                res[f"{lab} | {t}"] = o
                gt = "" if t not in gate else (" [gate PASS]" if gate[t] else " [gate FAIL]")
                print(f"     R_D vs A {lab:8s} {SHORT[t]:14s}{gt}: {SE.fmt_pair(o)}")
            o = SE.paired(rows["A"], rows["RD"], eps, col, rng)
            res[f"{lab} | all"] = o
            print(f"     R_D vs A {lab:8s} {'all four':14s}: {SE.fmt_pair(o)}")
        out["tests"] = res

    print("\n== 6. mechanism: each stage's palm miss (cm, median), D_S's error of the sent command at the stage's end (palm, "
          "cm, median), reaches with a table contact (hits)")  # fmt: skip
    arms = ("A", "RD") if have_rd else ("A",)
    for t in TASKS:
        ks = [k for k in eps if k[0] == t and all(k in rows[a] for a in arms)]
        for st in ("pregrasp", "entry"):
            line = f"  {SHORT[t]:14s} {st:8s} (n {len(ks)}): miss " + ", ".join(f"{a} {SE.med([SE.stage(rows[a][k], st).get('err_cm') for k in ks]):.2f}" for a in arms)
            line += " | D_S's error " + ", ".join(
                f"{a} {SE.med([s.get('pred_err_palm_cm') for k, lg in logs[a].items() if k[0] == t for s in lg['stages_correct'] if s['stage'] == st]):.2f}"
                for a in ("AL",) + (("RD",) if have_rd else ()))  # fmt: skip
            line += " | table contact " + ", ".join(
                f"{a} {sum(any(h[0] == st and 'table' in str(h[2]) for h in (rows[a][k].get('hits') or [])) for k in ks)}" for a in arms)  # fmt: skip
            print(line)
    if g.gate and (g.gate / "err_D_S.json").exists():
        R = json.loads((g.gate / "err_D_S.json").read_text())["sets"]["AL"]["rows"]
        for t in TASKS + ("all",):
            o = R.get(f"{t}/both")
            if o:
                print(f"  AL {SHORT.get(t, t):14s} model / direct (13 cm point, both stages): {o['model_over_direct']:.3f} [{o['mod_ci'][0]:.3f}, {o['mod_ci'][1]:.3f}]; "
                      f"model {o['median_cm']:.2f} cm, direct {o['direct_cm']:.2f} cm")  # fmt: skip

    if have_rd and g.f1:
        print("\n== 7. F1 on R_D's commands (AL's for reference): (a) D_S's error / its val median > 1.3, (b) sd_range's three "
              "checks' share < 80 % (F1-0 beside), (c) the sent commands' IK > 2 cm share >= AL's + 10 pp -> not trustworthy")  # fmt: skip
        ea = {a: json.loads((g.f1 / f"err_{a}.json").read_text())["sets"][a]["rows"] for a in ("AL", "RD") if (g.f1 / f"err_{a}.json").exists()}
        fr = {a: json.loads((g.f1 / f"f1pre_{a}.json").read_text())["cells"] for a in ("AL", "RD") if (g.f1 / f"f1pre_{a}.json").exists()}
        rg = json.loads((g.f1 / "range.json").read_text())["sets"] if (g.f1 / "range.json").exists() else {}
        f1 = {}
        for t in TASKS:
            ik = {}
            for a in ("AL", "RD"):
                v = [s.get("ik13_cm") for k, lg in logs[a].items() if k[0] == t for s in lg["stages_correct"] if s.get("ik13_cm") is not None]
                ik[a] = float(np.mean([x > 2.0 for x in v])) if v else float("nan")
            for a in ("AL", "RD"):
                ra = ea.get(a, {}).get(f"{t}/both", {}).get("ratio", float("nan"))
                s0 = fr.get(a, {}).get(f"{t}/both", {}).get("share", float("nan"))
                S = [x["in_range"] for x in rg.get(a, []) if x["kept"] and x["task"] == t]
                sb = float(np.mean(S)) if S else float("nan")
                bad = [w for w, c in (("(a) ratio > 1.3", ra > 1.3), ("(b) in range < 80 %", not sb >= 0.8),
                                      ("(c) IK > 2 cm up >= 10 pp", a == "RD" and ik["RD"] - ik["AL"] >= 0.10)) if c]  # fmt: skip
                f1[f"{a}|{t}"] = dict(ratio=ra, in_range=sb, f1_0=s0, ik2=ik[a], bad=bad)
                print(f"  {a} {SHORT[t]:14s}: (a) {ra:.3f}x, (b) {100 * sb:5.1f} % (F1-0 {100 * s0:5.1f} %), (c) IK > 2 cm {100 * ik[a]:5.1f} % -> "
                      + ("NOT trustworthy (" + ", ".join(bad) + ")" if bad else "trustworthy"))  # fmt: skip
        out["f1"] = f1

    if have_rd:
        print("\n== 8. the commands")
        for st in ("pregrasp", "entry"):
            S = [s for lg in logs["RD"].values() for s in lg["stages_correct"] if s["stage"] == st]
            ik = [s.get("ik13_cm") for s in S if s.get("ik13_cm") is not None]
            sh = [x for s in S for x in (s.get("shrink") or [])]
            print(f"  RD {st:8s}: n {len(S)}, sent IK > 2 cm {sum(v > 2 for v in ik)}; correction median {SE.med([s.get('corr_cm') for s in S]):.2f} cm, "
                  f"max {max([s.get('corr_cm') or 0 for s in S] or [0]):.2f}; shrunk candidates {sum(x['shrunk'] for x in sh)} of {len(sh)}; "
                  f"fallbacks {sum(bool(s.get('model_fallback')) for s in S)}; plan_same {sum(s.get('plan_same') is True for s in S)}; "
                  f"think median {SE.med([s.get('think_s') for s in S]):.1f} s, max {max([s.get('think_s') or 0 for s in S] or [0]):.1f}")  # fmt: skip
    if g.json:
        g.json.write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
