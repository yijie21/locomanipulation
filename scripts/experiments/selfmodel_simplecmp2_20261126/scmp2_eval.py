"""selfmodel_simplecmp2_20261126: the registered evaluation; read-only.

    python3 scmp2_eval.py --run <this run> --cmp <selfmodel_simplecmp_20261126 run> --main <main metric bench dir>
        --range <scmp_range.py json> [--json F]

The statistics are selfmodel_simplecmp_20261126's (scmp_eval.py, imported): a layout is (task, index); the layout sign
test (per layout the successes over its episodes, + / - where the two arms differ, exact two-sided binomial), the
episode McNemar test beside it, the success difference's 95 % interval from 4000 resamples of the layouts (seed 0).

E1 (BendPick, stance jittered by up to 6 cm, --jitter-rep 1 and 2): A = AJ1 + AJ2 (60 episodes, key ep + 100 rep),
AL = ALJ (dryR, must equal AJ), R_D = RDJ. A layout is the index (10 layouts, 6 episodes each: 3 levels x 2 reps).
E2 (BendPick, no jitter): A = the main metric's A, R_D = selfmodel_simplecmp's RD, RE (only the entry corrected), RP
(only the pregrasp). V1 (TabletopGrasp's 28 routed episodes): J0 and J_D = selfmodel_simplecmp's J0 / JD, J0S (J0's
paths at J_D's durations), JDF (J_D's paths at J0's durations).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "selfmodel_simplecmp_20261126"))
import scmp_eval as SE  # noqa: E402

VAL, TIID = SE.VAL, SE.TIID
LIMITS = SE.LIMITS
BP, TG = "BendPickMP", "TabletopGraspMP"


def rows_rep(d: Path, arm: str, reps=(1, 2)) -> dict:
    """<arm><k>_<task>.jsonl for k in reps, keyed (task, ep + 100 k)."""
    out = {}
    for k in reps:
        for (t, e), r in SE.rows_of(d, f"{arm}{k}").items():
            assert int(r.get("jitter_rep", -1)) == k, (arm, k, t, e)
            out[(t, e + 100 * k)] = r
    return out


def logs_rep(run: Path, arm: str, reps=(1, 2)) -> dict:
    out = {}
    for k in reps:
        for (t, e), g in SE.corr_logs(run, f"{arm}{k}").items():
            out[(t, e + 100 * k)] = g
    return out


def count(R: dict, keys, col) -> int:
    return sum(bool(R[k].get(col)) for k in keys if k in R)


def miss(R: dict, keys, st) -> float:
    return SE.med([SE.stage(R[k], st).get("err_cm") for k in keys if k in R])


def same_stage(a: dict, b: dict) -> bool:
    return {k: v for k, v in a.items() if k != "correct"} == {k: v for k, v in b.items() if k != "correct"}


def f1_of(rg: dict, arms: list, logs: dict, task: str, rng) -> dict | None:
    S = [x for a in arms for x in rg.get(a, []) if x["kept"] and x["task"] == task]
    if not S:
        return None
    e = [x["err_cm"] for x in S]
    ci = SE.boot_med(e, [(x["task"], x["index"]) for x in S], rng)
    ik = [s.get("ik13_cm") for g in logs.values() if g["task"] == task for s in g["stages_correct"] if s.get("ik13_cm") is not None]
    return dict(n=len(S), med=float(np.median(e)), ci=ci, in_range=float(np.mean([x["in_range"] for x in S])),
                ik2=float(np.mean([v > 2.0 for v in ik])) if ik else float("nan"))  # fmt: skip


def fmt_f1(nm: str, o: dict) -> str:
    return (f"  {nm:5s}: n {o['n']:3d}, D median {o['med']:.3f} cm [{o['ci'][0]:.3f}, {o['ci'][1]:.3f}] = {o['med'] / VAL:.2f}x SIMPLE val "
            f"[{o['ci'][0] / VAL:.2f}, {o['ci'][1] / VAL:.2f}], {o['med'] / TIID:.2f}x test_iid task-walk; in range {100 * o['in_range']:.1f} %; "
            f"IK > 2 cm {100 * o['ik2']:.1f} %")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--cmp", type=Path, required=True)
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--range", type=Path, required=True)
    ap.add_argument("--json", type=Path, default=None)
    g = ap.parse_args()
    rng = np.random.default_rng(0)
    rg = json.loads(g.range.read_text())["sets"]
    out: dict = {}
    R = dict(AJ=rows_rep(g.run / "rows", "AJ"), ALJ=rows_rep(g.run / "rows", "ALJ"), RDJ=rows_rep(g.run / "rows", "RDJ"),
             A=SE.rows_of(g.main / "rows", "A"), RD=SE.rows_of(g.cmp / "rows", "RD"), RE=SE.rows_of(g.run / "rows", "RE"),
             RP=SE.rows_of(g.run / "rows", "RP"), J0=SE.rows_of(g.cmp / "rows", "J0"), JD=SE.rows_of(g.cmp / "rows", "JD"),
             J0S=SE.rows_of(g.run / "rows", "J0S"), JDF=SE.rows_of(g.run / "rows", "JDF"))  # fmt: skip
    L = dict(ALJ=logs_rep(g.run, "ALJ"), RDJ=logs_rep(g.run, "RDJ"), RE=SE.corr_logs(g.run, "RE"), RP=SE.corr_logs(g.run, "RP"),
             J0S=SE.corr_logs(g.run, "J0S"), JDF=SE.corr_logs(g.run, "JDF"))  # fmt: skip
    print("== rows: " + ", ".join(f"{a} {len(r)}" for a, r in R.items()))
    e1 = [(BP, e + 100 * k) for k in (1, 2) for e in range(30)]
    e2 = [(BP, e) for e in range(30)]
    v1 = sorted(k for k in R["J0"] if k[0] == TG)

    # 1. consistency
    print("\n== 1. consistency")
    for k in (1, 2):
        ks = [x for x in e1 if x[1] // 100 == k and x in R["AJ"] and x in R["ALJ"]]
        same = [x for x in ks if SE.clean(R["AJ"][x]) == SE.clean(R["ALJ"][x])]
        print(f"  E1 rep {k}: AJ = ALJ (every row and stage key but wall_s, --correct's fields, sonic_vr_frame): {len(same)} / {len(ks)}"
              + ("" if len(same) == len(ks) else f"; differ: {sorted(set(ks) - set(same))[:10]}"))  # fmt: skip
        out[f"same_AJ_ALJ_{k}"] = [len(same), len(ks)]
    jit_ok = [x for x in e1 if all(x in R[a] for a in ("AJ", "ALJ", "RDJ"))
              and R["AJ"][x].get("jitter") == R["ALJ"][x].get("jitter") == R["RDJ"][x].get("jitter") is not None]  # fmt: skip
    print(f"  E1: the same stance jitter in AJ, ALJ and RDJ: {len(jit_ok)} / {len(e1)}; jitter |ahead|, |left| max "
          f"{max(abs(R['AJ'][x]['jitter'][0]) for x in jit_ok) if jit_ok else float('nan'):.4f}, "
          f"{max(abs(R['AJ'][x]['jitter'][1]) for x in jit_ok) if jit_ok else float('nan'):.4f} m")  # fmt: skip
    re_ok = [k for k in e2 if k in R["RE"] and k in R["A"] and same_stage(SE.stage(R["RE"][k], "pregrasp"), SE.stage(R["A"][k], "pregrasp"))]
    rp_ok = [k for k in e2 if k in R["RP"] and k in R["RD"] and same_stage(SE.stage(R["RP"][k], "pregrasp"), SE.stage(R["RD"][k], "pregrasp"))]
    print(f"  E2: RE's pregrasp = A's (every stage key but correct): {len(re_ok)} / {len(e2)}; RP's pregrasp = R_D's: {len(rp_ok)} / {len(e2)}")
    for a, b, c in (("J0S", "J0", "JD"), ("JDF", "JD", "J0")):
        ks = [k for k in v1 if k in R[a]]
        p_same = sum(SE.stage(R[a][k], "pregrasp").get("joint_path_cm") == SE.stage(R[b][k], "pregrasp").get("joint_path_cm")
                     and SE.stage(R[a][k], "pregrasp").get("S0") == SE.stage(R[b][k], "pregrasp").get("S0") for k in ks)  # fmt: skip
        d = [abs(SE.stage(R[a][k], "pregrasp")["correct"]["dur_s"] - SE.stage(R[c][k], "pregrasp")["correct"]["dur_s"]) for k in ks]
        print(f"  V1 {a}: path and the command's joints = {b}'s {p_same} / {len(ks)}; pregrasp duration (correct.dur_s) - {c}'s: "
              f"max |diff| {max(d) if d else float('nan'):.5f} s ({sum(x <= 1.01e-4 for x in d)} / {len(ks)} within the record's 1e-4 s)")  # fmt: skip
    for a in ("RDJ", "RE", "RP"):
        S = [s for lg in L[a].values() if lg["task"] == BP for s in lg["stages_correct"] if s.get("rule") in ("R", "S")]
        print(f"  {a}: corrected stages {len(S)}, plan_same {sum(s.get('plan_same') is True for s in S)}, "
              f"fallbacks {sum(bool(s.get('model_fallback')) for s in S)}")  # fmt: skip

    # 2. E1
    print("\n== 2. E1: BendPick, stance jittered (up to 6 cm), A vs R_D, 60 episodes (2 reps x 30); a layout = the index")
    for lab, col in LIMITS:
        for k in (1, 2):
            ks = [x for x in e1 if x[1] // 100 == k]
            print(f"  {lab:8s} rep {k}: A {count(R['AJ'], ks, col)}, AL {count(R['ALJ'], ks, col)}, R_D {count(R['RDJ'], ks, col)} (of 30)")
        print(f"  {lab:8s} both : A {count(R['AJ'], e1, col)}, AL {count(R['ALJ'], e1, col)}, R_D {count(R['RDJ'], e1, col)} (of 60)")
    res = {}
    for lab, col in LIMITS:
        o = SE.paired(R["AJ"], R["RDJ"], e1, col, rng)
        res[f"E1 R_D vs A | {lab}"] = o
        print(f"  E1 R_D vs A {lab:8s}: {SE.fmt_pair(o)}")
    for k in (1, 2):
        o = SE.paired(R["AJ"], R["RDJ"], [x for x in e1 if x[1] // 100 == k], "success_45", rng)
        print(f"  E1 R_D vs A 45 s rep {k}: {SE.fmt_pair(o)}")
    ks = [x for x in e1 if x in R["AJ"] and x in R["RDJ"]]
    for st in ("pregrasp", "entry"):
        print(f"  {st:8s} palm miss median (n {len(ks)}): A {miss(R['AJ'], ks, st):.2f}, R_D {miss(R['RDJ'], ks, st):.2f} cm | "
              f"A's failures {miss(R['AJ'], [x for x in ks if not R['AJ'][x].get('success_45')], st):.2f}, "
              f"A's successes {miss(R['AJ'], [x for x in ks if R['AJ'][x].get('success_45')], st):.2f} cm | D's error of the sent command "
              f"AL {SE.med([s.get('pred_err_palm_cm') for lg in L['ALJ'].values() if lg['task'] == BP for s in lg['stages_correct'] if s['stage'] == st]):.2f}, "
              f"R_D {SE.med([s.get('pred_err_palm_cm') for lg in L['RDJ'].values() if lg['task'] == BP for s in lg['stages_correct'] if s['stage'] == st]):.2f} cm")  # fmt: skip
    f1 = {nm: f1_of(rg, [f"{nm}1", f"{nm}2"], L[nm], BP, rng) for nm in ("ALJ", "RDJ")}
    print("  F1 on the jittered starts (BendPick, both reps; D's error at the task label, sd_range's in-range, the sent commands' IK):")
    for nm in ("ALJ", "RDJ"):
        if f1[nm]:
            print(fmt_f1(nm, f1[nm]))
    o, r0 = f1["RDJ"], f1["ALJ"]
    bad = [w for w, c in (("ratio > 1.3", o and o["med"] / VAL > 1.3), ("in range < 80 %", o and o["in_range"] < 0.8),
                          ("IK > 2 cm share up >= 10 pp", o and r0 and o["ik2"] - r0["ik2"] >= 0.10)) if c]  # fmt: skip
    f1_ok = bool(o) and not bad
    print(f"  F1 RDJ BendPick: {'trustworthy' if f1_ok else 'NOT trustworthy (' + ', '.join(bad) + ')'}")
    o45 = res["E1 R_D vs A | 45 s"]
    a45 = o45.get("x", 0)
    d45 = o45.get("y", 0) - a45
    ent = miss(R["RDJ"], ks, "entry")
    lp, lm = o45.get("layouts_plus", 0), o45.get("layouts_minus", 0)
    ref = [w for w, c in (("R_D - A < 0", d45 < 0), ("R_D - A = 0 with room (A < 54)", d45 == 0 and a45 < 54),
                          ("layouts the other way", lm > lp), ("R_D entry miss median > 2.8 cm", ent > 2.8)) if c]  # fmt: skip
    if ref:
        v = "REFUTED (" + ", ".join(ref) + ")"
    elif a45 >= 54:
        v = "INSUFFICIENT (A succeeds in >= 54 / 60: the holding line cannot be reached; not support)"
    elif d45 >= 5 and o45.get("p_sign", 1) < 0.05 and lp > lm and f1_ok:
        v = "HOLDS (R_D - A >= +5, layout sign test p < 0.05, F1 trustworthy)"
    else:
        v = "NEITHER (" + ", ".join(w for w, c in (("R_D - A < +5", d45 < 5), ("p >= 0.05", o45.get("p_sign", 1) >= 0.05),
                                                       ("F1 not trustworthy", not f1_ok)) if c) + ")"  # fmt: skip
    print(f"  E1 verdict (45 s): R_D - A = {d45:+d}, A {a45} / 60, R_D entry miss median {ent:.2f} cm -> {v}")
    out["E1"] = dict(verdict=v, d45=d45, a45=a45, entry=ent, f1=f1, f1_ok=f1_ok)

    # 3. E2
    print("\n== 3. E2: BendPick, no jitter: only the entry corrected (RE), only the pregrasp (RP); A and R_D from before")
    for lab, col in LIMITS:
        print(f"  {lab:8s}: A {count(R['A'], e2, col)}, R_D {count(R['RD'], e2, col)}, RE {count(R['RE'], e2, col)}, RP {count(R['RP'], e2, col)} (of 30)")
    for nm, a, b in (("RE vs A", "A", "RE"), ("RP vs A", "A", "RP"), ("RE vs R_D", "RD", "RE"), ("RP vs R_D", "RD", "RP")):
        o = SE.paired(R[a], R[b], e2, "success_45", rng)
        res[f"E2 {nm} | 45 s"] = o
        print(f"  E2 {nm:9s} 45 s: {SE.fmt_pair(o)}")
    ks2 = [k for k in e2 if all(k in R[a] for a in ("A", "RD", "RE", "RP"))]
    for st in ("pregrasp", "entry"):
        print(f"  {st:8s} palm miss median (n {len(ks2)}): " + ", ".join(f"{a} {miss(R[a], ks2, st):.2f}" for a in ("A", "RD", "RE", "RP")) + " cm")
    dre = count(R["RE"], e2, "success_45") - count(R["A"], e2, "success_45")
    drp = count(R["RP"], e2, "success_45") - count(R["A"], e2, "success_45")
    v2 = ("the entry's miss explains it (RE - A >= +14)" if dre >= 14 else
          "the entry's miss does NOT explain it (RE - A <= +10)" if dre <= 10 else "partly (RE - A +11 to +13)")  # fmt: skip
    print(f"  E2 verdict (45 s): RE - A = {dre:+d}, RP - A = {drp:+d} -> {v2}" + ("; the pregrasp alone recovers most (RP - A >= +14)" if drp >= 14 else ""))
    out["E2"] = dict(dre=dre, drp=drp, verdict=v2)

    # 4. V1
    print("\n== 4. V1: TabletopGrasp's routed episodes, the paths of one arm at the other's pregrasp durations")
    for lab, col in LIMITS:
        print(f"  {lab:8s}: J0 {count(R['J0'], v1, col)}, J_D {count(R['JD'], v1, col)}, J0S {count(R['J0S'], v1, col)}, JDF {count(R['JDF'], v1, col)} (of {len(v1)})")
    for nm, a, b in (("J0S vs J0", "J0", "J0S"), ("JDF vs J_D", "JD", "JDF"), ("JDF vs J0", "J0", "JDF"), ("J0S vs J_D", "JD", "J0S")):
        for lab, col in LIMITS:
            o = SE.paired(R[a], R[b], v1, col, rng)
            res[f"V1 {nm} | {lab}"] = o
            print(f"  V1 {nm:10s} {lab:8s}: {SE.fmt_pair(o)}")
    ksv = [k for k in v1 if all(k in R[a] for a in ("J0", "JD", "J0S", "JDF"))]
    for st in ("pregrasp", "entry"):
        print(f"  {st:8s} palm miss median (n {len(ksv)}): " + ", ".join(f"{a} {miss(R[a], ksv, st):.2f}" for a in ("J0", "JD", "J0S", "JDF")) + " cm")
    print("  pregrasp duration median: " + ", ".join(f"{a} {SE.med([SE.stage(R[a][k], 'pregrasp').get('joint_dur_s') for k in ksv]):.2f}" for a in ("J0", "JD", "J0S", "JDF"))
          + " s; t_success median: " + ", ".join(f"{a} {SE.med([R[a][k].get('t_success') for k in ksv]):.2f}" for a in ("J0", "JD", "J0S", "JDF")) + " s")  # fmt: skip
    for a in ("J0", "JD", "J0S", "JDF"):
        late = [k[1] for k in ksv if R[a][k].get("success_45") and not R[a][k].get("success_official")]
        print(f"  {a:3s}: within 45 s but after the 16.0 s limit {len(late)} {late}")
    j0s = count(R["J0S"], v1, "success_official")
    jdf = count(R["JDF"], v1, "success_official")
    conf = miss(R["JDF"], ksv, "pregrasp") - miss(R["JD"], ksv, "pregrasp")
    v3 = ("the duration explains it (J0S <= 18 and JDF >= 21)" if j0s <= 18 and jdf >= 21 else
          "the duration does NOT explain it (J0S >= 21 or JDF <= 18)" if j0s >= 21 or jdf <= 18 else "partly")  # fmt: skip
    print(f"  V1 verdict (official): J0S {j0s}, JDF {jdf} -> {v3}; JDF's pregrasp miss - J_D's {conf:+.2f} cm"
          + (" (> +0.5: the faster path also lands worse)" if conf > 0.5 else ""))  # fmt: skip
    out["V1"] = dict(j0s=j0s, jdf=jdf, verdict=v3, conf=conf)
    out["tests"] = res

    # 5. the commands
    print("\n== 5. the commands")
    for a in ("RDJ", "RE", "RP"):
        for st in ("pregrasp", "entry"):
            S = [s for lg in L[a].values() if lg["task"] == BP for s in lg["stages_correct"] if s["stage"] == st and s.get("rule") in ("R", "S")]
            if not S:
                continue
            ik = [s.get("ik13_cm") for s in S if s.get("ik13_cm") is not None]
            sh = [x for s in S for x in (s.get("shrink") or [])]
            print(f"  {a:3s} {st:8s}: n {len(S)}, sent IK > 2 cm {sum(v > 2 for v in ik)}; correction median {SE.med([s.get('corr_cm') for s in S]):.2f} cm, "
                  f"max {max([s.get('corr_cm') or 0 for s in S] or [0]):.2f}; shrunk {sum(x['shrunk'] for x in sh)} of {len(sh)}; "
                  f"think median {SE.med([s.get('think_s') for s in S]):.1f} s")  # fmt: skip
    if g.json:
        g.json.write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
