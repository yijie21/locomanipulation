"""selfmodel_simplecmp2_20261126: the pre-registration checks' comparisons; read-only.

    python3 scmp2_smoke.py --dev <smoke dir, development tree> --test <smoke dir, test tree> --route1 <simplecmp
        checks/route1> --cmp <selfmodel_simplecmp run> --main <main metric bench dir>

Development episodes (TabletopGrasp 30 / 36): Ak plain A; AJk A with the stance jittered (rep 1), ALJk the same with
--correct dryR, RDJk with R; Rk R; REk R on the entry only; RPk R on the pregrasp only; J0k selfmodel_simplecmp's dev J0
plans replayed (dryR, as its checks/route1 J0); J0Ik the same retimed to their own durations (--joint-dur); J0Sk the same
retimed to that check's J_D durations; with --dev2 (the live durations: scmp2_jdur.py's {"plan": ...} values) J0Ik,
J0Sk and JDFk (the J_D plans at J0's durations) come from there. Test episodes, the switches off or as before: A0 (BendPick 0, 1) against the main
metric's A; Jt (TabletopGrasp 0, --avoid <main via> --joint) against the main metric's J; Rt (BendPick 0, R) against
selfmodel_simplecmp's R_D.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "selfmodel_simplecmp_20261126"))
import scmp_cmp as SC  # noqa: E402
import scmp_eval as SE  # noqa: E402

TG, BP = "TabletopGraspMP", "BendPickMP"


def same(x: dict, y: dict, what: str) -> None:
    d = SC.diff(SE.clean(x), SE.clean(y))
    print(f"  {what}: " + ("identical" if not d else f"DIFFER in {len(d)} keys: {d[:12]}"))


def st(r: dict, name: str) -> dict:
    return {k: v for k, v in SE.stage(r, name).items() if k != "correct"}


def corr_stages(r: dict) -> str:
    return ", ".join(f"{s['stage']} {(s.get('correct') or {}).get('rule', '-')}" for s in r["stages"] if s["stage"] in ("pregrasp", "entry"))


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("dev", "test", "route1", "cmp", "main"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    ap.add_argument("--dev2", type=Path, default=None)
    g = ap.parse_args()
    D = {a: SE.rows_of(g.dev / "rows", a) for a in ("Ak", "AJk", "ALJk", "RDJk", "Rk", "REk", "RPk", "J0k", "J0Ik", "J0Sk")}
    if g.dev2:
        D.update({a: SE.rows_of(g.dev2 / "rows", a) for a in ("J0Ik", "J0Sk", "JDFk")})
    T = {a: SE.rows_of(g.test / "rows", a) for a in ("A0", "Jt", "Rt")}
    print("rows: " + ", ".join(f"{a} {len(v)}" for a, v in {**D, **T}.items()))
    k30 = (TG, 30)
    print("\n== stance jitter (dev TabletopGrasp 30, rep 1)")
    for a in ("AJk", "ALJk", "RDJk"):
        r = D[a][k30]
        print(f"  {a}: jitter {r.get('jitter')} rep {r.get('jitter_rep')}; stand {r.get('stand')}; success_45 {r.get('success_45')}; "
              f"pregrasp miss {SE.stage(r, 'pregrasp').get('err_cm')}, entry {SE.stage(r, 'entry').get('err_cm')} cm; --correct: {corr_stages(r)}")  # fmt: skip
    print(f"  Ak (no jitter): stand {D['Ak'][k30].get('stand')}; success_45 {D['Ak'][k30].get('success_45')}")
    same(D["AJk"][k30], D["ALJk"][k30], "AJk = ALJk (dryR records, sends the goal)")
    print("\n== only one stage corrected (dev TabletopGrasp 30)")
    for a in ("Rk", "REk", "RPk"):
        r = D[a][k30]
        print(f"  {a}: --correct: {corr_stages(r)}; pregrasp miss {SE.stage(r, 'pregrasp').get('err_cm')}, entry "
              f"{SE.stage(r, 'entry').get('err_cm')} cm; success_45 {r.get('success_45')}")  # fmt: skip
    d = SC.diff(st(D["REk"][k30], "pregrasp"), st(D["Ak"][k30], "pregrasp"))
    print(f"  REk's pregrasp = Ak's (every stage key but correct): {'identical' if not d else d}")
    d = SC.diff(st(D["RPk"][k30], "pregrasp"), st(D["Rk"][k30], "pregrasp"))
    print(f"  RPk's pregrasp = Rk's: {'identical' if not d else d}")
    d = SC.diff(st(D["RPk"][k30], "entry"), st(D["Rk"][k30], "entry"))
    print(f"  RPk's entry vs Rk's (RPk's not corrected): {'identical' if not d else f'differ in {len(d)} keys'}")
    print("\n== joint paths (dev TabletopGrasp 30 / 36; simplecmp checks/route1's plans)")
    J0, JD = SE.rows_of(g.route1 / "rows", "J0"), SE.rows_of(g.route1 / "rows", "JD")
    for e in (30, 36):
        k = (TG, e)
        same(D["J0k"][k], J0[k], f"ep {e}: J0k = route1 J0 (the replay)")
        for a, ref in (("J0Ik", J0), ("J0Sk", JD), ("JDFk", J0)):
            if k not in D.get(a, {}):
                continue
            p, pr = SE.stage(D[a][k], "pregrasp"), SE.stage(ref[k], "pregrasp")
            print(f"  ep {e} {a}: joint_dur_set {p.get('joint_dur_set')}; correct.dur_s {p['correct'].get('dur_s')} (the asked arm's "
                  f"{pr['correct'].get('dur_s')}); joint_dur_s {p.get('joint_dur_s')}; joint_path_cm {p.get('joint_path_cm')} "
                  f"(J0 {SE.stage(J0[k], 'pregrasp').get('joint_path_cm')}); pregrasp miss {p.get('err_cm')} (J0 "
                  f"{SE.stage(J0[k], 'pregrasp').get('err_cm')}, J_D {SE.stage(JD[k], 'pregrasp').get('err_cm')}); t_success "
                  f"{D[a][k].get('t_success')} (J0 {J0[k].get('t_success')}, J_D {JD[k].get('t_success')})")  # fmt: skip
        d = SC.diff(SE.clean(D["J0Ik"][k]), SE.clean(J0[k]))
        print(f"  ep {e}: J0Ik vs route1 J0: {'identical' if not d else f'differ in {len(d)} keys: {d[:8]}'}")
        if k in D.get("JDFk", {}):
            p, q = SE.stage(D["JDFk"][k], "pregrasp"), SE.stage(JD[k], "pregrasp")
            print(f"  ep {e}: JDFk's path = route1 J_D's: {p.get('joint_path_cm') == q.get('joint_path_cm') and p.get('S0') == q.get('S0')}")
    print("\n== the switches off, test episodes, against earlier rows")
    A, J = SE.rows_of(g.main / "rows", "A"), SE.rows_of(g.main / "rows", "J")
    RD = SE.rows_of(g.cmp / "rows", "RD")
    for k in sorted(T["A0"]):
        same(T["A0"][k], A[k], f"{k[0]} {k[1]}: A0 = the main metric's A")
    for k in sorted(T["Jt"]):
        same(T["Jt"][k], J[k], f"{k[0]} {k[1]}: Jt = the main metric's J")
    for k in sorted(T["Rt"]):
        same(T["Rt"][k], RD[k], f"{k[0]} {k[1]}: Rt = simplecmp's R_D")
        a = [s.get("correct", {}).get("u") for s in T["Rt"][k]["stages"] if s.get("correct")]
        b = [s.get("correct", {}).get("u") for s in RD[k]["stages"] if s.get("correct")]
        print(f"  {k[0]} {k[1]}: Rt's sent commands = R_D's: {a == b}")


if __name__ == "__main__":
    main()
