"""selfmodel_hatask_20261126 revision 2, section 1: ht_inv.py (R's copy run in HA) against lmlcmp3's frozen lc3_run.py, in
LocoManiLab's environment (the interpreter, PYTHONPATH and code copy lc3_formal.sh runs lc3_run with), read-only.

    cd $G && PYTHONPATH=$G:$C0/src CUDA_VISIBLE_DEVICES= $PY ht_equiv.py --lc3 $C0/scripts/experiments/selfmodel_lmlcmp3_20261126 \
        --runs <ht_task out>[,<out>...] --wide '<the HA run's L.WIDE, json>' [--json F]

1. Text. inspect.getsource, byte for byte:
   - ht_inv.shrink == lc3_run.shrink;
   - ht_inv.clip_t == lml_tasks.clip_t (the one lc3_run imports as LT);
   - ht_inv.reachable_one's block (between its "verbatim" marker lines) == lc3_run.episode's block from the line with
     "t, evs = gh.copy(), []" to the line with "best = evs[bi]", both dedented, except one line: lc3_run's
     `t = LT.clip_t(t + (gh - pos), "right")` is `t = LT.clip_t(t + (gh - pos), side)`;
   - the constants: SHRINK_IK, SHRINK_STEPS (lc3_run), ITERS, LAMBDA (lml_tasks), and WIDE (lml_tasks' L.WIDE against
     the HA run's, --wide).
2. Numbers. Every right-hand one-hand R command in the runs' R logs (a group): the goal, each plan's IK error and each
   prediction the HA run recorded, keyed by the candidate's bytes. lc3_run's own shrink and lc3_run.episode's block
   (exec'd from its source, with lml_tasks as LT, method "reachable") run on a driver and a model that answer from
   those records (a candidate the HA run never planned or predicted is a failure). Required equal, bit for bit: every
   candidate (so every shrunk command), every loss, each shrink's record (alpha, shrunk, n_plans, ik0, corr0, corr),
   the number of iterations, the chosen index and the chosen command. REACHWM_FDVEL is off on both sides (lc3_run
   asserts it; the dev runs use LocoManiLab's D, trained with it off: the runs' model flags are checked).
   Left-hand commands (lc3_run has none): each step's update against lml_tasks.clip_t(..., "left") and lc3_run.shrink
   on the same records.
3. Two hands (ht_inv.reachable_two, new code): per step each hand's next candidate equals the one-hand rule
   (lml_tasks.clip_t of t + goal - prediction, then the recorded common shrink fraction), and each loss equals the sum
   of the two one-hand losses, bit for bit.
Prints per group and "EQUIV PASS" / "EQUIV FAIL" (exit 1).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import textwrap
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def block(src: str, first: str, last: str, inclusive: bool) -> list[str]:
    lines = src.splitlines()
    i = next(k for k, s in enumerate(lines) if first in s)
    j = next(k for k in range(i, len(lines)) if last in lines[k])
    sel = lines[i : j + 1] if inclusive else lines[i + 1 : j]
    return textwrap.dedent("\n".join(sel)).splitlines()


def key(v) -> bytes:
    return np.asarray(v, np.float64).tobytes()


class Recorded:
    """A driver (plan -> IK error) and a model (predict -> pos, std) answering from one R command's records."""

    def __init__(self, r: dict, side: str | None):
        self.ik, self.pred, self.miss, self.conflict = {}, {}, [], []
        for q in r["ik_queries"]:
            k = key(q["t"])
            v = (q.get("err") or "LinAlgError") if q["ik"] is None else float(q["ik"])
            if k in self.ik and self.ik[k] != v:
                self.conflict.append(("ik", q["t"]))
            self.ik[k] = v
        for q in r["pred_queries"]:
            k = key(q["t"])
            v = (np.asarray(q["pos"], np.float64), np.asarray(q["std"], np.float64))
            if k in self.pred and not all(np.array_equal(a, b) for a, b in zip(self.pred[k], v)):
                self.conflict.append(("pred", q["t"]))
            self.pred[k] = v

    def plan(self, x, ref, c):
        k = key(c["target"])
        if k not in self.ik:
            self.miss.append(("ik", np.asarray(c["target"]).tolist()))
            raise KeyError("a plan the HA run never made")
        v = self.ik[k]
        if isinstance(v, str):
            raise ValueError(v) if v == "ValueError" else np.linalg.LinAlgError(v)
        return None, None, {"ik_err": v}

    def feat(self, cc):
        pl = self.plan(None, None, cc)
        return pl, {"key": key(cc["target"])}

    def predict(self, z, names=("D",)):
        if z["key"] not in self.pred:
            self.miss.append(("pred", z["key"]))
            raise KeyError("a prediction the HA run never made")
        return {"D": self.pred[z["key"]]}


N_SYN = 200


class SynDr:
    """plan -> ik_err = max(0, |target - centre| - radius) (+ a plan failure inside a small ball, to cover that branch)."""

    def __init__(self, centre, radius, fail_c=None):
        self.c, self.r, self.fail_c = np.asarray(centre, np.float64), float(radius), fail_c

    def err(self, t) -> float:
        t = np.asarray(t, np.float64)
        if self.fail_c is not None and float(np.linalg.norm(t - self.fail_c)) < 0.01:
            raise np.linalg.LinAlgError("synthetic")
        return max(0.0, float(np.linalg.norm(t - self.c)) - self.r)

    def plan(self, x, ref, c):
        return None, None, {"ik_err": self.err(c["target"])}


class SynDr2:
    def __init__(self, dr: SynDr, other: str):
        self.dr, self.other = dr, other

    def plan(self, x, ref, c2):
        e = {s: (self.dr.err(c2[s]["target"]) if s != self.other else 0.0) for s in c2}
        return None, None, {"ik_err": e}


def synthetic(LC, INV) -> dict:
    rng = np.random.default_rng(20261126)
    n1 = n2 = n_sh = 0
    for i in range(N_SYN):
        gh = rng.uniform([0.2, -0.4, 0.6], [0.6, 0.2, 1.2])
        d = rng.normal(size=3)
        t = gh + rng.uniform(0.02, 0.15) * d / np.linalg.norm(d)
        r = rng.uniform(0.0, 0.06)
        fail = (gh + 0.5 * (t - gh)) if i % 10 == 0 else None  # every 10th: the first bisection point fails to plan
        dr = SynDr(gh, r, fail)
        a1, s1 = LC.shrink(dr, None, None, dict(target=gh.copy()), gh, t.copy())
        b1, r1 = INV.shrink(dr, None, None, dict(target=gh.copy()), gh, t.copy())
        same = key(a1) == key(b1) and set(s1) == set(r1) and all(
            (s1[k] == r1[k]) if not isinstance(s1[k], float) else same_f(s1[k], r1[k]) for k in s1)  # fmt: skip
        n1 += bool(same)
        n_sh += bool(s1["shrunk"])
        gl = gh * np.array([1.0, -1.0, 1.0])
        tl = gl + np.array([0.05, -0.03, 0.02])
        t2, s2 = INV.shrink2(SynDr2(dr, "left"), None, None, {"right": dict(target=gh.copy()), "left": dict(target=gl.copy())},
                             {"right": gh, "left": gl}, {"right": t.copy(), "left": tl.copy()})  # fmt: skip
        al = s2["alpha"]
        n2 += bool(key(t2["right"]) == key(b1) and same_f(al, r1["alpha"]) and s2["shrunk"] == r1["shrunk"] and s2["n_plans"] == r1["n_plans"]
                   and key(t2["left"]) == key(gl + al * (tl - gl) if s2["shrunk"] else tl))  # fmt: skip
    return dict(n=N_SYN, one_ok=n1, two_ok=n2, shrunk=n_sh)


def same_f(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return np.float64(a).tobytes() == np.float64(b).tobytes()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lc3", type=Path, required=True, help="lmlcmp3's frozen code dir (lc3_run.py)")
    ap.add_argument("--runs", required=True, help="ht_task output dirs (their R/log/*.json), comma-separated")
    ap.add_argument("--wide", required=True, help="the HA run's L.WIDE (json [[lo], [hi]])")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    assert os.environ.get("REACHWM_FDVEL", "0") == "0"
    sys.path.insert(0, str(a.lc3))
    import lc3_run as LC  # noqa: E402

    LT = LC.LT
    sys.path.insert(0, str(HERE))
    import ht_inv as INV  # noqa: E402

    INV.set_L(LT.L)
    out, ok = dict(files={}), True
    for nm, mod in (("lc3_run", LC), ("lml_tasks", LT), ("ht_inv", INV)):
        out["files"][nm] = dict(path=mod.__file__, md5=hashlib.md5(Path(mod.__file__).read_bytes()).hexdigest()[:8])
    print("files:", json.dumps(out["files"]))
    # 1. text
    t1 = dict(shrink=inspect.getsource(INV.shrink) == inspect.getsource(LC.shrink),
              clip_t=inspect.getsource(INV.clip_t) == inspect.getsource(LT.clip_t))  # fmt: skip
    lb = block(inspect.getsource(LC.episode), "t, evs = gh.copy(), []", "best = evs[bi]", True)
    ib = block(inspect.getsource(INV.reachable_one), "# -- verbatim from lc3_run.episode", "# -- end of the verbatim block --", False)
    allowed = {('t = LT.clip_t(t + (gh - pos), "right")', "t = LT.clip_t(t + (gh - pos), side)")}
    diffs = [(x.strip(), y.strip()) for x, y in zip(lb, ib) if x != y]
    t1["block_lines"] = [len(lb), len(ib)]
    t1["block"] = len(lb) == len(ib) and set(diffs) <= allowed and len(diffs) == 1
    t1["block_diffs"] = diffs
    wide = json.loads(a.wide)
    t1["constants"] = (INV.SHRINK_IK == LC.SHRINK_IK and INV.SHRINK_STEPS == LC.SHRINK_STEPS and INV.ITERS == LT.ITERS
                       and INV.LAMBDA == LT.LAMBDA and all(np.array_equal(np.asarray(w, float), np.asarray(v, float)) for w, v in zip(wide, LT.L.WIDE)))  # fmt: skip
    print("text:", json.dumps(t1))
    ok &= t1["shrink"] and t1["clip_t"] and t1["block"] and t1["constants"]
    out["text"] = t1
    code = compile("\n".join(lb), "<lc3_run.episode block>", "exec")
    # 2. numbers
    groups, left, twos = [], [], []
    for run in a.runs.split(","):
        for lp in sorted(Path(run).glob("R/log/*.json")):
            lg = json.loads(lp.read_text())
            fl = lg.get("model", {}) or {}
            if fl.get("flags", {}).get("FDVEL"):
                print(f"FAIL {lp.name}: the run's model reader had FDVEL on")
                ok = False
            for cm in lg.get("commands", []):
                r = cm.get("r")
                if r is None:
                    continue
                g = dict(ep=lp.stem, label=cm["label"], f_cmd=cm["f_cmd"])
                if "side" in r and r["side"] == "right":
                    gh = np.asarray(r["gh"], np.float64)
                    rec = Recorded(r, "right")
                    ns = dict(np=np, gh=gh, method="reachable", LT=LT, shrink=LC.shrink, dr=rec, rows={0: None}, f=0, frames={0: None},
                              c=dict(target=gh.copy()), feat=rec.feat, MD=rec)  # fmt: skip
                    try:
                        exec(code, ns)
                    except KeyError as e:
                        g.update(ok=False, why=f"replay asked for a record the HA run does not have: {e}")
                        groups.append(g)
                        ok = False
                        continue
                    evs, shr, bi = ns["evs"], ns["shr"], ns["bi"]
                    inv = r["inv"]
                    cand = len(evs) == len(inv) and all(key(e["t"]) == key(v["t"]) for e, v in zip(evs, inv))
                    loss = len(evs) == len(inv) and all(same_f(e["l"], v["l"]) for e, v in zip(evs, inv))
                    shk = len(shr) == len(r["shrink"]) and all(
                        all(same_f(s.get(k), v.get(k)) if k not in ("shrunk", "n_plans", "it") else s.get(k) == v.get(k) for k in set(s) | set(v))
                        for s, v in zip(shr, r["shrink"]))  # fmt: skip
                    pick = bi == r["inv_best"] and ("t_exec" not in r or key(evs[bi]["t"]) == key(r["t_exec"]))
                    g.update(n_iter=[len(evs), r["n_iter"]], candidates=cand, losses=loss, shrink=shk, chosen=pick,
                             shrunk=sum(1 for s in shr if s["shrunk"]), conflicts=len(rec.conflict))  # fmt: skip
                    g["ok"] = cand and loss and shk and pick and len(evs) == r["n_iter"] and not rec.conflict
                    groups.append(g)
                    ok &= g["ok"]
                elif "side" in r:  # left hand: each update against lml_tasks.clip_t(..., "left") + lc3_run.shrink
                    gh = np.asarray(r["gh"], np.float64)
                    rec = Recorded(r, "left")
                    inv, good = r["inv"], True
                    for i in range(len(inv) - 1):
                        if inv[i].get("pos") is None:
                            good = False
                            break
                        nxt = LT.clip_t(np.asarray(inv[i]["t"], np.float64) + (gh - np.asarray(inv[i]["pos"], np.float64)), "left")
                        try:
                            t2, _ = LC.shrink(rec, None, None, dict(target=gh.copy()), gh, nxt)
                        except KeyError:
                            good = False
                            break
                        good &= key(t2) == key(inv[i + 1]["t"])
                    g.update(ok=bool(good), n_iter=r["n_iter"])
                    left.append(g)
                    ok &= bool(good)
                else:  # two hands
                    sides = r["sides"]
                    gh2 = {s: np.asarray(r["gh"][s], np.float64) for s in sides}
                    inv, good = r["inv"], True
                    for i, e in enumerate(inv):
                        if e.get("pos") is None:
                            continue
                        l1 = sum(float(((np.asarray(e["pos"][s]) - gh2[s]) ** 2).sum() + LT.LAMBDA * (np.asarray(e["std"][s]) ** 2).sum()) for s in sides)
                        good &= same_f(l1, e["l"])
                        if i + 1 < len(inv):
                            sh = r["shrink"][i]
                            nxt = {s: LT.clip_t(np.asarray(e["t"][s], np.float64) + (gh2[s] - np.asarray(e["pos"][s], np.float64)), s) for s in sides}
                            al = float(sh["alpha"]) if sh["shrunk"] else None
                            exp = {s: (gh2[s] + al * (nxt[s] - gh2[s])) if al is not None else nxt[s] for s in sides}
                            good &= all(key(exp[s]) == key(inv[i + 1]["t"][s]) for s in sides)
                    g.update(ok=bool(good), n_iter=r["n_iter"])
                    twos.append(g)
                    ok &= bool(good)
    for g in groups:
        print("right:", json.dumps(g))
    for g in left:
        print("left:", json.dumps(g))
    for g in twos:
        print("two-hand:", json.dumps(g))
    n_ok = sum(g["ok"] for g in groups)
    print(f"right-hand groups: {n_ok}/{len(groups)} bit-identical (shrunk at least once in {sum(1 for g in groups if g.get('shrunk'))}); "
          f"left-hand {sum(g['ok'] for g in left)}/{len(left)}; two-hand {sum(g['ok'] for g in twos)}/{len(twos)}")  # fmt: skip
    if len(groups) < 20:
        print(f"FAIL: {len(groups)} right-hand groups (the coordinator asked for >= 20)")
        ok = False
    # 4. the shrink's bisection (synthetic: the dev groups may never shrink): N_SYN seeded cases, an IK error that grows
    # with the distance from a reachable centre near the goal; lc3_run.shrink and ht_inv.shrink bit for bit, and
    # ht_inv.shrink2 with a second hand that is always reachable against ht_inv.shrink on the first
    syn = synthetic(LC, INV)
    print("synthetic shrink:", json.dumps(syn))
    ok &= syn["one_ok"] == syn["n"] and syn["two_ok"] == syn["n"] and syn["shrunk"] >= syn["n"] // 2
    out.update(right=groups, left=left, two=twos, synthetic=syn, ok=bool(ok))
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))
    print("EQUIV", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
