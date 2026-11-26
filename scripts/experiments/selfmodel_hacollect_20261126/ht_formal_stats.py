"""selfmodel_hatask_20261126: the formal runs' report for one controller (registration's metrics and statistics; revision
1: the main metric is grasp-and-lift; revision 2 sections 4 and 6; revision 3's two descriptive items), read-only on the
episode logs and data files. Written before any formal episode was read.

    cd <repo_b>; REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1 LOCO_REPO=<repo_b> PYTHONDONTWRITEBYTECODE=1 \
        $PY ht_formal_stats.py --runs <formal/<ctrl>_ppbox>,<formal/<ctrl>_doubledesk> --touch <hd_touch.py> [--json F] [--structure]

--structure: only the counts and the structural checks (no metric value); used to try the script on the code-check runs.

1. Pairs: (task, seed) with both arms' logs. Episodes that ended in error (revision 2 section 6) are listed and left out.
2. Consistency (the table's row 2): pairs whose prefix digests differ; > 5 % -> the controller's results are not printed.
3. Main metric, grasp-and-lift (the log's lifted_step: the object >= 5 cm over its reset height for >= 50 steps):
   A and R counts, R - A in points, discordant pairs (R only / A only), McNemar exact two-sided p. Per task counts.
   Sensitivity: the pairs with identical prefixes only. Holm across the three controllers is done when all are in
   (--holm "ctrl=p,..." adds the others' raw p).
4. Secondary: success within 3x (HA's reward), success within the official budget; the same statistics, raw p only.
5. Landing error of the commands through the model (label x side, per task): the simulated hand point at the end label
   (data.load's pos_f) to the perceived target of that command (the log's goal; A's command), cm. A and R medians, the
   paired R - A median with a 95 % interval from 1000 seed resamples (seed 0), the pair count. P&PBox's squeeze per hand
   is revision 2 section 4. Frame check: each segment's target equals the log's executed target of that hand.
6. Per arm: how episodes ended, the furthest step, HA falls. Revision 3: DoubleDesk's per-episode max lift (A, R);
   P&PBox episodes stuck at find_shelf (A, R).
7. R's commands through the model: share with a shrink, correction and predicted miss medians. Contacts (hd_touch's
   rule) and pelvis displacement on the reaches through the model, per arm (descriptive).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("LOCO_REPO", "/wuji-vefps-D/wuji-il/yijie/ws/code/loco"))
sys.path.insert(0, str(REPO / "scripts" / "experiments" / "reachwm_20261030"))
import data as DT  # noqa: E402

TASKS = ("ppbox", "doubledesk")
SEARCH = {"ppbox": "search_shelf", "doubledesk": "search_basket"}
PLACE = {"ppbox": "place", "doubledesk": "over_basket"}
NOT_SEEN = {"ppbox": "shelf not seen", "doubledesk": "basket not seen"}
NB = 1000


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def holm(ps: dict) -> dict:
    ks = sorted(ps, key=lambda k: ps[k])
    out, run = {}, 0.0
    for i, k in enumerate(ks):
        run = max(run, min(1.0, (len(ks) - i) * ps[k]))
        out[k] = run
    return out


def how(lg: dict) -> str:
    e, s = str(lg["end"]), str(lg.get("sequence"))
    if e == "success":
        return "success"
    if e == "budget":
        return "timeout"
    if e.startswith("error"):
        return "error"
    if e == "fell":
        return "HA fall"
    if s.startswith("abandoned: ik") or "ik " in s:
        return "plan failure"
    return f"held ({s})"


def furthest(lg: dict) -> str:
    t, ph = lg["task"], {name for _, name in lg["phases"]}
    st = [("grasp", "close" in ph), ("grasp-and-lift", lg.get("lifted_step") is not None), ("search", SEARCH[t] in ph),
          ("carry", "carry" in ph), ("place", PLACE[t] in ph), ("open", "release" in ph)]  # fmt: skip
    f = "before grasp"
    for k, v in st:
        if v:
            f = k
    return f


def touch_of(z, hit, chain: int, side: str):
    """hd_touch's verdict for the record row of this chain and hand (side > 0 is right, as hd_real.side_name)."""
    if z is None or hit is None:
        return None
    j = [k for k in np.flatnonzero(np.asarray(z["chain"]).astype(int) == int(chain)) if (float(z["side"][k]) > 0) == (side == "right")]
    return bool(hit[j[0]]) if j else None


def binary(pairs: list, key) -> dict:
    a = np.array([bool(key(p["A"])) for p in pairs])
    r = np.array([bool(key(p["R"])) for p in pairs])
    b, c = int((r & ~a).sum()), int((a & ~r).sum())
    n = len(pairs)
    return dict(n=n, A=int(a.sum()), R=int(r.sum()), diff_pp=100 * (r.sum() - a.sum()) / n if n else float("nan"),
                R_only=b, A_only=c, p=mcnemar(b, c))  # fmt: skip


def show_bin(tag: str, o: dict) -> str:
    return (f"{tag}: A {o['A']}/{o['n']} ({100 * o['A'] / o['n']:.1f} %), R {o['R']}/{o['n']} ({100 * o['R'] / o['n']:.1f} %), "
            f"R - A {o['diff_pp']:+.1f} points; discordant R only {o['R_only']}, A only {o['A_only']}; McNemar exact p {o['p']:.4g}")  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--touch", type=Path, required=True, help="the frozen hd_touch.py (revision 1's contact rule)")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--structure", action="store_true")
    ap.add_argument("--holm", default="", help="the other controllers' raw main p: ctrl=p,...")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("hd_touch", a.touch)
    TOUCH = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(TOUCH)
    rng = np.random.default_rng(0)
    out = dict(runs=a.runs, flags=dict(FDVEL=bool(DT.FDVEL), GOAL_REF=bool(DT.GOAL_REF), APPROACH=bool(DT.APPROACH), QFIX=bool(DT.QFIX)))
    # 1. pairs
    pairs, errors, missing, ctrl = [], [], [], set()
    for run in a.runs.split(","):
        run = Path(run)
        names = sorted({p.stem for arm in ("A", "R") for p in (run / arm / "log").glob("*.json")})
        for nm in names:
            lg = {arm: json.loads((run / arm / "log" / f"{nm}.json").read_text()) if (run / arm / "log" / f"{nm}.json").exists() else None
                  for arm in ("A", "R")}  # fmt: skip
            if lg["A"] is None or lg["R"] is None:
                missing.append(nm)
                continue
            ctrl.add(lg["A"]["controller"])
            bad = [arm for arm in ("A", "R") if str(lg[arm]["end"]).startswith("error")]
            if bad:
                errors.append(dict(name=nm, arms=bad, ends={arm: lg[arm]["end"] for arm in bad}, phase={arm: lg[arm].get("phase_at_end") for arm in bad}))
                continue
            pairs.append(dict(run=run, name=nm, task=lg["A"]["task"], seed=int(lg["A"]["seed"]), A=lg["A"], R=lg["R"],
                              same=lg["A"].get("prefix") is not None and lg["A"].get("prefix") == lg["R"].get("prefix")))  # fmt: skip
    print(f"controller {sorted(ctrl)}; runs {a.runs}; reader flags {out['flags']}")
    per_task = ", ".join(f"{t} {sum(p['task'] == t for p in pairs)}" for t in TASKS)
    print(f"pairs {len(pairs)} ({per_task}); "
          f"ended in error (left out, revision 2 section 6) {len(errors)}; one arm missing {len(missing)} {missing[:6]}")  # fmt: skip
    for e in errors:
        print("  error:", json.dumps(e))
    out.update(n_pairs=len(pairs), errors=errors, missing=missing)
    # 2. consistency
    nd = sum(not p["same"] for p in pairs)
    share = nd / len(pairs) if pairs else 0.0
    print(f"consistency: prefixes identical {len(pairs) - nd}/{len(pairs)}; differ {nd} ({100 * share:.1f} %): "
          + ", ".join(p["name"] for p in pairs if not p["same"])[:400])  # fmt: skip
    out["prefix_differ"] = nd
    # 5 (structure first): segments of the commands through the model
    land, frame_ok, frame_bad, a_goal_eq, a_goal_ne, no_seg, touch_rows = [], 0, [], 0, 0, 0, []
    for p in pairs:
        for arm in ("A", "R"):
            lg = p[arm]
            npz = p["run"] / arm / "data" / f"{p['name']}.npz"
            D = DT.load(str(npz)) if npz.exists() else None
            z = np.load(npz, allow_pickle=True) if npz.exists() else None
            hit = TOUCH.touching(z) if z is not None else None
            for cm in lg.get("commands", []):
                if not cm.get("through"):
                    continue
                for s, g in cm["goal"].items():
                    if arm == "A":
                        if np.allclose(np.asarray(cm.get("exec", {}).get(s, [np.nan] * 3), float), np.asarray(g, float), atol=1e-6):
                            a_goal_eq += 1
                        else:
                            a_goal_ne += 1
                    rows = [] if D is None or cm.get("chain") is None else [
                        i for i in np.flatnonzero(np.asarray(D["chain"]) == cm["chain"]) if (float(D["side"][i]) > 0) == (s == "right")]
                    if not rows:
                        no_seg += 1
                        continue
                    i = rows[0]
                    ex = np.asarray(cm["exec"][s], float)
                    if np.allclose(np.asarray(D["target"][i], float), ex, atol=1e-4):
                        frame_ok += 1
                    else:
                        frame_bad.append(dict(name=p["name"], arm=arm, label=cm["label"], side=s, dmax=float(np.abs(np.asarray(D["target"][i], float) - ex).max())))
                    so, k0, n_ = int(D["so"][i]), int(D["k0"][i]), int(D["n_steps"][i])
                    W = np.asarray(D["W"], np.float64)
                    r0, r1 = so + k0, so + n_ - 1
                    dp = DT.hv(np.r_[W[r1, :2] - W[r0, :2], 0.0][None], np.array([W[r0, 3]]))[0, :2] * 100
                    land.append(dict(task=p["task"], seed=p["seed"], arm=arm, label=cm["label"], side=s,
                                     err=100 * float(np.linalg.norm(np.asarray(D["pos_f"][i], float) - np.asarray(g, float))),
                                     touching=touch_of(z, hit, cm["chain"], s),
                                     fwd=float(dp[0]), lat=float(dp[1])))  # fmt: skip
    print(f"commands through the model, per hand: segments found {len(land)}, none recorded {no_seg}; frame check (segment target == "
          f"the log's executed target) {frame_ok} equal, {len(frame_bad)} differ {frame_bad[:3]}; A's executed == goal {a_goal_eq}, differ {a_goal_ne}")  # fmt: skip
    out.update(frame_ok=frame_ok, frame_bad=frame_bad, a_goal_eq=a_goal_eq, a_goal_ne=a_goal_ne, no_seg=no_seg)
    if a.structure:
        print("structure only: no metric printed")
        return 0
    if share > 0.05:
        print("more than 5 % of the pairs differ before the first pre-grasp: the table's row 2 -- results not printed")
        return 0
    if frame_bad:
        print("the frame check failed: landing errors not printed")
    # 3, 4
    res = {}
    for key, tag, f in (("lift", "grasp-and-lift (main, revision 1)", lambda lg: lg.get("lifted_step") is not None),
                        ("success3", "success within 3x (secondary)", lambda lg: bool(lg.get("success"))),
                        ("success_official", "success within the official budget (secondary)", lambda lg: bool(lg.get("success_official")))):  # fmt: skip
        o = binary(pairs, f)
        o["tasks"] = {t: binary([p for p in pairs if p["task"] == t], f) for t in TASKS}
        o["same_prefix"] = binary([p for p in pairs if p["same"]], f)
        res[key] = o
        print("\n" + show_bin(tag, o))
        for t in TASKS:
            if o["tasks"][t]["n"]:
                print("  " + show_bin(t, o["tasks"][t]))
        print("  " + show_bin(f"identical prefixes only ({o['same_prefix']['n']} pairs)", o["same_prefix"]))
    if a.holm:
        ps = {k: float(v) for k, v in (x.split("=") for x in a.holm.split(","))}
        ps[",".join(sorted(ctrl))] = res["lift"]["p"]
        res["holm"] = holm(ps)
        print("Holm (main metric, controllers):", json.dumps(res["holm"]))
    out["binary"] = res
    # 5
    if not frame_bad:
        print("\nlanding error of the commands through the model (cm; hand point at the end label to the perceived target):")
        out["landing"] = {}
        for t in TASKS:
            for lab, s in sorted({(x["label"], x["side"]) for x in land if x["task"] == t}):
                ea = {x["seed"]: x["err"] for x in land if x["task"] == t and x["label"] == lab and x["side"] == s and x["arm"] == "A"}
                er = {x["seed"]: x["err"] for x in land if x["task"] == t and x["label"] == lab and x["side"] == s and x["arm"] == "R"}
                sd = sorted(set(ea) & set(er))
                d = np.array([er[k] - ea[k] for k in sd])
                ci = (float("nan"), float("nan"))
                if len(d):
                    bs = [np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(NB)]
                    ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
                o = dict(nA=len(ea), nR=len(er), n_pairs=len(sd), A=float(np.median(list(ea.values()))) if ea else float("nan"),
                         R=float(np.median(list(er.values()))) if er else float("nan"), diff=float(np.median(d)) if len(d) else float("nan"), ci=ci)  # fmt: skip
                out["landing"][f"{t} {lab} {s}"] = o
                print(f"  {t} {lab} {s}: A median {o['A']:.2f} (n {o['nA']}), R median {o['R']:.2f} (n {o['nR']}); paired R - A median "
                      f"{o['diff']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] over {o['n_pairs']} seeds")  # fmt: skip
    # 6
    print("\nhow episodes ended, the furthest step, HA falls (per task and arm):")
    out["ends"] = {}
    for t in TASKS:
        for arm in ("A", "R"):
            lgs = [p[arm] for p in pairs if p["task"] == t]
            if not lgs:
                continue
            e = {}
            for lg in lgs:
                e[how(lg)] = e.get(how(lg), 0) + 1
            fs = {}
            for lg in lgs:
                fs[furthest(lg)] = fs.get(furthest(lg), 0) + 1
            fell = sum(lg.get("fell_step") is not None for lg in lgs)
            out["ends"][f"{t} {arm}"] = dict(ends=e, furthest=fs, fell=fell)
            print(f"  {t} {arm} (n {len(lgs)}): ends {json.dumps(e)}; furthest {json.dumps(fs)}; HA fall {fell}")
    dd = sorted((p["seed"], p["A"].get("max_lift_cm"), p["R"].get("max_lift_cm")) for p in pairs if p["task"] == "doubledesk")
    if dd:
        print("\nrevision 3 (1) DoubleDesk max lift per episode, cm (seed: A, R):")
        print("  " + "; ".join(f"{s}: {x}, {y}" for s, x, y in dd))
        za, zr = sum(x == 0 for _, x, _ in dd), sum(y == 0 for _, _, y in dd)
        print(f"  episodes with max lift 0.0: A {za}/{len(dd)}, R {zr}/{len(dd)}")
        out["dd_max_lift"] = dd
    pp = [p for p in pairs if p["task"] == "ppbox"]
    if pp:
        sa = sum(p["A"].get("sequence") == NOT_SEEN["ppbox"] for p in pp)
        sr = sum(p["R"].get("sequence") == NOT_SEEN["ppbox"] for p in pp)
        print(f"revision 3 (2) P&PBox stuck at find_shelf (sequence 'shelf not seen'): A {sa}/{len(pp)}, R {sr}/{len(pp)}")
        out["ppbox_find_shelf"] = dict(A=sa, R=sr, n=len(pp))
    # 7
    rr = [cm["r"] for p in pairs for cm in p["R"].get("commands", []) if cm.get("through") and cm.get("r")]
    if rr:
        shr = sum(any(s.get("shrunk") for s in r.get("shrink", [])) for r in rr)
        corr = [r["corr_cm"] for r in rr if isinstance(r.get("corr_cm"), (int, float))]
        pm = [r["pred_miss_cm"] if isinstance(r.get("pred_miss_cm"), (int, float)) else max(r["pred_miss_cm"].values()) for r in rr if r.get("pred_miss_cm") is not None]
        print(f"\nR's commands through the model: {len(rr)}; with a shrink {shr} ({100 * shr / len(rr):.1f} %); correction median "
              f"{np.median(corr) if corr else float('nan'):.2f} cm; predicted miss median {np.median(pm) if pm else float('nan'):.2f} cm (two-hand: the larger hand)")  # fmt: skip
        out["r_cmds"] = dict(n=len(rr), shrunk=shr)
    print("contacts (hd_touch's rule) and pelvis displacement on the reaches through the model (descriptive):")
    for t in TASKS:
        for arm in ("A", "R"):
            xs = [x for x in land if x["task"] == t and x["arm"] == arm]
            if xs:
                tc = sum(bool(x["touching"]) for x in xs)
                print(f"  {t} {arm}: hands {len(xs)}, touching {tc}; pelvis forward median {np.median([x['fwd'] for x in xs]):.1f} cm, "
                      f"lateral median {np.median([x['lat'] for x in xs]):.1f} cm")  # fmt: skip
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
