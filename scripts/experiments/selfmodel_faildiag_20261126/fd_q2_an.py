"""selfmodel_faildiag_20261126 question 2: the measurement groups (servo, servopos, gtgrasp, gtservo) against
lmlcmp3's A (alone) and R (reachable), paired by episode (measurement only; the servo and truth groups use simulator
truth and never enter a method comparison).

Per (controller, task, group, episode): success and reason; for the servo groups whether the servo converged (each
segment inside 0.5 cm / 5 deg, the record's servo_done) and the last residual; on dumbbell / watering can (step logs
for every group) whether the object left its look pose before the hand's arrival (fd_q1_moved.episode, the corrected
definition; for the servo groups the arrival counts from the FIRST entry command, the servo's repeats are part of the
approach), recomputed per group since the walks can differ.

Paired comparisons: McNemar exact (two-sided) on the episodes both groups have; also on the pairs where the object was
not moved before the arrival in either run.

    python fd_q2_an.py --out <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import fd_q1_moved as M

FD = M.FD
L3 = Path("/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126")
TASKS = {"IThorLivingRoomApproachTableAndLiftDumbbell": "dumbbell", "IThorLivingRoomApproachTableAndLiftWateringCan": "wcan",
         "IThorBathroomLiftCandleFromFloor": "candle", "IThorKitchenApproachTableAndLiftApple": "apple"}  # fmt: skip
TN = {v: k for k, v in TASKS.items()}
BACKENDS = ("scalebfm", "sonic")
# group -> (episode root, method dir); A and R: lmlcmp3's records (dumbbell / watering can: question 1's bit-identical
# rerun, which also has the step logs)
GROUPS = ("A", "R", "servo", "servopos", "gtgrasp", "gtservo")
PAIRS = (("servo", "A"), ("gtservo", "A"), ("servo", "servopos"), ("gtservo", "servo"), ("gtgrasp", "R"), ("R", "A"))


def paths(group: str, be: str, task: str, ep: int) -> tuple[Path | None, Path | None]:
    f = f"ep{ep:+05d}"
    if group in ("A", "R"):
        m = "alone" if group == "A" else "reachable"
        if TASKS[task] in ("dumbbell", "wcan"):
            return FD / "rerun" / be / "episodes" / task / m / f"{f}.json", FD / "rerun" / be / "steplog" / task / m / f"{f}.npz"
        return L3 / be / "episodes" / task / m / f"{f}.json", None
    hits = glob.glob(str(FD / group / be / "episodes" / task / "*" / f"{f}.json"))
    if not hits:
        return None, None
    p = Path(hits[0])
    return p, FD / group / be / "steplog" / task / p.parent.name / f"{f}.npz"


def first_waits(r: dict) -> dict:
    """The record with each command key's first non-correction-walk fire only (the servo repeats dropped), so that
    fd_q1_moved.episode's arrival counts from the first entry command."""
    seen, ws = set(), []
    for w in r.get("waits", []):
        if w.get("then") == "correction walk":
            ws.append(w)
        elif w["key"] not in seen:
            seen.add(w["key"])
            ws.append(w)
    return dict(r, waits=ws)


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def one(group: str, be: str, task: str, ep: int) -> dict | None:
    pj, pz = paths(group, be, task, ep)
    if pj is None or not pj.exists():
        return None
    r = json.loads(pj.read_text())
    out = dict(success=bool(r.get("success")), reason=r.get("reason"), close_dist_cm=r.get("close_dist_cm"), close_rot_err_deg=r.get("close_rot_err_deg"))
    if "servo" in r:
        sd = r.get("servo_done") or {}
        last = {k: (v[-1] if v else None) for k, v in r["servo"].items()}
        out.update(conv1=bool(sd.get("cmd1")), conv2=bool(sd.get("cmd2")), reached_entry="cmd2" in r["servo"] and bool(r["servo"]["cmd2"]),
                   res_p_cm=(last.get("cmd2") or {}).get("err_p_cm"), res_r_deg=(last.get("cmd2") or {}).get("err_r_deg"),
                   n_servo=sum(len(v) for v in r["servo"].values()), revert=any(e.get("revert") for v in r["servo"].values() for e in v))  # fmt: skip
    if "gt_n" in r:
        out.update(gt_n=r["gt_n"])
    plan_ok = bool(r.get("plan") and r["plan"].get("fail") is None)
    out["plan_ok"] = plan_ok
    if pz is not None and pz.exists() and plan_ok:
        z = np.load(pz)
        r1 = first_waits(r)
        if TASKS[task] in ("dumbbell", "wcan"):
            e = M.episode(r1, z, False)
            out.update(moved=e["moved"], at_arr_cm=e["at_arr_cm"], phase=e.get("phase"))
        # the entry's push: the object's move from the first entry command to the close command (the servo's repeats
        # sit in between)
        w = {x["key"]: int(x["fire"]) for x in r1["waits"] if x.get("then") != "correction walk" and x.get("fire") is not None}
        if "cmd2" in w and "close" in w:
            C = {str(c): i for i, c in enumerate(z["cols"])}
            o = z["rows"][:, [C["obj_px"], C["obj_py"], C["obj_pz"]]]
            out["entry_push_cm"] = float(100 * np.linalg.norm(o[w["close"]] - o[w["cmd2"]]))
    return out


def pct(k: int, n: int) -> str:
    return f"{k}/{n} ({100 * k / n:.0f} %)" if n else "0/0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    eps = defaultdict(set)  # (backend, task) -> the question 2 episodes, from the job list
    for ln in (FD / "jobs_q2.txt").read_text().split("\n"):
        t = ln.split()
        if not t:
            continue
        kv = dict(zip(t[1::2], t[2::2]))
        eps[(kv["--backend"], kv["--task"])].update(int(x) for x in kv["--ep-list"].split(","))
    per = []
    for (be, task), E in sorted(eps.items()):
        for ep in sorted(E):
            row = dict(backend=be, task=TASKS[task], ep=ep)
            for g in GROUPS:
                v = one(g, be, task, ep)
                if v is not None and g in ("gtgrasp", "gtservo") and v["reason"] == "stance_or_ik" and not v["plan_ok"]:
                    row[f"{g}_none"] = True  # no true grasp passed the filters: reported apart, out of the denominators
                elif v is not None:
                    row[g] = v
            per.append(row)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "q2_episodes.json").write_text(json.dumps(per, indent=0, default=float))

    def block(rows: list[dict], label: str) -> None:
        print(f"\n== {label}: {len(rows)} episodes; no true grasp passed the filters (out of the gt groups' denominators): "
              f"gtgrasp {sum(bool(x.get('gtgrasp_none')) for x in rows)}, gtservo {sum(bool(x.get('gtservo_none')) for x in rows)}")
        for g in GROUPS:
            G = [x[g] for x in rows if g in x]
            if not G:
                continue
            s = f"  {g:9s} success {pct(sum(v['success'] for v in G), len(G))}; reasons {dict(Counter(v['reason'] for v in G).most_common())}"
            if "conv2" in G[0]:
                C2 = [v for v in G if v["conv2"]]
                res = [v["res_p_cm"] for v in G if v.get("res_p_cm") is not None]
                rr = [v["res_r_deg"] for v in G if v.get("res_r_deg") is not None]
                s += (f"\n            servo converged: entry {pct(len(C2), len(G))}, pre-grasp {pct(sum(v['conv1'] for v in G), len(G))}; success when the entry converged {pct(sum(v['success'] for v in C2), len(C2))}; "
                      f"entry residual p50 {np.median(res) if res else float('nan'):.2f} cm / {np.median(rr) if rr else float('nan'):.1f} deg (n {len(res)}); reverted {sum(v['revert'] for v in G)}")  # fmt: skip
            Ep = [v for v in G if "entry_push_cm" in v]
            if Ep:
                pu = [v for v in Ep if v["entry_push_cm"] > M.MOVE_CM]
                s += f"\n            object pushed > {M.MOVE_CM} cm between the first entry command and the close {pct(len(pu), len(Ep))}, their success {pct(sum(v['success'] for v in pu), len(pu))}"
            Mv = [v for v in G if "moved" in v]
            if Mv:
                nm = [v for v in Mv if not v["moved"]]
                s += f"\n            object moved before the arrival {pct(len(Mv) - len(nm), len(Mv))}; success when not moved {pct(sum(v['success'] for v in nm), len(nm))}"
                if "conv2" in G[0]:
                    nc = [v for v in nm if v["conv2"]]
                    s += f"; not moved and the entry converged: success {pct(sum(v['success'] for v in nc), len(nc))}"
            print(s)
        for g1, g2 in PAIRS:
            P = [x for x in rows if g1 in x and g2 in x]
            if not P:
                continue
            b = sum(x[g1]["success"] and not x[g2]["success"] for x in P)
            c = sum(x[g2]["success"] and not x[g1]["success"] for x in P)
            s = f"  pair {g1} v {g2}: n {len(P)}, {g1} {sum(x[g1]['success'] for x in P)} v {g2} {sum(x[g2]['success'] for x in P)}, only {g1} {b}, only {g2} {c}, McNemar p {mcnemar(b, c):.3g}"
            Q = [x for x in P if x[g1].get("moved") is False and x[g2].get("moved") is False]
            if Q:
                b2 = sum(x[g1]["success"] and not x[g2]["success"] for x in Q)
                c2 = sum(x[g2]["success"] and not x[g1]["success"] for x in Q)
                s += f"; neither moved: n {len(Q)}, {sum(x[g1]['success'] for x in Q)} v {sum(x[g2]['success'] for x in Q)}, only {b2} / {c2}, p {mcnemar(b2, c2):.3g}"
            print(s)

    for be in BACKENDS:
        for tn in ("dumbbell", "wcan", "candle", "apple"):
            R = [x for x in per if x["backend"] == be and x["task"] == tn]
            if R:
                block(R, f"{be} {tn}")
        block([x for x in per if x["backend"] == be and x["task"] in ("dumbbell", "wcan")], f"{be} dumbbell + watering can (the decision set)")
    missing = {g: sum(g not in x and not x.get(f"{g}_none") for x in per if not (g == "servopos" and x["backend"] != "sonic") and not (g == "servopos" and x["task"] not in ("dumbbell", "wcan"))) for g in GROUPS}
    print(f"\nmissing records per group: {missing}")
    print("moved per episode (group: backend task ep) listed in q2_episodes.json")


if __name__ == "__main__":
    main()
