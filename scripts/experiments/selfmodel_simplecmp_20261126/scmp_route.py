"""selfmodel_simplecmp_20261126: the route arm's goals and plans (S-full's steps with model D in place of jrefwm).

S-full (precision.py shifts2's SF, selfmodel_sfull_20261125) moved J's pregrasp goal by -(g + p): the reference's jaw at
the end of J's plan minus the look's goal plus the SIMPLE-trained via-point model's predicted offset from that end, that is
minus the model's predicted landing's miss of the goal, in the heading frame of the pelvis yaw at the command; then one
cuRobo replan (plan2.py, IK seed 1531), at 2/3 and 1/3 of the shift where it does not plan, J's own plan where none does.
Here the predicted landing is model D's (the JL arm: J run with --correct-rule dryR, which records D's prediction of the
palm at the end of J's joint path from the state at the command, rule "joint"):

    shift_h = -(pred_hp - goal_hp)      (D's palm, the goal palm; the command row's heading frame; stage "pregrasp")

and everything after it is S-full's: prep goal_wrist + from_head(k shift_h, yaw) for k in 1, 2/3, 1/3 (precision.SCALES),
plan2.py, the first scale that plans, J's own plan otherwise. A second arm J0 moves nothing (shift 0, the same replan):
cuRobo's replans of one input differ run to run (curobo-replan-nondeterminism), so J0 is the route arm's partner planned
in the same run.

    PYTHONPATH=src $PY -I scmp_route.py shifts <P> <JL corr dir> <bench dir>    -> <P>/<arm>/step1, step2*, <P>/shifts.jsonl
    (plan2.py per arm, scale and routed episode: scmp_run.sh)
    PYTHONPATH=src $PY -I scmp_route.py pick <P> <bench dir>                     -> <P>/<arm>/via, <P>/pick.jsonl

<bench dir> is the main metric's (bench_rgbd_20261124/scalebfm: via/, step1/, step2/, try_act02/), read only.
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

import numpy as np

SCALES = (("1", 1.0, "step2"), ("2/3", 2 / 3, "step2_s2of3"), ("1/3", 1 / 3, "step2_s1of3"))  # precision.SCALES
ARMS = {"JD": True, "J0": False}  # arm: shifted (tries the smaller scales)


def from_head(v_h, yaw: float) -> np.ndarray:  # precision.from_head
    c, s = np.cos(yaw), np.sin(yaw)
    v = np.asarray(v_h, float)
    return np.r_[c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]


def shifts(a) -> None:
    P, B, C = Path(a.p), Path(a.bench), Path(a.jl)
    for arm in ARMS:
        (P / arm / "step1").mkdir(parents=True, exist_ok=True)
        for _, _, d in SCALES:
            (P / arm / d).mkdir(parents=True, exist_ok=True)
    out = open(P / "shifts.jsonl", "w")
    n = 0
    for f in sorted(glob.glob(str(C / "log" / "*.jsonl"))):
        for line in open(f):
            lg = json.loads(line)
            name = f"{lg['task']}_L{lg['level']}_e{lg['index']}"
            vf = B / "via" / f"via_{name}.npz"
            pre = [s for s in lg["stages_correct"] if s["stage"] == "pregrasp"]
            if not vf.exists() or not pre or pre[0]["rule"] != "joint":
                continue  # not routed: J's pregrasp was the straight one (J = A there)
            s = pre[0]
            sh = -(np.asarray(s["pred_hp"], float) - np.asarray(s["goal_hp"], float))
            yaw = float(s["yaw"])
            v0 = dict(np.load(vf, allow_pickle=True))
            src = str(v0["src"])
            pr = dict(np.load(B / "step2" / f"prep_{name}.npz"))
            for arm, shifted in ARMS.items():
                for _, k, d in SCALES if shifted else SCALES[:1]:
                    prz = dict(pr)
                    prz["goal_wrist"] = np.asarray(pr["goal_wrist"], float) + (from_head(k * sh, yaw) if shifted else 0.0)
                    np.savez(P / arm / d / f"prep_{name}.npz", **prz)
                shutil.copy(B / "step1" / f"obst_{name}.npz", P / arm / "step1" / f"obst_{name}.npz")
            row = dict(name=name, ep=10 * int(lg["level"]) + int(lg["index"]), task=lg["task"],
                       act=0.02 if "try_act02" in src else 0.03, yaw=yaw, plan_src_J=src,
                       pred_hp=s["pred_hp"], goal_hp=s["goal_hp"], ref_end_h=s.get("ref_end_h"),
                       pred_miss_palm_cm=s.get("pred_miss_palm_cm"), in_table=s.get("in_table"),
                       shift_h_cm={"JD": (100 * sh).round(3).tolist(), "J0": [0.0, 0.0, 0.0]},
                       shift_cm=round(100 * float(np.linalg.norm(sh)), 3))  # fmt: skip
            out.write(json.dumps(row) + "\n")
            n += 1
    out.close()
    print(f"{n} routed episodes -> {P / 'shifts.jsonl'}")


def pick(a) -> None:
    """precision.pick with this entry's arms: the first scale that planned, J's own plan where none did."""
    P, B = Path(a.p), Path(a.bench)
    out = open(P / "pick.jsonl", "w")
    for arm in ARMS:
        (P / arm / "via").mkdir(exist_ok=True)
    for line in open(P / "shifts.jsonl"):
        r = json.loads(line)
        name = r["name"]
        v0 = dict(np.load(B / "via" / f"via_{name}.npz", allow_pickle=True))
        Q0 = np.load(Path(str(v0["src"])) / f"plan_{name}.npz")["Q"]
        for arm, shifted in ARMS.items():
            tried, pl, scale = [], None, "fallback"
            for lab, _, d in SCALES if shifted else SCALES[:1]:
                pf, rf = P / arm / d / f"plan_{name}.npz", P / arm / d / f"plan_row_{r['task']}_{r['ep']}.json"
                if not (pf.exists() or rf.exists()):
                    continue
                st_ = json.loads(rf.read_text().splitlines()[0]).get("status") if rf.exists() and rf.stat().st_size else None
                tried.append(dict(scale=lab, status=st_))
                z = np.load(pf) if pf.exists() else None
                if z is not None and "Q" in z.files:
                    pl, scale = z, lab
                    break
            v = dict(v0)
            row = dict(name=name, ep=r["ep"], task=r["task"], arm=arm, planned=pl is not None, scale=scale, tried=tried)
            if pl is not None:
                v["src"] = np.array(str((P / arm / dict((x[0], x[2]) for x in SCALES)[scale]).resolve()))
                Q = pl["Q"]
                row.update(same_as_J=bool(Q.shape == Q0.shape and np.allclose(Q, Q0, atol=1e-6)), T=len(Q), T_J=len(Q0),
                           end_dq_deg=round(float(np.degrees(np.abs(Q[-1] - Q0[-1]).max())), 3) if Q.shape[1:] == Q0.shape[1:] else None)  # fmt: skip
            np.savez(P / arm / "via" / f"via_{name}.npz", **v)
            out.write(json.dumps(row) + "\n")
    out.close()
    print(open(P / "pick.jsonl").read(), end="")


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("shifts")
    s.add_argument("p")
    s.add_argument("jl")
    s.add_argument("bench")
    k = sp.add_parser("pick")
    k.add_argument("p")
    k.add_argument("bench")
    a = ap.parse_args()
    {"shifts": shifts, "pick": pick}[a.cmd](a)


if __name__ == "__main__":
    main()
