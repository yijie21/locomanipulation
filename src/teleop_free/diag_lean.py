"""M4: how much waist lean does the policy deliver, and what reach does it buy?

Phase 1 bets on waist pitch replacing the squat: in the planner channel the
legs belong to the deploy's k-planner, so the 3 waist joints inside
upper_body_position[17] are the only way to bring the hand down and forward.
Two numbers decide whether that bet is sound:

  1. delivery ratio -- an earlier streamed-channel probe measured only ~2/3
     of a commanded lean arriving, and the IK cap was set to 0.20 rad on the
     strength of it, against a 0.52 rad hardware range.
  2. reach gain -- centimetres of forward/downward palm travel per radian of
     delivered lean, with the arm joints held fixed.

The arm is frozen at one forward-reach pose throughout, so every change in
the palm comes from the torso.

    python src/teleop_free/diag_lean.py
Output: outputs/diag/lean.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stream_reach  # noqa: E402
from arm_ik import ArmIK  # noqa: E402
from primitives import (  # noqa: E402
    LocomotionMode,
    Primitives,
    StreamMode,
)
from privileged_pub import EpisodeControlClient  # noqa: E402
import paths

SCENE_META = f"{paths.REPO}/outputs/scene/sim/scene_meta.json"
OUT = f"{paths.REPO}/outputs/diag/lean.json"
SIDE = "right"
LEANS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.52]
HOLD_S = 2.5


def pitch_of(quat) -> float:
    R = Primitives._rot(quat)
    return float(np.arcsin(np.clip(-R[2, 0], -1, 1)))


def restand(p, reset, label):
    p.cmd.stream_mode = StreamMode.PLANNER_VR_3PT
    p.cmd.upper_body = None
    p.cmd.set_upper_body_vel(None)
    p.cmd.reset_pose()
    reset.pin_arms()
    reset.reset(label=label)
    time.sleep(0.3)
    p.cmd.start_control()
    p.cmd.hold(3.0, mode=LocomotionMode.IDLE)
    stream_reach.reanchor(p, side=SIDE)
    return p.obs()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("reach", "neutral"), default="reach",
                    help="reach: hold a far-extended arm pose (balance-expensive); "
                         "neutral: leave the arm where standing left it")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    meta = json.load(open(SCENE_META))
    p = Primitives(meta=meta, verbose=False)
    reset = EpisodeControlClient()
    ik = ArmIK()
    ik.lo[14], ik.hi[14] = -0.05, 0.52          # let the solver use the range
    s = restand(p, reset, "lean")
    if s["fallen"]:
        sys.exit("robot did not stand")

    # one forward-reach arm pose, solved once with the waist locked at 0
    aim_pf = np.array([0.42, -0.20, -0.08])
    if args.arm == "reach":
        q_arm, res, _ = ik.solve(SIDE, aim_pf, None, q_mj29=np.array(s["joints"]),
                                 use_waist=False)
        print(f"[lean] extended arm pose solved, IK res {res * 100:.1f} cm "
              f"(the aim is outside the arm's reach on purpose)")
    else:
        q_arm = np.array(s["joints"])
        print("[lean] arm left at the standing pose")

    out = []
    for kind in ("planner", "streamed"):
        s = p.obs()
        if s["fallen"]:
            s = restand(p, reset, "restand")
        reset.pin_arms()
        arm = (stream_reach.PlannerArm(p, SIDE) if kind == "planner"
               else stream_reach.StreamedArm(p, SIDE))
        arm.start()
        q0 = arm.q.copy()
        if args.arm == "reach":
            q0[15:29] = q_arm[15:29]
        arm.move_to(q0, speed=0.8)
        arm.hold(1.5)
        base = p.obs()
        base_palm = np.array(base["palms"][SIDE]["pos"])
        base_pel = np.array(base["pelvis"]["pos"])
        for wp in LEANS:
            q = arm.q.copy()
            q[14] = wp
            if args.arm == "reach":
                q[15:29] = q_arm[15:29]
            arm.move_to(q, speed=0.5)
            arm.hold(HOLD_S)
            st = p.obs()
            palm = np.array(st["palms"][SIDE]["pos"])
            pel = np.array(st["pelvis"]["pos"])
            R = Primitives._rot(st["pelvis"]["quat"])
            rec = dict(
                channel=kind, arm=args.arm, cmd=wp,
                delivered=float(st["joints"][14]),
                pelvis_pitch=pitch_of(st["pelvis"]["quat"]),
                pelvis_z=float(pel[2]),
                palm_fwd=float((R.T @ (palm - pel))[0]),
                palm_z=float(palm[2]),
                d_fwd=float(np.linalg.norm((palm - base_palm)[:2])),
                d_z=float(palm[2] - base_palm[2]),
                pelvis_drift=float(np.linalg.norm((pel - base_pel)[:2])),
                fallen=bool(st["fallen"]))
            out.append(rec)
            print(f"[lean] {kind:9} cmd {wp:.2f} -> {rec['delivered']:+.3f} "
                  f"({rec['delivered'] / wp * 100 if wp > 0.01 else 0:4.0f}%)  "
                  f"pelvis pitch {np.degrees(rec['pelvis_pitch']):+5.1f}d  "
                  f"palm fwd {rec['palm_fwd'] * 100:5.1f}cm z {rec['palm_z']:.3f}"
                  f"  drift {rec['pelvis_drift'] * 100:4.1f}cm"
                  f"{'  FELL' if rec['fallen'] else ''}")
            if rec["fallen"]:
                break
        q = arm.q.copy(); q[14] = 0.0
        arm.move_to(q, speed=0.5)
        arm.stop()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[lean] wrote {args.out}")
    for kind in ("planner", "streamed"):
        v = [r for r in out if r["channel"] == kind and r["cmd"] > 0.01
             and not r["fallen"]]
        if not v:
            continue
        ratio = np.median([r["delivered"] / r["cmd"] for r in v])
        top = max(v, key=lambda r: r["cmd"])
        print(f"    {kind:9} delivery ratio {ratio:.2f}; at cmd {top['cmd']:.2f} "
              f"the palm moved {top['d_fwd'] * 100:.1f} cm forward, "
              f"{top['d_z'] * 100:+.1f} cm in z")


if __name__ == "__main__":
    main()
