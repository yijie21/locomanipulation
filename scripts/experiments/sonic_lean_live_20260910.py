"""Isolated instance-1 physical waist diagnostic, adapted from 78536ae diag_lean.

Historical angle sweep, with reset between channels and time-series FK torso
world pitch added. No grasp expert or training dataset is modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from locomani import paths
from locomani.backends.sonic import stream_reach
from locomani.backends.sonic.primitives import LocomotionMode, Primitives, StreamMode
from locomani.primitives.arm_ik import ArmIK
from locomani.sim.privileged_pub import EpisodeControlClient

SCENE_META = str(paths.SCENE_META)
OUT = "outputs/diagnostics/sonic_lean_20260910/live/result.json"
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
    ap.add_argument(
        "--arm",
        choices=("reach", "neutral"),
        default="reach",
        help="reach: hold a far-extended arm pose (balance-expensive); neutral: leave the arm where standing left it",
    )
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    if os.environ.get("TF_EPISODE_PORT") != "5592" or os.environ.get("TF_SONIC_PORT") != "5566":
        raise RuntimeError("diagnostic requires isolated instance 1 ports")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    meta = json.load(open(SCENE_META))
    p = Primitives(meta=meta, verbose=False)
    from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber

    debug = ZMQStateSubscriber(port=5567)
    reset = EpisodeControlClient()
    ik = ArmIK()
    ik.lo[14], ik.hi[14] = -0.05, 0.52  # let the solver use the range
    s = restand(p, reset, "lean")
    if s["fallen"]:
        sys.exit("robot did not stand")

    # one forward-reach arm pose, solved once with the waist locked at 0
    aim_pf = np.array([0.42, -0.20, -0.08])
    if args.arm == "reach":
        q_arm, res, _ = ik.solve(SIDE, aim_pf, None, q_mj29=np.array(s["joints"]), use_waist=False)
        print(
            f"[lean] extended arm pose solved, IK res {res * 100:.1f} cm "
            f"(the aim is outside the arm's reach on purpose)"
        )
    else:
        q_arm = np.array(s["joints"])
        print("[lean] arm left at the standing pose")

    out = []
    for kind in ("planner", "streamed"):
        s = restand(p, reset, "lean_" + kind)
        if s["fallen"]:
            s = restand(p, reset, "restand")
        reset.pin_arms()
        arm = stream_reach.PlannerArm(p, SIDE) if kind == "planner" else stream_reach.StreamedArm(p, SIDE)
        arm.start()
        traces = []
        original_push = arm._push
        torso_bid = ik.model.body("torso_link").id

        def trace_push(q, original_push=original_push):
            original_push(q)
            state = p.obs()
            message = debug.get_msg()
            debug_data = (
                None
                if message is None
                else {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in message.items()}
            )
            ik.fk(np.array(state["joints"]), SIDE)
            torso = Primitives._rot(state["pelvis"]["quat"]) @ ik.data.xmat[torso_bid].reshape(3, 3)
            traces.append(
                dict(
                    deploy_debug=debug_data,
                    wall=time.time(),
                    sim_t=state["t"],
                    age=time.time() - state["wall"],
                    requested=list(np.asarray(q, dtype=float)),
                    joints=state["joints"],
                    pelvis=state["pelvis"],
                    torso_pitch=float(np.arcsin(np.clip(-torso[2, 0], -1, 1))),
                    fallen=state["fallen"],
                )
            )
            if state["fallen"]:
                with open(args.out + ".failure.trace.json", "w") as f:
                    json.dump(traces, f)
                raise RuntimeError("fell during diagnostic; stop and inspect trace")

        arm._push = trace_push
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
                channel=kind,
                arm=args.arm,
                cmd=wp,
                delivered=float(st["joints"][14]),
                pelvis_pitch=pitch_of(st["pelvis"]["quat"]),
                pelvis_z=float(pel[2]),
                palm_fwd=float((R.T @ (palm - pel))[0]),
                palm_z=float(palm[2]),
                d_fwd=float(np.linalg.norm((palm - base_palm)[:2])),
                d_z=float(palm[2] - base_palm[2]),
                pelvis_drift=float(np.linalg.norm((pel - base_pel)[:2])),
                fallen=bool(st["fallen"]),
                torso_pitch=float(traces[-1]["torso_pitch"]),
                steady_waist_pitch=float(np.mean([x["joints"][14] for x in traces[-50:]])),
                steady_torso_pitch=float(np.mean([x["torso_pitch"] for x in traces[-50:]])),
                max_observation_age=float(max(x["age"] for x in traces[-50:])),
            )
            out.append(rec)
            with open(args.out, "w") as f:
                json.dump(out, f, indent=2)
            with open(args.out + "." + kind + ".trace.json", "w") as f:
                json.dump(traces, f)
            print(
                f"[lean] {kind:9} cmd {wp:.2f} -> {rec['delivered']:+.3f} "
                f"({rec['delivered'] / wp * 100 if wp > 0.01 else 0:4.0f}%)  "
                f"pelvis pitch {np.degrees(rec['pelvis_pitch']):+5.1f}d  "
                f"palm fwd {rec['palm_fwd'] * 100:5.1f}cm z {rec['palm_z']:.3f}"
                f"  drift {rec['pelvis_drift'] * 100:4.1f}cm"
                f"{'  FELL' if rec['fallen'] else ''}"
            )
            if rec["fallen"]:
                break
        q = arm.q.copy()
        q[14] = 0.0
        arm.move_to(q, speed=0.5)
        arm.stop()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    debug.close()
    p.close()
    print(f"[lean] wrote {args.out}")
    for kind in ("planner", "streamed"):
        v = [r for r in out if r["channel"] == kind and r["cmd"] > 0.01 and not r["fallen"]]
        if not v:
            continue
        ratio = np.median([r["delivered"] / r["cmd"] for r in v])
        top = max(v, key=lambda r: r["cmd"])
        print(
            f"    {kind:9} delivery ratio {ratio:.2f}; at cmd {top['cmd']:.2f} "
            f"the palm moved {top['d_fwd'] * 100:.1f} cm in the xy plane, "
            f"{top['d_z'] * 100:+.1f} cm in z"
        )


if __name__ == "__main__":
    main()
