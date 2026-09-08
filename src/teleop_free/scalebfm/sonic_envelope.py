"""The ScaleBFM reach-envelope sweep, fed to SONIC.

Same pad targets as run_spike.run_envelope (world frame: initial pelvis xy +
x forward, +-0.20 m sideways, 0.76 + z high; palm pitch 60 deg; 2 s each),
sent the only way SONIC accepts arm targets: our ArmIK -> 17 upper-body joint
targets on the planner channel (BimanualArms, waist locked as in the crate
expert). Standing, IDLE, no squat command. Records the delivered pad error and
the pelvis displacement per target to envelope.json in the spike's format, and
records the sim state trace (trace/envelope_state_trace.npz) from the first
target on, so it can be rendered with the spike's camera for a side-by-side.

    INSTANCE=2 bash scripts/run_sonic_envelope.sh
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from agent_server import LocomotionMode, StreamMode  # noqa: E402
from primitives import DT, Primitives, upper_body_from_mj  # noqa: E402
from privileged_pub import EpisodeControlClient  # noqa: E402
import paths  # noqa: E402
import upper_ref  # noqa: E402
from stream_reach import reanchor  # noqa: E402
from bimanual_carry import BimanualArms, rot_y  # noqa: E402
from g1_joints import ARM_MJ  # noqa: E402

OBJ = "blue_crate"
PALM_PITCH_DEG = 60.0
PELVIS_Z = 0.76
XS = np.arange(0.20, 0.76, 0.05)
ZS = np.array([0.10, 0.0, -0.10, -0.20, -0.30, -0.40, -0.50])
HOLD_S = float(os.environ.get("HOLD_S", 2.0))
SHOULDER_PITCH_MIN_DEG = -80.0
# SQUAT=1: the fair variant. Per target, command the k-planner's own crouch
# (LocomotionMode.IDEL_SQUAT + absolute pelvis `height`, the one depth lever
# SONIC has; calibrated delivered z ~= height + 0.02, saturating below 0.40)
# so that the pad target sits within the arm's reach band below the pelvis.
SQUAT = os.environ.get("SQUAT", "0") == "1"
SQUAT_REACH_BAND = float(os.environ.get("SQUAT_BAND", 0.10))   # target no lower than this below the pelvis (arm-only band with the expert bounds is ~0.15)
SQUAT_HEIGHT_BIAS = 0.02


def main():
    inst = int(os.environ.get("INSTANCE", "0"))
    for k, base in (("TF_SONIC_PORT", 5556), ("TF_OUT_PORT", 5557), ("TF_KEYBOARD_PORT", 5580),
                    ("TF_PRIVILEGED_PORT", 5581), ("TF_EPISODE_PORT", 5582)):
        os.environ.setdefault(k, str(base + 10 * inst))
    out_dir = os.environ["SIM_VIDEO_DIR"]
    meta = json.load(open(paths.SCENE_META))
    p = Primitives(meta=meta)
    reset = EpisodeControlClient()

    def stand_up():
        p.cmd.reset_pose()
        reset.pin_arms()
        # crate out of the way, robot at its nominal start (-1.5, 0) facing +x
        reset.reset(objects={OBJ: {"dxy": [3.0, 3.0], "yaw": 0.0}}, robot={"dxy": [0.0, 0.0], "yaw": 0.0})
        time.sleep(0.3)
        p.cmd.stream_mode = StreamMode.PLANNER_VR_3PT
        p.cmd.upper_body = None
        p.cmd.start_control()
        upper_ref.shared().wait(3.0)
        p.cmd.hold(2.0, mode=LocomotionMode.IDLE)
        reanchor(p)
        p.cmd.hold(1.5, mode=LocomotionMode.IDLE, facing=(1.0, 0.0, 0.0))
        s = p.obs()
        q_free = upper_ref.shared().upper17()
        q_free = upper_body_from_mj(s["joints"]) if q_free is None else q_free
        ramp = upper_ref.UpperRamp(p.cmd, q_free)
        p.cmd.stream_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY
        arms = BimanualArms(p)
        for sd in ("left", "right"):
            arms.arm.ik.lo[ARM_MJ[sd][0]] = np.radians(SHOULDER_PITCH_MIN_DEG)
        ramp.settle()
        arms.start(q17=ramp.q)
        arms.arm.q = np.asarray(s["joints"], dtype=float).copy()
        return arms, s

    arms, s = stand_up()
    p0 = np.asarray(s["pelvis"]["pos"][:3], dtype=float)
    print(f"[sonic-env] pelvis at {np.round(p0, 3)}, targets relative to xy {np.round(p0[:2], 3)} and z {PELVIS_Z}")
    rot = {sd: rot_y(np.radians(PALM_PITCH_DEG)) for sd in ("left", "right")}
    reset.start_video("envelope")
    rows = []
    t_start = time.time()
    for z in ZS:
        for x in XS:
            tg = {sd: np.array([p0[0] + x, p0[1] + (0.20 if sd == "left" else -0.20), PELVIS_Z + z]) for sd in ("left", "right")}
            if SQUAT:
                pel_want = float(np.clip(PELVIS_Z + z + SQUAT_REACH_BAND, 0.42, PELVIS_Z))
                arms.arm.mode = LocomotionMode.IDEL_SQUAT
                arms.arm.height = float(np.clip(pel_want - SQUAT_HEIGHT_BIAS, 0.40, 0.78))
            fell = False
            # new target: solve from zeros first (the crate expert's "validated
            # branch" rule) -- warm-starting from the previous target's pose
            # left the IK on an arm-out branch with 20 cm residuals (first squat run)
            arms.arm.q = np.zeros(29)
            arms.arm.q = arms.solve(s, tg, rot, iters=200)
            for _ in range(int(HOLD_S / DT)):
                s = p.obs()
                if s["fallen"]:
                    fell = True
                    break
                arms.push(arms.solve(s, tg, rot, iters=60))
                time.sleep(DT)
            s = p.obs()
            err = {sd: arms.grip_point(s, sd)[0] - tg[sd] for sd in tg}
            pel = np.asarray(s["pelvis"]["pos"], dtype=float)
            pel_d = pel - np.array([p0[0], p0[1], PELVIS_Z])
            q_ik = arms.arm.q
            ik_res = {sd: float(np.linalg.norm(
                np.asarray(s["pelvis"]["pos"]) + Primitives._rot(s["pelvis"]["quat"]) @ (
                    arms.arm.ik.fk(q_ik, sd)[0] + arms.arm.ik.fk(q_ik, sd)[1] @ np.asarray(arms_offset())) - tg[sd]) * 100)
                for sd in tg}
            row = dict(x=float(x), z=float(z), squat_height=(arms.arm.height if SQUAT else None), err_cm={sd: [round(float(v) * 100, 1) for v in err[sd]] for sd in err},
                       err_norm_cm=float(np.mean([np.linalg.norm(err[sd]) for sd in err]) * 100),
                       ik_residual_cm=ik_res,
                       pelvis_dxyz_cm=[round(float(v) * 100, 1) for v in pel_d], fallen=bool(fell),
                       t_s=round(time.time() - t_start, 2))
            rows.append(row)
            print(f"x {x:.2f} z {z:+.2f}: err {row['err_norm_cm']:5.1f} cm (IK residual L {ik_res['left']:.1f} R {ik_res['right']:.1f})  "
                  f"pelvis d {row['pelvis_dxyz_cm']}  {'FELL' if fell else ''}", flush=True)
            if fell:
                break
        if fell:
            arms.stop()
            reset.end_video()
            arms, s = stand_up()
            reset.start_video("envelope_b")
    arms.stop()
    reset.end_video()
    json.dump(rows, open(os.path.join(out_dir, "envelope.json"), "w"), indent=1)
    print("\nmean pad error (cm), rows z rel. pelvis, cols x rel. pelvis")
    print("      " + " ".join(f"{x:5.2f}" for x in XS))
    for z in ZS:
        vals = [next((r["err_norm_cm"] for r in rows if abs(r["x"] - x) < 1e-6 and abs(r["z"] - z) < 1e-6), float("nan")) for x in XS]
        print(f"{z:+.2f} " + " ".join(f"{v:5.1f}" for v in vals))


def arms_offset():
    from bimanual_carry import OFFSET
    return OFFSET


if __name__ == "__main__":
    main()
