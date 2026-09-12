"""Is the VR 3-point channel live once the encoder is actually in "teleop" mode?

Ledger bottle.md 34 measured the wrist target at 0.01 of the commanded motion
on BOTH checkpoints and closed the planner wrist channel as "a cue, not a
servo". The deploy source says otherwise: the target IS the servo input of
encoder mode 1 ("teleop"), and the deploy enters that mode on the RISING edge
of "the planner message carries `vr_position`" (zmq_manager.hpp:679). Our
commander sent that field from boot, so the edge fired before the planner
initialised, `Planner()` reset the mode to 0, and every reset would have done
the same (cpp:3770, zmq_manager.hpp:498). The deploy logs of both 2026-09-12
runs show exactly that order: "VR 3-point control enabled" BEFORE "Setting
planner_motion encode mode to 0", then one bounce to 'teleop' and back to 'g1'
for the rest of the run.

This probe enters the channel the way the PICO manager does -- planner up,
robot standing, then the field appears with the MEASURED pose as its first
value -- and measures, in order, each leg gated by the one before it:

  0. entry    : does entering "teleop" mode move anything? (palm + pelvis drift)
  1. reach    : right wrist POINT to [0.35, -0.22, -0.05] (pelvis frame), OPEN loop (kp=0)
                -> the ratio 34 reported as 0.01; the deploy log's
                "[Mode Filter] Switched to mode 'teleop'" is printed alongside
  2. servo    : the same target with the proportional correction on
  3. rotation : palm-down 60 deg about the pelvis y axis at the reach point
  4. squat    : IDEL_SQUAT 0.10 m with the world target FIXED
                -> how far the palm drifts in the world while the body sinks,
                i.e. whether the policy compensates the retreat and pitch that
                `project_squat` had to predict
  5. lean     : neck point pitched 0.30 rad forward, wrists held
                -> delivered waist pitch and what it does to the reach

    INSTANCE=0 . scripts/paths.sh; export CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0
    SONIC_POLICY=policy/sonic_v1_1 bash scripts/stack/start_stack.sh "vr3pt probe"
    SONIC_POLICY=policy/sonic_v1_1 PYTHONPATH=src $VENV_SIM_PY -m locomani.backends.sonic.diag_vr3pt \\
        --json outputs/diag/<run>/vr3pt_v1_1.json

Needs a live stack; changes no production constant. SONIC_POLICY must match
the stack's (the probe only records the name).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation

from locomani import paths
from locomani.backends.sonic.primitives import LocomotionMode, Primitives, StreamMode
from locomani.backends.sonic.streamed_arm import reanchor
from locomani.backends.sonic.teleop_arm import VR_OFFSET, TeleopArm
from locomani.sim.privileged_pub import EpisodeControlClient

SIDE = "right"
# Where the WRIST POINT (wrist_yaw_link + 0.18 m along its x) is asked to go, pelvis
# frame: diag_wrist.py's "plausible pre-grasp wrist point". The first run used a
# relative move of +0.15/-0.08 from the hanging rest pose, which puts the point
# 35 cm below the pelvis -- outside the standing envelope, so its 0.71 ratio mixed
# tracking with reach; an absolute grasp-band target separates the two.
REACH_POINT = np.array([0.35, -0.22, -0.05])
SETTLE = 3.0
SQUAT_M = 0.10
LEAN_RAD = 0.30
ROT_DEG = 60.0
LIVE_M = 0.03  # below this the palm did not follow the target at all


def deploy_log_modes(instance: int, since: int) -> list[str]:
    """Encoder-mode lines the deploy printed after byte `since`."""
    path = os.path.join(paths.session_root(instance), "deploy.log")
    try:
        with open(path, "rb") as fh:
            fh.seek(since)
            tail = fh.read().decode(errors="replace").splitlines()
    except OSError:
        return []
    keys = ("Mode Filter", "VR 3-point control", "encode mode", "Intended encoder")
    return [ln.strip() for ln in tail if any(k in ln for k in keys)]


def log_size(instance: int) -> int:
    path = os.path.join(paths.session_root(instance), "deploy.log")
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def restand(p: Primitives, reset: EpisodeControlClient, label: str) -> dict:
    """One clean standing start in plain planner mode -- NO VR field yet."""
    p.cmd.stream_mode = StreamMode.PLANNER
    p.cmd.upper_body = None
    p.cmd.set_upper_body_vel(None)
    p.cmd.reset_pose()  # also clears vr_targets
    reset.pin_arms()
    reset.reset(label=label)
    time.sleep(0.3)
    p.cmd.start_control()
    p.cmd.hold(3.0, mode=LocomotionMode.IDLE)
    reanchor(p, side=SIDE)
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
    return p.obs()


def pose_row(p: Primitives, arm: TeleopArm, s: dict) -> dict:
    palm = s["palms"][SIDE]
    pel = s["pelvis"]
    R = Primitives._rot(pel["quat"])
    return dict(
        palm_world=[float(v) for v in palm["pos"]],
        palm_root=[float(v) for v in p.world_to_root_full(palm["pos"], s)],
        point_root=[float(v) for v in arm.measured_point(SIDE, s)[0]],
        target_root=[float(v) for v in arm.target_point(SIDE, s)[0]] if SIDE in arm.target else None,
        pelvis_world=[float(v) for v in pel["pos"]],
        pelvis_pitch_deg=float(np.degrees(np.arcsin(-np.clip(R[2, 0], -1, 1)))),
        waist_pitch_rad=float(np.asarray(s["joints"])[14]),
        fallen=bool(s["fallen"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--instance", type=int, default=int(os.environ.get("INSTANCE", "0")))
    ap.add_argument("--legs", default="0,1,2,3,4,5", help="comma list of legs to run")
    args = ap.parse_args()
    legs = {int(x) for x in args.legs.split(",") if x.strip()}

    p = Primitives()
    reset = EpisodeControlClient()
    out = dict(policy=paths.SONIC_POLICY, side=SIDE, legs={}, log={})
    print(f"\npolicy {paths.SONIC_POLICY}   side {SIDE}   instance {args.instance}")

    s = restand(p, reset, "vr3pt")
    if s["fallen"]:
        print("could not stand -- stopping")
        return
    mark = log_size(args.instance)

    # -- 0. entry -----------------------------------------------------------
    arm = TeleopArm(p, SIDE, kp=0.0)
    s0 = arm.start()
    before = pose_row(p, arm, s0)
    arm.hold(2.0)
    s1 = p.obs()
    after = pose_row(p, arm, s1)
    drift_palm = float(np.linalg.norm(np.array(after["palm_world"]) - np.array(before["palm_world"])))
    drift_pel = float(np.linalg.norm(np.array(after["pelvis_world"]) - np.array(before["pelvis_world"])))
    modes = deploy_log_modes(args.instance, mark)
    out["log"]["entry"] = modes
    print(f"== 0. ENTRY: palm drift {drift_palm * 100:.1f} cm, pelvis drift {drift_pel * 100:.1f} cm over 2 s")
    for ln in modes:
        print(f"   deploy: {ln}")
    out["legs"]["entry"] = dict(before=before, after=after, palm_drift_m=drift_palm, pelvis_drift_m=drift_pel)
    if s1["fallen"]:
        print("FELL on entry -- stopping")
        _dump(out, args.json)
        return
    teleop_on = any("teleop" in ln for ln in modes)
    print(f"   encoder mode switched to 'teleop': {teleop_on}")

    # -- 1. reach, open loop -------------------------------------------------
    if 1 in legs:
        s = p.obs()
        R = Primitives._rot(s["pelvis"]["quat"])
        pel = np.array(s["pelvis"]["pos"])
        start_pt = arm.measured_point(SIDE, s)[0]
        # link target whose POINT lands on REACH_POINT, rotation unchanged
        rot_pf = R.T @ arm.target[SIDE]["rot"]
        goal = pel + R @ (REACH_POINT - rot_pf @ VR_OFFSET[SIDE])
        arm.move_to(goal, settle=SETTLE)
        s = p.obs()
        row = pose_row(p, arm, s)
        asked = float(np.linalg.norm(REACH_POINT - start_pt))
        got = float(np.linalg.norm(np.array(row["point_root"]) - start_pt))
        resid = float(np.linalg.norm(np.array(row["point_root"]) - np.array(row["target_root"])))
        out["legs"]["reach_open"] = dict(**row, asked_m=asked, moved_m=got, ratio=got / asked, residual_m=resid)
        out["log"]["reach"] = deploy_log_modes(args.instance, mark)
        print(
            f"== 1. REACH open loop: asked {asked:.3f} m, point moved {got:.3f} m, ratio {got / asked:.2f}, residual {resid:.3f} m"
        )
        if s["fallen"]:
            print("FELL -- stopping")
            _dump(out, args.json)
            return
        if got < LIVE_M:
            print(
                f"CHANNEL NOT LIVE (moved {got:.3f} < {LIVE_M}). Rotation/squat/lean would measure nothing. STOPPING."
            )
            _dump(out, args.json)
            _finish(p, arm)
            return

    # -- 2. servo ------------------------------------------------------------
    if 2 in legs:
        arm.kp = 0.6
        arm.hold(2.5)
        s = p.obs()
        row = pose_row(p, arm, s)
        resid = float(np.linalg.norm(np.array(row["point_root"]) - np.array(row["target_root"])))
        out["legs"]["reach_servo"] = dict(**row, residual_m=resid)
        print(f"== 2. SERVO (kp 0.6): residual {resid:.3f} m")

    # -- 3. rotation ---------------------------------------------------------
    if 3 in legs:
        s = p.obs()
        R = Primitives._rot(s["pelvis"]["quat"])
        rot0 = arm.target[SIDE]["rot"].copy()
        meas0 = Primitives._rot(s["palms"][SIDE]["quat"])
        # palm down: rotate about the pelvis y axis, expressed in the world
        axis_w = R[:, 1]
        rot1 = Rotation.from_rotvec(np.deg2rad(-ROT_DEG) * axis_w).as_matrix() @ rot0
        arm.move_to(arm.target[SIDE]["pos"], rot1, settle=SETTLE)
        s = p.obs()
        meas1 = Primitives._rot(s["palms"][SIDE]["quat"])
        delivered = float(np.degrees(np.linalg.norm(Rotation.from_matrix(meas0.T @ meas1).as_rotvec())))
        resid_rot = float(np.degrees(np.linalg.norm(Rotation.from_matrix(meas1.T @ rot1).as_rotvec())))
        row = pose_row(p, arm, s)
        out["legs"]["rotation"] = dict(**row, asked_deg=ROT_DEG, delivered_deg=delivered, residual_deg=resid_rot)
        print(
            f"== 3. ROTATION: asked {ROT_DEG:.0f} deg, delivered {delivered:.1f} deg (ratio {delivered / ROT_DEG:.2f}), residual {resid_rot:.1f} deg"
        )

    # -- 4. squat with the world target fixed --------------------------------
    if 4 in legs:
        s = p.obs()
        palm0 = np.array(s["palms"][SIDE]["pos"])
        pel0 = np.array(s["pelvis"]["pos"])
        yaw = float(s["pelvis"]["yaw"])
        fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        drop = arm.squat(SQUAT_M, settle=2.5)
        s = p.obs()
        palm1 = np.array(s["palms"][SIDE]["pos"])
        pel1 = np.array(s["pelvis"]["pos"])
        row = pose_row(p, arm, s)
        out["legs"]["squat"] = dict(
            **row,
            asked_drop_m=SQUAT_M,
            delivered_drop_m=drop,
            pelvis_retreat_m=float(-(pel1 - pel0) @ fwd),
            palm_world_drift_m=float(np.linalg.norm(palm1 - palm0)),
            palm_world_drift_xyz=[float(v) for v in (palm1 - palm0)],
            residual_m=arm.residual(s),
        )
        print(
            f"== 4. SQUAT {SQUAT_M:.2f} m: delivered {drop:.3f} m, pelvis retreat {out['legs']['squat']['pelvis_retreat_m'] * 100:.1f} cm, "
            f"pitch {row['pelvis_pitch_deg']:.1f} deg; palm WORLD drift {np.linalg.norm(palm1 - palm0) * 100:.1f} cm, residual {arm.residual(s) * 100:.1f} cm"
        )
        arm.stand(settle=2.0)

    # -- 5. lean via the neck point ------------------------------------------
    if 5 in legs:
        s = p.obs()
        wp0 = float(np.asarray(s["joints"])[14])
        palm0 = np.array(s["palms"][SIDE]["pos"])
        arm.lean = LEAN_RAD
        arm.hold(SETTLE)
        s = p.obs()
        row = pose_row(p, arm, s)
        wp1 = float(np.asarray(s["joints"])[14])
        out["legs"]["lean"] = dict(
            **row,
            asked_rad=LEAN_RAD,
            delivered_waist_rad=wp1 - wp0,
            ratio=(wp1 - wp0) / LEAN_RAD,
            palm_world_drift_m=float(np.linalg.norm(np.array(s["palms"][SIDE]["pos"]) - palm0)),
            residual_m=arm.residual(s),
        )
        print(
            f"== 5. LEAN {LEAN_RAD:.2f} rad via the neck: waist pitch {wp1 - wp0:+.3f} rad (ratio {(wp1 - wp0) / LEAN_RAD:.2f}), "
            f"pelvis pitch {row['pelvis_pitch_deg']:.1f} deg, palm world drift {out['legs']['lean']['palm_world_drift_m'] * 100:.1f} cm, residual {arm.residual(s) * 100:.1f} cm"
        )
        arm.lean = 0.0
        arm.hold(1.5)

    out["log"]["end"] = deploy_log_modes(args.instance, mark)
    out["servo_stats"] = arm.stats
    _finish(p, arm)
    _dump(out, args.json)


def _finish(p: Primitives, arm: TeleopArm) -> None:
    try:
        arm.stop()
    finally:
        p.cmd.pause_control()
        p.close()


def _dump(out: dict, path: str | None) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
