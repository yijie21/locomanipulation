"""Run the RGB-guided crate expert; reset and privileged evaluation stay outside its policy."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from locomani import paths
from locomani.backends.sonic import upper_ref
from locomani.backends.sonic.agent_server import HAND_OPEN, LocomotionMode
from locomani.backends.sonic.primitives import Primitives
from locomani.backends.sonic.streamed_arm import PlannerArm, StreamedArm, reanchor
from locomani.pipeline.execution_record import RoundResult
from locomani.primitives.arm_ik import ArmIK
from locomani.sim import ports
from locomani.sim.privileged_pub import EpisodeControlClient
from locomani.tasks.crate.bimanual_ik import BimanualIK
from locomani.tasks.crate.config import CrateTaskConfig
from locomani.tasks.crate.evaluation import evaluate
from locomani.tasks.crate.joint_reference import JointReference
from locomani.tasks.crate.perception import ApproachTarget, CratePerception
from locomani.tasks.crate.round_runner import prime
from locomani.tasks.crate.visual_approach import VisualApproach, initial_robot_xy
from locomani.tasks.crate.visual_low_reach import VisualLowReach
from locomani.tasks.crate.visual_policy import ExpertCommand, VisualCratePolicy
from locomani.tasks.crate.visual_recording import VisualRecording, record_attempt


def reset_ready(p, reset, meta, cfg, rng, solver, *, whole_body=False):
    """Place at the sampled table-edge distance; all initialization ends before recording."""
    jitter = rng.uniform(-cfg.position_jitter, cfg.position_jitter, size=2)
    yaw = np.pi / 2 + np.radians(rng.uniform(-cfg.yaw_jitter_deg, cfg.yaw_jitter_deg))
    start_distance = float(rng.uniform(cfg.start_edge_min, cfg.start_edge_max))
    start_xy = initial_robot_xy(meta, start_distance)
    p.cmd.reset_pose()
    p.cmd.set_hands(left=HAND_OPEN, right=HAND_OPEN)
    if whole_body:
        p.cmd.exit_streamed()
        p.cmd.hold(1.5, mode=LocomotionMode.IDLE, facing=p._facing(p.obs()))
    reset.pin_arms()
    reset.reset(
        objects={"blue_crate": {"dxy": jitter.tolist(), "yaw": float(yaw)}},
        robot={"dxy": (start_xy - meta["robot_start"]["xy"]).tolist(), "yaw": 0.0},
    )
    time.sleep(0.4)
    p.cmd.start_control()
    p.hold_yaw = 0.0
    if whole_body:
        upper_ref.shared().wait(3.0)
        p.cmd.hold(1.5, mode=LocomotionMode.IDLE, facing=(1, 0, 0))
        reanchor(p)
    p.cmd.hold(3.0, mode=LocomotionMode.IDLE, facing=(1, 0, 0))
    if not whole_body:
        p.cmd.heading_offset = float(p.pelvis_yaw(p.obs()))
    arm = PlannerArm(p, "right", send_vel=False, left_guard=False)
    arm.start()
    # The same ready pose is used for every basket placement, before task observations begin.
    ready = solver.solve(
        np.zeros(29), [[0.27, 0.22, 0.13], [0.27, -0.22, 0.13]], [np.eye(3), np.eye(3)], iterations=160
    )
    if whole_body:
        ready.joints[:] = 0
        ready.joints[15:22] = [-0.35, 0.2, 0, 0.8, 0, 0, 0]
        ready.joints[22:29] = [-0.35, -0.2, 0, 0.8, 0, 0, 0]
    q0 = arm.q.copy()
    for k in range(1, 126):
        u = k / 125
        arm.q = q0 + (3 * u * u - 2 * u * u * u) * (ready.joints - q0)
        arm._push(arm.q)
    arm.hold(1.2)
    return arm, {
        "object_dxy": jitter.tolist(),
        "object_yaw": float(yaw),
        "start_edge_distance": start_distance,
        "start_xy": start_xy.tolist(),
    }


def run_attempt(p, reset, camera, meta, config, directory, label, rng):
    cfg = config.visual
    model = ArmIK(paths.SCENE_XML)
    whole_body = config.carry.reach_control == "whole_body"
    solver = BimanualIK(
        model,
        waist_pitch_max=config.carry.waist_pitch_max if whole_body else cfg.waist_pitch_max,
        waist_dofs=config.carry.waist_dofs if whole_body else "pitch",
    )
    low = None
    perception = CratePerception(ArmIK(paths.SCENE_XML), meta["crate_visual"], max_reprojection_px=cfg.reprojection_px)
    policy = VisualCratePolicy(meta["crate_visual"]["grasp_points"], cfg)
    approach = VisualApproach(cfg)
    arm, placement = reset_ready(p, reset, meta, cfg, rng, solver, **({"whole_body": True} if whole_body else {}))
    reference = JointReference(arm.q, solver, speed=cfg.joint_speed, acceleration=cfg.joint_acceleration)
    initial = p.obs()
    if whole_body:
        distance = float(initial_robot_xy(meta, 0)[0] - initial["pelvis"]["pos"][0])
        if not 1.2 <= distance <= 1.5:
            arm.stop()
            return RoundResult(
                False,
                "invalid_initial_distance",
                False,
                {
                    "metrics": dict(
                        placement=placement,
                        initial_edge_distance=distance,
                        frames=0,
                        produced=False,
                        dataset_kind="painted_crate_rgb_low_reach_video_pilot",
                    )
                },
            )
    recorder = VisualRecording(directory / label)
    reset.start_video(label)
    started = time.time()
    estimate, last_image_time = None, 0.0
    last_camera_received = last_command_time = started
    last_motion = {"movement": np.zeros(3), "speed": -1.0}
    last_phase, reason, error = None, "no_camera_frames", None
    try:
        while time.time() - started < cfg.walk_timeout + cfg.timeout_seconds + 2:
            now = time.time()
            raw = p.obs()
            if raw["fallen"]:
                reason = "external_fall_abort"
                break
            message = camera.read()
            if message is None or not {"ego_view", "left_wrist", "right_wrist"} <= set(message["images"]):
                arm.mode = LocomotionMode.IDLE
                last_motion = {"movement": np.zeros(3), "speed": -1.0}
                if now - last_camera_received > cfg.vision_timeout:
                    reason = "camera_unavailable"
                    break
                arm._push(arm.q)
                continue
            now = time.time()
            last_camera_received = now
            stamp = max(float(v) for v in message["timestamps"].values())
            if now - min(float(v) for v in message["timestamps"].values()) > cfg.vision_timeout:
                reason = "stale_camera_abort"
                break
            if stamp == last_image_time:
                arm._push(arm.q, **({} if low is not None else last_motion))
                continue
            last_image_time = stamp
            images = message["images"]
            joints = np.asarray(raw["joints"], dtype=float)
            measured, rotations = solver.tool_poses(joints)
            new_estimate = perception.update(images, joints, stamp)
            estimate = new_estimate  # an occluded frame cannot refresh a true-state target
            openings = np.array([2 * (np.mean(raw["palms"][side]["hand_q"][:2]) + 0.023) for side in ("left", "right")])
            navigation = None
            if approach is not None:
                target = (
                    ApproachTarget(estimate.position, stamp, "full_rgb_pose")
                    if estimate is not None
                    else perception.approach_target(images, joints, stamp)
                )
                nav = approach.step(now, target, raw["pelvis"]["quat"], grasp_pose_available=estimate is not None)
                arm.mode = LocomotionMode.SLOW_WALK if nav.speed > 0 else LocomotionMode.IDLE
                yaw = p.pelvis_yaw(raw)
                direction = nav.direction.copy()
                direction[:2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]] @ direction[:2]
                q = arm.q.copy()
                last_motion = {"movement": direction, "speed": nav.speed if nav.speed > 0 else -1.0}
                arm._push(q, **last_motion)
                positions, reference_rotations = solver.tool_poses(q)
                command = ExpertCommand(positions, reference_rotations, 0.0, nav.phase, nav.reason, bool(nav.reason))
                navigation = {
                    "target": None if target is None else target.position,
                    "source": None if target is None else target.source,
                    "movement": direction,
                    "speed": nav.speed,
                    "relative_speed": approach.relative_speed,
                    "angular_speed": approach.angular_speed,
                    "ready": nav.ready,
                }
                if nav.ready:
                    approach = None
            elif whole_body:
                if low is None:
                    # Body planning cannot begin without a current RGB fit.
                    if estimate is None:
                        arm._push(arm.q)
                        continue
                    arm = StreamedArm(p, "right", ik=model, left_guard=False)
                    arm.start(hold_planner_pose=True)
                    handoff_state = p.obs()
                    low = VisualLowReach(
                        solver,
                        meta["crate_visual"],
                        config,
                        handoff_state["joints"],
                        handoff_state["pelvis"]["quat"],
                        reference_joints=arm.q,
                        reference_quaternion=arm.root_quat,
                    )
                    continue  # reacquire RGB/encoders after the bounded mode handoff
                body_command = low.step(now, estimate, joints, raw["pelvis"]["quat"], openings)
                q = body_command.joints
                closed = np.zeros_like(HAND_OPEN)
                closed[:2] = -0.023
                arm.hands = {
                    side: HAND_OPEN + body_command.closure * (closed - HAND_OPEN) for side in ("left", "right")
                }
                arm.q, arm.root_quat = q, body_command.root_quaternion
                arm._push(q, reference_velocity=body_command.velocity)
                points, attitudes = solver.tool_poses(q)
                command = ExpertCommand(
                    points,
                    attitudes,
                    body_command.closure,
                    body_command.phase,
                    body_command.reason,
                    body_command.finished,
                )
            else:
                command = policy.step(now, estimate, measured, rotations, openings)
                correction = np.clip(
                    cfg.tracking_gain * (command.positions - measured), -cfg.tracking_cap, cfg.tracking_cap
                )
                solution = solver.solve(
                    joints, command.positions + correction, command.rotations, iterations=18, posture=arm.q
                )
                q = reference.step(solution.joints, now - last_command_time)
                closed = np.zeros_like(HAND_OPEN)
                closed[:2] = -0.023
                arm.hands = {side: HAND_OPEN + command.closure * (closed - HAND_OPEN) for side in ("left", "right")}
                arm.q = q
                arm._push(q)
            last_command_time = now
            if command.phase != last_phase:
                print(f"[{label}] {command.phase} at {now - started:.2f}s", flush=True)
                last_phase = command.phase
            recorder.append(
                images,
                {
                    "wall_time": now,
                    "image_timestamps": message["timestamps"],
                    "state_seq": raw.get("seq"),
                    "state_wall_time": raw.get("wall"),
                    "simulation_time": raw.get("t"),
                    "joints": joints,
                    "base_quaternion": raw["pelvis"]["quat"],
                    "navigation": navigation,
                    "hand_openings": openings,
                    "reference_joints": q,
                    "root_reference": None if low is None else dict(quaternion=arm.root_quat, velocity=low.velocity),
                    "low_reach": None if low is None else low.diagnostics,
                    "tool_reference": command.positions,
                    "closure": command.closure,
                    "phase": command.phase,
                    "reason": command.reason,
                    "vision": None
                    if estimate is None
                    else {
                        "position": estimate.position,
                        "rotation": estimate.rotation,
                        "reprojection_px": estimate.reprojection_px,
                        "detections": estimate.detections,
                    },
                },
            )
            if command.finished:
                reason = command.reason
                break
    except Exception:
        reason, error = "runtime_error", traceback.format_exc()
    finally:
        control_reason = reason
        for cleanup in (recorder.close, reset.end_video, arm.stop):
            try:
                cleanup()
            except Exception:
                reason, error = "runtime_error", (error or "") + traceback.format_exc()
    try:
        final = p.obs()
        result = evaluate(initial, final, reason, lift_height=cfg.lift_height, max_tilt_deg=cfg.max_tilt_deg)
    except Exception:
        error = (error or "") + traceback.format_exc()
        result = {"ok": False, "why": "runtime_error"}
    result.update(
        label=label,
        seconds=time.time() - started,
        frames=recorder.frames,
        placement=placement,
        dataset_kind="painted_crate_rgb_low_reach_video_pilot" if whole_body else "painted_crate_rgb_walk_video_pilot",
        produced=False,
        control_reason=control_reason,
        intervention=control_reason.startswith("external_"),
    )
    if "pelvis" in initial:
        result["initial_edge_distance"] = float(initial_robot_xy(meta, 0)[0] - initial["pelvis"]["pos"][0])
    if error is not None:
        result["error"] = error
    print(f"[{label}] {result}", flush=True)
    return RoundResult(result["ok"], result["why"], result["produced"], {"metrics": result})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--label", default="visual_crate")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    config = CrateTaskConfig.load(args.config) if args.config else CrateTaskConfig()
    config = config.with_overrides(args.overrides)
    meta = json.loads(Path(paths.SCENE_META).read_text())
    if "crate_visual" not in meta:
        parser.error("build a distinct painted scene with python -m locomani.tasks.crate.visual_scene")
    from gear_sonic.camera.composed_camera import ComposedCameraClientSensor

    camera = ComposedCameraClientSensor(server_ip="localhost", port=ports.port("TF_CAMERA_PORT"))
    p, reset = Primitives(), EpisodeControlClient()
    directory = args.manifest.parent / "observations"
    directory.mkdir(parents=True, exist_ok=True)
    try:
        prime(p, reset)
        for index in range(args.rounds):
            label, seed = f"{args.label}_{index:02d}", args.seed + index
            result = record_attempt(
                args.manifest,
                label,
                seed,
                lambda: run_attempt(p, reset, camera, meta, config, directory, label, np.random.default_rng(seed)),
            )
            if result.why == "runtime_error":
                raise RuntimeError(result.execution["metrics"]["error"])
    finally:
        camera.close()
        reset.close()
        p.close()


if __name__ == "__main__":
    main()
