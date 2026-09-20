"""Run the camera-guided crate expert with explicit RGB or simulation-assisted pose sources."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from locomani import paths
from locomani.backends.sonic import upper_ref
from locomani.backends.sonic.agent_server import HAND_OPEN, LocomotionMode
from locomani.backends.sonic.pose_preview import PosePreview
from locomani.backends.sonic.primitives import Primitives, upper_body_from_mj
from locomani.backends.sonic.streamed_arm import PlannerArm, StreamedArm, reanchor
from locomani.pipeline.execution_record import RoundResult
from locomani.pipeline.provenance import require_pose_ownership
from locomani.primitives.arm_ik import ArmIK
from locomani.sim import ports
from locomani.sim.privileged_pub import EpisodeControlClient
from locomani.tasks.crate.assisted_guidance import AssistedGuidance
from locomani.tasks.crate.bimanual_ik import BimanualIK
from locomani.tasks.crate.config import CrateTaskConfig
from locomani.tasks.crate.coordinated_reach import CoordinatedReach
from locomani.tasks.crate.evaluation import evaluate
from locomani.tasks.crate.grasp_region import GRASP_POINTS
from locomani.tasks.crate.joint_reference import JointReference
from locomani.tasks.crate.perception import ApproachTarget, CratePerception
from locomani.tasks.crate.round_runner import prime
from locomani.tasks.crate.visible_target import VisibleTarget
from locomani.tasks.crate.visual_approach import ApproachPreparation, VisualApproach, initial_robot_xy
from locomani.tasks.crate.visual_low_reach import VisualLowReach
from locomani.tasks.crate.visual_policy import ExpertCommand, VisualCratePolicy
from locomani.tasks.crate.visual_recording import VisualRecording, record_attempt


def prepare_stationary_ready(
    p,
    reset,
    cfg,
    solver,
    *,
    placement: Callable[[str], None],
    whole_body: bool = False,
    on_phase: Callable[[str, dict], None] | None = None,
):
    """Run the visual expert's stationary controller preparation and admission.

    ``placement`` owns scene-specific reset contents. This lets floor-only
    diagnostics use the original controller sequence without inventing a crate
    or maintaining another ready-state initializer.
    """

    def phase(name: str, **details) -> None:
        if on_phase is not None:
            on_phase(name, details)

    phase("preparation_started", whole_body=whole_body)
    p.cmd.reset_pose()
    p.cmd.set_hands(left=HAND_OPEN, right=HAND_OPEN)
    if whole_body:
        p.cmd.exit_streamed()
        p.cmd.hold(1.5, mode=LocomotionMode.IDLE, facing=p._facing(p.obs()))
        phase("pre_reset_idle_complete", duration_s=1.5)
    reset.pin_arms()
    phase("simulation_pins_released")
    phase("reset_commanded", reset_id="initial_placement")
    placement("initial_placement")
    time.sleep(0.4)
    phase("initial_placement_settled", duration_s=0.4)
    p.cmd.start_control()
    p.hold_yaw = 0.0
    phase("control_started")
    if whole_body:
        upper_ref.shared().wait(3.0)
        phase("upper_reference_ready", timeout_s=3.0)
        p.cmd.hold(1.5, mode=LocomotionMode.IDLE, facing=(1, 0, 0))
        phase("reanchor_started")
        reanchor(p)
        phase("reanchor_complete")
    p.cmd.hold(3.0, mode=LocomotionMode.IDLE, facing=(1, 0, 0))
    if whole_body:
        phase("post_reanchor_idle_complete", before_s=1.5, after_s=3.0)
    if not whole_body:
        p.cmd.heading_offset = float(p.pelvis_yaw(p.obs()))
    arm = PlannerArm(p, "right", send_vel=False, left_guard=False)
    arm.start()
    phase("planner_arm_started", settle_seconds="default")

    # The original visual expert's ready pose and transition are deliberately
    # kept here as one owner. Whole-body operation overrides the solver's
    # single-arm ready output exactly as it did before this extraction.
    ready = solver.solve(
        np.zeros(29),
        [[0.27, 0.22, 0.13], [0.27, -0.22, 0.13]],
        [np.eye(3), np.eye(3)],
        iterations=160,
    )
    if whole_body:
        ready.joints[:] = 0
        ready.joints[15:22] = [-0.35, 0.2, 0, 0.8, 0, 0, 0]
        ready.joints[22:29] = [-0.35, -0.2, 0, 0.8, 0, 0, 0]
    q0 = arm.q.copy()
    phase("ready_transition_started", frames=125)
    for k in range(1, 126):
        u = k / 125
        arm.q = q0 + (3 * u * u - 2 * u * u * u) * (ready.joints - q0)
        arm._push(arm.q)
    phase("ready_transition_complete", frames=125)
    arm.hold(1.2)
    phase("ready_hold_complete", duration_s=1.2)

    stance_ready = True
    state = None
    quiet_since = None
    if whole_body:
        # Establish the sampled episode only after the ready reference exists.
        phase("reset_commanded", reset_id="prepared_placement")
        placement("prepared_placement")
        time.sleep(0.3)
        phase("prepared_placement_settled", duration_s=0.3)
        arm.start(upper_reference=upper_body_from_mj(ready.joints), settle_seconds=0.0)
        phase("planner_reference_reseeded", settle_seconds=0.0)

        state = p.obs()
        began = previous = float(state["t"])
        progress_wall, quiet_since, stance_ready = time.time(), None, False
        phase("quiet_admission_started")
        while state["t"] - began < cfg.walk_timeout:
            stamp = float(state["t"])
            if stamp < previous or time.time() - progress_wall > cfg.vision_timeout:
                break
            if stamp > previous:
                previous, progress_wall = stamp, time.time()
            _w, x, y, _z = state["pelvis"]["quat"]
            upright = 1 - 2 * (x * x + y * y) >= np.cos(np.radians(cfg.upright_tolerance_deg))
            quiet = upright and state["pelvis"]["pos"][2] >= 0.72 and np.linalg.norm(state["pelvis"]["vel"]) < 0.06
            quiet_since = (stamp if quiet_since is None else quiet_since) if quiet else None
            if quiet_since is not None and stamp - quiet_since >= cfg.settle_seconds:
                stance_ready = True
                break
            arm._push(arm.q)
            state = p.obs()
        phase(
            "quiet_admitted" if stance_ready else "quiet_admission_failed",
            quiet_duration_sim_s=(float(state["t"]) - quiet_since if quiet_since is not None else 0.0),
        )

    return arm, {
        "initialization_resets": 2 if whole_body else 1,
        "initial_stance_ready": stance_ready,
        "ready_reference": ready.joints.copy(),
        "admission_state": state,
    }


def reset_ready(p, reset, meta, cfg, rng, solver, *, whole_body=False):
    """Place at the sampled table-edge distance; all initialization ends before recording."""
    jitter = rng.uniform(-cfg.position_jitter, cfg.position_jitter, size=2)
    yaw = np.pi / 2 + np.radians(rng.uniform(-cfg.yaw_jitter_deg, cfg.yaw_jitter_deg))
    start_distance = float(rng.uniform(cfg.start_edge_min, cfg.start_edge_max))
    start_xy = initial_robot_xy(meta, start_distance)

    def placement(_reset_id: str) -> None:
        reset.reset(
            objects={"blue_crate": {"dxy": jitter.tolist(), "yaw": float(yaw)}},
            robot={"dxy": (start_xy - meta["robot_start"]["xy"]).tolist(), "yaw": 0.0},
        )

    arm, preparation = prepare_stationary_ready(
        p,
        reset,
        cfg,
        solver,
        placement=placement,
        whole_body=whole_body,
    )
    preparation.pop("admission_state")
    preparation.pop("ready_reference")
    return arm, {
        **preparation,
        "object_dxy": jitter.tolist(),
        "object_yaw": float(yaw),
        "start_edge_distance": start_distance,
        "start_xy": start_xy.tolist(),
    }


def run_attempt(p, reset, camera, meta, config, directory, label, rng, *, record=False):
    cfg = config.visual
    model = ArmIK(paths.SCENE_XML)
    whole_body = config.carry.reach_control == "whole_body"
    solver = BimanualIK(
        model,
        waist_pitch_max=config.carry.waist_pitch_max if whole_body else cfg.waist_pitch_max,
        waist_dofs=config.carry.waist_dofs if whole_body else "pitch",
    )
    low = None
    assisted = cfg.pose_source == "assisted"
    geometry = {"grasp_points": GRASP_POINTS} if assisted else meta["crate_visual"]
    dataset_kind = (
        "ordinary_crate_assisted_low_reach_video_pilot"
        if assisted
        else "painted_crate_rgb_low_reach_video_pilot"
        if whole_body
        else "painted_crate_rgb_walk_video_pilot"
    )
    policy = VisualCratePolicy(geometry["grasp_points"], cfg)
    approach = VisualApproach(cfg)
    stand_attempts = approach.stand_attempts  # the list outlives the approach object
    arm, placement = reset_ready(p, reset, meta, cfg, rng, solver, **({"whole_body": True} if whole_body else {}))
    if not placement["initial_stance_ready"]:
        arm.stop()
        return RoundResult(
            False,
            "initial_stance_unstable",
            False,
            {"metrics": dict(placement=placement, frames=0, produced=False, dataset_kind=dataset_kind)},
        )
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
                        dataset_kind=dataset_kind,
                    )
                },
            )
    perception = (
        AssistedGuidance(ArmIK(paths.SCENE_XML))
        if assisted
        else CratePerception(ArmIK(paths.SCENE_XML), geometry, max_reprojection_px=cfg.reprojection_px)
    )
    recorder = VisualRecording(directory / label)
    target_cache = VisibleTarget()
    preparation = ApproachPreparation(solver, cfg, arm.q) if assisted and whole_body else None
    # The transport journals every reference send, even during mode priming.
    if hasattr(p, "cmd"):
        p.cmd.reference_observer = recorder.observe_reference
    reset.start_video(label)
    produced = False
    if record:
        # 'c': the exporter goes IDLE -> RECORDING. Every initialization step
        # ended in reset_ready, so frame 0 of the episode is the task. The
        # matching save/discard key is sent once the attempt is evaluated.
        p.cmd.key("c")
        produced = True
    started = time.time()
    estimate, last_image_time = None, 0.0
    last_camera_received = last_command_time = started
    last_motion = {"movement": np.zeros(3), "speed": -1.0}
    last_phase, reason, error = None, "no_camera_frames", None
    final = initial
    last_control_time = None
    images, message = None, None
    last_simulation_time, last_progress_wall = float(initial["t"]) if assisted else 0.0, started
    try:
        while (
            float(final["t"]) - float(initial["t"]) if assisted else time.time() - started
        ) < cfg.walk_timeout + cfg.timeout_seconds + 2:
            now = time.time()
            raw = p.obs()
            final = raw  # evaluate the observed attempt, before mode exit or video finalization
            motion_clock = {"motion_time": float(raw["t"])} if assisted else {}
            if assisted:
                if raw["t"] < last_simulation_time:
                    reason = "simulation_clock_reversed"
                    break
                if raw["t"] > last_simulation_time:
                    last_simulation_time, last_progress_wall = float(raw["t"]), now
                elif now - last_progress_wall > cfg.vision_timeout:
                    reason = "simulation_stalled"
                    break
            if raw["fallen"]:
                reason = "external_fall_abort"
                break
            if assisted and raw["t"] == last_control_time:
                time.sleep(0.002)
                continue  # no new physical sample: do not append a contradictory held pose
            last_control_time = raw.get("t")
            incoming = camera.read()
            fresh_image = False
            if incoming is not None and {"ego_view", "left_wrist", "right_wrist"} <= set(incoming["images"]):
                last_camera_received = now
                stamp = max(float(v) for v in incoming["timestamps"].values())
                if now - min(float(v) for v in incoming["timestamps"].values()) > cfg.vision_timeout:
                    reason = "stale_camera_abort"
                    break
                fresh_image = stamp != last_image_time
                if fresh_image:
                    message, images = incoming, incoming["images"]
                    last_image_time = stamp
                    estimate = perception.update(images, raw if assisted else np.asarray(raw["joints"]), stamp)
                    if assisted:
                        target_cache.update(estimate, raw)
            if now - last_camera_received > cfg.vision_timeout:
                reason = "camera_unavailable"
                break
            if assisted:
                estimate = target_cache.at(raw)
            joints = np.asarray(raw["joints"], dtype=float)
            measured, rotations = solver.tool_poses(joints)
            recorder.context = dict(
                state_seq=raw.get("seq"),
                simulation_time=raw.get("t"),
                state_wall_time=raw.get("wall"),
                image_frame=(recorder.frames if fresh_image else recorder.frames - 1) if images is not None else None,
                image_timestamp=last_image_time or None,
                phase=last_phase or "walk",
            )
            if (not assisted and not fresh_image) or images is None:
                arm._push(arm.q, **({} if low is not None else last_motion))
                continue
            openings = np.array([2 * (np.mean(raw["palms"][side]["hand_q"][:2]) + 0.023) for side in ("left", "right")])
            navigation = None
            if approach is not None:
                target = (
                    ApproachTarget(
                        estimate.position,
                        estimate.timestamp,
                        "visible_simulation_pose" if assisted else "full_rgb_pose",
                    )
                    if estimate is not None
                    else None
                    if assisted
                    else perception.approach_target(images, joints, stamp)
                )
                nav = approach.step(
                    now,
                    target,
                    raw["pelvis"]["quat"],
                    grasp_pose_available=estimate is not None,
                    **({"base_velocity": raw["pelvis"]["vel"]} if assisted else {}),
                    **motion_clock,
                )
                admission = None
                if nav.ready and assisted:
                    probe = CoordinatedReach(
                        solver, config, raw, reference_joints=joints, reference_quaternion=raw["pelvis"]["quat"]
                    )
                    check = probe.step(now, estimate, raw, openings, **motion_clock)
                    admission = probe.diagnostics
                    if check.reason == "unsupported_assisted_reach":
                        if approach.retry_closer():
                            nav = approach.step(
                                now, target, raw["pelvis"]["quat"], grasp_pose_available=True, **motion_clock
                            )
                        else:
                            nav = replace(nav, ready=False, reason="approach_unsupported_reach")
                arm.mode = LocomotionMode.SLOW_WALK if nav.speed > 0 else LocomotionMode.IDLE
                yaw = p.pelvis_yaw(raw)
                direction = nav.direction.copy()
                direction[:2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]] @ direction[:2]
                q, velocity = (
                    (arm.q.copy(), np.zeros(29)) if preparation is None else preparation.step(float(raw["t"]), estimate)
                )
                arm.q = q
                recorder.context["phase"] = nav.phase
                last_motion = {"movement": direction, "speed": nav.speed if nav.speed > 0 else -1.0}
                arm._push(q, reference_velocity=velocity, **last_motion)
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
                    "reach_admission": admission,
                    "stand_attempts": list(stand_attempts),
                }
                if nav.ready:
                    approach = None
            elif whole_body:
                if low is None:
                    # Body planning needs a currently admitted visible target.
                    if estimate is None:
                        arm._push(arm.q)
                        continue
                    recorder.context["phase"] = "handoff"
                    if fresh_image:
                        recorder.append(
                            images,
                            dict(
                                wall_time=now,
                                image_timestamps=message["timestamps"],
                                state_seq=raw.get("seq"),
                                simulation_time=raw.get("t"),
                                joints=joints,
                                base_quaternion=raw["pelvis"]["quat"],
                                base_position=raw["pelvis"]["pos"],
                                reference_joints=arm.q,
                                phase="handoff",
                            ),
                        )
                    arm = StreamedArm(
                        p,
                        "right",
                        ik=model,
                        left_guard=False,
                        preview=(
                            PosePreview(acceleration=cfg.joint_acceleration, joint_limits=(solver.lo, solver.hi))
                            if assisted
                            else None
                        ),
                    )
                    arm.start(hold_planner_pose=True)
                    handoff_state = p.obs()
                    if assisted:
                        low = CoordinatedReach(
                            solver,
                            config,
                            handoff_state,
                            reference_joints=arm.q,
                            reference_quaternion=arm.root_quat,
                        )
                    else:
                        low = VisualLowReach(
                            solver,
                            geometry,
                            config,
                            handoff_state["joints"],
                            handoff_state["pelvis"]["quat"],
                            reference_joints=arm.q,
                            reference_quaternion=arm.root_quat,
                        )
                    continue  # reacquire RGB/encoders after the bounded mode handoff
                body_command = (
                    low.step(now, estimate, raw, openings, **motion_clock)
                    if assisted
                    else low.step(now, estimate, joints, raw["pelvis"]["quat"], openings)
                )
                q = body_command.joints
                closed = np.zeros_like(HAND_OPEN)
                closed[:2] = -0.023
                arm.hands = {
                    side: HAND_OPEN + body_command.closure * (closed - HAND_OPEN) for side in ("left", "right")
                }
                arm.q, arm.root_quat = q, body_command.root_quaternion
                recorder.context["phase"] = body_command.phase
                if not body_command.finished or body_command.reason == "completed":
                    arm._push(
                        q,
                        reference_velocity=body_command.velocity,
                        **({"reference_time": raw["t"]} if assisted else {}),
                    )
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
            if fresh_image:
                recorder.append(
                    images,
                    {
                        "wall_time": now,
                        "image_timestamps": message["timestamps"],
                        "state_seq": raw.get("seq"),
                        "state_wall_time": raw.get("wall"),
                        "simulation_time": raw.get("t"),
                        "motion_clock": "simulation" if assisted else "wall",
                        "joints": joints,
                        "base_quaternion": raw["pelvis"]["quat"],
                        "base_position": raw["pelvis"]["pos"],
                        "base_velocity": raw["pelvis"].get("vel"),
                        "guidance": perception.diagnostics if assisted else {"pose_source": "rgb_markers"},
                        "navigation": navigation,
                        "hand_openings": openings,
                        "reference_joints": q,
                        "command_id": recorder.command_count - 1,
                        "root_reference": None
                        if low is None
                        else dict(quaternion=arm.root_quat, velocity=low.velocity),
                        "low_reach": None if low is None else low.diagnostics,
                        "tool_reference": command.positions,
                        "closure": command.closure,
                        "phase": command.phase,
                        "reason": command.reason,
                        "vision": None
                        if estimate is None
                        else {
                            "source": "simulation" if assisted else "rgb_markers",
                            "position": estimate.position,
                            "rotation": estimate.rotation,
                            "reprojection_px": estimate.reprojection_px,
                            "detections": estimate.detections,
                        },
                    },
                )
            if assisted and command.phase == "hold" and recorder.training_tail is None:
                recorder.mark_training_end(raw)
            if command.finished:
                reason = command.reason
                break
    except Exception:
        reason, error = "runtime_error", traceback.format_exc()
    finally:
        control_reason = reason
        if hasattr(p, "cmd"):
            p.cmd.reference_observer = None
        for cleanup in (recorder.close, reset.end_video, arm.stop):
            try:
                cleanup()
            except Exception:
                reason, error = "runtime_error", (error or "") + traceback.format_exc()
    try:
        result = evaluate(initial, final, reason, lift_height=cfg.lift_height, max_tilt_deg=cfg.max_tilt_deg)
    except Exception:
        error = (error or "") + traceback.format_exc()
        result = {"ok": False, "why": "runtime_error"}
    result.update(
        label=label,
        seconds=time.time() - started,
        frames=recorder.frames,
        placement=placement,
        dataset_kind=dataset_kind,
        produced=produced,
        task=config.prompt,
        target="blue_crate",
        control_reason=control_reason,
        intervention=control_reason.startswith("external_"),
        evaluation_state_seq=final.get("seq"),
    )
    if assisted:
        result["reference_stream"] = dict(
            clock="simulation",
            rate_hz=1 / PosePreview.dt,
            future_frames=PosePreview.future_frames,
            history_frames=PosePreview.history_frames,
            prediction="shared_tangent_braking",
        )
    result["command_count"] = recorder.command_count
    result["stand_attempts"] = list(stand_attempts)
    if result["ok"] and recorder.training_tail is not None:
        result["training_tail"] = recorder.training_tail
        # The converter's trim reads row["execution"]["metrics"]["training_tail"]
        # (verification_tail.training_prefix_length); the same marker under the
        # single-arm line's path, so one converter serves both lines.
        result["execution"] = {"metrics": {"training_tail": dict(recorder.training_tail)}}
    if produced:
        # 'c' saves, 'x' discards; the exporter writes the episode from its own
        # loop, and the 2 s IDLE hold gives it rows past the training boundary.
        # Sent on every path that started recording, including runtime errors,
        # so the exporter never stays in RECORDING and the manifest join holds.
        p.cmd.key("c" if result["ok"] else "x")
        p.cmd.hold(2.0, mode=LocomotionMode.IDLE, facing=(1, 0, 0))
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
    parser.add_argument(
        "--record",
        action="store_true",
        help="drive the data exporter: 'c' at task start, 'c'/'x' after evaluation; refused without an exporter",
    )
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.record:
        ports.require_exporter("--record")
    config = CrateTaskConfig.load(args.config) if args.config else CrateTaskConfig()
    config = config.with_overrides(args.overrides)
    if config.carry.reach_control == "whole_body":
        require_pose_ownership()
    meta = json.loads(Path(paths.SCENE_META).read_text())
    if config.visual.pose_source == "markers" and "crate_visual" not in meta:
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
                lambda: run_attempt(
                    p, reset, camera, meta, config, directory, label, np.random.default_rng(seed), record=args.record
                ),
            )
            if result.why == "runtime_error":
                raise RuntimeError(result.execution["metrics"]["error"])
    finally:
        camera.close()
        reset.close()
        p.close()


if __name__ == "__main__":
    main()
