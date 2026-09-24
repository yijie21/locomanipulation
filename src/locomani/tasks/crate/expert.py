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
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from locomani import paths
from locomani.backends.sonic import upper_ref
from locomani.backends.sonic.agent_server import HAND_OPEN, LocomotionMode
from locomani.backends.sonic.config import DEFAULT_BACKEND
from locomani.backends.sonic.pose_preview import PosePreview
from locomani.backends.sonic.primitives import DT, Primitives, upper_body_from_mj
from locomani.backends.sonic.streamed_arm import PlannerArm, StreamedArm, reanchor
from locomani.pipeline.execution_record import RoundResult
from locomani.pipeline.provenance import require_pose_ownership
from locomani.primitives.arm_ik import ArmIK
from locomani.primitives.foot_watch import FootWatch, GuardBrake
from locomani.sim import ports
from locomani.sim.privileged_pub import EpisodeControlClient
from locomani.tasks.crate.assisted_guidance import AssistedGuidance
from locomani.tasks.crate.bimanual_ik import BimanualIK
from locomani.tasks.crate.config import CrateTaskConfig
from locomani.tasks.crate.coordinated_reach import CoordinatedReach, CoordinatedTarget
from locomani.tasks.crate.evaluation import evaluate
from locomani.tasks.crate.ghost_trace import GhostTargetTrace
from locomani.tasks.crate.grasp_region import GRASP_BELOW_RIM_M, RimRegion, grasp_points_for, target_key
from locomani.tasks.crate.joint_reference import JointReference
from locomani.tasks.crate.perception import ApproachTarget, CratePerception
from locomani.tasks.crate.retreat import retreat
from locomani.tasks.crate.round_runner import prime
from locomani.tasks.crate.visible_target import VisibleTarget
from locomani.tasks.crate.visual_approach import (
    ApproachPreparation,
    VisualApproach,
    approach_heading,
    edge_distance,
    initial_robot_xy,
    near_edge,
    scene_object_yaw,
)
from locomani.tasks.crate.visual_low_reach import VisualLowReach
from locomani.tasks.crate.visual_policy import ExpertCommand, VisualCratePolicy
from locomani.tasks.crate.visual_recording import VisualRecording, record_attempt


class PlaceholderCamera:
    """backend.render offline: the sim renders nothing, so every read is three black frames at the published
    camera size. Only the assisted expert may run on it (its pose and visibility are geometric).

    Frames change at the live stream's ~30 Hz, not on every read, so `fresh_image` toggles like live. The stamp is
    the read time: the consumer must sample its own `now` after reading, or the frame is "from the future" and the
    reach rejects it (per-read stamps ended every offline round `vision_lost`; the 30 Hz stamp still froze the
    reach clock at 0 for two offline pilots, 2026-09-23/24)."""

    shapes = dict.fromkeys(("ego_view", "left_wrist", "right_wrist"), (480, 640, 3))  # sim CAMERA_CONFIGS
    period = 1 / 30

    def __init__(self):
        self.stamp = 0.0

    def read(self):
        now = time.time()
        if now - self.stamp >= self.period:
            self.stamp = now
        return {
            "images": {n: np.zeros(s, np.uint8) for n, s in self.shapes.items()},
            "timestamps": dict.fromkeys(self.shapes, self.stamp),
        }

    def close(self):
        pass


def stand_up(p, reset, *, placement, whole_body, phase, heading):
    """Rest arms, place the scene, start control and stand in IDLE (>= 3 s) facing `heading`; arms stay the planner's."""
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
    p.hold_yaw = float(heading)
    facing = (float(np.cos(heading)), float(np.sin(heading)), 0.0)
    phase("control_started")
    if whole_body:
        upper_ref.shared().wait(3.0)
        phase("upper_reference_ready", timeout_s=3.0)
        p.cmd.hold(1.5, mode=LocomotionMode.IDLE, facing=facing)
        phase("reanchor_started")
        reanchor(p)
        phase("reanchor_complete")
    p.cmd.hold(3.0, mode=LocomotionMode.IDLE, facing=facing)
    if whole_body:
        phase("post_reanchor_idle_complete", before_s=1.5, after_s=3.0)
    if not whole_body:
        p.cmd.heading_offset = float(p.pelvis_yaw(p.obs()))


class HangingWalk:
    """approach_stop=guard's walk: plain planner frames, the walking policy's own swinging arms -- the condition the
    guard brake was calibrated in (walk scene, diag_brake.trial). Arms raised during the walk put the fingers into
    the warehouse box 0.9-1.6 s after the start in 6/10 rounds (box_room_grasp_20260923). Implements the part of
    PlannerArm the approach loop drives; the arms are readied in place after the brake (prepare_stationary_ready)."""

    def __init__(self, p, heading):
        self.p, self.mode = p, LocomotionMode.IDLE
        self._face = (float(np.cos(heading)), float(np.sin(heading)), 0.0)
        self.q = np.asarray(p.obs()["joints"], dtype=float).copy()
        self.next_t = time.time()

    def _push(self, _q=None, *, reference_velocity=None, movement=(0.0, 0.0, 0.0), speed=-1.0) -> None:
        self.p.cmd.frame(mode=self.mode, facing=self._face, movement=movement, speed=speed)
        self.next_t += DT
        lag = self.next_t - time.time()
        if lag > 0:
            time.sleep(lag)
        else:
            self.next_t = time.time()

    def stop(self) -> None:
        pass


def lifted_ready(solver, cfg, start):
    """Ready pose that lifts the hands in place: tool points at cfg.ready_tool (lateral mirrored), palms pitched
    cfg.ready_pitch_deg, solved from the walking arms with the waist locked. The forward pose put the fingertips
    0.42 m ahead, on the warehouse box's near wall at stances under 0.51 m (2026-09-24); this keeps them within
    0.20 m through the joint blend, so the near wall is first met by the reach, which sees it."""
    forward, lateral, up = (float(v) for v in cfg.ready_tool)
    attitude = Rotation.from_euler("y", cfg.ready_pitch_deg, degrees=True).as_matrix()
    lo, hi = solver.lo.copy(), solver.hi.copy()
    lo[12:15], hi[12:15] = 0.0, 0.0
    start = np.asarray(start, dtype=float)
    ready = solver.solve(
        start,
        [[forward, lateral, up], [forward, -lateral, up]],
        [attitude, attitude],
        iterations=250,
        rot_weight=0.15,
        posture=start,
        posture_weight=0.01,
        joint_limits=(lo, hi),
    )
    if np.max(ready.errors) > 0.02:
        raise SystemExit(
            f"visual.ready_pose lifted: tool targets {cfg.ready_tool} missed by {np.round(ready.errors, 3).tolist()} m "
            f"(elbow_min_deg {cfg.elbow_min_deg}); the pose must be delivered, not approximated"
        )
    return ready


def prepare_stationary_ready(
    p,
    reset,
    cfg,
    solver,
    *,
    placement: Callable[[str], None],
    whole_body: bool = False,
    on_phase: Callable[[str, dict], None] | None = None,
    heading: float = 0.0,
):
    """Run the visual expert's stationary controller preparation and admission.

    ``placement`` owns scene-specific reset contents. This lets floor-only
    diagnostics use the original controller sequence without inventing a crate
    or maintaining another ready-state initializer.
    """

    def phase(name: str, **details) -> None:
        if on_phase is not None:
            on_phase(name, details)

    stand_up(p, reset, placement=placement, whole_body=whole_body, phase=phase, heading=heading)
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
    if whole_body and cfg.ready_pose == "lifted":
        ready = lifted_ready(solver, cfg, arm.q)
    elif whole_body:
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


def start_placement(reset, meta, cfg, rng):
    """The sampled episode's reset: object jitter and yaw, robot start pose. Returns (placement, its record)."""
    jitter = rng.uniform(-cfg.position_jitter, cfg.position_jitter, size=2)
    # The object keeps the scene's own yaw (pi/2 in the synthetic table scenes,
    # 0 in the logistics room: long axis across the walk-in either way).
    yaw = scene_object_yaw(meta) + np.radians(rng.uniform(-cfg.yaw_jitter_deg, cfg.yaw_jitter_deg))
    heading = approach_heading(meta)
    base = np.asarray(meta["robot_start"]["xy"], dtype=float)
    if cfg.start_pose == "scene":
        start_xy = base + cfg.start_advance * np.array([np.cos(heading), np.sin(heading)])
        start_distance = edge_distance(meta, start_xy)
    else:
        start_distance = float(rng.uniform(cfg.start_edge_min, cfg.start_edge_max))
        start_xy = initial_robot_xy(meta, start_distance)

    def placement(_reset_id: str) -> None:
        reset.reset(
            objects={target_key(meta): {"dxy": jitter.tolist(), "yaw": float(yaw)}},
            robot={"dxy": (start_xy - base).tolist(), "yaw": heading},
        )

    return placement, {
        "object_dxy": jitter.tolist(),
        "object_yaw": float(yaw),
        "start_edge_distance": start_distance,
        "start_xy": start_xy.tolist(),
    }


def reset_ready(p, reset, meta, cfg, rng, solver, *, whole_body=False):
    """Place at the sampled table-edge distance; all initialization ends before recording."""
    placement, record = start_placement(reset, meta, cfg, rng)
    arm, preparation = prepare_stationary_ready(
        p,
        reset,
        cfg,
        solver,
        placement=placement,
        whole_body=whole_body,
        heading=approach_heading(meta),
    )
    preparation.pop("admission_state")
    preparation.pop("ready_reference")
    return arm, {**preparation, **record}


def reset_hanging(p, reset, meta, cfg, rng):
    """approach_stop=guard: place and stand with the planner's own arms; the ready arms come after the brake."""
    placement, record = start_placement(reset, meta, cfg, rng)
    heading = approach_heading(meta)
    stand_up(p, reset, placement=placement, whole_body=True, phase=lambda *_a, **_k: None, heading=heading)
    # Standing up drifts the robot ~11 cm and turns it ~12 deg off the walk-in axis (first hanging round,
    # 2026-09-23); re-place it, as prepare_stationary_ready's second reset does, and wait out the landing.
    placement("prepared_placement")
    time.sleep(0.3)
    walk = HangingWalk(p, heading)
    state = p.obs()
    began, quiet_since, stance_ready = float(state["t"]), None, False
    while float(state["t"]) - began < cfg.walk_timeout and not state["fallen"]:
        _w, x, y, _z = state["pelvis"]["quat"]
        upright = 1 - 2 * (x * x + y * y) >= np.cos(np.radians(cfg.upright_tolerance_deg))
        quiet = upright and state["pelvis"]["pos"][2] >= 0.72 and np.linalg.norm(state["pelvis"]["vel"]) < 0.06
        quiet_since = (float(state["t"]) if quiet_since is None else quiet_since) if quiet else None
        if quiet_since is not None and float(state["t"]) - quiet_since >= 1.0:
            stance_ready = True
            break
        walk._push()
        state = p.obs()
    offset = np.asarray(state["pelvis"]["pos"][:2]) - np.asarray(record["start_xy"])
    return walk, {
        "initialization_resets": 2,
        "initial_stance_ready": stance_ready,
        "start_offset_m": offset.tolist(),
        "start_yaw_error_deg": float(np.degrees((p.pelvis_yaw(state) - heading + np.pi) % (2 * np.pi) - np.pi)),
        **record,
    }


def box_front_line(state, key, box, axis):
    """Distance along `axis` of the box's near face at its observed pose (world origin), from its `box` footprint."""
    obj = state["objects"][key]
    w, x, y, z = obj["quat"]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    local = rotation.T @ axis
    hx, hy = (float(v) for v in box["half_extent"][:2])
    centre = np.asarray(obj["pos"][:2], dtype=float) + rotation @ np.asarray(box.get("centre", (0, 0, 0))[:2], float)
    return float(axis @ centre - abs(local[0]) * hx - abs(local[1]) * hy)


def run_attempt(p, reset, camera, meta, config, directory, label, rng, *, record=False, scene_xml=None):
    cfg = config.visual
    # The run's own scene, not the one `locomani.paths` froze when it was imported: an
    # in-process caller attaches to a stack and only then learns which run it is serving
    # (docs/handoff/flat-functional-scripts-20260922.md).
    scene_xml = scene_xml or paths.SCENE_XML
    model = ArmIK(scene_xml)
    whole_body = config.carry.reach_control == "whole_body"
    solver = BimanualIK(
        model,
        waist_pitch_max=config.carry.waist_pitch_max if whole_body else cfg.waist_pitch_max,
        waist_dofs=config.carry.waist_dofs if whole_body else "pitch",
        elbow_min_deg=cfg.elbow_min_deg,
    )
    low = None
    assisted = cfg.pose_source == "assisted"
    key = target_key(meta)
    target_meta = (meta.get("objects") or {}).get(key) or {}
    grasp_points = grasp_points_for(target_meta, cfg.box_grasp_below_rim)
    rim_above_grasp = cfg.box_grasp_below_rim if target_meta.get("box") else GRASP_BELOW_RIM_M
    # The closing gate's across-wall band: the crate bar's 1.8 cm unless the config widens it (visual.capture_across).
    capture_region = RimRegion(across=cfg.capture_across) if cfg.capture_across > 0 else RimRegion()
    geometry = {"grasp_points": grasp_points} if assisted else meta["crate_visual"]
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
    # approach_stop=guard runs the walk-scene sequence (brake_then_reach_20260923): walk with the planner's own
    # arms, guard-brake, ready the arms in place, reach. The line is the nearer of the target surface's edge and
    # the box's front face; distances below are along the walk-in axis from the world origin.
    guard = None
    if cfg.approach_stop == "guard":
        if not (assisted and whole_body):
            raise ValueError("visual.approach_stop=guard needs pose_source assisted and carry.reach_control whole_body")
        edge_xy, walk_axis = near_edge(meta)
        guard = dict(
            watch=FootWatch(scene_xml),
            rule=GuardBrake(extra=cfg.guard_extra, margin=cfg.guard_margin, stride0=cfg.guard_stride0),
            axis=walk_axis,
            line=float(walk_axis @ edge_xy),
            closest=-np.inf,  # frontmost low-body point relative to the line, max over the attempt (+ = crossed)
            # the same for any geom reaching below the box top, over the walk only (the reach crosses it by design)
            walk_closest=-np.inf,
            brake=None,
        )
    retreat_record = None
    arm, placement = (
        reset_hanging(p, reset, meta, cfg, rng)
        if guard is not None
        else reset_ready(p, reset, meta, cfg, rng, solver, **({"whole_body": True} if whole_body else {}))
    )
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
    body_top = np.inf
    if guard is not None and target_meta.get("box"):
        guard["line"] = min(guard["line"], box_front_line(initial, key, target_meta["box"], guard["axis"]))
        box = target_meta["box"]
        body_top = float(initial["objects"][key]["pos"][2]) + float(box["centre"][2]) + float(box["half_extent"][2])
    if whole_body:
        distance = edge_distance(meta, initial["pelvis"]["pos"])
        if cfg.start_pose == "sampled" and not 1.2 <= distance <= 1.5:
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
        AssistedGuidance(
            ArmIK(scene_xml), key=key, box=target_meta.get("box"), rendered=not isinstance(camera, PlaceholderCamera)
        )
        if assisted
        else CratePerception(ArmIK(scene_xml), geometry, max_reprojection_px=cfg.reprojection_px)
    )
    recorder = VisualRecording(directory / label)
    # Ghost-hands rule (2026-09-24): every run records the commanded grip targets next to its state trace so the
    # third-person video can draw them as translucent grippers (never the ego/wrist views).
    ghost = GhostTargetTrace(str(directory.parent / "trace"), solver.offset)
    target_cache = VisibleTarget()
    preparation = ApproachPreparation(solver, cfg, arm.q) if assisted and whole_body and guard is None else None
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
            # Freshness is judged against `now`, so sample it AFTER the read: a frame stamped even 0.1 ms after
            # `now` fails the reach's `0 <= now - timestamp` and that tick is "waiting_for_vision". The live
            # stream stamps in the sim, before this loop; the placeholder stamps at the read, after the `now`
            # above, and at the reach's 66-125 ms ticks every read re-stamped -- the reach clock never advanced
            # in any offline run (2026-09-23/24).
            now = time.time()
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
                brake_now = False
                if guard is not None:
                    landed, front, stride = guard["watch"].update(raw, np.zeros(2), guard["axis"])
                    brake_now, used = guard["rule"].should_brake(
                        landed, front, stride, guard["watch"].last_strides, guard["line"]
                    )
                    if brake_now and guard["brake"] is None:
                        guard["brake"] = dict(front_ankle_to_line=guard["line"] - front, stride=used, t=raw.get("t"))
                nav = approach.step(
                    now,
                    target,
                    raw["pelvis"]["quat"],
                    grasp_pose_available=estimate is not None,
                    **({"base_velocity": raw["pelvis"]["vel"]} if assisted else {}),
                    **motion_clock,
                    guard_brake=brake_now,
                )
                admission = None
                if nav.ready and assisted and guard is None:
                    probe = CoordinatedReach(
                        solver,
                        config,
                        raw,
                        reference_joints=joints,
                        reference_quaternion=raw["pelvis"]["quat"],
                        target=CoordinatedTarget(
                            object_key=key,
                            grasp_points_local=grasp_points,
                            rim_above_grasp=rim_above_grasp,
                            region=capture_region,
                        ),
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
                if nav.facing_offset:
                    # Turn in place: re-aim the held planner heading at the target
                    # from the CURRENT yaw. The new heading stays held afterwards.
                    face = yaw + nav.facing_offset
                    arm._face = (float(np.cos(face)), float(np.sin(face)), 0.0)
                direction = nav.direction.copy()
                direction[:2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]] @ direction[:2]
                if guard is not None and nav.speed > 0:
                    direction = np.array([*guard["axis"], 0.0])  # world walk-in axis, as diag_brake walks
                q, velocity = (
                    (joints.copy() if guard is not None else arm.q.copy(), np.zeros(29))
                    if preparation is None
                    else preparation.step(float(raw["t"]), estimate)
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
                    "facing_offset": nav.facing_offset,
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
                    if isinstance(arm, HangingWalk):
                        # Guard stance is quiet: ready the arms in place, as the walk scene did after its brake
                        # (a no-op placement keeps the robot where the brake left it). Blocks for ~10 s.
                        print(f"[{label}] arm preparation at {now - started:.2f}s", flush=True)
                        arm, prepared = prepare_stationary_ready(
                            p,
                            reset,
                            cfg,
                            solver,
                            placement=lambda _reset_id: None,
                            whole_body=True,
                            heading=approach_heading(meta),
                        )
                        placement["arm_preparation"] = dict(
                            initial_stance_ready=prepared["initial_stance_ready"],
                            pelvis_xy=[float(v) for v in p.obs()["pelvis"]["pos"][:2]],
                        )
                        if not prepared["initial_stance_ready"]:
                            reason = "arm_preparation_unstable"
                            break
                        last_camera_received = last_progress_wall = time.time()
                        continue
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
                            target=CoordinatedTarget(
                                object_key=key,
                                grasp_points_local=grasp_points,
                                rim_above_grasp=rim_above_grasp,
                                region=capture_region,
                            ),
                        )
                        ghost.start(label)
                        low.ghost = ghost
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
                closed[:2] = cfg.close_target
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
                closed[:2] = cfg.close_target
                arm.hands = {side: HAND_OPEN + command.closure * (closed - HAND_OPEN) for side in ("left", "right")}
                arm.q = q
                arm._push(q)
            last_command_time = now
            if guard is not None:
                past = guard["watch"].frontmost_low(raw, np.zeros(2), guard["axis"]) - guard["line"]
                guard["closest"] = max(guard["closest"], past)
                if isinstance(arm, HangingWalk):
                    past = guard["watch"].frontmost_low(raw, np.zeros(2), guard["axis"], top=body_top) - guard["line"]
                    guard["walk_closest"] = max(guard["walk_closest"], past)
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
        if reason == "completed" and cfg.retreat_distance > 0 and isinstance(arm, StreamedArm):
            print(f"[{label}] retreat at {time.time() - started:.2f}s", flush=True)
            reason, retreat_record, final = retreat(
                p, arm, cfg, key, approach_heading(meta), float(initial["objects"][key]["pos"][2])
            )
    except Exception:
        reason, error = "runtime_error", traceback.format_exc()
    finally:
        control_reason = reason
        if hasattr(p, "cmd"):
            p.cmd.reference_observer = None
        for cleanup in (recorder.close, reset.end_video, arm.stop, ghost.stop):
            try:
                cleanup()
            except Exception:
                reason, error = "runtime_error", (error or "") + traceback.format_exc()
    try:
        result = evaluate(initial, final, reason, lift_height=cfg.lift_height, max_tilt_deg=cfg.max_tilt_deg, key=key)
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
        target=key,
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
    if guard is not None:
        result["guard"] = dict(
            line=guard["line"],
            brake=guard["brake"],
            frontmost_past_line_max=guard["closest"],
            walk_body_past_line_max=guard["walk_closest"],
        )
    if retreat_record is not None:
        result["retreat"] = retreat_record
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
        heading = approach_heading(meta)
        p.cmd.hold(2.0, mode=LocomotionMode.IDLE, facing=(float(np.cos(heading)), float(np.sin(heading)), 0.0))
    if "pelvis" in initial:
        result["initial_edge_distance"] = edge_distance(meta, initial["pelvis"]["pos"])
    if error is not None:
        result["error"] = error
    print(f"[{label}] {result}", flush=True)
    return RoundResult(result["ok"], result["why"], result["produced"], {"metrics": result})


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--label", default="visual_crate")
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--scene-dir",
        type=Path,
        help="the run's own scene copy; defaults to whatever LOCOMANI_SIM_DIR named at import",
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument(
        "--record",
        action="store_true",
        help="drive the data exporter: 'c' at task start, 'c'/'x' after evaluation; refused without an exporter",
    )
    args = parser.parse_args(argv)
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    config = CrateTaskConfig.load(args.config) if args.config else CrateTaskConfig()
    run(
        config.with_overrides(args.overrides),
        rounds=args.rounds,
        seed=args.seed,
        manifest=args.manifest,
        label=args.label,
        record=args.record,
        scene_dir=args.scene_dir,
    )


def run(config, *, rounds, seed, manifest, label="visual_crate", record=False, scene_dir=None, render=None):
    """Every attempt this configuration asks for, against the scene it names.

    The entry for a caller that already HAS a configuration: the collection entry point builds one
    and calls this, so neither side assembles a command line for a process that is never started
    (docs/handoff/flat-functional-scripts-20260922.md, decisions 2 and 11). `main` is the CLI edge
    over it and nothing else.

    `scene_dir` is the run's own scene copy. Defaulting to `locomani.paths` keeps every existing
    CLI and diagnostic caller working, but an in-process caller must pass it: those module
    attributes were resolved when `locomani.paths` was imported, which is before a process that
    attaches to a stack knows which run it is serving.
    """
    if record:
        ports.require_exporter("--record")
    if config.carry.reach_control == "whole_body":
        require_pose_ownership()
    scene_meta = Path(scene_dir) / "scene_meta.json" if scene_dir else Path(paths.SCENE_META)
    scene_xml = str(Path(scene_dir) / "scene_locomani.xml") if scene_dir else paths.SCENE_XML
    meta = json.loads(scene_meta.read_text())
    if config.visual.pose_source == "markers" and "crate_visual" not in meta:
        raise SystemExit("build a distinct painted scene with python -m locomani.tasks.crate.visual_scene")
    manifest = Path(manifest)
    # `render` like `scene_dir`: an in-process caller hands over the run's backend.render; the import-time
    # DEFAULT_BACKEND only knows it when this process was started with the stack's environment
    if (render or DEFAULT_BACKEND.render) == "offline":
        if config.visual.pose_source != "assisted":
            raise SystemExit(
                "backend.render offline needs visual.pose_source assisted: the markers expert reads pixels"
            )
        print("[expert] backend.render offline: placeholder frames, no camera stream", flush=True)
        camera = PlaceholderCamera()
    else:
        from gear_sonic.camera.composed_camera import ComposedCameraClientSensor

        camera = ComposedCameraClientSensor(server_ip="localhost", port=ports.port("TF_CAMERA_PORT"))
    p, reset = Primitives(), EpisodeControlClient()
    directory = manifest.parent / "observations"
    directory.mkdir(parents=True, exist_ok=True)
    try:
        prime(p, reset)
        for index in range(rounds):
            round_label, round_seed = f"{label}_{index:02d}", seed + index
            result = record_attempt(
                manifest,
                round_label,
                round_seed,
                lambda: run_attempt(
                    p,
                    reset,
                    camera,
                    meta,
                    config,
                    directory,
                    round_label,
                    np.random.default_rng(round_seed),
                    record=record,
                    scene_xml=scene_xml,
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
