"""The visual expert's per-tick machinery, extracted from `expert.run_attempt` (pure extraction, 2026-09-28).

`TickSource.next()` is the top of the old loop: one privileged sample, the camera read, the guidance
estimate, the recorder context. `walk_step` is the `approach is not None` branch, `reach_step` the
`low.step` branch, `markers_step` the RGB-markers branch, `record_tick` the recorder row and
`guard_track` the brake-line bookkeeping. `run_attempt` still owns object construction (so the offline
tests' monkeypatches on `expert.*` keep working) and the phase order; the skills catalogue calls the same
pieces to run the walk-in and the reach as two separate primitives.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np

from locomani.backends.sonic.agent_server import HAND_OPEN, LocomotionMode
from locomani.tasks.crate.perception import ApproachTarget
from locomani.tasks.crate.visual_policy import ExpertCommand


@dataclass
class Tick:
    raw: dict
    now: float
    motion_clock: dict
    fresh_image: bool
    images: dict | None
    message: dict | None
    estimate: object
    hold_only: bool  # no usable image this tick: push the held reference and take the next sample
    joints: np.ndarray | None = None
    measured: object = None
    rotations: object = None
    openings: np.ndarray | None = None


class TickSource:
    """One physical sample + camera frame + guidance estimate per call; abort reasons as strings."""

    def __init__(self, p, camera, perception, target_cache, recorder, cfg, *, assisted, solver, initial, started):
        self.p, self.camera, self.perception, self.target_cache = p, camera, perception, target_cache
        self.recorder, self.cfg, self.assisted, self.solver = recorder, cfg, assisted, solver
        self.initial, self.started = initial, started
        self.final, self.final_wall = initial, started
        self.estimate, self.images, self.message, self.stamp = None, None, None, None
        self.last_image_time, self.last_control_time, self.last_phase = 0.0, None, None
        self.last_camera_received = self.last_progress_wall = started
        self.last_simulation_time = float(initial["t"]) if assisted else 0.0

    def touch(self) -> None:
        """After a blocking sub-phase (turn, arm preparation): the camera and the clock were not being read."""
        self.last_camera_received = self.last_progress_wall = time.time()

    def elapsed(self) -> float:
        if self.assisted:
            return float(self.final["t"]) - float(self.initial["t"])
        return time.time() - self.started

    def next(self) -> Tick | str | None:
        """None: no new physical sample yet. str: abort reason. Tick: a sample to act on."""
        cfg, assisted = self.cfg, self.assisted
        now = time.time()
        raw = self.p.obs()
        self.final, self.final_wall = raw, time.time()
        motion_clock = {"motion_time": float(raw["t"])} if assisted else {}
        if assisted:
            if raw["t"] < self.last_simulation_time:
                return "simulation_clock_reversed"
            if raw["t"] > self.last_simulation_time:
                self.last_simulation_time, self.last_progress_wall = float(raw["t"]), now
            elif now - self.last_progress_wall > cfg.vision_timeout:
                return "simulation_stalled"
        if raw["fallen"]:
            return "external_fall_abort"
        if assisted and raw["t"] == self.last_control_time:
            time.sleep(0.002)
            return None
        self.last_control_time = raw.get("t")
        incoming = self.camera.read()
        # Freshness is judged against `now`, so sample it AFTER the read (see the 2026-09-23/24 note in git history:
        # a frame stamped after `now` never advanced the reach clock offline).
        now = time.time()
        fresh_image = False
        if incoming is not None and {"ego_view", "left_wrist", "right_wrist"} <= set(incoming["images"]):
            self.last_camera_received = now
            self.stamp = max(float(v) for v in incoming["timestamps"].values())
            if now - min(float(v) for v in incoming["timestamps"].values()) > cfg.vision_timeout:
                return "stale_camera_abort"
            fresh_image = self.stamp != self.last_image_time
            if fresh_image:
                self.message, self.images = incoming, incoming["images"]
                self.last_image_time = self.stamp
                self.estimate = self.perception.update(
                    self.images, raw if assisted else np.asarray(raw["joints"]), self.stamp
                )
                if assisted:
                    self.target_cache.update(self.estimate, raw)
        if now - self.last_camera_received > cfg.vision_timeout:
            return "camera_unavailable"
        if assisted:
            self.estimate = self.target_cache.at(raw)
        joints = np.asarray(raw["joints"], dtype=float)
        measured, rotations = self.solver.tool_poses(joints)
        self.recorder.context = dict(
            state_seq=raw.get("seq"),
            simulation_time=raw.get("t"),
            state_wall_time=raw.get("wall"),
            image_frame=(self.recorder.frames if fresh_image else self.recorder.frames - 1)
            if self.images is not None
            else None,
            image_timestamp=self.last_image_time or None,
            phase=self.last_phase or "walk",
        )
        tick = Tick(raw, now, motion_clock, fresh_image, self.images, self.message, self.estimate, hold_only=False)
        if (not assisted and not fresh_image) or self.images is None:
            return replace(tick, hold_only=True)
        tick.joints, tick.measured, tick.rotations = joints, measured, rotations
        tick.openings = np.array(
            [2 * (np.mean(raw["palms"][side]["hand_q"][:2]) + 0.023) for side in ("left", "right")]
        )
        return tick


@dataclass
class WalkState:
    """What the walk-in branch reads and updates between ticks."""

    p: object
    approach: object  # VisualApproach
    arm: object
    guard: dict | None
    preparation: object  # ApproachPreparation | None
    solver: object
    cfg: object
    assisted: bool
    perception: object
    stand_attempts: list
    probe_factory: object  # (raw, joints) -> CoordinatedReach, for the depth-mode reach admission
    ticks: TickSource
    last_motion: dict


def walk_step(tick: Tick, w: WalkState):
    """One walk-in tick: brake decision, approach command, arm reference, planner frame. -> (command, navigation, nav)."""
    raw, now, joints, estimate = tick.raw, tick.now, tick.joints, tick.estimate
    if estimate is not None:
        target = ApproachTarget(
            estimate.position, estimate.timestamp, "visible_simulation_pose" if w.assisted else "full_rgb_pose"
        )
    elif w.assisted:
        target = None
    else:
        target = w.perception.approach_target(tick.images, joints, w.ticks.stamp)
    brake_now = False
    guard = w.guard
    if guard is not None:
        landed, front, stride = guard["watch"].update(raw, np.zeros(2), guard["axis"])
        if getattr(w.cfg, "guard_brake", "landing") == "phase":
            brake_now, used = phase_brake(guard, raw, landed, w.cfg)
        else:
            brake_now, used = guard["rule"].should_brake(
                landed, front, stride, guard["watch"].last_strides, guard["line"]
            )
        if brake_now and getattr(w.approach, "phase", "walk") == "walk":  # not while a guard re-walk backs off
            brake = dict(front_ankle_to_line=guard["line"] - front, stride=used, t=raw.get("t"))
            guard["brake"] = guard["brake"] or brake  # the first walk-in's, as before the guard re-walk
            guard.setdefault("brakes", []).append(brake)  # every walk-in's (TODO 2.4.10.23.2)
    nav = w.approach.step(
        now,
        target,
        raw["pelvis"]["quat"],
        grasp_pose_available=estimate is not None,
        **({"base_velocity": raw["pelvis"]["vel"]} if w.assisted else {}),
        **tick.motion_clock,
        guard_brake=brake_now,
    )
    admission = None
    if nav.ready and w.assisted and guard is None:
        probe = w.probe_factory(raw, joints)
        check = probe.step(now, estimate, raw, tick.openings, **tick.motion_clock)
        admission = probe.diagnostics
        if check.reason == "unsupported_assisted_reach":
            if w.approach.retry_closer():
                nav = w.approach.step(
                    now, target, raw["pelvis"]["quat"], grasp_pose_available=True, **tick.motion_clock
                )
            else:
                nav = replace(nav, ready=False, reason="approach_unsupported_reach")
    arm = w.arm
    arm.mode = LocomotionMode.SLOW_WALK if nav.speed > 0 else LocomotionMode.IDLE
    yaw = w.p.pelvis_yaw(raw)
    if nav.facing_offset:
        face = yaw + nav.facing_offset  # re-aim the held heading from the CURRENT yaw; held afterwards
        arm._face = (float(np.cos(face)), float(np.sin(face)), 0.0)
    direction = nav.direction.copy()
    direction[:2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]] @ direction[:2]
    if guard is not None and nav.speed > 0:
        # world walk-in axis, as diag_brake walks; backwards for a guard re-walk's backing off
        direction = np.array([*guard["axis"], 0.0]) * (-1.0 if nav.phase == "back" else 1.0)
    q, velocity = (
        (joints.copy() if guard is not None else arm.q.copy(), np.zeros(29))
        if w.preparation is None
        else w.preparation.step(float(raw["t"]), estimate)
    )
    arm.q = q
    w.ticks.recorder.context["phase"] = nav.phase
    w.last_motion = {"movement": direction, "speed": nav.speed if nav.speed > 0 else -1.0}
    arm._push(q, reference_velocity=velocity, **w.last_motion)
    positions, reference_rotations = w.solver.tool_poses(q)
    command = ExpertCommand(positions, reference_rotations, 0.0, nav.phase, nav.reason, bool(nav.reason))
    navigation = {
        "target": None if target is None else target.position,
        "source": None if target is None else target.source,
        "movement": direction,
        "speed": nav.speed,
        "facing_offset": nav.facing_offset,
        "relative_speed": w.approach.relative_speed,
        "angular_speed": w.approach.angular_speed,
        "ready": nav.ready,
        "reach_admission": admission,
        "stand_attempts": list(w.stand_attempts),
    }
    return command, navigation, nav


def phase_brake(guard: dict, raw: dict, landed: bool, cfg) -> tuple[bool, float]:
    """visual.guard_brake=phase (TODO 3.14.1): brake on any tick once the pelvis plus the phase-dependent band reaches
    the line minus the margin. The band is where the frontmost low-body point settles ahead of the pelvis after an
    IDLE at that gait phase (time since the last foot landing / the last step period). -> (brake now, band used)."""
    times = guard.setdefault("landing_t", [])
    if landed:
        times.append(float(raw["t"]))
    c0, a1, b1 = cfg.guard_phase_band
    band = c0
    if len(times) >= 2:
        phase = ((float(raw["t"]) - times[-1]) / max(times[-1] - times[-2], 1e-3)) % 1.0
        band = c0 + a1 * np.cos(2 * np.pi * phase) + b1 * np.sin(2 * np.pi * phase)
    pelvis = float(np.asarray(raw["pelvis"]["pos"][:2]) @ guard["axis"])
    return bool(pelvis + band >= guard["line"] - cfg.guard_margin), float(band)


def reach_step(tick: Tick, *, arm, low, solver, cfg, assisted: bool, recorder):
    """One coordinated-reach tick on the POSE stream (or the markers low reach). -> ExpertCommand."""
    raw = tick.raw
    body_command = (
        low.step(tick.now, tick.estimate, raw, tick.openings, **tick.motion_clock)
        if assisted
        else low.step(tick.now, tick.estimate, tick.joints, raw["pelvis"]["quat"], tick.openings)
    )
    q = body_command.joints
    closed = np.zeros_like(HAND_OPEN)
    closed[:2] = cfg.close_target
    arm.hands = {side: HAND_OPEN + body_command.closure * (closed - HAND_OPEN) for side in ("left", "right")}
    arm.q, arm.root_quat = q, body_command.root_quaternion
    recorder.context["phase"] = body_command.phase
    if not body_command.finished or body_command.reason == "completed":
        arm._push(q, reference_velocity=body_command.velocity, **({"reference_time": raw["t"]} if assisted else {}))
    points, attitudes = solver.tool_poses(q)
    return ExpertCommand(
        points, attitudes, body_command.closure, body_command.phase, body_command.reason, body_command.finished
    )


def markers_step(tick: Tick, *, arm, policy, reference, solver, cfg, last_command_time):
    """The historical RGB-markers walk expert's tick (no whole-body reach)."""
    command = policy.step(tick.now, tick.estimate, tick.measured, tick.rotations, tick.openings)
    correction = np.clip(cfg.tracking_gain * (command.positions - tick.measured), -cfg.tracking_cap, cfg.tracking_cap)
    solution = solver.solve(
        tick.joints, command.positions + correction, command.rotations, iterations=18, posture=arm.q
    )
    q = reference.step(solution.joints, tick.now - last_command_time)
    closed = np.zeros_like(HAND_OPEN)
    closed[:2] = cfg.close_target
    arm.hands = {side: HAND_OPEN + command.closure * (closed - HAND_OPEN) for side in ("left", "right")}
    arm.q = q
    arm._push(q)
    return command


def guard_track(guard: dict | None, raw: dict, *, hanging: bool, body_top: float) -> None:
    """Keep the frontmost-past-line maxima the result reports (whole body, and the low body while walking)."""
    if guard is None:
        return
    past = guard["watch"].frontmost_low(raw, np.zeros(2), guard["axis"]) - guard["line"]
    guard["closest"] = max(guard["closest"], past)
    if hanging:
        past = guard["watch"].frontmost_low(raw, np.zeros(2), guard["axis"], top=body_top) - guard["line"]
        guard["walk_closest"] = max(guard["walk_closest"], past)


def record_tick(recorder, tick: Tick, command, navigation, *, low, arm, perception, assisted: bool) -> None:
    raw, estimate = tick.raw, tick.estimate
    recorder.append(
        tick.images,
        {
            "wall_time": tick.now,
            "image_timestamps": tick.message["timestamps"],
            "state_seq": raw.get("seq"),
            "state_wall_time": raw.get("wall"),
            "simulation_time": raw.get("t"),
            "motion_clock": "simulation" if assisted else "wall",
            "joints": tick.joints,
            "base_quaternion": raw["pelvis"]["quat"],
            "base_position": raw["pelvis"]["pos"],
            "base_velocity": raw["pelvis"].get("vel"),
            "guidance": perception.diagnostics if assisted else {"pose_source": "rgb_markers"},
            "navigation": navigation,
            "hand_openings": tick.openings,
            "reference_joints": arm.q,
            "command_id": recorder.command_count - 1,
            "root_reference": None if low is None else dict(quaternion=arm.root_quat, velocity=low.velocity),
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
