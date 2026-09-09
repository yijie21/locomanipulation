"""Measured Dex1 capture geometry and bounded, crouched local alignment.

Only the opt-in recovery policy uses this path. A pregrasp waypoint is not
an execution prerequisite. Collision screening remains a sampled kinematic
check, not a guarantee of whole-body dynamics.
"""

import time
from copy import deepcopy

import adaptive_grasp
import numpy as np
from grasp_capture import IK_POSITION_TOL, IK_ROTATION_TOL, closure_geometry, height_correction
from primitives import HAND_CLOSED_RIGHT, Primitives
from ready_grasp_policy import OFFSET, templates, terminal_rotation


def bounded_rotation(measured, desired, max_lead=0.04):
    """Bound the attitude request relative to the delivered wrist attitude."""
    from scipy.spatial.transform import Rotation

    delta = Rotation.from_matrix(measured.T @ desired).as_rotvec()
    delta *= min(1.0, max_lead / max(float(np.linalg.norm(delta)), 1e-9))
    return measured @ Rotation.from_rotvec(delta).as_matrix()


class CaptureStability:
    """Speed hysteresis never relaxes geometry, clearance, or closing speed."""

    def __init__(self, hysteresis=True):
        self.since = None
        self.last = None
        self.hysteresis = hysteresis

    def update(self, now, safe, speed):
        exit_speed = 0.045 if self.hysteresis else 0.035
        if (
            not safe
            or not np.isfinite(speed)
            or speed >= exit_speed
            or (self.hysteresis and self.last is not None and now - self.last > 0.1)
        ):
            self.since = None
        self.last = now
        if not safe or not np.isfinite(speed) or speed >= exit_speed:
            return False
        if self.since is None and speed < 0.035:
            self.since = now
        return self.since is not None and now - self.since >= 0.25 and speed < 0.035


def closure_clear(policy, state):
    """Screen the current arms and sampled finger closure against the table."""
    closed = HAND_CLOSED_RIGHT[:2]
    for fraction in np.linspace(0.0, 1.0, 6):
        trial = deepcopy(state)
        fingers = np.asarray(state["palms"][policy.side]["hand_q"])
        trial["palms"][policy.side]["hand_q"] = (fingers + fraction * (closed - fingers)).tolist()
        error = policy.reference_check(trial, np.asarray(state["joints"]), "close")
        if error:
            return error
    return None


def run(policy, ghost):
    from stream_reach import SQUAT_HEIGHT_BIAS, _servo_continuous, toppled

    p, arm, ex = policy.p, policy.arm, policy.execution
    side, key = policy.side, policy.task.target.key
    state = p.obs()
    target = policy.task.grip(state)
    rel = p.world_to_root_full(target, state)
    _, _, local_rot = min(templates(), key=lambda row: np.linalg.norm(row[1] - rel))
    rot = terminal_rotation(Primitives._rot(state["pelvis"]["quat"]), local_rot, getattr(policy, "config", None))
    start = time.monotonic()
    height0 = arm.height
    ref = np.asarray(state["palms"][side]["pos"]).copy()
    trim = np.zeros(3)
    timely = getattr(getattr(policy, "config", None), "timely_capture", False)
    continuous = getattr(getattr(policy, "config", None), "continuous_capture", False)
    point_control = getattr(getattr(policy, "config", None), "tool_point_grasp", False)
    if point_control:
        ref += Primitives._rot(state["palms"][side]["quat"]) @ OFFSET
    capture_anchor = None
    capture_rotation = None
    feedback = None
    ready_ticks = 0
    stable = CaptureStability(hysteresis=timely)
    previous = None
    touched = False
    last_log = 0.0
    stats = dict(
        samples=[],
        height_start=height0,
        can_reposition=False,
        alignment_command_ticks=0,
        height_adjustment_ticks=0,
        seated_hold_ticks=0,
        first_safe_elapsed_s=None,
    )
    stats["active_stabilization_ticks"] = 0
    stats["active_reacquire_count"] = 0
    stats["controlled_point"] = "grasp_center" if point_control else "wrist"
    ex.metrics.setdefault("measured_grasp", []).append(stats)
    ex.enter("grasp.local_align")
    policy.ramp.settle()
    arm._q_prev = arm.q.copy()
    arm.next_t = time.time()
    while time.monotonic() - start < 8.0 and not policy.expired():
        state = p.obs()
        now = time.monotonic()
        if state["fallen"] or toppled(state, key):
            return False, "local alignment disturbed target or base"
        target = policy.task.grip(state)
        palm = np.asarray(state["palms"][side]["pos"])
        measured_rot = Primitives._rot(state["palms"][side]["quat"])
        controlled = palm + measured_rot @ OFFSET if point_control else palm
        relative = measured_rot.T @ (target - palm)
        touching_now = side in state["objects"][key].get("touching", [])
        touched |= touching_now
        geometry = closure_geometry(state, side, target)
        precision_ready = (
            geometry["ready"]
            and abs(geometry["depth_m"]) < 0.015
            and abs(geometry["lateral_signed_m"]) < 0.015
            and abs(geometry["world_height_error_m"]) < 0.015
        )
        # Precision is a planning preference, never a hidden five-second
        # timer before accepting an already safe, measured grasp.
        capture_candidate = (
            geometry["ready"] if timely else precision_ready or (geometry["ready"] and now - start >= 5.0)
        )
        obs_dt = 0.02 if previous is None else max(now - previous[0], 0.001)
        relative_speed = (
            float("inf")
            if previous is None
            else float(np.linalg.norm(relative - previous[1]) / max(now - previous[0], 0.001))
        )
        previous = (now, relative.copy())
        clear = closure_clear(policy, state) if capture_candidate else None
        safe = capture_candidate and clear is None
        if point_control:
            if safe and capture_anchor is None:
                capture_anchor = controlled.copy()
                capture_rotation = measured_rot.copy()
                ref = controlled.copy()
            elif not safe:
                capture_anchor = capture_rotation = None
        close_now = stable.update(now, safe, relative_speed)
        ready_ticks = ready_ticks + 1 if safe and relative_speed < 0.035 else 0
        if continuous:
            close_now = ready_ticks >= 3
            if safe and feedback is None:
                from capture_feedback import CaptureFeedback

                feedback = CaptureFeedback(state, side, policy.task.grip)
        if safe and stats["first_safe_elapsed_s"] is None:
            stats["first_safe_elapsed_s"] = now - start
        if now - last_log >= 0.2:
            stats["samples"].append(
                dict(
                    elapsed_s=now - start,
                    **geometry,
                    precision_ready=precision_ready,
                    relative_speed_m_s=relative_speed if np.isfinite(relative_speed) else None,
                    clearance=clear,
                    height=arm.height,
                    palm_world=palm.tolist(),
                    object_world=p.object_pos(key, state).tolist(),
                    grip_world=target.tolist(),
                    pelvis_world=list(map(float, state["pelvis"]["pos"])),
                    orientation_error_rad=float(
                        np.arccos(np.clip((np.trace(measured_rot.T @ rot) - 1) / 2, -1.0, 1.0))
                    ),
                )
            )
            last_log = now
        if close_now:
            # Preserve the delivered grasp pose through closure. The old
            # terminal template must not pull an already seated jaw away.
            plan = adaptive_grasp.Grasp(
                ok=True,
                rot=measured_rot,
                wrist=palm.copy(),
                grip=target.copy(),
                approach=measured_rot[:, 0],
                travel=0.0,
            )
            stats.update(close_geometry=geometry, alignment_s=now - start, height_end=arm.height, touched=touched)
            ex.enter("grasp.capture_ready", **geometry)
            return _servo_continuous(
                p,
                arm,
                key,
                plan,
                side,
                ghost=ghost,
                surface_top=policy.task.target.surface_top,
                execution=ex,
                reference_guard=policy.reference_check,
                transport_guard=policy.guard,
                close_from_current=True,
                grip_target=policy.task.grip,
                require_bilateral=timely,
                capture_feedback=feedback,
                grasp_point_control=point_control,
                preserve_capture_reference=getattr(policy.config, "capture_reference_handover", False),
            )
        if (safe or (capture_candidate and not timely)) and not continuous and not point_control:
            # Already seated: stabilize without chasing a nominal waypoint.
            stats["seated_hold_ticks"] += 1
            arm._push(arm.q)
            continue
        if continuous and feedback is not None:
            desired, _, lost = feedback.update(state, obs_dt)
            if lost:
                # Before any closure, a drifting hand may resume the same
                # short alignment. A displaced object must never be chased.
                if (
                    np.linalg.norm(policy.task.grip(state) - feedback.origin) >= 0.02
                    or stats["active_reacquire_count"] >= 2
                ):
                    return False, "active capture could not stabilize without chasing the object"
                feedback = None
                ready_ticks = 0
                stats["active_reacquire_count"] += 1
                continue
            ref += (desired - ref) * min(1.0, 0.0008 / max(np.linalg.norm(desired - ref), 1e-9))
            ref = palm + (ref - palm) * min(1.0, 0.015 / max(np.linalg.norm(ref - palm), 1e-9))
            command = ref + np.clip(0.25 * (ref - palm), -0.004, 0.004)
            seed = arm.q.copy()
            seed[:12] = state["joints"][:12]
            q, pe, re = arm.ik.solve(
                side,
                p.world_to_root_full(command, state),
                Primitives._rot(state["pelvis"]["quat"]).T @ feedback.rot,
                q_mj29=seed,
                use_waist=False,
                iters=30,
            )
            if pe > 0.02 or re > 0.15:
                return False, "active capture outside comfortable IK"
            q = seed + np.clip(q - seed, -0.02, 0.02)
            rejection = policy.reference_check(state, q, "in")
            if rejection:
                return False, rejection
            arm.q = q
            arm._push(q)
            stats["active_stabilization_ticks"] += 1
            continue
        preferred = target - getattr(getattr(policy, "config", None), "grasp_depth_preference", 0.0) * rot[:, 0]
        desired = preferred if point_control else preferred - rot @ OFFSET
        rotation_goal = bounded_rotation(measured_rot, rot) if point_control else rot
        if point_control and capture_anchor is not None:
            # Hold the measured grasp point while the body settles, without
            # freezing joints or chasing a more precise nominal waypoint.
            desired, rotation_goal = capture_anchor, capture_rotation
            stats["active_stabilization_ticks"] += 1
        error = desired - controlled
        entry = rot[:, 0]
        depth = float(error @ entry)
        lateral_error = error - depth * entry
        if np.linalg.norm(lateral_error) > 0.018:
            desired -= max(depth, 0.0) * entry  # align before advancing
        if np.linalg.norm(error) > getattr(getattr(policy, "config", None), "local_reach_limit", 0.20):
            stats["can_reposition"] = not touched
            return False, "local target outside short reach"
        # Measured tracking trim; do not integrate against a moving object.
        if not touching_now and capture_anchor is None:
            trim += 0.01 * lateral_error
            trim = np.clip(trim, [-0.04, -0.04, -0.04], [0.04, 0.04, 0.04])
        aim = desired if capture_anchor is not None else desired + trim
        delta = aim - ref
        step = getattr(policy.config, "local_point_speed", 0.125) * min(obs_dt, 0.04) if point_control else 0.0025
        lead_limit = getattr(policy.config, "local_point_lead", 0.035) if point_control else 0.035
        ref += delta * min(1.0, step / max(np.linalg.norm(delta), 1e-6))
        ref = controlled + (ref - controlled) * min(1.0, lead_limit / max(np.linalg.norm(ref - controlled), 1e-6))
        q, pe, re = arm.ik.solve(
            side,
            p.world_to_root_full(ref, state),
            Primitives._rot(state["pelvis"]["quat"]).T @ rotation_goal,
            q_mj29=arm.q,
            use_waist=False,
            iters=30,
            **(dict(point_offset=OFFSET, rot_weight=0.1) if point_control else {}),
        )
        rotation_tolerance = (
            IK_ROTATION_TOL if getattr(getattr(policy, "config", None), "tool_point_grasp", False) else 0.15
        )
        if pe > IK_POSITION_TOL or re > rotation_tolerance:
            stats["ik_rejection"] = dict(
                position_error_m=float(pe), rotation_error_rad=float(re), rotation_tolerance_rad=rotation_tolerance
            )
            stats["can_reposition"] = not touched
            return False, "local alignment outside comfortable IK"
        q = arm.q + np.clip(q - arm.q, -0.025, 0.025)
        rejection = policy.reference_check(state, q, "in")
        if rejection:
            stats["can_reposition"] = not touched
            return False, rejection
        # Height is a secondary actuator, after the arm has had time to
        # converge. Stay within 2.5 cm of this squat and preserve arm clearance.
        height_error = height_correction(geometry)
        if now - start > 2.0 and height_error and not touching_now:
            lo = max(height0 - 0.025, policy.standing_z - policy.config.max_drop - SQUAT_HEIGHT_BIAS)
            hi = min(height0 + 0.025, policy.standing_z - 0.03 - SQUAT_HEIGHT_BIAS)
            height = float(np.clip(arm.height + np.clip(height_error * 0.03, -0.0004, 0.0004), lo, hi))
            trial = deepcopy(state)
            trial["pelvis"]["pos"][2] += height - arm.height
            if policy.reference_check(trial, q, "in") is None:
                stats["height_adjustment_ticks"] += int(abs(height - arm.height) > 1e-6)
                arm.height = height
        stats["alignment_command_ticks"] += 1
        arm.q = q
        arm._push(q)
    stats.update(height_end=arm.height, touched=touched)
    # A tracking timeout alone is not evidence that standing/repeating helps.
    return False, "local alignment timed out without a justified stand correction"
