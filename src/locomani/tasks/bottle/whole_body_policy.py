"""Camera-admitted bottle reach through one supported descent/closure/lift reference."""

import json
import time
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from locomani.backends.sonic.agent_server import StreamMode
from locomani.backends.sonic.pose_preview import PosePreview
from locomani.backends.sonic.primitives import (
    HAND_CLOSED_RIGHT,
    HAND_OPEN,
    LocomotionMode,
    Primitives,
)
from locomani.backends.sonic.streamed_arm import StreamedArm
from locomani.pipeline.provenance import require_pose_ownership
from locomani.primitives.arm_ik import ArmIK
from locomani.primitives.contacts import bilateral_dex1_contact
from locomani.primitives.supported_motion import SupportedMotion, bounded_correction_step
from locomani.primitives.whole_body_ik import WholeBodyIK
from locomani.tasks.bottle.assisted_target import BottleTargetObserver
from locomani.tasks.bottle.grasp_capture import (
    DEX1_PAD_X_MAX,
    HEIGHT_TOL,
    LATERAL_TOL,
    capture_axial_clearance,
    capture_tool_goal,
    closure_geometry,
)
from locomani.tasks.bottle.loaded_handback import (
    AsyncLoadedEndpointPlanner,
    LoadedHandbackStabilityGate,
    commanded_palm_rotation,
    loaded_lift_height,
    loaded_liftoff_event,
    loaded_orientation_metrics,
    loaded_orientation_supervision,
    loaded_progress_governor,
    loaded_world_path,
)
from locomani.tasks.bottle.ready_grasp_policy import OFFSET
from locomani.tasks.bottle.stance_reach import (
    AsyncStanceReachPlanner,
    StanceReachEvaluator,
    cap_clearing_entry_step,
    cap_clearing_target_contact_allowed,
)
from locomani.tasks.bottle.walking_reach import WalkingReach

# Arm meshes that legitimately touch the bottle in the middle-pad grasp: the Dex1
# housing sits 0-4 mm from the held bottle (housing_clearance_review.json).
GRASP_ENVELOPE_MESHES = ("Dex1_base_link",)


def smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def bounded_takeover_brake_step(q, root_quat, velocity, angular, dt, acceleration, limits):
    """Advance a nonzero POSE takeover reference to rest without a velocity reset."""
    q = np.asarray(q, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    angular = np.asarray(angular, dtype=float)
    speed = max(float(np.max(np.abs(velocity))), float(np.linalg.norm(angular)))
    scale = max(0.0, 1 - float(acceleration) * float(dt) / speed) if speed else 0.0
    next_velocity = velocity * scale
    next_angular = angular * scale
    following = np.clip(
        q + 0.5 * (velocity + next_velocity) * dt,
        np.asarray(limits[0], dtype=float),
        np.asarray(limits[1], dtype=float),
    )
    root = Rotation.from_quat(np.asarray(root_quat, dtype=float)[[1, 2, 3, 0]])
    root = Rotation.from_rotvec(0.5 * (angular + next_angular) * dt) * root
    return (
        following,
        root.as_quat()[[3, 0, 1, 2]],
        next_velocity,
        next_angular,
        bool(speed == 0 or scale == 0),
    )


def guarded_capture_depth(
    lateral_error_m,
    height_error_m,
    capture_depth_m,
    retreat_depth_m,
    lateral_tolerance_m,
    height_tolerance_m,
):
    """Continuously retreat axial entry until lateral and height alignment are safe."""
    lateral_retreat = smooth((abs(float(lateral_error_m)) - lateral_tolerance_m) / (LATERAL_TOL - lateral_tolerance_m))
    height_retreat = smooth((abs(float(height_error_m)) - height_tolerance_m) / (HEIGHT_TOL - height_tolerance_m))
    retreat = max(float(lateral_retreat), float(height_retreat))
    return float(capture_depth_m + retreat * (retreat_depth_m - capture_depth_m))


def lean_angle(pelvis_pos, target, lean_max):
    """Torso lean target for the supported reach: the height term (the hand must get down to the
    grip) or the distance term (the shoulder must get forward: 0 at 0.30 m, full at 0.60 m
    pelvis-to-grip), whichever asks for more, capped at lean_max."""
    height = (float(pelvis_pos[2]) + 0.12 - float(target[2])) / 0.6
    d_xy = float(np.linalg.norm(np.asarray(target[:2], dtype=float) - np.asarray(pelvis_pos[:2], dtype=float)))
    distance = (d_xy - 0.30) / 0.30 * lean_max
    return float(np.clip(max(height, distance), 0.0, lean_max))


def minjerk(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * u * (10 - 15 * u + 6 * u * u)


def reach_path(start, goal, tau):
    """Minimum-jerk tool path from the takeover point to the grip: the slide-in runs over the
    whole duration, the descent over its middle 60 % so the hand is at grip height before the
    jaws reach the bottle. Zero velocity at both ends, so the close starts from rest."""
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    out = start + minjerk(tau) * (goal - start)
    out[2] = start[2] + minjerk((tau - 0.2) / 0.6) * (goal[2] - start[2])
    return out


class ReferenceFilter:
    """Two cascaded first-order low-passes on the sent joint reference; velocity from the
    filtered stream. Off when cutoff_hz is 0. The per-tick whole-body solve (3 iterations) and
    its finite-difference velocity carried ~24 % of their velocity power above 2 Hz."""

    def __init__(self, cutoff_hz):
        self.tau = 1.0 / (2 * np.pi * cutoff_hz)
        self.s1 = self.s2 = None

    def step(self, joints, dt):
        q = np.asarray(joints, dtype=float)
        if self.s1 is None:
            self.s1, self.s2 = q.copy(), q.copy()
            return q.copy(), np.zeros(len(q))
        a = 1 - np.exp(-dt / self.tau)
        self.s1 += a * (q - self.s1)
        previous = self.s2.copy()
        self.s2 += a * (self.s1 - self.s2)
        return self.s2.copy(), (self.s2 - previous) / dt


class TerminalMotionObserver:
    """Estimate signed terminal motion from one continuous position window."""

    def __init__(
        self,
        acceleration,
        lookahead_seconds,
        *,
        window_seconds=0.20,
        min_samples=6,
        max_gap_seconds=0.05,
    ):
        self.acceleration = float(acceleration)
        self.lookahead_seconds = float(lookahead_seconds)
        self.window_seconds = float(window_seconds)
        self.max_gap_seconds = float(max_gap_seconds)
        self.min_samples = int(min_samples)
        if (
            not np.isfinite(self.acceleration)
            or self.acceleration <= 0
            or not np.isfinite(self.lookahead_seconds)
            or self.lookahead_seconds < 0
            or not np.isfinite(self.window_seconds)
            or self.window_seconds <= 0
            or not np.isfinite(self.max_gap_seconds)
            or self.max_gap_seconds <= 0
            or type(min_samples) is not int
            or self.min_samples < 2
        ):
            raise ValueError("terminal observer bounds must be positive and finite")
        self.samples = []
        self.last_stamp = None
        self.phase = None
        self.last_reset_reason = "cold_start"

    def reset(self, reason):
        self.samples.clear()
        self.last_stamp = None
        self.phase = None
        self.last_reset_reason = str(reason)

    @staticmethod
    def _least_squares_rate(stamps, positions):
        centred = stamps - np.mean(stamps)
        denominator = float(centred @ centred)
        if denominator <= 0:
            return None
        return float(centred @ (positions - np.mean(positions)) / denominator)

    def step(self, stamp, position, entry_axis, depth_m, *, phase="reach"):
        stamp = float(stamp)
        position = np.asarray(position, dtype=float)
        axis = np.asarray(entry_axis, dtype=float)
        depth_m = float(depth_m)
        if (
            not np.isfinite(stamp)
            or position.shape != (3,)
            or not np.all(np.isfinite(position))
            or axis.shape != (3,)
            or not np.all(np.isfinite(axis))
            or not np.isfinite(depth_m)
        ):
            raise ValueError("terminal observer sample must be finite 3D geometry")
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 0:
            raise ValueError("terminal observer entry axis must be nonzero")
        axis = axis / axis_norm

        reset_this_sample = None
        if self.phase is not None and phase != self.phase:
            self.reset("phase_boundary")
            reset_this_sample = self.last_reset_reason
        if self.last_stamp is not None:
            dt = stamp - self.last_stamp
            if dt <= 0:
                self.reset("non_increasing_time")
                reset_this_sample = self.last_reset_reason
            elif dt > self.max_gap_seconds:
                self.reset("clock_gap")
                reset_this_sample = self.last_reset_reason

        self.phase = phase
        self.last_stamp = stamp
        self.samples.append((stamp, float(position @ axis)))
        cutoff = stamp - self.window_seconds
        while len(self.samples) > 1 and self.samples[1][0] <= cutoff:
            self.samples.pop(0)

        stamps = np.asarray([sample[0] for sample in self.samples], dtype=float)
        positions = np.asarray([sample[1] for sample in self.samples], dtype=float)
        held_seconds = float(stamps[-1] - stamps[0])
        raw_velocity = None
        if len(self.samples) >= 2:
            raw_velocity = float((positions[-1] - positions[-2]) / (stamps[-1] - stamps[-2]))
        trend_velocity = self._least_squares_rate(stamps, positions)
        oscillation_envelope = None
        stopping_displacement = None
        predicted_interval = None
        if trend_velocity is not None:
            intercept = float(np.mean(positions) - trend_velocity * np.mean(stamps))
            residual = positions - (intercept + trend_velocity * stamps)
            # Use the full detrended peak-to-peak excursion on each side. This
            # deliberately overbounds an unknown oscillation phase at capture.
            oscillation_envelope = float(np.max(residual) - np.min(residual))
            stopping_displacement = signed_stopping_displacement(
                trend_velocity,
                self.acceleration,
                self.lookahead_seconds,
            )
            predicted_centre = depth_m - stopping_displacement
            predicted_interval = [
                predicted_centre - oscillation_envelope,
                predicted_centre + oscillation_envelope,
            ]

        enough_samples = len(self.samples) >= self.min_samples
        enough_history = held_seconds + 1e-9 >= self.window_seconds
        ready = bool(enough_samples and enough_history and predicted_interval is not None)
        if ready:
            readiness_reason = None
        elif not enough_samples:
            readiness_reason = "insufficient_samples"
        else:
            readiness_reason = "insufficient_history"
        return {
            "ready": ready,
            "readiness_reason": readiness_reason,
            "reset_reason": reset_this_sample,
            "last_reset_reason": self.last_reset_reason,
            "sample_count": len(self.samples),
            "held_seconds": held_seconds,
            "window_seconds": self.window_seconds,
            "raw_velocity_m_s": raw_velocity,
            "trend_velocity_m_s": trend_velocity,
            "oscillation_envelope_m": oscillation_envelope,
            "stopping_displacement_m": stopping_displacement,
            "predicted_depth_interval_m": predicted_interval,
        }


class TerminalRetreatCommand:
    """Track a nonnegative retreat with bounded speed and acceleration."""

    def __init__(self, speed_limit, acceleration_limit):
        self.speed_limit = float(speed_limit)
        self.acceleration_limit = float(acceleration_limit)
        if (
            not np.isfinite(self.speed_limit)
            or self.speed_limit <= 0
            or not np.isfinite(self.acceleration_limit)
            or self.acceleration_limit <= 0
        ):
            raise ValueError("terminal retreat bounds must be positive and finite")
        self.value = np.zeros(1, dtype=float)
        self.rate = np.zeros(1, dtype=float)
        self.target = 0.0

    def step(self, target, dt, *, hold=False):
        target = float(target)
        dt = float(dt)
        if not np.isfinite(target) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("terminal retreat update must be positive-time and finite")
        if not hold:
            self.target = max(0.0, target)
        self.value, self.rate = bounded_correction_step(
            self.value,
            self.rate,
            [self.target],
            dt,
            self.speed_limit,
            self.acceleration_limit,
        )
        return float(self.value[0]), float(self.rate[0])


def terminal_retreat_speed_limit(hand_speed, acceleration, window_seconds):
    """Keep retreat response bandwidth fixed as the hand speed limit changes."""
    hand_speed = float(hand_speed)
    acceleration = float(acceleration)
    window_seconds = float(window_seconds)
    if (
        not np.isfinite(hand_speed)
        or hand_speed <= 0
        or not np.isfinite(acceleration)
        or acceleration <= 0
        or not np.isfinite(window_seconds)
        or window_seconds <= 0
    ):
        raise ValueError("terminal retreat speed inputs must be positive and finite")
    return min(hand_speed, acceleration * window_seconds)


def signed_stopping_displacement(velocity, acceleration, lookahead_seconds):
    """Signed lookahead plus constant-acceleration stopping displacement."""
    velocity = float(velocity)
    acceleration = float(acceleration)
    lookahead_seconds = float(lookahead_seconds)
    if (
        not np.isfinite(velocity)
        or not np.isfinite(acceleration)
        or acceleration <= 0
        or not np.isfinite(lookahead_seconds)
        or lookahead_seconds < 0
    ):
        raise ValueError("stopping-displacement inputs must be finite with positive acceleration")
    magnitude = abs(velocity)
    return float(np.sign(velocity) * (magnitude * lookahead_seconds + magnitude**2 / (2.0 * acceleration)))


def momentum_guarded_depth(
    guarded_depth_m,
    commanded_retreat_m,
    maximum_depth_m=0.10,
):
    """Retreat the active goal smoothly while retaining the cap-clearance bound."""
    retreat = max(0.0, float(commanded_retreat_m))
    uncapped = float(guarded_depth_m) + retreat
    depth = min(float(maximum_depth_m), uncapped)
    return depth, retreat, uncapped > depth


def predictive_capture_geometry(geometry, observer):
    """Require the full conservative observer interval to occupy the capture window."""
    result = dict(geometry)
    depth_min = float(result["preferred_depth_m"] - result["depth_tolerance_m"])
    depth_max = float(result["preferred_depth_m"] + result["depth_tolerance_m"])
    predicted_interval = observer["predicted_depth_interval_m"]
    observer_ready = bool(observer["ready"])
    interval_ready = bool(
        observer_ready
        and predicted_interval is not None
        and depth_min <= float(predicted_interval[0])
        and float(predicted_interval[1]) <= depth_max
    )
    current_geometry_ready = bool(result["ready"])
    result["ready"] = current_geometry_ready and interval_ready
    result["rejection_reasons"] = list(result["rejection_reasons"])
    if not observer_ready:
        result["rejection_reasons"].append("terminal_observer_not_ready")
    elif not interval_ready:
        result["rejection_reasons"].append("predicted_depth_interval")
    result["stopping"] = {
        "current_geometry_ready": current_geometry_ready,
        **observer,
        "accepted_depth_interval_m": [depth_min, depth_max],
        "admitted": interval_ready,
    }
    return result


def entry_step(
    command,
    measured,
    target,
    entry_axis,
    dt,
    hand_speed,
    final_depth_m=0.018,
    *,
    brake_acceleration=None,
):
    """One continuous, cap-clearing step toward the bottle's capture region."""
    return cap_clearing_entry_step(
        command,
        measured,
        target,
        entry_axis,
        dt,
        hand_speed,
        final_depth_m,
        brake_acceleration=brake_acceleration,
        terminal_speed_fn=terminal_entry_speed,
    )


def terminal_entry_speed(
    remaining,
    nominal_speed,
    acceleration,
):
    """Decelerate the entry command continuously as its active goal approaches."""
    remaining = max(0.0, float(remaining))
    braking_speed = np.sqrt(2.0 * float(acceleration) * remaining)
    return float(min(float(nominal_speed), braking_speed))


def advance_entry_command(
    initial,
    command,
    measured,
    target,
    entry_axis,
    tau,
    dt,
    hand_speed,
    final_depth_m,
    *,
    brake_acceleration,
):
    """Apply one dynamic axial goal to timed and per-tick entry paths."""
    goal = capture_tool_goal(target, entry_axis, final_depth_m)
    if tau < 1.0:
        following = reach_path(initial, goal, tau)
        axis = np.asarray(entry_axis, dtype=float)
        axis /= np.linalg.norm(axis)
        return following, float((np.asarray(target) - following) @ axis)
    return entry_step(
        command,
        measured,
        target,
        entry_axis,
        dt,
        hand_speed,
        final_depth_m,
        brake_acceleration=brake_acceleration,
    )


def point_in_palm_frame(state, side, world_position):
    """World point relative to a rotating measured palm frame."""
    palm = state["palms"][side]
    return Primitives._rot(palm["quat"]).T @ (
        np.asarray(world_position, dtype=float) - np.asarray(palm["pos"], dtype=float)
    )


def object_in_palm_frame(state, side, object_position):
    """Object position relative to a rotating measured palm frame."""
    return point_in_palm_frame(state, side, object_position)


def advance_load_progress(progress, dt, duration, contact_scale):
    """Advance a monotone load trajectory by its smoothly gated physical time."""
    return min(1.0, progress + min(dt, 0.1) * float(np.clip(contact_scale, 0.0, 1.0)) / duration)


def advance_loaded_progress(progress, dt, duration, contact_scale, tracking_scale):
    """Advance loaded motion with continuous contact and measured tracking load scales."""
    return advance_load_progress(
        progress,
        dt,
        duration,
        float(np.clip(contact_scale, 0.0, 1.0)) * float(np.clip(tracking_scale, 0.0, 1.0)),
    )


def push_loaded_reference(
    arm,
    sent,
    sent_velocity,
    sent_root_angular_velocity,
    state,
    dt,
    *,
    endpoint_pending,
    request_count,
):
    """Send every POSE tick and return evidence when an endpoint worker is pending."""
    stamp = float(state["t"])
    arm._push(
        sent,
        reference_velocity=sent_velocity,
        reference_time=stamp,
        root_angular_velocity=sent_root_angular_velocity,
    )
    if not endpoint_pending:
        return None
    return {
        "seq": state.get("seq"),
        "t": stamp,
        "wall": state.get("wall"),
        "dt": float(dt),
        "request_count": int(request_count),
    }


class CaptureSettleGate:
    """Admit bounded measured capture geometry over a physical-time window."""

    def __init__(
        self,
        quiet_seconds,
        min_samples,
        position_diameter_max,
        drift_speed_max,
        rotation_diameter_max,
        angular_drift_max,
    ):
        self.quiet_seconds = float(quiet_seconds)
        self.min_samples = int(min_samples)
        self.position_diameter_max = float(position_diameter_max)
        self.drift_speed_max = float(drift_speed_max)
        self.rotation_diameter_max = float(rotation_diameter_max)
        self.angular_drift_max = float(angular_drift_max)
        self.last_stamp = None
        self.samples = []

    @staticmethod
    def _least_squares_rate(stamps, values):
        if len(stamps) < 2:
            return None
        centred_time = stamps - np.mean(stamps)
        denominator = float(centred_time @ centred_time)
        if denominator <= 0:
            return None
        slope = np.sum(centred_time[:, None] * values, axis=0) / denominator
        return float(np.linalg.norm(slope))

    def _status(self, centred):
        count = len(self.samples)
        held_seconds = 0.0 if count < 2 else self.samples[-1][0] - self.samples[0][0]
        if count:
            positions = np.asarray([sample[1] for sample in self.samples])
            rotations = np.asarray([sample[2] for sample in self.samples])
            position_diameter = float(np.max(np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)))
            stamps = np.asarray([sample[0] for sample in self.samples])
            drift_speed = self._least_squares_rate(stamps, positions)
            rotation_diameter = max(
                (
                    float(Rotation.from_matrix(rotations[j] @ rotations[i].T).magnitude())
                    for i in range(count)
                    for j in range(i + 1, count)
                ),
                default=0.0,
            )
            rotation_vectors = Rotation.from_matrix(rotations @ rotations[0].T).as_rotvec()
            angular_drift = self._least_squares_rate(stamps, rotation_vectors)
        else:
            position_diameter = 0.0
            drift_speed = None
            rotation_diameter = 0.0
            angular_drift = None

        bounded = (
            position_diameter <= self.position_diameter_max
            and drift_speed is not None
            and drift_speed <= self.drift_speed_max
            and rotation_diameter <= self.rotation_diameter_max
            and angular_drift is not None
            and angular_drift <= self.angular_drift_max
        )
        window_ready = count >= self.min_samples and held_seconds >= self.quiet_seconds
        return dict(
            centred=bool(centred),
            quiet=bool(bounded),
            sample_count=count,
            held_seconds=float(held_seconds),
            position_diameter_m=position_diameter,
            relative_drift_m_s=drift_speed,
            rotation_diameter_rad=rotation_diameter,
            angular_drift_rad_s=angular_drift,
            ready=bool(centred and window_ready and bounded),
        )

    def step(self, stamp, centred, relative, palm_rotation):
        stamp = float(stamp)
        relative = np.asarray(relative, dtype=float)
        palm_rotation = np.asarray(palm_rotation, dtype=float)
        if self.last_stamp is not None and stamp < self.last_stamp:
            raise ValueError("capture settling physical clock moved backwards")
        if not centred:
            self.samples.clear()
            if self.last_stamp is None or stamp > self.last_stamp:
                self.last_stamp = stamp
            return self._status(False)
        if self.last_stamp is not None and stamp == self.last_stamp:
            return self._status(True)

        self.last_stamp = stamp
        self.samples.append((stamp, relative.copy(), palm_rotation.copy()))
        cutoff = stamp - self.quiet_seconds
        while len(self.samples) > self.min_samples and self.samples[1][0] <= cutoff:
            self.samples.pop(0)
        return self._status(True)


class CaptureAdmissionConsumer:
    """Enter close once when capture is admitted during reach."""

    def __init__(self):
        self.entered_close = False

    def consume(self, phase, capture_geometry, enter_close):
        if phase != "reach" or self.entered_close or not capture_geometry["ready"]:
            return phase

        self.entered_close = True
        enter_close()
        return "close"


class ContactLoadController:
    """Bilateral-contact hysteresis and smooth load-rate control on the physical clock."""

    def __init__(self, confirm_seconds, loss_grace_seconds, recovery_timeout, speed_max):
        self.confirm_seconds = float(confirm_seconds)
        self.loss_grace_seconds = float(loss_grace_seconds)
        self.recovery_timeout = float(recovery_timeout)
        self.speed_max = float(speed_max)
        self.last_stamp = None
        self.last_relative = None
        self.filtered_speed = 0.0
        self.good_since = None
        self.wait_since = None
        self.loss_since = None
        self.admitted = False
        self.scale = 0.0

    def step(self, stamp, bilateral, relative, *, required):
        stamp = float(stamp)
        relative = np.asarray(relative, dtype=float)
        dt = 0.0 if self.last_stamp is None else stamp - self.last_stamp
        if dt < 0:
            raise ValueError("contact controller physical clock moved backwards")
        if dt > 0 and self.last_relative is not None:
            instantaneous = float(np.linalg.norm(relative - self.last_relative) / dt)
            alpha = 1.0 - np.exp(-dt / self.confirm_seconds)
            self.filtered_speed += alpha * (instantaneous - self.filtered_speed)
        quiet = self.filtered_speed <= self.speed_max
        good = bool(bilateral and quiet)

        if good:
            if self.good_since is None:
                self.good_since = stamp
            if stamp - self.good_since >= self.confirm_seconds:
                self.admitted = True
                self.loss_since = None
                self.wait_since = None
        else:
            self.good_since = None
            if self.admitted and self.loss_since is None:
                self.loss_since = stamp
            if self.loss_since is not None and stamp - self.loss_since >= self.loss_grace_seconds:
                self.admitted = False

        ready = self.admitted and self.loss_since is None
        if required and not ready:
            if self.wait_since is None:
                self.wait_since = stamp
        elif not required:
            self.wait_since = None
        timed_out = bool(
            required and not ready and self.wait_since is not None and stamp - self.wait_since >= self.recovery_timeout
        )

        target_scale = 1.0 if ready else 0.0
        ramp_seconds = self.confirm_seconds if ready else self.loss_grace_seconds
        max_delta = dt / ramp_seconds if dt > 0 else 0.0
        self.scale += float(np.clip(target_scale - self.scale, -max_delta, max_delta))
        self.last_stamp = stamp
        self.last_relative = relative.copy()
        return dict(
            bilateral=bool(bilateral),
            quiet=quiet,
            admitted=ready,
            filtered_speed_m_s=float(self.filtered_speed),
            progress_scale=float(self.scale),
            loss_seconds=0.0 if self.wait_since is None else float(stamp - self.wait_since),
            timed_out=timed_out,
        )


def close_transition(arm_q, supported_q, targets, rotations, actual, measured_rotations):
    """Preserve the sent joint reference and report the measured Cartesian error."""
    targets = np.asarray(targets, dtype=float).copy()
    rotations = np.asarray(rotations, dtype=float).copy()
    actual = np.asarray(actual, dtype=float)
    measured_rotations = np.asarray(measured_rotations, dtype=float)
    diagnostics = dict(
        sent_q_continuity_error_rad=float(np.max(np.abs(np.asarray(supported_q) - np.asarray(arm_q)))),
        actual_tool_error=np.linalg.norm(targets - actual, axis=1),
        actual_rotation_error_deg=np.degrees(
            Rotation.from_matrix(rotations @ measured_rotations.transpose(0, 2, 1)).magnitude()
        ),
    )
    return targets, rotations, diagnostics


def freeze_rotation_reference(reference_rotations, rotation_feedback, tracking_gain):
    """Hold the active reference rotation with its current tracking compensation."""
    reference_rotations = np.asarray(reference_rotations, dtype=float)
    correction = Rotation.from_rotvec(float(tracking_gain) * np.asarray(rotation_feedback, dtype=float)).as_matrix()
    return correction.transpose(0, 2, 1) @ reference_rotations


def begin_live_translation_hold(motion, measured):
    """Hold measured world tools without pausing bounded translation feedback."""
    reference = motion.body.tools(motion.x).copy()
    measured = np.asarray(measured, dtype=float)
    if measured.shape != reference.shape or not np.isfinite(measured).all():
        raise ValueError("measured tools must match the finite active reference")
    return measured.copy(), reference


def loaded_support_problem_snapshot(motion):
    """Copy the live fixed-support optimization problem for an isolated endpoint worker."""
    body = motion.body
    return {
        "heading": body.heading.copy(),
        "feet_pos": body.feet_pos.copy(),
        "feet_rot": body.feet_rot.copy(),
        "com0": body.com0.copy(),
        "lo": body.lo.copy(),
        "hi": body.hi.copy(),
        "hand_weights": body.hand_weights.copy(),
        "com_weight": float(body.com_weight),
    }


class LoadedEndpointEvaluator:
    """Evaluate a loaded endpoint on private IK/collision state with the live support problem."""

    def __init__(self, ik, guard, config):
        self.ik = ik
        self.guard = guard
        self.config = config

    def _clear(self, state, candidate, candidate_base):
        for side in ("left", "right", "body"):
            hit = self.guard.check(
                state,
                candidate,
                side,
                allow_target_contact=side == "right",
                candidate_base=candidate_base,
                ignore_target=side == "right",
            )
            if not hit.clear:
                return f"whole-body clearance: {side}:{hit.link}:{hit.obstacle}"
        return None

    def evaluate(self, payload):
        started = time.perf_counter()
        state = payload["state"]
        support = payload["support_problem"]
        body = WholeBodyIK(
            self.ik,
            state,
            self.config,
            tool_offset=OFFSET,
            hand_weights=support["hand_weights"],
            com_weight=support["com_weight"],
        )
        for name in ("heading", "feet_pos", "feet_rot", "com0", "lo", "hi"):
            setattr(body, name, np.asarray(support[name], dtype=float).copy())

        endpoint_targets = np.asarray(payload["targets"], dtype=float).copy()
        endpoint_targets[1] = np.asarray(payload["endpoint"], dtype=float)
        reference_x = np.asarray(payload["reference_x"], dtype=float)
        reference_root = np.asarray(payload["reference_root"], dtype=float)
        pose = body.solve(
            endpoint_targets,
            payload["rotations"],
            seed=reference_x,
            posture=payload["standing_reference"],
            posture_weights=payload["standing_weights"],
            torso_rotation=body.heading,
            iterations=self.config.stance_solver_iterations,
        )
        torso_rotation = self.ik.data.xmat[body.torso].reshape(3, 3).copy()
        nominal_candidate_joints = pose.joints.copy()
        measured_joints = np.asarray(state["joints"], dtype=float)
        joint_tracking_offset = measured_joints - reference_x[4:]
        candidate_joints = nominal_candidate_joints + joint_tracking_offset
        delta_rotation = pose.rotation @ Primitives._rot(reference_root).T
        candidate_quat = Rotation.from_matrix(delta_rotation @ Primitives._rot(state["pelvis"]["quat"])).as_quat()[
            [3, 0, 1, 2]
        ]
        candidate_base = np.r_[
            np.asarray(state["pelvis"]["pos"], dtype=float) + pose.position - reference_x[:3],
            candidate_quat,
        ]
        nominal_candidate_base = np.r_[
            pose.position,
            Rotation.from_matrix(pose.rotation).as_quat()[[3, 0, 1, 2]],
        ]
        clearance = self._clear(state, candidate_joints, candidate_base)
        standing_reference = np.asarray(payload["standing_reference"], dtype=float)
        hand_error = float(max(pose.hand_errors))
        foot_error = float(max(pose.foot_errors))
        com_error = float(np.linalg.norm(pose.com_shift))
        reason = clearance
        if reason is None and hand_error > self.config.stance_ik_tolerance:
            reason = "endpoint hand error"
        if reason is None and foot_error > 0.025:
            reason = "endpoint fixed-foot error"
        if reason is None and com_error > 0.04:
            reason = "endpoint COM shift"
        return {
            "admitted": reason is None,
            "reason": reason,
            "endpoint": np.asarray(payload["endpoint"], dtype=float).copy(),
            "nominal_candidate_joints": nominal_candidate_joints,
            "candidate_joints": candidate_joints,
            "joint_tracking_offset": joint_tracking_offset,
            "nominal_candidate_base": nominal_candidate_base,
            "candidate_base": candidate_base,
            "hand_errors": pose.hand_errors.copy(),
            "foot_errors": pose.foot_errors.copy(),
            "com_shift": pose.com_shift.copy(),
            "standing_leg_error_rad": float(np.max(np.abs(pose.joints[:12] - standing_reference[:12]))),
            "endpoint_root_height_m": float(candidate_base[2]),
            "endpoint_torso_lean_rad": float(np.arccos(np.clip(torso_rotation[2, 2], -1.0, 1.0))),
            "solve_seconds": time.perf_counter() - started,
            "source": payload.get("source"),
            "support_problem": {
                name: np.asarray(support[name], dtype=float).copy()
                for name in ("heading", "feet_pos", "feet_rot", "com0", "lo", "hi", "hand_weights")
            },
            "support_com_weight": float(support["com_weight"]),
            "reference_x": reference_x.copy(),
            "reference_root": reference_root.copy(),
        }


def torso_offset(ik, motion, point):
    """A world point expressed in the reference torso frame of the commanded body state."""
    motion.body.put(motion.x)
    frame = ik.data.xmat[motion.body.torso].reshape(3, 3)
    return frame.T @ (np.asarray(point) - ik.data.xpos[motion.body.torso])


def lift_target(ik, motion, offset, rise):
    """The torso-frame offset re-expressed in the current commanded torso, raised by `rise` in world z."""
    motion.body.put(motion.x)
    frame = ik.data.xmat[motion.body.torso].reshape(3, 3)
    return ik.data.xpos[motion.body.torso] + frame @ offset + np.array([0.0, 0.0, rise])


class WholeBodyGraspPolicy:
    def __init__(self, robot, task, config, arm_ramp, execution):
        require_pose_ownership(hand_handoff=True)
        self.p, self.task, self.config, self.ramp, self.execution = robot, task, config, arm_ramp, execution
        self.cfg = config.whole_body
        self.ik = ArmIK()
        self.guard = task.transport_guard(self.ik, config)
        self.guard.finger_only_target_contact = True
        self.guard.allow_target_contact_meshes(GRASP_ENVELOPE_MESHES)
        self.stance_planner = None
        self.stance_guard = None
        if self.cfg.natural_walk_prep:
            stance_ik = ArmIK()
            self.stance_guard = task.transport_guard(stance_ik, config)
            self.stance_guard.finger_only_target_contact = True
            self.stance_guard.allow_target_contact_meshes(GRASP_ENVELOPE_MESHES)
            self.stance_planner = AsyncStanceReachPlanner(
                StanceReachEvaluator(stance_ik, self.cfg, self._stance_feasibility)
            )
        loaded_ik = ArmIK()
        loaded_guard = task.transport_guard(loaded_ik, config)
        loaded_guard.finger_only_target_contact = True
        loaded_guard.allow_target_contact_meshes(GRASP_ENVELOPE_MESHES)
        self.loaded_endpoint_planner = AsyncLoadedEndpointPlanner(
            LoadedEndpointEvaluator(loaded_ik, loaded_guard, self.cfg)
        )
        bottle_radius = float(max(np.asarray(task.target.box["half_extent"], dtype=float)[:2]))
        self.capture_clearance = capture_axial_clearance(
            self.cfg.capture_depth_m,
            self.cfg.capture_depth_tolerance_m,
            bottle_radius,
        )
        self.capture_retreat_depth_m = float(DEX1_PAD_X_MAX - OFFSET[0] + bottle_radius + config.transport_margin)
        if not self.capture_clearance["ready"]:
            raise ValueError("whole-body centred capture window does not fit the configured bottle in both Dex1 pads")
        self.target = BottleTargetObserver(self.ik, task, self.cfg.vision_timeout)
        self.error = None
        self.arm = self.motion = None
        self.completed = False
        self.walking = WalkingReach(
            self.ik,
            self.cfg,
            arm_ramp.q,
            brake_depth=config.visible_brake_depth,
            brake_edge=config.visible_brake_edge,
            feasibility=self._walking_feasibility if self.cfg.natural_walk_prep else None,
            stance_planner=self.stance_planner,
        )
        self.stats = execution.metrics.setdefault(
            "whole_body", dict(samples=[], walking=[], references=[], source="assisted", clock="simulation")
        )
        self.stats["capture_window"] = dict(
            preferred_depth_m=self.cfg.capture_depth_m,
            tolerance_m=self.cfg.capture_depth_tolerance_m,
            lateral_tolerance_m=self.cfg.capture_lateral_tolerance_m,
            height_tolerance_m=self.cfg.capture_height_tolerance_m,
            retreat_depth_m=self.capture_retreat_depth_m,
            bottle_radius_m=bottle_radius,
            **self.capture_clearance,
        )
        self.previous_observer = robot.cmd.reference_observer
        robot.cmd.reference_observer = self.observe_reference
        self.context = {}
        self.closed = False
        self._walking_edge_distance = None
        self._approach_settlement_observer = None
        self._base_settled = False
        self._settled_request = None
        self._settled_plan = None
        self._settled_request_count = 0
        self._settled_wait_started = None
        self._settled_preparation_samples = []
        self._settled_preparation_source = None
        self._stream_released = False

    def set_approach_settlement_observer(self, observer):
        """Continue approach stop diagnostics through the physical brake settle."""
        self._approach_settlement_observer = observer

    def _observe_approach_settlement(
        self,
        state,
        *,
        complete=False,
        settled=False,
        completion_reason=None,
    ):
        observer = getattr(self, "_approach_settlement_observer", None)
        if observer is not None:
            observer(
                state,
                complete=complete,
                settled=settled,
                completion_reason=completion_reason,
            )

    def observe_reference(self, channel, payload):
        if self.previous_observer is not None:
            self.previous_observer(channel, payload)
        # Current consumed labels remain exporter-owned. Log sent reference
        # identity separately, including buffered future samples during handoff.
        if channel == "pose":
            # A preview contains ~50 future frames on every tick. Record its
            # identity/endpoints, not quadratic copies of the same trajectory;
            # samples below retain the current joint reference and velocity.
            payload = dict(payload)
            payload["preview_frames"] = len(payload["frame_index"])
            for key in ("joint_pos", "joint_vel", "body_quat", "frame_index"):
                payload[key] = [payload[key][0], payload[key][-1]]
        self.stats["references"].append(dict(channel=channel, sent_wall=time.time(), **self.context, payload=payload))

    def close(self):
        if not self.closed:
            if getattr(self, "stance_planner", None) is not None:
                self.stance_planner.close()
            if getattr(self, "loaded_endpoint_planner", None) is not None:
                self.loaded_endpoint_planner.close()
            self.target.close()
            self.p.cmd.reference_observer = self.previous_observer
            self.closed = True
            normalized = json.loads(
                json.dumps(
                    self.stats, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else float(value)
                )
            )
            self.stats.clear()
            self.stats.update(normalized)

    def _walking_feasibility(self, state, candidate, candidate_base):
        """Admit a future stance/pregrasp with the task's real swept guard."""
        checks = []
        for side in ("right", "left", "body"):
            result = self.guard.check(state, candidate, side, candidate_base=candidate_base)
            checks.append(dict(side=side, **result.to_dict()))
            if not result.clear:
                return dict(
                    clear=False,
                    reason=f"{side}:{result.link}:{result.obstacle}",
                    required_margin=float(self.guard.margin),
                    checks=checks,
                )
        return dict(
            clear=True,
            reason=None,
            required_margin=float(self.guard.margin),
            positive_clearance=True,
            checks=checks,
        )

    def _stance_feasibility(
        self,
        state,
        candidate,
        candidate_base,
        *,
        allow_target_contact=False,
        path=None,
        endpoint_only=False,
    ):
        """Run worker-only swept checks on the worker's private guard."""
        checks = []
        sweep_state = deepcopy(state)
        if endpoint_only:
            sweep_state["joints"] = np.asarray(candidate, dtype=float).copy()
            sweep_state["pelvis"] = dict(sweep_state["pelvis"])
            sweep_state["pelvis"].update(
                pos=np.asarray(candidate_base, dtype=float)[:3].copy(),
                quat=np.asarray(candidate_base, dtype=float)[3:].copy(),
            )
        waypoints = [{"joints": candidate, "candidate_base": candidate_base}] if path is None else path
        for path_index, waypoint in enumerate(waypoints):
            waypoint_joints = np.asarray(waypoint["joints"], dtype=float)
            waypoint_base = np.asarray(waypoint["candidate_base"], dtype=float)
            waypoint_target_contact = bool(
                allow_target_contact and (path is None or waypoint.get("allow_target_contact", False))
            )
            for side in ("right", "left", "body"):
                result = self.stance_guard.check(
                    sweep_state,
                    waypoint_joints,
                    side,
                    candidate_base=waypoint_base,
                    allow_target_contact=waypoint_target_contact,
                )
                checks.append(dict(path_index=path_index, side=side, **result.to_dict()))
                if not result.clear:
                    return dict(
                        clear=False,
                        reason=f"{side}:{result.link}:{result.obstacle}",
                        required_margin=float(self.stance_guard.margin),
                        path_source=(None if path is None else "supported_cap_clearing_cartesian_reference"),
                        checks=checks,
                    )
            sweep_state["joints"] = waypoint_joints.copy()
            sweep_state["pelvis"] = dict(sweep_state["pelvis"])
            sweep_state["pelvis"].update(
                pos=waypoint_base[:3].copy(),
                quat=waypoint_base[3:].copy(),
            )
        return dict(
            clear=True,
            reason=None,
            required_margin=float(self.stance_guard.margin),
            positive_clearance=True,
            path_source=(None if path is None else "supported_cap_clearing_cartesian_reference"),
            path_samples=len(waypoints),
            checks=checks,
        )

    def walk_tick(self, state, *, edge_distance=None, navigation_heading=None, navigation_target=None):
        if edge_distance is not None:
            self._walking_edge_distance = float(edge_distance)
        else:
            edge_distance = self._walking_edge_distance
        if self.walking.last == float(state["t"]):
            return self.error is None
        started = time.perf_counter()
        target = self.target.observe(state)
        observed = time.perf_counter()
        self.context = dict(seq=state["seq"], t=state["t"], phase="prepare", camera=self.target.diagnostics.copy())
        previous = self.walking.q.copy()
        try:
            wanted, velocity = self.walking.step(
                state,
                target,
                ego_px=self.target.diagnostics.get("ego_height_px"),
                camera=self.target.diagnostics,
                edge_distance=edge_distance,
                navigation_heading=navigation_heading,
                navigation_target=navigation_target,
            )
        except ValueError as exc:
            self.error = str(exc)
            self.stats["failure"] = dict(state=state, previous_t=self.walking.last, reason=self.error)
            return False
        planned = time.perf_counter()
        predicted = np.asarray(state["joints"], dtype=float).copy()
        delta = wanted - previous
        predicted[12:15] += delta[:3]
        predicted[15:22] += delta[3::2]
        predicted[22:29] += delta[4::2]
        # The prepared hand is meant to arrive beside the bottle: a finger
        # predicted at the bottle's surface at handoff is the goal, not a
        # collision (it rejected 12/81 rounds, all parked under 0.44 m).
        self.error = self._clear(state, predicted, True)
        if self.error:
            self.stats["failure"] = dict(state=state, reference=wanted, reason=self.error)
            return False
        self.p.cmd.set_upper_body(wanted)
        self.p.cmd.set_upper_body_vel(velocity)
        elapsed = dict(camera=observed - started, planning=planned - observed, clearance=time.perf_counter() - planned)
        self.stats["last_walk_timing"] = dict(t=state["t"], **elapsed)
        if self.walking.diagnostics.get("plan_updated"):
            self.stats["walking"].append(dict(**self.walking.diagnostics, camera=self.target.diagnostics.copy()))
        if self.walking.diagnostics.get("stance_plan_status") == "ready":
            self.stats["prestop_stance_plan"] = self.walking.diagnostics.get("stance_plan_result")
        return True

    def _preparation_hold_tick(self, state):
        """Brake and hold the owned pregrasp reference on the physical clock."""
        previous = self.walking.q.copy()
        try:
            wanted, velocity = self.walking.brake(state)
        except ValueError as exc:
            self.error = str(exc)
            self.stats["failure"] = dict(
                state=state,
                previous_t=self.walking.last,
                reason=self.error,
                preparation_hold=True,
            )
            return False

        delta = wanted - previous
        predicted = np.asarray(state["joints"], dtype=float).copy()
        predicted[12:15] += delta[:3]
        predicted[15:22] += delta[3::2]
        predicted[22:29] += delta[4::2]
        self.error = self._clear(state, predicted, True)
        if self.error:
            self.stats["failure"] = dict(
                state=state,
                reference=wanted,
                reason=self.error,
                preparation_hold=True,
            )
            return False

        self.p.cmd.set_upper_body(wanted)
        self.p.cmd.set_upper_body_vel(velocity)
        self.stats["preparation_hold"] = dict(
            t=float(state["t"]),
            reference=wanted.copy(),
            velocity=velocity.copy(),
            reference_speed_rad_s=float(np.max(np.abs(velocity))),
        )
        return True

    def _hold_planner(self, state, *, stream_mode=None):
        self.p.cmd.frame(
            mode=LocomotionMode.IDLE,
            height=-1.0,
            speed=0.0,
            movement=(0, 0, 0),
            facing=self.p._facing(state),
            stream_mode=stream_mode,
        )

    def _settled_owned_reference(self, state):
        arm = getattr(self, "arm", None)
        if arm is not None and np.asarray(arm.q).shape == (29,):
            speed = max(
                float(np.max(np.abs(np.asarray(arm.reference_velocity, dtype=float)))),
                float(np.linalg.norm(np.asarray(arm.root_angular_velocity, dtype=float))),
            )
            return (
                np.asarray(arm.q, dtype=float).copy(),
                np.asarray(arm.root_quat, dtype=float).copy(),
                speed,
                "pose_owned_reference",
            )

        joints = np.asarray(state["joints"], dtype=float).copy()
        upper = np.asarray(self.walking.q, dtype=float)
        joints[12:15] = upper[:3]
        joints[15:22] = upper[3::2]
        joints[22:29] = upper[4::2]
        return (
            joints,
            np.asarray(state["pelvis"]["quat"], dtype=float).copy(),
            float(np.max(np.abs(np.asarray(self.walking.v, dtype=float)))),
            "walking_owned_upper_reference",
        )

    def _settled_preparation_status(self, state, command_speed, reference_source):
        samples = getattr(self, "_settled_preparation_samples", [])
        previous_source = getattr(self, "_settled_preparation_source", None)
        if previous_source != reference_source:
            samples = []
        self._settled_preparation_source = reference_source

        window_seconds = max(0.10, float(self.cfg.capture_quiet_seconds))
        if command_speed > 0.05:
            self._settled_preparation_samples = []
            return {
                "ready": False,
                "reason": "reference_moving",
                "reference_speed_rad_s": float(command_speed),
                "sample_count": 0,
                "held_seconds": 0.0,
            }

        palm = state.get("palms", {}).get("right")
        if palm is None:
            self._settled_preparation_samples = []
            return {
                "ready": False,
                "reason": "right_palm_measurement_unavailable",
                "reference_speed_rad_s": float(command_speed),
                "sample_count": 0,
                "held_seconds": 0.0,
            }

        stamp = float(state["t"])
        sample = (
            stamp,
            np.asarray(state["joints"], dtype=float).copy(),
            np.asarray(palm["pos"], dtype=float).copy(),
            Primitives._rot(palm["quat"]),
        )
        if samples and stamp < samples[-1][0]:
            samples = []
        if samples and stamp == samples[-1][0]:
            samples[-1] = sample
        else:
            samples.append(sample)
        while len(samples) > 1 and stamp - samples[1][0] >= window_seconds:
            samples.pop(0)
        self._settled_preparation_samples = samples

        stamps = np.asarray([item[0] for item in samples], dtype=float)
        held_seconds = 0.0 if len(samples) < 2 else float(stamps[-1] - stamps[0])
        joint_speed = hand_speed = angular_speed = None
        if len(samples) >= 2 and held_seconds > 0:
            centred = stamps - float(np.mean(stamps))
            denominator = float(centred @ centred)
            joints = np.asarray([item[1] for item in samples])
            hands = np.asarray([item[2] for item in samples])
            rotations = np.asarray([item[3] for item in samples])
            joint_slope = np.sum(centred[:, None] * joints, axis=0) / denominator
            hand_slope = np.sum(centred[:, None] * hands, axis=0) / denominator
            rotation_vectors = Rotation.from_matrix(rotations @ rotations[0].T).as_rotvec()
            rotation_slope = np.sum(centred[:, None] * rotation_vectors, axis=0) / denominator
            joint_speed = float(np.max(np.abs(joint_slope)))
            hand_speed = float(np.linalg.norm(hand_slope))
            angular_speed = float(np.linalg.norm(rotation_slope))

        ready = bool(
            len(samples) >= 3
            and held_seconds + 1e-9 >= window_seconds
            and joint_speed is not None
            and joint_speed <= 0.20
            and hand_speed <= self.cfg.capture_drift_speed_max
            and angular_speed <= self.cfg.capture_angular_drift_max
        )
        return {
            "ready": ready,
            "reason": None if ready else "measured_preparation_not_quiet",
            "reference_speed_rad_s": float(command_speed),
            "sample_count": len(samples),
            "held_seconds": held_seconds,
            "window_seconds": window_seconds,
            "measured_joint_speed_rad_s": joint_speed,
            "measured_hand_speed_m_s": hand_speed,
            "measured_hand_angular_speed_rad_s": angular_speed,
            "joint_speed_limit_rad_s": 0.20,
            "hand_speed_limit_m_s": self.cfg.capture_drift_speed_max,
            "hand_angular_speed_limit_rad_s": self.cfg.capture_angular_drift_max,
        }

    def _settled_support_diagnostics(self, request, state, target):
        tolerance = float(self.cfg.stance_plan_tolerance_m)
        target_error = (
            None if target is None else float(np.linalg.norm(np.asarray(target, dtype=float) - request["target"]))
        )
        root_translation_error = float(
            np.linalg.norm(np.asarray(state["pelvis"]["pos"], dtype=float)[:2] - request["pelvis_position"][:2])
        )
        current_root = Primitives._rot(state["pelvis"]["quat"])
        current_yaw = np.arctan2(current_root[1, 0], current_root[0, 0])
        request_yaw = np.arctan2(request["root_rotation"][1, 0], request["root_rotation"][0, 0])
        yaw_error = np.arctan2(np.sin(current_yaw - request_yaw), np.cos(current_yaw - request_yaw))
        joint_delta = np.abs(np.asarray(state["joints"], dtype=float) - request["joints"])
        joint_index = int(np.argmax(joint_delta))
        joint_error = float(joint_delta[joint_index])
        yaw_tolerance = float(np.radians(self.cfg.stance_plan_yaw_tolerance_deg))
        joint_tolerance = float(self.cfg.stance_plan_joint_tolerance_rad)
        matches = bool(
            target_error is not None
            and target_error <= tolerance
            and root_translation_error <= tolerance
            and abs(yaw_error) <= yaw_tolerance
            and joint_error <= joint_tolerance
        )
        return {
            "matches": matches,
            "request_key": request.get("key"),
            "request_seq": request.get("state", {}).get("seq"),
            "request_physical_time": request.get("state", {}).get("t"),
            "physical_time": state.get("t"),
            "target_error_m": target_error,
            "target_tolerance_m": tolerance,
            "root_translation_error_m": root_translation_error,
            "root_translation_tolerance_m": tolerance,
            "root_yaw_error_rad": float(yaw_error),
            "root_yaw_tolerance_rad": yaw_tolerance,
            "max_joint_error_rad": joint_error,
            "max_joint_error_index": joint_index,
            "joint_tolerance_rad": joint_tolerance,
        }

    def _settled_support_matches(self, request, state, target):
        return self._settled_support_diagnostics(request, state, target)["matches"]

    def _record_settled_invalidation(self, reason, request, state, target, reference_source):
        diagnostic = self._settled_support_diagnostics(request, state, target)
        diagnostic.update(
            reason=reason,
            request_reference_source=request.get("reference_source"),
            current_reference_source=reference_source,
        )
        self.stats.setdefault("settled_stance_invalidations", []).append(diagnostic)

    def _settled_stance_tick(self, state, *, support_quiet=True, allow_submission=True):
        if getattr(self, "stance_planner", None) is None:
            return {"status": "legacy"}

        if not support_quiet:
            self._settled_preparation_samples = []
            diagnostics = {
                "status": "preparing",
                "ready": False,
                "reason": "support_moving",
                "sample_count": 0,
                "held_seconds": 0.0,
            }
            self.stats["settled_stance_plan"] = diagnostics
            return diagnostics
        (
            reference_joints,
            reference_root,
            preparation_speed,
            reference_source,
        ) = self._settled_owned_reference(state)
        preparation = self._settled_preparation_status(state, preparation_speed, reference_source)
        if not preparation["ready"]:
            diagnostics = {
                "status": "preparing",
                "reference_source": reference_source,
                **preparation,
            }
            self.stats["settled_stance_plan"] = diagnostics
            return diagnostics
        if not allow_submission:
            diagnostics = {
                "status": "preparing",
                "reason": "support_quiet_dwell",
                "reference_source": reference_source,
                **preparation,
            }
            self.stats["settled_stance_plan"] = diagnostics
            return diagnostics
        target = self.target.observe(state)
        if target is None:
            if self._settled_request is not None:
                self._record_settled_invalidation(
                    "target_unavailable",
                    self._settled_request,
                    state,
                    None,
                    reference_source,
                )
            return {"status": "unavailable", "reason": "settled_target_unavailable"}
        target = np.asarray(target, dtype=float)
        if self._settled_request is not None and self._settled_request["reference_source"] != reference_source:
            self._record_settled_invalidation(
                "reference_source_changed",
                self._settled_request,
                state,
                target,
                reference_source,
            )
            self._settled_request = None
            self._settled_plan = None
            return {"status": "stale", "reason": "settled_reference_source_changed"}
        if self._settled_request is not None and not self._settled_support_matches(
            self._settled_request, state, target
        ):
            self._record_settled_invalidation(
                "support_or_target_changed",
                self._settled_request,
                state,
                target,
                reference_source,
            )
            self._settled_request = None
            self._settled_plan = None
            return {"status": "stale", "reason": "settled_support_or_target_changed"}

        if self._settled_request is None:
            self._settled_request_count += 1
            key = self._settled_request_count
            root = Primitives._rot(state["pelvis"]["quat"])
            desired_rotation = Rotation.from_euler("z", np.arctan2(root[1, 0], root[0, 0])).as_matrix()
            request = {
                "key": key,
                "target": target.copy(),
                "pelvis_position": np.asarray(state["pelvis"]["pos"], dtype=float).copy(),
                "root_rotation": root.copy(),
                "grasp_rotation": desired_rotation.copy(),
                "joints": np.asarray(state["joints"], dtype=float).copy(),
                "reference_joints": reference_joints.copy(),
                "reference_root": reference_root.copy(),
                "reference_source": reference_source,
                "reference_measurement_offset_linf_rad": float(
                    np.max(np.abs(reference_joints - np.asarray(state["joints"], dtype=float)))
                ),
                "state": deepcopy(state),
                "submitted_wall": time.monotonic(),
            }
            response = self.stance_planner.request(
                "settled",
                key,
                {
                    "state": state,
                    "target": target,
                    "desired_rotation": desired_rotation,
                    "posture": request["joints"],
                    "reference_joints": request["reference_joints"],
                    "reference_root": request["reference_root"],
                    "verify_path": False,
                },
                priority=True,
            )
            self._settled_request = request
        else:
            response = self.stance_planner.poll("settled", self._settled_request["key"])

        status = response["status"]
        diagnostics = {
            "status": status,
            "request_count": self._settled_request_count,
            "poll_seconds": float(response.get("poll_seconds", 0.0)),
            "age_seconds": time.monotonic() - self._settled_request["submitted_wall"],
            "request": {
                "key": self._settled_request["key"],
                "seq": self._settled_request["state"].get("seq"),
                "t": self._settled_request["state"].get("t"),
                "state": self._settled_request["state"],
                "target": self._settled_request["target"],
                "reference_joints": self._settled_request["reference_joints"],
                "reference_root": self._settled_request["reference_root"],
                "reference_source": self._settled_request["reference_source"],
                "reference_measurement_offset_linf_rad": self._settled_request["reference_measurement_offset_linf_rad"],
                "grasp_rotation": self._settled_request["grasp_rotation"],
            },
        }
        if status == "ready":
            result = response["result"]
            diagnostics["result"] = result
            completed_age = time.monotonic() - float(result["completed_wall"])
            worker_seconds = float(result.get("batch_solve_seconds", result.get("solve_seconds", 0.0)))
            diagnostics.update(
                worker_solve_seconds=worker_seconds,
                queue_and_poll_seconds=max(0.0, diagnostics["age_seconds"] - worker_seconds),
                completed_age_seconds=completed_age,
            )
            if completed_age > self.cfg.stance_plan_freshness_seconds:
                diagnostics.update(status="stale", reason="settled_plan_expired")
                self._settled_request = None
            elif result.get("feasible", False):
                self._settled_plan = result
                diagnostics["status"] = "ready"
            else:
                diagnostics.update(
                    status="infeasible",
                    reason=result.get("reason", "settled_whole_body_infeasible"),
                )
        elif status == "missing":
            diagnostics.update(status="unavailable", reason="settled_plan_missing")
        self.stats["settled_stance_plan"] = diagnostics
        return diagnostics

    def _brake(self):
        self._settled_request = getattr(self, "_settled_request", None)
        state = self.p.obs()
        start, quiet, last_progress = float(state["t"]), None, time.monotonic()
        previous = start
        while time.monotonic() - last_progress < 2:
            stamp = float(state["t"])
            self._hold_planner(state)
            self._observe_approach_settlement(state)
            if state["fallen"] or (stamp - start > 3 and getattr(self, "_settled_wait_started", None) is None):
                self._observe_approach_settlement(
                    state,
                    complete=True,
                    completion_reason="fallen" if state["fallen"] else "brake physical timeout",
                )
                return False
            if not self._preparation_hold_tick(state):
                final_state = self.p.obs()
                self._observe_approach_settlement(
                    final_state,
                    complete=True,
                    completion_reason="brake planner failure",
                )
                return False
            # Planning may occupy several sensor periods. Settle and diagnostics
            # use a new physical observation taken after that work finishes.
            state = self.p.obs()
            stamp = float(state["t"])
            self._observe_approach_settlement(state)
            if state["fallen"] or (stamp - start > 3 and getattr(self, "_settled_wait_started", None) is None):
                self._observe_approach_settlement(
                    state,
                    complete=True,
                    completion_reason="fallen" if state["fallen"] else "brake physical timeout",
                )
                return False
            if stamp > previous:
                previous, last_progress = stamp, time.monotonic()
            support_quiet = np.linalg.norm(state["pelvis"]["vel"][:2]) < 0.06
            quiet = (stamp if quiet is None else quiet) if support_quiet else None
            support_admitted = quiet is not None and stamp - quiet >= 0.25
            settled = self._settled_stance_tick(
                state,
                support_quiet=support_quiet,
                allow_submission=support_admitted,
            )
            wait_started = getattr(self, "_settled_wait_started", None)
            if wait_started is None and self._settled_request is not None:
                self._settled_wait_started = self._settled_request["submitted_wall"]
                wait_started = self._settled_wait_started
            if wait_started is not None and time.monotonic() - wait_started > self.cfg.stance_settled_timeout:
                self.error = "whole-body settled stance planning timed out"
                self._observe_approach_settlement(
                    state,
                    complete=True,
                    completion_reason=self.error,
                )
                return False
            if support_admitted:
                if settled["status"] == "preparing":
                    time.sleep(0.01)
                    continue
                if settled["status"] in ("ready", "legacy"):
                    self._observe_approach_settlement(
                        state,
                        complete=True,
                        settled=True,
                        completion_reason=(
                            "brake quiet" if settled["status"] == "legacy" else "brake quiet and stance solved"
                        ),
                    )
                    return True
                if settled["status"] in ("infeasible", "unavailable", "closed"):
                    self.error = f"whole-body {settled.get('reason', 'settled stance unavailable')}"
                    self._observe_approach_settlement(
                        state,
                        complete=True,
                        completion_reason=self.error,
                    )
                    return False
                if settled["status"] == "stale":
                    quiet = stamp
            time.sleep(0.01)
        self._observe_approach_settlement(
            state,
            complete=True,
            completion_reason="brake physical clock stalled",
        )
        return False

    def settle_for_admission(self):
        """Finish the supported brake once before stand-state admission checks."""
        if getattr(self, "_base_settled", False):
            return True
        self.execution.enter("whole_body.brake")
        self._base_settled = self._brake()
        return self._base_settled

    def _measured_tools(self, state):
        return np.array(
            [
                np.asarray(state["palms"][side]["pos"]) + Primitives._rot(state["palms"][side]["quat"]) @ OFFSET
                for side in ("left", "right")
            ]
        )

    def _clear(self, state, candidate, contact, candidate_base=None, *, attached=False):
        """Predicted-collision gate. `contact`: the right fingers may touch the
        target (finger_only_target_contact). `attached`: the target is in the
        right hand, so it is no obstacle to that arm at all; table and body
        clearance still apply."""
        for side in ("left", "right", "body"):
            hit = self.guard.check(
                state,
                candidate,
                side,
                allow_target_contact=contact and side == "right",
                candidate_base=candidate_base,
                ignore_target=attached and side == "right",
            )
            if not hit.clear:
                return f"whole-body clearance: {side}:{hit.link}:{hit.obstacle}"
        return None

    def _loaded_endpoint_request(self, state, motion, targets, rotations, desired_rotation):
        reference_tools = motion.body.tools(motion.x).copy()
        loaded_start = reference_tools[1] - self.cfg.tracking_gain * motion.feedback[1]
        commanded_rise = loaded_lift_height(self.cfg.lift_height, self.cfg.stance_ik_tolerance)
        endpoint, path = loaded_world_path(
            loaded_start,
            desired_rotation[:, 0],
            commanded_rise,
            1.0,
            carry_distance=self.cfg.lift_height,
        )
        standing_reference = self.walking.comfortable.copy()
        standing_weights = np.full(29, 0.04)
        standing_weights[:12] = 0.20
        standing_weights[12:15] = self.cfg.comfort_waist_weight
        standing_weights[22:29] = self.cfg.comfort_elbow_wrist_weight
        source = {name: state.get(name) for name in ("seq", "t", "wall")}
        palm_relative = object_in_palm_frame(
            state,
            "right",
            state["objects"][self.task.target.key]["pos"],
        )
        contract = {
            "targets": np.asarray(targets, dtype=float).copy(),
            "rotations": np.asarray(rotations, dtype=float).copy(),
            "loaded_start": loaded_start.copy(),
            "root_position": np.asarray(state["pelvis"]["pos"], dtype=float).copy(),
            "root_rotation": Primitives._rot(state["pelvis"]["quat"]),
            "measured_foot_positions": motion.measured_foot_positions(state),
            "palm_relative": palm_relative.copy(),
            "source": source,
        }
        payload = {
            "state": state,
            "targets": np.asarray(targets, dtype=float).copy(),
            "rotations": np.asarray(rotations, dtype=float).copy(),
            "endpoint": endpoint,
            "reference_x": motion.x.copy(),
            "reference_root": motion.root.copy(),
            "standing_reference": standing_reference,
            "standing_weights": standing_weights,
            "support_problem": loaded_support_problem_snapshot(motion),
            "source": source,
        }
        return {
            "payload": payload,
            "contract": contract,
            "path": path,
            "loaded_start": loaded_start,
            "endpoint": endpoint,
            "commanded_rise_m": commanded_rise,
            "tracking_margin_m": self.cfg.stance_ik_tolerance,
            "standing_reference": standing_reference,
            "standing_weights": standing_weights,
        }

    def _loaded_endpoint_contract_difference(self, request, state, motion, targets, rotations):
        contract = request["contract"]
        current_root = Primitives._rot(state["pelvis"]["quat"])
        root_yaw_error = float(
            abs(
                np.arctan2(
                    np.sin(
                        np.arctan2(current_root[1, 0], current_root[0, 0])
                        - np.arctan2(contract["root_rotation"][1, 0], contract["root_rotation"][0, 0])
                    ),
                    np.cos(
                        np.arctan2(current_root[1, 0], current_root[0, 0])
                        - np.arctan2(contract["root_rotation"][1, 0], contract["root_rotation"][0, 0])
                    ),
                )
            )
        )
        palm_relative = object_in_palm_frame(
            state,
            "right",
            state["objects"][self.task.target.key]["pos"],
        )
        current_reference_tools = motion.body.tools(motion.x).copy()
        current_loaded_start = current_reference_tools[1] - self.cfg.tracking_gain * motion.feedback[1]
        target_error = float(np.max(np.linalg.norm(np.asarray(targets, dtype=float) - contract["targets"], axis=1)))
        rotation_error = float(
            np.max(
                Rotation.from_matrix(
                    np.asarray(rotations, dtype=float) @ contract["rotations"].transpose(0, 2, 1)
                ).magnitude()
            )
        )
        root_xy_error = float(
            np.linalg.norm(np.asarray(state["pelvis"]["pos"], dtype=float)[:2] - contract["root_position"][:2])
        )
        palm_relative_error = float(np.linalg.norm(palm_relative - contract["palm_relative"]))
        loaded_start_error = float(np.linalg.norm(current_loaded_start - contract["loaded_start"]))
        current_foot_positions = motion.measured_foot_positions(state)
        request_foot_motion = np.linalg.norm(current_foot_positions - contract["measured_foot_positions"], axis=1)
        cumulative_foot_drift = np.asarray(motion.measured_foot_drift(state), dtype=float)
        max_request_foot_motion = float(np.max(request_foot_motion))
        max_cumulative_foot_drift = float(np.max(cumulative_foot_drift))
        bilateral = bilateral_dex1_contact(state["objects"][self.task.target.key], "right")
        checks = {
            "target": target_error <= self.cfg.stance_plan_tolerance_m,
            "rotation_target": rotation_error <= np.radians(self.cfg.stance_plan_yaw_tolerance_deg),
            "root_xy": root_xy_error <= self.cfg.stance_plan_tolerance_m,
            "root_yaw": root_yaw_error <= np.radians(self.cfg.stance_plan_yaw_tolerance_deg),
            "closed_grip": bilateral and palm_relative_error <= self.cfg.stance_plan_tolerance_m,
            "loaded_start": loaded_start_error <= self.cfg.stance_plan_tolerance_m,
            "fixed_feet": max_request_foot_motion <= self.cfg.stance_plan_tolerance_m,
        }
        reason = next((name for name, passed in checks.items() if not passed), None)
        return {
            "matches": reason is None,
            "reason": reason,
            "checks": checks,
            "target_error_m": target_error,
            "target_tolerance_m": self.cfg.stance_plan_tolerance_m,
            "rotation_target_error_rad": rotation_error,
            "rotation_target_tolerance_rad": np.radians(self.cfg.stance_plan_yaw_tolerance_deg),
            "root_xy_error_m": root_xy_error,
            "root_xy_tolerance_m": self.cfg.stance_plan_tolerance_m,
            "root_yaw_error_rad": root_yaw_error,
            "root_yaw_tolerance_rad": np.radians(self.cfg.stance_plan_yaw_tolerance_deg),
            "palm_relative_error_m": palm_relative_error,
            "palm_relative_tolerance_m": self.cfg.stance_plan_tolerance_m,
            "loaded_start_error_m": loaded_start_error,
            "loaded_start_tolerance_m": self.cfg.stance_plan_tolerance_m,
            "request_foot_motion_m": request_foot_motion,
            "max_request_foot_motion_m": max_request_foot_motion,
            "request_foot_motion_tolerance_m": self.cfg.stance_plan_tolerance_m,
            "cumulative_foot_drift_m": cumulative_foot_drift,
            "max_cumulative_foot_drift_m": max_cumulative_foot_drift,
            "request_source": contract["source"],
            "current_source": {name: state.get(name) for name in ("seq", "t", "wall")},
        }

    def _release_streamed(self):
        """Release POSE ownership once, including when normal handback setup fails."""
        if self.arm is None or getattr(self, "_stream_released", False):
            return
        try:
            self.arm.stop(preserve_reference=True)
        except Exception:
            if not getattr(self.arm, "_stream_released", False):
                try:
                    self.p.cmd.exit_streamed()
                finally:
                    self.arm._stream_released = True
            raise
        finally:
            self._stream_released = True

    def execute(self, *, grip_world, squat_hint, ghost):
        ok = False
        try:
            if not self.settle_for_admission():
                return False, self.error or "whole-body base did not stabilize"
            ok, why = self._reach()
            if ok:
                ok, why = self._brake_handoff()
            self.completed = ok
            self.stats["outcome"] = dict(ok=ok, reason=why, state=self.p.obs())
            return ok, why
        finally:
            if self.arm is not None and not getattr(self, "_stream_released", False):
                # Success returns to walking only after measured upright recovery.
                # Failure first pins a bounded stopped reference (never rise
                # through an obstructed outstretched hand as a cleanup), then
                # hands that same pose back to the planner. Leaving the deploy
                # in streamed mode across the round boundary made the next
                # reset land 0.10-0.12 m off and repeat to the end of the batch:
                # 8/8 stand-stage failures on 2026-09-14 followed a whole-body
                # abort, four of them in one unbroken run.
                if not ok:
                    try:
                        self.arm._push(self.arm.q, reference_velocity=np.zeros(29))
                    except (RuntimeError, ValueError) as exc:
                        self.stats["cleanup_brake_error"] = str(exc)
                try:
                    self._release_streamed()
                except (RuntimeError, ValueError) as exc:
                    self.stats["cleanup_release_error"] = str(exc)
            # Serialization and camera teardown belong to round cleanup. Doing
            # them while carrying can starve planner commands past its timeout.
            self.p.cmd.reference_observer = self.previous_observer

    def _loaded_stability_gate(self):
        return LoadedHandbackStabilityGate(
            quiet_seconds=self.cfg.capture_terminal_window_seconds,
            min_samples=self.cfg.capture_min_samples,
            max_gap_seconds=self.cfg.capture_terminal_max_gap_seconds,
            hand_position_diameter_max=self.cfg.capture_position_diameter_max,
            hand_speed_max=self.cfg.capture_drift_speed_max,
            hand_rotation_diameter_max=self.cfg.capture_rotation_diameter_max,
            hand_angular_speed_max=self.cfg.capture_angular_drift_max,
            object_relative_speed_max=self.cfg.contact_speed_max,
        )

    def _loaded_stability_step(self, gate, state):
        obj = state["objects"][self.task.target.key]
        palm = state["palms"]["right"]
        return gate.step(
            float(state["t"]),
            np.asarray(palm["pos"], dtype=float),
            Primitives._rot(palm["quat"]),
            np.asarray(state["pelvis"]["pos"], dtype=float),
            Primitives._rot(state["pelvis"]["quat"]),
            np.asarray(state["joints"], dtype=float)[:12],
            object_in_palm_frame(state, "right", obj["pos"]),
            bilateral_dex1_contact(obj, "right"),
        )

    def _brake_handoff(self):
        """Brake, measure loaded stability, switch once, and verify the idle tail."""
        state = self.p.obs()
        try:
            prediction = self.arm.stopping_reference()
        except (RuntimeError, ValueError) as exc:
            self.stats["handback"] = dict(
                admitted=False,
                completed=False,
                phase="prediction_failed",
                reason=str(exc),
            )
            return False, f"whole-body handback prediction failed: {exc}"

        reference_q = np.asarray(self.arm.q, dtype=float)
        measured_q = np.asarray(state["joints"], dtype=float)
        predicted_q = measured_q + prediction["joints"] - reference_q
        reference_root = Primitives._rot(self.arm.root_quat)
        stopping_root = Primitives._rot(prediction["root"])
        delta_rotation = stopping_root @ reference_root.T
        measured_root = Primitives._rot(state["pelvis"]["quat"])
        predicted_quat = Rotation.from_matrix(delta_rotation @ measured_root).as_quat()[[3, 0, 1, 2]]
        candidate_base = [*state["pelvis"]["pos"], *predicted_quat]
        reason = self._clear(
            state,
            predicted_q,
            True,
            candidate_base=candidate_base,
            attached=True,
        )
        handback = dict(
            admitted=reason is None,
            completed=False,
            phase="endpoint_admission",
            reason=reason,
            prediction=prediction,
            packet_start=dict(
                joints=reference_q.copy(),
                velocity=np.asarray(self.arm.reference_velocity, dtype=float).copy(),
                root=np.asarray(self.arm.root_quat, dtype=float).copy(),
                root_angular_velocity=np.asarray(self.arm.root_angular_velocity, dtype=float).copy(),
            ),
            actual_start=dict(
                joints=measured_q.copy(),
                q_tracking_error=float(np.max(np.abs(measured_q - reference_q))),
            ),
        )
        self.stats["handback"] = handback
        if reason is not None:
            return False, f"whole-body handback {reason}"

        self.arm.hands["right"] = HAND_CLOSED_RIGHT.copy()
        try:
            braking = self.arm.brake_to_stop(prediction)
        except (RuntimeError, ValueError) as exc:
            handback.update(phase="braking_failed", reason=str(exc))
            return False, f"whole-body handback braking failed: {exc}"

        gate = self._loaded_stability_gate()
        handback.update(braking=braking, phase="pre_switch_stability")
        started_wall = last_progress_wall = time.monotonic()
        previous_stamp = None
        last_hold_stamp = None
        stable_state = None
        while True:
            state = self.p.obs()
            stamp = float(state["t"])
            now = time.monotonic()
            if state.get("fallen", False):
                handback.update(phase="pre_switch_failed", reason="fallen")
                return False, "whole-body handback fell before ownership switch"
            if previous_stamp is None or stamp > previous_stamp:
                last_progress_wall = now
            previous_stamp = stamp
            if now - last_progress_wall > 2:
                handback.update(phase="pre_switch_timeout", reason="physical_clock_stalled")
                return False, "whole-body handback physical clock stalled"
            if now - started_wall > self.cfg.stance_settled_timeout:
                handback.update(phase="pre_switch_timeout", reason="stability_timeout")
                return False, "whole-body handback stability timed out"
            if last_hold_stamp is None or stamp > last_hold_stamp:
                self.arm._push(
                    self.arm.q,
                    reference_velocity=np.zeros(29),
                    reference_time=stamp,
                    root_angular_velocity=np.zeros(3),
                )
                last_hold_stamp = stamp
            try:
                stability = self._loaded_stability_step(gate, state)
            except (KeyError, ValueError) as exc:
                handback.update(phase="pre_switch_failed", reason=str(exc))
                return False, f"whole-body handback stability measurement failed: {exc}"
            handback["pre_switch"] = stability
            if stability["ready"]:
                stable_state = state
                break
            time.sleep(0.005)

        switch_stamp = float(stable_state["t"])
        release_requested_wall = time.time()
        try:
            self._release_streamed()
        except (RuntimeError, ValueError) as exc:
            handback.update(phase="switch_failed", reason=str(exc))
            return False, f"whole-body handback ownership switch failed: {exc}"
        release_returned_wall = time.time()
        handback.update(
            phase="post_switch_stability",
            switch=dict(
                seq=stable_state.get("seq"),
                t=switch_stamp,
                state_wall=stable_state.get("wall"),
                sample_count=stability["sample_count"],
                release_requested_wall=release_requested_wall,
                release_returned_wall=release_returned_wall,
                requested_mode="planner_frozen_upper_body_idle",
                observation_scope="release_request_only; no controller consumption acknowledgement",
            ),
        )

        # The release call and the wait for a fresh state put the first
        # post-switch sample beyond the regular gap limit (measured 0.060 s
        # against 0.05 s in stage3_2_aligned8, with the state stream itself
        # continuous at 0.02 s). Keep the pre-switch samples so motion caused
        # by the switch is still measured, but allow that one boundary gap up
        # to the quiet window itself.
        gate.expect_boundary(self.cfg.capture_terminal_window_seconds)
        post_started_wall = last_progress_wall = time.monotonic()
        previous_stamp = switch_stamp
        first_post_switch = None
        while True:
            state = self.p.obs()
            stamp = float(state["t"])
            now = time.monotonic()
            if first_post_switch is None:
                first_post_switch = dict(
                    seq=state.get("seq"),
                    t=stamp,
                    state_wall=state.get("wall"),
                    gap_from_switch_sample_s=float(stamp - switch_stamp),
                    boundary_tolerance_s=float(self.cfg.capture_terminal_window_seconds),
                )
                handback["switch"]["first_post_switch"] = first_post_switch
            if state.get("fallen", False):
                handback.update(phase="post_switch_failed", reason="fallen")
                return False, "whole-body handback fell after ownership switch"
            if stamp > previous_stamp:
                last_progress_wall = now
            previous_stamp = stamp
            if now - last_progress_wall > 2:
                handback.update(phase="post_switch_timeout", reason="physical_clock_stalled")
                return False, "whole-body handback post-switch physical clock stalled"
            if now - post_started_wall > self.cfg.stance_settled_timeout:
                handback.update(phase="post_switch_timeout", reason="stability_timeout")
                return False, "whole-body handback post-switch stability timed out"
            try:
                stability = self._loaded_stability_step(gate, state)
            except (KeyError, ValueError) as exc:
                handback.update(phase="post_switch_failed", reason=str(exc))
                return False, f"whole-body handback post-switch measurement failed: {exc}"
            handback["post_switch"] = stability
            if not stability["ready"]:
                reason = stability.get("reset_reason") or "measured_motion"
                handback.update(phase="post_switch_failed", reason=reason)
                return False, f"whole-body handback post-switch stability lost: {reason}"
            if stamp - switch_stamp >= self.cfg.capture_terminal_window_seconds:
                handback.update(
                    completed=True,
                    phase="complete",
                    reason=None,
                    actual_final=dict(
                        joints=np.asarray(state["joints"], dtype=float).copy(),
                        q_tracking_error=float(
                            np.max(
                                np.abs(np.asarray(state["joints"], dtype=float) - np.asarray(self.arm.q, dtype=float))
                            )
                        ),
                    ),
                    post_switch_window_end=dict(
                        seq=state.get("seq"),
                        t=stamp,
                        state_wall=state.get("wall"),
                        elapsed_physical_seconds=float(stamp - switch_stamp),
                    ),
                )
                return True, "continuous whole-body grasp upright recovery with stable handback"
            time.sleep(0.005)

    def prepare_carry(self):
        """The same reference already recovered upright with the bottle secured."""
        return self.completed

    def _admit_takeover_stop(self, state):
        try:
            prediction = self.arm.stopping_reference()
        except (RuntimeError, ValueError) as exc:
            return None, f"whole-body POSE takeover stop prediction failed: {exc}"
        reference_q = np.asarray(self.arm.q, dtype=float)
        measured_q = np.asarray(state["joints"], dtype=float)
        predicted_q = measured_q + np.asarray(prediction["joints"], dtype=float) - reference_q
        reference_root = Primitives._rot(self.arm.root_quat)
        stopping_root = Primitives._rot(prediction["root"])
        delta_rotation = stopping_root @ reference_root.T
        measured_root = Primitives._rot(state["pelvis"]["quat"])
        predicted_quat = Rotation.from_matrix(delta_rotation @ measured_root).as_quat()[[3, 0, 1, 2]]
        reason = self._clear(
            state,
            predicted_q,
            False,
            candidate_base=[*state["pelvis"]["pos"], *predicted_quat],
        )
        if reason:
            return None, f"whole-body POSE takeover stop {reason}"
        self.stats["post_takeover_brake"] = {
            "admitted": True,
            "prediction": prediction,
            "inherited_joint_speed_rad_s": float(np.max(np.abs(np.asarray(self.arm.reference_velocity, dtype=float)))),
            "inherited_root_angular_speed_rad_s": float(
                np.linalg.norm(np.asarray(self.arm.root_angular_velocity, dtype=float))
            ),
        }
        return prediction, None

    def _post_takeover_stance(self):
        """Require a fresh quiet support solve after POSE takes ownership."""
        if self.stance_planner is None:
            return self.p.obs(), None
        self._settled_request = None
        self._settled_plan = None
        state = self.p.obs()
        prediction, prediction_error = self._admit_takeover_stop(state)
        if prediction_error is not None:
            return None, prediction_error
        started = time.monotonic()
        quiet = None
        previous = None
        last_progress = started
        max_tick_seconds = 0.0
        reference_braked = False
        first_plan_submitted = None

        def timing_fields(now, status, stamp=None):
            result = status.get("result", {})
            worker_seconds = result.get("batch_solve_seconds", result.get("solve_seconds"))
            plan_seconds = 0.0 if first_plan_submitted is None else max(0.0, now - first_plan_submitted)
            return {
                "preparation_wall_seconds": max(
                    0.0,
                    (first_plan_submitted if first_plan_submitted is not None else now) - started,
                ),
                "support_quiet_physical_seconds": (
                    0.0 if quiet is None or stamp is None else max(0.0, float(stamp) - quiet)
                ),
                "planner_elapsed_seconds": plan_seconds,
                "planner_worker_seconds": worker_seconds,
                "planner_waiting_seconds": (
                    None if worker_seconds is None else max(0.0, plan_seconds - float(worker_seconds))
                ),
                "total_wall_seconds": max(0.0, now - started),
            }

        while True:
            now = time.monotonic()
            if first_plan_submitted is None:
                if now - started > self.cfg.stance_settled_timeout:
                    self.stats["post_takeover_stance"] = {
                        "status": "preparation_timeout",
                        **timing_fields(now, {"status": "preparation_timeout"}, previous),
                    }
                    return None, "whole-body post-takeover stance preparation timed out"
            elif now - first_plan_submitted > self.cfg.stance_settled_timeout:
                self.stats["post_takeover_stance"] = {
                    "status": "planning_timeout",
                    **timing_fields(now, {"status": "planning_timeout"}, previous),
                }
                return None, "whole-body post-takeover stance planning timed out"
            tick_started = time.perf_counter()
            state = self.p.obs()
            stamp = float(state["t"])
            if state["fallen"]:
                return None, "whole-body fell during POSE takeover settle"
            if previous is not None and stamp < previous:
                return None, "whole-body physical state clock moved backwards during takeover"
            dt = 0.0 if previous is None else stamp - previous
            if dt > 0.5:
                return None, "whole-body physical state clock gap during takeover settle"
            if previous is None or stamp > previous:
                previous, last_progress = stamp, time.monotonic()
            elif time.monotonic() - last_progress > 2:
                return None, "whole-body physical clock stalled during takeover settle"

            self._hold_planner(state, stream_mode=StreamMode.POSE)
            if not reference_braked and dt > 0:
                (
                    self.arm.q,
                    self.arm.root_quat,
                    self.arm.reference_velocity,
                    self.arm.root_angular_velocity,
                    reference_braked,
                ) = bounded_takeover_brake_step(
                    self.arm.q,
                    self.arm.root_quat,
                    self.arm.reference_velocity,
                    self.arm.root_angular_velocity,
                    dt,
                    self.arm.preview.acceleration,
                    (self.arm.preview.lo, self.arm.preview.hi),
                )
            self.arm._push(
                self.arm.q,
                reference_velocity=self.arm.reference_velocity,
                reference_time=stamp,
                root_angular_velocity=self.arm.root_angular_velocity,
            )
            state = self.p.obs()
            stamp = float(state["t"])
            speed = float(np.linalg.norm(np.asarray(state["pelvis"]["vel"], dtype=float)[:2]))
            if reference_braked and speed < 0.06:
                quiet = stamp if quiet is None else quiet
            else:
                quiet = None

            status = {"status": "moving"}
            if quiet is not None and stamp - quiet >= 0.25:
                status = self._settled_stance_tick(state)
                if first_plan_submitted is None and self._settled_request is not None:
                    first_plan_submitted = self._settled_request["submitted_wall"]
                if status["status"] == "ready":
                    max_tick_seconds = max(max_tick_seconds, time.perf_counter() - tick_started)
                    self.stats["post_takeover_stance"] = {
                        "status": "ready",
                        "max_tick_seconds": max_tick_seconds,
                        "measured_xy_speed_m_s": speed,
                        "plan": status,
                        **timing_fields(time.monotonic(), status, stamp),
                    }
                    return state, None
                if status["status"] in ("infeasible", "unavailable", "closed"):
                    self.stats["post_takeover_stance"] = {
                        "status": status["status"],
                        "max_tick_seconds": max_tick_seconds,
                        "measured_xy_speed_m_s": speed,
                        "plan": status,
                        **timing_fields(time.monotonic(), status, stamp),
                    }
                    return None, (f"whole-body {status.get('reason', 'post-takeover stance unavailable')}")
                if status["status"] == "stale":
                    quiet = stamp

            max_tick_seconds = max(max_tick_seconds, time.perf_counter() - tick_started)
            self.stats["post_takeover_stance"] = {
                "status": status["status"],
                "max_tick_seconds": max_tick_seconds,
                "measured_xy_speed_m_s": speed,
                **timing_fields(time.monotonic(), status, stamp),
            }
            time.sleep(0.005)

    def _verified_grasp_rotation(self, state):
        if self._settled_plan is not None:
            return (
                np.asarray(self._settled_plan["grasp_rotation"], dtype=float).copy(),
                "settled_stance_plan",
            )
        if self.walking.rotation is not None:
            return np.asarray(self.walking.rotation, dtype=float).copy(), "walking_preparation"
        root = Primitives._rot(state["pelvis"]["quat"])
        return (
            Rotation.from_euler("z", np.arctan2(root[1, 0], root[0, 0])).as_matrix(),
            "measured_root_yaw",
        )

    def _settled_control_targets(self):
        if self._settled_plan is None:
            return None, None
        posture = np.asarray(
            self._settled_plan.get("control_posture_joints", self._settled_plan["joints"]),
            dtype=float,
        ).copy()
        torso = np.asarray(
            self._settled_plan.get("control_torso_rotation", self._settled_plan["torso_rotation"]),
            dtype=float,
        ).copy()
        return posture, torso

    def _reach(self):
        state = self.p.obs()
        target = self.target.observe(state)
        if target is None:
            return False, "whole-body target not visible in student cameras"
        if self.stance_planner is not None and not (
            self._settled_plan is not None and self._settled_plan.get("feasible", False)
        ):
            return False, "whole-body settled stance plan unavailable"
        self.arm = StreamedArm(self.p, "right", ik=self.ik, left_guard=False)
        physical = self.ik.model.jnt_range[self.ik.jids]
        self.ik.lo, self.ik.hi = physical[:, 0].copy(), physical[:, 1].copy()
        self.ik.hi[14] = min(self.ik.hi[14], self.cfg.waist_pitch_max)
        self.arm.preview = PosePreview(
            acceleration=self.cfg.joint_acceleration,
            joint_limits=(self.ik.lo, self.ik.hi),
            continuous_handoff=True,
        )
        self.filter = ReferenceFilter(self.cfg.pose_lowpass_hz) if self.cfg.pose_lowpass_hz > 0 else None
        self.context["phase"] = "handoff"
        self._stream_released = False
        self.arm.start(hold_planner_pose=True, inherit_planner_velocity=True)
        state, takeover_error = self._post_takeover_stance()
        if takeover_error is not None:
            return False, takeover_error
        target = self.target.observe(state)
        if target is None:
            return False, "whole-body target stale after POSE takeover"
        settled_control_posture, planned_torso_rotation = self._settled_control_targets()
        comfort_reference = comfort_weights = None
        if settled_control_posture is not None:
            comfort_reference = settled_control_posture
            comfort_weights = np.full(29, 0.04)
            comfort_weights[12:15] = self.cfg.comfort_waist_weight
            comfort_weights[25:29] = self.cfg.comfort_elbow_wrist_weight
        elif self.cfg.comfort_posture:
            comfort_reference = np.asarray(self.arm.q, dtype=float).copy()
            comfort_reference[12:14] = 0.0
            comfort_reference[25:29] = self.walking.comfortable[25:29]
            comfort_weights = np.full(29, 0.04)
            comfort_weights[12:14] = self.cfg.comfort_waist_weight
            comfort_weights[25:29] = self.cfg.comfort_elbow_wrist_weight
        self.motion = SupportedMotion(
            self.ik,
            state,
            self.cfg,
            self.arm.q,
            self.arm.root_quat,
            tool_offset=OFFSET,
            hand_weights=(0.35, 1.0),
            reference_velocity=self.arm.reference_velocity,
            root_velocity=np.asarray(state["pelvis"].get("vel", [0, 0, 0]), dtype=float),
            root_angular_velocity=self.arm.root_angular_velocity,
            posture_reference=comfort_reference,
            posture_weights=comfort_weights,
            continuity_weight=self.cfg.ik_velocity_continuity_weight,
        )
        motion = self.motion
        initial = self._measured_tools(state)
        rotations = np.array([Primitives._rot(state["palms"][side]["quat"]) for side in ("left", "right")])
        desired_rot, desired_rotation_source = self._verified_grasp_rotation(state)
        turn = Slerp([0, 1], Rotation.from_matrix([rotations[1], desired_rot]))
        # Preparation already moved the hand: lean follows bottle height, not
        # the remaining hand error (which would erase the bend after good preparation).
        lean_target = (
            lean_angle(state["pelvis"]["pos"], target, self.cfg.lean_max)
            if self._settled_plan is None
            else float(np.clip(self._settled_plan["torso_pitch_rad"], 0.0, self.cfg.lean_max))
        )
        motion.body.put(motion.x)
        torso_reference = self.ik.data.xmat[motion.body.torso].reshape(3, 3).copy()
        self.stats["initial"] = dict(
            state=state,
            reference=self.arm.q,
            root=self.arm.root_quat,
            support_offset=motion.support_offset,
            handoff=self.arm.handoff_diagnostics,
            motion_initial_velocity=motion.v.copy(),
            preview=self.arm.preview.diagnostics.copy(),
            comfort_reference=comfort_reference,
            comfort_weights=comfort_weights,
            continuity_weight=self.cfg.ik_velocity_continuity_weight,
            settled_stance_plan=self._settled_plan,
            grasp_rotation=desired_rot.copy(),
            entry_axis=desired_rot[:, 0].copy(),
            grasp_rotation_source=desired_rotation_source,
            lean_source="settled_stance_solve" if self._settled_plan is not None else "legacy_geometry",
            lean_target_rad=lean_target,
        )
        phase, progress, closure = "reach", 0.0, 0.0
        load_control = ContactLoadController(
            self.cfg.contact_confirm_seconds,
            self.cfg.contact_loss_grace_seconds,
            self.cfg.contact_recovery_timeout,
            self.cfg.contact_speed_max,
        )
        load_status = None
        capture_gate = CaptureSettleGate(
            self.cfg.capture_quiet_seconds,
            self.cfg.capture_min_samples,
            self.cfg.capture_position_diameter_max,
            self.cfg.capture_drift_speed_max,
            self.cfg.capture_rotation_diameter_max,
            self.cfg.capture_angular_drift_max,
        )
        capture_status = None
        capture_admission = CaptureAdmissionConsumer()
        began = previous = float(state["t"])
        last_wall = time.monotonic()
        obj0 = np.asarray(state["objects"][self.task.target.key]["pos"]).copy()
        targets = initial.copy()
        terminal_observer = TerminalMotionObserver(
            self.cfg.capture_brake_acceleration,
            self.cfg.capture_brake_lookahead_seconds,
            window_seconds=self.cfg.capture_terminal_window_seconds,
            min_samples=self.cfg.capture_min_samples,
            max_gap_seconds=self.cfg.capture_terminal_max_gap_seconds,
        )
        terminal_retreat = TerminalRetreatCommand(
            terminal_retreat_speed_limit(
                self.cfg.hand_speed,
                self.cfg.capture_brake_acceleration,
                self.cfg.capture_terminal_window_seconds,
            ),
            self.cfg.capture_brake_acceleration,
        )
        terminal_status = None
        terminal_retreat_m = 0.0
        terminal_retreat_rate_m_s = 0.0
        capture_torso = None
        last_sample, ready_since = -1.0, None
        loaded_endpoint_request = None
        loaded_endpoint_request_count = 0
        loaded_endpoint_invalidations = []
        loaded_endpoint_consumptions = []
        loaded_endpoint_hold_sends = []
        captured_object_in_palm = None
        orientation_bad_since = None
        loaded_stability_gate = None
        loaded_stability = None
        self.execution.enter("grasp.whole_body_reach")
        while time.monotonic() - last_wall < 2:
            state = self.p.obs()
            stamp = float(state["t"])
            dt = stamp - previous
            if dt == 0:
                time.sleep(0.005)
                continue
            if dt < 0 or dt > 0.5:
                return False, "whole-body physical state clock gap"
            previous, last_wall = stamp, time.monotonic()
            if stamp - began > self.config.recovery_timeout:
                return False, f"whole-body {phase} timed out"
            obj = state["objects"][self.task.target.key]
            actual = self._measured_tools(state)
            if (
                state["fallen"]
                or np.linalg.norm(np.asarray(obj["pos"])[:2] - obj0[:2]) > 0.05
                and phase in ("reach", "close")
            ):
                return False, "whole-body disturbed target or base"
            target = self.target.observe(state)
            if target is None and phase in ("reach", "close"):
                return False, "whole-body target visibility expired"
            geometry = (
                None
                if target is None
                else closure_geometry(
                    state,
                    "right",
                    target,
                    preferred_depth_m=self.cfg.capture_depth_m,
                    depth_tolerance_m=self.cfg.capture_depth_tolerance_m,
                    lateral_tolerance_m=self.cfg.capture_lateral_tolerance_m,
                    height_tolerance_m=self.cfg.capture_height_tolerance_m,
                )
            )
            guarded_depth_m = (
                None
                if geometry is None
                else guarded_capture_depth(
                    geometry["lateral_signed_m"],
                    geometry["world_height_error_m"],
                    self.cfg.capture_depth_m,
                    self.capture_retreat_depth_m,
                    self.cfg.capture_lateral_tolerance_m,
                    self.cfg.capture_height_tolerance_m,
                )
            )
            dynamic_guarded_depth_m = None
            goal_depth_capped = None
            if geometry is not None:
                terminal_status = terminal_observer.step(
                    stamp,
                    actual[1],
                    desired_rot[:, 0],
                    geometry["depth_m"],
                    phase=phase,
                )
                stopping_displacement = terminal_status["stopping_displacement_m"]
                terminal_retreat_m, terminal_retreat_rate_m_s = terminal_retreat.step(
                    0.0 if stopping_displacement is None else max(0.0, stopping_displacement),
                    dt,
                    hold=not terminal_status["ready"],
                )
                dynamic_guarded_depth_m, _, goal_depth_capped = momentum_guarded_depth(
                    guarded_depth_m,
                    terminal_retreat_m,
                )
                geometry = predictive_capture_geometry(geometry, terminal_status)
            self.context = dict(
                seq=state["seq"],
                t=stamp,
                phase=phase,
                camera=self.target.diagnostics.copy(),
                observer_target=None if target is None else np.asarray(target).tolist(),
                capture_geometry=geometry,
                capture_alignment=(
                    None
                    if geometry is None
                    else dict(
                        lateral_error_m=geometry["lateral_signed_m"],
                        height_error_m=geometry["world_height_error_m"],
                        guarded_depth_m=guarded_depth_m,
                        retreat_m=guarded_depth_m - self.cfg.capture_depth_m,
                        terminal_observer=terminal_status,
                        goal_stopping_retreat_m=terminal_retreat_m,
                        goal_stopping_retreat_rate_m_s=terminal_retreat_rate_m_s,
                        dynamic_guarded_depth_m=dynamic_guarded_depth_m,
                        goal_depth_capped=goal_depth_capped,
                        predicted_stop_admitted=geometry["stopping"]["admitted"],
                    )
                ),
                terminal_motion=terminal_status,
            )
            torso_target = None
            if phase == "reach":
                tau = (stamp - began) / self.cfg.reach_seconds if self.cfg.reach_seconds > 0 else 1.0
                targets[1], entry_gap = advance_entry_command(
                    initial[1],
                    targets[1],
                    actual[1],
                    target,
                    desired_rot[:, 0],
                    tau,
                    dt,
                    self.cfg.hand_speed,
                    dynamic_guarded_depth_m,
                    brake_acceleration=self.cfg.capture_brake_acceleration,
                )
                progress = min(1.0, progress + min(dt, 0.1) / 1.0)
                rotations[1] = turn(smooth(progress)).as_matrix()
                torso_target = (
                    planned_torso_rotation
                    if planned_torso_rotation is not None
                    else motion.body.heading @ Rotation.from_euler("y", lean_target).as_matrix()
                )
                self.context.update(approach_part="direct", entry_gap=float(entry_gap))
                capture_status = capture_gate.step(
                    stamp,
                    geometry["ready"],
                    point_in_palm_frame(state, "right", target),
                    Primitives._rot(state["palms"]["right"]["quat"]),
                )
                self.context.update(
                    capture_settle=capture_status,
                    observer_target=np.asarray(target).tolist(),
                    capture_geometry=geometry,
                    terminal_motion=terminal_status,
                )

                def enter_close():
                    nonlocal capture_torso, rotations, targets

                    # Keep the active supported pose at the close boundary. A
                    # measurement reset here moved the wrist by the remaining
                    # capture depth just as the fingers contacted the bottle.
                    measured_rotations = np.array(
                        [Primitives._rot(state["palms"][side]["quat"]) for side in ("left", "right")]
                    )
                    previous_position_targets = targets.copy()
                    targets, previous_reference_positions = begin_live_translation_hold(motion, actual)
                    reference_rotations = self.ik.data.xmat[motion.body.hands].reshape(2, 3, 3).copy()
                    previous_rotation_targets = rotations.copy()
                    rotations = freeze_rotation_reference(
                        reference_rotations,
                        motion.rotation_feedback,
                        self.cfg.tracking_gain,
                    )
                    motion.freeze_rotation_feedback()
                    targets, rotations, transition = close_transition(
                        self.arm.q,
                        motion.joints,
                        targets,
                        rotations,
                        actual,
                        measured_rotations,
                    )
                    correction = Rotation.from_rotvec(self.cfg.tracking_gain * motion.rotation_feedback).as_matrix()
                    corrected_brake_targets = targets + self.cfg.tracking_gain * motion.feedback
                    transition["translation_brake"] = dict(
                        raw_target_change_m=np.linalg.norm(targets - previous_position_targets, axis=1),
                        raw_target_measured_error_m=np.linalg.norm(targets - actual, axis=1),
                        corrected_target_retreat_m=np.linalg.norm(
                            previous_reference_positions - corrected_brake_targets,
                            axis=1,
                        ),
                        previous_active_reference_actual_error_m=np.linalg.norm(
                            previous_reference_positions - actual,
                            axis=1,
                        ),
                    )
                    transition["rotation_freeze"] = dict(
                        raw_target_change_deg=np.degrees(
                            Rotation.from_matrix(rotations @ previous_rotation_targets.transpose(0, 2, 1)).magnitude()
                        ),
                        corrected_hold_error_deg=np.degrees(
                            Rotation.from_matrix(
                                correction @ rotations @ reference_rotations.transpose(0, 2, 1)
                            ).magnitude()
                        ),
                    )
                    motion.body.put(motion.x)
                    capture_torso = self.ik.data.xmat[motion.body.torso].reshape(3, 3).copy()
                    motion.freeze_posture()
                    self.execution.enter("grasp.close")
                    self.stats["capture"] = dict(
                        t=stamp,
                        geometry=geometry,
                        admission="current_geometry",
                        settle=capture_status,
                        transition=transition,
                    )

                phase = capture_admission.consume(phase, geometry, enter_close)
            elif phase == "close":
                torso_target = capture_torso
                closure = min(1.0, closure + min(dt, 0.1) / self.cfg.close_seconds)
                contact = bilateral_dex1_contact(obj, "right")
                palm_relative = object_in_palm_frame(state, "right", obj["pos"])
                load_status = load_control.step(stamp, contact, palm_relative, required=closure >= 1)
                if closure >= 1:
                    if load_status["timed_out"]:
                        return False, "whole-body no sustained bilateral contact"
                    if load_status["admitted"] and captured_object_in_palm is None:
                        captured_object_in_palm = Primitives._rot(state["palms"]["right"]["quat"]).T @ Primitives._rot(
                            obj["quat"]
                        )
                    if loaded_endpoint_request is not None or load_status["admitted"]:
                        if loaded_endpoint_request is None:
                            loaded_endpoint_request_count += 1
                            loaded_endpoint_request = self._loaded_endpoint_request(
                                state,
                                motion,
                                targets,
                                rotations,
                                desired_rot,
                            )
                            loaded_endpoint_request["key"] = loaded_endpoint_request_count
                            loaded_endpoint_request["submitted_wall"] = time.monotonic()
                            endpoint_response = self.loaded_endpoint_planner.request(
                                loaded_endpoint_request["key"],
                                loaded_endpoint_request["payload"],
                            )
                        else:
                            endpoint_response = self.loaded_endpoint_planner.poll(loaded_endpoint_request["key"])
                            if endpoint_response["status"] == "missing":
                                endpoint_response = self.loaded_endpoint_planner.request(
                                    loaded_endpoint_request["key"],
                                    loaded_endpoint_request["payload"],
                                )
                        endpoint_status = {
                            "status": endpoint_response["status"],
                            "request_count": loaded_endpoint_request_count,
                            "request_source": loaded_endpoint_request["contract"]["source"],
                            "request_age_seconds": time.monotonic() - loaded_endpoint_request["submitted_wall"],
                            "poll_seconds": float(endpoint_response.get("poll_seconds", 0.0)),
                            "invalidations": loaded_endpoint_invalidations,
                            "consumptions": loaded_endpoint_consumptions,
                            "hold_reference_sends": loaded_endpoint_hold_sends,
                        }
                        self.stats["loaded_endpoint"] = endpoint_status
                        if endpoint_response["status"] == "closed":
                            return False, "whole-body loaded endpoint planner closed"
                        if endpoint_response["status"] == "ready":
                            endpoint_result = endpoint_response["result"]
                            endpoint_contract = self._loaded_endpoint_contract_difference(
                                loaded_endpoint_request,
                                state,
                                motion,
                                targets,
                                rotations,
                            )
                            consumed_wall = time.monotonic()
                            submitted_wall = float(
                                endpoint_result.get(
                                    "submitted_wall",
                                    loaded_endpoint_request["submitted_wall"],
                                )
                            )
                            completed_wall = float(endpoint_result.get("completed_wall", consumed_wall))
                            request_source_time = loaded_endpoint_request["contract"]["source"].get("t")
                            endpoint_event = {
                                **endpoint_contract,
                                "request_count": loaded_endpoint_request_count,
                                "submitted_wall": submitted_wall,
                                "completed_wall": completed_wall,
                                "consumed_wall": consumed_wall,
                                "request_age_seconds": consumed_wall - submitted_wall,
                                "result_age_seconds": consumed_wall - completed_wall,
                                "request_source_age_seconds": (
                                    None if request_source_time is None else stamp - float(request_source_time)
                                ),
                                "solve_seconds": float(endpoint_result.get("solve_seconds", 0.0)),
                                "result_source": endpoint_result.get("source"),
                            }
                            if not endpoint_contract["matches"]:
                                endpoint_event["status"] = "invalidated"
                                loaded_endpoint_invalidations.append(endpoint_event)
                                self.stats["loaded_endpoint"] = {
                                    **endpoint_event,
                                    "invalidations": loaded_endpoint_invalidations,
                                    "consumptions": loaded_endpoint_consumptions,
                                    "hold_reference_sends": loaded_endpoint_hold_sends,
                                }
                                loaded_endpoint_request = None
                            elif not endpoint_result["admitted"]:
                                endpoint_event["status"] = "infeasible"
                                endpoint_event["result"] = endpoint_result
                                self.stats["loaded_endpoint"] = {
                                    **endpoint_event,
                                    "invalidations": loaded_endpoint_invalidations,
                                    "consumptions": loaded_endpoint_consumptions,
                                    "hold_reference_sends": loaded_endpoint_hold_sends,
                                }
                                reason = endpoint_result.get("reason") or "standing endpoint outside tolerance"
                                return False, f"whole-body loaded path {reason}"
                            else:
                                current_request = self._loaded_endpoint_request(
                                    state,
                                    motion,
                                    targets,
                                    rotations,
                                    desired_rot,
                                )
                                phase, progress = "lift", 0.0
                                motion.resume_translation_feedback()
                                # Rotation compensation stays frozen until the
                                # bottle actually leaves the table: resuming it
                                # here twists the hand around a pinned bottle.
                                object_lifted = False
                                loaded_hold_frozen = False
                                loaded_stability_gate = self._loaded_stability_gate()
                                relative_load = palm_relative.copy()
                                loaded_start = current_request["loaded_start"].copy()
                                loaded_tracking_offset = actual[1] - loaded_start
                                loaded_rotation_offset = Primitives._rot(state["palms"]["right"]["quat"]) @ (
                                    commanded_palm_rotation(
                                        rotations[1], motion.rotation_feedback[1], self.cfg.tracking_gain
                                    ).T
                                )
                                standing_reference = current_request["standing_reference"].copy()
                                standing_weights = current_request["standing_weights"].copy()
                                motion.retarget_posture(standing_reference, standing_weights)
                                clearance_rise = max(
                                    0.035,
                                    self.cfg.lift_height - self.cfg.capture_height_tolerance_m,
                                )
                                loaded_rise_margin = self.cfg.stance_ik_tolerance
                                loaded_commanded_rise = current_request["commanded_rise_m"]
                                carry_endpoint = current_request["endpoint"].copy()
                                loaded_path = current_request["path"]
                                loaded_clearance_fraction = loaded_path["clearance_fraction"]
                                orientation_bad_since = None
                                endpoint_event.update(
                                    status="consumed",
                                    consume_source={name: state.get(name) for name in ("seq", "t", "wall")},
                                    consumed_endpoint=carry_endpoint.copy(),
                                    result=endpoint_result,
                                )
                                loaded_endpoint_consumptions.append(endpoint_event)
                                self.stats["loaded_endpoint"] = {
                                    **endpoint_event,
                                    "invalidations": loaded_endpoint_invalidations,
                                    "consumptions": loaded_endpoint_consumptions,
                                    "hold_reference_sends": loaded_endpoint_hold_sends,
                                }
                                self.stats["loaded_path"] = dict(
                                    frame="world",
                                    start=loaded_start.copy(),
                                    endpoint=carry_endpoint.copy(),
                                    required_measured_rise_m=self.cfg.lift_height,
                                    commanded_rise_m=loaded_commanded_rise,
                                    tracking_margin_m=loaded_rise_margin,
                                    clearance_rise_m=clearance_rise,
                                    entry_axis=desired_rot[:, 0].copy(),
                                    standing_reference=standing_reference.copy(),
                                    path=loaded_path,
                                    endpoint_admission=endpoint_result,
                                )
                                self.execution.enter("grasp.lift_and_recover")
            else:
                torso_target = (
                    planned_torso_rotation
                    if planned_torso_rotation is not None
                    else motion.body.heading @ Rotation.from_euler("y", lean_target).as_matrix()
                )
                contact = bilateral_dex1_contact(obj, "right")
                palm_relative = object_in_palm_frame(state, "right", obj["pos"])
                load_status = load_control.step(stamp, contact, palm_relative, required=True)
                if np.linalg.norm(palm_relative - relative_load) > 0.04:
                    return False, "whole-body bottle slipped during recovery"
                if load_status["timed_out"]:
                    return False, "whole-body bilateral contact recovery timed out"
                translation_tracking_error = float(np.linalg.norm((actual[1] - targets[1]) - loaded_tracking_offset))
                palm_rotation = Primitives._rot(state["palms"]["right"]["quat"])
                current_rotation_offset = palm_rotation @ (
                    commanded_palm_rotation(rotations[1], motion.rotation_feedback[1], self.cfg.tracking_gain).T
                )
                rotation_tracking_error = float(
                    Rotation.from_matrix(current_rotation_offset @ loaded_rotation_offset.T).magnitude()
                )
                object_rise = float(obj["pos"][2]) - obj0[2]
                if loaded_liftoff_event(object_rise, self.cfg.loaded_liftoff_rise_m, contact, object_lifted):
                    # Load transfer: the grip relation that will be carried is
                    # the one at liftoff, not the one latched while the table
                    # still pinned the bottle. Re-latch exactly once, record
                    # the on-table rotation as evidence, resume rotation
                    # compensation on the now free bottle.
                    on_table = loaded_orientation_metrics(
                        palm_rotation,
                        Primitives._rot(obj["quat"]),
                        captured_object_in_palm,
                    )
                    captured_object_in_palm = on_table["object_in_palm"].copy()
                    object_lifted = True
                    orientation_bad_since = None
                    # Resume bounded rotation compensation on the free bottle.
                    # stage3_5_frozen8 kept it frozen: the palm then tilted up
                    # to 0.3 rad during the lift and three lifts failed on
                    # bottle tilt; with it on (stage3_4) tilt stayed < 0.24.
                    # Tracking error is measured against the corrected command,
                    # so the compensation's own action is not counted as error.
                    motion.resume_rotation_feedback()
                    self.stats["loaded_liftoff"] = dict(
                        t=stamp,
                        seq=state.get("seq"),
                        rise_m=object_rise,
                        threshold_m=self.cfg.loaded_liftoff_rise_m,
                        on_table_grip_axis_swing_rad=on_table["grip_axis_swing_rad"],
                        on_table_grip_rotation_rad=on_table["grip_rotation_error_rad"],
                        object_tilt_rad=on_table["object_tilt_rad"],
                        rotation_feedback_norm_rad=float(np.linalg.norm(motion.rotation_feedback)),
                        rotation_tracking_error_rad=rotation_tracking_error,
                    )
                orientation_metrics = loaded_orientation_metrics(
                    palm_rotation,
                    Primitives._rot(obj["quat"]),
                    captured_object_in_palm,
                )
                loaded_metrics = {
                    "translation_error_m": translation_tracking_error,
                    "rotation_error_rad": rotation_tracking_error,
                    **orientation_metrics,
                }
                loaded_governor = loaded_progress_governor(
                    loaded_metrics,
                    {
                        "translation_error_m": (
                            self.cfg.loaded_tracking_translation_full_m,
                            self.cfg.loaded_tracking_translation_stop_m,
                        ),
                        "rotation_error_rad": (
                            self.cfg.loaded_tracking_rotation_full_rad,
                            self.cfg.loaded_tracking_rotation_stop_rad,
                        ),
                        "object_tilt_rad": (
                            self.cfg.loaded_object_tilt_full_rad,
                            self.cfg.loaded_object_tilt_stop_rad,
                        ),
                        "grip_axis_swing_rad": (
                            self.cfg.loaded_grip_swing_full_rad,
                            self.cfg.loaded_grip_swing_stop_rad,
                        ),
                    },
                    lifted=object_lifted,
                )
                orientation_supervision = loaded_orientation_supervision(
                    orientation_metrics,
                    {
                        "object_tilt_rad": self.cfg.loaded_object_tilt_stop_rad,
                        "grip_axis_swing_rad": self.cfg.loaded_grip_swing_stop_rad,
                    },
                    {
                        "object_tilt_rad": self.cfg.loaded_object_tilt_fail_rad,
                        "grip_axis_swing_rad": self.cfg.loaded_grip_swing_fail_rad,
                    },
                    stamp,
                    orientation_bad_since,
                    self.cfg.loaded_orientation_recovery_seconds,
                    lifted=object_lifted,
                )
                orientation_bad_since = orientation_supervision["bad_since"]
                orientation_stopped = orientation_supervision["stopped"]
                orientation_stop_duration = orientation_supervision["stop_duration_seconds"]
                if orientation_supervision["failed"]:
                    failure_reason = (
                        orientation_supervision["reason"]
                        .replace("object_tilt_rad", "bottle tilt")
                        .replace("grip_axis_swing_rad", "grip axis swing")
                    )
                    self.stats["loaded_orientation_failure"] = {
                        **orientation_supervision,
                        "reason": failure_reason,
                        "source": {name: state.get(name) for name in ("seq", "t", "wall")},
                        "metrics": loaded_metrics,
                        "governor": loaded_governor,
                    }
                    if orientation_supervision["timed_out"]:
                        return False, f"whole-body loaded orientation recovery timed out: {failure_reason}"
                    return False, f"whole-body loaded {failure_reason} limit"
                loaded_stability = self._loaded_stability_step(loaded_stability_gate, state)
                quiet_tracking_admitted = bool(loaded_stability["ready"])
                progress = advance_loaded_progress(
                    progress,
                    dt,
                    self.cfg.lift_seconds,
                    load_status["progress_scale"],
                    loaded_governor["scale"],
                )
                if object_rise < clearance_rise:
                    progress = min(progress, loaded_clearance_fraction)
                if progress >= 1 and not loaded_hold_frozen:
                    # Endpoint reached: hold the corrected reference still so the
                    # measured quiet window can form; both compensation loops
                    # otherwise keep the hand moving at their own rate limits.
                    motion.freeze_translation_feedback()
                    motion.freeze_rotation_feedback()
                    loaded_hold_frozen = True
                    self.stats["loaded_hold"] = dict(
                        t=stamp,
                        seq=state.get("seq"),
                        rise_m=object_rise,
                        translation_feedback_norm_m=float(np.linalg.norm(motion.feedback[1])),
                        rotation_feedback_norm_rad=float(np.linalg.norm(motion.rotation_feedback[1])),
                        translation_error_m=translation_tracking_error,
                        rotation_error_rad=rotation_tracking_error,
                    )
                targets[1], loaded_path_status = loaded_world_path(
                    loaded_start,
                    desired_rot[:, 0],
                    loaded_commanded_rise,
                    progress,
                    carry_distance=self.cfg.lift_height,
                )
                if loaded_path_status["phase"] == "carry_inward" and object_rise < clearance_rise:
                    return False, "whole-body inward carry before measured table clearance"
                self.context.update(
                    loaded_path=loaded_path_status,
                    loaded_tracking=dict(
                        admitted=quiet_tracking_admitted,
                        translation_error_m=translation_tracking_error,
                        rotation_error_rad=rotation_tracking_error,
                        object_tilt_rad=orientation_metrics["object_tilt_rad"],
                        grip_rotation_error_rad=orientation_metrics["grip_rotation_error_rad"],
                        grip_axis_swing_rad=orientation_metrics["grip_axis_swing_rad"],
                        progress_scale=loaded_governor["scale"],
                        scale_components=loaded_governor["components"],
                        lifted=object_lifted,
                        orientation_stopped=orientation_stopped,
                        orientation_stop_duration_seconds=orientation_stop_duration,
                        stability=loaded_stability,
                    ),
                    load=load_status,
                )
                if object_rise >= clearance_rise:
                    if planned_torso_rotation is None:
                        torso_target = (
                            motion.body.heading
                            @ Rotation.from_euler("y", lean_target * (1 - smooth(progress))).as_matrix()
                        )
                    else:
                        torso_target = Slerp(
                            [0, 1],
                            Rotation.from_matrix([planned_torso_rotation, motion.body.heading]),
                        )(smooth(progress)).as_matrix()
                motion.body.put(
                    np.r_[
                        state["pelvis"]["pos"],
                        Rotation.from_matrix(Primitives._rot(state["pelvis"]["quat"])).as_euler("xyz")[1],
                        state["joints"],
                    ]
                )
                torso = motion.body.ik.data.xmat[motion.body.torso].reshape(3, 3)
                lean = np.arccos(np.clip(torso[2, 2], -1, 1))
                standing_leg_error = float(
                    np.max(np.abs(np.asarray(state["joints"], dtype=float)[:12] - standing_reference[:12]))
                )
                ready = (
                    progress >= 1
                    and float(obj["pos"][2]) - obj0[2] >= 0.06
                    and lean < np.radians(12)
                    and load_status["admitted"]
                    and quiet_tracking_admitted
                )
                ready_since = (stamp if ready_since is None else ready_since) if ready else None
                if ready_since is not None and stamp - ready_since >= 0.2:
                    self.stats["loaded_upright"] = dict(
                        t=stamp,
                        lean_deg=np.degrees(lean),
                        rise=float(obj["pos"][2]) - obj0[2],
                        standing_leg_error_rad=standing_leg_error,
                        tracking=self.context["loaded_tracking"],
                    )
                    return True, "continuous whole-body grasp and upright recovery"
            if phase == "reach":
                motion.blend_posture(progress)
            elif phase == "lift":
                motion.blend_posture(progress)
            previous_x, previous_root = motion.x.copy(), motion.root.copy()
            drift = motion.measured_foot_drift(state)
            if max(drift) > 0.05:
                self.stats["failure"] = dict(state=state, measured_foot_drift=drift)
                return False, "whole-body measured foot drift"
            if torso_target is not None:
                turn_delta = Rotation.from_matrix(torso_target @ torso_reference.T).as_rotvec()
                turn_delta *= min(1.0, 0.25 * dt / max(np.linalg.norm(turn_delta), 1e-9))
                torso_reference = Rotation.from_rotvec(turn_delta).as_matrix() @ torso_reference
            pose = motion.step(
                targets,
                rotations,
                actual,
                dt,
                torso_rotation=torso_reference,
                measured_rotations=(
                    np.array([Primitives._rot(state["palms"][side]["quat"]) for side in ("left", "right")])
                    if phase in ("reach", "lift")
                    else None
                ),
            )
            # SONIC references include systematic decoder tracking error. Sweep
            # the next reference increment from the measured body, not a nominal
            # reference body placed inside the real scene as if it were measured.
            predicted = deepcopy(state)
            predicted["joints"] = np.asarray(state["joints"]) + pose.joints - previous_x[4:]
            delta_rotation = Primitives._rot(motion.root) @ Primitives._rot(previous_root).T
            predicted["pelvis"] = dict(
                state["pelvis"],
                pos=np.asarray(state["pelvis"]["pos"]) + pose.position - previous_x[:3],
                quat=Rotation.from_matrix(delta_rotation @ Primitives._rot(state["pelvis"]["quat"])).as_quat()[
                    [3, 0, 1, 2]
                ],
            )
            reason = self._clear(
                state,
                predicted["joints"],
                phase != "reach" or cap_clearing_target_contact_allowed(actual[1], target),
                candidate_base=[*predicted["pelvis"]["pos"], *predicted["pelvis"]["quat"]],
                attached=phase == "lift",
            )
            if reason:
                self.stats["failure"] = dict(
                    state=state,
                    predicted=predicted,
                    reference=pose.joints,
                    target=target,
                    planned_tool=targets[1],
                    phase=phase,
                    reason=reason,
                )
                return False, reason
            if max(pose.foot_errors) > 0.025 or np.linalg.norm(pose.com_shift) > 0.04:
                self.stats["failure"] = dict(
                    state=state,
                    predicted=predicted,
                    reference=pose.joints,
                    foot_error=pose.foot_errors,
                    com_shift=pose.com_shift,
                    phase=phase,
                )
                return False, "whole-body support reference outside tolerance"
            sent, sent_velocity = (
                (pose.joints, motion.v[4:]) if self.filter is None else self.filter.step(pose.joints, dt)
            )
            sent_root_angular_velocity = Rotation.from_matrix(delta_rotation).as_rotvec() / dt
            self.arm.q, self.arm.root_quat = sent, motion.root
            self.arm.hands["right"] = HAND_OPEN + closure * (HAND_CLOSED_RIGHT - HAND_OPEN)
            hold_send = push_loaded_reference(
                self.arm,
                sent,
                sent_velocity,
                sent_root_angular_velocity,
                state,
                dt,
                endpoint_pending=phase == "close" and loaded_endpoint_request is not None,
                request_count=loaded_endpoint_request_count,
            )
            if hold_send is not None:
                loaded_endpoint_hold_sends.append(hold_send)
                self.stats["loaded_endpoint"]["hold_reference_sends"] = loaded_endpoint_hold_sends
            if stamp - last_sample >= 0.1:
                self.stats["samples"].append(
                    dict(
                        seq=state["seq"],
                        t=stamp,
                        phase=phase,
                        progress=progress,
                        capture_settle=capture_status,
                        observer_target=None if target is None else np.asarray(target).tolist(),
                        capture_geometry=geometry,
                        terminal_motion=terminal_status,
                        contact=load_status,
                        target=targets[1].tolist(),
                        tool=actual[1].tolist(),
                        object=list(obj["pos"]),
                        waist=list(state["joints"][12:15]),
                        reference=list(pose.joints),
                        velocity=list(motion.v[4:]),
                        root_angular_velocity=list(sent_root_angular_velocity),
                        **motion.diagnostics,
                    )
                )
                last_sample = stamp
        return False, "whole-body physical state stopped"
