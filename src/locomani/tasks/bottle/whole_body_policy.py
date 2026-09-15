"""Camera-admitted bottle reach through one supported descent/closure/lift reference."""

import json
import time
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

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
from locomani.primitives.supported_motion import SupportedMotion
from locomani.tasks.bottle.assisted_target import BottleTargetObserver
from locomani.tasks.bottle.grasp_capture import (
    DEX1_PAD_X_MAX,
    HEIGHT_TOL,
    LATERAL_TOL,
    capture_axial_clearance,
    capture_tool_goal,
    closure_geometry,
)
from locomani.tasks.bottle.ready_grasp_policy import OFFSET
from locomani.tasks.bottle.walking_reach import WalkingReach


def smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


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


class ForwardSpeedFilter:
    """Smooth measured axial advance without hiding a pulse from capture admission."""

    def __init__(self, lookahead_seconds):
        lookahead_seconds = float(lookahead_seconds)
        if not np.isfinite(lookahead_seconds) or lookahead_seconds <= 0:
            raise ValueError("forward-speed filter lookahead must be positive finite")
        self.attack_tau = 0.5 * lookahead_seconds
        self.release_tau = lookahead_seconds
        self.value = 0.0

    def step(self, measured_forward_speed, dt):
        measured_forward_speed = float(measured_forward_speed)
        dt = float(dt)
        if not np.isfinite(measured_forward_speed) or not np.isfinite(dt) or dt <= 0:
            raise ValueError("forward-speed filter sample must be positive-time finite")
        measured_forward_speed = max(0.0, measured_forward_speed)
        tau = self.attack_tau if measured_forward_speed > self.value else self.release_tau
        blend = 1.0 - np.exp(-dt / tau)
        self.value += blend * (measured_forward_speed - self.value)
        return self.value


def stopping_retreat(forward_speed, acceleration, lookahead_seconds):
    """Axial room needed for lookahead plus constant-acceleration braking."""
    forward_speed = float(forward_speed)
    acceleration = float(acceleration)
    lookahead_seconds = float(lookahead_seconds)
    if (
        not np.isfinite(forward_speed)
        or not np.isfinite(acceleration)
        or acceleration <= 0
        or not np.isfinite(lookahead_seconds)
        or lookahead_seconds < 0
    ):
        raise ValueError("stopping-retreat inputs must be finite with positive acceleration")
    forward_speed = max(0.0, forward_speed)
    return forward_speed * lookahead_seconds + forward_speed**2 / (2.0 * acceleration)


def momentum_guarded_depth(
    guarded_depth_m,
    filtered_forward_speed,
    acceleration,
    lookahead_seconds,
    maximum_depth_m=0.10,
):
    """Retreat the active goal smoothly while retaining the cap-clearance bound."""
    retreat = stopping_retreat(filtered_forward_speed, acceleration, lookahead_seconds)
    uncapped = float(guarded_depth_m) + retreat
    depth = min(float(maximum_depth_m), uncapped)
    return depth, retreat, uncapped > depth


def predictive_capture_geometry(
    geometry,
    measured_forward_speed,
    filtered_forward_speed,
    acceleration,
    lookahead_seconds,
):
    """Require current and predicted stopped depth to occupy the capture interval."""
    result = dict(geometry)
    prediction_speed = max(
        0.0,
        float(measured_forward_speed),
        float(filtered_forward_speed),
    )
    retreat = stopping_retreat(prediction_speed, acceleration, lookahead_seconds)
    predicted_depth = float(result["depth_m"]) - retreat
    depth_min = float(result["preferred_depth_m"] - result["depth_tolerance_m"])
    depth_max = float(result["preferred_depth_m"] + result["depth_tolerance_m"])
    stopped_depth_ready = depth_min <= predicted_depth <= depth_max
    current_geometry_ready = bool(result["ready"])
    result["ready"] = current_geometry_ready and stopped_depth_ready
    result["rejection_reasons"] = list(result["rejection_reasons"])
    if not stopped_depth_ready:
        result["rejection_reasons"].append("predicted_stopping_depth")
    result["stopping"] = dict(
        current_geometry_ready=current_geometry_ready,
        prediction_speed_m_s=prediction_speed,
        stopping_retreat_m=retreat,
        predicted_depth_m=predicted_depth,
        accepted_depth_interval_m=[depth_min, depth_max],
        admitted=stopped_depth_ready,
    )
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
    measured_forward_speed=0.0,
    brake_acceleration=None,
    brake_lookahead_seconds=0.0,
):
    """One continuous, cap-clearing step toward the bottle's capture region."""
    above = max(measured[2], command[2]) - target[2]
    gap = final_depth_m + 0.10 * smooth((above - 0.018) / 0.04)
    goal = target - gap * entry_axis
    delta = goal - command
    behind = min(float((target - hand) @ entry_axis) for hand in (measured, command))
    if delta[2] < 0 and above > 0.035:
        delta[2] *= smooth((behind - 0.05) / 0.05)
    speed = min(hand_speed, 0.02 + 0.5 * np.linalg.norm(goal - measured))
    step_dt = min(dt, 0.1)
    step = delta * min(1.0, speed * step_dt / max(np.linalg.norm(delta), 1e-9))
    if brake_acceleration is not None:
        remaining = float((goal - measured) @ entry_axis)
        advancing_speed_cap = terminal_entry_speed(
            remaining,
            measured_forward_speed,
            speed,
            brake_acceleration,
            brake_lookahead_seconds,
        )
        axial_step = float(step @ entry_axis)
        if axial_step > 0:
            step -= entry_axis * max(0.0, axial_step - advancing_speed_cap * step_dt)
    return command + step, gap


def terminal_entry_speed(
    remaining,
    advancing_speed,
    nominal_speed,
    acceleration,
    lookahead_seconds,
):
    """Cap terminal advance using measured motion and bounded stopping distance."""
    remaining = max(0.0, float(remaining))
    advancing_speed = max(0.0, float(advancing_speed))
    effective_remaining = max(0.0, remaining - advancing_speed * float(lookahead_seconds))
    braking_speed = np.sqrt(2.0 * float(acceleration) * effective_remaining)
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
    measured_forward_speed,
    brake_acceleration,
    brake_lookahead_seconds,
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
        measured_forward_speed=measured_forward_speed,
        brake_acceleration=brake_acceleration,
        brake_lookahead_seconds=brake_lookahead_seconds,
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

    def walk_tick(self, state, *, edge_distance=None):
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
        return True

    def _hold_planner(self, state):
        self.p.cmd.frame(
            mode=LocomotionMode.IDLE, height=-1.0, speed=0.0, movement=(0, 0, 0), facing=self.p._facing(state)
        )

    def _brake(self):
        state = self.p.obs()
        start, quiet, last_progress = float(state["t"]), None, time.monotonic()
        previous = start
        while time.monotonic() - last_progress < 2:
            stamp = float(state["t"])
            self._hold_planner(state)
            self._observe_approach_settlement(state)
            if state["fallen"] or stamp - start > 3:
                self._observe_approach_settlement(
                    state,
                    complete=True,
                    completion_reason="fallen" if state["fallen"] else "brake physical timeout",
                )
                return False
            if not self.walk_tick(state):
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
            if state["fallen"] or stamp - start > 3:
                self._observe_approach_settlement(
                    state,
                    complete=True,
                    completion_reason="fallen" if state["fallen"] else "brake physical timeout",
                )
                return False
            if stamp > previous:
                previous, last_progress = stamp, time.monotonic()
            quiet = (stamp if quiet is None else quiet) if np.linalg.norm(state["pelvis"]["vel"][:2]) < 0.06 else None
            if quiet is not None and stamp - quiet >= 0.25:
                self._observe_approach_settlement(
                    state,
                    complete=True,
                    settled=True,
                    completion_reason="brake quiet",
                )
                return True
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
            if self.arm is not None:
                # Success returns to walking only after measured upright recovery.
                # Failure first pins a bounded stopped reference (never rise
                # through an obstructed outstretched hand as a cleanup), then
                # hands that same pose back to the planner. Leaving the deploy
                # in streamed mode across the round boundary made the next
                # reset land 0.10-0.12 m off and repeat to the end of the batch:
                # 8/8 stand-stage failures on 2026-09-14 followed a whole-body
                # abort, four of them in one unbroken run.
                if not ok:
                    self.arm._push(self.arm.q, reference_velocity=np.zeros(29))
                self.arm.stop(preserve_reference=True)
            # Serialization and camera teardown belong to round cleanup. Doing
            # them while carrying can starve planner commands past its timeout.
            self.p.cmd.reference_observer = self.previous_observer

    def _brake_handoff(self):
        """Brake the admitted loaded reference before returning it to the planner."""
        state = self.p.obs()
        try:
            prediction = self.arm.stopping_reference()
        except (RuntimeError, ValueError) as exc:
            self.stats["handback"] = dict(admitted=False, reason=str(exc))
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
            handback["reason"] = str(exc)
            return False, f"whole-body handback braking failed: {exc}"
        final_state = self.p.obs()
        final_measured_q = np.asarray(final_state["joints"], dtype=float)
        handback.update(
            braking=braking,
            actual_final=dict(
                joints=final_measured_q.copy(),
                q_tracking_error=float(np.max(np.abs(final_measured_q - np.asarray(self.arm.q)))),
            ),
        )
        return True, "continuous whole-body grasp upright recovery and stopped handback"

    def prepare_carry(self):
        """The same reference already recovered upright with the bottle secured."""
        return self.completed

    def _reach(self):
        state = self.p.obs()
        target = self.target.observe(state)
        if target is None:
            return False, "whole-body target not visible in student cameras"
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
        self.arm.start(hold_planner_pose=True, inherit_planner_velocity=True)
        state = self.p.obs()
        comfort_reference = comfort_weights = None
        if self.cfg.comfort_posture:
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
        root = Primitives._rot(state["pelvis"]["quat"])
        desired_rot = self.walking.rotation
        if desired_rot is None:
            desired_rot = Rotation.from_euler("z", np.arctan2(root[1, 0], root[0, 0])).as_matrix()
        turn = Slerp([0, 1], Rotation.from_matrix([rotations[1], desired_rot]))
        # Preparation already moved the hand: lean follows bottle height, not
        # the remaining hand error (which would erase the bend after good preparation).
        lean_target = lean_angle(state["pelvis"]["pos"], target, self.cfg.lean_max)
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
        previous_actual = initial.copy()
        entry_speed_filter = ForwardSpeedFilter(self.cfg.capture_brake_lookahead_seconds)
        capture_torso = None
        last_sample, ready_since = -1.0, None
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
            measured_entry_speed = max(
                0.0,
                float((actual[1] - previous_actual[1]) @ desired_rot[:, 0]) / dt,
            )
            filtered_entry_speed = entry_speed_filter.step(measured_entry_speed, dt)
            previous_actual = actual.copy()
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
            goal_stopping_retreat_m = None
            goal_depth_capped = None
            if guarded_depth_m is not None:
                dynamic_guarded_depth_m, goal_stopping_retreat_m, goal_depth_capped = momentum_guarded_depth(
                    guarded_depth_m,
                    filtered_entry_speed,
                    self.cfg.capture_brake_acceleration,
                    self.cfg.capture_brake_lookahead_seconds,
                )
            if geometry is not None:
                geometry = predictive_capture_geometry(
                    geometry,
                    measured_entry_speed,
                    filtered_entry_speed,
                    self.cfg.capture_brake_acceleration,
                    self.cfg.capture_brake_lookahead_seconds,
                )
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
                        filtered_entry_speed_m_s=filtered_entry_speed,
                        goal_stopping_retreat_m=goal_stopping_retreat_m,
                        dynamic_guarded_depth_m=dynamic_guarded_depth_m,
                        goal_depth_capped=goal_depth_capped,
                        prediction_speed_m_s=geometry["stopping"]["prediction_speed_m_s"],
                        predicted_stopping_retreat_m=geometry["stopping"]["stopping_retreat_m"],
                        predicted_stop_depth_m=geometry["stopping"]["predicted_depth_m"],
                        predicted_stop_admitted=geometry["stopping"]["admitted"],
                    )
                ),
                measured_entry_speed_m_s=measured_entry_speed,
                filtered_entry_speed_m_s=filtered_entry_speed,
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
                    measured_forward_speed=max(measured_entry_speed, filtered_entry_speed),
                    brake_acceleration=self.cfg.capture_brake_acceleration,
                    brake_lookahead_seconds=self.cfg.capture_brake_lookahead_seconds,
                )
                progress = min(1.0, progress + min(dt, 0.1) / 1.0)
                rotations[1] = turn(smooth(progress)).as_matrix()
                torso_target = motion.body.heading @ Rotation.from_euler("y", lean_target).as_matrix()
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
                    measured_entry_speed_m_s=measured_entry_speed,
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
                    if load_status["admitted"]:
                        phase, progress = "lift", 0.0
                        lift_start = motion.body.tools(motion.x).copy() - self.cfg.tracking_gain * motion.feedback
                        motion.resume_translation_feedback()
                        relative_load = palm_relative.copy()
                        # The hand keeps its pose relative to the reference torso
                        # while the body stands up: no retract, no folding the
                        # elbow to hold a world point the shoulder is moving away
                        # from. Only the explicit lift is added, in world z.
                        lift_offset = torso_offset(self.ik, motion, lift_start[1])
                        self.execution.enter("grasp.lift_and_recover")
            else:
                torso_target = motion.body.heading @ Rotation.from_euler("y", lean_target).as_matrix()
                contact = bilateral_dex1_contact(obj, "right")
                palm_relative = object_in_palm_frame(state, "right", obj["pos"])
                load_status = load_control.step(stamp, contact, palm_relative, required=True)
                if np.linalg.norm(palm_relative - relative_load) > 0.04:
                    return False, "whole-body bottle slipped during recovery"
                if load_status["timed_out"]:
                    return False, "whole-body bilateral contact recovery timed out"
                progress = advance_load_progress(
                    progress,
                    dt,
                    self.cfg.lift_seconds,
                    load_status["progress_scale"],
                )
                targets[1] = lift_target(self.ik, motion, lift_offset, self.cfg.lift_height * smooth(progress))
                if float(obj["pos"][2]) - obj0[2] >= 0.035:
                    torso_target = (
                        motion.body.heading @ Rotation.from_euler("y", lean_target * (1 - smooth(progress))).as_matrix()
                    )
                motion.body.put(
                    np.r_[
                        state["pelvis"]["pos"],
                        Rotation.from_matrix(Primitives._rot(state["pelvis"]["quat"])).as_euler("xyz")[1],
                        state["joints"],
                    ]
                )
                torso = motion.body.ik.data.xmat[motion.body.torso].reshape(3, 3)
                lean = np.arccos(np.clip(torso[2, 2], -1, 1))
                ready = (
                    progress >= 1
                    and float(obj["pos"][2]) - obj0[2] >= 0.06
                    and lean < np.radians(12)
                    and load_status["admitted"]
                )
                ready_since = (stamp if ready_since is None else ready_since) if ready else None
                if ready_since is not None and stamp - ready_since >= 0.2:
                    self.stats["loaded_upright"] = dict(
                        t=stamp, lean_deg=np.degrees(lean), rise=float(obj["pos"][2]) - obj0[2]
                    )
                    return True, "continuous whole-body grasp and upright recovery"
            if phase == "reach":
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
                    if phase == "reach"
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
                phase != "reach" or abs(actual[1, 2] - target[2]) < 0.035,
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
            self.arm._push(
                sent,
                reference_velocity=sent_velocity,
                reference_time=stamp,
                root_angular_velocity=sent_root_angular_velocity,
            )
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
                        measured_entry_speed_m_s=measured_entry_speed,
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
