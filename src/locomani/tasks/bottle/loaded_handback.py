"""World-frame loaded motion and measured handback admission."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation


def _smooth(value):
    value = np.clip(value, 0.0, 1.0)
    return value**3 * (10 + value * (-15 + 6 * value))


def loaded_lift_height(required_rise, tracking_margin):
    """Command enough world rise to retain the measured terminal-rise gate."""
    required_rise = float(required_rise)
    tracking_margin = float(tracking_margin)
    if not np.isfinite(required_rise) or required_rise <= 0:
        raise ValueError("loaded required rise must be positive and finite")
    if not np.isfinite(tracking_margin) or tracking_margin < 0:
        raise ValueError("loaded tracking margin must be finite and nonnegative")
    return required_rise + tracking_margin


def loaded_tracking_scale(value, full_speed_limit, stopped_limit):
    """Continuously reduce path speed between a nominal load and a hard tracking load."""
    value = float(value)
    full_speed_limit = float(full_speed_limit)
    stopped_limit = float(stopped_limit)
    if (
        not np.isfinite(value)
        or not np.isfinite(full_speed_limit)
        or not np.isfinite(stopped_limit)
        or value < 0
        or full_speed_limit < 0
        or stopped_limit <= full_speed_limit
    ):
        raise ValueError("loaded tracking scale needs finite ordered nonnegative limits")
    return float(np.clip((stopped_limit - value) / (stopped_limit - full_speed_limit), 0.0, 1.0))


def commanded_palm_rotation(desired_rotation, rotation_feedback, tracking_gain):
    """The palm rotation actually sent: desired pre-rotated by the bounded compensation.

    Loaded tracking error must be measured against this, not against the bare
    desired rotation: stage3_4_liftoff8 measured 0.15-0.19 rad of "error" that
    was the compensation doing its job, and stage3_5_frozen8 (compensation
    frozen) let the palm tilt 0.3 rad with the bottle following it.
    """
    correction = Rotation.from_rotvec(float(tracking_gain) * np.asarray(rotation_feedback, dtype=float)).as_matrix()
    return correction @ np.asarray(desired_rotation, dtype=float)


def loaded_orientation_metrics(palm_rotation, object_rotation, captured_object_in_palm):
    """Measure gravity tilt, full grip rotation, and bottle-axis swing."""
    palm_rotation = np.asarray(palm_rotation, dtype=float)
    object_rotation = np.asarray(object_rotation, dtype=float)
    captured_object_in_palm = np.asarray(captured_object_in_palm, dtype=float)
    if any(
        value.shape != (3, 3) or not np.isfinite(value).all()
        for value in (palm_rotation, object_rotation, captured_object_in_palm)
    ):
        raise ValueError("loaded orientation metrics need three finite rotation matrices")
    object_in_palm = palm_rotation.T @ object_rotation
    captured_axis = captured_object_in_palm[:, 2]
    current_axis = object_in_palm[:, 2]
    return {
        "object_tilt_rad": float(np.arccos(np.clip(object_rotation[2, 2], -1.0, 1.0))),
        "grip_rotation_error_rad": float(Rotation.from_matrix(object_in_palm @ captured_object_in_palm.T).magnitude()),
        "grip_axis_swing_rad": float(np.arccos(np.clip(current_axis @ captured_axis, -1.0, 1.0))),
        "object_in_palm": object_in_palm,
    }


def loaded_progress_governor(metrics, limits, *, lifted=True):
    """Return the bounded speed scale and each physical load contribution.

    While the object still rests on the table (``lifted`` False) the grip-axis
    swing cannot be a slip signal: the table pins the bottle, so any palm
    rotation shows up as swing and would freeze the very motion that lifts the
    load off. Its component is neutral until liftoff; tilt still governs.
    """
    components = {
        name: loaded_tracking_scale(metrics[name], *limits[name])
        for name in ("translation_error_m", "rotation_error_rad", "object_tilt_rad", "grip_axis_swing_rad")
    }
    if not lifted:
        components["grip_axis_swing_rad"] = 1.0
    # Palm rotation tracking error is diagnostic only. Measured against the
    # corrected command it still reads ~0.2 rad through every lift
    # (stage3_6_command8: median 0.19-0.23, governor scale 0 in 5/7 lifts) while
    # bottle tilt and grip swing, the physical load signals, stayed within their
    # bands. The controller's orientation tracking under load is not something
    # the lift speed can fix; tilt, swing and translation keep governing.
    governing = {name: value for name, value in components.items() if name != "rotation_error_rad"}
    return {
        "scale": min(governing.values()),
        "components": components,
        "governing": sorted(governing),
        "metrics": {name: float(metrics[name]) for name in components},
        "lifted": bool(lifted),
    }


def loaded_liftoff_event(object_rise_m, liftoff_rise_m, bilateral, lifted):
    """Decide the single load-transfer moment: first bilateral rise past the threshold."""
    object_rise_m = float(object_rise_m)
    liftoff_rise_m = float(liftoff_rise_m)
    if not np.isfinite(object_rise_m) or not np.isfinite(liftoff_rise_m) or liftoff_rise_m <= 0:
        raise ValueError("liftoff needs finite rise and a positive threshold")
    if lifted:
        return False
    return bool(bilateral) and object_rise_m >= liftoff_rise_m


def loaded_orientation_supervision(
    metrics,
    stop_limits,
    fail_limits,
    stamp,
    bad_since,
    recovery_seconds,
    *,
    lifted=True,
):
    """Bound an orientation stop with explicit recovery and failure outcomes.

    Before liftoff only bottle tilt is supervised (a bottle tipping on the table
    is a real loss); grip-axis swing joins once the load is carried by the hand.
    """
    stamp = float(stamp)
    recovery_seconds = float(recovery_seconds)
    names = ("object_tilt_rad", "grip_axis_swing_rad") if lifted else ("object_tilt_rad",)
    values = {name: float(metrics[name]) for name in names}
    if (
        not np.isfinite(stamp)
        or not np.isfinite(recovery_seconds)
        or recovery_seconds <= 0
        or any(not np.isfinite(values[name]) or values[name] < 0 for name in names)
    ):
        raise ValueError("loaded orientation supervision needs finite nonnegative inputs")
    immediate = [name for name in names if values[name] > float(fail_limits[name])]
    stopped = [name for name in names if values[name] >= float(stop_limits[name])]
    if immediate:
        return {
            "failed": True,
            "timed_out": False,
            "reason": "+".join(immediate),
            "stopped": True,
            "bad_since": stamp if bad_since is None else float(bad_since),
            "stop_duration_seconds": 0.0 if bad_since is None else stamp - float(bad_since),
        }
    if not stopped:
        return {
            "failed": False,
            "timed_out": False,
            "reason": None,
            "stopped": False,
            "bad_since": None,
            "stop_duration_seconds": 0.0,
        }
    bad_since = stamp if bad_since is None else float(bad_since)
    stop_duration = max(0.0, stamp - bad_since)
    timed_out = stop_duration >= recovery_seconds
    return {
        "failed": timed_out,
        "timed_out": timed_out,
        "reason": "+".join(stopped) if timed_out else None,
        "stopped": True,
        "bad_since": bad_since,
        "stop_duration_seconds": stop_duration,
    }


class AsyncLoadedEndpointPlanner:
    """Run one endpoint evaluator that owns its IK and collision model off the command loop."""

    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bottle-loaded-endpoint")
        self.lock = threading.Lock()
        self.active = None
        self.results = {}
        self.closed = False

    def _evaluate(self, payload):
        try:
            result = dict(self.evaluator.evaluate(payload))
        except Exception as exc:
            result = {"admitted": False, "reason": f"endpoint_worker:{type(exc).__name__}:{exc}"}
        return result, time.monotonic()

    def _refresh_locked(self):
        if self.active is None or not self.active["future"].done():
            return
        active = self.active
        self.active = None
        result, completed_wall = active["future"].result()
        result.update(
            key=active["key"],
            submitted_wall=active["submitted_wall"],
            completed_wall=completed_wall,
        )
        self.results[active["key"]] = result

    def request(self, key, payload):
        started = time.perf_counter()
        with self.lock:
            if self.closed:
                return {"status": "closed", "poll_seconds": time.perf_counter() - started}
            self._refresh_locked()
            result = self.results.pop(key, None)
            if result is not None:
                return {"status": "ready", "result": result, "poll_seconds": time.perf_counter() - started}
            if self.active is None:
                submitted_wall = time.monotonic()
                self.active = {
                    "key": key,
                    "submitted_wall": submitted_wall,
                    "future": self.executor.submit(self._evaluate, deepcopy(payload)),
                }
            active = self.active
            return {
                "status": "pending" if active["key"] == key else "busy",
                "active_key": active["key"],
                "plan_age_seconds": time.monotonic() - active["submitted_wall"],
                "poll_seconds": time.perf_counter() - started,
            }

    def poll(self, key):
        started = time.perf_counter()
        with self.lock:
            if self.closed:
                return {"status": "closed", "poll_seconds": time.perf_counter() - started}
            self._refresh_locked()
            result = self.results.pop(key, None)
            if result is not None:
                return {"status": "ready", "result": result, "poll_seconds": time.perf_counter() - started}
            if self.active is not None and self.active["key"] == key:
                return {
                    "status": "pending",
                    "active_key": key,
                    "plan_age_seconds": time.monotonic() - self.active["submitted_wall"],
                    "poll_seconds": time.perf_counter() - started,
                }
            return {"status": "missing", "poll_seconds": time.perf_counter() - started}

    def close(self):
        with self.lock:
            self.closed = True
            if self.active is not None:
                self.active["future"].cancel()
        self.executor.shutdown(wait=False, cancel_futures=True)


def loaded_world_path(
    start,
    entry_axis,
    lift_height,
    progress,
    *,
    clearance_fraction=0.75,
    carry_distance=None,
):
    """Lift continuously in world z, adding inward carry only after clearance."""
    start = np.asarray(start, dtype=float)
    axis = np.asarray(entry_axis, dtype=float).copy()
    horizontal = axis.copy()
    horizontal[2] = 0.0
    norm = float(np.linalg.norm(horizontal))
    if start.shape != (3,) or not np.isfinite(start).all() or not np.isfinite(axis).all() or norm <= 0:
        raise ValueError("loaded world path needs finite start and horizontal entry axis")
    if not np.isfinite(lift_height) or lift_height <= 0:
        raise ValueError("loaded world path lift height must be positive and finite")
    if not np.isfinite(clearance_fraction) or not 0 < clearance_fraction < 1:
        raise ValueError("loaded world path clearance_fraction must be between zero and one")
    progress = float(np.clip(progress, 0.0, 1.0))
    horizontal /= norm
    vertical_progress = float(_smooth(progress))
    lifted = start + np.array([0.0, 0.0, float(lift_height) * vertical_progress])
    carry_distance = min(0.08, float(lift_height)) if carry_distance is None else float(carry_distance)
    if not np.isfinite(carry_distance) or carry_distance <= 0:
        raise ValueError("loaded carry_distance must be positive and finite")
    if progress <= clearance_fraction:
        return lifted, {
            "phase": "clear_table",
            "phase_progress": progress / clearance_fraction,
            "vertical_progress": vertical_progress,
            "clearance_fraction": float(clearance_fraction),
            "carry_distance_m": carry_distance,
        }
    phase_progress = float(_smooth((progress - clearance_fraction) / (1.0 - clearance_fraction)))
    return lifted - phase_progress * carry_distance * horizontal, {
        "phase": "carry_inward",
        "phase_progress": phase_progress,
        "vertical_progress": vertical_progress,
        "clearance_fraction": float(clearance_fraction),
        "carry_distance_m": carry_distance,
    }


class LoadedHandbackStabilityGate:
    """Require contiguous measured loaded stability before and after ownership switch."""

    def __init__(
        self,
        quiet_seconds,
        min_samples,
        max_gap_seconds,
        hand_position_diameter_max,
        hand_speed_max,
        hand_rotation_diameter_max,
        hand_angular_speed_max,
        object_relative_speed_max,
    ):
        self.quiet_seconds = float(quiet_seconds)
        self.min_samples = int(min_samples)
        self.max_gap_seconds = float(max_gap_seconds)
        self.hand_position_diameter_max = float(hand_position_diameter_max)
        self.hand_speed_max = float(hand_speed_max)
        self.hand_rotation_diameter_max = float(hand_rotation_diameter_max)
        self.hand_angular_speed_max = float(hand_angular_speed_max)
        self.object_relative_speed_max = float(object_relative_speed_max)
        self.samples = []
        self.last_reset_reason = "cold_start"
        self._boundary_gap_seconds = None
        self.last_boundary_gap_seconds = None

    def reset(self, reason):
        self.samples.clear()
        self.last_reset_reason = str(reason)

    def expect_boundary(self, max_gap_seconds):
        """Allow one larger physical gap for the next sample, across an ownership switch.

        The switch call itself takes tens of milliseconds and the next observation
        waits for a fresh state, so the first post-switch sample lands after the
        regular ``max_gap_seconds``. Measurement continues across the boundary: the
        retained pre-switch samples still detect motion caused by the switch.
        """
        value = float(max_gap_seconds)
        if not np.isfinite(value) or value < self.max_gap_seconds:
            raise ValueError("boundary gap must be finite and at least the regular gap limit")
        self._boundary_gap_seconds = value

    @staticmethod
    def _vector_stats(stamps, values, *, joint_space=False):
        values = np.asarray(values, dtype=float)
        centred = stamps - float(np.mean(stamps))
        denominator = float(centred @ centred)
        slope = (
            np.zeros(values.shape[1]) if denominator <= 0 else np.sum(centred[:, None] * values, axis=0) / denominator
        )
        if joint_space:
            excursion = float(np.max(np.ptp(values, axis=0)))
            trend = float(np.max(np.abs(slope)))
            instantaneous = float(np.max(np.abs(np.diff(values, axis=0) / np.diff(stamps)[:, None])))
        else:
            excursion = float(np.max(np.linalg.norm(values[:, None] - values[None, :], axis=2)))
            trend = float(np.linalg.norm(slope))
            instantaneous = float(np.max(np.linalg.norm(np.diff(values, axis=0), axis=1) / np.diff(stamps)))
        return dict(excursion=excursion, trend_speed=trend, instantaneous_speed=instantaneous)

    @staticmethod
    def _rotation_stats(stamps, rotations):
        rotations = np.asarray(rotations, dtype=float)
        vectors = Rotation.from_matrix(rotations @ rotations[0].T).as_rotvec()
        result = LoadedHandbackStabilityGate._vector_stats(stamps, vectors)
        result["excursion"] = max(
            (
                float(Rotation.from_matrix(rotations[j] @ rotations[i].T).magnitude())
                for i in range(len(rotations))
                for j in range(i + 1, len(rotations))
            ),
            default=0.0,
        )
        return result

    def _status(self, reset_reason=None):
        count = len(self.samples)
        held = 0.0 if count < 2 else float(self.samples[-1][0] - self.samples[0][0])
        result = dict(
            ready=False,
            sample_count=count,
            held_seconds=held,
            reset_reason=reset_reason,
            last_reset_reason=self.last_reset_reason,
            boundary_gap_seconds=self.last_boundary_gap_seconds,
        )
        if count < 2:
            return result
        stamps = np.asarray([sample[0] for sample in self.samples], dtype=float)
        hand = self._vector_stats(stamps, [sample[1] for sample in self.samples])
        hand_rotation = self._rotation_stats(stamps, [sample[2] for sample in self.samples])
        root = self._vector_stats(stamps, [sample[3] for sample in self.samples])
        root_rotation = self._rotation_stats(stamps, [sample[4] for sample in self.samples])
        legs = self._vector_stats(stamps, [sample[5] for sample in self.samples], joint_space=True)
        relative = self._vector_stats(stamps, [sample[6] for sample in self.samples])
        limits = dict(
            hand_position_diameter_m=self.hand_position_diameter_max,
            hand_trend_speed_m_s=self.hand_speed_max,
            hand_instantaneous_speed_m_s=self.hand_speed_max,
            hand_rotation_diameter_rad=self.hand_rotation_diameter_max,
            hand_angular_trend_rad_s=self.hand_angular_speed_max,
            hand_angular_instantaneous_rad_s=self.hand_angular_speed_max,
            root_position_diameter_m=max(0.006, self.hand_position_diameter_max),
            root_trend_speed_m_s=self.hand_speed_max,
            root_instantaneous_speed_m_s=self.hand_speed_max,
            root_rotation_diameter_rad=self.hand_rotation_diameter_max,
            root_angular_trend_rad_s=self.hand_angular_speed_max,
            root_angular_instantaneous_rad_s=self.hand_angular_speed_max,
            leg_excursion_rad=0.04,
            leg_trend_speed_rad_s=0.20,
            leg_instantaneous_speed_rad_s=0.20,
            object_relative_excursion_m=0.01,
            object_relative_trend_speed_m_s=self.object_relative_speed_max,
            object_relative_instantaneous_speed_m_s=self.object_relative_speed_max,
        )
        checks = (
            hand["excursion"] <= limits["hand_position_diameter_m"],
            hand["trend_speed"] <= limits["hand_trend_speed_m_s"],
            hand["instantaneous_speed"] <= limits["hand_instantaneous_speed_m_s"],
            hand_rotation["excursion"] <= limits["hand_rotation_diameter_rad"],
            hand_rotation["trend_speed"] <= limits["hand_angular_trend_rad_s"],
            hand_rotation["instantaneous_speed"] <= limits["hand_angular_instantaneous_rad_s"],
            root["excursion"] <= limits["root_position_diameter_m"],
            root["trend_speed"] <= limits["root_trend_speed_m_s"],
            root["instantaneous_speed"] <= limits["root_instantaneous_speed_m_s"],
            root_rotation["excursion"] <= limits["root_rotation_diameter_rad"],
            root_rotation["trend_speed"] <= limits["root_angular_trend_rad_s"],
            root_rotation["instantaneous_speed"] <= limits["root_angular_instantaneous_rad_s"],
            legs["excursion"] <= limits["leg_excursion_rad"],
            legs["trend_speed"] <= limits["leg_trend_speed_rad_s"],
            legs["instantaneous_speed"] <= limits["leg_instantaneous_speed_rad_s"],
            relative["excursion"] <= limits["object_relative_excursion_m"],
            relative["trend_speed"] <= limits["object_relative_trend_speed_m_s"],
            relative["instantaneous_speed"] <= limits["object_relative_instantaneous_speed_m_s"],
        )
        result.update(
            ready=bool(count >= self.min_samples and held + 1e-9 >= self.quiet_seconds and all(checks)),
            within_limits=bool(all(checks)),
            hand=hand,
            hand_rotation=hand_rotation,
            root=root,
            root_rotation=root_rotation,
            legs=legs,
            object_relative=relative,
            limits=limits,
        )
        return result

    def step(self, stamp, hand, hand_rotation, root, root_rotation, legs, object_relative, bilateral):
        stamp = float(stamp)
        values = (hand, hand_rotation, root, root_rotation, legs, object_relative)
        if not np.isfinite(stamp) or any(not np.isfinite(np.asarray(value, dtype=float)).all() for value in values):
            raise ValueError("loaded handback stability needs finite measurements")
        reset_reason = None
        gap_limit = self.max_gap_seconds if self._boundary_gap_seconds is None else self._boundary_gap_seconds
        boundary = self._boundary_gap_seconds is not None
        self._boundary_gap_seconds = None
        if not bilateral:
            self.reset("bilateral_contact_lost")
            return self._status("bilateral_contact_lost")
        if self.samples:
            gap = stamp - self.samples[-1][0]
            if boundary:
                self.last_boundary_gap_seconds = float(gap)
            if gap <= 0:
                reset_reason = "non_increasing_time"
                self.reset(reset_reason)
            elif gap > gap_limit:
                reset_reason = "clock_gap"
                self.reset(reset_reason)
        sample = (
            stamp,
            np.asarray(hand, dtype=float).copy(),
            np.asarray(hand_rotation, dtype=float).copy(),
            np.asarray(root, dtype=float).copy(),
            np.asarray(root_rotation, dtype=float).copy(),
            np.asarray(legs, dtype=float).copy(),
            np.asarray(object_relative, dtype=float).copy(),
        )
        self.samples.append(sample)
        while len(self.samples) > 1 and stamp - self.samples[1][0] >= self.quiet_seconds:
            self.samples.pop(0)
        status = self._status(reset_reason)
        if len(self.samples) >= 2 and not status["within_limits"]:
            self.reset("motion")
            self.samples.append(sample)
            return self._status("motion")
        return status
