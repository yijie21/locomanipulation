"""Measured, low-speed execution of one statically planned open-box rim grasp.

The static planner may move its floating pelvis while preserving foot anchors.  SONIC's
POSE channel cannot command that translation, so this module recomputes every arm/waist
reference in the *observed* pelvis frame and treats pelvis position as feedback only.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from locomani.backends.sonic.agent_server import HAND_CLOSED, HAND_OPEN, LocomotionMode
from locomani.backends.sonic.pose_preview import PosePreview
from locomani.backends.sonic.primitives import upper_body_from_mj
from locomani.backends.sonic.streamed_arm import PlannerArm, StreamedArm
from locomani.pipeline.execution_record import RoundResult
from locomani.primitives.g1_joints import ARM_MJ
from locomani.primitives.whole_body_ik import WholeBodyIK, plan_low_reach
from locomani.tasks.crate.box_rim_execution_config import (
    BoxRimExecutionConfig,
    execution_config_from_dict,
)
from locomani.tasks.crate.box_rim_geometry import OpenBoxGeometry, RigidPose
from locomani.tasks.crate.box_rim_planner import BoxRimPlan, BoxRimPlanner, RobotState
from locomani.tasks.crate.carry_config import CrateCarryConfig
from locomani.tasks.crate.coordinated_reach import CoordinatedReach, CoordinatedTarget
from locomani.tasks.crate.grasp_region import RimRegion
from locomani.tasks.crate.perception import ApproachTarget, CrateEstimate
from locomani.tasks.crate.visual_approach import VisualApproach
from locomani.tasks.crate.visual_config import CrateVisualConfig

HAND_READY = np.zeros_like(HAND_OPEN)
ARM_INDICES = np.asarray([*ARM_MJ["left"], *ARM_MJ["right"]], dtype=int)


@dataclass(frozen=True)
class _CoordinatedControlConfig:
    visual: CrateVisualConfig
    carry: CrateCarryConfig


def _vector(value, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {length} finite values")
    return result.copy()


def _yaw_quaternion(yaw: float) -> np.ndarray:
    return Rotation.from_euler("z", yaw).as_quat()[[3, 0, 1, 2]]


def _quaternion_angle(a, b) -> float:
    ra = Rotation.from_quat(_vector(a, 4, "quaternion")[[1, 2, 3, 0]])
    rb = Rotation.from_quat(_vector(b, 4, "quaternion")[[1, 2, 3, 0]])
    return float((ra.inv() * rb).magnitude())


class BoxRimRuntime(Protocol):
    """The execution policy's small, testable boundary around live SONIC control."""

    def observe(self) -> dict: ...
    def hold_ready(self) -> None: ...
    def ready_step(self, joints: np.ndarray, hands: np.ndarray) -> None: ...
    def walk_step(self, target_xy: np.ndarray, target_yaw: float, speed: float) -> None: ...
    def stop_walk(self) -> None: ...
    def enter_pose(
        self, acceleration: float, joint_limits: tuple[np.ndarray, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray] | None: ...
    def pose_step(
        self,
        joints: np.ndarray,
        root_quaternion_wxyz: np.ndarray,
        hands: np.ndarray,
        *,
        reference_velocity: np.ndarray | None = None,
    ) -> None: ...
    def safe_stop(self) -> None: ...
    def close(self) -> None: ...


class SonicBoxRimRuntime:
    """Concrete adapter over the existing PlannerArm/StreamedArm POSE interfaces."""

    def __init__(
        self,
        primitives,
        ik,
        *,
        approach_mode="supported_pose",
        initial_reference=None,
        initial_hands=None,
        planner_arm=None,
    ):
        self.p = primitives
        self.ik = ik
        self.approach_mode = approach_mode
        default_hands = HAND_OPEN if approach_mode == "coordinated" else HAND_READY
        self.ready_hands = (
            default_hands.copy() if initial_hands is None else _vector(initial_hands, len(HAND_OPEN), "initial hands")
        )
        if planner_arm is not None and initial_reference is not None:
            raise ValueError("planner_arm and initial_reference are mutually exclusive")
        self.planner_arm = (
            planner_arm
            if planner_arm is not None
            else PlannerArm(primitives, "right", send_vel=False, ik=ik, left_guard=False)
        )
        if planner_arm is not None:
            if self.planner_arm.q is None:
                raise ValueError("planner_arm must already be started")
        elif initial_reference is None:
            self.planner_arm.start()
        else:
            reference = _vector(initial_reference, 29, "initial reference")
            self.planner_arm.start(
                upper_reference=upper_body_from_mj(reference),
                settle_seconds=0.0,
            )
            self.planner_arm.q = reference
            self.planner_arm.hands = {side: self.ready_hands.copy() for side in ("left", "right")}
            self.planner_arm.mode = LocomotionMode.IDLE
            self.planner_arm._push(
                self.planner_arm.q,
                reference_velocity=np.zeros(29, dtype=float),
            )
        self.pose_arm: StreamedArm | None = None

    def observe(self) -> dict:
        return self.p.obs()

    def hold_ready(self) -> None:
        observed = self.observe()
        # Keep the stable PlannerArm takeover reference fixed.  Feeding noisy
        # measured joints back through finite-difference velocity destabilises the
        # standing controller; an explicit zero velocity means a deliberate hold,
        # not an assumed achieved pose.
        self.planner_arm.hands = {side: self.ready_hands.copy() for side in ("left", "right")}
        self.planner_arm.mode = LocomotionMode.IDLE
        self.planner_arm._push(
            self.planner_arm.q,
            observed=observed,
            reference_velocity=np.zeros(29, dtype=float),
        )

    def ready_step(self, joints, hands) -> None:
        self.planner_arm.q = np.asarray(joints, dtype=float).copy()
        self.ready_hands = _vector(hands, len(HAND_OPEN), "ready hands")
        self.planner_arm.hands = {side: self.ready_hands.copy() for side in ("left", "right")}
        self.planner_arm.mode = LocomotionMode.IDLE
        self.planner_arm._push(self.planner_arm.q, reference_velocity=np.zeros(29, dtype=float))

    def walk_step(self, target_xy, target_yaw, speed) -> None:
        state = self.observe()
        delta = np.asarray(target_xy, dtype=float) - np.asarray(state["pelvis"]["pos"][:2], dtype=float)
        direction = delta / max(float(np.linalg.norm(delta)), 1e-9)
        self.planner_arm.hands = {side: self.ready_hands.copy() for side in ("left", "right")}
        self.planner_arm.mode = LocomotionMode.SLOW_WALK
        self.planner_arm._face = (math.cos(target_yaw), math.sin(target_yaw), 0.0)
        self.planner_arm._push(
            self.planner_arm.q,
            observed=state,
            movement=(float(direction[0]), float(direction[1]), 0.0),
            speed=float(speed),
        )

    def stop_walk(self) -> None:
        self.planner_arm.mode = LocomotionMode.IDLE
        self.planner_arm._push(self.planner_arm.q, observed=self.observe(), speed=-1.0)

    def enter_pose(self, acceleration, joint_limits) -> None:
        if self.approach_mode == "planner_arms":
            self.stop_walk()
            return None
        if self.pose_arm is not None:
            return self.pose_arm.q.copy(), self.pose_arm.root_quat.copy()
        self.stop_walk()
        preview = PosePreview(acceleration=acceleration, joint_limits=joint_limits)
        arm = StreamedArm(self.p, "right", ik=self.ik, left_guard=False, preview=preview)
        arm.hands = {side: self.ready_hands.copy() for side in ("left", "right")}
        arm.start(hold_planner_pose=True)
        self.pose_arm = arm
        return arm.q.copy(), arm.root_quat.copy()

    def pose_step(
        self,
        joints,
        root_quaternion_wxyz,
        hands,
        *,
        reference_velocity=None,
    ) -> None:
        if self.approach_mode == "planner_arms":
            self.ready_step(joints, hands)
            return
        if self.pose_arm is None:
            raise RuntimeError("POSE control has not started")
        self.pose_arm.q = np.asarray(joints, dtype=float).copy()
        self.pose_arm.root_quat = np.asarray(root_quaternion_wxyz, dtype=float).copy()
        hand = np.asarray(hands, dtype=float).copy()
        self.pose_arm.hands = {"left": hand.copy(), "right": hand.copy()}
        self.pose_arm._push(self.pose_arm.q, reference_velocity=reference_velocity)

    def safe_stop(self) -> None:
        if self.pose_arm is not None:
            self.pose_arm.hold(0.3)
        else:
            self.stop_walk()

    def close(self) -> None:
        try:
            self.safe_stop()
        finally:
            self.p.close()


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass
class _Journal:
    started_wall: float = field(default_factory=time.time)
    phase: str = "admission"
    events: list[dict] = field(default_factory=list)
    traces: list[dict] = field(default_factory=list)
    retries: int = 0
    travel_m: float = 0.0
    selected_plan: dict | None = None
    planning_attempts: list[dict] = field(default_factory=list)
    actual_stop_positions: list[list[float]] = field(default_factory=list)

    def event(self, phase_name: str, kind: str, **values) -> None:
        if "phase" in values:
            values["reported_phase"] = values.pop("phase")
        self.phase = phase_name
        self.events.append(_json_safe({"phase": phase_name, "kind": kind, **values}))

    def trace(self, state, **values) -> None:
        obj = values.pop("object", None)
        palms = {
            side: {
                "position": list(state["palms"][side]["pos"]),
                "quaternion_wxyz": list(state["palms"][side]["quat"]),
                "hand_q": list(state["palms"][side].get("hand_q", [])),
            }
            for side in ("left", "right")
        }
        objects = {
            key: {
                "position": list(value.get("pos", [])),
                "quaternion_wxyz": list(value.get("quat", [])),
                "touching": list(value.get("touching", [])),
                "contact_links": list(value.get("touch_links", [])),
            }
            for key, value in state["objects"].items()
        }
        row = {
            "phase": self.phase,
            "seq": state.get("seq"),
            "simulation_time": state.get("t"),
            "wall": state.get("wall"),
            "pelvis_position": list(state["pelvis"]["pos"]),
            "pelvis_quaternion_wxyz": list(state["pelvis"]["quat"]),
            "pelvis_velocity_mps": list(state["pelvis"].get("vel", [])),
            "joints": np.asarray(state["joints"]).tolist(),
            "palms": palms,
            "objects": objects,
            **values,
        }
        if obj is not None:
            row["object"] = obj
        self.traces.append(_json_safe(row))

    def snapshot(self) -> dict:
        return {
            "schema_version": 1,
            "phase": self.phase,
            "started_wall": self.started_wall,
            "events": self.events,
            "retries": self.retries,
            "travel_m": self.travel_m,
            "selected_plan": self.selected_plan,
            "planning_attempts": self.planning_attempts,
            "actual_stop_positions": self.actual_stop_positions,
            "trace": self.traces,
        }


class _Failure(RuntimeError):
    def __init__(self, reason: str, **diagnostics):
        super().__init__(reason)
        self.reason = reason
        self.diagnostics = diagnostics


class BoxRimExecutor:
    """Run admission, optional bounded reposition, reach, close and probe."""

    def __init__(
        self,
        planner: BoxRimPlanner,
        box: OpenBoxGeometry,
        runtime: BoxRimRuntime,
        target_key: str,
        config: BoxRimExecutionConfig | None = None,
    ):
        self.planner = planner
        self.box = box
        self.runtime = runtime
        self.target_key = str(target_key)
        self.config = BoxRimExecutionConfig() if config is None else config
        self.journal = _Journal()
        self._last_seq = None
        self._last_sim_time = None
        self._last_wall_progress = time.time()
        self._reposition_last_xy: np.ndarray | None = None
        self._low_reach_posture: np.ndarray | None = None
        self._low_reach_reference_joints: np.ndarray | None = None
        self._low_reach_root_quaternion: np.ndarray | None = None
        self._low_reach_execution_baseline: dict | None = None

    def execute(self) -> RoundResult:
        try:
            state = self._settle("admission")
            if self.config.approach_mode == "coordinated":
                self._execute_coordinated(state)
                self.journal.event("complete", "coordinated_lift_accepted")
                return RoundResult(True, "probe_lift_confirmed", False, self.journal.snapshot())
            if self.config.approach_mode == "planner_arms":
                self._refresh_box_pose(state)
                clearance = self._arm_only_escape_waypoint(state, {"step": 1})
                targets, _ = self._tool_geometry(state, state["joints"])
                initial_contacts = self._admit_initial_escape(state, clearance, targets)
                baseline = state
                state = self._move_initial_escape(state, clearance, targets, initial_contacts)
                self._check_observed_waypoint(baseline, state, clearance)
                state = self._settle("ready_settle")
            plan, state = self._plan_with_reposition(state)
            self.journal.selected_plan = plan.to_dict()
            self._execute_plan(plan, state)
            self.journal.event("complete", "probe_accepted")
            return RoundResult(True, "probe_lift_confirmed", False, self.journal.snapshot())
        except _Failure as failure:
            self.journal.event(self.journal.phase, "failure", reason=failure.reason, **failure.diagnostics)
            try:
                self.runtime.safe_stop()
            except Exception as stop_error:  # noqa: BLE001 - retain primary physical failure
                self.journal.event(self.journal.phase, "safe_stop_error", error=repr(stop_error))
            return RoundResult(False, failure.reason, False, self.journal.snapshot())
        except Exception as error:  # noqa: BLE001 - retain the phase journal for integration failures
            self.journal.event(
                self.journal.phase,
                "runtime_error",
                error=f"{type(error).__name__}: {error}",
            )
            try:
                self.runtime.safe_stop()
            except Exception as stop_error:  # noqa: BLE001
                self.journal.event(self.journal.phase, "safe_stop_error", error=repr(stop_error))
            return RoundResult(False, "runtime_error", False, self.journal.snapshot())

    def _observe(self, *, require_advance: bool = True) -> dict:
        state = self.runtime.observe()
        required = ("t", "wall", "joints", "pelvis", "palms", "objects", "fallen")
        missing = [name for name in required if name not in state]
        if missing:
            raise _Failure("invalid_observation", missing=missing)
        now = time.time()
        if now - float(state["wall"]) > self.config.observation_max_age_s:
            raise _Failure("stale_observation", age_s=now - float(state["wall"]))
        sim_time = float(state["t"])
        seq = state.get("seq", sim_time)
        advanced = self._last_sim_time is None or sim_time > float(self._last_sim_time)
        if advanced:
            self._last_wall_progress = now
        elif require_advance and now - self._last_wall_progress > self.config.watchdog_wall_s:
            raise _Failure("observation_watchdog_timeout", seq=seq, simulation_time=sim_time)
        if self._last_sim_time is not None and sim_time < float(self._last_sim_time) - 1e-9:
            raise _Failure("simulation_clock_reversed")
        self._last_seq, self._last_sim_time = seq, sim_time
        joints = _vector(state["joints"], 29, "observation.joints")
        _vector(state["pelvis"]["pos"], 3, "observation.pelvis.pos")
        _vector(state["pelvis"]["quat"], 4, "observation.pelvis.quat")
        if state["fallen"]:
            raise _Failure("robot_fallen")
        if self.target_key not in state["objects"]:
            raise _Failure("target_missing", target_key=self.target_key)
        state = dict(state)
        state["joints"] = joints
        return state

    def _robot_state(self, state) -> RobotState:
        return RobotState(state["joints"], state["pelvis"]["pos"], state["pelvis"]["quat"])

    def _observed_foot_positions(self, state) -> np.ndarray:
        ik = self.planner.ik
        model, data, mj = ik.model, ik.data, ik._mj
        data.qpos[:] = model.qpos0
        data.qpos[ik.base_qadr : ik.base_qadr + 3] = state["pelvis"]["pos"]
        data.qpos[ik.base_qadr + 3 : ik.base_qadr + 7] = state["pelvis"]["quat"]
        data.qpos[ik.qadr] = state["joints"]
        mj.mj_kinematics(model, data)
        feet = [model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
        return data.xpos[feet].copy()

    def _coordinated_control_config(self) -> _CoordinatedControlConfig:
        """Adapt explicit box-execution limits to the original coordinated controller."""
        waist_dofs = self.planner.options.waist_dofs
        waist_pitch_max = min(0.52, float(self.planner.options.waist_pitch_max_rad))
        solver_lo = np.asarray(self.planner.bimanual.lo, dtype=float)
        solver_hi = np.asarray(self.planner.bimanual.hi, dtype=float)
        frozen = np.flatnonzero(solver_hi <= solver_lo)
        permitted_frozen = {12, 13} if waist_dofs == "pitch" else set()
        incompatible = [int(index) for index in frozen if int(index) not in permitted_frozen]
        if incompatible:
            raise _Failure(
                "coordinated_solver_limits_incompatible",
                waist_dofs=waist_dofs,
                non_strict_joint_bounds=incompatible,
            )
        visual = CrateVisualConfig(
            pose_source="assisted",
            waist_pitch_max=waist_pitch_max,
            palm_pitch_deg=60.0,
            walk_speed=self.config.coordinated_walk_speed_mps,
            walk_speed_min=self.config.coordinated_walk_speed_mps,
            walk_brake_margin=self.config.coordinated_walk_brake_margin_m,
            settle_seconds=self.config.coordinated_settle_s,
            stand_off=self.config.coordinated_stand_off_m,
            stand_lateral_tolerance=self.config.coordinated_stand_lateral_tolerance_m,
            stand_depth_tolerance=0.04,
            stand_depth_retries=2,
            pregrasp_above=self.config.coordinated_pregrasp_above_m,
            approach_speed=self.config.coordinated_approach_speed_mps,
            joint_speed=self.config.maximum_joint_speed_rps,
            joint_acceleration=self.config.maximum_joint_acceleration_rps2,
            tracking_gain=self.config.coordinated_tracking_gain,
            tracking_cap=self.config.coordinated_tracking_cap_m,
            align_tolerance=min(0.018, self.config.align_position_tolerance_m),
            close_seconds=self.config.close_s,
            probe_height=self.config.probe_lift_m,
            max_tilt_deg=math.degrees(self.config.maximum_object_tilt_rad),
        )
        carry = CrateCarryConfig(
            reach_control="whole_body",
            waist_dofs=waist_dofs,
            joint_speed=self.config.maximum_joint_speed_rps,
            joint_acceleration=self.config.maximum_joint_acceleration_rps2,
            waist_pitch_max=waist_pitch_max,
            pelvis_height_min=self.config.minimum_pelvis_height_m,
        )
        return _CoordinatedControlConfig(visual=visual, carry=carry)

    def _coordinated_target(self, plan: BoxRimPlan) -> CoordinatedTarget:
        if plan.targets_world_m is None or plan.selection is None:
            raise _Failure("coordinated_plan_missing_targets")
        if self.box.forbidden_intervals:
            raise _Failure("coordinated_forbidden_rim_intervals_unsupported")
        rim_pair = plan.selection.get("rim_pair")
        if rim_pair == "x":
            capture_rotation_local = np.eye(3)
            tangent_axis = 1
        elif rim_pair == "y":
            capture_rotation_local = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            tangent_axis = 0
        else:
            raise _Failure("coordinated_plan_unknown_rim_pair", rim_pair=rim_pair)
        local_targets = (
            np.asarray(plan.targets_world_m, dtype=float) - self.box.pose.position
        ) @ self.box.pose.rotation
        tangent_limit = (
            float(self.box.size_m[tangent_axis]) / 2 - self.box.corner_clearance_m - self.box.wall_thickness_m
        )
        tangent_coordinates = local_targets[:, tangent_axis]
        usable_half_length = float(
            np.min(np.minimum(tangent_limit - tangent_coordinates, tangent_coordinates + tangent_limit))
        )
        if usable_half_length <= 0:
            raise _Failure("coordinated_capture_region_empty", rim_pair=rim_pair)
        return CoordinatedTarget(
            object_key=self.target_key,
            grasp_points_local=tuple(map(tuple, local_targets)),
            capture_rotation_local=capture_rotation_local,
            region=RimRegion(half_length=usable_half_length),
            information_source="simulation_truth",
            preserve_point_order=True,
        )

    @staticmethod
    def _coordinated_estimate(state, target_key: str) -> CrateEstimate:
        base = Rotation.from_quat(np.asarray(state["pelvis"]["quat"])[[1, 2, 3, 0]]).as_matrix()
        origin = np.asarray(state["pelvis"]["pos"], dtype=float)
        observed = state["objects"][target_key]
        object_rotation = Rotation.from_quat(np.asarray(observed["quat"])[[1, 2, 3, 0]]).as_matrix()
        return CrateEstimate(
            base.T @ (np.asarray(observed["pos"], dtype=float) - origin),
            base.T @ object_rotation,
            float(state["wall"]),
            None,
            {"source": "simulation_truth"},
        )

    def _coordinated_stance(self, state, approach: VisualApproach | None = None):
        control = self._coordinated_control_config()
        approach = VisualApproach(control.visual) if approach is None else approach
        initial_object = state["objects"][self.target_key]
        initial_object_position = np.asarray(initial_object["pos"], dtype=float).copy()
        initial_object_quaternion = np.asarray(initial_object["quat"], dtype=float).copy()
        self._reposition_last_xy = np.asarray(state["pelvis"]["pos"][:2], dtype=float).copy()
        self.journal.event(
            "approach",
            "measured_stance_feedback_started",
            stand_off_m=control.visual.stand_off,
            information_source="simulation_truth",
        )
        while True:
            self._accumulate_reposition_travel(state)
            observed_object = state["objects"][self.target_key]
            box_motion = float(
                np.linalg.norm(np.asarray(observed_object["pos"], dtype=float) - initial_object_position)
            )
            box_tilt = _quaternion_angle(observed_object["quat"], initial_object_quaternion)
            if state.get("furniture_touch"):
                self.runtime.stop_walk()
                raise _Failure("approach_furniture_collision", furniture_touch=state["furniture_touch"])
            if box_motion > self.config.maximum_box_motion_before_close_m:
                self.runtime.stop_walk()
                raise _Failure("box_moved_during_approach", displacement_m=box_motion)
            if box_tilt > self.config.maximum_box_tilt_before_close_rad:
                self.runtime.stop_walk()
                raise _Failure("box_tilted_during_approach", tilt_rad=box_tilt)
            estimate = self._coordinated_estimate(state, self.target_key)
            target = ApproachTarget(estimate.position.copy(), estimate.timestamp, "simulation_truth")
            command = approach.step(
                float(state["wall"]),
                target,
                state["pelvis"]["quat"],
                grasp_pose_available=True,
                motion_time=float(state["t"]),
                base_velocity=state["pelvis"].get("vel", [0.0, 0.0, 0.0]),
            )
            stance_error = np.array(
                [estimate.position[0] - control.visual.stand_off, estimate.position[1]], dtype=float
            )
            self.journal.trace(
                state,
                stance_target_m=[control.visual.stand_off, 0.0],
                stance_measured_m=estimate.position[:2].tolist(),
                stance_error_m=stance_error.tolist(),
                stance_phase=command.phase,
                stance_speed_command=float(command.speed),
                stance_travel_m=self.journal.travel_m,
                reference_hands=HAND_OPEN.tolist(),
                information_source="simulation_truth",
            )
            if command.reason:
                self.runtime.stop_walk()
                raise _Failure(command.reason, stance_attempts=approach.stand_attempts)
            if command.ready:
                self.runtime.stop_walk()
                settled = self._settle("approach_settle")
                self._accumulate_reposition_travel(settled)
                self.journal.event(
                    "approach_settle",
                    "measured_stance_ready",
                    attempts=approach.stand_attempts,
                    final_error_m=stance_error.tolist(),
                    path_length_m=self.journal.travel_m,
                )
                return settled, approach
            if command.speed <= 0:
                self.runtime.stop_walk()
                state = self._observe()
                continue
            yaw = float(state["pelvis"].get("yaw", 0.0))
            world_direction = Rotation.from_euler("z", yaw).apply(command.direction)
            target_xy = np.asarray(state["pelvis"]["pos"][:2], dtype=float) + world_direction[:2]
            self.runtime.walk_step(target_xy, yaw, command.speed)
            state = self._observe()

    def _execute_coordinated(self, state) -> None:
        """Drive one continuous whole-body reference from measured settled state."""
        state, approach = self._coordinated_stance(state)
        control = self._coordinated_control_config()

        # Match the original assisted chain: a failed whole-body admission asks
        # VisualApproach for a bounded closer stance before POSE takes ownership.
        while True:
            self._refresh_box_pose(state)
            plan, state = self._plan_with_keepalive(state)
            if not plan.static_valid:
                if approach.retry_closer():
                    state, approach = self._coordinated_stance(state, approach)
                    continue
                raise _Failure("no_safe_plan", planner_reason=plan.reason)
            target = self._coordinated_target(plan)
            estimate = self._coordinated_estimate(state, self.target_key)
            openings = np.array(
                [
                    2 * (np.mean(np.asarray(state["palms"][side]["hand_q"], dtype=float)[:2]) + 0.023)
                    for side in ("left", "right")
                ]
            )
            probe = CoordinatedReach(
                self.planner.bimanual,
                control,
                state,
                reference_joints=state["joints"],
                reference_quaternion=state["pelvis"]["quat"],
                target=target,
            )
            admission = probe.step(float(state["wall"]), estimate, state, openings, motion_time=float(state["t"]))
            self.journal.event(
                "planning",
                "coordinated_admission",
                accepted=not admission.finished,
                reason=admission.reason,
                diagnostics={
                    key: np.asarray(value).tolist()
                    for key, value in probe.diagnostics.items()
                    if key in ("admission_hand_error", "admission_foot_error", "admission_com_shift")
                },
            )
            if admission.reason == "unsupported_assisted_reach" and approach.retry_closer():
                state, approach = self._coordinated_stance(state, approach)
                continue
            if admission.finished:
                raise _Failure(admission.reason or "coordinated_admission_failed")
            break

        self.journal.selected_plan = plan.to_dict()
        handoff = self.runtime.enter_pose(
            self.config.maximum_joint_acceleration_rps2,
            (self.planner.bimanual.lo, self.planner.bimanual.hi),
        )
        if handoff is None:
            raise _Failure("missing_coordinated_pose_handoff")
        state = self._observe()
        reference_joints = np.asarray(handoff[0], dtype=float).copy()
        reference_root = np.asarray(handoff[1], dtype=float).copy()
        controller = CoordinatedReach(
            self.planner.bimanual,
            control,
            state,
            reference_joints=reference_joints,
            reference_quaternion=reference_root,
            target=target,
        )
        self.journal.event(
            "coordinated_reach",
            "controller_started",
            rim_pair=plan.selection["rim_pair"],
            local_targets_m=np.asarray(target.grasp_points_local).tolist(),
            capture_rotation_local=target.capture_rotation_local.tolist(),
            reference_joints=reference_joints.tolist(),
            reference_root_quaternion_wxyz=reference_root.tolist(),
            information_source=target.information_source,
        )
        initial_object = state["objects"][self.target_key]
        initial_object_position = np.asarray(initial_object["pos"], dtype=float).copy()
        initial_object_quaternion = np.asarray(initial_object["quat"], dtype=float).copy()
        foot_anchors = self._observed_foot_positions(state)
        closed = np.zeros_like(HAND_OPEN)
        closed[:2] = -0.023
        capture_was_ready = False
        bilateral_since = None

        while True:
            estimate = self._coordinated_estimate(state, self.target_key)
            openings = np.array(
                [
                    2 * (np.mean(np.asarray(state["palms"][side]["hand_q"], dtype=float)[:2]) + 0.023)
                    for side in ("left", "right")
                ]
            )
            command = controller.step(
                float(state["wall"]),
                estimate,
                state,
                openings,
                motion_time=float(state["t"]),
            )
            hands = HAND_OPEN + command.closure * (closed - HAND_OPEN)
            diagnostics = controller.diagnostics
            capture_ready = bool(diagnostics.get("capture_ready", False))
            capture_was_ready = capture_was_ready or capture_ready
            if command.closure > 0 and not capture_was_ready:
                raise _Failure("close_without_measured_capture")
            if command.phase in ("observe", "reach") and not np.allclose(hands, HAND_OPEN):
                raise _Failure("hands_not_open_before_capture")

            observed_object = state["objects"][self.target_key]
            object_motion = float(
                np.linalg.norm(np.asarray(observed_object["pos"], dtype=float) - initial_object_position)
            )
            object_tilt = _quaternion_angle(observed_object["quat"], initial_object_quaternion)
            if command.closure <= 0 and object_motion > self.config.maximum_box_motion_before_close_m:
                raise _Failure("box_moved_before_capture", displacement_m=object_motion)
            if command.closure <= 0 and object_tilt > self.config.maximum_box_tilt_before_close_rad:
                raise _Failure("box_tilted_before_capture", tilt_rad=object_tilt)

            targets = np.asarray(diagnostics.get("observed_grips"), dtype=float)
            if targets.shape != (2, 3):
                raise _Failure("coordinated_targets_unavailable")
            current_box = OpenBoxGeometry(
                self.box.size_m,
                self.box.wall_thickness_m,
                RigidPose(observed_object["pos"], observed_object["quat"]),
                local_com_m=self.box.local_com_m,
                corner_clearance_m=self.box.corner_clearance_m,
                forbidden_intervals=self.box.forbidden_intervals,
            )
            measured_allowed, measured_rejected = self._collision_reasons(
                state,
                state["joints"],
                targets,
                box=current_box,
                gripper_q=self._observed_gripper_q(state),
            )
            reference_allowed, reference_rejected = self._collision_reasons(
                state,
                command.joints,
                targets,
                box=current_box,
                gripper_q=self._commanded_gripper_q(hands),
                position=controller.x[:3],
                quaternion_wxyz=command.root_quaternion,
                # The reference is a plan the controller tracks with centimetre lag, not the body:
                # entering along the side walls the finger links overlap the wall by up to its
                # thickness in the plan while the measured hands stay clear (2026-09-22). The
                # measured check above keeps the real penetration limit.
                max_target_penetration_m=current_box.wall_thickness_m,
            )
            if measured_rejected or reference_rejected or state.get("furniture_touch"):
                raise _Failure(
                    "coordinated_collision",
                    measured_contacts=measured_rejected,
                    reference_contacts=reference_rejected,
                    furniture_touch=state.get("furniture_touch", []),
                )

            observed_foot_errors = np.linalg.norm(self._observed_foot_positions(state) - foot_anchors, axis=1)
            reference_foot_errors = np.asarray(diagnostics.get("foot_error", []), dtype=float)

            touching = set(observed_object.get("touching", []))
            if touching == {"left", "right"}:
                bilateral_since = float(state["t"]) if bilateral_since is None else bilateral_since
            else:
                bilateral_since = None
            if command.phase in ("lift", "stand", "hold"):
                if touching != {"left", "right"}:
                    raise _Failure("lost_bilateral_contact_during_lift", touching=sorted(touching))
                if float(state["t"]) - bilateral_since < self.config.contact_hold_s:
                    raise _Failure("bilateral_contact_not_held_before_lift")
                attachment = diagnostics.get("attachment_error")
                if attachment is not None and float(attachment) > self.config.maximum_slip_m:
                    raise _Failure("grasp_slip", slip_m=float(attachment))

            self.runtime.pose_step(
                command.joints,
                command.root_quaternion,
                hands,
                reference_velocity=command.velocity,
            )
            observed_tools, _ = self._tool_geometry(state, state["joints"])
            self.journal.trace(
                state,
                reference_joints=np.asarray(command.joints).tolist(),
                reference_root_quaternion_wxyz=np.asarray(command.root_quaternion).tolist(),
                reference_hands=np.asarray(hands).tolist(),
                closure=float(command.closure),
                capture_ready=capture_ready,
                capture_errors=np.asarray(diagnostics.get("capture_error", [])).tolist(),
                observed_tools_world_m=observed_tools.tolist(),
                targets_world_m=targets.tolist(),
                observed_root_position_m=list(state["pelvis"]["pos"]),
                observed_root_quaternion_wxyz=list(state["pelvis"]["quat"]),
                observed_foot_error_m=observed_foot_errors.tolist(),
                reference_foot_error_m=reference_foot_errors.tolist(),
                measured_allowed_contacts=measured_allowed,
                reference_allowed_contacts=reference_allowed,
                object_displacement_m=object_motion,
                object_tilt_rad=object_tilt,
                touching=sorted(touching),
                phase=command.phase,
                information_source=target.information_source,
            )
            if command.finished:
                if command.reason not in ("", "completed"):
                    raise _Failure(command.reason, controller_diagnostics=diagnostics)
                return
            state = self._observe()

    def _is_upright(self, state) -> bool:
        rotation = Rotation.from_quat(np.asarray(state["pelvis"]["quat"])[[1, 2, 3, 0]])
        tilt = float(np.arccos(np.clip(rotation.as_matrix()[2, 2], -1.0, 1.0)))
        speed = float(np.linalg.norm(np.asarray(state["pelvis"].get("vel", [math.inf] * 3), dtype=float)))
        return (
            tilt <= math.radians(self.config.upright_tolerance_deg)
            and float(state["pelvis"]["pos"][2]) >= self.config.minimum_pelvis_height_m
            and speed <= self.config.settle_linear_speed_mps
        )

    def _settle(self, phase: str) -> dict:
        self.journal.event(phase, "settle_started")
        first = self._observe(require_advance=False)
        if phase == "reposition_settle":
            self._accumulate_reposition_travel(first)
        began, quiet_since = float(first["t"]), None
        state = first
        while float(state["t"]) - began <= self.config.settle_timeout_s:
            self.runtime.hold_ready()
            state = self._observe()
            if phase == "reposition_settle":
                self._accumulate_reposition_travel(state)
            upright = self._is_upright(state)
            quiet_since = float(state["t"]) if upright and quiet_since is None else quiet_since
            if not upright:
                quiet_since = None
            quat = np.asarray(state["pelvis"]["quat"], dtype=float)
            tilt = float(
                np.arccos(
                    np.clip(
                        Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()[2, 2],
                        -1.0,
                        1.0,
                    )
                )
            )
            self.journal.trace(
                state,
                settled_sample=upright,
                pelvis_speed_mps=float(np.linalg.norm(state["pelvis"].get("vel", [math.inf] * 3))),
                pelvis_tilt_rad=tilt,
            )
            if quiet_since is not None and float(state["t"]) - quiet_since >= self.config.settle_s:
                self.journal.event(phase, "settled", simulation_time=float(state["t"]))
                self.journal.trace(state)
                return state
        raise _Failure("settle_timeout", phase=phase)

    def _plan_with_keepalive(self, state) -> tuple[BoxRimPlan, dict]:
        # Static IK is CPU-bound.  Keep the supported planner channel alive while it
        # runs, then reject the result unless state advanced after the submitted
        # snapshot.  One planner instance is used by exactly one future at a time.
        self.runtime.hold_ready()
        before = self._observe()
        self._refresh_box_pose(before)
        self.journal.event(
            self.journal.phase,
            "planning_snapshot",
            robot_state={
                "joints": np.asarray(before["joints"]).tolist(),
                "pelvis_position": list(before["pelvis"]["pos"]),
                "pelvis_quaternion_wxyz": list(before["pelvis"]["quat"]),
            },
            box_pose={
                "position": self.box.pose.position.tolist(),
                "quaternion_wxyz": self.box.pose.quaternion_wxyz.tolist(),
            },
        )
        started_wall = time.time()
        deadline = time.monotonic() + self.config.planning_timeout_s
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="box-rim-plan") as pool:
            future = pool.submit(
                self.planner.plan,
                self._robot_state(before),
                deadline_monotonic=deadline,
                support_contact_policy=(
                    "diagnostic_floor" if self.config.approach_mode == "coordinated" else "bounded_growth"
                ),
            )
            while not future.done():
                self.runtime.hold_ready()
                self._observe()
            plan = future.result()
        after = self._observe()
        if plan.reason == "planning_deadline_exceeded":
            raise _Failure("planning_timeout", wall_seconds=time.time() - started_wall)
        if after.get("seq", after["t"]) == before.get("seq", before["t"]) and after["t"] <= before["t"]:
            raise _Failure("state_did_not_advance_after_planning")
        self.journal.planning_attempts.append(
            {
                "kind": "measured_replan",
                "robot_state": {
                    "joints": np.asarray(before["joints"]).tolist(),
                    "pelvis_position": list(before["pelvis"]["pos"]),
                    "pelvis_quaternion_wxyz": list(before["pelvis"]["quat"]),
                },
                "box_pose": {
                    "position": self.box.pose.position.tolist(),
                    "quaternion_wxyz": self.box.pose.quaternion_wxyz.tolist(),
                },
                "plan": plan.to_dict(),
            }
        )
        return plan, after

    def _refresh_box_pose(self, state) -> None:
        observed = state["objects"][self.target_key]
        self.box = OpenBoxGeometry(
            self.box.size_m,
            self.box.wall_thickness_m,
            RigidPose(observed["pos"], observed["quat"]),
            local_com_m=self.box.local_com_m,
            corner_clearance_m=self.box.corner_clearance_m,
            forbidden_intervals=self.box.forbidden_intervals,
        )
        self.planner.box = self.box

    def _plan_with_reposition(self, state) -> tuple[BoxRimPlan, dict]:
        self.journal.event("planning", "plan_started")
        plan, state = self._plan_with_keepalive(state)
        if plan.static_valid:
            self.journal.event("planning", "plan_selected", candidate=plan.selected_candidate)
            return plan, state
        if not self.config.enable_reposition or self.config.reposition_retries == 0:
            raise _Failure("no_safe_reposition", planner_reason=plan.reason)
        for retry in range(self.config.reposition_retries):
            candidate = self._choose_reposition(state)
            if candidate is None:
                raise _Failure("no_safe_reposition", planner_reason=plan.reason)
            target_xy, target_yaw, preview = candidate
            self.journal.retries += 1
            self.journal.event(
                "reposition",
                "candidate_selected",
                retry=retry + 1,
                target_xy=target_xy.tolist(),
                target_yaw=target_yaw,
                preview_candidate=preview.selected_candidate,
            )
            state = self._walk_to(target_xy, target_yaw)
            state = self._settle("reposition_settle")
            settled_position = list(state["pelvis"]["pos"])
            self.journal.actual_stop_positions.append(settled_position)
            self.journal.event(
                "reposition_settle",
                "actual_settled_stop",
                position=settled_position,
                yaw=float(state["pelvis"].get("yaw", 0.0)),
                path_length_m=self.journal.travel_m,
            )
            plan, state = self._plan_with_keepalive(state)
            if plan.static_valid:
                self.journal.event("planning", "plan_selected_after_reposition", candidate=plan.selected_candidate)
                return plan, state
        raise _Failure("no_plan_after_reposition", planner_reason=plan.reason)

    def _choose_reposition(self, state):
        origin = np.asarray(state["pelvis"]["pos"], dtype=float)
        yaw = float(
            state["pelvis"].get(
                "yaw", Rotation.from_quat(np.asarray(state["pelvis"]["quat"])[[1, 2, 3, 0]]).as_euler("zyx")[0]
            )
        )
        rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        toward_box = self.box.pose.position[:2] - origin[:2]
        box_distance = float(np.linalg.norm(toward_box))
        candidates = []
        if box_distance > 1e-9:
            direction = toward_box / box_distance
            face_yaw = math.atan2(direction[1], direction[0])
            yaw_error = math.atan2(math.sin(face_yaw - yaw), math.cos(face_yaw - yaw))
            target_yaw = yaw + float(
                np.clip(
                    yaw_error,
                    -self.config.maximum_reposition_yaw_change_rad,
                    self.config.maximum_reposition_yaw_change_rad,
                )
            )
            step = max(
                float(np.linalg.norm(np.asarray(candidate[:2], dtype=float)))
                for candidate in self.config.reposition_candidates
            )
            for distance in np.arange(
                step,
                self.config.maximum_total_reposition_m + 1e-12,
                step,
            ):
                candidates.append(
                    (
                        origin[:2] + float(distance) * direction,
                        target_yaw,
                        "toward_box",
                    )
                )
        for forward, lateral, dyaw in self.config.reposition_candidates:
            candidates.append(
                (
                    origin[:2] + rotation @ np.array([forward, lateral]),
                    yaw + dyaw,
                    "body_offset",
                )
            )
        for target_xy, target_yaw, strategy in candidates:
            hypothetical = RobotState(
                state["joints"],
                [target_xy[0], target_xy[1], origin[2]],
                _yaw_quaternion(target_yaw),
            )
            started_wall = time.time()
            deadline = time.monotonic() + self.config.planning_timeout_s
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="box-rim-preview") as pool:
                future = pool.submit(
                    self.planner.plan,
                    hypothetical,
                    deadline_monotonic=deadline,
                )
                while not future.done():
                    self.runtime.hold_ready()
                    state = self._observe()
                preview = future.result()
            if preview.reason == "planning_deadline_exceeded":
                raise _Failure(
                    "reposition_preview_timeout",
                    wall_seconds=time.time() - started_wall,
                )
            self.journal.planning_attempts.append(
                {
                    "kind": "reposition_preview",
                    "strategy": strategy,
                    "robot_state": {
                        "joints": hypothetical.joints.tolist(),
                        "pelvis_position": hypothetical.pelvis_position.tolist(),
                        "pelvis_quaternion_wxyz": hypothetical.pelvis_quaternion_wxyz.tolist(),
                    },
                    "box_pose": {
                        "position": self.box.pose.position.tolist(),
                        "quaternion_wxyz": self.box.pose.quaternion_wxyz.tolist(),
                    },
                    "plan": preview.to_dict(),
                }
            )
            if preview.static_valid:
                return target_xy, target_yaw, preview
        return None

    def _walk_to(self, target_xy, target_yaw):
        state = self._observe(require_advance=False)
        began = float(state["t"])
        start_xy = np.asarray(state["pelvis"]["pos"][:2], dtype=float)
        direction = np.asarray(target_xy, dtype=float) - start_xy
        direction = direction / max(float(np.linalg.norm(direction)), 1e-9)
        self._reposition_last_xy = start_xy.copy()
        while float(state["t"]) - began <= self.config.reposition_timeout_s:
            current_xy = np.asarray(state["pelvis"]["pos"][:2], dtype=float)
            position_error = float(np.linalg.norm(current_xy - target_xy))
            remaining_forward = float((np.asarray(target_xy) - current_xy) @ direction)
            lateral_axis = np.array([-direction[1], direction[0]])
            lateral_error = float(abs((current_xy - start_xy) @ lateral_axis))
            yaw = float(state["pelvis"].get("yaw", 0.0))
            yaw_error = abs(math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw)))
            velocity = np.asarray(state["pelvis"].get("vel", [0.0, 0.0, 0.0]), dtype=float)
            forward_speed = max(0.0, float(velocity[:2] @ direction))
            braking_threshold = (
                self.config.reposition_position_tolerance_m + forward_speed * self.config.reposition_braking_horizon_s
            )
            if remaining_forward <= braking_threshold:
                self.runtime.stop_walk()
                state = self._observe()
                self._accumulate_reposition_travel(state)
                brake_xy = np.asarray(state["pelvis"]["pos"][:2], dtype=float)
                brake_yaw = float(state["pelvis"].get("yaw", 0.0))
                self.journal.event(
                    "reposition",
                    "brake_command",
                    requested_xy=np.asarray(target_xy).tolist(),
                    brake_xy=brake_xy.tolist(),
                    remaining_forward_m=remaining_forward,
                    measured_forward_speed_mps=forward_speed,
                    braking_threshold_m=braking_threshold,
                    lateral_error_m=float(abs((brake_xy - start_xy) @ lateral_axis)),
                    requested_yaw=target_yaw,
                    brake_yaw=brake_yaw,
                    yaw_error_rad=float(
                        abs(
                            math.atan2(
                                math.sin(target_yaw - brake_yaw),
                                math.cos(target_yaw - brake_yaw),
                            )
                        )
                    ),
                )
                return state
            self.runtime.walk_step(target_xy, target_yaw, self.config.reposition_speed_mps)
            state = self._observe()
            self._accumulate_reposition_travel(state)
            self.journal.trace(
                state,
                position_error_m=position_error,
                remaining_forward_m=remaining_forward,
                lateral_error_m=lateral_error,
                yaw_error_rad=yaw_error,
                measured_forward_speed_mps=forward_speed,
                braking_threshold_m=braking_threshold,
            )
        self.runtime.stop_walk()
        raise _Failure("reposition_timeout", target_xy=np.asarray(target_xy).tolist())

    def _accumulate_reposition_travel(self, state) -> None:
        current_xy = np.asarray(state["pelvis"]["pos"][:2], dtype=float)
        if self._reposition_last_xy is not None:
            self.journal.travel_m += float(np.linalg.norm(current_xy - self._reposition_last_xy))
        self._reposition_last_xy = current_xy.copy()
        if self.journal.travel_m > self.config.maximum_total_reposition_m:
            self.runtime.stop_walk()
            raise _Failure(
                "reposition_travel_budget_exceeded",
                actual_path_length_m=self.journal.travel_m,
            )

    def _tool_geometry(self, state, joints):
        local_positions, local_rotations = self.planner.bimanual.tool_poses(joints)
        root_rotation = Rotation.from_quat(np.asarray(state["pelvis"]["quat"])[[1, 2, 3, 0]]).as_matrix()
        positions = np.asarray(state["pelvis"]["pos"]) + local_positions @ root_rotation.T
        rotations = np.array([root_rotation @ value for value in local_rotations])
        return positions, rotations

    def _tool_geometry_at(self, joints, position, quaternion_wxyz):
        local_positions, local_rotations = self.planner.bimanual.tool_poses(joints)
        root_rotation = Rotation.from_quat(np.asarray(quaternion_wxyz)[[1, 2, 3, 0]]).as_matrix()
        positions = np.asarray(position) + local_positions @ root_rotation.T
        rotations = np.array([root_rotation @ value for value in local_rotations])
        return positions, rotations

    def _support_diagnostics(self, state, joints, position, quaternion_wxyz, *, foot_tolerance_m=None):
        ik = self.planner.ik
        model, data, mj = ik.model, ik.data, ik._mj
        feet = [model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
        pelvis = model.body("pelvis").id

        def measured(q, root_position, root_quaternion):
            data.qpos[:] = model.qpos0
            data.qpos[ik.base_qadr : ik.base_qadr + 3] = root_position
            data.qpos[ik.base_qadr + 3 : ik.base_qadr + 7] = root_quaternion
            data.qpos[ik.qadr] = q
            mj.mj_kinematics(model, data)
            mj.mj_comPos(model, data)
            return data.xpos[feet].copy(), data.subtree_com[pelvis, :2].copy()

        baseline_feet, baseline_com = measured(
            state["joints"],
            state["pelvis"]["pos"],
            state["pelvis"]["quat"],
        )
        reference_feet, reference_com = measured(joints, position, quaternion_wxyz)
        margins = np.minimum(
            np.asarray(joints) - self.planner.joint_ranges[:, 0],
            self.planner.joint_ranges[:, 1] - np.asarray(joints),
        )
        foot_errors = np.linalg.norm(reference_feet - baseline_feet, axis=1)
        com_shift = reference_com - baseline_com
        diagnostics = {
            "foot_anchor_errors_m": foot_errors.tolist(),
            "robot_com_shift_m": com_shift.tolist(),
            "minimum_joint_margin_rad": float(margins.min()),
        }
        reasons = []
        foot_tolerance = self.planner.options.foot_anchor_tolerance_m if foot_tolerance_m is None else foot_tolerance_m
        if float(foot_errors.max()) > foot_tolerance:
            reasons.append("foot_anchor_error")
        if float(np.linalg.norm(com_shift)) > self.planner.options.com_shift_tolerance_m:
            reasons.append("robot_com_shift")
        if float(margins.min()) < self.planner.options.min_joint_margin_rad:
            reasons.append("joint_limit_margin")
        return diagnostics, reasons

    def _solve_at_observed_root(self, state, targets, rotations, seed, *, posture=None):
        root = self._robot_state(state)
        local_positions = (targets - root.pelvis_position) @ root.rotation
        local_rotations = np.array([root.rotation.T @ value for value in rotations])
        margin = self.planner.options.min_joint_margin_rad + 1e-6
        limits = (
            np.maximum(self.planner.bimanual.lo, self.planner.joint_ranges[:, 0] + margin),
            np.minimum(self.planner.bimanual.hi, self.planner.joint_ranges[:, 1] - margin),
        )
        return self.planner.bimanual.solve(
            seed,
            local_positions,
            local_rotations,
            iterations=max(80, self.planner.options.coarse_iterations),
            posture=seed if posture is None else posture,
            posture_weight=0.002,
            joint_limits=limits,
        ).joints

    def _solve_tracking_at_observed_root(self, state, targets, rotations, seed):
        if self.config.approach_mode == "legacy_low_reach":
            return self._solve_at_observed_root(
                state,
                targets,
                rotations,
                seed,
                posture=self._low_reach_posture,
            )
        return self._solve_at_observed_root(state, targets, rotations, seed)

    @staticmethod
    def _observed_gripper_q(state) -> np.ndarray:
        return np.concatenate(
            [
                _vector(state["palms"][side]["hand_q"], 2, f"observation.palms.{side}.hand_q")
                for side in ("left", "right")
            ]
        )

    @staticmethod
    def _commanded_gripper_q(hands) -> np.ndarray:
        hand = np.asarray(hands, dtype=float)
        if hand.ndim != 1 or hand.size < 2 or not np.isfinite(hand[:2]).all():
            raise ValueError("hand command must contain at least two finite values")
        return np.tile(hand[:2], 2)

    def _collision_reasons(
        self,
        state,
        joints,
        targets,
        *,
        box=None,
        gripper_q=None,
        position=None,
        quaternion_wxyz=None,
        max_target_penetration_m=None,
    ):
        box = self.box if box is None else box
        gripper_q = self._observed_gripper_q(state) if gripper_q is None else gripper_q
        if max_target_penetration_m is None:
            max_target_penetration_m = min(self.planner.options.max_target_penetration_m, box.wall_thickness_m / 4)
        allowed, rejected = self.planner.collision.check(
            self._robot_state(state),
            joints,
            state["pelvis"]["pos"] if position is None else position,
            state["pelvis"]["quat"] if quaternion_wxyz is None else quaternion_wxyz,
            box,
            targets,
            target_contact_tolerance_m=self.planner.options.target_contact_tolerance_m,
            max_target_penetration_m=max_target_penetration_m,
            support_baseline_max_penetration_m=self.planner.options.support_baseline_max_penetration_m,
            support_penetration_growth_tolerance_m=self.planner.options.support_penetration_growth_tolerance_m,
            gripper_q=gripper_q,
            support_contact_policy=(
                "diagnostic_floor" if self.config.approach_mode == "coordinated" else "bounded_growth"
            ),
            # The continuous reach slides the fingers along the rim on the way in; the static
            # planner's intended-pad rule would reject that as a wall collision (2026-09-22).
            finger_wall_policy=("any_wall" if self.config.approach_mode == "coordinated" else "intended_pad"),
        )
        return [value.to_dict() for value in allowed], [value.to_dict() for value in rejected]

    def _admit_hand_sweep(
        self,
        state,
        joints,
        targets,
        start,
        end,
        *,
        position=None,
        quaternion_wxyz=None,
    ):
        for fraction in np.linspace(0.0, 1.0, self.config.swept_samples):
            gripper_q = start + fraction * (end - start)
            _, rejected = self._collision_reasons(
                state,
                joints,
                targets,
                gripper_q=gripper_q,
                position=position,
                quaternion_wxyz=quaternion_wxyz,
            )
            if rejected:
                raise _Failure(
                    "unsafe_swept_reference",
                    segment="open_at_clearance",
                    fraction=float(fraction),
                    contacts=rejected,
                )

    def _transit_targets(self, state, pregrasp_targets):
        rotation = self.box.pose.rotation
        local_pelvis = rotation.T @ (np.asarray(state["pelvis"]["pos"], dtype=float) - self.box.pose.position)
        front_axis = int(np.argmax(np.abs(local_pelvis[:2])))
        front_sign = 1.0 if local_pelvis[front_axis] >= 0 else -1.0
        clearance = self.config.pregrasp_clearance_m + 2.5 * self.box.wall_thickness_m
        front_offset = 4.0 * self.box.wall_thickness_m
        local = (np.asarray(pregrasp_targets) - self.box.pose.position) @ rotation
        raised_front_local = local.copy()
        raised_front_local[:, front_axis] = front_sign * (self.box.size_m[front_axis] / 2.0 + front_offset)
        raised_front_local[:, 2] = self.box.size_m[2] + clearance
        above_rim_local = local.copy()
        above_rim_local[:, 2] = self.box.size_m[2] + clearance
        return (
            self.box.pose.position + raised_front_local @ rotation.T,
            self.box.pose.position + above_rim_local @ rotation.T,
        )

    @staticmethod
    def _interpolate_rotations(start, end, fraction):
        values = []
        for begin, finish in zip(start, end):
            begin_rotation = Rotation.from_matrix(begin)
            relative = begin_rotation.inv() * Rotation.from_matrix(finish)
            values.append((begin_rotation * Rotation.from_rotvec(fraction * relative.as_rotvec())).as_matrix())
        return np.asarray(values)

    def _plan_supported_transit(
        self,
        state,
        targets,
        rotations,
        pregrasp_targets,
        *,
        deadline_monotonic,
    ):
        raised_front_targets, above_rim_targets = self._transit_targets(state, pregrasp_targets)
        initial_positions, initial_rotations = self._tool_geometry(state, state["joints"])
        whole = WholeBodyIK(
            self.planner.ik,
            self._robot_state(state).whole_body_state(),
            self.planner.options,
            tool_offset=self.planner.bimanual.offset,
            joint_limits=(self.planner.bimanual.lo, self.planner.bimanual.hi),
        )
        lo, hi = whole.lo.copy(), whole.hi.copy()
        shift = self.planner.options.max_pelvis_xy_change_m
        measured_position = np.asarray(state["pelvis"]["pos"], dtype=float)
        lo[:2] = np.maximum(lo[:2], measured_position[:2] - shift)
        hi[:2] = np.minimum(hi[:2], measured_position[:2] + shift)
        lo[2] = max(lo[2], measured_position[2] - self.planner.options.max_pelvis_height_change_m)
        hi[2] = min(hi[2], measured_position[2] + self.planner.options.max_pelvis_height_change_m)
        route = []
        previous_joints = np.asarray(state["joints"], dtype=float)
        previous_position = measured_position
        desired_start_positions = initial_positions
        desired_start_rotations = initial_rotations
        seed = whole.initial.copy()
        subdivisions = 6
        segments = (
            ("raised_front", raised_front_targets, rotations, self.config.pregrasp_position_tolerance_m, "ready"),
            ("above_rim", above_rim_targets, rotations, self.config.pregrasp_position_tolerance_m, "open"),
            ("pregrasp", pregrasp_targets, rotations, self.config.pregrasp_position_tolerance_m, "open"),
            ("align", targets, rotations, self.config.align_position_tolerance_m, "open"),
        )
        for segment, desired_end_positions, desired_end_rotations, tolerance, hand_phase in segments:
            for step, fraction in enumerate(np.linspace(0.0, 1.0, subdivisions + 1)[1:], start=1):
                if time.monotonic() >= deadline_monotonic:
                    raise _Failure("transit_planning_deadline_exceeded", segment=segment, step=step)
                desired_positions = desired_start_positions + fraction * (
                    desired_end_positions - desired_start_positions
                )
                desired_rotations = self._interpolate_rotations(
                    desired_start_rotations,
                    desired_end_rotations,
                    float(fraction),
                )
                if self.config.approach_mode == "planner_arms":
                    solution_joints = self._solve_at_observed_root(
                        state, desired_positions, desired_rotations, previous_joints
                    )
                    solution_position = measured_position
                    solution_rotation = self._robot_state(state).rotation
                else:
                    pose = whole.solve(
                        desired_positions,
                        desired_rotations,
                        seed=seed,
                        bounds=(lo, hi),
                        posture=previous_joints,
                        iterations=self.planner.options.whole_body_iterations,
                    )
                    solution_joints = pose.joints
                    solution_position = pose.position
                    solution_rotation = pose.rotation
                quaternion = Rotation.from_matrix(solution_rotation).as_quat()[[3, 0, 1, 2]]
                actual_positions, actual_rotations = self._tool_geometry_at(
                    solution_joints,
                    solution_position,
                    quaternion,
                )
                position_error = float(np.max(np.linalg.norm(actual_positions - desired_positions, axis=1)))
                orientation_error = float(
                    max(
                        Rotation.from_matrix(wanted @ found.T).magnitude()
                        for found, wanted in zip(actual_rotations, desired_rotations)
                    )
                )
                support, support_reasons = self._support_diagnostics(
                    state,
                    solution_joints,
                    solution_position,
                    quaternion,
                )
                if (
                    position_error > tolerance
                    or orientation_error > self.config.align_orientation_tolerance_rad
                    or support_reasons
                ):
                    raise _Failure(
                        "unsupported_transit_waypoint",
                        segment=segment,
                        step=step,
                        position_error_m=position_error,
                        orientation_error_rad=orientation_error,
                        support_reasons=support_reasons,
                        **support,
                    )
                route.append(
                    {
                        "segment": segment,
                        "step": step,
                        "hand_phase": hand_phase,
                        "joints": solution_joints.copy(),
                        "pelvis_position": solution_position.copy(),
                        "pelvis_quaternion_wxyz": quaternion,
                        "targets_world_m": desired_positions,
                        "target_rotations": desired_rotations,
                        "position_tolerance_m": tolerance,
                        "position_error_m": position_error,
                        "orientation_error_rad": orientation_error,
                        **support,
                    }
                )
                previous_joints = solution_joints
                previous_position = solution_position
                pitch = Rotation.from_matrix(solution_rotation).as_euler("xyz")[1]
                seed = np.r_[previous_position, pitch, previous_joints]
            desired_start_positions = desired_end_positions
            desired_start_rotations = desired_end_rotations
        return route

    def _plan_low_reach_transit(
        self,
        state,
        targets,
        rotations,
        pregrasp_targets,
        *,
        deadline_monotonic,
    ):
        """Build the old expert's planted-foot bend then fixed-root arm approach."""
        if time.monotonic() >= deadline_monotonic:
            raise _Failure("transit_planning_deadline_exceeded", segment="low_reach", step=0)
        margin = self.planner.options.min_joint_margin_rad + 1e-6
        limits = (
            np.maximum(self.planner.bimanual.lo, self.planner.joint_ranges[:, 0] + margin),
            np.minimum(self.planner.bimanual.hi, self.planner.joint_ranges[:, 1] - margin),
        )
        low_pose = plan_low_reach(
            self.planner.ik,
            self._robot_state(state).whole_body_state(),
            targets,
            rotations,
            self.planner.options,
            tool_offset=self.planner.bimanual.offset,
            joint_limits=limits,
        )
        low_quaternion = Rotation.from_matrix(low_pose.rotation).as_quat()[[3, 0, 1, 2]]
        support, support_reasons = self._support_diagnostics(
            state,
            low_pose.joints,
            low_pose.position,
            low_quaternion,
        )
        if float(np.max(low_pose.hand_errors)) > self.planner.options.hand_position_tolerance_m or support_reasons:
            raise _Failure(
                "unsupported_low_reach_posture",
                hand_errors_m=low_pose.hand_errors.tolist(),
                support_reasons=support_reasons,
                **support,
            )

        raised_front_targets, above_rim_targets = self._transit_targets(state, pregrasp_targets)
        route = []
        measured_root = self._robot_state(state)
        arm_start_positions, arm_start_rotations = self._tool_geometry(state, state["joints"])
        arm_previous_joints = np.asarray(state["joints"], dtype=float).copy()
        for step, fraction in enumerate(np.linspace(0.0, 1.0, 7)[1:], start=1):
            desired_positions = arm_start_positions + fraction * (raised_front_targets - arm_start_positions)
            desired_rotations = self._interpolate_rotations(arm_start_rotations, rotations, float(fraction))
            local_positions = (desired_positions - measured_root.pelvis_position) @ measured_root.rotation
            local_rotations = np.array([measured_root.rotation.T @ value for value in desired_rotations])
            solution = self.planner.bimanual.solve(
                arm_previous_joints,
                local_positions,
                local_rotations,
                iterations=max(180, self.planner.options.coarse_iterations),
                posture=arm_previous_joints,
                posture_weight=0.002,
                rot_weight=0.25,
                joint_limits=limits,
            )
            actual_positions, actual_rotations = self._tool_geometry_at(
                solution.joints,
                measured_root.pelvis_position,
                measured_root.pelvis_quaternion_wxyz,
            )
            position_error = float(np.max(np.linalg.norm(actual_positions - desired_positions, axis=1)))
            orientation_error = float(
                max(
                    Rotation.from_matrix(wanted @ found.T).magnitude()
                    for found, wanted in zip(actual_rotations, desired_rotations)
                )
            )
            diagnostics, reasons = self._support_diagnostics(
                state,
                solution.joints,
                measured_root.pelvis_position,
                measured_root.pelvis_quaternion_wxyz,
            )
            if (
                position_error > self.config.low_reach_tracking_tolerance_m
                or orientation_error > self.config.align_orientation_tolerance_rad
                or reasons
            ):
                raise _Failure(
                    "unsupported_low_reach_arm_clearance",
                    step=step,
                    position_error_m=position_error,
                    orientation_error_rad=orientation_error,
                    support_reasons=reasons,
                    **diagnostics,
                )
            route.append(
                {
                    "segment": "low_reach_arm_clearance",
                    "step": step,
                    "hand_phase": "ready",
                    "joints": solution.joints.copy(),
                    "pelvis_position": measured_root.pelvis_position.copy(),
                    "pelvis_quaternion_wxyz": measured_root.pelvis_quaternion_wxyz.copy(),
                    "targets_world_m": desired_positions.copy(),
                    "target_rotations": desired_rotations.copy(),
                    "position_tolerance_m": self.config.low_reach_tracking_tolerance_m,
                    "position_error_m": position_error,
                    "orientation_error_rad": orientation_error,
                    **diagnostics,
                }
            )
            arm_previous_joints = solution.joints.copy()
        previous_joints = low_pose.joints.copy()
        desired_start_positions = raised_front_targets
        desired_start_rotations = rotations

        def solve_waypoint(segment, step, desired_positions, desired_rotations, hand_phase):
            nonlocal previous_joints
            if time.monotonic() >= deadline_monotonic:
                raise _Failure("transit_planning_deadline_exceeded", segment=segment, step=step)
            local_positions = (desired_positions - low_pose.position) @ low_pose.rotation
            local_rotations = np.array([low_pose.rotation.T @ value for value in desired_rotations])
            solution = self.planner.bimanual.solve(
                previous_joints,
                local_positions,
                local_rotations,
                iterations=max(180, self.planner.options.coarse_iterations),
                posture=low_pose.joints,
                posture_weight=0.002,
                rot_weight=0.25,
                joint_limits=limits,
            )
            actual_positions, actual_rotations = self._tool_geometry_at(
                solution.joints,
                low_pose.position,
                low_quaternion,
            )
            position_error = float(np.max(np.linalg.norm(actual_positions - desired_positions, axis=1)))
            orientation_error = float(
                max(
                    Rotation.from_matrix(wanted @ found.T).magnitude()
                    for found, wanted in zip(actual_rotations, desired_rotations)
                )
            )
            diagnostics, reasons = self._support_diagnostics(
                state,
                solution.joints,
                low_pose.position,
                low_quaternion,
            )
            if (
                position_error > self.config.low_reach_tracking_tolerance_m
                or orientation_error > self.config.align_orientation_tolerance_rad
                or reasons
            ):
                raise _Failure(
                    "unsupported_low_reach_waypoint",
                    segment=segment,
                    step=step,
                    position_error_m=position_error,
                    orientation_error_rad=orientation_error,
                    support_reasons=reasons,
                    **diagnostics,
                )
            waypoint = {
                "segment": segment,
                "step": step,
                "hand_phase": hand_phase,
                "joints": solution.joints.copy(),
                "pelvis_position": low_pose.position.copy(),
                "pelvis_quaternion_wxyz": low_quaternion.copy(),
                "targets_world_m": np.asarray(desired_positions).copy(),
                "target_rotations": np.asarray(desired_rotations).copy(),
                "position_tolerance_m": self.config.low_reach_tracking_tolerance_m,
                "position_error_m": position_error,
                "orientation_error_rad": orientation_error,
                **diagnostics,
            }
            previous_joints = solution.joints.copy()
            return waypoint

        bend = solve_waypoint(
            "low_reach_bend",
            1,
            raised_front_targets,
            rotations,
            "ready",
        )
        route.append(bend)
        self._low_reach_posture = low_pose.joints.copy()
        self._low_reach_root_quaternion = low_quaternion.copy()

        subdivisions = 6
        segments = (
            ("above_rim", above_rim_targets, rotations, "open"),
            ("pregrasp", pregrasp_targets, rotations, "open"),
            ("align", targets, rotations, "open"),
        )
        for segment, desired_end_positions, desired_end_rotations, hand_phase in segments:
            for step, fraction in enumerate(np.linspace(0.0, 1.0, subdivisions + 1)[1:], start=1):
                desired_positions = desired_start_positions + fraction * (
                    desired_end_positions - desired_start_positions
                )
                desired_rotations = self._interpolate_rotations(
                    desired_start_rotations,
                    desired_end_rotations,
                    float(fraction),
                )
                route.append(
                    solve_waypoint(
                        segment,
                        step,
                        desired_positions,
                        desired_rotations,
                        hand_phase,
                    )
                )
            desired_start_positions = desired_end_positions
            desired_start_rotations = desired_end_rotations
        return route

    def _admit_supported_segment(
        self,
        state,
        start_joints,
        start_position,
        start_quaternion,
        waypoint,
        targets,
        gripper_q,
    ):
        begin_rotation = Rotation.from_quat(np.asarray(start_quaternion)[[1, 2, 3, 0]])
        end_rotation = Rotation.from_quat(np.asarray(waypoint["pelvis_quaternion_wxyz"])[[1, 2, 3, 0]])
        relative = begin_rotation.inv() * end_rotation
        for fraction in np.linspace(0.0, 1.0, self.config.swept_samples):
            joints = start_joints + fraction * (waypoint["joints"] - start_joints)
            position = start_position + fraction * (waypoint["pelvis_position"] - start_position)
            quaternion = (begin_rotation * Rotation.from_rotvec(fraction * relative.as_rotvec())).as_quat()[
                [3, 0, 1, 2]
            ]
            support, support_reasons = self._support_diagnostics(state, joints, position, quaternion)
            if support_reasons:
                raise _Failure(
                    "unsupported_swept_reference",
                    segment=waypoint["segment"],
                    step=waypoint["step"],
                    fraction=float(fraction),
                    support_reasons=support_reasons,
                    **support,
                )
            _, rejected = self._collision_reasons(
                state,
                joints,
                targets,
                gripper_q=gripper_q,
                position=position,
                quaternion_wxyz=quaternion,
            )
            if rejected:
                raise _Failure(
                    "unsafe_swept_reference",
                    segment=waypoint["segment"],
                    step=waypoint["step"],
                    fraction=float(fraction),
                    contacts=rejected,
                )

    def _admit_low_reach_bend(
        self,
        state,
        start_joints,
        start_position,
        start_quaternion,
        waypoint,
        targets,
        gripper_q,
    ):
        """Admit bend endpoints strictly and its reference path without fake foot FK."""
        begin_rotation = Rotation.from_quat(np.asarray(start_quaternion)[[1, 2, 3, 0]])
        end_rotation = Rotation.from_quat(np.asarray(waypoint["pelvis_quaternion_wxyz"])[[1, 2, 3, 0]])
        relative = begin_rotation.inv() * end_rotation
        samples = np.linspace(0.0, 1.0, self.config.swept_samples)
        for index, fraction in enumerate(samples):
            joints = start_joints + fraction * (waypoint["joints"] - start_joints)
            position = start_position + fraction * (waypoint["pelvis_position"] - start_position)
            quaternion = (begin_rotation * Rotation.from_rotvec(fraction * relative.as_rotvec())).as_quat()[
                [3, 0, 1, 2]
            ]
            support, support_reasons = self._support_diagnostics(state, joints, position, quaternion)
            # A joint/root interpolation is not a statically planted kinematic
            # state. Endpoints must anchor the feet; during the bend, the live
            # measured feet are checked on every command below.
            if index != len(samples) - 1:
                support_reasons = [reason for reason in support_reasons if reason != "foot_anchor_error"]
            if support_reasons:
                raise _Failure(
                    "unsupported_low_reach_bend",
                    fraction=float(fraction),
                    support_reasons=support_reasons,
                    **support,
                )
            _, rejected = self._collision_reasons(
                state,
                joints,
                targets,
                gripper_q=gripper_q,
                position=position,
                quaternion_wxyz=quaternion,
            )
            if index != len(samples) - 1:
                rejected = [
                    contact
                    for contact in rejected
                    if not (
                        contact["classification"] == "robot_environment_collision"
                        and "ankle_roll_link" in (contact["body_1"] + " " + contact["body_2"])
                        and "floor" in (contact["geom_1"] + " " + contact["geom_2"])
                    )
                ]
            if rejected:
                raise _Failure(
                    "unsafe_swept_reference",
                    segment=waypoint["segment"],
                    step=waypoint["step"],
                    fraction=float(fraction),
                    contacts=rejected,
                )

    @staticmethod
    def _contact_pair(contact):
        return tuple(sorted((str(contact["geom_1"]), str(contact["geom_2"]))))

    @staticmethod
    def _contact_depths(contacts):
        return [{"geoms": list(pair), "penetration_m": float(depth)} for pair, depth in sorted(contacts.items())]

    def _initial_escape_contacts(self, rejected):
        contacts = {}
        for contact in rejected:
            penetration = max(0.0, -float(contact["distance_m"]))
            if (
                contact["classification"] != "robot_self_collision"
                or penetration > self.config.initial_escape_max_penetration_m
            ):
                raise _Failure(
                    "initial_escape_not_admissible",
                    maximum_penetration_m=self.config.initial_escape_max_penetration_m,
                    contacts=rejected,
                )
            pair = self._contact_pair(contact)
            contacts[pair] = max(contacts.get(pair, 0.0), penetration)
        return contacts

    def _resolving_contact_depths(
        self,
        rejected,
        initial_contacts,
        previous_contacts,
        *,
        phase,
        progress,
        tracking: bool = False,
    ):
        current = {pair: 0.0 for pair in initial_contacts}
        for contact in rejected:
            pair = self._contact_pair(contact)
            penetration_now = max(0.0, -float(contact["distance_m"]))
            if (
                contact["classification"] == "robot_self_collision"
                and pair not in initial_contacts
                and penetration_now <= self.config.initial_escape_max_penetration_m
            ):
                # The hand leaving the hip grazes other link pairs on the way (wrist-yaw, finger vs
                # hip roll/yaw at 1-100 um, 2026-09-22). Within the same depth envelope that admits
                # the initial contacts, a graze is tracked like them: it may not deepen and must
                # clear before the escape completes, but it is not a new collision.
                initial_contacts[pair] = penetration_now
                current[pair] = 0.0
            if contact["classification"] != "robot_self_collision" or pair not in initial_contacts:
                raise _Failure(
                    "initial_escape_new_collision",
                    phase=phase,
                    progress=float(progress),
                    contacts=rejected,
                    initial_contacts=self._contact_depths(initial_contacts),
                )
            current[pair] = max(current.get(pair, 0.0), max(0.0, -float(contact["distance_m"])))
        for pair in initial_contacts:
            previous = previous_contacts.get(pair, initial_contacts[pair])
            penetration = current[pair]
            # Reference paths remain strictly monotone. During their bounded
            # execution, controller/physics transients may revisit only the
            # exact initial pairs, within a small explicit depth envelope.
            limit = (
                min(
                    initial_contacts[pair] + self.config.initial_escape_tracking_growth_m,
                    self.config.initial_escape_max_penetration_m,
                )
                if tracking
                else previous
            )
            if penetration > limit + 1e-9:
                raise _Failure(
                    "initial_escape_penetration_increased",
                    phase=phase,
                    progress=float(progress),
                    pair=list(pair),
                    previous_penetration_m=float(previous),
                    penetration_m=float(penetration),
                    permitted_penetration_m=float(limit),
                )
        return current

    def _admit_initial_escape(self, state, waypoint, targets):
        start_joints = np.asarray(state["joints"], dtype=float)
        start_position = np.asarray(state["pelvis"]["pos"], dtype=float)
        start_quaternion = np.asarray(state["pelvis"]["quat"], dtype=float)
        gripper_q = self._observed_gripper_q(state)
        _, rejected = self._collision_reasons(
            state,
            start_joints,
            targets,
            gripper_q=gripper_q,
            position=start_position,
            quaternion_wxyz=start_quaternion,
        )
        initial_contacts = self._initial_escape_contacts(rejected)
        previous_contacts = dict(initial_contacts)
        begin_rotation = Rotation.from_quat(start_quaternion[[1, 2, 3, 0]])
        end_rotation = Rotation.from_quat(np.asarray(waypoint["pelvis_quaternion_wxyz"])[[1, 2, 3, 0]])
        relative = begin_rotation.inv() * end_rotation
        cleared_fraction = 0.0 if not initial_contacts else None
        for fraction in np.linspace(0.0, 1.0, self.config.initial_escape_samples):
            joints = start_joints + fraction * (waypoint["joints"] - start_joints)
            position = start_position + fraction * (waypoint["pelvis_position"] - start_position)
            quaternion = (begin_rotation * Rotation.from_rotvec(fraction * relative.as_rotvec())).as_quat()[
                [3, 0, 1, 2]
            ]
            support, support_reasons = self._support_diagnostics(state, joints, position, quaternion)
            if support_reasons:
                raise _Failure(
                    "unsupported_initial_escape_reference",
                    fraction=float(fraction),
                    support_reasons=support_reasons,
                    **support,
                )
            _, rejected = self._collision_reasons(
                state,
                joints,
                targets,
                gripper_q=gripper_q,
                position=position,
                quaternion_wxyz=quaternion,
            )
            previous_contacts = self._resolving_contact_depths(
                rejected,
                initial_contacts,
                previous_contacts,
                phase="reference",
                progress=float(fraction),
            )
            if not any(depth > 0.0 for depth in previous_contacts.values()) and cleared_fraction is None:
                cleared_fraction = float(fraction)
        if cleared_fraction is None:
            raise _Failure(
                "initial_escape_not_cleared",
                contacts=self._contact_depths(previous_contacts),
            )
        self.journal.event(
            "ready_escape",
            "reference_admitted",
            segment=waypoint["segment"],
            step=waypoint["step"],
            initial_contacts=self._contact_depths(initial_contacts),
            cleared_fraction=cleared_fraction,
        )
        return initial_contacts

    def _plan_supported_transit_with_keepalive(
        self,
        state,
        targets,
        rotations,
        pregrasp_targets,
        *,
        hold_pose=None,
    ):
        deadline = time.monotonic() + self.config.planning_timeout_s
        # The worker owns planner.ik while solving.  This loop only sends the
        # already-owned PlannerArm reference and reads observations; neither path
        # performs FK or otherwise touches that mutable model/data pair.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="box-rim-transit") as pool:
            future = pool.submit(
                self._plan_supported_transit,
                state,
                targets,
                rotations,
                pregrasp_targets,
                deadline_monotonic=deadline,
            )
            while not future.done():
                if hold_pose is None:
                    self.runtime.hold_ready()
                else:
                    self.runtime.pose_step(*hold_pose)
                observed = self._observe()
                self._check_box_not_pushed(observed)
            route = future.result()
        observed = self._observe()
        self._check_box_not_pushed(observed)
        return route, observed

    def _plan_low_reach_transit_with_keepalive(
        self,
        state,
        targets,
        rotations,
        pregrasp_targets,
    ):
        deadline = time.monotonic() + self.config.planning_timeout_s
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="box-rim-low-reach") as pool:
            future = pool.submit(
                self._plan_low_reach_transit,
                state,
                targets,
                rotations,
                pregrasp_targets,
                deadline_monotonic=deadline,
            )
            while not future.done():
                self.runtime.hold_ready()
                observed = self._observe()
                self._check_box_not_pushed(observed)
            route = future.result()
        observed = self._observe()
        self._check_box_not_pushed(observed)
        return route, observed

    def _check_observed_waypoint(self, baseline_state, state, waypoint):
        positions, rotations = self._tool_geometry(state, state["joints"])
        position_error = float(np.max(np.linalg.norm(positions - waypoint["targets_world_m"], axis=1)))
        orientation_error = float(
            max(
                Rotation.from_matrix(wanted @ found.T).magnitude()
                for found, wanted in zip(rotations, waypoint["target_rotations"])
            )
        )
        support, support_reasons = self._support_diagnostics(
            baseline_state,
            state["joints"],
            state["pelvis"]["pos"],
            state["pelvis"]["quat"],
            foot_tolerance_m=self.config.observed_foot_anchor_tolerance_m,
        )
        if (
            position_error > waypoint["position_tolerance_m"]
            or orientation_error > self.config.align_orientation_tolerance_rad
            or support_reasons
        ):
            raise _Failure(
                "transit_tracking_error",
                segment=waypoint["segment"],
                step=waypoint["step"],
                position_error_m=position_error,
                orientation_error_rad=orientation_error,
                support_reasons=support_reasons,
                **support,
            )
        self.journal.event(
            waypoint["segment"],
            "waypoint_tracked",
            step=waypoint["step"],
            position_error_m=position_error,
            orientation_error_rad=orientation_error,
            **support,
        )

    def _command_joints(self, state):
        if self.config.approach_mode == "legacy_low_reach" and self._low_reach_reference_joints is not None:
            return self._low_reach_reference_joints.copy()
        return np.asarray(state["joints"], dtype=float).copy()

    def _command_root_quaternion(self, state):
        if self.config.approach_mode == "legacy_low_reach" and self._low_reach_root_quaternion is not None:
            return self._low_reach_root_quaternion.copy()
        return np.asarray(state["pelvis"]["quat"], dtype=float).copy()

    def _open_at_clearance(self, state, joints, targets):
        self.journal.event("transit_open", "motion_started")
        began = float(state["t"])
        while float(state["t"]) - began < self.config.close_s:
            u = np.clip((float(state["t"]) - began) / self.config.close_s, 0.0, 1.0)
            blend = u * u * (3.0 - 2.0 * u)
            hands = HAND_READY + blend * (HAND_OPEN - HAND_READY)
            self.runtime.pose_step(joints, self._command_root_quaternion(state), hands)
            state = self._observe()
            self._check_box_not_pushed(state)
            _, rejected = self._collision_reasons(state, state["joints"], targets)
            if rejected or state.get("furniture_touch"):
                raise _Failure(
                    "observed_collision_during_motion",
                    contacts=rejected,
                    furniture_touch=state.get("furniture_touch", []),
                )
            self.journal.trace(
                state,
                target_key=self.target_key,
                reference_progress=float(u),
                reference_joints=np.asarray(joints).tolist(),
                reference_root_quaternion_wxyz=list(state["pelvis"]["quat"]),
                reference_hands=hands.tolist(),
            )
        self.runtime.pose_step(joints, self._command_root_quaternion(state), HAND_OPEN)
        state = self._observe()
        _, rejected = self._collision_reasons(state, state["joints"], targets)
        if rejected or state.get("furniture_touch"):
            raise _Failure(
                "observed_collision_during_motion",
                contacts=rejected,
                furniture_touch=state.get("furniture_touch", []),
            )
        return state

    def _check_box_not_pushed(self, state):
        obj = state["objects"][self.target_key]
        displacement = float(np.linalg.norm(np.asarray(obj["pos"]) - self.box.pose.position))
        tilt = _quaternion_angle(obj["quat"], self.box.pose.quaternion_wxyz)
        if displacement > self.config.maximum_box_motion_before_close_m:
            raise _Failure("box_pushed_before_close", displacement_m=displacement)
        if tilt > self.config.maximum_box_tilt_before_close_rad:
            raise _Failure("box_tilted_before_close", tilt_rad=tilt)

    def _move_reference(self, phase, start_state, target_joints, hands, target_root_quat=None):
        self.journal.event(phase, "motion_started")
        q0 = self._command_joints(start_state)
        delta = float(np.max(np.abs(target_joints - q0)))
        duration = max(
            0.4,
            1.5 * delta / self.config.maximum_joint_speed_rps,
            math.sqrt(6.0 * delta / self.config.maximum_joint_acceleration_rps2),
        )
        began = float(start_state["t"])
        root_quat = self._command_root_quaternion(start_state)
        target_root_quat = root_quat if target_root_quat is None else np.asarray(target_root_quat, dtype=float)
        roots = Rotation.from_quat(np.stack((root_quat, target_root_quat))[:, [1, 2, 3, 0]])
        state = start_state
        settle = self.config.waypoint_settle_s if self.config.approach_mode == "planner_arms" else 0.0
        while float(state["t"]) - began < duration + settle:
            u = np.clip((float(state["t"]) - began) / duration, 0.0, 1.0)
            blend = u * u * (3.0 - 2.0 * u)
            command = q0 + blend * (target_joints - q0)
            velocity = (6.0 * u * (1.0 - u) / duration) * (target_joints - q0)
            relative = roots[0].inv() * roots[1]
            command_root = (roots[0] * Rotation.from_rotvec(blend * relative.as_rotvec())).as_quat()[[3, 0, 1, 2]]
            if self.config.approach_mode == "legacy_low_reach":
                self.runtime.pose_step(
                    command,
                    command_root,
                    hands,
                    reference_velocity=velocity,
                )
            else:
                self.runtime.pose_step(command, command_root, hands)
            state = self._observe()
            self._check_box_not_pushed(state)
            if self.config.approach_mode == "legacy_low_reach":
                baseline = self._low_reach_execution_baseline or start_state
                support, support_reasons = self._support_diagnostics(
                    baseline,
                    state["joints"],
                    state["pelvis"]["pos"],
                    state["pelvis"]["quat"],
                    foot_tolerance_m=self.config.observed_foot_anchor_tolerance_m,
                )
                if support_reasons:
                    raise _Failure(
                        "observed_low_reach_tracking_rejected",
                        support_reasons=support_reasons,
                        **support,
                    )
            _, rejected = self._collision_reasons(
                state, state["joints"], np.asarray(self.journal.selected_plan["targets_world_m"])
            )
            if rejected or state.get("furniture_touch"):
                raise _Failure(
                    "observed_collision_during_motion",
                    contacts=rejected,
                    furniture_touch=state.get("furniture_touch", []),
                )
            self.journal.trace(
                state,
                target_key=self.target_key,
                reference_progress=float(u),
                reference_joints=command.tolist(),
                reference_root_quaternion_wxyz=command_root.tolist(),
                reference_hands=np.asarray(hands).tolist(),
            )
        if self.config.approach_mode == "legacy_low_reach":
            self.runtime.pose_step(
                target_joints,
                target_root_quat,
                hands,
                reference_velocity=np.zeros(29, dtype=float),
            )
        else:
            self.runtime.pose_step(target_joints, target_root_quat, hands)
        if self.config.approach_mode == "legacy_low_reach":
            self._low_reach_reference_joints = np.asarray(target_joints, dtype=float).copy()
        return self._observe()

    def _move_low_reach_bend(self, start_state, start_reference, waypoint):
        """Execute the historical >=4 s smooth bend from the POSE handoff reference."""
        self.journal.event("low_reach_bend", "motion_started")
        q0 = np.asarray(start_reference[0], dtype=float).copy()
        q1 = np.asarray(waypoint["joints"], dtype=float).copy()
        root0 = np.asarray(start_reference[1], dtype=float).copy()
        root1 = np.asarray(waypoint["pelvis_quaternion_wxyz"], dtype=float).copy()
        delta = float(np.max(np.abs(q1 - q0)))
        duration = max(
            self.config.low_reach_minimum_bend_s,
            1.875 * delta / self.config.maximum_joint_speed_rps,
            math.sqrt(5.8 * delta / self.config.maximum_joint_acceleration_rps2),
        )
        roots = Slerp(
            [0.0, 1.0],
            Rotation.from_quat(np.stack((root0, root1))[:, [1, 2, 3, 0]]),
        )
        began = float(start_state["t"])
        state = start_state
        while float(state["t"]) - began < duration:
            u = float(np.clip((float(state["t"]) - began) / duration, 0.0, 1.0))
            blend = float(np.clip(u**3 * (10.0 - 15.0 * u + 6.0 * u**2), 0.0, 1.0))
            speed = float(30.0 * u**2 * (1.0 - u) ** 2 / duration)
            command = q0 + blend * (q1 - q0)
            velocity = speed * (q1 - q0)
            command_root = roots(blend).as_quat()[[3, 0, 1, 2]]
            self.runtime.pose_step(
                command,
                command_root,
                HAND_READY,
                reference_velocity=velocity,
            )
            state = self._observe()
            self._check_box_not_pushed(state)
            baseline = self._low_reach_execution_baseline or start_state
            support, support_reasons = self._support_diagnostics(
                baseline,
                state["joints"],
                state["pelvis"]["pos"],
                state["pelvis"]["quat"],
                foot_tolerance_m=self.config.observed_foot_anchor_tolerance_m,
            )
            _, rejected = self._collision_reasons(
                state,
                state["joints"],
                np.asarray(self.journal.selected_plan["targets_world_m"]),
            )
            if support_reasons or rejected or state.get("furniture_touch"):
                raise _Failure(
                    "observed_low_reach_bend_rejected",
                    support_reasons=support_reasons,
                    contacts=rejected,
                    furniture_touch=state.get("furniture_touch", []),
                    **support,
                )
            self.journal.trace(
                state,
                target_key=self.target_key,
                reference_progress=blend,
                reference_joints=command.tolist(),
                reference_velocity=velocity.tolist(),
                reference_root_quaternion_wxyz=command_root.tolist(),
                reference_hands=HAND_READY.tolist(),
            )
        self.runtime.pose_step(
            q1,
            root1,
            HAND_READY,
            reference_velocity=np.zeros(29, dtype=float),
        )
        self._low_reach_reference_joints = q1.copy()
        self._low_reach_root_quaternion = root1.copy()
        state = self._observe()
        self.journal.event(
            "low_reach_bend",
            "completed",
            duration_s=float(state["t"]) - began,
        )
        return state

    def _check_initial_escape_state(
        self,
        baseline_state,
        state,
        targets,
        initial_contacts,
        previous_contacts,
        *,
        progress,
        reference_joints,
        reference_root_quaternion_wxyz,
    ):
        support, support_reasons = self._support_diagnostics(
            baseline_state,
            state["joints"],
            state["pelvis"]["pos"],
            state["pelvis"]["quat"],
            foot_tolerance_m=self.config.observed_foot_anchor_tolerance_m,
        )
        if support_reasons:
            raise _Failure(
                "unsupported_observed_initial_escape",
                progress=float(progress),
                support_reasons=support_reasons,
                **support,
            )
        _, rejected = self._collision_reasons(state, state["joints"], targets)
        self.journal.trace(
            state,
            reference_progress=float(progress),
            reference_joints=np.asarray(reference_joints, dtype=float).tolist(),
            escape_observed_contacts=rejected,
        )
        current_contacts = self._resolving_contact_depths(
            rejected,
            initial_contacts,
            previous_contacts,
            phase="observed",
            progress=float(progress),
            tracking=True,
        )
        self.journal.trace(
            state,
            target_key=self.target_key,
            reference_progress=float(progress),
            reference_joints=np.asarray(reference_joints, dtype=float).tolist(),
            reference_root_quaternion_wxyz=np.asarray(reference_root_quaternion_wxyz, dtype=float).tolist(),
            reference_hands=HAND_READY.tolist(),
            initial_escape_contacts=self._contact_depths(current_contacts),
            initial_escape_support=support,
        )
        if state.get("furniture_touch"):
            raise _Failure(
                "observed_collision_during_initial_escape",
                progress=float(progress),
                furniture_touch=state.get("furniture_touch", []),
            )
        return current_contacts

    def _move_initial_escape(self, start_state, waypoint, targets, initial_contacts):
        self.journal.event("ready_escape", "motion_started")
        q0 = np.asarray(start_state["joints"], dtype=float)
        target_joints = np.asarray(waypoint["joints"], dtype=float)
        delta = float(np.max(np.abs(target_joints - q0)))
        duration = max(
            0.4,
            1.5 * delta / self.config.maximum_joint_speed_rps,
            math.sqrt(6.0 * delta / self.config.maximum_joint_acceleration_rps2),
        )
        began = float(start_state["t"])
        root_quat = np.asarray(start_state["pelvis"]["quat"], dtype=float)
        target_root_quat = np.asarray(waypoint["pelvis_quaternion_wxyz"], dtype=float)
        roots = Rotation.from_quat(np.stack((root_quat, target_root_quat))[:, [1, 2, 3, 0]])
        relative = roots[0].inv() * roots[1]
        previous_contacts = dict(initial_contacts)
        state = start_state
        while float(state["t"]) - began < duration:
            u = np.clip((float(state["t"]) - began) / duration, 0.0, 1.0)
            blend = u * u * (3.0 - 2.0 * u)
            command = q0 + blend * (target_joints - q0)
            command_root = (roots[0] * Rotation.from_rotvec(blend * relative.as_rotvec())).as_quat()[[3, 0, 1, 2]]
            self.runtime.ready_step(command, HAND_READY)
            state = self._observe()
            self._check_box_not_pushed(state)
            previous_contacts = self._check_initial_escape_state(
                start_state,
                state,
                targets,
                initial_contacts,
                previous_contacts,
                progress=float(blend),
                reference_joints=command,
                reference_root_quaternion_wxyz=command_root,
            )
        self.runtime.ready_step(target_joints, HAND_READY)
        state = self._observe()
        self._check_box_not_pushed(state)
        previous_contacts = self._check_initial_escape_state(
            start_state,
            state,
            targets,
            initial_contacts,
            previous_contacts,
            progress=1.0,
            reference_joints=target_joints,
            reference_root_quaternion_wxyz=target_root_quat,
        )
        if any(depth > 0.0 for depth in previous_contacts.values()):
            raise _Failure(
                "initial_escape_not_cleared",
                contacts=self._contact_depths(previous_contacts),
            )
        self.journal.event(
            "ready_escape",
            "measured_clear",
            duration_s=float(state["t"]) - began,
        )
        return state

    def _move_planner_clearance(self, start_state, waypoint, targets, baseline_state):
        """Raise the hands while the legacy PlannerArm still owns the upper body."""
        self.journal.event("low_reach_arm_clearance", "motion_started", step=waypoint["step"])
        q0 = np.asarray(start_state["joints"], dtype=float)
        q1 = np.asarray(waypoint["joints"], dtype=float)
        delta = float(np.max(np.abs(q1 - q0)))
        duration = max(
            0.4,
            1.5 * delta / self.config.maximum_joint_speed_rps,
            math.sqrt(6.0 * delta / self.config.maximum_joint_acceleration_rps2),
        )
        began = float(start_state["t"])
        state = start_state
        while float(state["t"]) - began < duration:
            u = float(np.clip((float(state["t"]) - began) / duration, 0.0, 1.0))
            blend = u * u * (3.0 - 2.0 * u)
            command = q0 + blend * (q1 - q0)
            self.runtime.ready_step(command, HAND_READY)
            state = self._observe()
            self._check_box_not_pushed(state)
            support, support_reasons = self._support_diagnostics(
                baseline_state,
                state["joints"],
                state["pelvis"]["pos"],
                state["pelvis"]["quat"],
                foot_tolerance_m=self.config.observed_foot_anchor_tolerance_m,
            )
            _, rejected = self._collision_reasons(state, state["joints"], targets)
            if support_reasons or rejected or state.get("furniture_touch"):
                raise _Failure(
                    "observed_low_reach_clearance_rejected",
                    step=waypoint["step"],
                    support_reasons=support_reasons,
                    contacts=rejected,
                    furniture_touch=state.get("furniture_touch", []),
                    **support,
                )
            self.journal.trace(
                state,
                target_key=self.target_key,
                reference_progress=blend,
                reference_joints=command.tolist(),
                reference_hands=HAND_READY.tolist(),
            )
        self.runtime.ready_step(q1, HAND_READY)
        return self._observe()

    def _arm_only_escape_waypoint(self, state, supported_waypoint):
        joints = np.asarray(state["joints"], dtype=float).copy()
        # First abduct both shoulders without rotating the wrists next to the
        # hips. Coupled wrist/shoulder tracking can otherwise sweep a finger
        # inward even when the ideal synchronized reference clears the hip.
        joints[16] += self.config.ready_shoulder_clearance_rad
        joints[23] -= self.config.ready_shoulder_clearance_rad
        position = np.asarray(state["pelvis"]["pos"], dtype=float).copy()
        quaternion = np.asarray(state["pelvis"]["quat"], dtype=float).copy()
        targets, rotations = self._tool_geometry_at(joints, position, quaternion)
        return {
            **supported_waypoint,
            "segment": "ready_lateral",
            "joints": joints,
            "pelvis_position": position,
            "pelvis_quaternion_wxyz": quaternion,
            "targets_world_m": targets,
            "target_rotations": rotations,
            "position_tolerance_m": self.config.pregrasp_position_tolerance_m,
        }

    def _prepare_collision_free_transit(self, state, targets, rotations, pregrasp_targets):
        plan_transit = (
            self._plan_low_reach_transit_with_keepalive
            if self.config.approach_mode == "legacy_low_reach"
            else self._plan_supported_transit_with_keepalive
        )
        route, state = plan_transit(
            state,
            targets,
            rotations,
            pregrasp_targets,
        )
        pose_active = False
        for attempt in range(self.config.initial_escape_attempts + 1):
            _, rejected = self._collision_reasons(state, state["joints"], targets)
            if not rejected:
                return route, state, pose_active
            if attempt >= self.config.initial_escape_attempts:
                raise _Failure(
                    "initial_escape_attempts_exhausted",
                    attempts=self.config.initial_escape_attempts,
                    contacts=rejected,
                )
            supported_waypoint = route[0]
            waypoint = self._arm_only_escape_waypoint(state, supported_waypoint)
            initial_contacts = self._admit_initial_escape(state, waypoint, targets)
            # Match the legacy expert: arm-only preparation stays in PLANNER
            # mode; full-body POSE takeover follows clear route admission.
            state = self._observe()
            self._check_box_not_pushed(state)
            waypoint = self._arm_only_escape_waypoint(state, supported_waypoint)
            takeover_contacts = self._admit_initial_escape(state, waypoint, targets)
            # A shallow contact can momentarily vanish between observations.
            # Keep the exact observed starting pairs across that boundary;
            # do not relabel their recurrence as an unseen contact pair.
            for pair, depth in takeover_contacts.items():
                initial_contacts[pair] = max(initial_contacts.get(pair, 0.0), depth)
            self.journal.event(
                "ready_escape", "ready_start_contacts", initial_contacts=self._contact_depths(initial_contacts)
            )
            escape_baseline = state
            state = self._move_initial_escape(state, waypoint, targets, initial_contacts)
            self._check_observed_waypoint(escape_baseline, state, waypoint)
            self.journal.event(
                "ready_escape",
                "completed",
                attempt=attempt + 1,
                segment=waypoint["segment"],
                step=waypoint["step"],
            )
            route, state = plan_transit(
                state,
                targets,
                rotations,
                pregrasp_targets,
            )
        raise AssertionError("initial escape loop must return or fail")

    def _execute_plan(self, plan, state):
        targets = np.asarray(plan.targets_world_m, dtype=float)
        rotations = np.array(
            [
                Rotation.from_quat(np.asarray(value)[[1, 2, 3, 0]]).as_matrix()
                for value in plan.selection["target_quaternions_wxyz"]
            ]
        )
        pregrasp_targets = targets + self.config.pregrasp_clearance_m * self.box.pose.rotation[:, 2]
        route, state, pose_active = self._prepare_collision_free_transit(
            state,
            targets,
            rotations,
            pregrasp_targets,
        )
        execution_baseline = state
        if self.config.approach_mode == "legacy_low_reach":
            self._low_reach_execution_baseline = execution_baseline
        ready_gripper_q = self._observed_gripper_q(state)
        open_gripper_q = self._commanded_gripper_q(HAND_OPEN)
        previous_joints = np.asarray(state["joints"], dtype=float)
        previous_position = np.asarray(state["pelvis"]["pos"], dtype=float)
        previous_quaternion = np.asarray(state["pelvis"]["quat"], dtype=float)
        previous_hand_phase = "ready"
        for waypoint in route:
            if waypoint["hand_phase"] != previous_hand_phase:
                self._admit_hand_sweep(
                    state,
                    previous_joints,
                    targets,
                    ready_gripper_q,
                    open_gripper_q,
                    position=previous_position,
                    quaternion_wxyz=previous_quaternion,
                )
                previous_hand_phase = waypoint["hand_phase"]
            gripper_q = ready_gripper_q if waypoint["hand_phase"] == "ready" else open_gripper_q
            admit_segment = (
                self._admit_low_reach_bend if waypoint["segment"] == "low_reach_bend" else self._admit_supported_segment
            )
            admit_segment(
                state,
                previous_joints,
                previous_position,
                previous_quaternion,
                waypoint,
                targets,
                gripper_q,
            )
            previous_joints = waypoint["joints"]
            previous_position = waypoint["pelvis_position"]
            previous_quaternion = waypoint["pelvis_quaternion_wxyz"]
        self.journal.event(
            "planning",
            "transit_admitted",
            hand_phase="ready_jaw_q0_at_measured_body_state",
            static_plan_role="feasibility_witness",
            waypoint_diagnostics=[
                {
                    **{key: value for key, value in waypoint.items() if key != "target_rotations"},
                    "joints": waypoint["joints"].tolist(),
                    "pelvis_position": waypoint["pelvis_position"].tolist(),
                    "pelvis_quaternion_wxyz": waypoint["pelvis_quaternion_wxyz"].tolist(),
                    "targets_world_m": waypoint["targets_world_m"].tolist(),
                }
                for waypoint in route
            ],
        )
        if self.config.approach_mode == "legacy_low_reach":
            bend_index = next(index for index, waypoint in enumerate(route) if waypoint["segment"] == "low_reach_bend")
            for clearance in route[:bend_index]:
                state = self._move_planner_clearance(state, clearance, targets, execution_baseline)
                self._check_observed_waypoint(execution_baseline, state, clearance)
            route, state = self._plan_low_reach_transit_with_keepalive(state, targets, rotations, pregrasp_targets)
            bend_index = next(index for index, waypoint in enumerate(route) if waypoint["segment"] == "low_reach_bend")
            route = route[bend_index:]
            previous_joints = np.asarray(state["joints"], dtype=float)
            previous_position = np.asarray(state["pelvis"]["pos"], dtype=float)
            previous_quaternion = np.asarray(state["pelvis"]["quat"], dtype=float)
            for waypoint in route:
                admit_segment = (
                    self._admit_low_reach_bend
                    if waypoint["segment"] == "low_reach_bend"
                    else self._admit_supported_segment
                )
                admit_segment(
                    state,
                    previous_joints,
                    previous_position,
                    previous_quaternion,
                    waypoint,
                    targets,
                    ready_gripper_q if waypoint["hand_phase"] == "ready" else open_gripper_q,
                )
                previous_joints = waypoint["joints"]
                previous_position = waypoint["pelvis_position"]
                previous_quaternion = waypoint["pelvis_quaternion_wxyz"]

        handoff_reference = None
        if not pose_active:
            handoff_reference = self.runtime.enter_pose(
                self.config.maximum_joint_acceleration_rps2,
                (self.planner.bimanual.lo, self.planner.bimanual.hi),
            )
        state = self._observe()
        if self.config.approach_mode == "legacy_low_reach":
            if handoff_reference is None:
                raise _Failure("missing_pose_handoff_reference")
            bend_index = next(index for index, waypoint in enumerate(route) if waypoint["segment"] == "low_reach_bend")
            low_root = self._low_reach_root_quaternion.copy()
            handoff_root = np.asarray(handoff_reference[1], dtype=float).copy()
            self._low_reach_reference_joints = np.asarray(handoff_reference[0], dtype=float).copy()
            self._low_reach_root_quaternion = handoff_root
            for clearance in route[:bend_index]:
                clearance = dict(clearance)
                clearance["pelvis_position"] = np.asarray(state["pelvis"]["pos"], dtype=float)
                clearance["pelvis_quaternion_wxyz"] = handoff_root
                self._admit_low_reach_bend(
                    state,
                    self._command_joints(state),
                    clearance["pelvis_position"],
                    handoff_root,
                    clearance,
                    targets,
                    self._observed_gripper_q(state),
                )
                state = self._move_reference(
                    "low_reach_arm_clearance",
                    state,
                    clearance["joints"],
                    HAND_READY,
                    target_root_quat=handoff_root,
                )
                self._check_observed_waypoint(execution_baseline, state, clearance)
            bend = route[bend_index]
            self._low_reach_root_quaternion = low_root
            self._admit_low_reach_bend(
                state,
                self._command_joints(state),
                np.asarray(state["pelvis"]["pos"], dtype=float),
                handoff_root,
                bend,
                targets,
                self._observed_gripper_q(state),
            )
            state = self._move_low_reach_bend(
                state,
                (self._command_joints(state), handoff_root),
                bend,
            )
            route = route[bend_index + 1 :]
        previous_hand_phase = "ready"
        for waypoint in route:
            if waypoint["hand_phase"] != previous_hand_phase:
                state = self._open_at_clearance(state, self._command_joints(state), targets)
                previous_hand_phase = waypoint["hand_phase"]
            phase = {
                "low_reach_arm_clearance": "low_reach_arm_clearance",
                "low_reach_bend": "low_reach_bend",
                "raised_front": "transit_raise",
                "above_rim": "transit_above",
                "pregrasp": "pregrasp",
                "align": "align",
            }[waypoint["segment"]]
            hands = HAND_READY if waypoint["hand_phase"] == "ready" else HAND_OPEN
            if self.config.approach_mode in ("planner_arms", "legacy_low_reach"):
                waypoint = dict(waypoint)
                waypoint["joints"] = self._solve_tracking_at_observed_root(
                    state,
                    waypoint["targets_world_m"],
                    waypoint["target_rotations"],
                    self._command_joints(state),
                )
                waypoint["pelvis_position"] = np.asarray(state["pelvis"]["pos"], dtype=float)
                waypoint["pelvis_quaternion_wxyz"] = self._command_root_quaternion(state)
                self._admit_supported_segment(
                    state,
                    np.asarray(state["joints"], dtype=float),
                    waypoint["pelvis_position"],
                    waypoint["pelvis_quaternion_wxyz"],
                    waypoint,
                    targets,
                    self._observed_gripper_q(state),
                )
            state = self._move_reference(
                phase,
                state,
                waypoint["joints"],
                hands,
                target_root_quat=waypoint["pelvis_quaternion_wxyz"],
            )
            self._check_observed_waypoint(execution_baseline, state, waypoint)
        state = self._closed_loop_align(state, targets, rotations)
        # From here onward targets and hand assignment are immutable.
        self.journal.event("align", "grasp_locked", candidate=plan.selected_candidate)
        state = self._close(state, targets, rotations)
        self._probe(state, targets, rotations)

    def _closed_loop_align(self, state, targets, rotations):
        began, valid_since = float(state["t"]), None
        reference = self._command_joints(state)
        previous_sim_time = float(state["t"])
        while float(state["t"]) - began <= self.config.align_timeout_s:
            self._check_box_not_pushed(state)
            positions, actual_rotations = self._tool_geometry(state, state["joints"])
            position_errors = np.linalg.norm(positions - targets, axis=1)
            rotation_errors = np.array(
                [
                    Rotation.from_matrix(wanted @ found.T).magnitude()
                    for found, wanted in zip(actual_rotations, rotations)
                ]
            )
            valid = (
                float(position_errors.max()) <= self.config.align_position_tolerance_m
                and float(rotation_errors.max()) <= self.config.align_orientation_tolerance_rad
            )
            valid_since = float(state["t"]) if valid and valid_since is None else valid_since
            if not valid:
                valid_since = None
            self.journal.trace(
                state,
                target_key=self.target_key,
                position_errors_m=position_errors.tolist(),
                orientation_errors_rad=rotation_errors.tolist(),
                reference_joints=reference.tolist(),
                reference_hands=HAND_OPEN.tolist(),
            )
            if valid_since is not None and float(state["t"]) - valid_since >= self.config.align_hold_s:
                return state
            solved = self._solve_tracking_at_observed_root(state, targets, rotations, reference)
            dt = float(state["t"]) - previous_sim_time
            if dt <= 0:
                if self.config.approach_mode == "legacy_low_reach":
                    self.runtime.pose_step(
                        reference,
                        self._command_root_quaternion(state),
                        HAND_OPEN,
                        reference_velocity=np.zeros(29, dtype=float),
                    )
                else:
                    self.runtime.pose_step(reference, self._command_root_quaternion(state), HAND_OPEN)
                state = self._observe()
                continue
            step = self.config.maximum_joint_speed_rps * dt
            previous_reference = reference.copy()
            reference += np.clip(solved - reference, -step, step)
            _, rejected = self._collision_reasons(state, reference, targets)
            if rejected:
                raise _Failure("align_collision", contacts=rejected)
            if self.config.approach_mode == "legacy_low_reach":
                self.runtime.pose_step(
                    reference,
                    self._command_root_quaternion(state),
                    HAND_OPEN,
                    reference_velocity=(reference - previous_reference) / dt,
                )
            else:
                self.runtime.pose_step(reference, self._command_root_quaternion(state), HAND_OPEN)
            state = self._observe()
            if self.config.approach_mode == "legacy_low_reach":
                self._low_reach_reference_joints = reference.copy()
            previous_sim_time = float(state["t"])
        raise _Failure("align_timeout")

    def _close(self, state, targets, rotations):
        self.journal.event("close", "closure_started")
        began = float(state["t"])
        root_quat = self._command_root_quaternion(state)
        q = self._command_joints(state)
        while float(state["t"]) - began < self.config.close_s:
            self._check_box_not_pushed(state)
            positions, actual_rotations = self._tool_geometry(state, state["joints"])
            if float(np.max(np.linalg.norm(positions - targets, axis=1))) > self.config.align_position_tolerance_m:
                raise _Failure("capture_lost_during_close")
            normal_errors = [
                float(Rotation.from_matrix(wanted @ found.T).magnitude())
                for found, wanted in zip(actual_rotations, rotations)
            ]
            if max(normal_errors) > self.config.align_orientation_tolerance_rad:
                raise _Failure("jaw_normal_error_during_close", errors_rad=normal_errors)
            alpha = np.clip((float(state["t"]) - began) / self.config.close_s, 0.0, 1.0)
            hands = HAND_OPEN + alpha * (HAND_CLOSED - HAND_OPEN)
            self.runtime.pose_step(q, root_quat, hands)
            state = self._observe()
            obj = state["objects"][self.target_key]
            self.journal.trace(
                state,
                closure=float(alpha),
                object={"touching": obj.get("touching", []), "contact_links": obj.get("touch_links", [])},
                reference_joints=q.tolist(),
                reference_root_quaternion_wxyz=root_quat.tolist(),
                reference_hands=np.asarray(hands).tolist(),
            )
        touching = set(state["objects"][self.target_key].get("touching", []))
        if touching != {"left", "right"}:
            raise _Failure("empty_or_unilateral_grip", touching=sorted(touching))
        hold_since = float(state["t"])
        while float(state["t"]) - hold_since < self.config.contact_hold_s:
            self.runtime.pose_step(q, root_quat, HAND_CLOSED)
            state = self._observe()
            if set(state["objects"][self.target_key].get("touching", [])) != {"left", "right"}:
                raise _Failure("lost_bilateral_contact_before_probe")
        return state

    def _probe(self, state, targets, rotations):
        self.journal.event("probe", "probe_started", requested_lift_m=self.config.probe_lift_m)
        obj0 = state["objects"][self.target_key]
        object_position0 = np.asarray(obj0["pos"], dtype=float)
        object_quaternion0 = np.asarray(obj0["quat"], dtype=float)
        palms0 = np.array([state["palms"][side]["pos"] for side in ("left", "right")], dtype=float)
        relative0 = palms0 - object_position0
        probe_targets = targets.copy()
        probe_targets[:, 2] += self.config.probe_lift_m
        q0 = self._command_joints(state)
        q_target = self._solve_tracking_at_observed_root(state, probe_targets, rotations, q0)
        began = float(state["t"])
        root_quat = self._command_root_quaternion(state)
        joint_delta = float(np.max(np.abs(q_target - q0)))
        move_duration = max(
            0.4,
            1.5 * joint_delta / self.config.maximum_joint_speed_rps,
            math.sqrt(6.0 * joint_delta / self.config.maximum_joint_acceleration_rps2),
        )
        if move_duration + self.config.contact_hold_s > self.config.probe_timeout_s:
            raise _Failure(
                "probe_reference_exceeds_timeout",
                required_s=move_duration + self.config.contact_hold_s,
                timeout_s=self.config.probe_timeout_s,
            )
        best_rise = 0.0
        valid_since = None
        while float(state["t"]) - began <= self.config.probe_timeout_s:
            u = np.clip((float(state["t"]) - began) / move_duration, 0.0, 1.0)
            blend = u * u * (3.0 - 2.0 * u)
            command = q0 + blend * (q_target - q0)
            current_obj = state["objects"][self.target_key]
            current_box = OpenBoxGeometry(
                self.box.size_m,
                self.box.wall_thickness_m,
                RigidPose(current_obj["pos"], current_obj["quat"]),
                local_com_m=self.box.local_com_m,
                corner_clearance_m=self.box.corner_clearance_m,
                forbidden_intervals=self.box.forbidden_intervals,
            )
            _, rejected = self._collision_reasons(state, command, probe_targets, box=current_box)
            if rejected or state.get("furniture_touch"):
                raise _Failure(
                    "probe_collision",
                    contacts=rejected,
                    furniture_touch=state.get("furniture_touch", []),
                )
            if self.config.approach_mode == "legacy_low_reach":
                velocity = (6.0 * u * (1.0 - u) / move_duration) * (q_target - q0)
                self.runtime.pose_step(
                    command,
                    root_quat,
                    HAND_CLOSED,
                    reference_velocity=velocity,
                )
                self._low_reach_reference_joints = command.copy()
            else:
                self.runtime.pose_step(command, root_quat, HAND_CLOSED)
            state = self._observe()
            obj = state["objects"][self.target_key]
            touching = set(obj.get("touching", []))
            position = np.asarray(obj["pos"], dtype=float)
            palms = np.array([state["palms"][side]["pos"] for side in ("left", "right")], dtype=float)
            rise = float(position[2] - object_position0[2])
            slip = float(np.max(np.linalg.norm((palms - position) - relative0, axis=1)))
            tilt = _quaternion_angle(obj["quat"], object_quaternion0)
            best_rise = max(best_rise, rise)
            self.journal.trace(
                state,
                object={
                    "rise_m": rise,
                    "touching": sorted(touching),
                    "contact_links": obj.get("touch_links", []),
                    "slip_m": slip,
                    "tilt_rad": tilt,
                },
                reference_joints=command.tolist(),
                reference_root_quaternion_wxyz=root_quat.tolist(),
                reference_hands=HAND_CLOSED.tolist(),
            )
            if touching != {"left", "right"}:
                raise _Failure("lost_contact_during_probe", touching=sorted(touching), rise_m=rise)
            if slip > self.config.maximum_slip_m:
                raise _Failure("grasp_slip", slip_m=slip, rise_m=rise)
            if tilt > self.config.maximum_object_tilt_rad:
                raise _Failure("object_tilt", tilt_rad=tilt, rise_m=rise)
            accepted = rise >= self.config.minimum_object_rise_m
            valid_since = float(state["t"]) if accepted and valid_since is None else valid_since
            if not accepted:
                valid_since = None
            if valid_since is not None and float(state["t"]) - valid_since >= self.config.contact_hold_s:
                self.journal.event(
                    "probe",
                    "measured_rise",
                    rise_m=rise,
                    touching=sorted(touching),
                    contact_links=obj.get("touch_links", []),
                    slip_m=slip,
                    tilt_rad=tilt,
                )
                return
        raise _Failure("probe_no_measured_rise", best_rise_m=best_rise)


__all__ = [
    "BoxRimExecutionConfig",
    "BoxRimExecutor",
    "BoxRimRuntime",
    "HAND_READY",
    "SonicBoxRimRuntime",
    "execution_config_from_dict",
]
