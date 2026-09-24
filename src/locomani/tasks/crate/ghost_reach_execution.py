"""Bounded dynamic reach trials against paired virtual world targets.

The virtual target supplies observations only.  It has no MuJoCo body, geom,
contact, attachment, or force.  Physical commands go through the existing
``CoordinatedReach`` whole-body controller with open hands and no locomotion
interface.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.backends.sonic.agent_server import HAND_OPEN
from locomani.tasks.crate.bimanual_ik import BimanualIK
from locomani.tasks.crate.carry_config import CrateCarryConfig
from locomani.tasks.crate.coordinated_reach import CoordinatedReach, CoordinatedTarget
from locomani.tasks.crate.perception import CrateEstimate
from locomani.tasks.crate.visual_config import CrateVisualConfig

READY_WAIST = (0.0, 0.0, 0.0)
READY_LEFT_ARM = (-0.35, 0.2, 0.0, 0.8, 0.0, 0.0, 0.0)
READY_RIGHT_ARM = (-0.35, -0.2, 0.0, 0.8, 0.0, 0.0, 0.0)
DEFAULT_BODYWARD_TILT_DEG = 30.0
PALM_PITCH_DEG = 90.0 - DEFAULT_BODYWARD_TILT_DEG
PREVIEW_TILT_DEG = DEFAULT_BODYWARD_TILT_DEG


def _finite_vector(value, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {length} finite values")
    return result


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class GhostReachCriteria:
    target_timeout_sim_s: float = 12.0
    success_hold_sim_s: float = 0.5
    position_tolerance_m: float = 0.020
    orientation_tolerance_deg: float = 15.0
    foot_horizontal_drift_m: float = 0.030
    foot_vertical_lift_m: float = 0.030
    wall_watchdog_s: float = 5.0

    def __post_init__(self):
        values = np.asarray(list(asdict(self).values()), dtype=float)
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("experiment criteria must be finite and positive")
        if self.success_hold_sim_s >= self.target_timeout_sim_s:
            raise ValueError("success_hold_sim_s must be shorter than target_timeout_sim_s")
        if self.orientation_tolerance_deg > 180:
            raise ValueError("orientation_tolerance_deg must not exceed 180")


@dataclass(frozen=True)
class GhostReachCase:
    index: int
    label_zh: str
    midpoint_robot_m: tuple[float, float, float]
    separation_m: float
    line_yaw_deg: float
    source_centres_robot_m: tuple[tuple[float, float, float], tuple[float, float, float]]
    bodyward_tilt_deg: float = DEFAULT_BODYWARD_TILT_DEG

    @classmethod
    def from_source(cls, raw: dict) -> "GhostReachCase":
        if not isinstance(raw, dict):
            raise ValueError("each source sample must be an object")
        required = {"index", "label_zh", "parameters", "capture_centres_xyz", "orientation_endpoints"}
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(f"source sample missing fields: {', '.join(missing)}")
        params = raw["parameters"]
        centres = raw["capture_centres_xyz"]
        if not isinstance(params, dict) or not isinstance(centres, dict):
            raise ValueError("source parameters and capture_centres_xyz must be objects")
        midpoint = np.array(
            [
                params["pair_midpoint_x_m"],
                params["pair_midpoint_y_m"],
                params["ground_height_z_m"],
            ],
            dtype=float,
        )
        separation = float(params["pair_separation_m"])
        yaw_deg = float(params["pair_line_yaw_deg"])
        bodyward_tilt_deg = float(raw.get("bodyward_tilt_deg", DEFAULT_BODYWARD_TILT_DEG))
        left = _finite_vector(centres["left"], 3, "capture_centres_xyz.left")
        right = _finite_vector(centres["right"], 3, "capture_centres_xyz.right")
        if not np.isfinite(midpoint).all() or not np.isfinite((separation, yaw_deg, bodyward_tilt_deg)).all():
            raise ValueError("source sample parameters must be finite")
        if separation <= 0:
            raise ValueError("pair_separation_m must be positive")
        if not 0.0 <= bodyward_tilt_deg <= 45.0:
            raise ValueError("bodyward_tilt_deg must be in [0, 45]")
        yaw = math.radians(yaw_deg)
        line = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        expected = np.array([midpoint + 0.5 * separation * line, midpoint - 0.5 * separation * line])
        supplied = np.array([left, right])
        if not np.allclose(supplied, expected, atol=1e-10):
            raise ValueError(f"source sample {raw['index']} capture centres disagree with parameters")
        return cls(
            index=int(raw["index"]),
            label_zh=str(raw["label_zh"]),
            midpoint_robot_m=tuple(map(float, midpoint)),
            separation_m=separation,
            line_yaw_deg=yaw_deg,
            source_centres_robot_m=tuple(tuple(map(float, row)) for row in supplied),
            bodyward_tilt_deg=bodyward_tilt_deg,
        )


@dataclass(frozen=True)
class WorldGhostTarget:
    centres_m: np.ndarray
    midpoint_m: np.ndarray
    capture_rotation: np.ndarray
    permitted_tool_rotation: np.ndarray
    initialized_robot_origin_xy_m: np.ndarray
    initialized_robot_yaw_rad: float

    def __post_init__(self):
        if np.asarray(self.centres_m).shape != (2, 3):
            raise ValueError("centres_m must be 2x3")
        for name in ("capture_rotation", "permitted_tool_rotation"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3, 3) or not np.allclose(value.T @ value, np.eye(3), atol=1e-7):
                raise ValueError(f"{name} must be a rotation")


def world_target(case: GhostReachCase, initial_state: dict) -> WorldGhostTarget:
    """Freeze one source sample in world after measured ready initialization."""
    origin_xy = _finite_vector(initial_state["pelvis"]["pos"], 3, "pelvis.pos")[:2]
    yaw = float(
        initial_state["pelvis"].get(
            "yaw",
            Rotation.from_quat(
                _finite_vector(initial_state["pelvis"]["quat"], 4, "pelvis.quat")[[1, 2, 3, 0]]
            ).as_euler("xyz")[2],
        )
    )
    robot_rotation = Rotation.from_euler("z", yaw).as_matrix()
    source = np.asarray(case.source_centres_robot_m, dtype=float)
    centres = source @ robot_rotation.T
    centres[:, :2] += origin_xy
    # Source z is a height above the world floor, not a pelvis-relative offset.
    centres[:, 2] = source[:, 2]
    midpoint = centres.mean(axis=0)
    line = centres[0] - centres[1]
    line /= np.linalg.norm(line)
    forward = robot_rotation @ np.array(
        [math.cos(math.radians(case.line_yaw_deg)), math.sin(math.radians(case.line_yaw_deg)), 0.0]
    )
    capture = np.column_stack([line, -forward, [0.0, 0.0, 1.0]])
    world_line_yaw = math.atan2(-line[0], line[1])
    permitted = (
        Rotation.from_euler("z", world_line_yaw).as_matrix()
        @ Rotation.from_euler("y", 90.0 - case.bodyward_tilt_deg, degrees=True).as_matrix()
    )
    return WorldGhostTarget(
        centres_m=centres,
        midpoint_m=midpoint,
        capture_rotation=capture,
        permitted_tool_rotation=permitted,
        initialized_robot_origin_xy_m=origin_xy,
        initialized_robot_yaw_rad=yaw,
    )


def synthetic_estimate(target: WorldGhostTarget, state: dict, timestamp: float) -> CrateEstimate:
    """Express the fixed world target in the current measured pelvis frame."""
    base = Rotation.from_quat(_finite_vector(state["pelvis"]["quat"], 4, "pelvis.quat")[[1, 2, 3, 0]]).as_matrix()
    origin = _finite_vector(state["pelvis"]["pos"], 3, "pelvis.pos")
    return CrateEstimate(
        position=base.T @ (target.midpoint_m - origin),
        rotation=base.T @ target.capture_rotation,
        timestamp=float(timestamp),
        reprojection_px=0.0,
        detections={"source": "fixed_world_synthetic_target"},
    )


@dataclass(frozen=True)
class GhostReachControlConfig:
    visual: CrateVisualConfig
    carry: CrateCarryConfig


ENVELOPE_FIELDS = frozenset(
    {
        "waist_pitch_max",
        "pelvis_pitch_max",
        "pelvis_height_min",
        "com_shift_max",
        "com_free_radius",
        "knee_max",
        "pelvis_forward_max",
        "base_anchor",
        "pelvis_height_target",
    }
)
TRACKING_FIELDS = frozenset({"tracking_gain", "tracking_cap"})


def original_reach_control_config(
    bodyward_tilt_deg: float = DEFAULT_BODYWARD_TILT_DEG,
    envelope: dict | None = None,
) -> GhostReachControlConfig:
    """Original assisted low-reach profile used by the bimanual expert.

    ``envelope`` overrides the whole-body support bounds only (see ENVELOPE_FIELDS);
    it never touches the visual profile, timing or acceptance thresholds.
    """
    if not math.isfinite(bodyward_tilt_deg) or not 0.0 <= bodyward_tilt_deg <= 45.0:
        raise ValueError("bodyward_tilt_deg must be finite and in [0, 45]")
    envelope = {} if envelope is None else dict(envelope)
    tracking = {name: envelope.pop(name) for name in list(envelope) if name in TRACKING_FIELDS}
    unknown = set(envelope) - ENVELOPE_FIELDS
    if unknown:
        raise ValueError(f"envelope has unsupported fields: {sorted(unknown)}")
    config = GhostReachControlConfig(
        visual=CrateVisualConfig(
            pose_source="assisted",
            palm_pitch_deg=90.0 - bodyward_tilt_deg,
            settle_seconds=0.3,
            stand_off=0.39,
            stand_lateral_tolerance=0.14,
            walk_brake_margin=0.10,
            pregrasp_above=0.04,
            approach_speed=0.16,
            tracking_gain=0.70,
            tracking_cap=0.04,
            close_seconds=0.8,
            position_jitter=0.05,
            yaw_jitter_deg=20.0,
        ),
        carry=CrateCarryConfig(
            reach_control="whole_body",
            waist_dofs="all",
            waist_pitch_max=0.52,
        ),
    )
    if envelope:
        config = replace(config, carry=replace(config.carry, **envelope))
    if tracking:
        config = replace(config, visual=replace(config.visual, **tracking))
    return config


class GhostReachRuntime(Protocol):
    walk_command_count: int

    def observe(self) -> dict: ...

    def enter_pose(self, acceleration, joint_limits) -> None: ...

    def pose_step(self, joints, root_quaternion_wxyz, hands, *, reference_velocity=None) -> None: ...

    def safe_stop(self) -> None: ...


def foot_positions(ik, state: dict) -> np.ndarray:
    model, data, mj = ik.model, ik.data, ik._mj
    data.qpos[:] = model.qpos0
    data.qpos[ik.base_qadr : ik.base_qadr + 3] = state["pelvis"]["pos"]
    data.qpos[ik.base_qadr + 3 : ik.base_qadr + 7] = state["pelvis"]["quat"]
    data.qpos[ik.qadr] = state["joints"]
    mj.mj_kinematics(model, data)
    feet = [model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
    return data.xpos[feet].copy()


class MeasuredCollisionChecker:
    """Classify measured floor support, robot self, and non-support contacts."""

    def __init__(self, ik):
        self.mj = ik._mj
        self.model = copy.deepcopy(ik.model)
        self.data = self.mj.MjData(self.model)
        self.base_qadr = ik.base_qadr
        self.joint_qadr = ik.qadr.copy()
        pelvis = self.model.body("pelvis").id
        self.robot_root = int(self.model.body_rootid[pelvis])
        self.hand_qadr: list[int] = []
        hand_suffixes = (
            "gripper_1",
            "gripper_2",
            "thumb_0",
            "thumb_1",
            "thumb_2",
            "middle_0",
            "middle_1",
            "index_0",
            "index_1",
        )
        for side in ("left", "right"):
            for suffix in hand_suffixes:
                name = f"{side}_hand_{suffix}_joint"
                jid = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_JOINT, name)
                if jid >= 0:
                    self.hand_qadr.append(int(self.model.jnt_qposadr[jid]))

    def check(self, state: dict) -> tuple[list[dict], list[dict]]:
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[self.base_qadr : self.base_qadr + 3] = state["pelvis"]["pos"]
        self.data.qpos[self.base_qadr + 3 : self.base_qadr + 7] = state["pelvis"]["quat"]
        self.data.qpos[self.joint_qadr] = state["joints"]
        hand_values = []
        for side in ("left", "right"):
            hand_values.extend(state.get("palms", {}).get(side, {}).get("hand_q", []))
        if self.hand_qadr and len(hand_values) == len(self.hand_qadr):
            self.data.qpos[self.hand_qadr] = hand_values
        self.mj.mj_forward(self.model, self.data)
        allowed, rejected = [], []
        for contact in self.data.contact[: self.data.ncon]:
            geom_ids = (int(contact.geom1), int(contact.geom2))
            body_ids = tuple(int(self.model.geom_bodyid[geom]) for geom in geom_ids)
            robot = tuple(int(self.model.body_rootid[body]) == self.robot_root for body in body_ids)
            if not any(robot):
                continue
            geom_names = tuple(self.model.geom(geom).name or f"geom_{geom}" for geom in geom_ids)
            body_names = tuple(self.model.body(body).name or f"body_{body}" for body in body_ids)
            diagnostic = {
                "classification": "robot_self_collision" if all(robot) else "robot_environment_collision",
                "geom_1": geom_names[0],
                "geom_2": geom_names[1],
                "body_1": body_names[0],
                "body_2": body_names[1],
                "distance_m": float(contact.dist),
            }
            support = False
            if robot.count(True) == 1:
                robot_index = 0 if robot[0] else 1
                other_index = 1 - robot_index
                other_geom = geom_ids[other_index]
                is_plane = int(self.model.geom_type[other_geom]) == int(self.mj.mjtGeom.mjGEOM_PLANE)
                support = is_plane and "ankle_roll_link" in body_names[robot_index]
            if support:
                diagnostic["classification"] = "diagnostic_floor_support_contact"
                allowed.append(diagnostic)
            else:
                rejected.append(diagnostic)
        return allowed, rejected


def _measured_tools(solver: BimanualIK, state: dict) -> tuple[np.ndarray, np.ndarray]:
    base = Rotation.from_quat(_finite_vector(state["pelvis"]["quat"], 4, "pelvis.quat")[[1, 2, 3, 0]]).as_matrix()
    origin = _finite_vector(state["pelvis"]["pos"], 3, "pelvis.pos")
    positions, rotations = solver.tool_poses(_finite_vector(state["joints"], 29, "joints"))
    return positions @ base.T + origin, base @ rotations


def _orientation_errors_deg(measured: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.degrees(np.array([(Rotation.from_matrix(target @ value.T)).magnitude() for value in measured]))


class GhostReachExecutor:
    """Run one initialized target without reset, recording, or stack ownership."""

    def __init__(
        self,
        solver: BimanualIK,
        runtime: GhostReachRuntime,
        *,
        criteria: GhostReachCriteria | None = None,
        control_config: GhostReachControlConfig | None = None,
        collision_checker: MeasuredCollisionChecker | None = None,
        controller_factory: Callable = CoordinatedReach,
        wall_clock: Callable[[], float] = time.monotonic,
        event_clock: Callable[[], float] = time.time,
    ):
        self.solver = solver
        self.runtime = runtime
        self.criteria = criteria or GhostReachCriteria()
        self.control_config = control_config or original_reach_control_config()
        self.collision_checker = collision_checker
        self.controller_factory = controller_factory
        self.wall_clock = wall_clock
        self.event_clock = event_clock

    def run_case(self, case: GhostReachCase, initial_state: dict) -> dict:
        criteria = self.criteria
        control_config = replace(
            self.control_config,
            visual=replace(self.control_config.visual, palm_pitch_deg=90.0 - case.bodyward_tilt_deg),
        )
        target = world_target(case, initial_state)
        initial_centres = target.centres_m.copy()
        initial_feet = foot_positions(self.solver.ik, initial_state)
        initial_tools, _ = _measured_tools(self.solver, initial_state)
        local_points = (
            (0.5 * case.separation_m, 0.0, 0.0),
            (-0.5 * case.separation_m, 0.0, 0.0),
        )
        controller_target = CoordinatedTarget(
            object_key="__virtual_ghost_target__",
            grasp_points_local=local_points,
            information_source="fixed_world_synthetic_target",
            preserve_point_order=True,
            reach_only=True,
        )
        controller = self.controller_factory(
            self.solver,
            control_config,
            initial_state,
            reference_joints=np.asarray(initial_state["joints"], dtype=float),
            reference_quaternion=np.asarray(initial_state["pelvis"]["quat"], dtype=float),
            target=controller_target,
        )
        self.runtime.enter_pose(
            control_config.visual.joint_acceleration,
            (np.asarray(self.solver.lo), np.asarray(self.solver.hi)),
        )
        began_sim = previous_sim = float(initial_state["t"])
        progress_wall = self.wall_clock()
        success_since = None
        traces: list[dict] = []
        final_state = initial_state
        final_reason = "target_timeout"
        status = "failed"
        last_seq = None
        while True:
            state = self.runtime.observe()
            final_state = state
            sim_time = float(state["t"])
            wall_now = self.wall_clock()
            terminal_reason = None
            if sim_time < previous_sim - 1e-9:
                terminal_reason = "simulation_clock_reversed"
            if sim_time > previous_sim + 1e-9:
                previous_sim, progress_wall = sim_time, wall_now
            elif wall_now - progress_wall >= criteria.wall_watchdog_s:
                terminal_reason = terminal_reason or "wall_watchdog_timeout"
            if state.get("fallen", False):
                terminal_reason = terminal_reason or "robot_fallen"
            feet = foot_positions(self.solver.ik, state)
            foot_horizontal = np.linalg.norm(feet[:, :2] - initial_feet[:, :2], axis=1)
            foot_vertical = np.abs(feet[:, 2] - initial_feet[:, 2])
            if np.max(foot_horizontal) > criteria.foot_horizontal_drift_m:
                terminal_reason = terminal_reason or "stance_drift_horizontal"
            if np.max(foot_vertical) > criteria.foot_vertical_lift_m:
                terminal_reason = terminal_reason or "stance_drift_vertical"
            measured_tools, measured_rotations = _measured_tools(self.solver, state)
            position_errors = np.linalg.norm(measured_tools - target.centres_m, axis=1)
            orientation_errors = _orientation_errors_deg(measured_rotations, target.permitted_tool_rotation)
            allowed_contacts: list[dict] = []
            rejected_contacts: list[dict] = []
            if self.collision_checker is not None:
                allowed_contacts, rejected_contacts = self.collision_checker.check(state)
                if rejected_contacts:
                    terminal_reason = terminal_reason or "non_support_collision"
            within = bool(
                np.max(position_errors) <= criteria.position_tolerance_m
                and np.max(orientation_errors) <= criteria.orientation_tolerance_deg
            )
            success_since = sim_time if within and success_since is None else success_since if within else None
            held_success = success_since is not None and sim_time - success_since >= criteria.success_hold_sim_s
            if sim_time - began_sim >= criteria.target_timeout_sim_s and not held_success:
                terminal_reason = terminal_reason or "target_timeout"
            if terminal_reason is None and not held_success:
                estimate = synthetic_estimate(target, state, self.event_clock())
                command = controller.step(
                    self.event_clock(),
                    estimate,
                    state,
                    [float("inf"), float("inf")],
                    motion_time=sim_time,
                )
            else:
                command = controller.result(terminal_reason or "measured_reach_held", True)
            trace = {
                "seq": state.get("seq"),
                "sim_time_s": sim_time,
                "pelvis_position_m": state["pelvis"]["pos"],
                "pelvis_quaternion_wxyz": state["pelvis"]["quat"],
                "measured_joints": state["joints"],
                "measured_tools_world_m": measured_tools,
                "reference": {
                    "pelvis_position_m": controller.x[:3],
                    "pelvis_quaternion_wxyz": command.root_quaternion,
                    "joints": command.joints,
                    "joint_velocity": command.velocity,
                },
                "fixed_targets_world_m": target.centres_m,
                "position_error_m": position_errors,
                "orientation_error_deg": orientation_errors,
                "feet_world_m": feet,
                "foot_horizontal_drift_m": foot_horizontal,
                "foot_vertical_lift_m": foot_vertical,
                "success_window_started_sim_s": success_since,
                "controller_phase": command.phase,
                "controller_reason": command.reason,
                "controller_diagnostics": controller.diagnostics,
                "allowed_contacts": allowed_contacts,
                "rejected_contacts": rejected_contacts,
                "terminal_reason": terminal_reason
                if terminal_reason is not None
                else ("measured_reach_held" if held_success else None),
            }
            if state.get("seq") != last_seq:
                traces.append(_json_safe(trace))
                last_seq = state.get("seq")
            if not np.array_equal(target.centres_m, initial_centres):
                raise RuntimeError("fixed world target mutated during trial")
            if terminal_reason is not None:
                final_reason = terminal_reason
                break
            if held_success:
                status, final_reason = "success", "measured_reach_held"
                break
            if command.finished:
                final_reason = f"controller_{command.reason or 'finished'}"
                break
            self.runtime.pose_step(
                command.joints,
                command.root_quaternion,
                HAND_OPEN,
                reference_velocity=command.velocity,
            )
            if self.runtime.walk_command_count:
                final_reason = "walking_command_forbidden"
                break
        try:
            self.runtime.safe_stop()
        except Exception as error:  # noqa: BLE001 - retain physical result and report stop failure
            stop_error = f"{type(error).__name__}: {error}"
        else:
            stop_error = None
        final_feet = foot_positions(self.solver.ik, final_state)
        final_tools, _ = _measured_tools(self.solver, final_state)
        measured_tool_displacement = np.linalg.norm(final_tools - initial_tools, axis=1)
        return _json_safe(
            {
                "index": case.index,
                "label_zh": case.label_zh,
                "status": status,
                "reason": final_reason,
                "initialization": {
                    "robot_origin_xy_m": target.initialized_robot_origin_xy_m,
                    "robot_yaw_rad": target.initialized_robot_yaw_rad,
                    "initial_feet_world_m": initial_feet,
                },
                "target": {
                    "source_centres_robot_m": case.source_centres_robot_m,
                    "fixed_centres_world_m": target.centres_m,
                    "permitted_tool_rotation": target.permitted_tool_rotation,
                    "controller_palm_pitch_deg": 90.0 - case.bodyward_tilt_deg,
                    "preview_bodyward_tilt_deg": case.bodyward_tilt_deg,
                    "tested_orientation_scope": "single fixed per-case tilt within allowed 0..45deg",
                },
                "final_feet_world_m": final_feet,
                "walk_command_count": self.runtime.walk_command_count,
                "execution_started": True,
                "execution_start_seq": initial_state.get("seq"),
                "execution_end_seq": final_state.get("seq"),
                "execution_start_sim_time_s": began_sim,
                "execution_end_sim_time_s": float(final_state["t"]),
                "initial_tools_world_m": initial_tools,
                "final_tools_world_m": final_tools,
                "measured_tool_displacement_m": measured_tool_displacement,
                "max_measured_tool_displacement_m": float(np.max(measured_tool_displacement)),
                "sim_duration_s": float(final_state["t"]) - began_sim,
                "stop_error": stop_error,
                "traces": traces,
            }
        )


def load_source_cases(path: Path) -> tuple[list[GhostReachCase], dict]:
    payload = path.read_bytes()
    raw = json.loads(payload)
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("source targets schema_version must be 1")
    samples = raw.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("source targets must contain a non-empty sample list")
    if len(samples) > 1_000_000:
        raise ValueError("source targets may contain at most 1000000 samples")
    cases = [GhostReachCase.from_source(sample) for sample in samples]
    if [case.index for case in cases] != list(range(1, len(cases) + 1)):
        raise ValueError("source sample indices must be unique, contiguous 1..N, and in order")
    identity = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "schema_version": raw["schema_version"],
        "seed": raw.get("seed"),
        "proposal_status": raw.get("proposal_status"),
        "frame": raw.get("frame"),
        "sample_count": len(cases),
    }
    return cases, identity


def summarize_cases(cases: list[dict]) -> dict:
    counts: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for case in cases:
        status = str(case["status"])
        reason = str(case["reason"])
        counts[status] = counts.get(status, 0) + 1
        reasons[reason] = reasons.get(reason, 0) + 1
    return {"total": len(cases), "status_counts": counts, "reason_counts": reasons}


__all__ = [
    "DEFAULT_BODYWARD_TILT_DEG",
    "GhostReachCase",
    "GhostReachCriteria",
    "GhostReachExecutor",
    "MeasuredCollisionChecker",
    "PALM_PITCH_DEG",
    "PREVIEW_TILT_DEG",
    "READY_LEFT_ARM",
    "READY_RIGHT_ARM",
    "READY_WAIST",
    "foot_positions",
    "load_source_cases",
    "original_reach_control_config",
    "summarize_cases",
    "synthetic_estimate",
    "world_target",
]
