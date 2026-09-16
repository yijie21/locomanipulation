"""Full-body bottle reach checks for predicted and measured standing support."""

import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from locomani.backends.sonic.primitives import STANDING_PELVIS_Z, Primitives
from locomani.primitives.supported_motion import SupportedMotion
from locomani.primitives.whole_body_ik import WholeBodyIK
from locomani.tasks.bottle.grasp_capture import capture_tool_goal
from locomani.tasks.bottle.ready_grasp_policy import OFFSET, ready_q

STANDING_SUPPORT_SOURCE = "neutral_standing_legs_candidate_upper_yaw_only_root"
REACH_PATH_SOURCE = "supported_cap_clearing_cartesian_reference"
CAP_CLEARING_CONTACT_HEIGHT_M = 0.035
SETTLED_REFERENCE_SPEED_MAX = 0.05


def _smooth(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3 - 2 * value)


def cap_clearing_target_contact_allowed(tool_position, target):
    """Match live finger-target contact admission near capture height."""
    tool_position = np.asarray(tool_position, dtype=float)
    target = np.asarray(target, dtype=float)
    return bool(abs(float(tool_position[2] - target[2])) < CAP_CLEARING_CONTACT_HEIGHT_M)


def cap_clearing_entry_step(
    command,
    measured,
    target,
    entry_axis,
    dt,
    hand_speed,
    final_depth_m,
    *,
    brake_acceleration=None,
    terminal_speed_fn=None,
):
    """Advance the cap-clearing Cartesian entry used live and in planning."""
    command = np.asarray(command, dtype=float)
    measured = np.asarray(measured, dtype=float)
    target = np.asarray(target, dtype=float)
    axis = np.asarray(entry_axis, dtype=float)
    axis /= np.linalg.norm(axis)
    above = max(measured[2], command[2]) - target[2]
    gap = float(final_depth_m) + 0.10 * _smooth((above - 0.018) / 0.04)
    goal = target - gap * axis
    delta = goal - command
    behind = min(float((target - hand) @ axis) for hand in (measured, command))
    if delta[2] < 0 and above > 0.035:
        delta[2] *= _smooth((behind - 0.05) / 0.05)
    speed = min(float(hand_speed), 0.02 + 0.5 * float(np.linalg.norm(goal - measured)))
    step_dt = min(float(dt), 0.1)
    step = delta * min(1.0, speed * step_dt / max(float(np.linalg.norm(delta)), 1e-9))
    if brake_acceleration is not None:
        remaining = max(0.0, float((goal - measured) @ axis))
        axial_speed = (
            min(speed, float(np.sqrt(2.0 * float(brake_acceleration) * remaining)))
            if terminal_speed_fn is None
            else float(terminal_speed_fn(remaining, speed, brake_acceleration))
        )
        axial_step = float(step @ axis)
        if axial_step > 0:
            step -= axis * max(0.0, axial_step - axial_speed * step_dt)
    return command + step, gap


def yaw_rotation(quat):
    rotation = Primitives._rot(quat)
    return Rotation.from_euler("z", np.arctan2(rotation[1, 0], rotation[0, 0])).as_matrix()


def standing_support_state(state, candidate_joints, candidate_base):
    """Build a double-support prior without copying an instantaneous gait leg pose."""
    candidate_joints = np.asarray(candidate_joints, dtype=float)
    candidate_base = np.asarray(candidate_base, dtype=float)
    if candidate_joints.shape != (29,) or candidate_base.shape != (7,):
        raise ValueError("standing support candidate must contain 29 joints and a 7D base")
    standing = deepcopy(state)
    joints = ready_q()
    joints[12:] = candidate_joints[12:]
    root = yaw_rotation(candidate_base[3:])
    quat = Rotation.from_matrix(root).as_quat()[[3, 0, 1, 2]]
    standing["joints"] = joints
    standing["pelvis"] = dict(standing["pelvis"])
    standing["pelvis"].update(
        pos=[candidate_base[0], candidate_base[1], STANDING_PELVIS_Z],
        quat=quat,
        vel=[0.0, 0.0, 0.0],
    )
    return standing


def shifted_stance_state(state, forward_m, yaw_rad):
    shifted = deepcopy(state)
    root = yaw_rotation(state["pelvis"]["quat"])
    root = Rotation.from_euler("z", yaw_rad).as_matrix() @ root
    position = np.asarray(state["pelvis"]["pos"], dtype=float).copy()
    position += float(forward_m) * root[:, 0]
    shifted["pelvis"] = dict(shifted["pelvis"])
    shifted["pelvis"].update(
        pos=position,
        quat=Rotation.from_matrix(root).as_quat()[[3, 0, 1, 2]],
        vel=[0.0, 0.0, 0.0],
    )
    return shifted, root


class StanceReachEvaluator:
    """Solve final capture geometry against one private IK and collision guard."""

    def __init__(self, ik, config, feasibility=None):
        self.ik = ik
        self.config = config
        self.feasibility = feasibility

    def _joint_limits(self):
        ranges = self.ik.model.jnt_range[self.ik.jids].T.copy()
        ranges[0, 12:15] = [
            -self.config.stance_waist_yaw_max,
            -self.config.stance_waist_roll_max,
            0.0,
        ]
        ranges[1, 12:15] = [
            self.config.stance_waist_yaw_max,
            self.config.stance_waist_roll_max,
            self.config.waist_pitch_max,
        ]
        return ranges

    def _posture_weights(self):
        weights = np.full(29, 0.04)
        weights[12:15] = self.config.comfort_waist_weight
        weights[25:29] = self.config.comfort_elbow_wrist_weight
        return weights

    def _cartesian_reach_path(
        self,
        state,
        target,
        desired_rotation,
        posture,
        torso_rotation,
        *,
        reference_joints=None,
        reference_root=None,
        cancel_event=None,
    ):
        """Sample the supported reference path that the live entry executes.

        Long joint interpolation from the incoming pose to the final IK pose is
        not a controller trajectory.  This nominal path starts from the known
        owned reference when available, keeps measured support fixed, advances
        the live cap-clearing tool law, and samples its rate-bounded references.
        Live execution retains the measured per-tick collision guard because
        plant tracking error can still make its exact joint path differ.
        """
        path_started = time.perf_counter()
        reference_joints = (
            np.asarray(state["joints"], dtype=float)
            if reference_joints is None
            else np.asarray(reference_joints, dtype=float)
        )
        reference_root = (
            np.asarray(state["pelvis"]["quat"], dtype=float)
            if reference_root is None
            else np.asarray(reference_root, dtype=float)
        )
        motion = SupportedMotion(
            self.ik,
            state,
            self.config,
            reference_joints,
            reference_root,
            tool_offset=OFFSET,
            hand_weights=(0.35, 1.0),
            posture_reference=posture,
            posture_weights=self._posture_weights(),
            continuity_weight=self.config.ik_velocity_continuity_weight,
        )
        positions = motion.body.tools(motion.x).copy()
        motion.body.put(motion.x)
        rotations = self.ik.data.xmat[motion.body.hands].reshape(2, 3, 3).copy()
        initial_right_rotation = rotations[1].copy()
        turn = Slerp([0, 1], Rotation.from_matrix([initial_right_rotation, desired_rotation]))
        torso_reference = self.ik.data.xmat[motion.body.torso].reshape(3, 3).copy()
        command = positions[1].copy()
        goal = capture_tool_goal(target, desired_rotation[:, 0], self.config.capture_depth_m)
        progress = 0.0
        dt = 0.10
        waypoints = []
        setup_finished = time.perf_counter()
        ik_step_seconds = 0.0
        terminal_fk_check_seconds = 0.0
        terminal_convergence_started = None
        previous_fk_tool = positions[1].copy()
        previous_fk_rotation = initial_right_rotation.copy()
        for _ in range(160):
            if cancel_event is not None and cancel_event.is_set():
                return waypoints, {
                    "source": REACH_PATH_SOURCE,
                    "model_scope": "nominal_owned_reference_with_measured_support",
                    "live_per_tick_guard_required": True,
                    "complete": False,
                    "cancelled": True,
                    "samples": len(waypoints),
                    "setup_seconds": setup_finished - path_started,
                    "ik_step_seconds": ik_step_seconds,
                    "terminal_fk_check_seconds": terminal_fk_check_seconds,
                    "terminal_convergence_seconds": (
                        None
                        if terminal_convergence_started is None
                        else time.perf_counter() - terminal_convergence_started
                    ),
                }
            previous_x = motion.x.copy()
            previous_root = motion.root.copy()
            measured_positions = motion.body.tools(motion.x).copy()
            motion.body.put(motion.x)
            measured_rotations = self.ik.data.xmat[motion.body.hands].reshape(2, 3, 3).copy()
            command, gap = cap_clearing_entry_step(
                command,
                measured_positions[1],
                target,
                desired_rotation[:, 0],
                dt,
                self.config.hand_speed,
                self.config.capture_depth_m,
                brake_acceleration=self.config.capture_brake_acceleration,
            )
            positions[1] = command
            progress = min(1.0, progress + dt)
            rotations[1] = turn(_smooth(progress)).as_matrix()
            motion.blend_posture(progress)
            torso_delta = Rotation.from_matrix(torso_rotation @ torso_reference.T).as_rotvec()
            torso_delta *= min(1.0, 0.25 * dt / max(float(np.linalg.norm(torso_delta)), 1e-9))
            torso_reference = Rotation.from_rotvec(torso_delta).as_matrix() @ torso_reference
            step_started = time.perf_counter()
            pose = motion.step(
                positions,
                rotations,
                measured_positions,
                dt,
                torso_rotation=torso_reference,
                measured_rotations=measured_rotations,
            )
            ik_step_seconds += time.perf_counter() - step_started
            delta_rotation = Primitives._rot(motion.root) @ Primitives._rot(previous_root).T
            previous_position = np.asarray(state["pelvis"]["pos"], dtype=float)
            if waypoints:
                previous_position = np.asarray(waypoints[-1]["candidate_base"][:3], dtype=float)
            candidate_position = previous_position + pose.position - previous_x[:3]
            previous_quat = (
                np.asarray(state["pelvis"]["quat"], dtype=float)
                if not waypoints
                else np.asarray(waypoints[-1]["candidate_base"][3:], dtype=float)
            )
            candidate_quat = Rotation.from_matrix(delta_rotation @ Primitives._rot(previous_quat)).as_quat()[
                [3, 0, 1, 2]
            ]
            terminal_check_started = time.perf_counter()
            motion.body.put(motion.x)
            fk_tools = motion.body.tools(motion.x)
            fk_rotations = self.ik.data.xmat[motion.body.hands].reshape(2, 3, 3)
            fk_torso = self.ik.data.xmat[motion.body.torso].reshape(3, 3)
            waypoints.append(
                {
                    "joints": pose.joints.copy(),
                    "candidate_base": np.r_[candidate_position, candidate_quat],
                    "commanded_tool": command.copy(),
                    "fk_tool": fk_tools[1].copy(),
                    "gap_m": float(gap),
                    "allow_target_contact": cap_clearing_target_contact_allowed(fk_tools[1], target),
                }
            )
            command_position_error = float(np.linalg.norm(command - goal))
            fk_position_error = float(np.linalg.norm(fk_tools[1] - goal))
            fk_rotation_error = float(Rotation.from_matrix(fk_rotations[1] @ desired_rotation.T).magnitude())
            fk_torso_error = float(Rotation.from_matrix(torso_rotation @ fk_torso.T).magnitude())
            fk_tool_speed = float(np.linalg.norm(fk_tools[1] - previous_fk_tool) / dt)
            fk_angular_speed = float(Rotation.from_matrix(fk_rotations[1] @ previous_fk_rotation.T).magnitude() / dt)
            foot_error = float(np.max(pose.foot_errors))
            com_shift = float(np.linalg.norm(pose.com_shift))
            terminal_reference_speed = float(np.max(np.abs(motion.v)))
            terminal_fk_check_seconds += time.perf_counter() - terminal_check_started
            previous_fk_tool = fk_tools[1].copy()
            previous_fk_rotation = fk_rotations[1].copy()
            if command_position_error <= 0.002 and terminal_convergence_started is None:
                terminal_convergence_started = time.perf_counter()
            if (
                command_position_error <= 0.002
                and fk_position_error <= 0.005
                and fk_rotation_error <= 0.15
                and fk_tool_speed <= self.config.capture_drift_speed_max
                and fk_angular_speed <= self.config.capture_angular_drift_max
                and foot_error <= 0.005
                and com_shift <= 0.015
                and terminal_reference_speed <= SETTLED_REFERENCE_SPEED_MAX
            ):
                return waypoints, {
                    "source": REACH_PATH_SOURCE,
                    "model_scope": "nominal_owned_reference_with_measured_support",
                    "live_per_tick_guard_required": True,
                    "complete": True,
                    "samples": len(waypoints),
                    "command_position_error_m": command_position_error,
                    "fk_position_error_m": fk_position_error,
                    "fk_rotation_error_rad": fk_rotation_error,
                    "fk_torso_error_rad": fk_torso_error,
                    "fk_tool_speed_m_s": fk_tool_speed,
                    "fk_angular_speed_rad_s": fk_angular_speed,
                    "foot_error_m": foot_error,
                    "com_shift_m": com_shift,
                    "terminal_reference_speed": terminal_reference_speed,
                    "terminal_reference_speed_limit": SETTLED_REFERENCE_SPEED_MAX,
                    "terminal": {
                        "joints": pose.joints.copy(),
                        "root_position": candidate_position.copy(),
                        "root_rotation": Primitives._rot(candidate_quat),
                        "torso_rotation": fk_torso.copy(),
                        "hand_errors": pose.hand_errors.copy(),
                        "foot_errors": pose.foot_errors.copy(),
                        "com_shift": pose.com_shift.copy(),
                    },
                    "setup_seconds": setup_finished - path_started,
                    "ik_step_seconds": ik_step_seconds,
                    "terminal_fk_check_seconds": terminal_fk_check_seconds,
                    "terminal_convergence_seconds": (
                        None
                        if terminal_convergence_started is None
                        else time.perf_counter() - terminal_convergence_started
                    ),
                }
        return waypoints, {
            "source": REACH_PATH_SOURCE,
            "model_scope": "nominal_owned_reference_with_measured_support",
            "live_per_tick_guard_required": True,
            "complete": False,
            "samples": len(waypoints),
            "command_position_error_m": float(np.linalg.norm(command - goal)),
            "fk_position_error_m": float(np.linalg.norm(fk_tools[1] - goal)),
            "fk_rotation_error_rad": float(Rotation.from_matrix(fk_rotations[1] @ desired_rotation.T).magnitude()),
            "fk_torso_error_rad": float(Rotation.from_matrix(torso_rotation @ fk_torso.T).magnitude()),
            "fk_tool_speed_m_s": fk_tool_speed,
            "fk_angular_speed_rad_s": fk_angular_speed,
            "foot_error_m": foot_error,
            "com_shift_m": com_shift,
            "terminal_reference_speed": float(np.max(np.abs(motion.v))),
            "terminal_reference_speed_limit": SETTLED_REFERENCE_SPEED_MAX,
            "setup_seconds": setup_finished - path_started,
            "ik_step_seconds": ik_step_seconds,
            "terminal_fk_check_seconds": terminal_fk_check_seconds,
            "terminal_convergence_seconds": (
                None if terminal_convergence_started is None else time.perf_counter() - terminal_convergence_started
            ),
        }

    def evaluate(
        self,
        state,
        target,
        desired_rotation,
        posture,
        *,
        allow_target_contact,
        verify_path=True,
        reference_joints=None,
        reference_root=None,
        cancel_event=None,
    ):
        started = time.perf_counter()
        state = deepcopy(state)
        target = np.asarray(target, dtype=float)
        desired_rotation = np.asarray(desired_rotation, dtype=float)
        posture = np.asarray(posture, dtype=float)
        if target.shape != (3,) or desired_rotation.shape != (3, 3) or posture.shape != (29,):
            raise ValueError("stance reach inputs have invalid geometry")

        body = WholeBodyIK(
            self.ik,
            state,
            self.config,
            tool_offset=OFFSET,
            joint_limits=self._joint_limits(),
            hand_weights=(0.35, 1.0),
            com_weight=20.0,
        )
        initial_x = body.initial.copy()
        positions = body.tools(initial_x)
        body.put(initial_x)
        rotations = self.ik.data.xmat[body.hands].reshape(2, 3, 3).copy()
        initial_torso = self.ik.data.xmat[body.torso].reshape(3, 3).copy()
        positions[1] = capture_tool_goal(
            target,
            desired_rotation[:, 0],
            self.config.capture_depth_m,
        )
        rotations[1] = desired_rotation
        try:
            pose = body.solve(
                positions,
                rotations,
                posture=posture,
                posture_weights=self._posture_weights(),
                iterations=self.config.stance_solver_iterations,
            )
        except (ValueError, RuntimeError) as exc:
            return {
                "feasible": False,
                "reason": f"whole_body_solver:{exc}",
                "final_ik_seconds": time.perf_counter() - started,
                "solve_seconds": time.perf_counter() - started,
            }
        final_solved = time.perf_counter()

        solved_x = body.seed.copy()
        body.put(solved_x)
        final_ik_torso_rotation = self.ik.data.xmat[body.torso].reshape(3, 3).copy()
        if verify_path:
            reach_path, reach_path_diagnostics = self._cartesian_reach_path(
                state,
                target,
                desired_rotation,
                pose.joints,
                final_ik_torso_rotation,
                reference_joints=reference_joints,
                reference_root=reference_root,
                cancel_event=cancel_event,
            )
        else:
            reach_path = None
            reach_path_diagnostics = {
                "source": REACH_PATH_SOURCE,
                "model_scope": "nominal_owned_reference_with_measured_support",
                "live_per_tick_guard_required": True,
                "complete": False,
                "verification": "deferred_endpoint_screen_only",
                "samples": 0,
            }
        terminal = reach_path_diagnostics.pop("terminal", None)
        if terminal is None:
            endpoint_source = "final_ik"
            endpoint_joints = pose.joints.copy()
            endpoint_position = pose.position.copy()
            endpoint_rotation = pose.rotation.copy()
            endpoint_torso_rotation = final_ik_torso_rotation
            endpoint_hand_errors = pose.hand_errors.copy()
            endpoint_foot_errors = pose.foot_errors.copy()
            endpoint_com_shift = pose.com_shift.copy()
        else:
            endpoint_source = "terminal_nominal_rollout"
            endpoint_joints = terminal["joints"]
            endpoint_position = terminal["root_position"]
            endpoint_rotation = terminal["root_rotation"]
            endpoint_torso_rotation = terminal["torso_rotation"]
            endpoint_hand_errors = terminal["hand_errors"]
            endpoint_foot_errors = terminal["foot_errors"]
            endpoint_com_shift = terminal["com_shift"]
            reach_path_diagnostics.update(
                terminal_joint_delta_from_final_ik_linf_rad=float(np.max(np.abs(endpoint_joints - pose.joints))),
                terminal_root_delta_from_final_ik_m=float(np.linalg.norm(endpoint_position - pose.position)),
                terminal_torso_delta_from_final_ik_rad=float(
                    Rotation.from_matrix(endpoint_torso_rotation @ final_ik_torso_rotation.T).magnitude()
                ),
            )
        torso_rotation = endpoint_torso_rotation
        torso_pitch = float(Rotation.from_matrix(body.heading.T @ torso_rotation).as_euler("xyz")[1])
        torso_action = float(Rotation.from_matrix(torso_rotation @ initial_torso.T).magnitude())
        physical = self.ik.model.jnt_range[self.ik.jids]
        span = physical[:, 1] - physical[:, 0]
        normalized_margin = (
            np.minimum(
                endpoint_joints - physical[:, 0],
                physical[:, 1] - endpoint_joints,
            )
            / span
        )
        natural_indices = np.r_[12:15, 22:29]
        natural_margin = float(np.min(normalized_margin[natural_indices]))
        posture_cost = float(
            np.linalg.norm((endpoint_joints[natural_indices] - posture[natural_indices]) / span[natural_indices])
        )
        joint_action = endpoint_joints - np.asarray(state["joints"], dtype=float)
        candidate_base = [
            *endpoint_position,
            *Rotation.from_matrix(endpoint_rotation).as_quat()[[3, 0, 1, 2]],
        ]
        rollout_finished = time.perf_counter()
        if reach_path_diagnostics.get("cancelled", False):
            clearance = {
                "clear": False,
                "reason": "cartesian_reach_path_cancelled",
                "checks": [],
            }
        else:
            clearance = (
                {"clear": True, "reason": None, "checks": []}
                if self.feasibility is None
                else self.feasibility(
                    state,
                    endpoint_joints,
                    candidate_base,
                    allow_target_contact=allow_target_contact,
                    path=reach_path,
                    endpoint_only=not verify_path,
                )
            )
        clearance_finished = time.perf_counter()
        if (
            verify_path
            and not reach_path_diagnostics["complete"]
            and not reach_path_diagnostics.get("cancelled", False)
        ):
            clearance = dict(clearance)
            clearance.update(clear=False, reason="cartesian_reach_path_incomplete")
        hand_error = float(np.max(endpoint_hand_errors))
        foot_error = float(np.max(endpoint_foot_errors))
        com_shift = float(np.linalg.norm(endpoint_com_shift))
        feasible = bool(
            hand_error <= self.config.stance_ik_tolerance
            and foot_error <= 0.005
            and com_shift <= 0.015
            and clearance.get("clear", False)
        )
        reason = None
        if hand_error > self.config.stance_ik_tolerance:
            reason = "final_hand_error"
        elif foot_error > 0.005:
            reason = "support_foot_error"
        elif com_shift > 0.015:
            reason = "support_com_shift"
        elif not clearance.get("clear", False):
            reason = clearance.get("reason", "final_swept_clearance")
        return {
            "feasible": feasible,
            "reason": reason,
            "endpoint_source": endpoint_source,
            "joints": endpoint_joints,
            "root_position": endpoint_position,
            "root_rotation": endpoint_rotation,
            "control_posture_joints": pose.joints,
            "control_torso_rotation": final_ik_torso_rotation,
            "grasp_rotation": desired_rotation.copy(),
            "entry_axis": desired_rotation[:, 0].copy(),
            "torso_rotation": torso_rotation,
            "torso_pitch_rad": torso_pitch,
            "torso_action_rad": torso_action,
            "hand_errors_m": endpoint_hand_errors,
            "foot_errors_m": endpoint_foot_errors,
            "com_shift_m": endpoint_com_shift,
            "natural_joint_margin": natural_margin,
            "natural_posture_cost": posture_cost,
            "joint_action_linf_rad": float(np.max(np.abs(joint_action))),
            "joint_action_l2_rad": float(np.linalg.norm(joint_action)),
            "left_arm_action_l2_rad": float(np.linalg.norm(joint_action[15:22])),
            "right_arm_action_l2_rad": float(np.linalg.norm(joint_action[22:29])),
            "clearance": clearance,
            "reach_path": reach_path_diagnostics,
            "feasibility_scope": "cartesian_path" if verify_path else "endpoint_only",
            "final_ik_seconds": final_solved - started,
            "cartesian_rollout_seconds": rollout_finished - final_solved,
            "clearance_seconds": clearance_finished - rollout_finished,
            "solve_seconds": time.perf_counter() - started,
            "selection_score": [
                -natural_margin,
                posture_cost,
                torso_action,
                float(np.linalg.norm(joint_action)),
                hand_error,
                foot_error,
                com_shift,
            ],
        }

    def evaluate_walking(self, payload):
        """Screen all endpoints cheaply, then verify one ranked Cartesian path."""
        started = time.perf_counter()
        checked = []
        endpoint_candidates = []
        verify_path = bool(payload.get("verify_path", True))
        yaw_margin = np.radians(self.config.stance_yaw_margin_deg)
        cancel_event = payload.get("_cancel_event")

        def cancelled():
            return cancel_event is not None and cancel_event.is_set()

        def cancelled_result():
            return {
                "feasible": False,
                "reason": "walking_plan_superseded",
                "chosen_index": None,
                "checked": checked,
                "cancelled": True,
                "support_prior_source": STANDING_SUPPORT_SOURCE,
                "solve_seconds": time.perf_counter() - started,
            }

        for index, source_candidate in enumerate(payload["candidates"]):
            if cancelled():
                return cancelled_result()
            candidate = deepcopy(source_candidate)
            pregrasp = (
                {"clear": True, "reason": None, "checks": []}
                if self.feasibility is None
                else self.feasibility(
                    payload["state"],
                    candidate["q"],
                    candidate["candidate_base"],
                    allow_target_contact=False,
                    endpoint_only=True,
                )
            )
            scenarios = {}
            expected_state = expected_rotation = None
            if pregrasp.get("clear", False):
                expected_state = standing_support_state(payload["state"], candidate["q"], candidate["candidate_base"])
                expected_state, expected_rotation = shifted_stance_state(expected_state, 0.0, 0.0)
                scenarios["expected"] = self.evaluate(
                    expected_state,
                    payload["target"],
                    expected_rotation,
                    candidate["q"],
                    allow_target_contact=True,
                    verify_path=False,
                )
            endpoint_feasible = bool(pregrasp.get("clear", False) and scenarios and scenarios["expected"]["feasible"])
            expected_score = list(scenarios["expected"]["selection_score"]) if scenarios else [float("inf")] * 7
            item = {
                "candidate_index": index,
                "candidate_endpoint": np.asarray(candidate["candidate_base"], dtype=float),
                "arm_score": float(candidate["score"]),
                "candidate_metadata": {
                    key: value for key, value in candidate.items() if key not in ("q", "candidate_base", "score")
                },
                "feasible": False,
                "endpoint_feasible": endpoint_feasible,
                "reason": (
                    pregrasp.get("reason")
                    if not pregrasp.get("clear", False)
                    else (
                        scenarios["expected"].get("reason")
                        if scenarios and not scenarios["expected"]["feasible"]
                        else "cartesian_path_verification_pending"
                    )
                ),
                "pregrasp_clearance": pregrasp,
                "scenarios": scenarios,
                "selection_score": [*expected_score, float(candidate["score"])],
            }
            checked.append(item)
            if endpoint_feasible:
                endpoint_candidates.append((item, candidate, expected_state, expected_rotation))

        endpoint_candidates.sort(key=lambda entry: entry[0]["selection_score"])
        chosen = None
        for item, candidate, expected_state, expected_rotation in endpoint_candidates:
            if cancelled():
                return cancelled_result()
            scenarios = {"expected": item["scenarios"]["expected"]}
            uncertainty_clear = True
            for name, yaw_rad in (
                ("forward_yaw_left", yaw_margin),
                ("forward_yaw_right", -yaw_margin),
            ):
                if cancelled():
                    return cancelled_result()
                scenario, rotation = shifted_stance_state(
                    standing_support_state(payload["state"], candidate["q"], candidate["candidate_base"]),
                    self.config.stance_forward_margin_m,
                    yaw_rad,
                )
                scenarios[name] = self.evaluate(
                    scenario,
                    payload["target"],
                    rotation,
                    candidate["q"],
                    allow_target_contact=True,
                    verify_path=False,
                )
                uncertainty_clear = uncertainty_clear and scenarios[name]["feasible"]
            if not uncertainty_clear:
                item.update(
                    scenarios=scenarios,
                    reason=next(
                        result.get("reason") for result in scenarios.values() if not result.get("feasible", False)
                    ),
                )
                continue
            if verify_path:
                scenarios["expected"] = self.evaluate(
                    expected_state,
                    payload["target"],
                    expected_rotation,
                    candidate["q"],
                    allow_target_contact=True,
                    verify_path=True,
                    cancel_event=cancel_event,
                )
            verified = bool(scenarios["expected"]["feasible"])
            scenario_results = list(scenarios.values())
            item.update(
                feasible=verified,
                reason=None if verified else scenarios["expected"].get("reason"),
                scenarios=scenarios,
                selection_score=[
                    *[max(result["selection_score"][position] for result in scenario_results) for position in range(7)],
                    float(candidate["score"]),
                ],
            )
            if verified:
                chosen = item
                break

        return {
            "feasible": chosen is not None,
            "reason": None if chosen is not None else "no_full_body_stance_candidate",
            "chosen_index": None if chosen is None else chosen["candidate_index"],
            "checked": checked,
            "support_prior_source": STANDING_SUPPORT_SOURCE,
            "path_verification": (
                "chosen_expected_cartesian_path; uncertainty_endpoints_only"
                if verify_path
                else "endpoint_only; live_per_tick_cartesian_guard"
            ),
            "solve_seconds": time.perf_counter() - started,
        }

    def evaluate_settled(self, payload):
        started = time.perf_counter()
        result = self.evaluate(
            payload["state"],
            payload["target"],
            payload["desired_rotation"],
            payload["posture"],
            allow_target_contact=True,
            verify_path=bool(payload.get("verify_path", True)),
            reference_joints=payload.get("reference_joints"),
            reference_root=payload.get("reference_root"),
            cancel_event=payload.get("_cancel_event"),
        )
        result["support_prior_source"] = "measured_post_stop_state"
        result["batch_solve_seconds"] = time.perf_counter() - started
        return result


class AsyncStanceReachPlanner:
    """Serialize private stance solves while control ticks submit and poll."""

    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bottle-stance")
        self.lock = threading.Lock()
        self.active = None
        self.pending_priority = None
        self.results = {}
        self.closed = False

    def _submit_locked(self, kind, key, payload):
        method = self.evaluator.evaluate_walking if kind == "walking" else self.evaluator.evaluate_settled
        worker_payload = deepcopy(payload)
        cancel_event = threading.Event()
        worker_payload["_cancel_event"] = cancel_event
        self.active = {
            "kind": kind,
            "key": key,
            "submitted_wall": time.monotonic(),
            "cancel_event": cancel_event,
            "future": self.executor.submit(method, worker_payload),
        }

    def _refresh_locked(self):
        if self.active is None or not self.active["future"].done():
            return
        active = self.active
        self.active = None
        try:
            result = active["future"].result()
        except CancelledError:
            result = {"feasible": False, "reason": "planner_worker_cancelled"}
        except Exception as exc:  # worker failures become conservative planner results
            result = {"feasible": False, "reason": f"planner_worker:{type(exc).__name__}:{exc}"}
        result = dict(result)
        result.update(
            kind=active["kind"],
            key=active["key"],
            submitted_wall=active["submitted_wall"],
            completed_wall=time.monotonic(),
        )
        if self.pending_priority is None:
            self.results[(active["kind"], active["key"])] = result
        else:
            kind, key, payload = self.pending_priority
            self.pending_priority = None
            self._submit_locked(kind, key, payload)

    def request(self, kind, key, payload, *, priority=False):
        started = time.perf_counter()
        with self.lock:
            if self.closed:
                return {"status": "closed", "poll_seconds": time.perf_counter() - started}
            self._refresh_locked()
            cached = self.results.pop((kind, key), None)
            if cached is not None:
                return {
                    "status": "ready",
                    "result": cached,
                    "poll_seconds": time.perf_counter() - started,
                }
            if self.active is None:
                self._submit_locked(kind, key, payload)
            elif priority and (self.active["kind"], self.active["key"]) != (kind, key):
                # A settled solve supersedes every walking result.  A running
                # Future cannot be killed, so queue exactly one private solve
                # and discard the old result when it finishes.
                self.results.clear()
                self.pending_priority = (kind, key, deepcopy(payload))
                self.active["cancel_event"].set()
                self.active["future"].cancel()
                self._refresh_locked()
            active = self.active
            return {
                "status": "pending",
                "active_kind": active["kind"],
                "active_key": active["key"],
                "plan_age_seconds": time.monotonic() - active["submitted_wall"],
                "poll_seconds": time.perf_counter() - started,
            }

    def poll(self, kind, key):
        started = time.perf_counter()
        with self.lock:
            if self.closed:
                return {"status": "closed", "poll_seconds": time.perf_counter() - started}
            self._refresh_locked()
            cached = self.results.pop((kind, key), None)
            if cached is not None:
                return {
                    "status": "ready",
                    "result": cached,
                    "poll_seconds": time.perf_counter() - started,
                }
            if self.active is not None and (self.active["kind"], self.active["key"]) == (kind, key):
                return {
                    "status": "pending",
                    "active_kind": kind,
                    "active_key": key,
                    "plan_age_seconds": time.monotonic() - self.active["submitted_wall"],
                    "poll_seconds": time.perf_counter() - started,
                }
            if self.pending_priority is not None and self.pending_priority[:2] == (kind, key):
                return {
                    "status": "queued",
                    "active_kind": None if self.active is None else self.active["kind"],
                    "active_key": None if self.active is None else self.active["key"],
                    "poll_seconds": time.perf_counter() - started,
                }
            return {"status": "missing", "poll_seconds": time.perf_counter() - started}

    def close(self):
        with self.lock:
            self.closed = True
            if self.active is not None:
                self.active["cancel_event"].set()
                self.active["future"].cancel()
            self.pending_priority = None
        self.executor.shutdown(wait=False, cancel_futures=True)
