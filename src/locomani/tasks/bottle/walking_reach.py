"""Slow camera-supervised arm preparation during gait.

Legacy profiles retain the fixed crate-style preparation.  The opt-in natural
profile starts from the incoming hanging reference, scores a small set of stop
and pregrasp pairs with real arm IK, and asks the task's swept collision guard
to admit the best few.  Camera corrections are deliberately slower than the
control loop and are applied before the existing acceleration-bounded joint
interpolator.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.backends.sonic.primitives import Primitives, upper_body_from_mj
from locomani.tasks.bottle.ready_grasp_policy import OFFSET, ready_q

HAND_X = 0.08
WORKSPACE_Y = (-0.24, -0.12)
WAIST_ENVELOPE = (np.array([-0.10, 0.0, 0.0]), np.array([0.10, 0.0, 0.12]))
ROT_WEIGHT = 1.0


class WalkingReach:
    def __init__(
        self,
        ik,
        config,
        upper_reference,
        *,
        brake_depth=0.31,
        brake_edge=0.0,
        feasibility=None,
    ):
        self.ik, self.config = ik, config
        self.brake_depth, self.brake_edge = float(brake_depth), float(brake_edge)
        self.feasibility = feasibility
        self.q = np.asarray(upper_reference, dtype=float).copy()
        self.v = np.zeros(17)
        self.wanted = self.q.copy()
        # Warm-start at the commanded reference. The other arm retains READY,
        # which is already the walking profile's table-clear posture.
        self.seed = ready_q()
        self.seed[12:15], self.seed[22:29] = self.q[:3], self.q[4::2]
        self.comfortable = self.seed.copy()
        # Shoulder slightly forward/outboard, flexed elbow, nearly neutral
        # wrist. This is a posture prior, not a forced tool orientation.
        natural = np.array([0.10, -0.30, 0.20, 1.02, 0.0, 0.12, 0.0])
        self.comfortable[22:29] = np.clip(natural, ik.lo[22:29], ik.hi[22:29])
        self.relative = self.root = self.rotation = self.raw_relative = None
        self.base_velocity = np.zeros(3)
        self.last = self.planned = None
        self.progress = 0.0
        self.raised = config.raise_trigger_px <= 0
        self.raise_reason = "configured_from_start" if self.raised else None
        self._ego_stamp = self._wrist_stamp = None
        self._ego_confirm = self._wrist_confirm = 0
        self.wrist_acquired = False
        self.raise_offset = 0.0
        self.roi_correction = np.zeros(3)  # pelvis-frame xyz
        self.roi_correction_time = None
        self.roi_updated = False
        self.roi_override_reason = None
        self.geometry = None
        self.geometry_anchor = None
        self.geometry_retry_time = None
        self.chosen_stop_depth = self.brake_depth
        self.chosen_stop_edge = self.brake_edge if self.brake_edge > 0 else None
        self.nominal_stop_depth = self.chosen_stop_depth
        self.nominal_stop_edge = self.chosen_stop_edge
        self.stop_lead = 0.0
        self.stop_speed = 0.0
        self.raw_gap_bound = None
        self.diagnostics = {}

    def _solve(self, goal, attitude, *, natural=False):
        ik = self.ik
        lo, hi = ik.lo, ik.hi
        ik.lo, ik.hi = lo.copy(), hi.copy()
        ik.lo[12:15] = np.maximum(lo[12:15], WAIST_ENVELOPE[0])
        ik.hi[12:15] = np.minimum(hi[12:15], WAIST_ENVELOPE[1])
        try:
            return ik.solve(
                "right",
                goal,
                None if natural else attitude,
                q_mj29=self.seed.copy(),
                point_offset=OFFSET,
                use_waist=True,
                iters=36 if natural else 30,
                rot_weight=0.0 if natural else ROT_WEIGHT,
                posture=self.comfortable if natural else self.seed,
                posture_weight=0.025 if natural else 0.005,
            )
        finally:
            ik.lo, ik.hi = lo, hi

    def _vision(self, camera, ego_px, dt):
        cfg = self.config
        self.roi_updated = False
        camera = {} if camera is None else camera
        visible = camera.get("visible", {})
        ego_stamp = camera.get("image_timestamps", {}).get("ego_view")
        ego_visible = visible.get("ego_view") == ego_stamp
        if not getattr(cfg, "wrist_roi_supervisor", False):
            if ego_px is not None and ego_px >= cfg.raise_trigger_px:
                self.raised, self.raise_reason = True, "ego_size"
        elif ego_stamp is not None and ego_stamp != self._ego_stamp:
            self._ego_stamp = ego_stamp
            if ego_visible and ego_px is not None and ego_px >= cfg.raise_trigger_px:
                self._ego_confirm += 1
            elif not ego_visible or ego_px is None or ego_px <= cfg.raise_release_px:
                self._ego_confirm = 0
            if self._ego_confirm >= cfg.vision_confirm_frames:
                self.raised, self.raise_reason = True, "ego_size_stable"

        if (
            getattr(cfg, "wrist_roi_supervisor", False)
            and not self.raised
            and self.raw_relative is not None
            and self.raw_relative[0] <= cfg.raise_late_depth
        ):
            self.raised, self.raise_reason = True, "late_depth_protection"

        roi = camera.get("wrist_roi")
        image_stamp = camera.get("image_timestamps", {}).get("right_wrist")
        new_image = image_stamp is not None and image_stamp != self._wrist_stamp
        if new_image:
            self._wrist_stamp = image_stamp
            if not self.wrist_acquired and (roi is None or roi.get("timestamp") != image_stamp):
                self._wrist_confirm = 0
        new_roi = new_image and roi is not None and roi.get("timestamp") == image_stamp
        if new_roi:
            valid = (
                float(roi.get("quality", 0.0)) > 0
                and np.asarray(roi.get("centroid", [])).shape == (2,)
                and np.asarray(roi.get("extent", [])).shape == (2,)
            )
            self._wrist_confirm = self._wrist_confirm + 1 if valid else 0
            if self._wrist_confirm >= cfg.vision_confirm_frames:
                self.wrist_acquired = True

            period = 1.0 / cfg.roi_correction_hz
            if self.wrist_acquired and (
                self.roi_correction_time is None or float(roi["timestamp"]) - self.roi_correction_time >= period
            ):
                error = np.asarray(roi["centroid"], dtype=float) - 0.5
                active = np.sign(error) * np.maximum(np.abs(error) - cfg.roi_deadband, 0.0)
                basis = np.asarray(roi.get("correction_basis_pelvis", []), dtype=float)
                if basis.shape != (3, 2) or not np.isfinite(basis).all():
                    basis = np.zeros((3, 2))
                increment = basis @ active * cfg.roi_gain
                self.roi_correction = np.clip(
                    self.roi_correction + increment, -cfg.roi_correction_max, cfg.roi_correction_max
                )
                norm = float(np.linalg.norm(self.roi_correction))
                if norm > cfg.roi_correction_max:
                    self.roi_correction *= cfg.roi_correction_max / norm
                self.roi_correction_time = float(roi["timestamp"])
                self.roi_updated = bool(np.linalg.norm(increment) > 0)

        if self.raised and not self.wrist_acquired and getattr(cfg, "wrist_roi_supervisor", False):
            self.raise_offset = min(cfg.find_raise_max, self.raise_offset + cfg.find_raise_rate * dt)
        return roi

    def _candidate_geometry(self, state, edge_distance):
        cfg = self.config
        anchor = np.array([self.relative[1], self.relative[2]], dtype=float)
        # An unavailable edge later in the same approach does not erase a
        # previously selected edge-aware stance. A newly available edge does
        # upgrade an earlier depth-only selection.
        edge_contract_matches = self.geometry is not None and (
            edge_distance is None or self.geometry.get("stop_edge") is not None
        )
        if edge_contract_matches and np.linalg.norm(anchor - self.geometry_anchor) < 0.04:
            if self.geometry.get("feasible", False):
                return self.geometry
            if self.geometry_retry_time is not None and float(state["t"]) - self.geometry_retry_time < 1.0:
                return self.geometry

        depths = self.brake_depth + np.linspace(-cfg.stance_search_span, cfg.stance_search_span, 5)
        specs = [(depth, cfg.pregrasp_gap_max, 0.0) for depth in depths]
        specs.extend(
            (self.brake_depth, cfg.pregrasp_gap_max, lateral)
            for lateral in (-cfg.pregrasp_lateral_span, cfg.pregrasp_lateral_span)
        )
        candidates = []
        heading_world = self.rotation[:, 0]
        admitted_target = np.asarray(state["pelvis"]["pos"], dtype=float) + self.root @ self.relative
        for requested_depth, gap, lateral in specs:
            stop_depth = float(requested_depth)
            # The edge clearance is a physical lower bound on how far the base
            # may travel. Fold it into the candidate depth instead of letting
            # a separate fixed edge gate silently override the selected stance.
            if self.brake_edge > 0 and edge_distance is not None:
                edge_limited_depth = float(self.raw_relative[0] - (edge_distance - self.brake_edge))
                stop_depth = max(stop_depth, edge_limited_depth)
            travel = max(0.0, float(self.raw_relative[0] - stop_depth))
            stop_edge = None if edge_distance is None else float(edge_distance - travel)
            base_pos = np.asarray(state["pelvis"]["pos"], dtype=float) + heading_world * travel
            candidate_base = np.r_[base_pos, np.asarray(state["pelvis"]["quat"], dtype=float)]
            command_gap = gap + cfg.walk_tracking_gap
            goal = np.array(
                [stop_depth - command_gap, self.relative[1] + lateral, self.relative[2] + cfg.walk_clearance]
            )
            solved, error, rot_error = self._solve(goal, None, natural=True)
            view = self._wrist_view(solved, candidate_base, admitted_target)
            posture_cost = float(np.linalg.norm((solved[22:29] - self.comfortable[22:29]) / 1.5))
            score = float(error) * 30 + posture_cost + abs(lateral) * 2 + abs(gap - 0.13) - view["quality"]
            candidates.append(
                dict(
                    stop_depth=float(stop_depth),
                    requested_stop_depth=float(requested_depth),
                    stop_edge=stop_edge,
                    gap=float(gap),
                    command_gap=float(command_gap),
                    lateral=float(lateral),
                    goal=goal,
                    q=solved,
                    ik_error=float(error),
                    rot_error=float(rot_error),
                    posture_cost=posture_cost,
                    score=score,
                    candidate_base=candidate_base,
                    orientation_policy="comfortable_posture_prior_no_cartesian_rotation",
                    wrist_view=view,
                )
            )
        candidates.sort(key=lambda item: item["score"])
        checked = []
        chosen = None
        for candidate in candidates[: cfg.stance_candidates]:
            clearance = (
                dict(clear=True, reason=None, min_distance=None, checks=[])
                if self.feasibility is None
                else self.feasibility(state, candidate["q"], candidate["candidate_base"])
            )
            candidate["clearance"] = clearance
            candidate["feasible"] = bool(
                candidate["ik_error"] <= cfg.stance_ik_tolerance
                and clearance.get("clear", False)
                and (not cfg.wrist_roi_supervisor or candidate["wrist_view"]["visible"])
            )
            checked.append(candidate)
            if candidate["feasible"]:
                chosen = candidate
                break
        fallback_reason = None
        if chosen is None:
            fallback_reason = "no_candidate_passed_ik_and_swept_clearance"
            if not candidates:
                # Edge geometry excluded every searched stop. Keep the physical
                # configured stop so the outer edge gate can halt safely.
                goal = np.array(
                    [self.brake_depth - cfg.pregrasp_gap_max, self.relative[1], self.relative[2] + cfg.walk_clearance]
                )
                solved, error, rot_error = self._solve(goal, None, natural=True)
                chosen = dict(
                    stop_depth=self.brake_depth,
                    stop_edge=self.brake_edge if self.brake_edge > 0 else None,
                    gap=cfg.pregrasp_gap_max,
                    lateral=0.0,
                    goal=goal,
                    q=solved,
                    ik_error=float(error),
                    rot_error=float(rot_error),
                    score=float(error) * 30,
                    candidate_base=None,
                    clearance=dict(clear=False, reason="edge_excluded_search", checks=[]),
                    feasible=False,
                )
            else:
                chosen = candidates[0]
                chosen = chosen.copy()
                chosen["stop_depth"] = self.brake_depth
                chosen["stop_edge"] = self.brake_edge if self.brake_edge > 0 else None
        chosen = chosen.copy()
        chosen["fallback_reason"] = fallback_reason
        chosen["selection_reason"] = (
            "lowest_score_passing_ik_view_and_swept_clearance"
            if fallback_reason is None
            else "conservative_configured_stop_with_arm_hold"
        )
        chosen["checked"] = [
            {key: value for key, value in item.items() if key not in ("q", "candidate_base")} for item in checked
        ]
        self.geometry, self.geometry_anchor = chosen, anchor
        self.geometry_retry_time = float(state["t"])
        self.nominal_stop_depth = float(chosen["stop_depth"])
        self.nominal_stop_edge = chosen["stop_edge"]
        self.chosen_stop_depth = self.nominal_stop_depth + self.stop_lead
        self.chosen_stop_edge = None if self.nominal_stop_edge is None else self.nominal_stop_edge + self.stop_lead
        self.seed = np.asarray(chosen["q"]).copy()
        return chosen

    def _update_stop_envelope(self, state):
        if self.rotation is None or not getattr(self.config, "natural_walk_prep", False):
            return
        heading = self.rotation[:, 0]
        measured = max(0.0, float(np.asarray(state["pelvis"].get("vel", [0, 0, 0])) @ heading))
        filtered = max(0.0, float(self.base_velocity @ heading))
        self.stop_speed = max(measured, filtered)
        cfg = self.config
        lead = self.stop_speed**2 / (2 * cfg.walk_stop_acceleration) + self.stop_speed * cfg.walk_stop_latency
        # Never shrink the stop target while approaching; doing so makes a
        # decelerating gait chase its own receding threshold.
        self.stop_lead = max(self.stop_lead, min(cfg.walk_stop_lead_max, lead))
        self.chosen_stop_depth = self.nominal_stop_depth + self.stop_lead
        self.chosen_stop_edge = None if self.nominal_stop_edge is None else self.nominal_stop_edge + self.stop_lead

    def _wrist_view(self, joints, candidate_base, target):
        """Project the already-admitted target through candidate wrist FK."""
        ik, m, d = self.ik, self.ik.model, self.ik.data
        d.qpos[ik.qadr] = joints
        d.qpos[ik.base_qadr : ik.base_qadr + 7] = candidate_base
        ik._mj.mj_fwdPosition(m, d)
        camera = m.camera("right_wrist").id
        local = (np.asarray(target) - d.cam_xpos[camera]) @ d.cam_xmat[camera].reshape(3, 3)
        depth = float(-local[2])
        tangent = float(np.tan(np.radians(m.cam_fovy[camera]) / 2))
        normalized = local[:2] / max(depth, 1e-9)
        # Width is at least height for the deployed wrist stream. A square
        # frustum is a conservative admission before actual pixels arrive.
        margin = min(tangent - abs(normalized[0]), tangent - abs(normalized[1]))
        return dict(
            visible=bool(depth > 0.03 and margin > 0),
            depth=depth,
            normalized=normalized,
            quality=float(np.clip(margin / max(tangent, 1e-9), 0.0, 1.0)),
        )

    @staticmethod
    def _public_geometry(geometry):
        if geometry is None:
            return None
        return {key: value for key, value in geometry.items() if key not in ("q", "candidate_base")}

    def step(self, state, target, *, ego_px=None, camera=None, edge_distance=None):
        stamp = float(state["t"])
        dt = 0 if self.last is None else stamp - self.last
        if dt < 0 or dt > 0.5:
            raise ValueError("walking reach physical state clock gap")
        if self.last == stamp:
            return self.q.copy(), self.v.copy()
        self.last = stamp
        updated = False

        if target is not None:
            root = Primitives._rot(state["pelvis"]["quat"])
            relative = root.T @ (np.asarray(target) - state["pelvis"]["pos"])
            self.raw_relative = relative.copy()
            if self.relative is None:
                self.relative, self.root = relative.copy(), root.copy()
            alpha = 1 - np.exp(-dt / self.config.walk_filter_seconds)
            velocity = np.asarray(state["pelvis"].get("vel", [0, 0, 0]), dtype=float).copy()
            velocity[2] = 0
            self.base_velocity += (1 - np.exp(-dt / (self.config.walk_filter_seconds / 3))) * (
                velocity - self.base_velocity
            )
            self.relative -= self.root.T @ self.base_velocity * dt
            self.relative += alpha * (relative - self.relative)
            self.root = (
                Rotation.from_rotvec(alpha * Rotation.from_matrix(root @ self.root.T).as_rotvec()).as_matrix()
                @ self.root
            )
        self._update_stop_envelope(state)

        roi = self._vision(camera, ego_px, dt)
        if not self.raised:
            self.wanted = self.q.copy()
        if (
            target is not None
            and self.raised
            and (self.planned is None or stamp - self.planned >= 1 / self.config.walk_plan_hz)
        ):
            self.planned, updated = stamp, True
            cfg = self.config
            self.rotation = Rotation.from_euler("z", np.arctan2(self.root[1, 0], self.root[0, 0])).as_matrix()
            heading = self.root.T @ self.rotation[:, 0]
            floor = self.relative[2] + cfg.walk_clearance
            self.progress = float(np.clip((self.brake_depth + 0.9 - self.relative[0]) / 0.9, 0, 1))
            command_feasible = True
            if getattr(cfg, "natural_walk_prep", False):
                geometry = self._candidate_geometry(state, edge_distance)
                self._update_stop_envelope(state)
                goal = np.asarray(geometry["goal"]).copy()
                if not geometry.get("feasible", False):
                    command_feasible = False
                    self.wanted = self.q.copy()
                    solved, error, rot_error = self.seed.copy(), geometry["ik_error"], geometry["rot_error"]
                    goal = np.asarray(geometry["goal"]).copy()
                else:
                    goal += self.roi_correction
                    goal[2] = max(goal[2] + self.raise_offset, floor)
                    heading_local = self.root.T @ self.rotation[:, 0]
                    reserve = max(float(geometry["gap"]), cfg.pregrasp_gap_max) + cfg.walk_tracking_gap
                    reserve += self.stop_lead
                    raw_forward = float(self.raw_relative @ heading_local)
                    self.raw_gap_bound = raw_forward - reserve
                    command_forward = float(goal @ heading_local)
                    if command_forward > self.raw_gap_bound:
                        goal -= heading_local * (command_forward - self.raw_gap_bound)
                    solved, error, rot_error = self._solve(goal, None, natural=True)
                    if self.roi_updated and self.feasibility is not None:
                        corrected_clearance = self.feasibility(state, solved, geometry["candidate_base"])
                        if not corrected_clearance.get("clear", False):
                            self.roi_override_reason = corrected_clearance.get("reason", "swept_clearance")
                            self.roi_correction[:] = 0
                            goal = np.asarray(geometry["goal"]).copy()
                            solved, error, rot_error = self._solve(goal, None, natural=True)
                        else:
                            self.roi_override_reason = None
            else:
                advance_speed = max(0, float(self.base_velocity @ self.rotation[:, 0]))
                gap = 0.12 + advance_speed / cfg.walk_plan_hz
                behind = self.raw_relative[0] - heading[0] * gap
                goal = np.array([min(HAND_X, behind), np.clip(self.relative[1], *WORKSPACE_Y), floor])
                solved, error, rot_error = self._solve(goal, self.root.T @ self.rotation)
            if command_feasible:
                self.seed = solved
                self.wanted = upper_body_from_mj(solved)
            self.diagnostics.update(goal=goal.copy(), ik_error=float(error), rot_error=float(rot_error))
        elif target is None:
            self.wanted = self.q.copy()

        speed = min(self.config.walk_joint_speed, self.config.joint_speed)
        acceleration = min(1.0, self.config.joint_acceleration)
        error = self.wanted - self.q
        braking_speed = np.sqrt(2 * acceleration * np.abs(error)) * 0.8
        wanted_v = np.sign(error) * np.minimum(np.minimum(4 * np.abs(error), braking_speed), speed)
        self.v += np.clip(wanted_v - self.v, -acceleration * dt, acceleration * dt)
        self.q += self.v * dt
        self.diagnostics.update(
            t=stamp,
            plan_updated=updated,
            relative=None if self.relative is None else self.relative.copy(),
            reference=self.q.copy(),
            velocity=self.v.copy(),
            target_available=target is not None,
            raised=self.raised,
            raise_reason=self.raise_reason,
            raise_offset=self.raise_offset,
            ego_px=ego_px,
            ego_confirm_frames=self._ego_confirm,
            wrist_acquired=self.wrist_acquired,
            wrist_confirm_frames=self._wrist_confirm,
            wrist_roi=roi,
            roi_correction=self.roi_correction.copy(),
            roi_hold=not updated or roi is None,
            roi_updated=self.roi_updated,
            roi_override_reason=self.roi_override_reason,
            chosen_geometry=self._public_geometry(self.geometry),
            chosen_stop_depth=self.chosen_stop_depth,
            chosen_stop_edge=self.chosen_stop_edge,
            nominal_stop_depth=self.nominal_stop_depth,
            nominal_stop_edge=self.nominal_stop_edge,
            stop_speed=self.stop_speed,
            stop_lead=self.stop_lead,
            raw_relative=None if self.raw_relative is None else self.raw_relative.copy(),
            raw_gap_bound=self.raw_gap_bound,
        )
        return self.q.copy(), self.v.copy()
