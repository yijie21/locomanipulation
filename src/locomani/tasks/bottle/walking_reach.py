"""Slow camera-supervised arm preparation during gait.

Legacy profiles retain the fixed crate-style preparation.  The opt-in natural
profile starts from the incoming hanging reference, scores a small set of stop
and pregrasp pairs with real arm IK, and asks the task's swept collision guard
to admit the best few.  Camera corrections are deliberately slower than the
control loop and are applied before the existing acceleration-bounded joint
interpolator.
"""

import time

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.backends.sonic.primitives import STANDING_PELVIS_Z, Primitives, upper_body_from_mj
from locomani.tasks.bottle.ready_grasp_policy import OFFSET, ready_q
from locomani.tasks.bottle.stance_reach import STANDING_SUPPORT_SOURCE, yaw_rotation

HAND_X = 0.08
WORKSPACE_Y = (-0.24, -0.12)
WAIST_ENVELOPE = (np.array([-0.10, 0.0, 0.0]), np.array([0.10, 0.0, 0.12]))
ROT_WEIGHT = 1.0
# The right arm's natural posture: shoulder slightly forward/outboard, flexed elbow, nearly neutral wrist (joints 22-28).
# The walking reach's posture prior, and the simple expert's CuRobo retract posture (simple_policy.ROUTE_JOINTS).
NATURAL_RIGHT_ARM = (0.10, -0.30, 0.20, 1.02, 0.0, 0.12, 0.0)


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
        stance_planner=None,
    ):
        self.ik, self.config = ik, config
        self.brake_depth, self.brake_edge = float(brake_depth), float(brake_edge)
        self.feasibility = feasibility
        self.stance_planner = stance_planner
        self.q = np.asarray(upper_reference, dtype=float).copy()
        self.v = np.zeros(17)
        self.wanted = self.q.copy()
        # Warm-start at the commanded reference. The other arm retains READY,
        # which is already the walking profile's table-clear posture.
        self.seed = ready_q()
        self.seed[12:15], self.seed[22:29] = self.q[:3], self.q[4::2]
        self.comfortable = self.seed.copy()
        # A posture prior, not a forced tool orientation.
        self.comfortable[22:29] = np.clip(np.array(NATURAL_RIGHT_ARM), ik.lo[22:29], ik.hi[22:29])
        self.relative = self.root = self.rotation = self.raw_relative = None
        self.world_target = None
        self.navigation_rotation = None
        self.navigation_target = None
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
        self._stance_request = None
        self._stance_valid_contract = None
        self._stance_request_count = 0
        self._stance_hit_count = 0
        self._stance_stale_count = 0
        self._stance_last_status = "disabled" if stance_planner is None else "idle"
        self._stance_last_result = None
        self._stance_poll_seconds = 0.0
        self._stance_retry_wall = None
        self._stance_ready_response = None
        self._stance_invalidations = []
        self._stance_consumptions = []
        self._stance_valid_request = None
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

    @staticmethod
    def _stance_candidate_identity(candidate):
        keys = ("requested_stop_depth", "gap", "lateral")
        if all(candidate.get(key) is not None for key in keys):
            return {
                "source": "search_parameters",
                **{key: float(candidate[key]) for key in keys},
            }
        base = np.asarray(candidate["candidate_base"], dtype=float)
        return {
            "source": "normalized_endpoint",
            "root_x": float(base[0]),
            "root_y": float(base[1]),
        }

    @staticmethod
    def _stance_candidate_identity_matches(left, right):
        if left["source"] != right["source"]:
            return False
        return (
            all(key == "source" or abs(float(left[key]) - float(right[key])) <= 1e-9 for key in left if key in right)
            and left.keys() == right.keys()
        )

    def _set_planning_frame(self, target, navigation_heading, navigation_target):
        """Keep live world observations separate from the fixed arrival heading."""
        self.world_target = np.asarray(target, dtype=float).copy()
        if navigation_heading is None:
            self.navigation_rotation = None
            self.navigation_target = None
            yaw = np.arctan2(self.root[1, 0], self.root[0, 0])
        else:
            heading = np.asarray(navigation_heading, dtype=float)
            if heading.shape not in ((2,), (3,)) or not np.isfinite(heading).all():
                raise ValueError("walking navigation heading must contain two or three finite values")
            norm = float(np.linalg.norm(heading[:2]))
            if norm <= 1e-9:
                raise ValueError("walking navigation heading must have a nonzero horizontal component")
            heading = heading[:2] / norm
            yaw = np.arctan2(heading[1], heading[0])
            self.navigation_rotation = Rotation.from_euler("z", yaw).as_matrix()
            if navigation_target is None:
                raise ValueError("walking navigation target is required with a fixed heading")
            navigation_target = np.asarray(navigation_target, dtype=float)
            if navigation_target.shape not in ((2,), (3,)) or not np.isfinite(navigation_target).all():
                raise ValueError("walking navigation target must contain two or three finite values")
            self.navigation_target = navigation_target[:2].copy()
        self.rotation = Rotation.from_euler("z", yaw).as_matrix()

    def _observed_target(self, state):
        if self.world_target is not None:
            return self.world_target.copy()
        return np.asarray(state["pelvis"]["pos"], dtype=float) + self.root @ self.relative

    def _target_depth(self, state, target):
        if state is None or self.rotation is None:
            return float(self.raw_relative[0])
        delta = np.asarray(target, dtype=float)[:2] - np.asarray(state["pelvis"]["pos"], dtype=float)[:2]
        return float(delta @ self.rotation[:2, 0])

    def _candidate_base(self, state, travel):
        base_pos = np.asarray(state["pelvis"]["pos"], dtype=float) + self.rotation[:, 0] * float(travel)
        if self.navigation_rotation is None:
            quat = np.asarray(state["pelvis"]["quat"], dtype=float)
        else:
            lateral = self.navigation_rotation[:2, 1]
            base_pos[:2] += lateral * float((self.navigation_target - base_pos[:2]) @ lateral)
            quat = Rotation.from_matrix(self.navigation_rotation).as_quat()[[3, 0, 1, 2]]
        return np.r_[base_pos, quat]

    def _stance_candidate_contract(self, candidate):
        base = np.asarray(candidate["candidate_base"], dtype=float)
        joints = candidate.get("q")
        return {
            "identity": self._stance_candidate_identity(candidate),
            "root_xy": base[:2].copy(),
            "source_root_z": float(base[2]),
            "normalized_root_z": float(STANDING_PELVIS_Z),
            "yaw_rotation": yaw_rotation(base[3:]),
            "stop_edge": None if candidate.get("stop_edge") is None else float(candidate["stop_edge"]),
            "upper_joints": None if joints is None else np.asarray(joints, dtype=float)[12:].copy(),
        }

    def _stance_contract(self, candidates, target, edge_distance, state=None):
        target_edge_offset = None
        if edge_distance is not None:
            target_edge_offset = float(self._target_depth(state, target) - edge_distance)
        return {
            "target": np.asarray(target, dtype=float).copy(),
            "candidates": [self._stance_candidate_contract(candidate) for candidate in candidates],
            "edge_available": edge_distance is not None,
            # Both terms are observed in the moving pelvis frame, so their
            # difference is the static world object-to-table-edge geometry.
            # Raw ego edge distance alone changes while the robot walks and
            # used to invalidate otherwise identical world-frame endpoints.
            "target_edge_offset": target_edge_offset,
            "support_prior_source": STANDING_SUPPORT_SOURCE,
            "source": (
                None
                if state is None
                else {
                    "seq": state.get("seq"),
                    "t": state.get("t"),
                    "wall": state.get("wall"),
                }
            ),
        }

    @staticmethod
    def _stance_contract_with_selected(contract, selected_index):
        return {**contract, "selected_candidate": contract["candidates"][int(selected_index)]}

    def _stance_contract_difference(self, left, right):
        if left is None or right is None:
            return {"matches": False, "reason": "missing_contract"}
        tolerance = self.config.stance_plan_tolerance_m
        yaw_tolerance = np.radians(self.config.stance_plan_yaw_tolerance_deg)
        joint_tolerance = self.config.stance_plan_joint_tolerance_rad
        edge_available_matches = left["edge_available"] == right["edge_available"]
        support_source_matches = left["support_prior_source"] == right["support_prior_source"]
        target_error = float(np.linalg.norm(left["target"] - right["target"]))
        target_edge_error = (
            0.0
            if not left["edge_available"] or not right["edge_available"]
            else abs(float(left["target_edge_offset"] - right["target_edge_offset"]))
        )
        selected = left.get("selected_candidate")
        if selected is None and len(left["candidates"]) == 1:
            selected = left["candidates"][0]
        candidates = (
            []
            if selected is None
            else [
                candidate
                for candidate in right["candidates"]
                if self._stance_candidate_identity_matches(selected["identity"], candidate["identity"])
            ]
        )
        current = None
        if candidates:
            current = min(candidates, key=lambda candidate: np.linalg.norm(selected["root_xy"] - candidate["root_xy"]))

        if current is None:
            root_xy_error = source_root_z_error = normalized_root_z_error = None
            yaw_error = stop_edge_error = joint_error = None
        else:
            root_xy_error = float(np.linalg.norm(selected["root_xy"] - current["root_xy"]))
            source_root_z_error = abs(float(selected["source_root_z"] - current["source_root_z"]))
            normalized_root_z_error = abs(float(selected["normalized_root_z"] - current["normalized_root_z"]))
            yaw_error = float(Rotation.from_matrix(selected["yaw_rotation"] @ current["yaw_rotation"].T).magnitude())
            if selected["stop_edge"] is None and current["stop_edge"] is None:
                stop_edge_error = 0.0
            elif selected["stop_edge"] is None or current["stop_edge"] is None:
                stop_edge_error = None
            else:
                stop_edge_error = abs(float(selected["stop_edge"] - current["stop_edge"]))
            if selected["upper_joints"] is None or current["upper_joints"] is None:
                joint_error = 0.0
            else:
                joint_error = float(np.max(np.abs(selected["upper_joints"] - current["upper_joints"])))

        checks = {
            "edge_availability": edge_available_matches,
            "support_prior_source": support_source_matches,
            "target": target_error <= tolerance,
            "target_edge": target_edge_error <= tolerance,
            "selected_candidate": current is not None,
            "normalized_root_xy": root_xy_error is not None and root_xy_error <= tolerance,
            "normalized_root_z": normalized_root_z_error is not None and normalized_root_z_error <= tolerance,
            "root_yaw": yaw_error is not None and yaw_error <= yaw_tolerance,
            "stop_edge": stop_edge_error is not None and stop_edge_error <= tolerance,
        }
        reason = next((name for name, passed in checks.items() if not passed), None)
        return {
            "matches": reason is None,
            "reason": reason,
            "checks": checks,
            "selected_candidate_identity": None if selected is None else selected["identity"],
            "current_candidate_identity": None if current is None else current["identity"],
            "target_error_m": target_error,
            "target_tolerance_m": tolerance,
            "target_edge_error_m": target_edge_error,
            "root_xy_error_m": root_xy_error,
            "source_root_z_error_m": source_root_z_error,
            "normalized_root_z_error_m": normalized_root_z_error,
            "yaw_error_rad": yaw_error,
            "yaw_tolerance_rad": yaw_tolerance,
            "stop_edge_error_m": stop_edge_error,
            "upper_joint_error_rad": joint_error,
            "upper_joint_tolerance_rad": joint_tolerance,
            "request_source": left.get("source"),
            "current_source": right.get("source"),
        }

    def _stance_contract_matches(self, left, right):
        return bool(self._stance_contract_difference(left, right)["matches"])

    def _record_stance_validation(self, status, request, result, difference):
        now = time.monotonic()
        event = {
            "status": status,
            "reason": difference["reason"],
            "request_key": None if request is None else request.get("key"),
            "selected_candidate_index": None if request is None else request.get("selected_candidate_index"),
            "request_age_seconds": None if request is None else now - float(request["submitted_wall"]),
            "result_age_seconds": (
                None
                if result is None or result.get("completed_wall") is None
                else now - float(result["completed_wall"])
            ),
            "result_solve_seconds": None if result is None else result.get("solve_seconds"),
            **difference,
        }
        if status == "consumed":
            self._stance_consumptions.append(event)
        else:
            self._stance_invalidations.append(event)
        return event

    def _publish_stance_geometry(self, chosen, anchor, contract=None, request=None):
        self.geometry = chosen
        self.geometry_anchor = anchor
        self.geometry_retry_time = self.last
        self.nominal_stop_depth = float(chosen["stop_depth"])
        self.nominal_stop_edge = chosen["stop_edge"]
        self.chosen_stop_depth = self.nominal_stop_depth + self.stop_lead
        self.chosen_stop_edge = None if self.nominal_stop_edge is None else self.nominal_stop_edge + self.stop_lead
        if chosen.get("feasible", False):
            self.seed = np.asarray(chosen["q"]).copy()
            self._stance_valid_contract = contract
            self._stance_valid_request = request
        return chosen

    @staticmethod
    def _conservative_local_candidate(candidates):
        """Keep the base outside while the standing plan is unresolved.

        A local arm candidate may begin collision-screened wrist preparation,
        but it cannot justify a closer stop. Prefer the largest object-relative
        stop depth, then the normal local score.
        """
        return min(candidates, key=lambda candidate: (-float(candidate["stop_depth"]), candidate["score"]))

    def _stance_retry_holds_geometry(self, state):
        """Poll cheaply during the local retry hold; recompute only for a ready result."""
        if self.geometry is None or self.geometry.get("feasible", False):
            return False
        if self.geometry_retry_time is None or float(state["t"]) - self.geometry_retry_time >= 1.0:
            return False
        if self._stance_request is None:
            return True
        response = self.stance_planner.poll("walking", self._stance_request["key"])
        self._stance_poll_seconds = float(response.get("poll_seconds", 0.0))
        self._stance_last_status = response["status"]
        if response["status"] == "ready":
            self._stance_ready_response = response
            return False
        if response["status"] in ("missing", "closed"):
            self._stance_request = None
            self._stance_retry_wall = time.monotonic()
        return True

    def _cached_stance_contract(self, state, edge_distance):
        """Rebuild the adopted candidate's geometric contract without running arm IK."""
        if self.geometry is None:
            return None
        identity = self._stance_candidate_identity(self.geometry)
        if identity["source"] != "search_parameters":
            return None

        requested_depth = float(identity["requested_stop_depth"])
        stop_depth = requested_depth
        admitted_target = self._observed_target(state)
        target_depth = self._target_depth(state, admitted_target)
        if self.brake_edge > 0 and edge_distance is not None:
            edge_limited_depth = float(target_depth - (edge_distance - self.brake_edge))
            stop_depth = max(stop_depth, edge_limited_depth)
        travel = max(0.0, float(target_depth - stop_depth))
        stop_edge = None if edge_distance is None else float(edge_distance - travel)
        current = self.geometry.copy()
        current.update(
            stop_depth=stop_depth,
            stop_edge=stop_edge,
            candidate_base=self._candidate_base(state, travel),
        )
        return self._stance_contract([current], admitted_target, edge_distance, state)

    def _cached_stance_geometry_is_valid(self, state, edge_distance):
        current_contract = self._cached_stance_contract(state, edge_distance)
        difference = self._stance_contract_difference(self._stance_valid_contract, current_contract)
        if difference["matches"]:
            self._stance_hit_count += 1
            return True
        if self._stance_valid_contract is not None:
            request = self._stance_valid_request
            result = None if request is None else request.get("result")
            self._record_stance_validation("consumed_plan_invalidated", request, result, difference)
        self._stance_valid_contract = None
        self._stance_valid_request = None
        return False

    def _async_candidate_geometry(self, state, edge_distance, candidates, admitted_target, anchor):
        cfg = self.config
        screened = []
        for source in candidates:
            candidate = source.copy()
            candidate["feasible"] = bool(
                candidate["ik_error"] <= cfg.stance_ik_tolerance
                and (not cfg.wrist_roi_supervisor or candidate["wrist_view"]["visible"])
            )
            screened.append(candidate)
        eligible = [candidate for candidate in screened[: cfg.stance_candidates] if candidate["feasible"]]
        available = [candidate for candidate in screened if candidate["feasible"]]
        current_contract = self._stance_contract(available, admitted_target, edge_distance, state)

        valid = False
        if self.geometry is not None and self.geometry.get("feasible", False):
            difference = self._stance_contract_difference(self._stance_valid_contract, current_contract)
            valid = bool(difference["matches"])
            if valid:
                self._stance_hit_count += 1
            elif self._stance_valid_contract is not None:
                valid_request = self._stance_valid_request
                valid_result = None if valid_request is None else valid_request.get("result")
                self._record_stance_validation("consumed_plan_invalidated", valid_request, valid_result, difference)
                self._stance_valid_contract = None
                self._stance_valid_request = None

        if self._stance_request is not None:
            response = self._stance_ready_response
            self._stance_ready_response = None
            if response is None:
                response = self.stance_planner.poll("walking", self._stance_request["key"])
            self._stance_poll_seconds = float(response.get("poll_seconds", 0.0))
            self._stance_last_status = response["status"]
            if response["status"] == "ready":
                request = self._stance_request
                self._stance_request = None
                result = response["result"]
                self._stance_last_result = result
                if result.get("feasible", False):
                    selected_index = int(result["chosen_index"])
                    request["selected_candidate_index"] = selected_index
                    selected_contract = self._stance_contract_with_selected(request["contract"], selected_index)
                    difference = self._stance_contract_difference(selected_contract, current_contract)
                    if difference["matches"]:
                        chosen = request["candidates"][selected_index].copy()
                        chosen.update(
                            feasible=True,
                            fallback_reason=None,
                            selection_reason=(
                                "full_body_endpoint_and_uncertainty_scenarios"
                                if result.get("path_verification", "").startswith("endpoint_only")
                                else "full_body_expected_and_uncertainty_scenarios"
                            ),
                            full_body=result,
                        )
                        self._stance_last_status = "ready"
                        request["result"] = result
                        self._record_stance_validation("consumed", request, result, difference)
                        return self._publish_stance_geometry(
                            chosen,
                            anchor,
                            selected_contract,
                            request,
                        )
                    self._stance_stale_count += 1
                    self._stance_last_status = "stale"
                    self._record_stance_validation("result_invalidated", request, result, difference)
                else:
                    self._stance_last_status = "infeasible"
                self._stance_retry_wall = time.monotonic()
            elif valid:
                return self.geometry
            elif response["status"] in ("missing", "closed"):
                self._stance_request = None
                self._stance_last_status = response["status"]
                self._stance_retry_wall = time.monotonic()

        if valid:
            return self.geometry

        if not eligible:
            chosen = candidates[0].copy()
            chosen.update(
                feasible=False,
                fallback_reason="no_candidate_passed_arm_ik_and_wrist_view",
                selection_reason="current_bounded_reference_held",
            )
            self._stance_last_status = "arm_infeasible"
            return self._publish_stance_geometry(chosen, anchor)

        retry_ready = self._stance_retry_wall is None or time.monotonic() - self._stance_retry_wall >= 1.0
        if self._stance_request is None and retry_ready:
            self._stance_request_count += 1
            key = self._stance_request_count
            payload_candidates = [
                {
                    "q": candidate["q"],
                    "candidate_base": candidate["candidate_base"],
                    "score": candidate["score"],
                }
                for candidate in eligible
            ]
            request_contract = self._stance_contract(eligible, admitted_target, edge_distance, state)
            response = self.stance_planner.request(
                "walking",
                key,
                {
                    "state": state,
                    "target": admitted_target,
                    "candidates": payload_candidates,
                    "verify_path": False,
                },
            )
            self._stance_poll_seconds = float(response.get("poll_seconds", 0.0))
            self._stance_last_status = response["status"]
            self._stance_request = {
                "key": key,
                "contract": request_contract,
                "candidates": eligible,
                "submitted_wall": time.monotonic(),
            }

        chosen = self._conservative_local_candidate(eligible).copy()
        chosen.update(
            feasible=False,
            preparation_feasible=True,
            fallback_reason=f"stance_plan_{self._stance_last_status}",
            selection_reason="conservative_local_preparation_while_stance_plan_unresolved",
        )
        return self._publish_stance_geometry(chosen, anchor)

    def _candidate_geometry(self, state, edge_distance):
        cfg = self.config
        admitted_target = self._observed_target(state)
        if self.navigation_rotation is None:
            anchor = np.array([self.relative[1], self.relative[2]], dtype=float)
        else:
            anchor = np.array(
                [
                    (admitted_target[:2] - self.navigation_target) @ self.navigation_rotation[:2, 1],
                    admitted_target[2] - STANDING_PELVIS_Z,
                ],
                dtype=float,
            )
        # An unavailable edge later in the same approach does not erase a
        # previously selected edge-aware stance. A newly available edge does
        # upgrade an earlier depth-only selection.
        edge_contract_matches = self.geometry is not None and (
            edge_distance is None or self.geometry.get("stop_edge") is not None
        )
        if (
            self.stance_planner is None
            and edge_contract_matches
            and np.linalg.norm(anchor - self.geometry_anchor) < 0.04
        ):
            if self.geometry.get("feasible", False):
                return self.geometry
        if (
            self.stance_planner is not None
            and edge_contract_matches
            and np.linalg.norm(anchor - self.geometry_anchor) < 0.04
            and self.geometry.get("feasible", False)
            and self._cached_stance_geometry_is_valid(state, edge_distance)
        ):
            return self.geometry
        if (
            self.geometry is not None
            and not self.geometry.get("feasible", False)
            and self._stance_retry_holds_geometry(state)
        ):
            return self.geometry

        depths = self.brake_depth + np.linspace(-cfg.stance_search_span, cfg.stance_search_span, 5)
        specs = [(depth, cfg.pregrasp_gap_max, 0.0) for depth in depths]
        specs.extend(
            (self.brake_depth, cfg.pregrasp_gap_max, lateral)
            for lateral in (-cfg.pregrasp_lateral_span, cfg.pregrasp_lateral_span)
        )
        candidates = []
        target_depth = self._target_depth(state, admitted_target)
        for requested_depth, gap, lateral in specs:
            stop_depth = float(requested_depth)
            # The edge clearance is a physical lower bound on how far the base
            # may travel. Fold it into the candidate depth instead of letting
            # a separate fixed edge gate silently override the selected stance.
            if self.brake_edge > 0 and edge_distance is not None:
                edge_limited_depth = float(target_depth - (edge_distance - self.brake_edge))
                stop_depth = max(stop_depth, edge_limited_depth)
            travel = max(0.0, float(target_depth - stop_depth))
            stop_edge = None if edge_distance is None else float(edge_distance - travel)
            candidate_base = self._candidate_base(state, travel)
            command_gap = gap + cfg.walk_tracking_gap
            candidate_relative = self.rotation.T @ (admitted_target - candidate_base[:3])
            goal = candidate_relative + np.array([-command_gap, lateral, cfg.walk_clearance])
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
        if self.stance_planner is not None and candidates:
            return self._async_candidate_geometry(
                state,
                edge_distance,
                candidates,
                admitted_target,
                anchor,
            )
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
        return {key: value for key, value in geometry.items() if key not in ("q", "candidate_base", "full_body")}

    def _public_stance_result(self):
        result = self._stance_last_result
        if result is None:
            return None
        metric_keys = (
            "feasible",
            "reason",
            "hand_errors_m",
            "foot_errors_m",
            "com_shift_m",
            "natural_joint_margin",
            "natural_posture_cost",
            "joint_action_linf_rad",
            "joint_action_l2_rad",
            "left_arm_action_l2_rad",
            "right_arm_action_l2_rad",
            "torso_pitch_rad",
            "torso_action_rad",
            "solve_seconds",
        )
        return {
            "feasible": result.get("feasible", False),
            "reason": result.get("reason"),
            "chosen_index": result.get("chosen_index"),
            "support_prior_source": result.get("support_prior_source"),
            "solve_seconds": result.get("solve_seconds"),
            "checked": [
                {
                    "candidate_index": candidate["candidate_index"],
                    "candidate_endpoint": candidate.get("candidate_endpoint"),
                    "feasible": candidate["feasible"],
                    "reason": candidate.get("reason"),
                    "selection_score": candidate.get("selection_score"),
                    "scenarios": {
                        name: {key: scenario.get(key) for key in metric_keys}
                        for name, scenario in candidate["scenarios"].items()
                    },
                }
                for candidate in result.get("checked", [])
            ],
        }

    def brake(self, state):
        """Advance the owned upper-body reference to rest without replanning."""
        stamp = float(state["t"])
        dt = 0 if self.last is None else stamp - self.last
        if dt < 0 or dt > 0.5:
            raise ValueError("walking reach physical state clock gap")
        if self.last == stamp:
            return self.q.copy(), self.v.copy()
        self.last = stamp
        if dt > 0:
            acceleration = min(1.0, self.config.joint_acceleration)
            previous_v = self.v.copy()
            self.v += np.clip(-self.v, -acceleration * dt, acceleration * dt)
            self.q += 0.5 * (previous_v + self.v) * dt
        self.wanted = self.q.copy()
        self.diagnostics.update(
            reference=self.q.copy(),
            velocity=self.v.copy(),
            preparation_hold=True,
        )
        return self.q.copy(), self.v.copy()

    def step(
        self,
        state,
        target,
        *,
        ego_px=None,
        camera=None,
        edge_distance=None,
        navigation_heading=None,
        navigation_target=None,
    ):
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
            self._set_planning_frame(target, navigation_heading, navigation_target)
        self._update_stop_envelope(state)

        roi = self._vision(camera, ego_px, dt)
        if (
            target is not None
            and not self.raised
            and self.stance_planner is not None
            and getattr(self.config, "natural_walk_prep", False)
            and (self.planned is None or stamp - self.planned >= 1 / self.config.walk_plan_hz)
        ):
            # Full-body checks start at the first camera-admitted target; the
            # separate ego-size trigger still owns when the arm may move.
            self.planned = stamp
            self._candidate_geometry(state, edge_distance)
            self._update_stop_envelope(state)
        if not self.raised:
            self.wanted = self.q.copy()
        if (
            target is not None
            and self.raised
            and (self.planned is None or stamp - self.planned >= 1 / self.config.walk_plan_hz)
        ):
            self.planned, updated = stamp, True
            cfg = self.config
            heading = self.root.T @ self.rotation[:, 0]
            floor = self.relative[2] + cfg.walk_clearance
            self.progress = float(np.clip((self.brake_depth + 0.9 - self.relative[0]) / 0.9, 0, 1))
            command_feasible = True
            if getattr(cfg, "natural_walk_prep", False):
                geometry = self._candidate_geometry(state, edge_distance)
                self._update_stop_envelope(state)
                goal = np.asarray(geometry["goal"]).copy()
                if not (geometry.get("feasible", False) or geometry.get("preparation_feasible", False)):
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
            stance_plan_status=self._stance_last_status,
            stance_plan_hit_rate=(self._stance_hit_count / max(1, self._stance_hit_count + self._stance_request_count)),
            stance_plan_requests=self._stance_request_count,
            stance_plan_stale_results=self._stance_stale_count,
            stance_plan_poll_seconds=self._stance_poll_seconds,
            stance_plan_age_seconds=(
                None if self._stance_request is None else time.monotonic() - self._stance_request["submitted_wall"]
            ),
            stance_plan_solve_seconds=(
                None if self._stance_last_result is None else self._stance_last_result.get("solve_seconds")
            ),
            stance_plan_result=self._public_stance_result(),
            stance_plan_invalidations=list(self._stance_invalidations),
            stance_plan_consumptions=list(self._stance_consumptions),
        )
        return self.q.copy(), self.v.copy()
