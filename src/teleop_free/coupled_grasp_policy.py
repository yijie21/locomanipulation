"""Opt-in measured squat + arm transport, with a shared collision screen.

The base does not translate intentionally. Its squat command and right-arm
reference are chosen together. Rejected descent proposals retain the current
arm reference and stop further requested descent. This sampled kinematic screen
is not a whole-body dynamics guarantee.
"""

import time
from copy import deepcopy

import numpy as np
from agent_server import LocomotionMode
from grasp_capture import IK_POSITION_TOL, IK_ROTATION_TOL, closure_geometry, planning_points
from primitives import Primitives
from ready_grasp_policy import OFFSET, PRE_GAP, ready_q, templates, terminal_rotation
from recoverable_grasp_policy import RecoverableGraspPolicy
from scipy.spatial.transform import Rotation


def screen_motion(guard, state, q, dz, forward):
    """Screen joint sweep and small base-motion uncertainty together."""
    for u in (0.0, 0.5, 1.0):
        trial = deepcopy(state)
        # A 5mm squat/backshift lookahead leaves space for delivery lag.
        uncertainty = 0.005 if dz < -1e-6 else 0.0
        trial["pelvis"]["pos"] = (
            np.asarray(state["pelvis"]["pos"])
            + [0.0, 0.0, u * (dz - uncertainty)]
            - u * uncertainty * np.asarray(forward)
        ).tolist()
        # Upper-joint references do not move the body.  Base/body clearance
        # belongs to stand selection and the terminal squat projection; an
        # existing hip/table near-contact must not veto every arm command.
        for side in (side for side in guard.hands if side != "body"):
            hit = guard.check(trial, q, side, allow_target_contact=False)
            if not hit.clear:
                return f"{side}:{hit.link}:{hit.obstacle}"
    return None


TRANSPORT_POSITION_TOL = 0.08
TRANSPORT_ROTATION_TOL = 0.20


def solve_transport_ik(ik, side, aim, rotation, seed):
    """Solve the transport waypoint IK.

    The caller still bounds joint increments and screens their swept motion.
    """
    return ik.solve(side, aim, rotation, q_mj29=seed, use_waist=False, iters=50)


def project_squat(state, drop, config):
    """Predict the delivered squat pose, including calibrated passive retreat.

    These are prediction parameters; no base translation is commanded.
    Zero defaults retain the original purely vertical prediction.
    """
    projected = deepcopy(state)
    delivered = max(0.0, drop - getattr(config, "squat_prediction_drop_shortfall", 0.0))
    rotation = Primitives._rot(state["pelvis"]["quat"])
    forward = rotation[:2, 0].copy()
    forward /= max(float(np.linalg.norm(forward)), 1e-9)
    backshift = delivered * getattr(config, "squat_prediction_backshift_ratio", 0.0)
    projected["pelvis"]["pos"] = (
        np.asarray(state["pelvis"]["pos"]) - np.array([*(backshift * forward), delivered])
    ).tolist()
    return projected


class CoupledGraspPolicy(RecoverableGraspPolicy):
    def brake(self):
        self._prepared = None
        return super().brake()

    def select_drop(self, state, hint):
        cached = getattr(self, "_prepared", None)
        self._prepared = None
        if cached is not None:
            drop, candidate, rot = cached
            # Re-evaluate the CURRENT geometry. Cached IK is only a candidate:
            # base drift, target motion and all terminal obstacles still veto it.
            if self.floor_clear(state["pelvis"]["pos"][:2], margin=0.06) and all(
                self.guard.check(state, np.asarray(state["joints"]), sd).clear for sd in ("left", "right")
            ):
                trial = project_squat(state, drop, getattr(self, "config", None))
                candidate = candidate.copy()
                candidate[:12] = np.asarray(state["joints"])[:12]
                palm, local_rot = self.arm.ik.fk(candidate, self.side)
                world_rot = Primitives._rot(trial["pelvis"]["quat"])
                trial["palms"][self.side]["pos"] = (np.asarray(trial["pelvis"]["pos"]) + world_rot @ palm).tolist()
                trial["palms"][self.side]["quat"] = (
                    Rotation.from_matrix(world_rot @ local_rot).as_quat()[[3, 0, 1, 2]].tolist()
                )
                trial["joints"] = candidate
                if closure_geometry(trial, self.side, self.task.grip(state))["ready"] and all(
                    self.guard.check(trial, candidate, sd, allow_target_contact=sd == self.side).clear
                    for sd in self.guard.hands
                ):
                    self.coupled_rotation = world_rot @ rot
                    records = self.execution.metrics.get("brake_planning", [])
                    if records and self.execution.stage != "recovery_reposition":
                        records[-1]["reused"] = True
                    self.execution.metrics.setdefault("cached_plan_reuses", []).append(
                        dict(stage=self.execution.stage, drop_m=float(drop))
                    )
                    return drop
        return super().select_drop(state, hint)

    def assess(self, state, drop):
        # Do not reject a stand because FIXED ready arms would hit the bottle
        # during descent. Only screen current geometry and a feasible goal;
        # the coupled controller screens each executed approach increment.
        if not 0.03 <= drop <= self.config.max_drop:
            return self.reject_prediction("coupled_squat_out_of_bounds")
        if not self.floor_clear(state["pelvis"]["pos"][:2], margin=0.06):
            return self.reject_prediction("coupled_base_floor")
        if not all(self.guard.check(state, np.asarray(state["joints"]), sd).clear for sd in ("left", "right")):
            return self.reject_prediction("coupled_initial_clearance")
        projected = project_squat(state, drop, self.config)
        preferred = self.task.grip(state)
        world_rot = Primitives._rot(projected["pelvis"]["quat"])
        for point in planning_points(preferred):
            rel = self.p.world_to_root_full(point, projected)
            _, _, local_rot = min(templates(), key=lambda row: np.linalg.norm(row[1] - rel))
            rot = terminal_rotation(world_rot, local_rot, self.config)
            local_rot = world_rot.T @ rot
            q = ready_q()
            q[:12] = np.asarray(state["joints"])[:12]
            candidate, pe, re = self.arm.ik.solve(
                self.side,
                self.p.world_to_root_full(point - rot @ OFFSET, projected),
                local_rot,
                q_mj29=q,
                use_waist=False,
                iters=80,
            )
            # Planning may land anywhere inside the same capture region the
            # executor accepts.  The measured closure geometry and terminal
            # collision checks below remain authoritative.
            if pe > IK_POSITION_TOL or re > IK_ROTATION_TOL:
                diagnostics = self.execution.metrics.setdefault("coupled_prediction_ik_errors", [])
                if len(diagnostics) < 40:
                    diagnostics.append(
                        dict(
                            drop_m=float(drop),
                            position_error_m=float(pe),
                            rotation_error_rad=float(re),
                            target_root=list(map(float, self.p.world_to_root_full(point - rot @ OFFSET, projected))),
                        )
                    )
                self.reject_prediction("coupled_terminal_ik")
                continue
            palm, actual_rot = self.arm.ik.fk(candidate, self.side)
            trial = deepcopy(projected)
            trial["palms"][self.side]["pos"] = (np.asarray(projected["pelvis"]["pos"]) + world_rot @ palm).tolist()
            trial["palms"][self.side]["quat"] = (
                Rotation.from_matrix(world_rot @ actual_rot).as_quat()[[3, 0, 1, 2]].tolist()
            )
            if not closure_geometry(trial, self.side, preferred)["ready"]:
                self.reject_prediction("coupled_terminal_capture")
                continue
            # Terminal collision check, not a sweep from fixed ready arms at
            # the bottom of the squat (which is not our approach path).
            trial["joints"] = candidate
            if not all(
                self.guard.check(trial, candidate, sd, allow_target_contact=(sd == self.side)).clear
                for sd in self.guard.hands
            ):
                self.reject_prediction("coupled_terminal_clearance")
                continue
            self.coupled_rotation = rot
            self._prepared = (drop, candidate.copy(), local_rot.copy())
            return True
        return False

    def finish(self, *, grip_world, squat_hint, ghost):
        ok = False
        try:
            ok, why = self.coupled_transport(float(squat_hint), ghost)
            if not ok:
                # A tracking/sweep failure is not evidence that standing up
                # through the same obstruction would help.
                self.execution.metrics.setdefault("measured_grasp", []).append(
                    dict(samples=[], can_reposition=False, coupled_failure=why)
                )
            return ok, why
        finally:
            self.finish_cleanup(ok)

    def coupled_transport(self, drop, ghost):
        from measured_grasp import run
        from stream_reach import SQUAT_HEIGHT_BIAS, toppled

        p, arm, ex = self.p, self.arm, self.execution
        state = p.obs()
        start_z = float(state["pelvis"]["pos"][2])
        target_z = start_z - drop
        start_palm = np.asarray(state["palms"][self.side]["pos"]).copy()
        # Keep the already-clear wrist attitude while leaving the bottle.
        # Rotating toward the terminal grasp at a close stop can sweep a long
        # Dex1 finger through the object even when the palm path moves away.
        target_rotation = Primitives._rot(state["palms"][self.side]["quat"])
        grasp_rotation = self.coupled_rotation
        preferred0 = self.task.grip(state)
        gap = getattr(self.config, "pregrasp_gap", PRE_GAP)
        safe_pregrasp = preferred0 - grasp_rotation @ OFFSET - gap * grasp_rotation[:, 0]
        safe_pregrasp[2] += getattr(self.config, "pregrasp_rise", 0.08)
        arm.mode = LocomotionMode.IDEL_SQUAT
        arm.height = target_z - SQUAT_HEIGHT_BIAS
        arm._face = p._facing(state)
        self.ramp.settle()
        arm._q_prev = arm.q.copy()
        arm.next_t = time.time()
        stats = dict(
            samples=[],
            descent_ticks=0,
            arm_ticks=0,
            blocked_ticks=0,
            requested_drop_m=drop,
            strategy="coupled_squat_to_high_pregrasp",
            safe_pregrasp_world=safe_pregrasp.tolist(),
            pregrasp_gap_m=gap,
            pregrasp_lateral_offset_m=0.0,
            pregrasp_ahead_of_pelvis_m=float(
                (safe_pregrasp - np.asarray(state["pelvis"]["pos"]))
                @ np.array([np.cos(p.pelvis_yaw(state)), np.sin(p.pelvis_yaw(state)), 0.0])
            ),
        )
        ex.metrics.setdefault("coupled_transport", []).append(stats)
        ex.enter("grasp.coupled_squat")
        self.hold()
        start = time.monotonic()
        last_log = 0.0
        last_z = start_z
        while time.monotonic() - start < 4.0 and not self.expired():
            state = p.obs()
            now = time.monotonic()
            if state["fallen"] or toppled(state, self.task.target.key):
                return False, "coupled transport disturbed target or base"
            palm = np.asarray(state["palms"][self.side]["pos"])
            br = Primitives._rot(state["pelvis"]["quat"])
            base = np.asarray(state["pelvis"]["pos"])
            z = float(base[2])
            preferred = self.task.grip(state)
            geometry = closure_geometry(state, self.side, preferred)
            delivered = start_z - z
            phase = float(np.clip(delivered / max(drop - 0.02, 0.001), 0.0, 1.0))
            phase = phase * phase * (3.0 - 2.0 * phase)
            aim = start_palm + phase * (safe_pregrasp - start_palm)
            aim_error = float(np.linalg.norm(aim - palm))
            if geometry["ready"] or (
                now - start > 0.8 and delivered >= drop - 0.02 and np.linalg.norm(safe_pregrasp - palm) < 0.06
            ):
                arm.height = z - SQUAT_HEIGHT_BIAS
                self.hold()
                stats.update(
                    delivered_drop_m=start_z - z,
                    transport_s=now - start,
                    handover="capture" if geometry["ready"] else "squat_delivered",
                )
                ex.transition.update(
                    squat_requested=drop, squat_delivered=start_z - z, reach_started=True, pregrasp_delivered=False
                )
                return run(self, ghost)
            forward = np.array([np.cos(p.pelvis_yaw(state)), np.sin(p.pelvis_yaw(state)), 0.0])
            seed = arm.q.copy()
            seed[:12] = np.asarray(state["joints"])[:12]
            q, pe, re = solve_transport_ik(
                arm.ik,
                self.side,
                p.world_to_root_full(aim, state),
                br.T @ target_rotation,
                seed,
            )
            if pe > TRANSPORT_POSITION_TOL or re > TRANSPORT_ROTATION_TOL:
                stats["blocked_ticks"] += 1
                arm.height = z - SQUAT_HEIGHT_BIAS
                self.hold()
                if stats["blocked_ticks"] > 10:
                    return False, "coupled safe-hand IK did not track"
                continue
            q = seed + np.clip(q - seed, -0.025, 0.025)
            # The arm is re-solved every control tick.  Screen the local leg
            # advance it can experience before the next correction, rather
            # than pairing today's arm joints with the full final squat.
            preview_dz = max(target_z - z, -0.008)
            rejection = screen_motion(self.guard, state, q, preview_dz, forward)
            if rejection:
                stats["blocked_ticks"] += 1
                arm.height = z - SQUAT_HEIGHT_BIAS
                self.hold()
                if stats["blocked_ticks"] > 10:
                    stats["last_rejection"] = rejection
                    return False, f"coupled safe-hand path blocked: {rejection}"
                continue
            rejection = self.reference_check(state, q, "prepare")
            if rejection:
                return False, rejection
            if now - last_log >= 0.2:
                stats["samples"].append(
                    dict(
                        elapsed_s=now - start,
                        pelvis_world=base.tolist(),
                        palm_world=palm.tolist(),
                        grip_world=preferred.tolist(),
                        command_height=target_z - SQUAT_HEIGHT_BIAS,
                        phase=phase,
                        aim_world=aim.tolist(),
                        aim_error_m=aim_error,
                        **geometry,
                    )
                )
                last_log = now
            stats["descent_ticks"] += int(z < last_z - 1e-5)
            stats["arm_ticks"] += int(np.max(np.abs(q - arm.q)) > 1e-6)
            last_z = z
            arm.height = target_z - SQUAT_HEIGHT_BIAS
            arm.q = q
            arm._push(q, observed=state)
        self.hold()
        stats["delivered_drop_m"] = start_z - float(p.obs()["pelvis"]["pos"][2])
        return False, "coupled squat was not delivered"
