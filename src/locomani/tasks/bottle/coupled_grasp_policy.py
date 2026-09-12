"""Opt-in measured squat + arm transport, with a shared collision screen.

The base does not translate intentionally. Its squat command and right-arm
reference are chosen together. Rejected descent proposals retain the current
arm reference and stop further requested descent. This sampled kinematic screen
is not a whole-body dynamics guarantee.
"""

import time
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.backends.sonic.agent_server import LocomotionMode
from locomani.backends.sonic.primitives import Primitives
from locomani.tasks.bottle.grasp_capture import IK_POSITION_TOL, IK_ROTATION_TOL, closure_geometry, planning_points
from locomani.tasks.bottle.ready_grasp_policy import OFFSET, PRE_GAP, ready_q, templates, terminal_rotation
from locomani.tasks.bottle.recoverable_grasp_policy import RecoverableGraspPolicy


def screen_motion(guard, state, q, dz, forward, *, contact_side=None):
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
            hit = guard.check(trial, q, side, allow_target_contact=side == contact_side)
            if not hit.clear:
                return f"{side}:{hit.link}:{hit.obstacle}"
    return None


TRANSPORT_POSITION_TOL = 0.08
# 0.20 rad = 11.5 deg was a bare literal, and it is SMALLER THAN THE PITCH THE
# SQUAT ITSELF DELIVERS. The transport holds the palm's WORLD attitude while the
# pelvis pitches forward through the crouch (the target is recomputed every tick
# as `br.T @ target_rotation` against the live pelvis rotation), so the pelvis's
# own pitch appears as a rotation residual on the arm. Measured from the pose
# sidecars at the deepest squat (2026-09-11, ledger bottle.md 23): release
# pitches +9.8 to +11.8 deg, v1.1 +12.1 to +12.5, with per-round peaks to
# +14.4 -- i.e. both checkpoints sit ON the old tolerance and v1.1 crosses it.
# The instrumented rounds show exactly that: pos_err 0.009 m against a 0.08
# tolerance (9x inside -- reach was never the problem) while rot_err ran
# 0.2013 / 0.2208 / 0.2323 rad at phase 0.94-0.99, the bottom of the descent.
# 0.35 rad = 20 deg covers the measured peak with margin and still refuses a
# grossly wrong wrist solution; the swept-motion screen (`screen_motion`) and
# `reference_check` below are what actually protect the object, and the
# alignment stage that follows servos the attitude error down to ~0.03 rad
# regardless. Re-derive this from the sidecars if the squat depth changes.
TRANSPORT_ROTATION_TOL = 0.35


def solve_transport_ik(ik, side, aim, rotation, seed):
    """Solve the transport waypoint IK.

    The caller still bounds joint increments and screens their swept motion.
    """
    return ik.solve(side, aim, rotation, q_mj29=seed, use_waist=False, iters=50)


def _pelvis_pitch_deg(state):
    """Pelvis lean, positive nose-down. `project_squat` predicts no pitch at all."""
    return float(np.degrees(np.arcsin(-np.clip(Primitives._rot(state["pelvis"]["quat"])[2, 0], -1.0, 1.0))))


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
            seed_snapshot = q.copy()
            target_root = self.p.world_to_root_full(point - rot @ OFFSET, projected)
            candidate, pe, re = self.arm.ik.solve(
                self.side, target_root, local_rot, q_mj29=q, use_waist=False, iters=80
            )
            # Planning may land anywhere inside the same capture region the
            # executor accepts.  The measured closure geometry and terminal
            # collision checks below remain authoritative.
            if pe > IK_POSITION_TOL or re > IK_ROTATION_TOL:
                diagnostics = self.execution.metrics.setdefault("coupled_prediction_ik_errors", [])
                if len(diagnostics) < 40:
                    # SHADOW SOLVE, recorded and never used: the same target,
                    # seed and rotation with the waist free. Re-solving these
                    # rejections OFFLINE does not reproduce the live residual
                    # even from the recorded seed (0.013 against 0.047, and the
                    # rotation residual 0.002 against 0.103 -- no template
                    # rotation reproduces it, ledger 32), so the only way to
                    # compare the two solves is to run both here.
                    _, waist_pe, waist_re = self.arm.ik.solve(
                        self.side, target_root, local_rot, q_mj29=q, use_waist=True, iters=80
                    )
                    # THE SAME CALL AGAIN, byte-identical arguments. Offline the
                    # recorded (target, rotation, seed) re-solve to 0.068/0.010
                    # while the live call reported 0.115/0.299 -- strictly worse
                    # on BOTH objectives, which a converged damped-least-squares
                    # solve cannot be. Either an input is not what it is recorded
                    # as, or the solver's state is shared: this repeat decides
                    # which, inside the process where it happens.
                    _, repeat_pe, repeat_re = self.arm.ik.solve(
                        self.side, target_root, local_rot, q_mj29=seed_snapshot, use_waist=False, iters=80
                    )
                    diagnostics.append(
                        dict(
                            repeat_position_error_m=float(repeat_pe),
                            repeat_rotation_error_rad=float(repeat_re),
                            seed_mutated=float(np.max(np.abs(q - seed_snapshot))),
                            waist_position_error_m=float(waist_pe),
                            waist_rotation_error_rad=float(waist_re),
                            # The rotation target itself, which nothing recorded:
                            # without it an offline re-solve cannot even be set up.
                            target_rot_local=[float(v) for v in np.asarray(local_rot).ravel()],
                            drop_m=float(drop),
                            position_error_m=float(pe),
                            rotation_error_rad=float(re),
                            target_root=[float(v) for v in target_root],
                            # The SEED, without which this cannot be re-solved
                            # offline: the live solve carries the MEASURED leg
                            # geometry (q[:12]) while `use_waist=False` pins the
                            # waist at the seed's value. Re-solving the recorded
                            # targets from a default-leg seed gave a median
                            # position error of 0.011 m against the 0.040 the
                            # live solve reported (2026-09-12, ledger 31), so the
                            # difficulty lives in the seed and the target alone
                            # does not reproduce it.
                            seed_q=[float(v) for v in np.asarray(seed_snapshot).ravel()],
                            use_waist=False,
                            # The prediction is the only thing between the real
                            # stand and this target, so record both frames: the
                            # target can then be re-derived offline for any
                            # backshift ratio without another run (ledger 32).
                            target_world=[float(v) for v in (point - rot @ OFFSET)],
                            pelvis_pre=[float(v) for v in state["pelvis"]["pos"]],
                            pelvis_pred=[float(v) for v in projected["pelvis"]["pos"]],
                            pelvis_quat=[float(v) for v in state["pelvis"]["quat"]],
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
        from locomani.backends.sonic.stream_reach import SQUAT_HEIGHT_BIAS, toppled
        from locomani.tasks.bottle.measured_grasp import run

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
        _pre_root = np.asarray(p.world_to_root(safe_pregrasp, state))
        # The squat's PASSIVE retreat, against what `project_squat` PREDICTED.
        # Nothing ever compared the two: the loop logged `pelvis_world` but no
        # round recorded the retreat itself, so the ratio (0.8, release-era)
        # could only be checked by reconstructing it from the samples. Measured
        # that way over 111 rounds it is 1.15x on release and 1.43x on v1.1
        # (ledger 32) -- i.e. the prediction is OPTIMISTIC, the delivered squat
        # ends 3.6 / 6.6 cm further from the object than the plan it approved.
        _p0 = np.asarray(state["pelvis"]["pos"]).copy()
        _yaw0 = float(p.pelvis_yaw(state))
        _fwd0 = np.array([np.cos(_yaw0), np.sin(_yaw0)])
        _pitch0 = _pelvis_pitch_deg(state)
        _predicted_retreat = float(
            getattr(self.config, "squat_prediction_backshift_ratio", 0.0)
            * max(0.0, drop - getattr(self.config, "squat_prediction_drop_shortfall", 0.0))
        )

        def _pose_delta(st):
            """Delivered base motion in the PRE-SQUAT frame; +retreat is backwards."""
            d = np.asarray(st["pelvis"]["pos"]) - _p0
            dyaw = float(p.pelvis_yaw(st)) - _yaw0
            return dict(
                drop_m=round(float(-d[2]), 4),
                retreat_m=round(float(-(d[0] * _fwd0[0] + d[1] * _fwd0[1])), 4),
                lateral_m=round(float(-d[0] * _fwd0[1] + d[1] * _fwd0[0]), 4),
                pitch_deg=round(_pelvis_pitch_deg(st) - _pitch0, 2),
                yaw_deg=round(float(np.degrees(np.arctan2(np.sin(dyaw), np.cos(dyaw)))), 2),
                predicted_retreat_m=round(_predicted_retreat, 4),
            )

        stats = dict(
            samples=[],
            descent_ticks=0,
            arm_ticks=0,
            blocked_ticks=0,
            blocked_samples=[],
            requested_drop_m=drop,
            strategy="coupled_squat_to_high_pregrasp",
            safe_pregrasp_world=safe_pregrasp.tolist(),
            pregrasp_gap_m=gap,
            pregrasp_lateral_offset_m=0.0,
            pregrasp_ahead_of_pelvis_m=float(
                (safe_pregrasp - np.asarray(state["pelvis"]["pos"]))
                @ np.array([np.cos(p.pelvis_yaw(state)), np.sin(p.pelvis_yaw(state)), 0.0])
            ),
            # 2026-09-11: `pregrasp_ahead_of_pelvis_m` separated the v1.1 rounds
            # that blocked here from the ones that did not (OK <= -0.044, blocked
            # >= -0.080, ledger 23) -- but one projection cannot say whether the
            # stand was too far out or turned too far, and the failure recorded
            # nothing but a tick count. These decompose it: the pregrasp point in
            # the PELVIS frame, plus the azimuth the arm is asked to fold to.
            pregrasp_root=_pre_root.tolist(),
            pregrasp_azimuth_deg=float(np.degrees(np.arctan2(_pre_root[1], _pre_root[0]))),
            pelvis_at_squat=dict(
                pos=[float(x) for x in state["pelvis"]["pos"]],
                yaw_deg=float(np.degrees(p.pelvis_yaw(state))),
            ),
            grip_root=p.world_to_root(preferred0, state).tolist(),
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
            stats["squat_delivered_pose"] = _pose_delta(state)
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
                    squat_requested=drop,
                    squat_delivered=start_z - z,
                    squat_retreat_m=stats["squat_delivered_pose"]["retreat_m"],
                    squat_retreat_predicted_m=round(_predicted_retreat, 4),
                    reach_started=True,
                    pregrasp_delivered=False,
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
                # The residual is the whole diagnosis and it used to be thrown
                # away: this failure recorded a tick count only. Bounded to the
                # first few so a 4 s loop cannot bloat the manifest.
                if len(stats["blocked_samples"]) < 5:
                    stats["blocked_samples"].append(
                        dict(
                            pos_err_m=round(float(pe), 4),
                            rot_err_rad=round(float(re), 4),
                            tol=[TRANSPORT_POSITION_TOL, TRANSPORT_ROTATION_TOL],
                            aim_root=[round(float(x), 4) for x in p.world_to_root(aim, state)],
                            phase=round(float(phase), 3),
                            delivered_m=round(float(delivered), 4),
                            pelvis_z=round(z, 4),
                        )
                    )
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
