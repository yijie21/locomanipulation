"""Grasp on the VR 3-point channel: command the wrist pose, no arm IK, no prediction.

`CoupledGraspPolicy` plans the squat with `project_squat` (a release-era retreat
constant), solves our own DLS IK against that prediction with the waist locked,
and REJECTS the stand when the solve misses (`coupled_terminal_ik`, the
dominant v1.1 failure in ledger bottle.md 31-33). This policy keeps the same
brake / stand / squat-depth scaffolding and replaces the execution: the wrist
target goes to the controller as a pelvis-frame pose through `TeleopArm`, is
re-expressed in the LIVE pelvis frame every tick, and is corrected on the
measured residual. Whatever the squat does to the base is compensated by
measurement, so nothing here predicts it and nothing rejects on a solve.

Measured on 2026-09-12 (diag_vr3pt, sonic_v1_1): reach ratio 1.16, servo
residual 2.6 cm, rotation 0.88, squat 0.10 m with the world target fixed ->
palm world drift 2.3 cm against a 6.7 cm pelvis retreat, neck lean 1.05.

Stages keep the names the GPU gate knows: grasp.coupled_squat,
grasp.local_align, grasp.capture_ready, grasp.close, grasp.lift.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.backends.sonic.agent_server import LocomotionMode
from locomani.backends.sonic.primitives import LIFT_HEIGHT, Primitives
from locomani.backends.sonic.teleop_arm import TeleopArm
from locomani.tasks.bottle import adaptive_grasp
from locomani.tasks.bottle.coupled_grasp_policy import CoupledGraspPolicy, project_squat
from locomani.tasks.bottle.grasp_capture import LATERAL_TOL, closure_geometry, planning_points
from locomani.tasks.bottle.measured_grasp import CaptureStability, closure_clear
from locomani.tasks.bottle.ready_grasp_policy import OFFSET, PRE_GAP, templates, terminal_rotation

LIFT_MIN = 0.06  # servo.py's rule: the object must actually leave the table
ALIGN_TIMEOUT_S = 12.0
SQUAT_TIMEOUT_S = 3.0
# Align before advancing along the entry axis -- but gate at the CAPTURE
# tolerance, not tighter. measured_grasp.run uses 0.018 because on the coupled
# channel we drove the joints and the palm held 1 cm; here the controller holds
# the arm and its steady-state lateral error is ~2 cm (probe: servo residual
# 2.6 cm), so a 1.8 cm gate is tighter than the channel can deliver and the
# depth advance never unblocks: run 5 round 5 sat at lat -2.4..-2.9 cm with
# depth frozen at +10.5 cm for the whole 12 s window.
LATERAL_ALIGN_M = LATERAL_TOL
# The palm-relative speed that counts as "settled" is measured over a WINDOW,
# not tick to tick. `CaptureStability`'s 0.035 m/s was calibrated against our
# own rate-limited joint stream; on this channel the policy holds the arm with
# its own tremor, and at a 20 ms baseline a 1 mm tremor reads as 0.05 m/s --
# above the threshold forever (run 5 round 4: `safe True` from t=2.1 s to
# t=7.4 s and still a timeout). Over 0.2 s the same tremor reads 0.005-0.01 m/s.
# The THRESHOLD is untouched; only the baseline of the estimate changes.
SPEED_WINDOW_S = 0.2
STEP_M = 0.004  # per-tick cap on the wrist target inside the local align (0.2 m/s)
HOVER_RISE_M = 0.15  # hover height above the grasp wrist while the body squats


def level_jaw(rot):
    """Rotate a palm attitude about its own entry axis (palm x) until the jaw
    axis (palm y) is horizontal. The FK templates carry a 0.20 rad jaw tilt --
    a comfort choice for OUR IK's branch -- and closure_geometry gates the
    MEASURED tilt at 0.30, so with a 0.1 rad attitude residual the gate sat
    on the edge (run 4 round 8: every position tolerance met for 12 s, tilt
    0.30-0.33). The controller delivers any attitude, so ask for a level jaw."""
    rot = np.asarray(rot, dtype=float)
    e, y, z = rot[:, 0], rot[:, 1], rot[:, 2]
    theta = float(np.arctan2(-y[2], z[2]))
    return Rotation.from_rotvec(theta * e).as_matrix() @ rot


class TeleopGraspPolicy(CoupledGraspPolicy):
    """Same brake/stand scaffolding as the coupled policy; execution on TeleopArm."""

    def __init__(self, *args):
        super().__init__(*args)
        self.tarm: TeleopArm | None = None
        self.execution.metrics["teleop_grasp"] = dict(channel="PLANNER_VR_3PT", encoder_mode="teleop")

    # -- planning: choose, never reject on a solve --------------------------

    def assess(self, state, drop):
        """Screen the stand geometrically and pick the terminal rotation.

        The coupled policy's IK screen is deliberately absent: the controller
        owns the arm and the waist on this channel, and a DLS solve with the
        waist locked is not a statement about what it can reach (ledger 32.2:
        73 % of those rejections were reachable with the waist free).
        """
        if not 0.03 <= drop <= self.config.max_drop:
            return self.reject_prediction("coupled_squat_out_of_bounds")
        if not self.floor_clear(state["pelvis"]["pos"][:2], margin=0.06):
            return self.reject_prediction("coupled_base_floor")
        if not all(self.guard.check(state, np.asarray(state["joints"]), sd).clear for sd in ("left", "right")):
            return self.reject_prediction("coupled_initial_clearance")
        # The template still chooses the ATTITUDE (which FK-comfortable centre
        # is nearest); the prediction is used for that choice only.
        projected = project_squat(state, drop, self.config)
        preferred = self.task.grip(state)
        world_rot = Primitives._rot(projected["pelvis"]["quat"])
        point = planning_points(preferred)[0]
        rel = self.p.world_to_root_full(point, projected)
        _, _, local_rot = min(templates(), key=lambda row: np.linalg.norm(row[1] - rel))
        rot = terminal_rotation(world_rot, local_rot, self.config)
        self.execution.metrics["teleop_grasp"]["template_jaw_tilt"] = abs(float(rot[2, 1]))
        self.coupled_rotation = level_jaw(rot)
        self._prepared = None
        self.execution.metrics["teleop_grasp"]["grip_root_at_plan"] = [float(v) for v in rel]
        return True

    def reference_check(self, state, q, phase):
        """Collision screen only. The joint 'comfort limits' of the baseline
        (WRIST_LIMIT, shoulder/elbow 0.65) bound OUR solver's branch choice; on
        this channel the controller chooses the joints, and the first pilot
        round sat inside every capture tolerance for 11 s while this veto read
        'baseline shoulder/elbow comfort limit' off the MEASURED arm."""
        for sd in ("left", "right"):
            hit = self.guard.check(state, q, sd, allow_target_contact=(sd == self.side and phase != "prepare"))
            if not hit.clear:
                error = f"baseline clearance: {hit.link} vs {hit.obstacle}"
                self.execution.metrics.setdefault(
                    "clearance_failure", dict(stage=self.execution.stage, phase=phase, reason=error)
                )
                return error
        return "recovery time budget exhausted" if self.expired() else None

    # -- execution ----------------------------------------------------------

    def hold(self):
        if self.tarm is not None:
            self.tarm.tick()
        else:
            super().hold()

    def finish(self, *, grip_world, squat_hint, ghost):
        ok = False
        try:
            ok, why = self.teleop_transport(float(squat_hint), ghost)
            if not ok:
                self.execution.metrics.setdefault("measured_grasp", []).append(
                    dict(samples=[], can_reposition=False, coupled_failure=why)
                )
            return ok, why
        finally:
            self.finish_cleanup(ok)

    def finish_cleanup(self, ok):
        p = self.p
        if self.tarm is not None:
            # Both outcomes: pin the upper body where it IS and drop the VR
            # field. A failed attempt leaves the arm exactly where it stopped
            # (no stand-up sweep), a successful one hands the held pose to the
            # frozen channel the carry tail runs on.
            self.tarm.stop()
            self.arm.mode, self.arm.height = self.tarm.mode, self.tarm.height
            self.tarm = None
        p.cmd.set_upper_body_vel(np.zeros(17))
        if not ok:
            super().hold()

    def teleop_transport(self, drop, ghost):
        from locomani.tasks.bottle.execution import SQUAT_HEIGHT_BIAS, toppled

        p, ex, side, key = self.p, self.execution, self.side, self.task.target.key
        state = p.obs()
        self.ramp.settle()
        tarm = TeleopArm(p, side)
        self.tarm = tarm
        tarm.start()
        rot = self.coupled_rotation
        grip0 = self.task.grip(state)
        gap = getattr(self.config, "pregrasp_gap", PRE_GAP)
        rise = getattr(self.config, "pregrasp_rise", 0.08)
        wrist0 = grip0 - rot @ OFFSET
        pregrasp = wrist0 - gap * rot[:, 0]
        pregrasp[2] += rise
        if ghost is not None:
            ghost(adaptive_grasp.Grasp(ok=True, rot=rot, wrist=wrist0, grip=grip0, approach=rot[:, 0]))
        z0 = float(state["pelvis"]["pos"][2])
        p0 = np.asarray(state["pelvis"]["pos"], dtype=float)
        yaw0 = float(p.pelvis_yaw(state))
        fwd0 = np.array([np.cos(yaw0), np.sin(yaw0)])
        stats = dict(
            requested_drop_m=drop,
            strategy="teleop_squat_to_pregrasp",
            pregrasp_world=pregrasp.tolist(),
            samples=[],
            align_samples=[],
        )
        ex.metrics.setdefault("coupled_transport", []).append(stats)

        # -- hover first, then squat under the held hand ----------------------------
        # Run 4 round 2 (2026-09-12): the hand was sent to the pregrasp (8 cm above
        # the grasp wrist) WHILE the body squatted; the pelvis dropped 8 cm,
        # retreated 8 cm and pitched, the lagging fingertips dipped into the
        # bottle top and knocked it over (sidecar quat_w 0.30 -> 0.67 inside the
        # squat). So: park the hand at a HOVER well above the grasp, squat with
        # the hover held in the world (the probe measured 2.3 cm of world drift
        # through a 0.10 m squat), and only then descend along the entry axis.
        ex.enter("grasp.coupled_squat")
        hover = pregrasp.copy()
        hover[2] = wrist0[2] + max(rise, HOVER_RISE_M)
        stats["hover_world"] = hover.tolist()
        tarm.move_to(hover, rot, settle=0.3)
        if drop >= 0.03:
            tarm.mode = LocomotionMode.IDEL_SQUAT
            tarm.height = float(np.clip(z0 - drop - SQUAT_HEIGHT_BIAS, 0.40, 0.78))
        start = time.monotonic()
        delivered = 0.0
        while time.monotonic() - start < SQUAT_TIMEOUT_S and not self.expired():
            state = tarm.tick()
            if state["fallen"] or toppled(state, key):
                return False, "teleop transport disturbed target or base"
            delivered = z0 - float(state["pelvis"]["pos"][2])
            if time.monotonic() - start > 0.6 and (drop < 0.03 or delivered >= drop - 0.02):
                break
        d = np.asarray(state["pelvis"]["pos"]) - p0
        stats["squat_delivered_pose"] = dict(
            drop_m=round(float(-d[2]), 4),
            retreat_m=round(float(-(d[0] * fwd0[0] + d[1] * fwd0[1])), 4),
            residual_m=round(tarm.residual(state), 4),
        )
        if drop >= 0.03 and delivered < drop - 0.03:
            stats["delivered_drop_m"] = delivered
            return False, "teleop squat was not delivered"
        # hold the delivered height from here on, exactly like the coupled policy
        tarm.height = float(state["pelvis"]["pos"][2]) - SQUAT_HEIGHT_BIAS
        tarm.move_to(pregrasp, rot, settle=0.2)
        state = p.obs()
        if state["fallen"] or toppled(state, key):
            return False, "teleop descent disturbed target or base"
        ex.transition.update(
            squat_requested=drop, squat_delivered=delivered, reach_started=True, pregrasp_delivered=True
        )

        # -- local align on the measured object ---------------------------------
        ex.enter("grasp.local_align")
        stable = CaptureStability(hysteresis=True)
        history: deque = deque()  # (t, object position in the palm frame) for the windowed speed
        start = time.monotonic()
        last_log = 0.0
        touched = False
        obj_z0 = float(p.object_pos(key, state)[2])
        while time.monotonic() - start < ALIGN_TIMEOUT_S and not self.expired():
            state = p.obs()
            now = time.monotonic()
            if state["fallen"] or toppled(state, key):
                return False, "local alignment disturbed target or base"
            target = self.task.grip(state)
            palm = np.asarray(state["palms"][side]["pos"])
            measured_rot = Primitives._rot(state["palms"][side]["quat"])
            geometry = closure_geometry(state, side, target)
            touched |= side in state["objects"][key].get("touching", [])
            relative = measured_rot.T @ (target - palm)
            history.append((now, relative.copy()))
            while len(history) > 2 and now - history[0][0] > SPEED_WINDOW_S:
                history.popleft()
            span = now - history[0][0]
            speed = (
                float("inf") if span < SPEED_WINDOW_S * 0.5 else float(np.linalg.norm(relative - history[0][1]) / span)
            )
            clear = closure_clear(self, state) if geometry["ready"] else None
            safe = geometry["ready"] and clear is None
            if now - last_log >= 0.2:
                resid = tarm.residual(state)
                stats["align_samples"].append(
                    dict(elapsed_s=now - start, **geometry, clearance=clear, residual_m=resid)
                )
                if len(stats["align_samples"]) % 5 == 1:
                    rot_err = float(np.degrees(np.linalg.norm(Rotation.from_matrix(measured_rot.T @ rot).as_rotvec())))
                    p.log(
                        f"[align] t {now - start:4.1f}s depth {geometry['depth_m'] * 100:+5.1f} lat "
                        f"{geometry['lateral_signed_m'] * 100:+5.1f} h {geometry['world_height_error_m'] * 100:+5.1f} cm "
                        f"tilt {geometry['jaw_tilt']:.2f} (target {abs(float(rot[2, 1])):.2f}) rot_err {rot_err:4.1f} deg "
                        f"resid {resid * 100:4.1f} cm safe {safe} clear {clear}"
                    )
                last_log = now
            if stable.update(now, safe, speed):
                stats.update(alignment_s=now - start, close_geometry=geometry, touched=touched)
                ex.enter("grasp.capture_ready", **geometry)
                break
            if safe:
                tarm.tick(state)  # seated: hold, let the speed settle
                continue
            desired = target - rot @ OFFSET
            error = desired - palm
            entry = rot[:, 0]
            depth = float(error @ entry)
            lateral = error - depth * entry
            if np.linalg.norm(lateral) > LATERAL_ALIGN_M:
                desired = desired - max(depth, 0.0) * entry  # align before advancing
            cur = tarm.target[side]["pos"]
            step = desired - cur
            n = float(np.linalg.norm(step))
            if n > STEP_M:
                step *= STEP_M / n
            tarm.target[side] = dict(pos=cur + step, rot=rot)
            tarm.tick(state)
        else:
            state = p.obs()
            geometry = closure_geometry(state, side, self.task.grip(state))
            stats["align_timeout"] = dict(residual_m=tarm.residual(state), geometry=geometry, touched=touched)
            return False, (
                "local alignment timed out: "
                + ",".join(geometry["rejection_reasons"] or ["speed"])
                + f" depth {geometry['depth_m'] * 100:+.1f} lat {geometry['lateral_signed_m'] * 100:+.1f}"
                + f" h {geometry['world_height_error_m'] * 100:+.1f} cm, residual {tarm.residual(state) * 100:.1f} cm"
            )

        # -- close ---------------------------------------------------------------
        ex.enter("grasp.close")
        if not tarm.close_hand(key):
            return False, "fingers closed without contact"

        # -- lift ----------------------------------------------------------------
        ex.enter("grasp.lift")
        state = p.obs()
        lift_to = np.asarray(state["palms"][side]["pos"]) + np.array([0.0, 0.0, LIFT_HEIGHT])
        tarm.move_to(lift_to, None, settle=1.0)
        state = p.obs()
        lifted = float(p.object_pos(key, state)[2]) - obj_z0
        stats["lifted_m"] = lifted
        if state["fallen"]:
            return False, "fell during the lift"
        if lifted < LIFT_MIN:
            return False, f"{key} rose only {lifted:.3f} m, need > {LIFT_MIN}"
        p.log(f"  teleop: lifted {lifted * 100:.1f} cm")
        return True, "teleop grasp lifted"
