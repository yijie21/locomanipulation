"""Visible-target reach with simulation-assisted, continuously coordinated body feedback."""

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from locomani.tasks.crate.grasp_region import GRASP_BELOW_RIM_M, GRASP_POINTS, RimRegion
from locomani.tasks.crate.visual_low_reach import LowReachCommand, blend, quaternion, rotation
from locomani.tasks.crate.whole_body_ik import WholeBodyIK


@dataclass(frozen=True)
class CoordinatedTarget:
    """Object-local grasp geometry consumed by :class:`CoordinatedReach`.

    ``capture_rotation_local`` has capture across, tangent and up axes as
    columns. The default exactly describes the historical blue-crate target.
    """

    object_key: str = "blue_crate"
    grasp_points_local: tuple[tuple[float, float, float], ...] = GRASP_POINTS
    capture_rotation_local: np.ndarray = field(default_factory=lambda: np.eye(3))
    region: RimRegion = field(default_factory=RimRegion)
    information_source: str = "simulation_assisted"
    preserve_point_order: bool = False
    reach_only: bool = False
    rim_above_grasp: float = GRASP_BELOW_RIM_M  # rim top height above the grasp points; lateral_hold_below_rim uses it

    def __post_init__(self):
        points = np.asarray(self.grasp_points_local, dtype=float)
        frame = np.asarray(self.capture_rotation_local, dtype=float)
        if points.shape != (2, 3) or not np.isfinite(points).all():
            raise ValueError("grasp_points_local must contain two finite 3D points")
        if frame.shape != (3, 3) or not np.isfinite(frame).all():
            raise ValueError("capture_rotation_local must be a finite 3x3 rotation")
        if not np.allclose(frame.T @ frame, np.eye(3), atol=1e-6) or np.linalg.det(frame) < 0.999999:
            raise ValueError("capture_rotation_local must be right-handed and orthonormal")
        if not self.object_key:
            raise ValueError("object_key must be non-empty")
        object.__setattr__(self, "grasp_points_local", tuple(map(tuple, points)))
        object.__setattr__(self, "capture_rotation_local", frame.copy())


class CoordinatedReach:
    """One supported body reference through descent, capture and paired lifting.

    Simulation supplies measured world poses; reference motion remains bounded,
    feet stay anchored, and unseen targets cannot advance the trajectory.
    """

    def __init__(
        self,
        solver,
        config,
        state,
        *,
        reference_joints,
        reference_quaternion,
        target: CoordinatedTarget | None = None,
    ):
        self.solver, self.config = solver, config
        self.target = CoordinatedTarget() if target is None else target
        self.region = self.target.region
        self.body = WholeBodyIK(
            solver.ik,
            state,
            config.carry,
            tool_offset=solver.offset,
            joint_limits=(solver.lo, solver.hi),
            com_shift_max=config.carry.com_shift_max,
            com_free_radius=config.carry.com_free_radius,
            knee_max=config.carry.knee_max,
            pelvis_forward_max=config.carry.pelvis_forward_max,
            pelvis_height_target=config.carry.pelvis_height_target,
            elbow_min_deg=config.visual.elbow_min_deg,
            jacobian=config.carry.ik_jacobian,
        )
        # The crouch serves the reach; standing up with the load returns the pelvis to its start height.
        self.crouch_height = config.carry.pelvis_height_target
        self.anchor_release = 0.0  # 0 pins the solver base to the measured pelvis; 1 frees it (stand-up)
        self.q = np.asarray(reference_joints).copy()
        self.root = np.asarray(reference_quaternion).copy()
        pitch = Rotation.from_matrix(rotation(self.root)).as_euler("xyz")[1]
        self.body.heading = rotation(self.root) @ Rotation.from_euler("y", -pitch).as_matrix()
        # The consumed planner reference is continuous; infer its translation from the same feet.
        self.x = np.r_[state["pelvis"]["pos"], pitch, self.q]
        self.body.put(self.x)
        self.x[:3] += np.mean(self.body.feet_pos - solver.ik.data.xpos[self.body.feet], axis=0)
        self.travel_lo, self.travel_hi = np.minimum(self.body.lo, self.x), np.maximum(self.body.hi, self.x)
        self.last_dt = None
        self.v = np.zeros(33)
        self.velocity = np.zeros(29)
        self.phase, self.closure = "observe", 0.0
        self.measured_base = None  # measured pelvis in the solver frame, set each step
        self.started = self.last_time = self.last_seen = None
        self.was_visible = False
        self.elapsed = 0.0
        self.ready_since = None
        self.feedback = np.zeros((2, 3))
        self.align_started, self.align_residual = None, None
        self.feedforward = np.zeros(3)
        self.rotation_feedback = np.zeros((2, 3))  # filtered predicted-vs-measured palm rotation vectors (world)
        self.ghost = None  # GhostTargetTrace: every commanded (pre-feedback) grip target + palm attitude, by state seq
        self.collision_guard = None  # HandCollisionGuard (visual.collision_guard mink): filters align/descend targets
        # visual.reach_route planned: callable(start_joints, goal_joints, state) -> (T, 29) collision-free joint path
        # from the reference to the pregrasp solve, or None when there is none (tasks/crate/reach_route.py); set by
        # the runner, which also skips the ready pose
        self.route_planner = None
        self.route = None  # the planned path, its timing and its hand poses, once planned (plan_route)
        self._seq = None
        self.contact_sides = ()  # hands the privileged state reports touching the target object this tick
        self.best_error, self.progress_time = np.inf, None
        self.retries = dict(descend=0, reseat=0)  # visual.descend_retries / reseat_retries used so far
        self.last_retry = None  # the latest retry (kind, after, count), kept in the diagnostics from then on
        self.attachment_feedback = np.zeros(2)  # per hand: filtered vertical drift from the closed hand-box offset, m
        self.reopen_targets = None
        self.diagnostics = {}
        self.torso_target = None
        self.previous_load = None
        self.hold_window = deque()

    def result(self, reason="", finished=False):
        return LowReachCommand(
            self.q.copy(), self.velocity.copy(), self.root.copy(), self.closure, self.phase, reason, finished
        )

    def enter(self, phase):
        self.phase, self.elapsed, self.ready_since = phase, 0.0, None
        if phase == "align" and self.collision_guard is not None:
            self.collision_guard.reset()  # the shadow restarts from the measured arms at every (re)alignment

    def guarded(self, tools, attitudes, state, dt):
        """visual.collision_guard: the intended grip targets and palm attitudes as the guard's shadow could reach
        them without a hand meeting an obstacle; unchanged without a guard."""
        if self.collision_guard is None:
            return tools, attitudes
        grips, rotations, clearance = self.collision_guard.filter(tools, attitudes, state, dt)
        self.diagnostics.update(
            guard_clearance=clearance,
            guard_shift=np.linalg.norm(grips - np.asarray(tools), axis=1),
        )
        return grips, rotations

    def retry(self, kind, now, after):
        """Try the descend again instead of ending the round on `after`: `descend` backs up to the pregrasp point
        (the align phase re-centres there), `reseat` first opens the jaws where the hands are. Returns this tick's
        command, or None when visual.<kind>_retries is used up. Every failure was terminal until 2026-09-27."""
        cfg = self.config.visual
        if self.retries[kind] >= (cfg.descend_retries if kind == "descend" else cfg.reseat_retries):
            return None
        self.retries[kind] += 1
        self.last_retry = dict(kind=kind, after=after, count=dict(self.retries), at=round(float(now), 3))
        self.diagnostics["retry"] = self.last_retry
        self.best_error, self.progress_time = np.inf, None
        if kind == "reseat":
            self.reopen_targets = self.body.tools(self.x) - cfg.tracking_gain * self.feedback
            self.enter("reopen")
        else:
            self.align_started = now
            self.enter("align")
        return self.result()

    def descend_goal(self, targets):
        """Where the staged descent aims: the grasp points, or contact_overdrive below them in contact mode."""
        cfg = self.config.visual
        if cfg.close_trigger == "contact" and cfg.contact_overdrive > 0:
            return targets - np.array([0.0, 0.0, cfg.contact_overdrive])
        return targets

    def start_descend(self, targets, first):
        """Straight down from the pregrasp point, at the same Cartesian speed limit."""
        cfg = self.config.visual
        drop = np.max(np.linalg.norm(self.descend_goal(targets) - first, axis=1))
        self.descend_from = self.body.tools(self.x)
        self.descend_duration = max(1.5, 1.875 * drop / (cfg.descend_speed or cfg.approach_speed))
        self.best_error, self.progress_time = np.inf, None
        self.enter("descend")

    def contact_ready(self, centres, obj_rotation, tools):
        """visual.close_trigger=contact: close when each jaw is physically around its wall, wherever the hand
        landed -- the wall inside the open jaw (|across| <= contact_across) and the grip point under the rim
        top by contact_below_rim, or that hand already touching the object with the grip point at most 1.5 cm
        above the rim (a pad or the palm has met the wall; descending further only jams). No palm-attitude
        gate: the delivered attitude is what SONIC gave, closing on it beats stalling 2-5 cm away."""
        cfg = self.config.visual
        error = (np.asarray(tools) - centres) @ obj_rotation  # per hand: across, along, up
        rim = self.target.rim_above_grasp
        ready = []
        for i, side in enumerate(("left", "right")):
            across, along, up = error[i]
            inside = abs(across) <= cfg.contact_across and abs(along) <= cfg.contact_along
            entered = up <= rim - cfg.contact_below_rim
            touched = side in self.contact_sides and up <= rim + 0.015
            ready.append(inside and (entered or touched))
        return bool(all(ready)), error

    def capture_ready(self, centres, obj_rotation, tools, tool_rotations):
        if self.config.visual.close_trigger == "contact":
            return self.contact_ready(centres, obj_rotation, tools)
        return self.region.contains(centres, obj_rotation, tools, tool_rotations)

    def forward_floor(self, targets, state):
        """Stand/hold hand targets kept at least visual.stand_forward_min ahead of the CURRENT pelvis along the
        body-to-object axis. A floor applied once at closing is not enough: the pelvis itself advances 11-17 cm
        while the torso straightens (2026-09-24), and a deep box rides into the torso."""
        cfg = self.config.visual
        if cfg.stand_forward_min <= 0:
            return targets
        pelvis_xy = np.asarray(state["pelvis"]["pos"][:2], dtype=float)
        forward = float((np.mean(targets, axis=0)[:2] - pelvis_xy) @ self.outward[:2])
        self.diagnostics["stand_forward_m"] = max(forward, cfg.stand_forward_min)
        if forward < cfg.stand_forward_min:
            return targets + (cfg.stand_forward_min - forward) * self.outward
        return targets

    def follow_attitudes(self, obj_rotation, dt):
        """Palm attitudes for the loaded phases: the reach attitudes, rotated about the hand-hand axis
        against the box's measured pitch since closing (visual.lift_pitch_follow_gain).

        A rim pinch is a hinge: the box's pitch about the line between the hands is held only by pad
        friction, so it follows the palms' attitude drift during the lift and swings to the tilt gate
        (2026-09-24). Rotating the commanded palms the other way by gain x pitch brings it back, as the
        crate line's bar grasp needed (pitch follow, 2026-09-13). Low-passed (0.2 s) and capped.
        """
        cfg = self.config.visual
        if cfg.lift_pitch_follow_gain <= 0 or getattr(self, "closed_rotation", None) is None:
            return self.attitudes
        axis = obj_rotation[:, 0]  # world direction of the box's grasp axis (end wall to end wall)
        relative = Rotation.from_matrix(obj_rotation @ self.closed_rotation.T).as_rotvec()
        pitch = float(relative @ axis)
        if dt > 0:
            self.pitch_follow += (1 - np.exp(-dt / 0.2)) * (pitch - self.pitch_follow)
        angle = float(
            np.clip(
                -cfg.lift_pitch_follow_gain * self.pitch_follow,
                -np.radians(cfg.lift_pitch_follow_cap_deg),
                np.radians(cfg.lift_pitch_follow_cap_deg),
            )
        )
        self.diagnostics.update(pitch_follow_measured_deg=np.degrees(pitch), pitch_follow_command_deg=np.degrees(angle))
        correction = Rotation.from_rotvec(angle * axis).as_matrix()
        return np.array([correction @ a for a in self.attitudes])

    def advance(
        self, targets, attitudes, measured, dt, measured_rotations=None, z_floor=None, hold_across=None, posture=None
    ):
        """Rate/acceleration limits belong inside the supported body solve, including hips.

        `z_floor` (per hand, world z): the height the measured grip point must not be driven below. A hand at
        or under it keeps its z feedback from growing downward and backs it off by the undershoot.
        `hold_across` = (mask per hand, world unit vector): a held hand's feedback may not change along the
        vector. Below the box rim a pad that meets a wall face is blocked, and the across-wall correction
        otherwise winds up to tracking_cap against it and shoves the wall until the box tips (2026-09-24).
        With visual.lateral_hold_common_mode only the differential part of that change is removed: the pair's
        mean shift stays free, because the stalls of 2026-09-27 were the PAIR drifting 3-5 cm to one side.
        `posture`: this tick's posture prior instead of the admission solve's (a planned route's joints).
        """
        if dt <= 0:
            return
        cfg = self.config.visual
        if self.ghost is not None:
            # The pure reference (before position/orientation feedback), as the ghost-hands rule requires.
            self.ghost.row(
                self._seq, {"left": targets[0], "right": targets[1]}, {"left": attitudes[0], "right": attitudes[1]}
            )
        predicted = self.body.tools(self.x)
        observed_error = np.clip(predicted - measured, -cfg.tracking_cap, cfg.tracking_cap)
        alpha = 1 - np.exp(-dt / 0.4)
        previous_z = self.feedback[:, 2].copy()
        previous = self.feedback.copy()
        self.feedback += alpha * (observed_error - self.feedback)
        if hold_across is not None:
            mask, axis = np.asarray(hold_across[0], dtype=bool), np.asarray(hold_across[1], dtype=float)
            along = (self.feedback - previous) @ axis
            if cfg.lateral_hold_common_mode:
                along = along - np.mean(along)  # the held hand loses only its share of the pair's spread change
            self.feedback[mask] -= along[mask, None] * axis
            self.diagnostics["across_feedback"] = self.feedback @ axis
        if z_floor is not None:
            under = np.asarray(measured)[:, 2] <= np.asarray(z_floor)
            backed = np.maximum(self.feedback[:, 2], previous_z) + alpha * (
                np.asarray(z_floor) - np.asarray(measured)[:, 2]
            )
            self.feedback[:, 2] = np.where(under, np.minimum(backed, cfg.tracking_cap), self.feedback[:, 2])
        if cfg.orientation_gain > 0 and measured_rotations is not None:
            # The palm-attitude counterpart of the position feedback: the rotation from what the controller
            # delivers to what the solver predicted, filtered, capped, and pre-applied to the command.
            predicted_rotations = self.solver.ik.data.xmat[self.body.hands].reshape(2, 3, 3)  # tools() put self.x
            error = np.array(
                [Rotation.from_matrix(p @ m.T).as_rotvec() for p, m in zip(predicted_rotations, measured_rotations)]
            )
            cap = np.radians(cfg.orientation_cap_deg)
            error *= np.minimum(1.0, cap / np.maximum(np.linalg.norm(error, axis=1, keepdims=True), 1e-9))
            self.rotation_feedback += (1 - np.exp(-dt / 0.4)) * (error - self.rotation_feedback)
            attitudes = np.array(
                [
                    Rotation.from_rotvec(cfg.orientation_gain * f).as_matrix() @ a
                    for f, a in zip(self.rotation_feedback, attitudes)
                ]
            )
        speed = np.r_[[self.config.carry.height_speed] * 3, cfg.palm_rotation_speed, [cfg.joint_speed] * 29]
        acceleration = np.r_[[0.2] * 3, cfg.joint_acceleration, [cfg.joint_acceleration] * 29]
        interval = dt if self.last_dt is None else (dt + self.last_dt) / 2
        self.last_dt = dt
        vlo = np.maximum(-speed, self.v - acceleration * interval)
        vhi = np.minimum(speed, self.v + acceleration * interval)
        braking_acceleration = acceleration / 2  # reserve braking margin for the next irregular interval
        vlo = np.maximum(
            vlo,
            -np.maximum(
                0,
                np.sqrt(2 * braking_acceleration * np.maximum(self.x - self.travel_lo, 0))
                - braking_acceleration * max(dt, 0.1),
            ),
        )
        vhi = np.minimum(
            vhi,
            np.maximum(
                0,
                np.sqrt(2 * braking_acceleration * np.maximum(self.travel_hi - self.x, 0))
                - braking_acceleration * max(dt, 0.1),
            ),
        )
        lo = np.maximum(self.travel_lo, self.x + vlo * dt)
        hi = np.minimum(self.travel_hi, self.x + vhi * dt)
        # Brake before physical limits. Tiny open bounds also cover an initially saturated axis.
        lo = np.minimum(lo, self.travel_hi - 1e-10)
        hi = np.maximum(hi, self.travel_lo + 1e-10)
        lo = np.minimum(lo, hi - 1e-10)
        anchor = self.config.carry.base_anchor
        if anchor != "reference" and self.measured_base is not None:
            # The streamed pose carries joints and root attitude only; the controller
            # places the pelvis itself. Solving from where it actually is keeps hand
            # corrections in joints the controller tracks instead of a virtual base shift.
            # measured_xy leaves height free so a lower pelvis_height_target can still
            # bend the streamed leg reference and be followed.
            n = 2 if anchor == "measured_xy" else 3
            if self.anchor_release <= 0:
                lo[:n], hi[:n] = self.measured_base[:n] - 0.005, self.measured_base[:n] + 0.005
            else:
                # The controller crouches by sitting back (~20 cm); pinned there, the solver can only
                # bend the knees further when asked to rise. Widen the window as the stand progresses
                # so the pelvis can come back over the feet inside the travel bounds.
                w = 0.005 + self.anchor_release * 0.5
                lo[:n] = np.maximum(lo[:n], self.measured_base[:n] - w)
                hi[:n] = np.minimum(hi[:n], self.measured_base[:n] + w)
                lo[:n] = np.minimum(lo[:n], hi[:n] - 1e-10)
        pose = self.body.solve(
            targets + cfg.tracking_gain * self.feedback,
            attitudes,
            seed=self.x + np.clip(self.v, vlo, vhi) * dt,
            bounds=(lo, hi),
            posture=self.posture if posture is None else posture,
            # TODO 2.4.10.20: a heavier pull of the 14 arm joints toward the admission solve's pose pins the elbow /
            # wrist-pitch null space the reach otherwise drifts along (the solver's own weight is 0.04 on all 29)
            posture_weights=(
                None
                if self.posture is None or cfg.reach_arm_posture_weight <= 0
                else np.r_[np.full(15, 0.04), np.full(14, cfg.reach_arm_posture_weight)]
            ),
            torso_rotation=self.torso_target,
            iterations=6,
            # The staged reach to the pregrasp point may trade palm attitude for position (TODO 2.4.10.13: the
            # wrist cameras only need the box in descend); align and descend keep the full weight.
            hand_rotation_weight=(
                cfg.reach_rotation_weight_transit
                if self.phase == "reach" and cfg.reach_rotation_weight_transit >= 0
                else cfg.reach_rotation_weight
            ),
            continuity_reference=self.x,
            continuity_dt=max(dt, 1e-3),
            continuity_weight=cfg.reach_continuity_weight,
        )
        next_x = self.body.seed.copy()
        self.v = (next_x - self.x) / dt
        self.x, self.q, self.root = next_x, pose.joints, quaternion(pose.rotation)
        self.velocity = self.v[4:].copy()
        self.diagnostics.update(
            reference_position=pose.position,
            foot_error=pose.foot_errors,
            com_shift=pose.com_shift,
            tracking_correction=self.feedback,
            orientation_correction_deg=np.degrees(np.linalg.norm(self.rotation_feedback, axis=1)),
            reference_hand_error=pose.hand_errors,
        )

    def pinned_bounds(self):
        """The solver's travel bounds with the base held at the measured pelvis, as `advance` holds it while the
        anchor is not released (carry.base_anchor measured / measured_xy)."""
        lo, hi = self.travel_lo.copy(), self.travel_hi.copy()
        anchor = self.config.carry.base_anchor
        if anchor != "reference" and self.measured_base is not None:
            n = 2 if anchor == "measured_xy" else 3
            lo[:n], hi[:n] = self.measured_base[:n] - 0.005, self.measured_base[:n] + 0.005
        return lo, hi

    def plan_route(self, goal_tools, final_torso, state):
        """visual.reach_route planned: this reach's own whole-body solve at the pregrasp points (base held, the
        admission posture as the prior, the final torso and palm attitudes) is the goal; `route_planner` returns a
        collision-free joint path from the reference to it, which is retimed (arm_route.retime_path, peak
        visual.reach_route_speed) and turned into hand poses on the start base. Returns a failure reason or ''."""
        from locomani.primitives.arm_route import retime_path

        cfg = self.config.visual
        if self.route_planner is None:
            raise RuntimeError("visual.reach_route planned needs CoordinatedReach.route_planner (the runner sets it)")
        goal = self.body.solve(
            goal_tools,
            self.attitudes,
            seed=self.x.copy(),
            bounds=self.pinned_bounds(),
            posture=self.posture,
            torso_rotation=final_torso,
            hand_rotation_weight=cfg.reach_rotation_weight,
        )
        self.diagnostics.update(route_goal_hand_error=goal.hand_errors)
        if max(goal.hand_errors) > 0.02:
            return "route_goal_unreachable"
        path = self.route_planner(self.q.copy(), goal.joints.copy(), state)
        if path is None or len(path) < 2:
            return "route_not_found"
        path, fraction, total = retime_path(np.asarray(path, dtype=float), cfg.reach_route_speed)
        base = self.x[:4].copy()
        positions, rotations = [], []
        for q in path:
            positions.append(self.body.tools(np.r_[base, q]))
            rotations.append(self.solver.ik.data.xmat[self.body.hands].reshape(2, 3, 3).copy())
        self.body.put(self.x)
        self.route = dict(
            joints=path, fraction=fraction, total=total, positions=np.array(positions), rotations=np.array(rotations)
        )
        self.duration = total
        self.diagnostics.update(route_waypoints=len(path), route_seconds=total)
        return ""

    def route_sample(self, t):
        """(hand positions, palm attitudes, joints, path fraction) `t` seconds into the planned route: one smoothstep
        over the path's joint arc, positions and joints linear and attitudes geodesic between waypoints."""
        r = self.route
        u = min(1.0, max(0.0, t / r["total"]))
        s = 3 * u * u - 2 * u**3
        k = int(np.clip(np.searchsorted(r["fraction"], s, side="right") - 1, 0, len(r["joints"]) - 2))
        w = min(1.0, max(0.0, (s - r["fraction"][k]) / max(float(r["fraction"][k + 1] - r["fraction"][k]), 1e-9)))
        positions = r["positions"][k] + w * (r["positions"][k + 1] - r["positions"][k])
        joints = r["joints"][k] + w * (r["joints"][k + 1] - r["joints"][k])
        attitudes = np.array(
            [
                a @ Rotation.from_rotvec(w * Rotation.from_matrix(a.T @ b).as_rotvec()).as_matrix()
                for a, b in zip(r["rotations"][k], r["rotations"][k + 1])
            ]
        )
        return positions, attitudes, joints, s

    def pregrasp(self, targets):
        """Staged reach waypoint: pregrasp_above over each grasp point, pregrasp_back toward the body."""
        cfg = self.config.visual
        return targets - cfg.pregrasp_back * self.outward + np.array([0.0, 0.0, cfg.pregrasp_above])

    def solver_frame_base(self, state):
        """Measured pelvis position expressed in the solver frame (feet coincide with the solver's feet)."""
        quat = np.asarray(state["pelvis"]["quat"], dtype=float)
        pitch = Rotation.from_matrix(rotation(quat)).as_euler("xyz")[1]
        self.body.put(np.r_[state["pelvis"]["pos"], pitch, np.asarray(state["joints"], dtype=float)])
        correction = np.mean(self.body.feet_pos - self.solver.ik.data.xpos[self.body.feet], axis=0)
        return np.asarray(state["pelvis"]["pos"], dtype=float) + correction

    def brake(self, dt):
        """A lost visual target stops progress and brakes the last physical reference."""
        if dt <= 0:
            return
        acceleration = np.r_[
            [0.2] * 3, self.config.visual.joint_acceleration, [self.config.visual.joint_acceleration] * 29
        ]
        interval = dt if self.last_dt is None else (dt + self.last_dt) / 2
        velocity = self.v + np.clip(-self.v, -acceleration * interval, acceleration * interval)
        x = np.clip(self.x + velocity * dt, self.travel_lo, self.travel_hi)
        self.v = (x - self.x) / dt
        self.x, self.last_dt = x, dt
        self.q, self.velocity = x[4:].copy(), self.v[4:].copy()
        self.root = quaternion(self.body.put(x))

    def step(self, wall_time, estimate, state, hand_openings, *, motion_time=None):
        """Advance by physical time; camera timestamps and freshness use wall time."""
        cfg = self.config.visual
        now = wall_time if motion_time is None else float(motion_time)
        self._seq = state.get("seq")
        self.contact_sides = tuple((state.get("objects") or {}).get(self.target.object_key, {}).get("touching") or ())
        if self.started is None:
            self.started = self.last_time = now
            self.started_wall = wall_time
        control_dt = max(float(now - self.last_time), 0.0)
        dt = min(control_dt, 0.1)  # cap trajectory progress, not the physical rate interval
        self.last_time = now
        if now - self.started > cfg.timeout_seconds:
            return self.result("timeout", True)
        if estimate is None or not 0 <= wall_time - estimate.timestamp <= cfg.vision_timeout:
            self.was_visible, self.ready_since = False, None
            self.brake(control_dt)
            stale = (
                wall_time - (self.last_seen if self.last_seen is not None else self.started_wall) > cfg.vision_timeout
            )
            return self.result("vision_lost" if stale else "waiting_for_vision", stale)
        dt = dt if self.was_visible else 0.0
        self.measured_base = self.solver_frame_base(state)
        self.was_visible, self.last_seen = True, estimate.timestamp
        self.elapsed += dt
        base, origin = rotation(state["pelvis"]["quat"]), np.asarray(state["pelvis"]["pos"])
        tools, tool_rotations = self.solver.tool_poses(np.asarray(state["joints"]))
        tools, tool_rotations = tools @ base.T + origin, base @ tool_rotations
        local_centres = estimate.transform(self.target.grasp_points_local)
        if not self.target.preserve_point_order:
            local_centres = local_centres[np.argsort(-local_centres[:, 1])]
        centres = local_centres @ base.T + origin
        obj_rotation, obj_position = (
            base @ estimate.rotation @ self.target.capture_rotation_local,
            origin + base @ estimate.position,
        )
        # A reachability probe owns exact world capture centres.  Ordinary
        # crate execution still permits the historical slide along the rim.
        targets = centres if self.target.reach_only else self.region.nearest(centres, obj_rotation[:, 1], tools)
        # The cached camera-admitted pose owns reach targets. Current simulation
        # state independently checks displacement, attachment and load safety.
        if self.target.object_key in state.get("objects", {}):
            observed_object = state["objects"][self.target.object_key]
            obj_position = np.asarray(observed_object["pos"])
            obj_rotation = rotation(observed_object["quat"]) @ self.target.capture_rotation_local
        self.diagnostics = dict(
            observed_grips=centres,
            measured_tools=tools,
            information_source=self.target.information_source,
            **({"retry": self.last_retry} if self.last_retry else {}),
        )
        if self.phase == "observe":
            axis = centres[0] - centres[1]
            yaw = np.arctan2(-axis[0], axis[1])
            attitude = (
                Rotation.from_euler("z", yaw).as_matrix()
                @ Rotation.from_euler("y", cfg.palm_pitch_deg, degrees=True).as_matrix()
            )
            self.attitudes = np.array([attitude] * 2)
            admission = self.body.solve(targets, self.attitudes)
            self.diagnostics.update(
                admission_hand_error=admission.hand_errors,
                admission_foot_error=admission.foot_errors,
                admission_com_shift=admission.com_shift,
            )
            if not admission.reachable:
                return self.result("unsupported_assisted_reach", True)
            self.posture = admission.joints.copy()
            final_torso = self.solver.ik.data.xmat[self.body.torso].reshape(3, 3).copy()
            self.body.put(self.x)
            initial_torso = self.solver.ik.data.xmat[self.body.torso].reshape(3, 3).copy()
            self.reach_rotation = Slerp([0, 1], Rotation.from_matrix([initial_torso, final_torso]))
            self.start_tools = self.body.tools(self.x)
            self.body.put(self.x)
            initial_attitudes = self.solver.ik.data.xmat[self.body.hands].reshape(2, 3, 3)
            self.rotations = [Slerp([0, 1], Rotation.from_matrix([a, attitude])) for a in initial_attitudes]
            # Horizontal unit vector from the body toward the object, across the grasp line.
            self.outward = Rotation.from_euler("z", yaw).apply([1.0, 0.0, 0.0])
            # Feed-forward of the controller's steady delivery gap (delivered hands short and high of the commanded
            # reference by a posture-dependent 6-8 cm, 2026-09-24): commanded outside the capped feedback, so the cap
            # is left for the residual. Forward along the body-to-object axis, up in the world.
            self.feedforward = cfg.reach_feedforward_forward * self.outward + np.array(
                [0.0, 0.0, cfg.reach_feedforward_up]
            )
            first = self.pregrasp(targets) if cfg.reach_path == "staged" else targets
            distance = np.max(np.linalg.norm(first - self.start_tools, axis=1))
            self.duration = max(3.0, 1.875 * distance / cfg.approach_speed)
            self.reach_object = obj_position.copy()
            if cfg.reach_route == "planned":
                failure = self.plan_route(first + self.feedforward, final_torso, state)
                if failure:
                    return self.result(failure, True)
            self.enter("reach")
        self.diagnostics.update(
            reference_elapsed=self.elapsed, planned_reach_seconds=self.duration, control_interval=control_dt
        )
        if self.phase == "reach":
            if np.linalg.norm(obj_position - self.reach_object) > self.region.half_length:
                return self.result("basket_displaced_before_capture", True)
            error = float(np.max(np.linalg.norm(tools - targets, axis=1)))
            if self.progress_time is None or error < self.best_error - 0.005:
                self.best_error, self.progress_time = error, now
            if not self.target.reach_only and self.elapsed > self.duration and now - self.progress_time > 3.0:
                return self.result("reach_stalled", True)
            if self.target.reach_only:
                self.diagnostics.update(capture_error=np.linalg.norm(tools - targets, axis=1), capture_ready=False)
            else:
                ready, errors = self.capture_ready(centres, obj_rotation, tools, tool_rotations)
                self.diagnostics.update(capture_error=errors, capture_ready=ready)
                self.ready_since = (now if self.ready_since is None else self.ready_since) if ready else None
                if self.ready_since is not None and now - self.ready_since >= 0.1:
                    self.closed_targets = self.body.tools(self.x)
                    self.enter("close")
            if self.phase == "reach":
                first = self.pregrasp(targets) if cfg.reach_path == "staged" else targets
                if self.route is not None:
                    # the planned path's hand poses (feed-forward already in its goal), its joints as the posture prior
                    commanded, attitudes, posture, u = self.route_sample(self.elapsed)
                    self.torso_target = self.reach_rotation(u).as_matrix()
                    self.advance(commanded, attitudes, tools, control_dt, tool_rotations, posture=posture)
                else:
                    u, _ = blend(self.elapsed, self.duration)
                    self.torso_target = self.reach_rotation(u).as_matrix()
                    commanded = self.start_tools + u * (first - self.start_tools)
                    attitudes = np.array([r(u).as_matrix() for r in self.rotations])
                    self.advance(commanded + self.feedforward, attitudes, tools, control_dt, tool_rotations)
                if cfg.reach_path == "staged" and self.elapsed >= self.duration:
                    if cfg.descend_align_tolerance > 0:
                        self.align_started = now
                        self.enter("align")
                    else:
                        self.start_descend(targets, first)
                    return self.result()  # this tick's reference is already sent; continue from the next one
        if self.phase == "reopen":
            # A reseat retry: let go where the hands are (the fingers open over close_seconds while the targets
            # hold), then back up to the pregrasp point and re-centre.
            self.closure = max(0.0, 1.0 - self.elapsed / cfg.close_seconds)
            self.advance(
                self.reopen_targets - cfg.tracking_gain * self.feedback,
                self.attitudes,
                tools,
                control_dt,
                tool_rotations,
            )
            if self.elapsed >= cfg.close_seconds + 0.2:
                self.align_started = now
                self.enter("align")
            return self.result()
        if self.phase == "align":
            # Hold at the pregrasp point, above the rim where nothing blocks a lateral correction, until both
            # measured grip points are centred across the wall (assisted: the estimate is the simulation's own
            # pose). Descending 2-6 cm off-centre jammed a pad on the wall face (2026-09-24).
            first = self.pregrasp(targets)
            across = ((tools - targets) @ obj_rotation)[:, 0]
            self.diagnostics.update(align_across=across, align_seconds=now - self.align_started)
            if (
                np.max(np.abs(across)) <= cfg.descend_align_tolerance
                or now - self.align_started >= cfg.descend_align_timeout
            ):
                self.align_residual = across.copy()
                self.start_descend(targets, first)
                return self.result()
            first, attitudes = self.guarded(first, self.attitudes, state, control_dt)
            self.advance(first + self.feedforward, attitudes, tools, control_dt, tool_rotations)
        if self.phase == "descend":
            # Same capture and stall rules as the direct reach, including reach_only probes that never close.
            if self.target.reach_only:
                ready, errors = False, np.linalg.norm(tools - targets, axis=1)
            else:
                ready, errors = self.capture_ready(centres, obj_rotation, tools, tool_rotations)
            self.diagnostics.update(capture_error=errors, capture_ready=ready)
            goal = self.descend_goal(targets)
            error = float(np.max(np.linalg.norm(tools - goal, axis=1)))
            if self.progress_time is None or error < self.best_error - 0.005:
                self.best_error, self.progress_time = error, now
            if not self.target.reach_only and self.elapsed > self.descend_duration and now - self.progress_time > 3.0:
                return self.retry("descend", now, "reach_stalled") or self.result("reach_stalled", True)
            self.ready_since = (now if self.ready_since is None else self.ready_since) if ready else None
            if self.ready_since is not None and now - self.ready_since >= 0.1:
                self.closed_targets = self.body.tools(self.x)
                self.enter("close")
            else:
                u, _ = blend(self.elapsed, self.descend_duration)
                commanded = self.descend_from + u * (goal - self.descend_from)
                commanded, attitudes = self.guarded(commanded, self.attitudes, state, control_dt)
                hold = None
                if cfg.lateral_hold_below_rim:
                    below = tools[:, 2] <= targets[:, 2] + self.target.rim_above_grasp + 0.01
                    hold = (below, obj_rotation[:, 0])
                    across = ((tools - targets) @ obj_rotation)[:, 0]
                    self.diagnostics.update(lateral_hold=below, across_common_mode=float(np.mean(across)))
                self.advance(
                    commanded + self.feedforward,
                    attitudes,
                    tools,
                    control_dt,
                    tool_rotations,
                    z_floor=targets[:, 2] if cfg.descend_z_guard else None,
                    hold_across=hold,
                )
        if self.phase == "close":
            # Maintain the paired world reference while smoothly braking any residual body motion.
            self.advance(
                self.closed_targets - cfg.tracking_gain * self.feedback,
                self.attitudes,
                tools,
                control_dt,
                tool_rotations,
            )
            self.closure = min(1.0, self.elapsed / cfg.close_seconds)
            if self.elapsed >= cfg.close_seconds + 0.2:
                if min(hand_openings) < 0.003:
                    return self.retry("reseat", now, "empty_gripper") or self.result("empty_gripper", True)
                if cfg.seat_depth > 0:
                    # Seated evenly and deep enough to survive the stand? Otherwise spend a reseat, never the round.
                    error = (tools - centres) @ obj_rotation  # per hand: across, along, up
                    depth = self.target.rim_above_grasp - error[:, 2]
                    skew = np.abs(error[0] - error[1])
                    seated = bool(np.all(depth >= cfg.seat_depth) and np.all(skew[1:] <= cfg.seat_skew))
                    self.diagnostics["seat"] = dict(
                        depth=depth.tolist(), skew_along=float(skew[1]), skew_up=float(skew[2]), ok=seated
                    )
                    if not seated and (again := self.retry("reseat", now, "seat_rejected")) is not None:
                        return again
                self.closed_tools = tools.copy()
                self.closed_relative = (tools - obj_position) @ obj_rotation
                self.closed_object = obj_position.copy()
                self.closed_rotation = obj_rotation.copy()
                self.pitch_follow = 0.0
                # The held FK already includes feedback; remove it before adding
                # the same correction to the new trajectory, preserving its start.
                self.lift_targets = self.body.tools(self.x) - cfg.tracking_gain * self.feedback
                self.stand_targets = self.lift_targets + self.start_tools.mean(axis=0) - self.lift_targets.mean(axis=0)
                self.stand_targets[:, 2] = np.maximum(
                    self.stand_targets[:, 2], self.lift_targets[:, 2] + cfg.lift_height
                )
                if cfg.stand_forward_min > 0:
                    # Keep the held load clear of the torso through the stand: the ready-pose position the
                    # targets return to sits inside a deep box's footprint (2026-09-24, warehouse).
                    pelvis_xy = np.asarray(state["pelvis"]["pos"][:2], dtype=float)
                    forward = float((self.stand_targets.mean(axis=0)[:2] - pelvis_xy) @ self.outward[:2])
                    if forward < cfg.stand_forward_min:
                        self.stand_targets = self.stand_targets + (cfg.stand_forward_min - forward) * self.outward
                    self.diagnostics["stand_forward_m"] = max(forward, cfg.stand_forward_min)
                self.body.put(self.x)
                planned_torso = self.solver.ik.data.xmat[self.body.torso].reshape(3, 3).copy()
                measured_torso = base @ self.solver.ik.body_pose(np.asarray(state["joints"]), "torso_link")[1]
                # Do not ask a lagging torso to bend farther at lift-off.
                initial_torso = min((planned_torso, measured_torso), key=lambda r: np.arccos(np.clip(r[2, 2], -1, 1)))
                self.torso_target = initial_torso.copy()
                self.stand_rotation = Slerp([0, 1], Rotation.from_matrix([initial_torso, self.body.heading]))
                self.stand_posture = self.posture.copy()
                self.recovery_started = None
                self.enter("lift")
        elif self.phase in ("lift", "stand", "hold"):
            drift = tools - (self.closed_relative @ obj_rotation.T + obj_position)  # per hand, world
            attachment = float(np.max(np.linalg.norm(drift, axis=1)))
            rise = float(obj_position[2] - self.closed_object[2])
            if cfg.attachment_gain > 0 and self.recovery_started is not None and control_dt > 0:
                # The pinch creeps UP the wall under load; command the hand that rose back down by what it rose.
                # Not during the probe lift, whose delivered-height test would read the correction as a short lift.
                alpha = 1 - np.exp(-control_dt / 0.3)
                self.attachment_feedback += alpha * (drift[:, 2] - self.attachment_feedback)
                self.attachment_feedback = np.clip(self.attachment_feedback, -cfg.attachment_cap, cfg.attachment_cap)
            self.diagnostics["attachment_feedback"] = self.attachment_feedback.copy()
            tilt = float(np.degrees(np.arccos(np.clip(obj_rotation[2, 2], -1, 1))))
            self.diagnostics.update(attachment_error=attachment, observed_rise=rise, observed_tilt=tilt)
            torso = base @ self.solver.ik.body_pose(np.asarray(state["joints"]), "torso_link")[1]
            lean = float(np.degrees(np.arccos(np.clip(torso[2, 2], -1, 1))))
            self.diagnostics["observed_torso_lean_deg"] = lean
            if self.previous_load is not None and control_dt > 0:
                previous_position, previous_torso = self.previous_load
                basket_speed = float(np.linalg.norm(obj_position - previous_position) / control_dt)
                torso_speed = float(Rotation.from_matrix(torso @ previous_torso.T).magnitude() / control_dt)
                self.diagnostics.update(observed_basket_speed=basket_speed, observed_torso_speed=torso_speed)
            self.previous_load = obj_position.copy(), torso.copy()
            if min(hand_openings) < 0.003 or (self.phase != "lift" and attachment > 0.06):
                if self.phase == "lift" and (again := self.retry("reseat", now, "lost_attachment")) is not None:
                    return again
                return self.result("lost_attachment", True)
            if tilt > cfg.max_tilt_deg:
                return self.result("basket_tilt", True)
            attitudes = self.follow_attitudes(obj_rotation, control_dt)
            if self.phase in ("lift", "stand"):
                delivered_probe = np.mean(tools[:, 2] - self.closed_tools[:, 2]) >= cfg.probe_height
                if self.recovery_started is None:
                    if delivered_probe:
                        if rise < cfg.probe_height * 0.35 or attachment > 0.03:
                            return self.retry("reseat", now, "probe_not_attached") or self.result(
                                "probe_not_attached", True
                            )
                        # A phase label changes, not the motion clock or reference.
                        self.recovery_started = self.elapsed
                        self.phase = "stand"
                    elif self.elapsed >= cfg.lift_seconds + 2:
                        return self.retry("reseat", now, "probe_not_attached") or self.result(
                            "probe_not_attached", True
                        )
                lift_u, _ = blend(self.elapsed, cfg.lift_seconds)
                recovery_time = 0 if self.recovery_started is None else self.elapsed - self.recovery_started
                stand_u, _ = blend(recovery_time, cfg.stand_seconds)
                lifted = self.lift_targets + [0, 0, cfg.lift_height]
                targets = self.lift_targets + [0, 0, lift_u * cfg.lift_height]
                targets += stand_u * (self.stand_targets - lifted)
                self.torso_target = self.stand_rotation(stand_u).as_matrix()
                self.posture = self.stand_posture + stand_u * (self.body.q0 - self.stand_posture)
                # Only a crouched start needs the base freed; the upright flow keeps its measured anchor.
                self.anchor_release = stand_u if self.crouch_height is not None else 0.0
                if self.crouch_height is not None:
                    # Keep the crouch through the lift; release it with the posture during the stand,
                    # or the solver keeps pulling the pelvis down under hands that rise to standing height.
                    self.body.pelvis_height_target = self.crouch_height + stand_u * (
                        self.body.base0[2] - self.crouch_height
                    )
                self.diagnostics["pelvis_height_target"] = self.body.pelvis_height_target
                targets = self.forward_floor(targets, state)
                self.diagnostics["loaded_targets_raw"] = targets.copy()
                targets[:, 2] -= cfg.attachment_gain * self.attachment_feedback
                self.diagnostics.update(lift_progress=lift_u, recovery_progress=stand_u, loaded_targets=targets.copy())
                self.advance(targets, attitudes, tools, control_dt, tool_rotations)
                if (
                    self.recovery_started is not None
                    and recovery_time >= cfg.stand_seconds
                    and self.elapsed >= cfg.lift_seconds
                    and lean <= cfg.upright_tolerance_deg
                    and attachment < 0.03
                    and rise >= cfg.lift_height * 0.6
                ):
                    self.body.pelvis_height_target = None
                    self.anchor_release = 1.0
                    self.enter("hold")
                elif recovery_time >= cfg.stand_seconds + 4:
                    return self.result("stand_not_delivered", True)
            else:
                targets = self.forward_floor(self.stand_targets, state)
                self.diagnostics["loaded_targets_raw"] = targets.copy()
                targets[:, 2] -= cfg.attachment_gain * self.attachment_feedback
                self.diagnostics["loaded_targets"] = targets.copy()
                self.advance(targets, attitudes, tools, control_dt, tool_rotations)
                if self.elapsed >= cfg.hold_seconds + 2.0:
                    return self.result("unstable_hold", True)
                stable = (
                    attachment < 0.03
                    and rise >= 0.6 * cfg.lift_height
                    and min(hand_openings) >= 0.003
                    and lean <= cfg.upright_tolerance_deg
                )
                if not stable:
                    self.hold_window.clear()
                    self.ready_since = None
                else:
                    anchor = np.asarray(state["pelvis"]["pos"], dtype=float) if cfg.hold_frame == "pelvis" else 0.0
                    self.hold_window.append((now, obj_position - anchor))
                    while len(self.hold_window) > 1 and self.hold_window[1][0] <= now - cfg.hold_seconds:
                        self.hold_window.popleft()
                    positions = np.array([p for _, p in self.hold_window])
                    radius = float(np.max(np.linalg.norm(positions - positions.mean(axis=0), axis=1)))
                    self.diagnostics["hold_position_radius"] = radius
                    self.ready_since = self.hold_window[0][0] if radius <= cfg.hold_position_tolerance else None
                if self.ready_since is not None and now - self.ready_since >= cfg.hold_seconds:
                    return self.result("completed", True)
        return self.result()
