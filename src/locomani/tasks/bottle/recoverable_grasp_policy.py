"""Bounded measured stand correction around the opt-in ready grasp policy.

Prospective geometry is a screening approximation; measured guards still own
execution. Recovery never continues after intentional target contact begins.
"""

import time
from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.backends.sonic import gait_speeds
from locomani.backends.sonic.agent_server import LocomotionMode
from locomani.backends.sonic.primitives import Primitives, upper_body_from_mj, wrap
from locomani.tasks.bottle.ready_grasp_policy import OFFSET, ReadyGraspPolicy, ready_q, templates


def retracted_q():
    q = ready_q()
    for shoulder, elbow in ((15, 18), (22, 25)):
        q[shoulder], q[elbow] = 0.50, -0.50
    return q


class RecoverableGraspPolicy(ReadyGraspPolicy):
    def __init__(self, *args):
        super().__init__(*args)
        self.meta = self.task.meta
        if self.meta is None:
            raise ValueError("recovery requires the round's resolved scene metadata")
        self.retracted = False
        self.deadline = None
        self.standing_z = None
        self.arm.mode, self.arm.height = LocomotionMode.IDLE, -1.0
        self.execution.metrics["recovery"] = dict(
            attempts=[], max_attempts=self.config.recovery_attempts, budget_s=self.config.recovery_timeout
        )

    def walk_tick(self, state):
        return self._push(state, retracted_q() if self.retracted else ready_q())

    def expired(self):
        return self.deadline is not None and time.monotonic() >= self.deadline

    def hold(self):
        self.p.cmd.frame(
            mode=self.arm.mode,
            height=self.arm.height,
            speed=0.0,
            movement=(0.0, 0.0, 0.0),
            facing=self.p._facing(self.p.obs()),
        )

    def finish_cleanup(self, ok):
        if ok:
            super().finish_cleanup(ok)
        else:
            # Do not invoke the baseline's unconditional stand-up while an
            # extended arm could sweep through the object or tabletop.
            self.p.cmd.set_upper_body_vel(np.zeros(17))
            self.hold()

    def reference_check(self, state, q, phase):
        error = super().reference_check(state, q, phase)
        if error and "clearance_failure" not in self.execution.metrics:
            # Exact guard inputs, before cleanup changes the posture.
            self.execution.metrics["clearance_failure"] = dict(
                stage=self.execution.stage,
                phase=phase,
                reason=error,
                joints=list(map(float, state["joints"])),
                reference=list(map(float, q)),
                pelvis={k: list(map(float, state["pelvis"][k])) for k in ("pos", "quat")},
                objects={
                    self.task.target.key: {
                        k: list(map(float, state["objects"][self.task.target.key][k])) for k in ("pos", "quat")
                    }
                },
                palms={sd: dict(hand_q=list(map(float, state["palms"][sd]["hand_q"]))) for sd in ("left", "right")},
            )
        return error or ("recovery time budget exhausted" if self.expired() else None)

    def floor_distances(self, xy):
        distances = []
        for surface in self.meta["surfaces"].values():
            half = np.asarray(surface["half_extent"][:2])
            delta = np.abs(np.asarray(xy) - surface["xy"]) - half
            distances.append(float(np.linalg.norm(np.maximum(delta, 0)) if (delta > 0).any() else max(delta)))
        return np.asarray(distances)

    def floor_clear(self, xy, margin=0.17):
        return bool(np.all(self.floor_distances(xy) >= margin))

    def retreat_clear(self, start, end):
        """Below the navigation margin, permit only outward translation."""
        before, after = self.floor_distances(start), self.floor_distances(end)
        return bool(
            np.all(before >= 0.06)
            and np.all(after >= 0.06)
            and np.all((after >= 0.17) | ((before < 0.17) & (after > before + 1e-6)))
        )

    def pose_clear(self, state, q):
        return all(self.guard.check(state, q, sd).clear for sd in ("left", "right", "body"))

    def reject_prediction(self, reason):
        counts = self.execution.metrics.setdefault("prediction_rejections", {})
        counts[reason] = counts.get(reason, 0) + 1
        return False

    def assess(self, state, drop):
        """Screen a squat and a path into the shared delivered capture region."""
        if not 0.03 <= drop <= self.config.max_drop:
            return self.reject_prediction("squat_out_of_bounds")
        if not self.floor_clear(state["pelvis"]["pos"][:2], margin=0.06):
            return self.reject_prediction("base_floor_clearance")
        q = ready_q()
        q[:12] = np.asarray(state["joints"])[:12]  # retain measured leg geometry
        base = np.asarray(state["pelvis"]["pos"])
        yaw = self.p.pelvis_yaw(state)
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        projected = deepcopy(state)
        for backshift in (0.0, self.config.backshift):
            for u in np.linspace(0, 1, 9):
                projected["pelvis"]["pos"] = (base - u * (forward * backshift + [0.0, 0.0, drop])).tolist()
                projected["joints"] = q
                for sd in ("left", "right", "body"):
                    hit = self.guard.check(projected, q, sd)
                    if not hit.clear:
                        return self.reject_prediction(f"squat_clearance:{hit.link}:{hit.obstacle}")
        projected["pelvis"]["pos"] = (base - forward * self.config.backshift - [0.0, 0.0, drop]).tolist()
        from locomani.tasks.bottle.grasp_capture import planning_points

        preferred = self.task.grip(state)
        for point in planning_points(preferred):
            if self.expired():
                return self.reject_prediction("budget")
            if self.assess_reach(deepcopy(projected), q.copy(), point, preferred):
                self.execution.metrics["prediction_capture_point"] = point.tolist()
                return True
        return False

    def assess_reach(self, projected, q, point, preferred):
        from locomani.tasks.bottle.grasp_capture import closure_geometry

        rel = self.p.world_to_root_full(point, projected)
        _, _, local_rot = min(templates(), key=lambda row: np.linalg.norm(row[1] - rel))
        world_rot = Primitives._rot(projected["pelvis"]["quat"])
        rot = world_rot @ local_rot
        wrist = point - rot @ OFFSET
        final = self.p.world_to_root_full(wrist, projected)
        terminal, pe, re = self.arm.ik.solve(self.side, final, local_rot, q_mj29=q, use_waist=False, iters=60)
        if pe > 0.015 or re > 0.15:
            return self.reject_prediction("terminal_ik")
        palm, _ = self.arm.ik.fk(q, self.side)
        if np.linalg.norm(final - palm) > 0.20:
            return self.reject_prediction("local_path_too_long")
        axis = local_rot[:, 0]
        depth = float((final - palm) @ axis)
        aligned = final - max(depth, 0.0) * axis
        begin = palm.copy()
        for end in (aligned, final):
            for u in np.linspace(0.25, 1.0, 4):
                target = begin + u * (end - begin)
                candidate, pe, re = self.arm.ik.solve(
                    self.side,
                    target,
                    local_rot,
                    q_mj29=q,
                    use_waist=False,
                    iters=80 if np.linalg.norm(target - final) < 1e-6 else 40,
                )
                if pe > 0.02 or re > 0.15:
                    return self.reject_prediction("path_ik")
                for sd in ("left", "right", "body"):
                    hit = self.guard.check(
                        projected,
                        candidate,
                        sd,
                        allow_target_contact=(sd == self.side and np.linalg.norm(target - final) < 0.08),
                    )
                    if not hit.clear:
                        return self.reject_prediction(f"path_clearance:{hit.link}:{hit.obstacle}")
                projected["joints"] = candidate
                q = candidate
            begin = end
        # IK convergence is not itself proof that the candidate captures.
        palm, rot = self.arm.ik.fk(q, self.side)
        trial = deepcopy(projected)
        trial["palms"][self.side]["pos"] = (np.asarray(projected["pelvis"]["pos"]) + world_rot @ palm).tolist()
        trial["palms"][self.side]["quat"] = Rotation.from_matrix(world_rot @ rot).as_quat()[[3, 0, 1, 2]].tolist()
        if not closure_geometry(trial, self.side, preferred)["ready"]:
            return self.reject_prediction("terminal_capture_region")
        return True

    def select_drop(self, state, hint):
        live = self.task.grip(state)
        z = float(state["pelvis"]["pos"][2])
        world_rot = Primitives._rot(state["pelvis"]["quat"])
        raw = [float(hint)] + [float(z - (live[2] - (world_rot @ centre)[2])) for _, centre, _ in templates()]
        # A template asking for a deeper squat may still be reachable with
        # the arm at the allowed squat limit; test that candidate as well.
        drops = list(dict.fromkeys(float(np.clip(d, 0.03, self.config.max_drop)) for d in raw))
        drops += [d for d in np.linspace(0.03, self.config.max_drop, 6) if d not in drops]
        for drop in drops:
            if self.expired():
                break
            if self.assess(state, drop):
                return drop
        return None

    def brake(self):
        self.execution.enter("recovery_brake")
        end = time.monotonic() + 3.0
        stable = None
        while time.monotonic() < end and not self.expired():
            state = self.p.obs()
            if state["fallen"] or not self.walk_tick(state):
                return False
            self.hold()
            speed = np.linalg.norm(state["pelvis"]["vel"][:2])
            if speed < 0.06:
                stable = stable or time.monotonic()
                self.prepare_brake(state)
                # Planning may take longer than a control tick. Never start
                # descent from the velocity sampled before that work.
                latest = self.p.obs()
                if latest["fallen"] or self.expired():
                    return False
                if np.linalg.norm(latest["pelvis"]["vel"][:2]) >= 0.06:
                    stable = None
                    continue
                if time.monotonic() - stable >= 0.25:
                    return True
            else:
                stable = None
            time.sleep(0.02)
        return False

    def prepare_brake(self, state):
        """Optional planning work while the base finishes decelerating."""

    def choose_stand(self, state):
        """Find a nearby feasible region; minimize correction rather than snap to one pose."""
        live = self.task.grip(state)
        base = np.asarray(state["pelvis"]["pos"])
        yaw = self.p.pelvis_yaw(state)
        options = []
        for turn in (0.0, -0.15, 0.15):
            angle = yaw + turn
            world_rot = Rotation.from_euler("z", turn).as_matrix() @ Primitives._rot(state["pelvis"]["quat"])
            quat = Rotation.from_matrix(world_rot).as_quat()[[3, 0, 1, 2]]
            forward = np.array([np.cos(angle), np.sin(angle)])
            for _, centre, _ in templates():
                world_centre = world_rot @ centre
                drop = float(np.clip(base[2] - (live[2] - world_centre[2]), 0.03, self.config.max_drop))
                for slack in (0.0, 0.03, 0.06):
                    xy = live[:2] - world_centre[:2] + (self.config.backshift - slack) * forward
                    distance = np.linalg.norm(xy - base[:2])
                    meaningful = distance >= 0.035 or abs(turn) >= 0.12
                    if distance <= 0.22 and meaningful and self.floor_clear(xy):
                        options.append((distance + 0.10 * abs(turn), xy, angle, drop, quat))
        for _, xy, angle, drop, quat in sorted(options, key=lambda x: x[0]):
            if self.expired():
                break
            trial = deepcopy(state)
            trial["pelvis"]["pos"] = [*map(float, xy), float(base[2])]
            trial["pelvis"]["quat"] = quat.tolist()
            trial["pelvis"]["yaw"] = float(angle)
            if self.assess(trial, drop):
                return xy, float(angle), float(drop)
        return None

    def withdraw_and_stand(self):
        """Retract while stationary, then rise under clearance checks, then permit walking."""
        self.execution.enter("recovery_retract")
        target = upper_body_from_mj(retracted_q())
        end = time.monotonic() + 4.0
        while time.monotonic() < end and not self.expired():
            state = self.p.obs()
            if state["fallen"] or not self._push(state, retracted_q()):
                return False
            self.hold()
            if np.max(np.abs(upper_body_from_mj(state["joints"])[3:] - target[3:])) < 0.15:
                break
            time.sleep(0.02)
        else:
            return False
        self.retracted = True
        self.execution.enter("recovery_stand")
        end = time.monotonic() + 4.0
        from locomani.tasks.bottle.execution import SQUAT_HEIGHT_BIAS

        while time.monotonic() < end and not self.expired():
            state = self.p.obs()
            z = float(state["pelvis"]["pos"][2])
            if state["fallen"]:
                return False
            next_z = min(self.standing_z, z + 0.015)
            projected = deepcopy(state)
            projected["pelvis"]["pos"][2] = next_z
            if not self.pose_clear(projected, retracted_q()) or not self.walk_tick(state):
                return False
            if z >= self.standing_z - 0.02:
                self.arm.mode, self.arm.height = LocomotionMode.IDLE, -1.0
                self.hold()
                return True
            self.arm.mode = LocomotionMode.IDEL_SQUAT
            self.arm.height = next_z - SQUAT_HEIGHT_BIAS
            self.hold()
            time.sleep(0.02)
        return False

    def move_to_region(self, goal):
        xy, yaw, _ = goal
        # The recovery walk's two speeds come from the same measured
        # per-checkpoint map as the approach's (gait_speeds.py, ledger 21): a
        # command carried across checkpoints changes speed by up to 13x. This
        # path is dormant in the collection profiles (recovery_attempts: 0) --
        # which is exactly why it would otherwise keep release-era numbers
        # unnoticed until someone turned recovery back on.
        _, _MID, _SHUFFLE = gait_speeds.approach_bands()
        self.execution.enter("recovery_reposition", xy=list(map(float, xy)), yaw=yaw)
        initial = self.p.obs()
        motion = dict(
            commanded_translation_ticks=0,
            commanded_turn_ticks=0,
            start_xy=list(map(float, initial["pelvis"]["pos"][:2])),
        )
        self.execution.metrics["recovery"].setdefault("motions", []).append(motion)
        end = time.monotonic() + 12.0
        stable = None
        try:
            while time.monotonic() < end and not self.expired():
                state = self.p.obs()
                base = np.asarray(state["pelvis"]["pos"][:2])
                delta = xy - base
                distance = np.linalg.norm(delta)
                if state["fallen"] or not self.floor_clear(base, margin=0.06) or not self.walk_tick(state):
                    return False
                dyaw = wrap(yaw - self.p.pelvis_yaw(state))
                if distance < 0.025 and abs(dyaw) < 0.12:
                    self.hold()
                    stable = stable or time.monotonic()
                    if time.monotonic() - stable >= 0.25:
                        return True
                else:
                    stable = None
                    movement = np.r_[delta / max(distance, 1e-6), 0.0] if distance >= 0.025 else np.zeros(3)
                    # Check the near-future reference before translating. The
                    # A close stop can retreat outward through the .17 m margin.
                    projected = deepcopy(state)
                    projected["pelvis"]["pos"] = (np.asarray(state["pelvis"]["pos"]) + movement * 0.03).tolist()
                    if not self.retreat_clear(base, projected["pelvis"]["pos"][:2]) or not self.pose_clear(
                        projected, retracted_q() if self.retracted else ready_q()
                    ):
                        return False
                    motion["commanded_translation_ticks"] += int(distance >= 0.025)
                    motion["commanded_turn_ticks"] += int(abs(dyaw) >= 0.12)
                    self.p.cmd.frame(
                        mode=LocomotionMode.SLOW_WALK,
                        speed=_SHUFFLE if distance < 0.12 else _MID,
                        movement=tuple(movement),
                        facing=(np.cos(yaw), np.sin(yaw), 0.0),
                    )
                time.sleep(0.02)
            return False
        finally:
            final = self.p.obs()
            motion["end_xy"] = list(map(float, final["pelvis"]["pos"][:2]))
            motion["displacement_m"] = float(np.linalg.norm(np.asarray(motion["end_xy"]) - motion["start_xy"]))
            motion["commanded_motion"] = bool(motion["commanded_translation_ticks"] or motion["commanded_turn_ticks"])
            self.hold()

    def execute(self, *, grip_world, squat_hint, ghost):
        self.deadline = time.monotonic() + self.config.recovery_timeout
        self.standing_z = float(self.p.obs()["pelvis"]["pos"][2])
        drop = float(squat_hint or 0.0)
        why = "current stand is not graspable"
        self._drop_hint = drop
        journal = self.execution.metrics["recovery"]["attempts"]
        succeeded = False
        visible = bool(getattr(self.config, "visible_expert", False))
        try:
            self.hold()
            if not self.brake():
                return False, "recovery base did not stop before planning"
            self.standing_z = float(self.p.obs()["pelvis"]["pos"][2])
            # Visible expert: the squat depth is the deterministic stand's own
            # drop and the stand is never corrected. assess() only screens it
            # (and prepares the grasp rotation); an infeasible stand discards
            # the episode instead of walking to a stand no image explains.
            attempts = 0 if visible else self.config.recovery_attempts
            for correction in range(attempts + 1):
                if self.expired():
                    return False, "recovery time budget exhausted"
                state = self.p.obs()
                from locomani.tasks.bottle.execution import toppled

                if state["fallen"] or toppled(state, self.task.target.key):
                    return False, "recovery target or base disturbed"
                start = len(self.execution.events)
                if visible:
                    fixed = float(np.clip(drop, 0.03, self.config.max_drop))
                    selected = fixed if self.assess(state, fixed) else None
                else:
                    selected = self.select_drop(state, drop)
                if selected is not None:
                    drop = selected
                    self.retracted = False
                    self.execution.enter("recovery_attempt", correction=correction, drop=drop)
                    ok, why = self.finish(grip_world=self.task.grip(state), squat_hint=drop, ghost=ghost)
                    journal.append(dict(correction=correction, ok=ok, reason=why))
                    if ok:
                        succeeded = True
                        return True, why
                    # Entry/contact failure is not a positioning retry. No
                    # opening, standing or walking while a grasp may exist.
                    if any(
                        e["stage"] in ("grasp.in", "grasp.close", "grasp.lift") for e in self.execution.events[start:]
                    ):
                        return False, why
                    local = self.execution.metrics.get("measured_grasp", [])
                    if local and not local[-1]["can_reposition"]:
                        return False, why
                else:
                    why = "current stand lacks a clear squat and short grasp path"
                    journal.append(dict(correction=correction, ok=False, reason=why))
                if correction == attempts:
                    break
                state = self.p.obs()
                if state["fallen"] or toppled(state, self.task.target.key):
                    return False, "recovery target or base disturbed"
                self.execution.enter("recovery_needed", reason=why)
                projected = deepcopy(state)
                projected["pelvis"]["pos"][2] = self.standing_z
                if self.choose_stand(projected) is None:
                    return False, "no meaningful feasible stand correction before rising"
                if not self.withdraw_and_stand():
                    return False, "recovery could not safely retract and stand"
                state = self.p.obs()
                goal = self.choose_stand(state)
                if goal is None:
                    return False, "no nearby collision-screened grasp region"
                if not self.move_to_region(goal):
                    return False, "recovery positioning failed or timed out"
                self._drop_hint = goal[2]
                if not self.brake():
                    return False, "recovery base did not stop after positioning"
                drop = goal[2]  # Recheck ACTUAL state next iteration, never trust arrival alone.
            return False, (
                f"visible expert: stand infeasible, no reposition ({why})"
                if visible
                else f"recovery attempts exhausted: {why}"
            )
        finally:
            self.p.cmd.set_upper_body_vel(np.zeros(17))
            if not succeeded:
                self.execution.metrics["terminal_hold"] = dict(
                    mode=int(self.arm.mode),
                    height=float(self.arm.height),
                    speed=0.0,
                    movement=[0.0, 0.0, 0.0],
                    facing=list(map(float, self.p._facing(self.p.obs()))),
                )
                self.hold()
