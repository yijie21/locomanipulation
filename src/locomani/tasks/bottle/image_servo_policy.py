"""Execute the RGB-error expert through the existing guarded SONIC arm channel.

The same feedback step runs during approach and at a parked table. Clock reads
bound execution/freshness and record diagnostics; they do not select arm goals.
Privileged collision/fall/contact checks remain independent veto/acceptance.
"""

import time
from dataclasses import asdict

import numpy as np

from locomani.backends.sonic.primitives import STANDING_PELVIS_Z, LocomotionMode, Primitives, upper_body_from_mj
from locomani.tasks.bottle import adaptive_grasp
from locomani.tasks.bottle.coupled_grasp_policy import screen_motion
from locomani.tasks.bottle.image_bottle import BottleImageObserver
from locomani.tasks.bottle.image_servo import ServoObservation, decide
from locomani.tasks.bottle.measured_grasp import bounded_rotation, closure_clear
from locomani.tasks.bottle.ready_grasp_policy import OFFSET, ready_q, templates
from locomani.tasks.bottle.recoverable_grasp_policy import RecoverableGraspPolicy, retracted_q


class ImageServoPolicy(RecoverableGraspPolicy):
    def __init__(self, *args):
        super().__init__(*args)
        if self.task.target.key != "water_bottle":
            raise ValueError("RGB cap calibration only supports water_bottle")
        self.observer = BottleImageObserver(self.arm.ik, self.config.image_calibration)
        self.limits = self.config.image_control
        self.stats = dict(
            samples=[],
            calibration=asdict(self.observer.calibration),
            control=asdict(self.limits),
            source="rgb_cap_cylinder_fit",
        )
        self.execution.metrics["image_servo"] = self.stats
        self._last_log = 0.0
        self._grip_world = None
        self.arm.q[12:15] = self.ramp.q[:3]
        self.arm.q[15:22] = self.ramp.q[3::2]
        self.arm.q[22:29] = self.ramp.q[4::2]
        self.arm._q_prev = self.arm.q.copy()
        self.arm.next_t = time.time()

    def close(self):
        self.p.cmd.set_upper_body_vel(np.zeros(17))
        self.observer.close()

    def _send(self, state, q, *, depth=None, target=None, height=None, contact=False):
        """Preserve reference continuity and publish finite-difference velocities."""
        q = self.arm.q + np.clip(q - self.arm.q, -self.limits.joint_step, self.limits.joint_step)
        error = self.reference_check(state, q, "enter" if contact else "prepare")
        if error:
            self.error = error
            return False
        self.arm.q = q
        self.ramp.q = upper_body_from_mj(q).copy()
        self.arm._face = self.p._facing(state)
        if self.arm.mode != LocomotionMode.IDEL_SQUAT:
            self.arm.mode = LocomotionMode.IDLE
            self.arm.height = -1.0
        movement = (0.0, 0.0, 0.0)
        speed = 0.0
        if height is not None:
            from locomani.backends.sonic.stream_reach import SQUAT_HEIGHT_BIAS

            self.arm.mode = LocomotionMode.IDEL_SQUAT
            self.arm.height = height - SQUAT_HEIGHT_BIAS
        if target is not None and height is None and self._needs_walk(target, state):
            self.arm.mode = LocomotionMode.SLOW_WALK
            self.arm.height = -1.0
            br = Primitives._rot(state["pelvis"]["quat"])
            relative_goal = np.array(
                [
                    self.config.visible_brake_depth,
                    -self.limits.approach_lateral,
                    0.0,
                ]
            )
            delta = br @ (target - relative_goal)
            delta[2] = 0.0
            movement = delta / max(np.linalg.norm(delta), 1e-6)
            u = float(
                np.clip((depth - self.config.visible_brake_depth) / max(self.config.visible_decel_band, 0.001), 0, 1)
            )
            speed = self.limits.walking_floor + (self.limits.walking_speed - self.limits.walking_floor) * u
            if depth <= self.limits.walking_mid_depth:
                speed = min(speed, self.limits.walking_mid)
            else:
                # Aim at the APPROACH GOAL, not at the bottle. The stand is
                # deliberately offset `approach_lateral` to the right, so
                # target[1] converges to -0.195 and never to zero: bearing to
                # the bottle then commands a permanent ~10 deg right turn every
                # tick and the robot pirouettes. Measured in attempt10, once the
                # perception fix stopped under-reporting depth and this branch
                # started running: heading swept +2400..+3200 deg inside a 3-8 cm
                # box, 0.6 m walked for 0.03 m of net travel. `goal` is the same
                # pelvis-frame vector `movement` is built from, so it goes to
                # zero as the robot arrives.
                goal = target - relative_goal
                yaw = self.p.pelvis_yaw(state) + float(
                    np.clip(np.arctan2(goal[1], goal[0]), -self.limits.far_yaw_lead, self.limits.far_yaw_lead)
                )
                self.arm._face = (float(np.cos(yaw)), float(np.sin(yaw)), 0.0)
            base_xy = np.asarray(state["pelvis"]["pos"][:2])
            if not self.floor_clear(
                base_xy + self.limits.edge_lookahead * movement[:2], margin=self.limits.edge_margin
            ):
                self.error = "RGB approach reached table-edge safety margin"
                return False
        # Use one mode throughout the tick; do not alternate IDLE/WALK in the
        # command publisher while updating the upper-body reference.
        self.arm._push(q, observed=state, movement=tuple(movement), speed=speed)
        return True

    def _needs_walk(self, target, state):
        drop = max(0.0, STANDING_PELVIS_Z - state["pelvis"]["pos"][2])
        depth_limit = self.limits.approach_end_depth + self.config.squat_prediction_backshift_ratio * drop
        return target[0] > depth_limit or abs(target[1] + self.limits.approach_lateral) > self.limits.lateral_tolerance

    def _goal(self, state, target, up, palm, rotation):
        """Far-field safe preparation connects to near-field current-error servo."""
        depth = float(target[0])
        # This is a fixed comfortable grasp attitude, not an episode-selected pose.
        _, centre, terminal = templates()[4]
        observation = ServoObservation(target, palm, rotation, terminal, up)
        command = decide(observation, self.limits)
        if depth <= self.limits.prealign_depth:
            return command, centre
        prep = self._view_q()
        prep[26] = 0.22
        prep_palm, prep_rot = self.arm.ik.fk(prep, self.side)
        neutral, neutral_rot = self.arm.ik.fk(ready_q(), self.side)
        u = float(
            np.clip(
                (self.limits.approach_start_depth - depth)
                / (self.limits.approach_start_depth - self.limits.approach_end_depth),
                0.0,
                1.0,
            )
        )
        from scipy.spatial.transform import Rotation

        rv = Rotation.from_matrix(neutral_rot.T @ prep_rot).as_rotvec()
        attitude = neutral_rot @ Rotation.from_rotvec(u * rv).as_matrix()
        from locomani.tasks.bottle.image_servo import ServoCommand

        return ServoCommand(neutral + u * (prep_palm - neutral), attitude, "visual_prepare"), centre

    def _view_q(self):
        q = retracted_q()
        q[25] = -self.limits.view_elbow
        return q

    def _tick(self, state):
        if state["fallen"]:
            self.error = "RGB servo: base fell"
            return None
        fit = self.observer.observe(state)
        if fit is None or fit["uncertainty_m"] > self.observer.calibration.approach_uncertainty_m:
            # The known occluder is the hand itself. Retracting it is a bounded,
            # proprioception-defined view action, never a blind target approach.
            self.stats["reacquire_ticks"] = self.stats.get("reacquire_ticks", 0) + 1
            self._send(state, self._view_q())
            return None
        br = Primitives._rot(state["pelvis"]["quat"])
        base = np.asarray(state["pelvis"]["pos"])
        up = br.T @ np.array([0.0, 0.0, 1.0])
        target = fit["centre"] - self.config.grip_below_center * up
        self._grip_world = base + br @ target
        palm = br.T @ (np.asarray(state["palms"][self.side]["pos"]) - base)
        rotation = br.T @ Primitives._rot(state["palms"][self.side]["quat"])
        command, comfortable = self._goal(state, target, up, palm, rotation)
        delta = command.aim - palm
        aim = palm + delta * min(
            1.0 + self.limits.tracking_gain, self.limits.cartesian_lead / max(np.linalg.norm(delta), 1e-9)
        )
        q, pe, re = self.arm.ik.solve(
            self.side, aim, bounded_rotation(rotation, command.rotation), q_mj29=self.arm.q, use_waist=False, iters=40
        )
        if pe > self.limits.ik_position_tolerance or re > self.limits.ik_rotation_tolerance:
            # The body can still reduce a below-workspace target's error.
            # Hold a valid arm reference while the screened squat progresses.
            self.stats["ik_hold_ticks"] = self.stats.get("ik_hold_ticks", 0) + 1
            q = self.arm.q.copy()
        height = None
        if not self._needs_walk(target, state):
            # Geometry requests squat; it does not wait for a timer or a
            # stage-start height. A 1mm proposal is screened together with arm.
            need = float((comfortable - target) @ up)
            desired_height = float(
                np.clip(base[2] - max(need, 0), STANDING_PELVIS_Z - self.config.max_drop, STANDING_PELVIS_Z)
            )
            from locomani.backends.sonic.stream_reach import SQUAT_HEIGHT_BIAS

            previous = (
                self.arm.height + SQUAT_HEIGHT_BIAS if self.arm.mode == LocomotionMode.IDEL_SQUAT else float(base[2])
            )
            height = previous + float(np.clip(desired_height - previous, -self.limits.body_step, self.limits.body_step))
            limited_q = self.arm.q + np.clip(q - self.arm.q, -self.limits.joint_step, self.limits.joint_step)
            rejection = screen_motion(
                self.guard,
                state,
                limited_q,
                height - base[2],
                br[:, 0],
                contact_side=self.side if command.intent in ("enter", "capture") else None,
            )
            if rejection:
                self.error = f"RGB servo sweep: {rejection}"
                return None
        if not self._send(
            state,
            q,
            depth=float(target[0]),
            target=target,
            height=height,
            contact=command.intent in ("enter", "capture"),
        ):
            return None
        now = time.monotonic()
        if now - self._last_log >= 0.10:
            truth = np.asarray(state["objects"][self.task.target.key]["pos"])
            row = dict(
                wall=time.time(),
                seq=state.get("seq"),
                intent=command.intent,
                target_root=target.tolist(),
                palm_root=palm.tolist(),
                aim_root=command.aim.tolist(),
                wrist_rotation=rotation.tolist(),
                target_world=self._grip_world.tolist(),
                base_world=base.tolist(),
                palm_world=state["palms"][self.side]["pos"],
                command_height=height,
                close_candidate=command.close,
                views=fit["selected"],
                uncertainty_m=fit["uncertainty_m"],
                detection=[dict(camera=v["camera"], pixels=v["pixels"], box=v["box"].tolist()) for v in fit["views"]],
                object_truth_world=truth.tolist(),
                perception_error_m=float(np.linalg.norm(base + br @ fit["centre"] - truth)),
            )
            self.stats["samples"].append(row)
            self._last_log = now
            self.p.log(
                f"[image-servo] {command.intent} depth={target[0]:.3f} error={np.linalg.norm(delta):.3f} vision={','.join(fit['selected'])} audit={row['perception_error_m']:.3f}"
            )
        return command, target, palm, rotation

    def approach(self):
        self.execution.enter("approach")
        end = time.monotonic() + self.config.approach_timeout
        while time.monotonic() < end:
            state = self.p.obs()
            result = self._tick(state)
            if self.error:
                return False, self.error
            if result is not None and not self._needs_walk(result[1], state):
                self.execution.metrics["first_stop"] = float(np.linalg.norm(result[1][:2]))
                return None
        return False, "RGB approach could not reach a visible grasp region"

    def _capture_ready(self, result, state):
        # CLOSE is a current-observation intent. Mechanical closure/contact
        # verification belongs to the existing low-level gripper primitive.
        return result[0].close and self.observer.latest["precise"] and closure_clear(self, state) is None

    def execute(self, *, grip_world, squat_hint, ghost):
        from locomani.backends.sonic.stream_reach import _servo_continuous

        end = time.monotonic() + self.config.recovery_timeout
        try:
            while time.monotonic() < end:
                state = self.p.obs()
                result = self._tick(state)
                if self.error:
                    return False, self.error
                if result is None or not self._capture_ready(result, state):
                    continue
                # Existing discrete close/lift controller, now entered only
                # after RGB says the object is actually between the jaws.
                palm = np.asarray(state["palms"][self.side]["pos"])
                rot = Primitives._rot(state["palms"][self.side]["quat"])
                plan = adaptive_grasp.Grasp(
                    ok=True, rot=rot, wrist=palm, grip=self._grip_world.copy(), approach=rot[:, 0], travel=0.0
                )
                self.execution.enter("grasp.capture_ready", source="rgb")
                return _servo_continuous(
                    self.p,
                    self.arm,
                    self.task.target.key,
                    plan,
                    self.side,
                    ghost=ghost,
                    surface_top=self.task.target.surface_top,
                    execution=self.execution,
                    reference_guard=self.reference_check,
                    transport_guard=self.guard,
                    close_from_current=True,
                    grip_target=lambda s: self._grip_world,
                    require_bilateral=True,
                )
            return False, "RGB grasp feedback timed out"
        finally:
            self.p.cmd.set_upper_body_vel(np.zeros(17))
