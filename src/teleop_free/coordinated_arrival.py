"""Opt-in single-arm preparation during approach and crouch.

The released WBC only delivers height in static squat mode. Walking prepares
one arm; the same reference continues into squat + pregrasp transport. Default
collection never constructs this controller.
"""

import time

import adaptive_grasp
import numpy as np
from agent_server import LocomotionMode
from primitives import Primitives, upper_body_from_mj
from scipy.spatial.transform import Rotation
from stream_reach import _HAND, ARM_MJ, DT, SAFE_GAP, SQUAT_HEIGHT_BIAS, PlannerArm, _servo_continuous, toppled


class CoordinatedArrival:
    def __init__(self, p, task, config, ramp, execution):
        if _HAND != "dex1":
            raise ValueError("coordinated arrival currently supports Dex1 only")
        self.p, self.task, self.config = p, task, config
        self.side = task.target.side
        self.ramp, self.execution = ramp, execution
        self.arm = PlannerArm(p, self.side, send_vel=True)
        s = p.obs()
        self.arm.q = np.array(s["joints"], dtype=float)
        self._sync_reference()
        self.guard = task.transport_guard(self.arm.ik, config)
        self.plan = task.initial(s)
        if not self.plan.ok:
            raise ValueError(f"coordinated initial plan: {self.plan.why}")
        self.reference = np.asarray(s["palms"][self.side]["pos"], dtype=float)
        self.rotation_reference = Primitives._rot(s["palms"][self.side]["quat"])
        self.rotation_bias = np.zeros(3)
        self.error = None
        self.last_t = time.monotonic()
        self.stats = dict(checks=0, rejected=None, nav_ticks=0)
        execution.metrics["coordinated_arrival"] = self.stats
        execution.metrics["reach_config"] = config.to_dict()

    def _sync_reference(self):
        self.arm.q[12:15] = self.ramp.q[:3]
        self.arm.q[15:22] = self.ramp.q[3::2]
        self.arm.q[22:29] = self.ramp.q[4::2]

    def pregrasp(self):
        return np.asarray(self.plan.wrist) - np.asarray(self.plan.rot)[:, 0] * SAFE_GAP

    def walk_tick(self, state):
        self.step(state, walking=True)
        if self.error:
            self.abort()
        return self.error is None

    def step(self, state, *, walking):
        """Update upper reference; caller alone sends the planner frame."""
        if self.error:
            return
        now = time.monotonic()
        dt = float(np.clip(now - self.last_t, 0.005, 0.05))
        self.last_t = now
        palm = np.asarray(state["palms"][self.side]["pos"])
        target = self.pregrasp().copy()
        if walking:
            target[2] = max(target[2], self.task.target.grip_z + 0.15)
        # Far targets are unreachable. Raise into a comfortable pose in front
        # of the active shoulder while the base closes the remaining distance.
        local = self.p.world_to_root_full(target, state)
        if walking and local[0] > 0.30:
            local[0] = 0.30
            local[1] = -0.26 if self.side == "right" else 0.26
            rot_base = Primitives._rot(state["pelvis"]["quat"])
            target = np.asarray(state["pelvis"]["pos"]) + rot_base @ local
        lead = self.reference - palm
        self.reference = palm + lead * min(1.0, 0.06 / max(np.linalg.norm(lead), 1e-6))
        delta = target - self.reference
        self.reference += delta * min(1.0, 0.24 * dt / max(np.linalg.norm(delta), 1e-6))
        cmd = self.reference.copy()
        rotation = np.asarray(self.plan.rot)
        measured_rot = Primitives._rot(state["palms"][self.side]["quat"])
        if not walking:
            error = Rotation.from_matrix(rotation @ measured_rot.T).as_rotvec()
            self.rotation_bias += 0.7 * dt * error
            self.rotation_bias *= min(1.0, 1.10 / max(np.linalg.norm(self.rotation_bias), 1e-6))
        rotation = Rotation.from_rotvec(self.rotation_bias).as_matrix() @ rotation
        current = self.rotation_reference
        rv = Rotation.from_matrix(current.T @ rotation).as_rotvec()
        rotation = current @ Rotation.from_rotvec(rv * min(1.0, 1.4 * dt / max(np.linalg.norm(rv), 1e-6))).as_matrix()
        self.rotation_reference = rotation
        low = float((adaptive_grasp.hand_points() @ rotation.T)[:, 2].min())
        floor = self.task.target.surface_top + 0.045 - low - float((rotation @ adaptive_grasp.OFFSET)[2])
        cmd[2] = max(cmd[2], floor)
        q, residual, _ = self.arm.ik.solve(
            self.side,
            self.p.world_to_root_full(cmd, state),
            Primitives._rot(state["pelvis"]["quat"]).T @ rotation,
            q_mj29=self.arm.q,
            use_waist=not walking,
            iters=10,
        )
        if residual > 0.08:
            self.error = f"coordinated IK residual {residual:.3f}"
            return
        other = "left" if self.side == "right" else "right"
        # Slightly bent elbow, arm beside the torso. Slew from the owned
        # reference, rather than switching the idle hand abruptly.
        q[ARM_MJ[other]] = [1.0, 0.18 if other == "left" else -0.18, 0.0, 0.4, 0.0, 0.0, 0.0]
        want = upper_body_from_mj(q)
        proposed = self.ramp.q + np.clip(want - self.ramp.q, -self.ramp.max_step, self.ramp.max_step)
        checked = np.asarray(state["joints"], dtype=float).copy()
        checked[12:15], checked[15:22], checked[22:29] = proposed[:3], proposed[3::2], proposed[4::2]
        for side in (self.side, other):
            swept = self.guard.check(state, checked, side)
            self.stats["checks"] += 1
            if not swept.clear:
                self.stats["rejected"] = swept.to_dict()
                self.error = f"coordinated clearance: {swept.link} vs {swept.obstacle}"
                return
        self.ramp.push(want)
        self._sync_reference()
        self.stats["nav_ticks"] += int(walking)

    def abort(self):
        self.ramp.settle()
        self.p.cmd.frame(
            mode=LocomotionMode.IDLE,
            speed=0.0,
            movement=(0.0, 0.0, 0.0),
            facing=self.p._facing(self.p.obs()),
            height=-1.0,
        )

    def finish(self, **kwargs):
        ok = False
        try:
            ok, why = self._finish(**kwargs)
            return ok, why
        finally:
            if not ok:
                self.abort()

    def _finish(self, *, grip_world, squat_hint, ghost):
        """Crouch while transporting, then hand measured pregrasp to closure."""
        p, arm, ex = self.p, self.arm, self.execution
        if self.error:
            return False, self.error
        # No start()/hold(): navigation already owns this exact upper reference.
        arm._face = p._facing(p.obs())
        arm._q_prev = arm.q.copy()
        arm.next_t = time.time()
        ex.enter("posture")
        s0 = p.obs()
        z0 = float(s0["pelvis"]["pos"][2])
        # Geometry-based desired body height, independent of how high the hand
        # happened to be in the walking animation.
        drop = float(np.clip(max(z0 - (float(grip_world[2]) - 0.08), squat_hint or 0.0), 0.0, self.config.max_drop))
        before = p.world_to_root(grip_world, s0)
        self.plan = self.task.initial(s0)
        p.log(
            f"  coordinated: initial palm {s0['palms'][self.side]['pos']}, pregrasp {self.pregrasp()}, drop {drop:.3f}"
        )
        if not self.plan.ok:
            return False, self.plan.why
        if ghost:
            ghost(self.plan)
        start = time.monotonic()
        delivered = False
        while time.monotonic() - start < 10.0:
            s = p.obs()
            if s["fallen"] or toppled(s, self.task.target.key):
                return False, "coordinated preparation fell or toppled target"
            elapsed = time.monotonic() - start
            alpha = min(1.0, elapsed / 2.5)
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            arm.mode = LocomotionMode.IDEL_SQUAT if drop > 0.03 else LocomotionMode.IDLE
            arm.height = float(np.clip(z0 - drop * alpha - SQUAT_HEIGHT_BIAS, 0.40, 0.78)) if drop > 0.03 else -1.0
            self.step(s, walking=False)
            if self.error:
                return False, self.error
            p.cmd.set_hands(left=arm.hands["left"], right=arm.hands["right"])
            p.cmd.frame(mode=arm.mode, facing=arm._face, height=arm.height)
            palm = np.asarray(s["palms"][self.side]["pos"])
            rotation = Primitives._rot(s["palms"][self.side]["quat"])
            angle = np.linalg.norm(Rotation.from_matrix(rotation.T @ self.plan.rot).as_rotvec())
            got = z0 - float(s["pelvis"]["pos"][2])
            if (
                elapsed >= 2.5
                and np.linalg.norm(palm - self.pregrasp()) < 0.04
                and angle < 0.23
                and got >= min(0.06, 0.6 * drop)
            ):
                delivered = True
                break
            time.sleep(DT)
        if not delivered:
            p.log(
                f"  coordinated timeout: distance {np.linalg.norm(palm - self.pregrasp()):.3f} m, angle {angle:.3f}, drop {got:.3f}"
            )
            return False, "coordinated pregrasp delivery timeout"
        actual = p.world_to_root(self.task.grip(s), s)
        ex.transition.update(
            squat_requested=drop,
            squat_delivered=got,
            pre_squat_depth=float(before[0]),
            post_depth=float(actual[0]),
            reach_started=True,
            pregrasp_delivered=True,
        )
        if not (
            self.config.post_min <= actual[0] <= self.config.post_max
            and abs(np.arctan2(actual[1], actual[0])) <= np.deg2rad(self.config.azimuth_deg)
        ):
            return False, "coordinated post-crouch geometry outside depth/cone"
        p.log(f"  coordinated: pregrasp delivered during crouch, drop {got * 100:.1f} cm")
        ex.enter("reach_validation")
        self.ramp.settle()
        arm._q_prev = arm.q.copy()
        arm.next_t = time.time()
        try:
            return _servo_continuous(
                p,
                arm,
                self.task.target.key,
                self.plan,
                self.side,
                replan=None,
                ghost=ghost,
                surface_top=self.task.target.surface_top,
                execution=ex,
                transport_guard=self.guard,
                start_at_pregrasp=True,
                rotation_bias=self.rotation_bias.copy(),
            )
        finally:
            arm.stop()
