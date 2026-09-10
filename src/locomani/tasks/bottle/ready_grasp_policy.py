"""Experimental ready → stop → squat → short reach tabletop policy.

Joint zero is already a bent-elbow, level-forearm pose in this MJCF. Build
sampling support from its FK, rather than sending unconstrained IK to random
stands. No orientation overdrive or outboard/up/down transport is used.
"""

import time
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.backends.sonic.agent_server import LocomotionMode, StreamMode
from locomani.backends.sonic.primitives import READY_UPPER, STANDING_PELVIS_Z, Primitives, upper_body_from_mj
from locomani.backends.sonic.primitives_const import GRASP_CENTRE_OFFSET_CONST, HAND
from locomani.backends.sonic.upper_ref import UpperRamp
from locomani.primitives.arm_ik import ArmIK
from locomani.primitives.g1_joints import ARM_MJ
from locomani.tasks.bottle import adaptive_grasp

OFFSET = np.asarray(GRASP_CENTRE_OFFSET_CONST)
WRIST_LIMIT = 0.45
PRE_GAP = 0.07


def terminal_rotation(root_rotation, local_rotation, config):
    """An upright bottle needs clearance above the fingers, including its neck.

    The complete arm/target collision screen remains authoritative.
    """
    return root_rotation @ local_rotation


def ready_q():
    q = np.zeros(29)
    q[15:22], q[22:29] = READY_UPPER[3::2], READY_UPPER[4::2]
    for sd in ("left", "right"):
        idx = list(ARM_MJ[sd])
        q[idx[0]], q[idx[3]] = 0.30, -0.30
    return q


@lru_cache(maxsize=1)
def templates():
    """FK-defined comfortable terminal grasp centres, not arbitrary xyz boxes."""
    if HAND != "dex1":
        raise ValueError("ready grasp baseline currently requires Dex1")
    ik = ArmIK()
    rows = []
    for pitch in (-0.05, 0.0, 0.05):
        for elbow in (0.08, 0.12, 0.16):
            q = ready_q()
            q[22], q[25] = pitch, elbow
            palm, rot = ik.fk(q, "right")
            centre = palm + rot @ OFFSET
            rows.append((q, centre, rot))
    return tuple(rows)


@lru_cache(maxsize=64)
def lower_grip_samples(grip_z, max_drop):
    ik = ArmIK()
    idx = list(ARM_MJ["right"])
    for j in (idx[0], idx[3]):
        ik.lo[j], ik.hi[j] = max(ik.lo[j], -0.65), min(ik.hi[j], 0.65)
    for j in idx[-3:]:
        ik.lo[j], ik.hi[j] = max(ik.lo[j], -WRIST_LIMIT), min(ik.hi[j], WRIST_LIMIT)
    ik.lo[idx[2]], ik.hi[idx[2]] = max(ik.lo[idx[2]], -0.5), min(ik.hi[idx[2]], 0.5)
    choices = []
    for q, centre, rot in templates():
        desired_drop = STANDING_PELVIS_Z - (grip_z - centre[2])
        drop = float(np.clip(desired_drop, 0.03, max_drop))
        if abs(desired_drop - drop) > 0.04:
            continue
        adjusted = centre.copy()
        adjusted[2] = grip_z - (STANDING_PELVIS_Z - drop)
        _, pe, re = ik.solve("right", adjusted - rot @ OFFSET, rot, q_mj29=q, use_waist=False, iters=60)
        if pe <= 0.015 and re <= 0.15:
            choices.append((adjusted, drop))
    return tuple(choices)


def sample_stand(plan, rng, config):
    choices = []
    for _, centre, _ in templates():
        drop = STANDING_PELVIS_Z - (plan.grip_z - centre[2])
        if 0.03 <= drop <= config.max_drop:
            choices.append((centre, drop))
    if not choices and config.ready_grasp_recovery:
        # A lower grip may require a few cm of arm motion beyond the FK
        # sampling template. Verify that motion instead of requesting a
        # deeper squat or abandoning sampling before the live policy runs.
        choices = list(lower_grip_samples(float(plan.grip_z), config.max_drop))
    if not choices:
        raise ValueError("target height has no comfortable baseline grasp; do not broaden IK")
    if getattr(config, "visible_expert", False):
        # One stand per object height, no draw and no noise: the stand is a
        # function of the (visible) grip height alone, so the same measured
        # pose never carries two different stop labels. Diversity comes from
        # the spawn. The median-drop template is the most comfortable middle.
        ordered = sorted(choices, key=lambda cd: (cd[1], cd[0][0]))
        centre, drop = ordered[len(ordered) // 2]
        return centre[:2] - [config.backshift, 0.0], float(drop)
    centre, drop = choices[int(rng.integers(len(choices)))]
    # Small perturbations within a comfortable FK neighbourhood. Account for
    # the expected squat backshift; live post-squat checks remain authoritative.
    return centre[:2] - [config.backshift, 0.0] + rng.uniform(-0.01, 0.01, 2), float(drop)


def prepare_ready(p, timeout=3.0):
    """Before recording: both arms ready, wrists neutral, no hidden reach phase."""
    from locomani.backends.sonic import upper_ref

    seed = upper_ref.shared().upper17()
    if seed is None:
        raise RuntimeError("ready baseline needs a fresh owned planner reference")
    target = upper_body_from_mj(ready_q())
    ramp = UpperRamp(p.cmd, seed)
    p.cmd.stream_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY
    end = time.monotonic() + timeout
    residual, joint_err, pushes = float("nan"), float("nan"), 0
    while time.monotonic() < end:
        state = p.obs()
        if state["fallen"]:
            return False
        residual = ramp.push(target)
        pushes += 1
        p.cmd.frame(mode=LocomotionMode.IDLE, facing=p._facing(state))
        measured = upper_body_from_mj(state["joints"])
        joint_err = float(np.max(np.abs(measured[3:] - target[3:])))
        if residual < 0.01 and joint_err < 0.30:
            ramp.settle()
            return True
        time.sleep(0.02)
    ramp.settle()
    # Say WHICH gate failed: the reference ramp not finished (residual) or the
    # deploy not tracking it (joint_err) -- the two need different fixes.
    print(
        f"[gym] ready posture NOT delivered in {timeout:.1f} s: ramp residual {residual:.3f}, "
        f"max joint err {joint_err:.2f} rad, {pushes} pushes"
    )
    return False


class ReadyGraspPolicy:
    approach_lateral = -0.19

    def __init__(self, p, task, config, ramp, execution):
        from locomani.backends.sonic.stream_reach import PlannerArm

        self.p, self.task, self.config = p, task, config
        self.ramp, self.execution = ramp, execution
        self.side = task.target.side
        if self.side != "right" or HAND != "dex1":
            raise ValueError("baseline calibrated for right-hand Dex1 bottle grasp only")
        self.arm = PlannerArm(p, self.side, send_vel=True)
        self.arm.q = np.asarray(p.obs()["joints"], dtype=float).copy()
        self.guard = task.transport_guard(self.arm.ik, config)
        # Prevent the optimizer using wrist twists/large torso rotations to
        # rescue an unsuitable stand. These limits apply through close/lift too.
        for sd in ("left", "right"):
            idx = list(ARM_MJ[sd])
            for j in (idx[0], idx[3]):
                self.arm.ik.lo[j] = max(self.arm.ik.lo[j], -0.65)
                self.arm.ik.hi[j] = min(self.arm.ik.hi[j], 0.65)
            for j in idx[-3:]:
                self.arm.ik.lo[j] = max(self.arm.ik.lo[j], -WRIST_LIMIT)
                self.arm.ik.hi[j] = min(self.arm.ik.hi[j], WRIST_LIMIT)
            self.arm.ik.lo[idx[2]] = max(self.arm.ik.lo[idx[2]], -0.5)
            self.arm.ik.hi[idx[2]] = min(self.arm.ik.hi[idx[2]], 0.5)
        self.error = None
        execution.metrics["ready_grasp_baseline"] = dict(wrist_limit=WRIST_LIMIT, short_reach_max=0.20)
        execution.metrics["reach_config"] = config.to_dict()

    def reference_check(self, state, q, phase):
        for sd in ("left", "right"):
            idx = list(ARM_MJ[sd])
            if np.max(np.abs(np.asarray(q)[idx[-3:]])) > WRIST_LIMIT + 1e-6:
                return "baseline wrist comfort limit"
            if abs(q[idx[2]]) > 0.5 + 1e-6:
                return "baseline shoulder yaw comfort limit"
            if max(abs(q[idx[0]]), abs(q[idx[3]])) > 0.65 + 1e-6:
                return "baseline shoulder/elbow comfort limit"
        # Before intended target contact, screen both arm paths. The retained
        # servo has its own contact-aware gates for entry, closure and lifting.
        for sd in ("left", "right"):
            hit = self.guard.check(state, q, sd, allow_target_contact=(sd == self.side and phase != "prepare"))
            if not hit.clear:
                return f"baseline clearance: {hit.link} vs {hit.obstacle}"
        return None

    def _push(self, state, q):
        wanted = upper_body_from_mj(q)
        u = self.ramp.q + np.clip(wanted - self.ramp.q, -self.ramp.max_step, self.ramp.max_step)
        checked = np.asarray(state["joints"], dtype=float).copy()
        checked[12:15], checked[15:22], checked[22:29] = u[:3], u[3::2], u[4::2]
        self.error = self.reference_check(state, checked, "prepare")
        if self.error:
            self.ramp.settle()
            return False
        self.ramp.push(wanted)
        self.arm.q = checked
        return True

    def walk_tick(self, state):
        return self._push(state, ready_q())

    def finish(self, *, grip_world, squat_hint, ghost):
        from locomani.backends.sonic.stream_reach import SQUAT_HEIGHT_BIAS, _servo_continuous, toppled

        p, arm, ex = self.p, self.arm, self.execution
        ok = False
        try:
            ex.enter("baseline_brake")
            start = time.monotonic()
            stable_since = None
            while time.monotonic() - start < 3.0:
                state = p.obs()
                now = time.monotonic()
                speed = float(np.linalg.norm(state["pelvis"]["vel"][:2]))
                if not self.walk_tick(state):
                    return False, self.error
                p.cmd.frame(mode=LocomotionMode.IDLE, facing=p._facing(state))
                stable_since = (stable_since or now) if speed < 0.06 else None
                if stable_since is not None and now - stable_since >= 0.25:
                    break
                time.sleep(0.02)
            else:
                return False, "baseline base did not stop"
            z0 = float(state["pelvis"]["pos"][2])
            drop = float(squat_hint or 0.0)
            if not 0.03 <= drop <= self.config.max_drop:
                return False, "baseline posture outside calibrated drop"
            arm._face = p._facing(state)
            arm.mode = LocomotionMode.IDEL_SQUAT
            arm.height = float(np.clip(z0 - drop - SQUAT_HEIGHT_BIAS, 0.40, 0.78))
            ex.enter("baseline_squat")
            start = time.monotonic()
            while time.monotonic() - start < 3.0:
                state = p.obs()
                if state["fallen"] or toppled(state, self.task.target.key):
                    return False, "baseline fell or toppled target"
                if self.config.measured_squat_handover and time.monotonic() - start > 0.8:
                    from locomani.tasks.bottle.arrival_validation import squat_handover
                    from locomani.tasks.bottle.grasp_capture import closure_geometry

                    geometry = closure_geometry(state, self.side, self.task.grip(state))
                    delivered = z0 - float(state["pelvis"]["pos"][2])
                    reason = squat_handover(geometry, delivered, drop)
                    if reason:
                        arm.height = float(state["pelvis"]["pos"][2]) - SQUAT_HEIGHT_BIAS
                        ex.metrics["squat_handover"] = dict(
                            reason=reason,
                            requested_drop_m=drop,
                            delivered_drop_m=delivered,
                            geometry=geometry,
                            hold_height=arm.height,
                        )
                        p.cmd.frame(mode=arm.mode, height=arm.height, facing=arm._face)
                        break
                if (
                    self.config.ready_grasp_recovery
                    and not self.config.measured_squat_handover
                    and time.monotonic() - start > 0.8
                ):
                    from locomani.tasks.bottle.measured_grasp import closure_geometry

                    if closure_geometry(state, self.side, self.task.grip(state))["ready"]:
                        # Stop descending once the delivered jaw is seated;
                        # the local controller still checks stability/closure clearance.
                        arm.height = float(state["pelvis"]["pos"][2]) - SQUAT_HEIGHT_BIAS
                        delivered = z0 - float(state["pelvis"]["pos"][2])
                        p.cmd.frame(mode=arm.mode, height=arm.height, facing=arm._face)
                        break
                if not self.walk_tick(state):
                    return False, self.error
                p.cmd.frame(mode=arm.mode, height=arm.height, facing=arm._face)
                delivered = z0 - float(state["pelvis"]["pos"][2])
                if (
                    not self.config.measured_squat_handover
                    and abs(delivered - drop) < 0.02
                    and time.monotonic() - start > 0.8
                ):
                    break
                time.sleep(0.02)
            else:
                return False, "baseline squat not delivered"
            if self.config.ready_grasp_recovery:
                from locomani.tasks.bottle.measured_grasp import run

                ex.transition.update(
                    squat_requested=drop, squat_delivered=delivered, reach_started=True, pregrasp_delivered=False
                )
                ok, why = run(self, ghost)
                return ok, why
            live = self.task.grip(state)
            rel = p.world_to_root_full(live, state)
            nearest = min(templates(), key=lambda row: np.linalg.norm(row[1] - rel))
            ex.metrics["ready_grasp_baseline"]["post_grip_full"] = rel.tolist()
            rot = Primitives._rot(state["pelvis"]["quat"]) @ nearest[2]
            wrist = live - rot @ OFFSET
            _, pos_error, rot_error = arm.ik.solve(
                self.side,
                p.world_to_root_full(wrist, state),
                Primitives._rot(state["pelvis"]["quat"]).T @ rot,
                q_mj29=arm.q,
                use_waist=False,
            )
            if pos_error > 0.015 or rot_error > 0.15:
                return False, "baseline stand has no comfortable terminal IK"
            low = float((adaptive_grasp.hand_points() @ rot.T)[:, 2].min())
            if live[2] + low < self.task.target.surface_top + 0.025:
                return False, "baseline grasp lacks tabletop clearance"
            pre = wrist - rot[:, 0] * PRE_GAP
            palm = np.asarray(state["palms"][self.side]["pos"])
            if np.linalg.norm(pre - palm) > 0.20:
                return False, "baseline reach exceeds short-path limit"
            plan = adaptive_grasp.Grasp(
                ok=True, rot=rot, wrist=wrist, grip=live, approach=rot[:, 0], travel=float(np.linalg.norm(wrist - palm))
            )
            ex.enter("baseline_pregrasp")
            start = time.monotonic()
            ref = palm.copy()
            while time.monotonic() - start < 4.0:
                state = p.obs()
                if state["fallen"] or toppled(state, self.task.target.key):
                    return False, "baseline preparation disturbed target or base"
                delta = pre - ref
                ref += delta * min(1.0, 0.003 / max(np.linalg.norm(delta), 1e-6))
                q, pos_err, rot_err = arm.ik.solve(
                    self.side,
                    p.world_to_root_full(ref, state),
                    Primitives._rot(state["pelvis"]["quat"]).T @ rot,
                    q_mj29=arm.q,
                    use_waist=False,
                    iters=30,
                )
                if pos_err > 0.02 or rot_err > 0.15:
                    return False, "baseline short reach has no comfortable IK"
                if not self._push(state, q):
                    return False, self.error
                p.cmd.frame(mode=arm.mode, height=arm.height, facing=arm._face)
                palm = np.asarray(state["palms"][self.side]["pos"])
                angle = np.linalg.norm(
                    Rotation.from_matrix(Primitives._rot(state["palms"][self.side]["quat"]).T @ rot).as_rotvec()
                )
                if np.linalg.norm(palm - pre) < 0.025 and angle < 0.23:
                    break
                time.sleep(0.02)
            else:
                return False, "baseline measured pregrasp not delivered"
            ex.transition.update(
                squat_requested=drop, squat_delivered=delivered, reach_started=True, pregrasp_delivered=True
            )
            self.ramp.settle()
            arm._q_prev = arm.q.copy()
            arm.next_t = time.time()
            ok, why = _servo_continuous(
                p,
                arm,
                self.task.target.key,
                plan,
                self.side,
                replan=None,
                ghost=ghost,
                surface_top=self.task.target.surface_top,
                execution=ex,
                transport_guard=self.guard,
                start_at_pregrasp=True,
                reference_guard=self.reference_check,
                pregrasp_gap=PRE_GAP,
            )
            return ok, why
        finally:
            self.finish_cleanup(ok)

    def finish_cleanup(self, ok):
        p, arm = self.p, self.arm
        p.cmd.set_upper_body_vel(np.zeros(17))
        if not ok:
            p.cmd.frame(
                mode=LocomotionMode.IDLE, height=-1.0, speed=0.0, movement=(0.0, 0.0, 0.0), facing=p._facing(p.obs())
            )
        else:
            arm.stop()
