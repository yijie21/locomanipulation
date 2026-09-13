"""Enter a supported hip hinge, then reuse carry's bilateral capture and lift gates."""

import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from locomani.backends.sonic.primitives import upper_body_from_mj
from locomani.backends.sonic.streamed_arm import StreamedArm
from locomani.tasks.crate.bimanual_carry import OFFSET, PALM_PITCH_DEG, UP_ABOVE, UP_SHORT, BimanualArms, rot_y, rot_z
from locomani.tasks.crate.carry_motion import CarryMotion
from locomani.tasks.crate.whole_body_ik import plan_low_reach


class WholeBodyArms(BimanualArms):
    """A paired arm servo over a held full-body pose reference, with the same task interface."""

    def __init__(self, p, config):
        super().__init__(p, config)
        self.arm = StreamedArm(p, "right", ik=self.arm.ik, left_guard=False)
        self.arm.ik.lo[12:15], self.arm.ik.hi[12:15] = self.solver.lo[12:15], self.solver.hi[12:15]
        self.entered = False

    def start(self, q17=None, *, settle_seconds=0.6):
        if self.entered:
            return
        self.arm.start()
        self.arm.hands = {k: v.astype(np.float32) for k, v in self.hands.items()}
        self.entered = True

    def stop(self):
        # Preserve the final grasp pose. The next reset releases the hands and
        # switches back to planner before resetting the standing robot.
        return

    def push(self, q, *, reference_velocity=None):
        sent_at = time.time()
        q = self.motion.step(q, sent_at) if self.motion is not None else np.asarray(q).copy()
        self.arm.q = q
        velocity = (
            self.motion.reference.velocity
            if self.motion is not None
            else np.zeros(29)
            if reference_velocity is None
            else reference_velocity
        )
        self.arm._push(q, reference_velocity=velocity)
        if self.motion_recording is not None:
            self.motion_recording.sample(
                sent_at,
                upper_body_from_mj(q),
                upper_body_from_mj(velocity),
                self.p.obs(),
                root_reference=dict(
                    quat=self.arm.root_quat.tolist(), joints=q.tolist(), velocity=np.asarray(velocity).tolist()
                ),
            )


def _phase_whole_body(ctx):
    """Plan at the observed stand; ramp continuously to a low staging pose."""
    from locomani.tasks.crate.carry_phases import admit_grasp, stand_metrics

    s = ctx.p.obs()
    arms = WholeBodyArms(ctx.p, ctx.config)
    rotation = rot_z(ctx.psi) @ rot_y(np.radians(PALM_PITCH_DEG))
    planning = time.time()
    pose = plan_low_reach(arms.arm.ik, s, list(ctx.bars.values()), [rotation, rotation], ctx.config, tool_offset=OFFSET)
    ctx.m.update(
        body_plan_seconds=time.time() - planning,
        body_plan_hand_errors=pose.hand_errors.tolist(),
        body_plan_foot_errors=pose.foot_errors.tolist(),
        body_plan_com_shift=pose.com_shift.tolist(),
        body_plan_pelvis=pose.position.tolist(),
        body_plan_rotation=pose.rotation.tolist(),
    )
    if not pose.reachable:
        ctx.m["why"] = "[whole_body] no supported low-reach reference"
        return ctx.m
    planned = dict(
        s,
        pelvis=dict(
            s["pelvis"],
            pos=pose.position.tolist(),
            quat=Rotation.from_matrix(pose.rotation).as_quat()[[3, 0, 1, 2]].tolist(),
        ),
    )
    goals = {side: ctx.bars[side] + np.array([*(-ctx.n * UP_SHORT), UP_ABOVE]) for side in ("left", "right")}
    arms.arm.q = pose.joints.copy()
    target = arms.solve(planned, goals, {side: rot_z(ctx.psi) for side in goals}, iters=180)
    # The full-body pose owns legs and root; both hands approach from above.
    ctx.arms = arms
    arms.motion_recording = ctx.motion_recording
    arms.start()
    q0 = arms.arm.q.copy()
    r0 = Rotation.from_quat(np.asarray(ctx.p.obs()["pelvis"]["quat"])[[1, 2, 3, 0]])
    roots = Slerp([0, 1], Rotation.from_matrix([r0.as_matrix(), pose.rotation]))
    distance = float(np.max(np.abs(target - q0)))
    duration = max(
        4.0, 1.875 * distance / ctx.config.joint_speed, np.sqrt(5.8 * distance / ctx.config.joint_acceleration)
    )
    start = time.time()
    while time.time() - start < duration + 0.4:
        t = time.time() - start
        u = min(t / duration, 1.0)
        speed = 30 * u * u * (1 - u) * (1 - u) / duration
        u = u**3 * (10 - 15 * u + 6 * u * u)
        arms.arm.root_quat = roots(u).as_quat()[[3, 0, 1, 2]].astype(np.float32)
        arms.push(q0 * (1 - u) + target * u, reference_velocity=(target - q0) * speed)
        s = ctx.p.obs()
        if s["fallen"]:
            ctx.m["why"] = "[whole_body] fell entering the low pose"
            return ctx.m
    arms.motion = CarryMotion(arms.arm, arms.solver, ctx.config, time.time())
    ctx.m["body_entry_seconds"] = time.time() - start
    s, yaw, along, lateral, edge = stand_metrics(ctx, " after hip hinge")
    ctx.m.update(yaw_err_deg=yaw, stand_err=along, lateral=lateral, edge_clear=edge)
    ctx.s = s
    return admit_grasp(ctx)
