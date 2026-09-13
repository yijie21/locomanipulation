"""Measured body-height control and a paired staging path before low grasps."""

import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from locomani.tasks.crate.settling import SettlingWindow


class HeightReference:
    """Slew-limited height integral, using fresh observed pelvis height.

    Planner height is not measured pelvis height. Integrating their task error
    accommodates that offset without treating a calibration bias as delivery.
    Stale observations cannot advance the command; planner bounds prevent windup.
    """

    def __init__(self, target, speed, measured, wall):
        self.target, self.speed = target, speed
        self.command = float(np.clip(measured, 0.40, 0.78))
        self.wall = wall

    def step(self, wall, measured):
        dt = wall - self.wall
        if dt <= 0:
            return self.command
        self.wall = wall
        if dt <= 0.1:
            rate = np.clip(self.target - measured, -self.speed, self.speed)
            self.command = float(np.clip(self.command + dt * rate, 0.40, 0.78))
        return self.command


def _phase_crouch(ctx):
    from locomani.tasks.crate.bimanual_carry import OBJ, UP_ABOVE, UP_SHORT, crate_frame, rot_z
    from locomani.tasks.crate.carry_phases import admit_grasp, stand_metrics

    arms, p = ctx.arms, ctx.p
    s = p.obs()
    start_z = float(s["pelvis"]["pos"][2])
    target_z = ctx.config.pelvis_height
    arms.height_control = HeightReference(target_z, ctx.config.height_speed, start_z, s["wall"])
    initial = {side: arms.grip_point(s, side) for side in arms.solver.sides}
    centre, bars, n, _, psi = crate_frame(p, s)
    goals = {side: bars[side] + np.array([*(-n * UP_SHORT), UP_ABOVE]) for side in arms.solver.sides}
    rotations = {
        side: Slerp([0, 1], Rotation.from_matrix([initial[side][1], rot_z(psi)])) for side in arms.solver.sides
    }
    quiet = SettlingWindow(ctx.config.settle_seconds)
    start = time.time()
    ready = False
    while time.time() - start < 20:
        s = p.obs()
        if s["fallen"]:
            ctx.m["why"] = "[crouch] fell while lowering"
            break
        if np.linalg.norm(np.asarray(s["objects"][OBJ]["pos"]) - centre) > 0.05:
            ctx.m["why"] = "[crouch] crate moved more than 5 cm before grasp"
            break
        # Measured descent paces both hand targets. An undelivered height
        # request cannot advance the hands down into the low table.
        u = float(np.clip((start_z - s["pelvis"]["pos"][2]) / max(start_z - target_z, 0.02), 0, 1))
        targets = {side: initial[side][0] * (1 - u) + goals[side] * u for side in arms.solver.sides}
        desired_rotations = {side: rotations[side](u).as_matrix() for side in arms.solver.sides}
        q = arms.solve(s, targets, desired_rotations, from_measured=True, rot_weight=0.25)
        arms.push(q)
        gap = np.max(np.abs(q[12:29] - arms.arm.q[12:29]))
        ready = quiet.step(
            s["wall"],
            s["pelvis"]["pos"],
            s["pelvis"]["quat"],
            reference_ready=abs(s["pelvis"]["pos"][2] - target_z) < 0.015 and gap < 0.02 and arms.motion.quiet,
        )
        if ready:
            break
    s = p.obs()
    ctx.m.update(
        crouch_seconds=time.time() - start,
        pelvis_height_target=target_z,
        pelvis_height_measured=float(s["pelvis"]["pos"][2]),
        planner_height_command=arms.arm.height,
        crouch_ready=ready,
        crouch_hand_errors_m={
            side: float(np.linalg.norm(arms.grip_point(s, side)[0] - goals[side])) for side in arms.solver.sides
        },
    )
    ctx.log(
        f"[crouch] {ctx.m['crouch_seconds']:.1f} s, pelvis {ctx.m['pelvis_height_measured']:.3f} / "
        f"{target_z:.3f} m, command {arms.arm.height:.3f}, ready {ready}"
    )
    if not ready:
        ctx.m["why"] = ctx.m["why"] or "[crouch] height or pose did not settle"
        return ctx.m
    s, yaw, along, lateral, edge = stand_metrics(ctx, " after crouch")
    ctx.m.update(yaw_err_deg=yaw, stand_err=along, lateral=lateral, edge_clear=edge)
    ctx.s = s
    ctx.centre, ctx.bars, ctx.n, ctx.y_hat, ctx.psi = crate_frame(p, s)
    return admit_grasp(ctx)
