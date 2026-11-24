"""A Motor that imagines instead of simulating (TODO 45, user 2026-11-04 proposal 3, "开始" 2026-11-08).

Same interface and the same logic as `Motor` (IK, timing, reach and walk outcomes, the trace), so a skills package runs
on it unchanged; only `tick` differs. It takes no physics step and calls no controller: the robot is written at the
reference pose, the scene's own geometry is queried, and three rules stand in for contact:

- an arm that would sink into fixed scenery stays where it was (the hand then misses, as it does in a real round);
- a free object the robot sinks into slides out of the way in the plane (a chair pushed, a mug knocked);
- fingers close until they touch; two fingers on one free body hold it rigidly to the wrist until the jaw opens.

Heavy free bodies give way to an arm only a little and then stop it. Nothing here is learned. It is the floor to compare a learned body and contact model against.
"""

from __future__ import annotations

import mujoco
import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import FINGERS, JAW_OPEN, SIDES, WRIST, Motor, Outcome
from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES
from locomani.primitives.whole_body_ik import rotvec

STRIDE = 5  # primitives advance 0.1 s per tick here; a skill that calls tick() itself still gets 0.02 s
BLOCK_M = 0.05  # penetration of an arm into fixed scenery past which the arm does not advance; at 0.015 the reference
# through a shelf edge (3.7 cm deep) stopped arms the real rounds got past: 75 false failures on grasp_atomic, 2026-11-08
BLOCK_FREE_M = (
    0.015  # the same against a free body (a chair after it has given way, an object a hand comes down on): at 0.05
)
# side_reach went from 33 of 38 seen to 8 of 38 on grasp_floor
PUSH_MIN_M = 0.001  # penetration below this moves nothing
HEAVY_KG = (
    1.5  # a free body at least this heavy gives way to an arm only HEAVY_GIVE_M, then stops it as fixed scenery does
)
HEAVY_GIVE_M = (
    0.08  # real rounds where an arm ran into a 5 kg chair and missed: chair moved median 6.8, max 8.8 cm (2026-11-08)
)


class KinMotor(Motor):
    def __init__(self, sim, policy, config=None, *, model_xml=None, track=(), jacobian="analytic"):
        super().__init__(sim, policy, config, model_xml=model_xml, track=track, jacobian=jacobian)
        m = sim.m
        root = m.body("pelvis").id
        self.robot_bodies = {b for b in range(m.nbody) if m.body_rootid[b] == root}
        self.free_bodies = {
            b
            for b in range(m.nbody)
            if b not in self.robot_bodies
            and m.body_jntnum[b] == 1
            and m.jnt_type[m.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE
        }
        self.heavy = {b for b in self.free_bodies if float(m.body_subtreemass[b]) >= HEAVY_KG}
        self.arm_joints, self.arm_dofs, self.arm_bodies = {}, {}, {}
        for s in SIDES:
            idx = [i for i, n in enumerate(MUJOCO_JOINT_NAMES) if n.startswith(s) and _is_arm(n)]
            assert len(idx) == 7, (s, idx)
            self.arm_joints[s] = np.array(idx)
            self.arm_dofs[s] = np.array([m.jnt_dofadr[m.joint(MUJOCO_JOINT_NAMES[i]).id] for i in idx])
            self.arm_bodies[s] = {b for b in self.robot_bodies if _is_arm_body(m.body(b).name, s)}
        self.waist_joints = np.array([i for i, n in enumerate(MUJOCO_JOINT_NAMES) if "waist" in n])
        self.waist_dofs = np.array([m.jnt_dofadr[m.joint(MUJOCO_JOINT_NAMES[i]).id] for i in self.waist_joints])
        self.finger_bodies = {b: (i, j) for i, s in enumerate(SIDES) for j, b in enumerate(self.fingers[s])}
        self.finger_lo = np.array([m.jnt_range[m.dof_jntid[v]][0] for v in sim.gv])
        self.finger_hi = np.array([m.jnt_range[m.dof_jntid[v]][1] for v in sim.gv])
        self.finger_joint = [int(m.dof_jntid[v]) for v in sim.gv]
        self._kin_clear()

    def _kin_clear(self) -> None:
        self.held: dict = {}  # side -> (body, position in the wrist frame, quaternion in the wrist frame, rest height)
        self.finger_stopped = np.zeros(4, bool)
        self.finger_closing = np.zeros(4, bool)
        self.grips: list[dict] = []  # one snapshot per grip call: what contact_gen.py replays on the real Motor
        self.kin_log: list[dict] = []
        self._pushed: dict = {}

    def reset(self, xy, yaw: float, *, warm_up: int = 0) -> None:
        super().reset(xy, yaw, warm_up=0)
        self._kin_clear()
        self.sim.d.qvel[:] = 0

    # -- geometry ---------------------------------------------------------------------------------------------------
    def _contacts(self) -> None:
        m, d = self.sim.m, self.sim.d
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_collision(m, d)

    def _free_adr(self, b: int) -> int:
        return int(self.sim.m.jnt_qposadr[self.sim.m.body_jntadr[b]])

    def _carry(self) -> None:
        """Held objects follow their wrists."""
        if not self.held:
            return
        m, d = self.sim.m, self.sim.d
        mujoco.mj_kinematics(m, d)
        for s, (b, rel_p, rel_q, _) in self.held.items():
            wp, wq = self.sim.body(WRIST[s])
            adr = self._free_adr(b)
            d.qpos[adr : adr + 3] = wp + R.mat(wq) @ rel_p
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, wq, rel_q)
            d.qpos[adr + 3 : adr + 7] = q

    def _fingers(self) -> None:
        """Each finger joint follows the jaw command until it has touched something; opening frees it and lets go."""
        d = self.sim.d
        gq = self.sim.gq
        target = np.clip(np.repeat(self.jaw, 2), self.finger_lo, self.finger_hi)
        self.finger_closing = target < d.qpos[gq] - 1e-6
        for j in range(4):
            if target[j] > d.qpos[gq[j]] + 1e-6:
                self.finger_stopped[j] = False
                d.qpos[gq[j]] = target[j]
                self._let_go(SIDES[j // 2])
            elif not self.finger_stopped[j]:
                d.qpos[gq[j]] = target[j]

    def _let_go(self, side: str) -> None:
        got = self.held.pop(side, None)
        if got is None:
            return
        d = self.sim.d
        adr = self._free_adr(got[0])
        d.qpos[adr + 2] = min(float(d.qpos[adr + 2]), got[3])  # back to the height it was taken from: no drop physics
        self.kin_log.append(dict(t=round(self.t, 2), event="let_go", side=side, body=self.sim.m.body(got[0]).name))

    def _maybe_grasp(self) -> None:
        for i, s in enumerate(SIDES):
            if s in self.held or self.jaw[i] >= JAW_OPEN - 1e-3:
                continue
            both = self.between_fingers(s) & self.free_bodies
            if not both:
                continue
            b = sorted(both)[0]
            wp, wq = self.sim.body(WRIST[s])
            op, oq = self.sim.d.xpos[b].copy(), self.sim.d.xquat[b].copy()
            inv, rel = np.zeros(4), np.zeros(4)
            mujoco.mju_negQuat(inv, wq)
            mujoco.mju_mulQuat(rel, inv, oq)
            self.held[s] = (b, R.mat(wq).T @ (op - wp), rel, float(op[2]))
            self.kin_log.append(dict(t=round(self.t, 2), event="grasp", side=s, body=self.sim.m.body(b).name))

    def _settle_contacts(self) -> set[str]:
        """Apply the contact rules to the contacts just computed; the sides whose arm sank into fixed scenery."""
        m, d = self.sim.m, self.sim.d
        blocked: dict[str, tuple[float, int]] = {}
        push: dict[int, np.ndarray] = {}
        held = {v[0] for v in self.held.values()}
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if (b1 in self.robot_bodies) == (b2 in self.robot_bodies):
                continue
            rb, ob, sign = (b1, b2, 1.0) if b1 in self.robot_bodies else (b2, b1, -1.0)
            if rb in self.finger_bodies and c.dist < 0:
                side, j = self.finger_bodies[rb]
                jj = 2 * side + j
                # a finger stops only on what it closes into; what it brushed on the way down does not jam it
                if self.finger_closing[jj] and float(-d.xaxis[self.finger_joint[jj]] @ (sign * c.frame[:3])) > 0.3:
                    self.finger_stopped[jj] = True
            if ob in held or c.dist >= -PUSH_MIN_M:
                continue
            arm = any(rb in self.arm_bodies[s] for s in SIDES)
            gives = ob in self.free_bodies and (
                ob not in self.heavy or not arm or self._pushed.get(ob, 0.0) < HEAVY_GIVE_M
            )
            if gives:
                closing = (
                    rb in self.finger_bodies
                    and self.finger_closing[2 * self.finger_bodies[rb][0] + self.finger_bodies[rb][1]]
                )
                if closing:
                    continue  # closing fingers do not shove what they close on
                n = sign * np.asarray(c.frame[:3], float)
                flat = float(np.linalg.norm(n[:2]))
                if flat >= 0.3:
                    step = -float(c.dist) * n[:2] / flat
                    if ob not in push or np.linalg.norm(step) > np.linalg.norm(push[ob]):
                        push[ob] = step
                    continue
                # pressing down: the object cannot go anywhere, so a hand coming down on it stops
            limit = BLOCK_FREE_M if ob in self.free_bodies else BLOCK_M
            for s in SIDES:
                if rb in self.arm_bodies[s] and c.dist < -limit and -c.dist > blocked.get(s, (0.0, 0))[0]:
                    blocked[s] = (-float(c.dist), ob)
        for b, step in push.items():
            adr = self._free_adr(b)
            d.qpos[adr : adr + 2] += step
            self._pushed[b] = self._pushed.get(b, 0.0) + float(np.linalg.norm(step))
        for s, (depth, ob) in blocked.items():
            self.kin_log.append(dict(t=round(self.t, 2), event="blocked", side=s, by=m.body(ob).name,
                                     depth_cm=round(100 * depth, 1)))  # fmt: skip
        return set(blocked)

    def _drive_wrist(self, side: str) -> None:
        """`hand` drives the wrist itself: a few damped least-squares steps of that arm and the waist toward it (the
        controller bends the waist to follow a wrist the arm alone cannot reach)."""
        m, d = self.sim.m, self.sim.d
        jaw, rot, _ = self.hand_override[side]
        target = np.asarray(jaw, float) - np.asarray(rot, float) @ R.OFF
        b = self.sim.bid[WRIST[side]]
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        joints = np.r_[self.arm_joints[side], self.waist_joints]
        dofs = np.r_[self.arm_dofs[side], self.waist_dofs]
        adr = self.kp.qadr[joints]
        lo, hi = self._range(joints)
        for _ in range(8):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            mujoco.mj_jacBody(m, d, jp, jr, b)
            Rw = d.xmat[b].reshape(3, 3)
            e = np.r_[target - d.xpos[b], 0.3 * rotvec(np.asarray(rot, float) @ Rw.T)]
            J = np.r_[jp[:, dofs], 0.3 * jr[:, dofs]]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e)
            d.qpos[adr] = np.clip(d.qpos[adr] + np.clip(dq, -0.15, 0.15), lo, hi)

    def _range(self, joints) -> tuple[np.ndarray, np.ndarray]:
        m = self.sim.m
        r = np.array([m.jnt_range[m.joint(MUJOCO_JOINT_NAMES[i]).id] for i in joints])
        return r[:, 0], r[:, 1]

    def grip(self, side: str, *, seconds: float | None = None, target: float | None = None) -> Outcome:
        m, d = self.sim.m, self.sim.d
        p, _ = self.pelvis()
        free = {m.body(b).name: [round(float(v), 5) for v in np.r_[d.xpos[b], d.xquat[b]]] for b in self.free_bodies}
        self.grips.append(dict(t=round(self.t, 2), side=side, pelvis=[round(float(v), 4) for v in p], yaw=round(self.yaw(), 4),
                               free=free, n_calls=len(self.calls)))  # fmt: skip
        return super().grip(side, seconds=seconds, target=target)

    # -- the tick ---------------------------------------------------------------------------------------------------
    def tick(self) -> bool:
        sim = self.sim
        d = sim.d
        if sim.fallen():
            self.fell_at = self.t if self.fell_at is None else self.fell_at
            return False
        pose = self.ref(self.t)
        before = {s: d.qpos[self.kp.qadr[self.arm_joints[s]]].copy() for s in SIDES}
        driven = np.zeros(29, bool)
        for s in self.hand_override:
            driven[self.arm_joints[s]] = True
            driven[self.waist_joints] = True
        d.qpos[:3], d.qpos[3:7] = pose.pos, pose.quat
        d.qpos[self.kp.qadr[~driven]] = np.asarray(pose.joints)[~driven]
        self._fingers()
        for s in list(self.hand_override):
            self._drive_wrist(s)
        self._carry()
        self._contacts()
        blocked = self._settle_contacts()
        if blocked:
            for s in blocked:
                d.qpos[self.kp.qadr[self.arm_joints[s]]] = before[s]
            self._carry()
            self._contacts()
        self._maybe_grasp()
        kp = self.kp(pose.pos, pose.quat, pose.joints)
        fut = {n: (np.asarray(p)[None], np.asarray(q)[None]) for n, (p, q) in kp.items()}
        for s, (jaw, rot, _) in self.hand_override.items():
            fut[WRIST[s]] = ((np.asarray(jaw) - np.asarray(rot) @ R.OFF)[None], R.quat_wxyz(rot)[None])
        self.trace.append(
            dict(
                t=self.t,
                primitive=self.primitive,
                mode=self.mode,
                robot=d.qpos[:7].copy(),
                joints=self.joints(),
                ref_pelvis=np.asarray(pose.pos).copy(),
                ref_wrists=np.array([fut[WRIST[s]][0][0] for s in SIDES]),
                wrists=np.array([sim.body(WRIST[s])[0] for s in SIDES]),
                jaw=self.jaw.copy(),
                fingers=d.qpos[sim.gq].copy(),
                objects={n: np.r_[d.xpos[b], d.xquat[b]] for n, b in self.track.items()},
            )
        )
        self.last_fut = fut
        self.k += 1
        return True

    def _run(self, name: str, update, timeout: float) -> Outcome:
        self.primitive, t0 = name, self.t
        while self.t - t0 < timeout:
            out = update(self.t)
            if out is not None:
                out.t0, out.t1 = t0, self.t
                return out
            self.k += STRIDE - 1
            if not self.tick():
                return Outcome(False, f"fell at {self.fell_at:.2f} s", t0, self.t)
        return Outcome(False, "timeout", t0, self.t)


def _is_arm(joint: str) -> bool:
    return any(k in joint for k in ("shoulder", "elbow", "wrist"))


def _is_arm_body(name: str, side: str) -> bool:
    return name.startswith(side) and any(k in name for k in ("shoulder", "elbow", "wrist", "dex1", "hand", "gripper"))


__all__ = ["KinMotor", "FINGERS"]
