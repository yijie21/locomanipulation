"""The imagined Motor with arrival errors and a jaw check (TODO 45.8, user 2026-11-09): kin_motor.py is left as it was,
frozen for shadow mode; this class adds two things on top of it.

- A walk and a reach no longer land exactly. Each draws its error from the errors real rounds showed for similar commands
  (arrival_data.py builds the pool), so one start run K times gives a share of successes: the trend, not one outcome.
- A jaw closes only on what fits: if a finger of the open jaw is already inside a light free body when the grip starts,
  the hand came down on the object, and nothing is held.

A skills package runs on it unchanged, and the errors reach it through what it measures, as they would in a real round.
"""

from __future__ import annotations

import mujoco
import numpy as np
from kin_motor import KinMotor

from locomani.backends.scalebfm import motor as MM
from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import JAW_OPEN, SIDES, WRIST
from locomani.primitives.whole_body_ik import rotvec

K_NEAR = 15  # draw among this many most similar real commands
REACH_SCALE = np.array([0.05, 0.05, 0.05, 0.10, 0.40, 0.03])  # target x, y, z and travel in metres; the wrist's
# pitch; how far off the last reach of that hand ended (a hand that could not get there still cannot, 2026-11-09)
FIT_PEN_M = 0.010  # a finger this far inside the object when the grip starts


class Pool:
    """Real arrival errors, each with the command that produced it."""

    def __init__(self, path: str):
        z = np.load(path)
        self.r_x, self.r_open, self.r_e = z["r_x"], z["r_open"], z["r_e"]
        self.w_dist, self.w_final, self.w_e, self.w_yaw = z["w_dist"], z["w_final"], z["w_e"], z["w_yaw"]

    def reach(self, rng, x: np.ndarray, jaw_open: bool) -> np.ndarray:
        """An error (heading frame, right-hand convention) of a real reach to a similar target."""
        idx = np.flatnonzero(self.r_open == jaw_open)
        d = np.linalg.norm((self.r_x[idx] - x) / REACH_SCALE, axis=1)
        near = idx[np.argpartition(d, min(K_NEAR, len(d) - 1))[:K_NEAR]]
        return self.r_e[near[rng.integers(len(near))]]

    def walk(self, rng, dist: float, final: bool) -> tuple[np.ndarray, float]:
        idx = np.flatnonzero(self.w_final == final)
        near = idx[np.argpartition(np.abs(self.w_dist[idx] - dist), min(K_NEAR, len(idx) - 1))[:K_NEAR]]
        i = near[rng.integers(len(near))]
        return self.w_e[i], float(self.w_yaw[i])


def _rot2(v: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


class ArrivalMotor(KinMotor):
    pool: Pool | None = None  # class level: one world per process, set by make_world
    noise = True
    fit = True
    round_seed = 0

    def _kin_clear(self) -> None:
        super()._kin_clear()
        self.rng = np.random.default_rng(self.round_seed)
        self.arm = {s: dict(e=np.zeros(3), start=np.zeros(3), end=np.zeros(3), goal=None, d0=1.0, u=1.0) for s in SIDES}
        self.walking: dict | None = None
        self.refused: set[str] = set()
        self.grip_pen: list[dict] = []

    # -- arrival errors ----------------------------------------------------------------------------------------------
    def reach(self, side, pos, rot=None, **kwargs):
        if self.noise and self.pool is not None and side != "both":
            self._plan_arm(side, np.asarray(pos, float), rot)
        return super().reach(side, pos, rot, **kwargs)

    def _plan_arm(self, side: str, goal: np.ndarray, rot) -> None:
        p0, _ = self.pelvis()
        yaw = self.yaw()
        jaw, Rw = self.jaw_centre(side)
        sign = -1.0 if side == SIDES[0] else 1.0  # the pool is kept as for the right hand
        t = _rot2(goal[:2] - p0[:2], -yaw)
        a = self.arm[side]
        travel = float(np.linalg.norm(goal - (jaw - a["e"])))
        pitch = float((Rw if rot is None else np.asarray(rot, float))[2, 0])
        x = np.array([t[0], sign * t[1], goal[2] - p0[2], travel, pitch, float(np.linalg.norm(a["e"]))])
        e = self.pool.reach(self.rng, x, abs(float(self.jaw[SIDES.index(side)]) - JAW_OPEN) < 1e-3)
        xy = _rot2(np.array([e[0], sign * e[1]]), yaw)
        a.update(start=a["e"].copy(), end=np.array([xy[0], xy[1], e[2]]), goal=goal, d0=max(travel, 1e-3), u=0.0)

    def walk_to(self, xy, yaw=None, *, speed=None, face="travel"):
        if self.noise and self.pool is not None:
            goal = np.asarray(xy, float)
            delta = goal - self.sim.d.qpos[:2]
            dist = float(np.linalg.norm(delta))
            end_yaw = (
                float(yaw)
                if yaw is not None
                else (float(np.arctan2(delta[1], delta[0])) if dist > 0.05 else self.yaw())
            )
            e, yaw_err = self.pool.walk(self.rng, dist, yaw is not None)
            self.walking = dict(goal=goal, d0=max(dist, 1e-3), e=_rot2(e, end_yaw), yaw=yaw_err, u=0.0)
        try:
            return super().walk_to(xy, yaw, speed=speed, face=face)
        finally:
            self.walking = None

    def _apply_walk(self) -> None:
        d, w = self.sim.d, self.walking
        nominal = d.qpos[:2] - self.servo.get("walk_push", np.zeros(2))  # the stop servo's work is in the sampled error
        w["u"] = max(w["u"], float(np.clip(1.0 - np.linalg.norm(w["goal"] - nominal) / w["d0"], 0.0, 1.0)))
        d.qpos[:2] = nominal + w["u"] * w["e"]
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, MM.yaw_quat(w["u"] * w["yaw"]), d.qpos[3:7].copy())
        d.qpos[3:7] = q

    def _apply_arms(self) -> None:
        m, d = self.sim.m, self.sim.d
        mujoco.mj_kinematics(m, d)
        for s in SIDES:
            a = self.arm[s]
            if s in self.hand_override:
                continue
            if a["goal"] is not None:
                jaw, _ = self.jaw_centre(s)
                a["u"] = max(a["u"], float(np.clip(1.0 - np.linalg.norm(jaw - a["goal"]) / a["d0"], 0.0, 1.0)))
                a["e"] = (1.0 - a["u"]) * a["start"] + a["u"] * a["end"]
            if np.linalg.norm(a["e"]) > 1e-3:
                self._shift_wrist(s, a["e"])

    def _shift_wrist(self, side: str, e: np.ndarray) -> None:
        """A few damped least-squares steps of that arm alone: the wrist moved by `e`, its attitude kept."""
        m, d = self.sim.m, self.sim.d
        b = self.sim.bid[WRIST[side]]
        target = d.xpos[b] + e
        Rw0 = d.xmat[b].reshape(3, 3).copy()
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        dofs, adr = self.arm_dofs[side], self.kp.qadr[self.arm_joints[side]]
        lo, hi = self._range(self.arm_joints[side])
        for _ in range(3):
            mujoco.mj_comPos(m, d)
            mujoco.mj_jacBody(m, d, jp, jr, b)
            Rw = d.xmat[b].reshape(3, 3)
            err = np.r_[target - d.xpos[b], 0.3 * rotvec(Rw0 @ Rw.T)]
            J = np.r_[jp[:, dofs], 0.3 * jr[:, dofs]]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), err)
            d.qpos[adr] = np.clip(d.qpos[adr] + np.clip(dq, -0.15, 0.15), lo, hi)
            mujoco.mj_kinematics(m, d)

    def _fingers(self) -> None:
        # tick() calls this right after writing the reference pose, and before contacts: where a pose error belongs
        if self.noise and self.pool is not None:
            if self.walking is not None:
                self._apply_walk()
            self._apply_arms()
        super()._fingers()

    # -- the jaw -----------------------------------------------------------------------------------------------------
    def grip(self, side: str, *, seconds: float | None = None, target: float | None = None):
        self.refused.discard(side)
        if self.fit:
            m, d = self.sim.m, self.sim.d
            fingers = set(self.fingers[side])
            light = self.free_bodies - self.heavy
            deep = 0.0
            for i in range(d.ncon):
                b1, b2 = int(m.geom_bodyid[d.contact[i].geom1]), int(m.geom_bodyid[d.contact[i].geom2])
                if (b1 in fingers and b2 in light) or (b2 in fingers and b1 in light):
                    deep = max(deep, -float(d.contact[i].dist))
            self.grip_pen.append(dict(t=round(self.t, 2), side=side, pen_cm=round(100 * deep, 2)))
            if deep > FIT_PEN_M:
                self.refused.add(side)
                self.kin_log.append(
                    dict(t=round(self.t, 2), event="does_not_fit", side=side, pen_cm=round(100 * deep, 2))
                )
        return super().grip(side, seconds=seconds, target=target)

    def between_fingers(self, side: str) -> set[int]:
        return set() if side in self.refused else super().between_fingers(side)


def make_world(fam: dict, pool_path: str | None, noise: bool, fit: bool):
    """As imagine.make_world, on this Motor."""
    from locomani.backends.scalebfm.config import ScaleBfmBackendConfig
    from locomani.tasks.scene import world as W

    MM.FF_SHORT = (0.0, 0.0, 0.0)
    ArrivalMotor.pool = Pool(pool_path) if pool_path and noise else None
    ArrivalMotor.noise, ArrivalMotor.fit = noise, fit
    W.Motor = ArrivalMotor
    policy = R.Policy("cpu", ScaleBfmBackendConfig())
    return W.SceneWorld(fam["scene_dir"], policy, controller="scalebfm", sleep=False)
