"""Geometric leg odometry (2026-11-03, after the walk model's post-walk readings turned out to decode the controller's
true-position input): where the pelvis went, from the robot's own readings only, with nothing learned.

Per 50 Hz motor tick it reads the joint encoders, the joint torques (the motors' own torque readings), the gyro and the
gravity reading. Through the robot's own kinematics (a private MjData, the pelvis at the origin) the encoders give both
ankles in the pelvis frame; the gravity reading levels them and the gyro, integrated about the vertical, gives the heading.
The torques give how much each foot carries: each leg's 6 torques through the transpose of its ankle Jacobian give the
force at the ankle, and its vertical part (sign flipped) is the load. A foot on the floor stays put, so the pelvis moves
by minus the change of that foot's offset. Which foot: the lower one, switching when the other gets lower by switch_m
(mode "height"), or the loads as weights (mode "load"). ScaleBFM shuffles (a swing foot lifts 3-11 mm) and its feet slide
1-9 cm while loaded (2026-11-03 probe), so the anchor drifts; on 60 recorded eval walks (record(), variants()) the best
was height, the sole point, 2 mm: 1.31 cm median against 2.06 for dead reckoning; every load-weighted variant read the
walk 1.3-1.9 cm short. The simulator's truth is never read (record() keeps it for scoring only).

Moved from reachwm_20261030/leg_odo.py (TODO 49.2): LegOdometry, TickFeed, JawFeed, OdoRoot, level. The probe / record /
variants tools (they need the selfwm Env) stayed behind; the two record-only environment switches (ODO_LOAD_POW, JAW_FEED_RAW)
are now the constant LOAD_POW = 2 and no raw record.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np

from locomani.paths import REPO

SIDES = ("left", "right")
_JN = json.load(open(Path(REPO) / "assets" / "scalebfm" / "model_22200_tensorrt_metadata.json"))["joint_names"]
LEGS = {
    s: [i for i, n in enumerate(_JN) if n.startswith(s) and any(k in n for k in ("hip", "knee", "ankle"))]
    for s in SIDES
}
LOAD_POW = 2.0  # (was env ODO_LOAD_POW, default 2) weights ~ load^LOAD_POW: the carrying foot dominates


def level(grav: np.ndarray) -> np.ndarray:
    """Body -> level frame (rows: forward projected on the floor, left, up) from the gravity reading in the body frame."""
    up = -np.asarray(grav, float)
    up /= np.linalg.norm(up)
    x = np.array([1.0, 0.0, 0.0]) - up[0] * up
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(up, x), up])


class LegOdometry:
    def __init__(self, model: mujoco.MjModel, qadr: np.ndarray, vadr: np.ndarray, point: str = "sole",
                 mode: str = "height", pow_: float = LOAD_POW, switch_m: float = 0.002):  # fmt: skip
        """qadr, vadr: the qpos and dof address of each of the 29 joints in the controller's order (runtime.Sim).
        point: the foot point taken as staying put, the ankle or the sole (centre of the foot's flat box or of its contact spheres); mode: "load"
        (weights ~ load^pow_, pow_ inf = the more loaded foot only) or "height" (the lower foot, switching when the other
        is lower by switch_m)."""
        assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE and model.jnt_qposadr[0] == 0, (
            "the pelvis free joint first"
        )
        self.m, self.d = model, mujoco.MjData(model)
        self.qadr, self.vadr = np.asarray(qadr), np.asarray(vadr)
        self.point, self.mode, self.pow, self.switch_m = point, mode, pow_, switch_m
        self.ank = {s: model.body(f"{s}_ankle_roll_link").id for s in SIDES}
        self.sole = {s: self._sole(model, self.ank[s]) for s in SIDES}
        self.jp, self.jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))

    @staticmethod
    def _sole(model, body: int) -> list:
        """The foot's sole geoms: its flat box, or on a model with point feet (SIMPLE's G1) its contact spheres."""
        on = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body]
        box = [g for g in on if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
        return box[:1] or [
            g for g in on if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and model.geom_contype[g]
        ]

    def feet(self, q, tau, grav):
        """Both ankles in the level frame (2, 3) and their loads (2,) in N."""
        d = self.d
        d.qpos[:] = self.m.qpos0
        d.qpos[0:3], d.qpos[3:7] = 0.0, (1.0, 0.0, 0.0, 0.0)
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        L = level(grav)
        f, load = np.zeros((2, 3)), np.zeros(2)
        for k, s in enumerate(SIDES):
            b = self.ank[s]
            f[k] = L @ (d.xpos[b] if self.point == "ankle" else d.geom_xpos[self.sole[s]].mean(axis=0))
            mujoco.mj_jacBody(self.m, d, self.jp, self.jr, b)
            J = np.r_[self.jp, self.jr][:, self.vadr[LEGS[s]]]
            w = np.linalg.lstsq(J.T, np.asarray(tau, float)[LEGS[s]], rcond=None)[0]
            load[k] = max(0.0, -(L @ w[:3])[2])
        return f, load

    def _rot(self) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    @staticmethod
    def _rate(gyro, grav) -> float:
        return float(-np.asarray(grav, float) @ np.asarray(gyro, float) / np.linalg.norm(grav))

    def start(self, q, tau, gyro, grav) -> None:
        self.yaw, self.p, self.last_rate = 0.0, np.zeros(3), self._rate(gyro, grav)
        f, _ = self.feet(q, tau, grav)
        self.prev = (self._rot() @ f.T).T
        self.stance = int(np.argmin(self.prev[:, 2]))

    def step(self, q, tau, gyro, grav, dt: float) -> None:
        rate = self._rate(gyro, grav)
        self.yaw += 0.5 * (rate + self.last_rate) * dt
        self.last_rate = rate
        f, load = self.feet(q, tau, grav)
        fw = (self._rot() @ f.T).T
        if self.mode == "height":
            o = 1 - self.stance
            if fw[o][2] < fw[self.stance][2] - self.switch_m:
                self.stance = o
            w = np.eye(2)[self.stance]
        elif math.isinf(self.pow):
            w = np.eye(2)[int(np.argmax(load))]
        else:
            w = load**self.pow
            w = w / w.sum() if w.sum() > 0 else np.full(2, 0.5)
        self.p = self.p - (w[:, None] * (fw - self.prev)).sum(0)
        self.prev = fw

    def pose(self) -> np.ndarray:
        """(forward m, left m, turn rad) in the frame of the pelvis at start()."""
        return np.r_[self.p[:2], self.yaw]


class TickFeed:
    """Feeds a LegOdometry with the robot's readings at every 50 Hz motor tick while `on` (wraps Motor.tick, as
    gen_reach.Recorder does). `truth` keeps the true pelvis rows for checks; the odometry never reads them."""

    def __init__(self, env, truth: bool = False):
        self.env, self.on, self.truth, self.rows = env, False, truth, []
        self.odo = LegOdometry(env.sim.m, env.sim.qadr, env.sim.vadr)
        m = env.m
        orig = m.tick

        def tick():
            ok = orig()
            if ok and self.on:
                self.odo.step(*self.read(), 0.02)
                if self.truth:
                    p, Rp = env.pelvis()
                    self.rows.append(np.r_[p, math.atan2(Rp[1, 0], Rp[0, 0]), self.odo.pose()])
            return ok

        m.tick = tick

    def read(self):
        sim = self.env.sim
        st = sim.state()
        return st["dof_pos"], sim.d.qfrc_actuator[sim.vadr].copy(), st["ang_vel"], self.env.self_state()[58:61]

    def begin(self) -> None:
        self.odo.start(*self.read())
        self.rows, self.on = [], True


class JawFeed(TickFeed):
    """The jaw pose in the world from joint readings, the gravity and gyro readings and leg odometry: the pelvis is
    anchored where a caller says it is (room_grasp: where the goals are anchored after the walk, so the anchor cancels),
    then followed by the odometry; the wrist comes from the arm's forward kinematics. Motor.jaw_source = feed.jaw makes
    the reach servo measure this instead of the simulator's wrist body."""

    @staticmethod
    def heading_rate(gyro, grav) -> float:
        """Rate of the heading of the body's forward axis. LegOdometry._rate (the vertical part of the gyro reading) counts
        a roll about a pitched forward axis as a turn: 9 deg off after a crouching reach."""
        up = -np.asarray(grav, float) / np.linalg.norm(grav)
        g = np.asarray(gyro, float)
        return float((up[1] * g[1] + up[2] * g[2]) / (up[1] ** 2 + up[2] ** 2))

    def anchor(self, p, yaw: float, wrist: dict, off) -> None:
        # standing reaches: both feet carry, so weight them by load (12 recorded reaches, pelvis error at the end, median:
        # lower foot only 3.4 cm, load-weighted 0.3 cm, both feet taken as fixed 0.35 cm but 24 cm when one steps)
        if self.odo.mode != "load":
            self.odo = LegOdometry(self.env.sim.m, self.env.sim.qadr, self.env.sim.vadr, mode="load", pow_=1.0)
        self.odo._rate = self.heading_rate
        self.p0, self.yaw0, self.off = np.asarray(p, float).copy(), float(yaw), np.asarray(off, float)
        self.wb = {s: self.odo.m.body(n).id for s, n in wrist.items()}
        self.begin()

    def pelvis(self):
        o = self.odo
        c, s = math.cos(self.yaw0), math.sin(self.yaw0)
        R0 = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return self.p0 + R0 @ o.p, R0 @ o._rot() @ level(self.read()[3])

    def jaw(self, side: str):
        o = self.odo
        pb, Rb = self.pelvis()
        b = self.wb[side]
        Rw = Rb @ o.d.xmat[b].reshape(3, 3)
        return pb + Rb @ o.d.xpos[b] + Rw @ self.off, Rw


class OdoRoot:
    """The pelvis position ScaleBFM is told, from leg odometry (env.Env with BFM_ROOT=odo). Its input is the targets minus
    this position, so one frame is enough: every time the motor anchors its stand reference to where the pelvis is (reset,
    the end of a walk) the odometry restarts there, and the policy sees that anchor plus the odometry since. The offset the
    anchor reads cancels in the targets; a robot would use its own belief there."""

    def __init__(self, env):
        self.env, self.sim, self.odo, self.anchor = (
            env,
            env.sim,
            LegOdometry(env.sim.m, env.sim.qadr, env.sim.vadr),
            None,
        )
        self.pending, m, orig = False, env.m, env.m._stand

        def stand(pose=None):
            orig(pose)
            if pose is None:
                self.pending = True

        m._stand = stand
        m.pol.root_source = self

    def read(self, st):
        sim = self.sim
        return st["dof_pos"], sim.d.qfrc_actuator[sim.vadr].copy(), st["ang_vel"], self.env.self_state()[58:61]

    def __call__(self, st, targets) -> np.ndarray:
        if self.pending:
            self.pending = False
            self.odo.start(*self.read(st))
            p, Rp = self.env.pelvis()
            c, s = math.cos(yaw := math.atan2(Rp[1, 0], Rp[0, 0])), math.sin(yaw)
            self.anchor, self.R0 = p.copy(), np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            return self.anchor.copy()
        if self.anchor is None:
            return st["root_pos"]
        self.odo.step(*self.read(st), 0.02)
        return self.anchor + self.R0 @ self.odo.p
