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

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    $PY scripts/experiments/reachwm_20261030/leg_odo.py probe [N]    # eval-like walks in the sim at 50 Hz, against truth
    $PY scripts/experiments/reachwm_20261030/leg_odo.py record [N]   # the same walks' readings -> approach/leg_odo_rec.npz
    $PY scripts/experiments/reachwm_20261030/leg_odo.py variants     # odometry variants on them -> leg_odo_variants.json
approach_eval.py's post_odo runs it during the eval walks.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "selfwm_20261029"))
sys.path.insert(0, str(HERE))
import env as E  # noqa: E402

SIDES = ("left", "right")
_JN = json.load(open(E.REPO / "assets" / "scalebfm" / "model_22200_tensorrt_metadata.json"))["joint_names"]
LEGS = {
    s: [i for i, n in enumerate(_JN) if n.startswith(s) and any(k in n for k in ("hip", "knee", "ankle"))]
    for s in SIDES
}
LOAD_POW = float(os.environ.get("ODO_LOAD_POW", "2"))  # weights ~ load^LOAD_POW: the carrying foot dominates


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
        self.raw = [] if os.environ.get("JAW_FEED_RAW") else None
        self.p0, self.yaw0, self.off = np.asarray(p, float).copy(), float(yaw), np.asarray(off, float)
        self.wb = {s: self.odo.m.body(n).id for s, n in wrist.items()}
        self.begin()

    def read(self):
        r = super().read()
        if getattr(self, "raw", None) is not None and self.on:
            p, Rp = self.env.pelvis()
            self.raw.append(np.r_[r[0], r[1], r[2], r[3], p, math.atan2(Rp[1, 0], Rp[0, 0])])
        return r

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


def probe(n: int = 30) -> None:
    """Eval-like open walks (approach_eval.case / direct_walk) at 50 Hz: leg odometry against the truth."""
    import approach_eval as AE
    import gen_reach as GR

    env = E.Env("open")
    feed = TickFeed(env, truth=True)
    env.reset(None)
    env.m.hold(1.0)
    e_dr, e_odo, t_dr, t_odo = [], [], [], []
    for i in range(n):
        c = AE.case(i)
        m = env.m
        env.reset(None)
        if c["walked"]:
            m.walk_to(np.array(c["walk0"][:2]), yaw=float(c["walk0"][2]))
        m.hold(GR.PRE_S)
        p0, Rp0 = env.pelvis()
        yaw0 = E.yaw_of(Rp0)
        walk = AE.direct_walk(c)
        feed.begin()
        GR.open_walk(m, float(walk[0]), float(walk[1]), float(walk[2]))
        m.hold(GR.PRE_S)
        feed.on = False
        p1, Rp1 = env.pelvis()
        d1 = E.rot_z(yaw0).T @ (p1 - p0)
        true = np.r_[d1[:2], (E.yaw_of(Rp1) - yaw0 + math.pi) % (2 * math.pi) - math.pi]
        est = feed.odo.pose()
        e_dr.append(100 * np.hypot(*(true[:2] - walk[:2])))
        e_odo.append(100 * np.hypot(*(est[:2] - true[:2])))
        t_dr.append(abs(math.degrees(true[2])))
        t_odo.append(abs(math.degrees(est[2] - true[2])))
        print(f"case {i}: walk {walk[0]:+.2f} m, truth {100 * true[0]:+.1f}/{100 * true[1]:+.1f} cm, odometry "
              f"{100 * est[0]:+.1f}/{100 * est[1]:+.1f} cm, error {e_odo[-1]:.2f} cm (dead reckoning {e_dr[-1]:.2f}), "
              f"turn err {t_odo[-1]:.2f} deg", flush=True)  # fmt: skip
    e_dr, e_odo = np.array(e_dr), np.array(e_odo)
    print(f"{n} walks at 50 Hz: dead reckoning median {np.median(e_dr):.2f} cm (p90 {np.percentile(e_dr, 90):.2f}); "
          f"leg odometry median {np.median(e_odo):.2f} cm (p90 {np.percentile(e_odo, 90):.2f}); turn {np.median(t_dr):.2f} -> "
          f"{np.median(t_odo):.2f} deg; odometry better in {int((e_odo < e_dr).sum())} of {n}", flush=True)  # fmt: skip


def record(n: int = 40) -> None:
    """Eval-like walks (approach_eval.case / direct_walk): the 50 Hz readings the odometry may use (encoders, torques,
    gyro, gravity) plus, for scoring only, the true pelvis pose; saved for offline variants (variants())."""
    import approach_eval as AE
    import gen_reach as GR

    env = E.Env("open")
    sim, m = env.sim, env.m
    rows, cur = [], {"on": False, "r": []}
    orig = m.tick

    def snap():
        st = sim.state()
        p, Rp = env.pelvis()
        cur["r"].append(np.r_[st["dof_pos"], sim.d.qfrc_actuator[sim.vadr], st["ang_vel"], env.self_state()[58:61],
                              p, math.atan2(Rp[1, 0], Rp[0, 0])])  # fmt: skip

    def tick():
        ok = orig()
        if ok and cur["on"]:
            snap()
        return ok

    m.tick = tick
    env.reset(None)
    m.hold(1.0)
    walks, segs = [], []
    for i in range(n):
        c = AE.case(i)
        env.reset(None)
        if c["walked"]:
            m.walk_to(np.array(c["walk0"][:2]), yaw=float(c["walk0"][2]))
        m.hold(GR.PRE_S)
        walk = AE.direct_walk(c)
        cur["r"], cur["on"] = [], True
        snap()  # the state at the sighting, without stepping the simulation
        GR.open_walk(m, float(walk[0]), float(walk[1]), float(walk[2]))
        m.hold(GR.PRE_S)
        cur["on"] = False
        segs.append(np.array(cur["r"], np.float32))
        walks.append(walk)
        print(f"recorded case {i}: {len(segs[-1])} ticks", flush=True)
    np.savez_compressed(OUT_REC, walks=np.array(walks), n=np.array([len(x) for x in segs]), rows=np.concatenate(segs))


OUT_REC = E.REPO / "outputs" / "reachwm_20261030" / "approach" / "leg_odo_rec.npz"
# a recorded row: q 0:29, tau 29:58, gyro 58:61, gravity 61:64 | truth: pelvis xyz 64:67, yaw 67


def replay(odo: "LegOdometry", R: np.ndarray) -> np.ndarray:
    odo.start(R[0, 0:29], R[0, 29:58], R[0, 58:61], R[0, 61:64])
    for r in R[1:]:
        odo.step(r[0:29], r[29:58], r[58:61], r[61:64], 0.02)
    return odo.pose()


def truth_of(R: np.ndarray) -> np.ndarray:
    y0 = float(R[0, 67])
    c, s = math.cos(y0), math.sin(y0)
    d = R[-1, 64:66] - R[0, 64:66]
    return np.r_[c * d[0] + s * d[1], -s * d[0] + c * d[1], (R[-1, 67] - y0 + math.pi) % (2 * math.pi) - math.pi]


def variants() -> None:
    """The recorded walks (record()) under odometry variants, scored against the truth."""
    import json as _json

    from locomani.backends.scalebfm import runtime as R
    from locomani.backends.scalebfm.config import ScaleBfmBackendConfig

    z = np.load(OUT_REC)
    n = z["n"]
    so = np.r_[0, np.cumsum(n)[:-1]]
    segs = [z["rows"][a : a + k].astype(np.float64) for a, k in zip(so, n)]
    walks = z["walks"]
    cfg = ScaleBfmBackendConfig(scene_dir=str(E.REPO / "outputs" / "scene" / "walk_mars"), object_name="")
    sim = R.Sim(_json.load(open(R.META)), backend=cfg)
    T = np.array([truth_of(S) for S in segs])
    dr = 100 * np.hypot(*(T[:, :2] - walks[:, :2]).T)
    print(f"{len(segs)} walks; dead reckoning median {np.median(dr):.2f} cm, p90 {np.percentile(dr, 90):.2f}")
    out = {}
    for name, kw in (("height ankle 5mm", dict(mode="height", point="ankle", switch_m=0.005)),
                     ("height ankle 2mm", dict(mode="height", point="ankle")),
                     ("height sole 2mm", dict(mode="height", point="sole", switch_m=0.002)),
                     ("load ankle p1", dict(mode="load", point="ankle", pow_=1.0)), ("load ankle p2", dict(mode="load", point="ankle", pow_=2.0)),
                     ("load ankle max", dict(mode="load", point="ankle", pow_=math.inf)), ("load sole p2", dict(mode="load", pow_=2.0)),
                     ("load sole p4", dict(mode="load", pow_=4.0)), ("load sole max", dict(mode="load", pow_=math.inf))):  # fmt: skip
        odo = LegOdometry(sim.m, sim.qadr, sim.vadr, **kw)
        P = np.array([replay(odo, S) for S in segs])
        e = 100 * np.hypot(*(P[:, :2] - T[:, :2]).T)
        de = 100 * (P[:, :2] - T[:, :2])
        te = np.degrees(np.abs(P[:, 2] - T[:, 2]))
        out[name] = dict(median=float(np.median(e)), p90=float(np.percentile(e, 90)), better=int((e < dr).sum()),
                         bias_fwd=float(de[:, 0].mean()), bias_left=float(de[:, 1].mean()), turn=float(np.median(te)))  # fmt: skip
        o = out[name]
        print(f"{name:18s} median {o['median']:.2f} cm p90 {o['p90']:.2f} | better than dead reckoning {o['better']}/{len(segs)} | "
              f"mean error fwd {o['bias_fwd']:+.2f} left {o['bias_left']:+.2f} cm | turn {o['turn']:.2f} deg", flush=True)  # fmt: skip
    _json.dump(dict(dead_reckoning=[float(np.median(dr)), float(np.percentile(dr, 90))], variants=out),
               open(OUT_REC.with_name("leg_odo_variants.json"), "w"), indent=1)  # fmt: skip


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "probe":
        probe(int(sys.argv[2]) if len(sys.argv) > 2 else 30)
    elif len(sys.argv) > 1 and sys.argv[1] == "variants":
        sys.path[:0] = [str(E.REPO / "src")]
        variants()
    elif len(sys.argv) > 1 and sys.argv[1] == "record":
        record(int(sys.argv[2]) if len(sys.argv) > 2 else 40)
