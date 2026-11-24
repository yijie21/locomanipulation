"""ScaleBFM motor layer: the nine primitives the scene RSI loop's skills compose (TODO 27.2.1, plan
docs/handoff/scene-rsi-plan-20261016.md D5; the ninth, `squeeze`, TODO 38.4).

`Motor` owns one `runtime.Sim` + `runtime.Policy` pair and a whole-body reference (pelvis pose, 29 joints, joint
velocity, two jaw targets). Each primitive moves that reference, steps the simulation at 50 Hz until it is done and
returns an `Outcome`; the next primitive starts from where the robot actually is. Nothing here knows a task, an object
class or a scene layout: skills (task level) decide where to walk, what to reach for and what counts as a grasp.

    walk_to(xy, yaw)      turn toward the goal, walk a speed profile, turn to `yaw`, servo the stop
    turn(yaw)             walk_to in place
    sidestep(dy)          walk_to sideways (or backward, `forward`), facing kept
    posture(z, pitch)     pelvis height and forward torso lean by whole-body IK, feet planted
    reach(side, pos, R)   one jaw centre (or both) along straight legs through `via`, IK at waypoints, servoed
    grip(side) / release(side)   one jaw, or both at once (side "both")
    squeeze(inward, seconds)     both jaw centres pushed towards each other and kept there (a two-hand hold)
    hold(seconds)

The ninth action, `squeeze` (TODO 38.4, D4): the two-hand hold of a box wider than a jaw. Each jaw-centre servo gets an
inward offset (pelvis frame, so it turns with the body) that stays on through posture, hold and walk_to until
`release` or a `reach` of that hand drops it; blocked by the object, the servo's lead winds up to its cap and the
controller keeps pressing. Nothing here decides where to stand, where to put the hands or how to lift.

Modes (network.py's mode table): a walk runs in mode 2 (pelvis + wrists; the policy places its own steps); everything
stationary runs in mode 4 (pelvis + wrists + both ankles, the ankles pinned where they stand). Standing 15 s in the ADT
apartment, mode 2 widened the stance 27.7 -> 36.1 cm and mode 4 26.5 -> 28.0 cm (2026-10-16, scratch stand_smoke).
A walk's speed sets the gait mode 2 chooses: below about 0.4 m/s it shuffles, from 0.5 m/s it steps (MotorConfig.walk_speed).
Arm keypoints are FK of the reference joints (as bfm_expert_warehouse); the jaw-centre + wrist-attitude integral servo
of controller-compare 25.3 (`pad_servo` there) shifts the wrist keypoint toward the reference's jaw centre.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch

from locomani.backends.scalebfm import runtime as R
from locomani.primitives.arm_ik import ArmIK
from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES
from locomani.primitives.whole_body_ik import WholeBodyIK, rotvec

SIDES = ("left", "right")
WRIST = {s: f"{s}_wrist_yaw_link" for s in SIDES}
ANKLE = {s: f"{s}_ankle_roll_link" for s in SIDES}
FINGERS = {s: (f"{s}_dex1_finger_1_link", f"{s}_dex1_finger_2_link") for s in SIDES}
MODE_WALK, MODE_STAND = 2, 4
JAW_OPEN = R.GRIP_OPEN
FF_SPEEDS, FF_SHORT = (0.25, 0.35, 0.55), (0.045, 0.036, 0.016)  # plan program's stop shortfall per speed (24.7.1)
ROUTE_JOINT_SPEED = (
    0.8  # rad/s peak on the most moved joint along a reach_route path (warehouse visual.reach_route_speed)
)


@dataclass(frozen=True)
class MotorConfig:
    # m/s, the warehouse expert's visual.walk_speed. Mode 2 shuffles at it -- 15 cm steps, 2 cm foot lift, 34-40 % of foot
    # travel sliding on the floor; at 0.5 it steps (48 cm, 3-5 cm, 7-9 %; outputs/scene_rsi_20261016/gait/). 0.5 from
    # 2026-10-20 to 10-21, back to 0.25 for steadier stops (user 2026-10-21): after a 0.5 m/s walk the body still sways
    # when the reach starts (pregrasp jaw error 2.27 vs 1.28 cm), and the trained families went 187 -> 169 / 632 (TODO 32.4.4)
    walk_speed: float = 0.25
    walk_acc: float = 0.6  # m/s^2 (runtime.WALK_ACC)
    # TODO 33.1.4 (2): take the body model's free-space miss off every one-hand reach's jaw target (the scene layer
    # installs `reach_offset`); off by default and left out of the motor signature while off (batch.motor_signature).
    # On (RSI_REACH_COMPENSATION=1 for scripts/rsi's rounds, batch.motor_config): the open-floor grid's closed-loop
    # check halved the miss.
    reach_compensation: bool = False
    # TODO 40: a reach with via points (the skills' pregrasp transits) follows a CuRobo route around the scene to its
    # own straight solve's goal instead of the straight legs (the scene layer installs `reach_route`); off by default
    # and left out of the motor signature while off. On: RSI_REACH_ROUTE=1 for scripts/rsi's rounds.
    reach_route: bool = False
    # TODO 40.4, with reach_route (both left out of the signature at these defaults): "open" also routes every reach
    # whose side's jaw is open and empty (entry, descend, raise; the target object is no obstacle there); pose_goal
    # lets a route to the goal fall back to the wrist pose, CuRobo choosing the arm, when the joint goal cannot be
    # planned. RSI_ROUTE_SCOPE=open, RSI_ROUTE_POSE=1 for scripts/rsi's rounds.
    reach_route_scope: str = "via"
    reach_route_pose_goal: bool = False
    turn_rate: float = 0.8  # rad/s peak of an in-place turn
    face_travel_deg: float = 20.0  # a walk longer than face_travel_min_m first turns toward its goal past this
    face_travel_min_m: float = 0.30
    stop_tolerance: float = 0.015  # m, pelvis to goal after the profile ends
    stop_ki: float = 1.0  # 1/s, post-stop integral on the pelvis keypoint (bfm_warehouse WALK_KI)
    stop_cap: float = 0.15  # m per axis (WALK_CAP)
    stop_timeout: float = 3.0  # s after the profile before the walk ends anyway
    settle_s: float = 0.5  # s standing after a walk, mode 4 pinned
    reach_speed: float = 0.15  # m/s jaw centre along a reach leg
    reach_min_s: float = 0.6  # shortest leg duration
    reach_step_m: float = 0.03  # IK waypoint spacing along a leg
    reach_settle_s: float = 1.5  # s the servo gets after the last leg
    reach_tolerance: float = 0.01  # m jaw-centre error that ends the settle early
    rotation_weight: float = 0.6  # WholeBodyIK hand_rotation_weight (rad vs 10 per metre)
    servo_ki: float = 2.0  # 1/s jaw-centre integral (bfm_warehouse SERVO_KI)
    servo_cap: float = 0.08  # m per axis
    servo_ki_rot: float = 2.0  # 1/s wrist-attitude integral
    servo_cap_rot: float = 0.5  # rad
    grip_seconds: float = 0.8  # jaw ramp to the close target
    jaw_close: float = -0.06  # jaw target past closed: the force is setpoint depth (dex1, 20 N clamp)
    posture_seconds: float = 2.0  # s for a posture change
    # whole-body IK support envelope (crate carry_config names)
    waist_pitch_max: float = 0.52
    waist_dofs: str = "all"
    pelvis_height_min: float = 0.40
    pelvis_pitch_max: float = 0.70
    knee_max: float = 1.5
    pelvis_forward_max: float = 0.20


@dataclass
class Outcome:
    ok: bool
    why: str = ""
    t0: float = 0.0
    t1: float = 0.0
    info: dict = field(default_factory=dict)


@dataclass
class Pose:
    """A whole-body reference at one instant."""

    pos: np.ndarray  # pelvis position, world
    quat: np.ndarray  # pelvis wxyz, world
    joints: np.ndarray  # 29, MuJoCo order
    velocity: np.ndarray  # 29, joint velocity (extrapolates the future frames)


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def tilt(quat) -> float:
    """Angle (rad) between a body's z axis and the world vertical."""
    z = R.mat(quat)[:, 2]
    return float(np.arctan2(np.hypot(z[0], z[1]), z[2]))


def minjerk(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * u * (10 - 15 * u + 6 * u * u)


def expmap(v) -> np.ndarray:
    a = float(np.linalg.norm(v))
    if a < 1e-9:
        return np.eye(3)
    k = np.asarray(v, float) / a
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


class Profile:
    """Distance along a line vs time: accelerate at `acc` to `v`, cruise, decelerate to rest exactly at `dist`
    (bfm_brake.Profile)."""

    def __init__(self, v: float, acc: float, dist: float):
        self.v, self.acc, self.dist = v, acc, max(float(dist), 0.0)
        ta = v / acc
        da = 0.5 * acc * ta * ta
        if self.dist <= 2 * da:
            self.vp = float(np.sqrt(acc * self.dist))
            self.ta, self.tc = self.vp / acc, 0.0
        else:
            self.vp, self.ta, self.tc = v, ta, (self.dist - 2 * da) / v
        self.t_dec = self.ta + self.tc
        self.t_end = self.t_dec + self.vp / acc

    def s(self, t: float) -> float:
        if t <= 0:
            return 0.0
        if t <= self.ta:
            return 0.5 * self.acc * t * t
        da = 0.5 * self.acc * self.ta * self.ta
        if t <= self.t_dec:
            return da + self.vp * (t - self.ta)
        if t >= self.t_end:
            return self.dist
        u = self.t_end - t
        return self.dist - 0.5 * self.acc * u * u


def turn_seconds(angle: float, rate: float) -> float:
    """A smoothstep turn's duration whose peak rate (1.5 * angle / T) is `rate`."""
    return max(0.6, 1.5 * abs(angle) / max(rate, 1e-6))


class Keypoints:
    """FK of the scene's robot: pelvis pose + 29 joints -> every ScaleBFM keypoint body, on a private MjData."""

    def __init__(self, model, names):
        self.m, self.d = model, mujoco.MjData(model)
        self.qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in MUJOCO_JOINT_NAMES])
        self.bids = {n: model.body(n).id for n in names}

    def __call__(self, pos, quat, joints) -> dict:
        self.d.qpos[:3], self.d.qpos[3:7], self.d.qpos[self.qadr] = pos, quat, joints
        mujoco.mj_kinematics(self.m, self.d)
        return {n: (self.d.xpos[b].copy(), self.d.xquat[b].copy()) for n, b in self.bids.items()}


def lowest_foot_point(m, d, bodies) -> float:
    """World z of the lowest corner of any collision geom's local box on `bodies` (geom_aabb, so a mesh counts by its
    box; runtime.Sim.reset leaves the mars feet ~10 cm off the floor, memory reset-seated-the-robot-10cm-too-high)."""
    low = np.inf
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    for g in range(m.ngeom):
        if m.geom_bodyid[g] not in bodies or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        rot = d.geom_xmat[g].reshape(3, 3)
        corners = d.geom_xpos[g] + (m.geom_aabb[g, :3] + signs * m.geom_aabb[g, 3:]) @ rot.T
        low = min(low, float(corners[:, 2].min()))
    return low


def _plain(x):
    """A primitive's argument as JSON: arrays and tuples to lists, numpy scalars to floats, anything else to its repr."""
    if isinstance(x, np.ndarray):
        return np.round(x.astype(float), 6).tolist()
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (bool, int, float, str)) or x is None:
        return x
    if isinstance(x, np.generic):
        return x.item()
    return repr(x)[:200]


def _logged(fn):
    """TODO 33.20: each primitive call goes on `self.calls` -- name, start and end time, its arguments, its outcome --
    so one command of a round can be run again alone from the round's start (scripts/rsi/isolate.py). A call made
    inside another (turn -> walk_to) is part of the outer one. Recording only: nothing here touches the simulation."""

    @functools.wraps(fn)
    def wrap(self, *args, **kwargs):
        if self._call_depth:
            return fn(self, *args, **kwargs)
        row = dict(name=fn.__name__, t0=round(self.t, 3), args=_plain(list(args)), kwargs=_plain(kwargs))
        self._call_depth += 1
        try:
            out = fn(self, *args, **kwargs)
        finally:
            self._call_depth -= 1
        row.update(t1=round(self.t, 3), ok=bool(getattr(out, "ok", True)), why=str(getattr(out, "why", ""))[:80])
        self.calls.append(row)
        return out

    return wrap


class Motor:
    """One robot in one scene under ScaleBFM. `t` is simulated time since `reset`; `trace` holds one row per tick."""

    def __init__(
        self,
        sim: R.Sim,
        policy: R.Policy,
        config: MotorConfig | None = None,
        *,
        model_xml: str | None = None,
        track: tuple[str, ...] = (),
        jacobian: str = "analytic",
    ):
        self.sim, self.pol, self.cfg = sim, policy, config or MotorConfig()
        self.kp = Keypoints(sim.m, policy.sel)
        self.ik = ArmIK(model_xml) if model_xml else None
        self.jacobian = jacobian
        self.track = {n: sim.m.body(n).id for n in track}
        self.foot_bodies = {sim.m.body(ANKLE[s]).id for s in SIDES}
        fingers = (
            FINGERS if sim.hand == "dex1" else {s: (f"{s}_hand_thumb_2_link", f"{s}_hand_index_1_link") for s in SIDES}
        )
        self.fingers = {s: tuple(sim.m.body(n).id for n in fingers[s]) for s in SIDES}
        self.k = 0
        self.mode: int | None = None  # set (and sent to the policy) by the first primitive
        self.ref = None  # callable t -> Pose
        self.jaw = np.full(2, JAW_OPEN)
        self.pinned: dict | None = None
        self.servo: dict = {}
        # walk_to's stop servo: a callable giving the pelvis xy it steers on (None: the simulator's)
        self.stop_xy = None
        self.servo_on = {s: False for s in SIDES}
        self.servo_target: dict = {}
        self.squeeze_offset: dict = {}  # side -> jaw-centre offset in the pelvis (yaw) frame: `squeeze`
        self.hand_override: dict = {}  # side -> (jaw centre, wrist rotation, velocity): `hand`
        self.last_fut: dict = {}
        self.trace: list[dict] = []
        self.primitive = ""
        self.fell_at: float | None = None
        self.stand_pose: Pose | None = None
        # TODO 33.1.2: called by `reach` before the arm moves, (side, jaw target, wrist rotation, pelvis position, pelvis
        # quaternion wxyz, pelvis_z, jaw open) -> dict merged into the reach's Outcome info -- the scene's body-model
        # prediction (locomani.tasks.scene.body.reach_note). None: nothing; it reads state only, never steps the sim.
        self.reach_note = None
        # TODO 33.1.4 (2): (side, jaw target, wrist rotation, pelvis position, pelvis quaternion wxyz, pelvis_z) -> the world
        # vector to take off the target, or None -- installed by the scene layer only when cfg.reach_compensation
        self.reach_offset = None
        # TODO 40: (start joints, goal joints, pelvis position, pelvis quaternion wxyz) -> (T, 29) collision-free joint
        # path from start to goal planned with the pelvis held there, or None; `last` holds the plan's facts --
        # installed by the scene layer only when cfg.reach_route (tasks/scene/route.py)
        self.reach_route = None
        self.calls: list[dict] = []  # TODO 33.20: the primitive calls of the current round (_logged), cleared by reset
        self._call_depth = 0

    @property
    def t(self) -> float:
        return self.k * R.DT

    # -- measured state ---------------------------------------------------------------------------------------------
    def pelvis(self):
        return self.sim.d.qpos[:3].copy(), self.sim.d.qpos[3:7].copy()

    def yaw(self) -> float:
        return R.yaw_of(self.sim.d.qpos[3:7])

    def joints(self) -> np.ndarray:
        return self.sim.d.qpos[self.kp.qadr].copy()

    def jaw_centre(self, side: str):
        """(position, wrist rotation) of the jaw centre, measured."""
        p, q = self.sim.body(WRIST[side])
        Rw = R.mat(q)
        return p + Rw @ R.OFF, Rw

    def jaw_opening(self, side: str) -> float:
        i = SIDES.index(side)
        return float(2 * (np.mean(self.sim.d.qpos[self.sim.gq[2 * i : 2 * i + 2]]) + 0.023))

    def wrist(self, side: str):
        """(position, quat) of `side`'s wrist body, measured."""
        return self.sim.body(WRIST[side])

    def finger_positions(self, side: str) -> list[np.ndarray]:
        """World positions of `side`'s two finger bodies (finger 1 on the wrist's -y side first)."""
        return [self.sim.d.xpos[f].copy() for f in self.fingers[side]]

    # -- contacts: the simulator's ground truth; the episode judge and scene.view.TruthView read them, a skill does not
    def between_fingers(self, side: str) -> set[int]:
        """Bodies (ids) touched by both fingers of `side` -- the grasped object, if any."""
        m, d = self.sim.m, self.sim.d
        touch = {f: set() for f in self.fingers[side]}
        for i in range(d.ncon):
            b1, b2 = int(m.geom_bodyid[d.contact[i].geom1]), int(m.geom_bodyid[d.contact[i].geom2])
            for a, b in ((b1, b2), (b2, b1)):
                if a in touch:
                    touch[a].add(b)
        f1, f2 = self.fingers[side]
        return touch[f1] & touch[f2]

    def gripping(self, body: int) -> set[str]:
        """Sides whose two fingers both touch `body` (what `held(X)` needs, TODO 27.9)."""
        return {s for s in SIDES if body in self.between_fingers(s)}

    def touching(self, body: int) -> set[str]:
        """Sides whose fingers touch `body`."""
        m, d = self.sim.m, self.sim.d
        out = set()
        for i in range(d.ncon):
            pair = (int(m.geom_bodyid[d.contact[i].geom1]), int(m.geom_bodyid[d.contact[i].geom2]))
            for s in SIDES:
                if body in pair and any(f in pair for f in self.fingers[s]):
                    out.add(s)
        return out

    # -- reset ------------------------------------------------------------------------------------------------------
    def reset(self, xy, yaw: float, *, warm_up: int = 3) -> None:
        """Robot standing at (xy, yaw) with its feet 5 mm above the floor, the policy history at that state; the
        scene's free objects stay where the model (or the caller, after this) put them."""
        sim = self.sim
        self.calls, self._call_depth = [], 0
        sim.reset(np.asarray(xy, float), float(yaw), 0.0)
        sim.d.qpos[2] += 0.005 - lowest_foot_point(sim.m, sim.d, self.foot_bodies)
        mujoco.mj_forward(sim.m, sim.d)
        st = sim.state()
        for _ in range(warm_up):  # the TorchScript first call differs by ~6e-8 (bfm_round.warm_up)
            self.pol.reset(st)
            self.pol(st, {})
        self.pol.reset(st)
        self.k, self.trace, self.fell_at = 0, [], None
        self.jaw[:] = JAW_OPEN
        self.servo, self.servo_target, self.squeeze_offset, self.hand_override = {}, {}, {}, {}
        self.servo_on = {s: False for s in SIDES}
        self._stand()
        self.stand_pose = self.ref(0.0)

    def place(self, body: str, pos, quat) -> None:
        """Put a free body at (pos, quat) before the round starts (after `reset`)."""
        m, d = self.sim.m, self.sim.d
        adr = int(m.jnt_qposadr[m.body_jntadr[m.body(body).id]])
        d.qpos[adr : adr + 3], d.qpos[adr + 3 : adr + 7] = pos, quat
        d.qvel[int(m.jnt_dofadr[m.body_jntadr[m.body(body).id]]) :][:6] = 0
        mujoco.mj_forward(m, d)

    # -- the tick ---------------------------------------------------------------------------------------------------
    def _set_mode(self, mode: int) -> None:
        if mode != self.mode:
            self.mode = mode
            self.pol.mode = torch.tensor([mode], dtype=torch.long, device=self.pol.mode.device)

    def _stand(self, pose: Pose | None = None) -> None:
        """Hold `pose` (default: the measured pelvis and joints) in mode 4 with the ankles pinned where they are."""
        if pose is None:
            pos, quat = self.pelvis()
            pose = Pose(pos, yaw_quat(R.yaw_of(quat)), self.joints(), np.zeros(29))
        self.pinned = {ANKLE[s]: self.sim.body(ANKLE[s]) for s in SIDES}
        self._set_mode(MODE_STAND)
        self.ref = lambda _t, p=pose: p

    def _futures(self) -> dict:
        out: dict = {}
        for j in self.pol.fidx:
            pose = self.ref(self.t + j * R.DT)
            kp = self.kp(pose.pos, pose.quat, pose.joints + pose.velocity * j * R.DT)
            if self.pinned is not None:
                kp.update(self.pinned)
            for n, (p, q) in kp.items():
                out.setdefault(n, ([], []))
                out[n][0].append(p)
                out[n][1].append(q)
        fut = self._servo({n: (np.array(v[0]), np.array(v[1])) for n, v in out.items()})
        for side, (jaw, rot, vel) in self.hand_override.items():
            lead = np.array([j * R.DT for j in self.pol.fidx])[:, None] * vel[None]
            fut[WRIST[side]] = (np.repeat((jaw - rot @ R.OFF)[None], self.pol.F, 0) + lead,
                                np.repeat(R.quat_wxyz(rot)[None], self.pol.F, 0))  # fmt: skip
        return fut

    def hand(self, side: str, jaw_centre=None, rotation=None, velocity=None) -> None:
        """Drive `side`'s wrist keypoint directly from the next tick on: the jaw centre at `jaw_centre` (world) with
        wrist rotation `rotation`, the future frames led along `velocity` (m/s); a closed-loop hand program (the seed
        grasp's entry, skills/grasp_program.py) sets it every tick. jaw_centre None hands the wrist back to the
        reference. Mode 2 and 4 read only the wrist of the arm, so this is what an IK solution would deliver."""
        if jaw_centre is None:
            self.hand_override.pop(side, None)
            return
        vel = np.zeros(3) if velocity is None else np.asarray(velocity, float)
        self.hand_override[side] = (np.asarray(jaw_centre, float), np.asarray(rotation, float), vel)

    def _servo(self, fut: dict) -> dict:
        """The jaw-centre + wrist-attitude integral (bfm_expert_warehouse.pad_servo) on the sides that have it on:
        integrate the measured jaw centre's error to the reference's (or `servo_target`'s) and shift the wrist."""
        cfg = self.cfg
        for side in SIDES:
            if not self.servo_on[side]:
                continue
            w = WRIST[side]
            p_ref, R_ref = fut[w][0][0], R.mat(fut[w][1][0])
            jaw_ref = self.servo_target.get(side, p_ref + R_ref @ R.OFF)
            if side in self.squeeze_offset:  # `squeeze`: the inward push rides on the reference, turned with the body
                jaw_ref = jaw_ref + R.rot_z(self.yaw()) @ self.squeeze_offset[side]
            p_meas, R_meas = (getattr(self, "jaw_source", None) or self.jaw_centre)(side)
            turn = self.servo.get(("turn", side), np.zeros(3)) + cfg.servo_ki_rot * R.DT * rotvec(R_ref @ R_meas.T)
            n = float(np.linalg.norm(turn))
            turn = turn * (cfg.servo_cap_rot / n) if n > cfg.servo_cap_rot else turn
            self.servo[("turn", side)] = turn
            prev = self.servo.get(("pos", side), np.zeros(3))
            lead = np.clip(prev + cfg.servo_ki * R.DT * (jaw_ref - p_meas), -cfg.servo_cap, cfg.servo_cap)
            if side in self.squeeze_offset:
                # in contact, the integral winds up: along the push it is capped at the offset itself, so the command
                # sits at most 2 x inward_m inside the wall (the box flew out of the hands at 12 cm, probe 2026-10-23);
                # across the push the lead is capped at the offset's size too (a hand held by friction would otherwise
                # wind its vertical and forward leads to servo_cap and drag the box)
                off = R.rot_z(self.yaw()) @ self.squeeze_offset[side]
                n = float(np.linalg.norm(off))
                if n > 1e-9:
                    u = off / n
                    along = min(float(lead @ u), n)
                    across = lead - u * float(lead @ u)
                    a = float(np.linalg.norm(across))
                    lead = (across * (n / a) if a > n else across) + u * along
            self.servo[("pos", side)] = lead
            Rt = expmap(turn)
            quats = np.array([R.quat_wxyz(Rt @ R.mat(q)) for q in fut[w][1]])
            shift = np.array([R.mat(q) @ R.OFF - Rt @ R.mat(q) @ R.OFF for q in fut[w][1]])
            fut[w] = (fut[w][0] + shift + lead, quats)
        return fut

    def tick(self) -> bool:
        """One 50 Hz step on the current reference. False once the robot has fallen (the step is not taken)."""
        sim = self.sim
        fut = self._futures()
        sim.set_jaw(self.jaw)
        if sim.fallen():
            self.fell_at = self.t if self.fell_at is None else self.fell_at
            return False
        self.trace.append(
            dict(
                t=self.t,
                primitive=self.primitive,
                mode=self.mode,
                robot=sim.d.qpos[:7].copy(),
                joints=self.joints(),
                ref_pelvis=fut["pelvis"][0][0].copy(),
                ref_wrists=np.array([fut[WRIST[s]][0][0] for s in SIDES]),
                wrists=np.array([sim.body(WRIST[s])[0] for s in SIDES]),
                jaw=self.jaw.copy(),
                fingers=sim.d.qpos[sim.gq].copy(),
                objects={n: np.r_[sim.d.xpos[b], sim.d.xquat[b]] for n, b in self.track.items()},
            )
        )
        sim.step(self.pol(sim.state(), fut))
        self.last_fut = fut
        self.k += 1
        return True

    def commanded_jaw(self, side: str) -> np.ndarray:
        """The jaw centre the last tick commanded (reference + servo lead): where a hand program taking over must
        start its command, or the servo's lead is dropped in one step."""
        p, q = self.last_fut[WRIST[side]]
        return p[0] + R.mat(q[0]) @ R.OFF

    def _run(self, name: str, update, timeout: float) -> Outcome:
        """Tick until `update(t)` returns an Outcome, the robot falls, or `timeout` s pass."""
        self.primitive, t0 = name, self.t
        while self.t - t0 < timeout:
            out = update(self.t)
            if out is not None:
                out.t0, out.t1 = t0, self.t
                return out
            if not self.tick():
                return Outcome(False, f"fell at {self.fell_at:.2f} s", t0, self.t)
        return Outcome(False, "timeout", t0, self.t)

    # -- locomotion -------------------------------------------------------------------------------------------------
    @_logged
    def walk_to(self, xy, yaw: float | None = None, *, speed: float | None = None, face: str = "travel") -> Outcome:
        """Walk the pelvis to `xy` and end facing `yaw` (None: the heading it walked on). face "travel": turn toward the goal first when
        it is farther than face_travel_min_m and off the heading by face_travel_deg; "fixed": keep the heading while
        moving (a sidestep or a step back)."""
        cfg = self.cfg
        goal = np.asarray(xy, float)
        p0, q0 = self.pelvis()
        yaw0 = R.yaw_of(q0)
        z = float(p0[2])
        q_stand = self.ref(self.t).joints.copy()
        delta = goal - p0[:2]
        dist = float(np.linalg.norm(delta))
        heading = yaw0
        if face == "travel" and dist > cfg.face_travel_min_m:
            travel = float(np.arctan2(delta[1], delta[0]))
            if abs(R.wrap(travel - yaw0)) > np.radians(cfg.face_travel_deg):
                heading = travel
        yaw_end = heading if yaw is None else float(yaw)  # None: keep the walking heading (a path's inner corner)
        segments = []  # (kind, duration, fn(u) -> (xy, yaw))
        if abs(R.wrap(heading - yaw0)) > 1e-3:
            a = R.wrap(heading - yaw0)
            segments.append((turn_seconds(a, cfg.turn_rate), lambda u, a=a: (p0[:2], yaw0 + R.smoothstep(u) * a)))
        ff = float(np.interp(speed or cfg.walk_speed, FF_SPEEDS, FF_SHORT))
        prof = Profile(speed or cfg.walk_speed, cfg.walk_acc, dist + ff if dist > 0.05 else dist)
        d = delta / max(dist, 1e-9)
        if dist > 1e-3:
            segments.append((prof.t_end, lambda u, T=prof.t_end: (p0[:2] + d * prof.s(u * T), heading)))
        a2 = R.wrap(yaw_end - heading)
        if abs(a2) > 1e-3:
            segments.append((turn_seconds(a2, cfg.turn_rate), lambda u, a=a2: (goal, heading + R.smoothstep(u) * a)))
        if not segments:
            return self.hold(cfg.settle_s)
        starts = np.cumsum([0.0] + [s[0] for s in segments])
        t_start = self.t
        total = float(starts[-1])

        def at(t):
            tt = t - t_start
            i = int(np.clip(np.searchsorted(starts, tt, side="right") - 1, 0, len(segments) - 1))
            dur, fn = segments[i]
            xy_, yw = fn(np.clip((tt - starts[i]) / max(dur, 1e-9), 0.0, 1.0))
            push = self.servo.get("walk_push", np.zeros(2))
            return Pose(np.r_[np.asarray(xy_) + push, z], yaw_quat(yw), q_stand, np.zeros(29))

        self.pinned = None
        self._set_mode(MODE_WALK)
        self.servo.pop("walk_push", None)
        self.ref = at
        stop = {}

        def update(t):
            tt = t - t_start
            if tt < total:
                return None
            err = goal - (self.sim.d.qpos[:2] if self.stop_xy is None else np.asarray(self.stop_xy(), float))
            self.servo["walk_push"] = np.clip(
                self.servo.get("walk_push", np.zeros(2)) + cfg.stop_ki * R.DT * err, -cfg.stop_cap, cfg.stop_cap
            )
            near = float(np.linalg.norm(err)) <= cfg.stop_tolerance
            if tt >= total + cfg.settle_s and (near or tt >= total + cfg.stop_timeout):
                stop["err"] = float(np.linalg.norm(err))
                return Outcome(
                    True,
                    "" if near else "stopped outside tolerance",
                    info=dict(
                        stop_err_cm=round(100 * stop["err"], 2),
                        yaw_err_deg=round(float(np.degrees(R.wrap(self.yaw() - yaw_end))), 2),
                        walked_m=round(dist, 3),
                    ),
                )
            return None

        out = self._run("walk_to", update, total + cfg.stop_timeout + cfg.settle_s + 1.0)
        self.servo.pop("walk_push", None)
        if self.fell_at is None:
            self._stand()
        return out

    @_logged
    def turn(self, yaw: float) -> Outcome:
        return self.walk_to(self.sim.d.qpos[:2].copy(), yaw)

    @_logged
    def sidestep(self, left: float = 0.0, forward: float = 0.0) -> Outcome:
        """Move the pelvis `left` (and `forward`) metres in its own frame, keeping the heading."""
        yaw = self.yaw()
        c, s = np.cos(yaw), np.sin(yaw)
        goal = self.sim.d.qpos[:2] + np.array([c * forward - s * left, s * forward + c * left])
        return self.walk_to(goal, yaw, face="fixed")

    # -- whole-body IK ----------------------------------------------------------------------------------------------
    def _solver(self, pelvis_height_target=None, at: Pose | None = None) -> WholeBodyIK:
        """The whole-body IK of the current state, or built on the reference pose `at` and seeded exactly there."""
        if self.ik is None:
            raise RuntimeError("Motor needs model_xml for whole-body IK (posture, reach)")
        if at is None:
            pos, quat = self.pelvis()
            joints = self.joints()
        else:
            pos, quat, joints = at.pos.copy(), at.quat.copy(), at.joints.copy()
        state = dict(joints=joints, pelvis=dict(pos=pos, quat=quat))
        cfg = self.cfg
        solver = WholeBodyIK(
            self.ik,
            state,
            cfg,
            tool_offset=tuple(R.OFF),
            knee_max=cfg.knee_max,
            pelvis_forward_max=cfg.pelvis_forward_max,
            pelvis_height_target=pelvis_height_target,
            jacobian=self.jacobian,
        )
        if pelvis_height_target is not None:
            # WholeBodyIK bounds the pelvis at most 2 cm above where it stands (a crouch solver); a posture that rises
            # back needs the bound at its target (motor_check posture_low, 2026-10-16: 0.58 -> 0.74 stayed at 0.58)
            solver.hi[2] = max(solver.hi[2], pelvis_height_target + 0.01)
        if at is not None:
            solver.seed = np.clip(solver.initial, solver.lo, solver.hi)
        return solver

    def _solve(self, solver, positions, rotations, *, weights=None, torso=None, seed=None, rotation_weight=None):
        if weights is not None:
            solver.hand_weights = np.asarray(weights, float)
        pose = solver.solve(
            positions,
            rotations,
            seed=seed,
            torso_rotation=torso,
            hand_rotation_weight=self.cfg.rotation_weight if rotation_weight is None else rotation_weight,
        )
        return pose

    def _pose_of(self, ik_pose, yaw: float) -> Pose:
        """WholeBodyIK's solution (base position, base rotation incl. pitch, 29 joints) as a reference pose."""
        return Pose(
            np.asarray(ik_pose.position, float), R.quat_wxyz(ik_pose.rotation), ik_pose.joints.copy(), np.zeros(29)
        )

    def _blend(self, poses: list[Pose], durations: list[float]):
        """A reference through `poses` (the first is where it starts), min-jerk in joints and pelvis per leg."""
        starts = np.cumsum([0.0] + list(durations))
        t_start = self.t

        def at(t):
            tt = t - t_start
            if tt >= starts[-1]:
                return poses[-1]
            i = int(np.clip(np.searchsorted(starts, tt, side="right") - 1, 0, len(durations) - 1))
            u = (tt - starts[i]) / max(durations[i], 1e-9)
            s, a, b = minjerk(u), poses[i], poses[i + 1]
            ds = 30 * u * u * (1 - u) ** 2 / max(durations[i], 1e-9)
            qa, qb = a.quat, b.quat if float(np.dot(a.quat, b.quat)) >= 0 else -b.quat
            q = (1 - s) * qa + s * qb
            return Pose(
                a.pos + s * (b.pos - a.pos),
                q / np.linalg.norm(q),
                a.joints + s * (b.joints - a.joints),
                ds * (b.joints - a.joints),
            )

        return at, float(starts[-1])

    def _routed(self, poses: list[Pose], durations: list[float], leg_ends: list[int], jaws=None, approach=False):
        """TODO 40: (reference, duration, facts) for a reach under `reach_route`. The straight solve keeps its IK
        poses; the route replaces the legs up to the goal's pose, or, when that plan fails (a pregrasp within the
        planner's clearance of its support), up to the last via point, the last leg staying straight. The route is
        retimed at ROUTE_JOINT_SPEED and never shorter than the legs it replaces; the pelvis and the joints the route
        does not plan move min-jerk between its ends. No plan keeps the straight reference. `jaws[k]`: the side's jaw
        centre (position, rotation, world) at the end of leg k, for the hook's pose goal and target (TODO 40.4); a
        route the hook planned to a pose goal ends where its path ends (the arm is CuRobo's own)."""
        from locomani.primitives.arm_route import retime_path, sample_path

        tries = []
        cuts = (len(leg_ends) - 1, len(leg_ends) - 2) if len(leg_ends) > 1 else (0,)
        for k in cuts:
            cut = leg_ends[k]
            a, b = poses[0], poses[cut]
            extra = {} if jaws is None else dict(jaw=jaws[k], approach=approach, pose_ok=k == len(leg_ends) - 1)
            try:
                path = self.reach_route(a.joints, b.joints, b.pos, b.quat, **extra)
            except Exception as e:  # noqa: BLE001 -- a route that fails must never end a round
                path = None
                tries.append(dict(error=repr(e)[:160]))
            else:
                tries.append(dict(getattr(self.reach_route, "last", {})))
            if path is not None:
                break
        if path is None:
            at, total = self._blend(poses, durations)
            return at, total, dict(to=None, tries=tries)
        path, frac, route_total = retime_path(path, ROUTE_JOINT_SPEED, min_seconds=float(sum(durations[:cut])))
        if cut == leg_ends[-1] and tries[-1].get("goal") == "pose":  # the arm is CuRobo's: stay where the path ends
            end = Pose(b.pos, b.quat, np.asarray(path[-1], float).copy(), np.zeros(29))
            tail, tail_total = (lambda t: end), 0.0
        else:
            tail, tail_total = self._blend(poses[cut:], durations[cut:])
        qa, qb = a.quat, b.quat if float(np.dot(a.quat, b.quat)) >= 0 else -b.quat
        t0 = self.t

        def at(t):
            tt = t - t0
            if tt >= route_total:
                return tail(t - route_total)
            q, v = sample_path(path, frac, route_total, tt)
            u = tt / route_total
            s = 3 * u * u - 2 * u**3
            quat = (1 - s) * qa + s * qb
            return Pose(a.pos + s * (b.pos - a.pos), quat / np.linalg.norm(quat), q, v)

        to = "goal" if cut == leg_ends[-1] else "last_via"
        return at, route_total + tail_total, dict(to=to, seconds_s=round(route_total, 2), tries=tries)

    def _stand_up(self, seconds: float) -> Outcome:
        cfg = self.cfg
        start = self.ref(self.t)
        pos, quat = self.pelvis()
        q = start.joints.copy()
        q[:15] = self.stand_pose.joints[:15]  # legs (0-11) and waist (12-14); arms as they are
        goal = Pose(np.r_[pos[:2], self.stand_pose.pos[2]], yaw_quat(R.yaw_of(quat)), q, np.zeros(29))
        at, total = self._blend([start, goal], [seconds])
        self._stand(start)
        self.ref = at
        t0 = self.t

        def update(t):
            if t - t0 < total + cfg.settle_s:
                return None
            return Outcome(
                True,
                info=dict(
                    pelvis_z=round(float(self.sim.d.qpos[2]), 3),
                    pelvis_z_ref=round(float(goal.pos[2]), 3),
                    torso_pitch_deg=round(float(np.degrees(tilt(self.sim.body("torso_link")[1]))), 1),
                ),
            )

        return self._run("posture", update, total + cfg.settle_s + 1.0)

    @_logged
    def posture(
        self, pelvis_z: float | None = None, torso_pitch: float | None = None, *, seconds: float | None = None
    ) -> Outcome:
        """Lower the pelvis to `pelvis_z` (m) and/or lean the torso `torso_pitch` (rad, world) by whole-body IK, feet
        planted, the hands riding along at low IK weight; mode 4. With neither given: stand back up -- the legs and
        waist of the standing pose `reset` recorded, the arms as they are (the IK, started from a squat, stopped at
        0.65 m on the way up, motor_check 2026-10-16).
        Mode 4 tracks no torso keypoint: the controller chooses the waist. Squatting with an upright reference torso
        it leaned 21 deg at 0.62 m and 56 deg at 0.60 m; a 20 / 34 deg reference lean came out as 8 / 16 deg. The
        info reports reference and delivered tilt."""
        cfg = self.cfg
        pos, quat = self.pelvis()
        yaw = R.yaw_of(quat)
        if pelvis_z is None and torso_pitch is None:
            return self._stand_up(seconds or cfg.posture_seconds)
        solver = self._solver(pelvis_height_target=pelvis_z)
        dz = 0.0 if pelvis_z is None else pelvis_z - float(pos[2])
        targets = np.array([self.jaw_centre(s)[0] + [0, 0, dz] for s in SIDES])
        rots = np.array([self.jaw_centre(s)[1] for s in SIDES])
        # torso upright unless a lean is asked: without a target the solver keeps whatever lean it starts from
        torso = R.rot_z(yaw) @ R.rot_y(0.0 if torso_pitch is None else torso_pitch)
        # the hands only keep the arms tidy here (weight 0.05): at 0.3 they drove a 0.52 m squat to a 63 deg lean and
        # held a 0.35 rad lean to 2.5 deg (motor_check 2026-10-16)
        ik_pose = self._solve(solver, targets, rots, weights=(0.05, 0.05), torso=torso)
        start = self.ref(self.t)
        goal = self._pose_of(ik_pose, yaw)
        at, total = self._blend([start, goal], [seconds or cfg.posture_seconds])
        self._stand(start)
        self.ref = at

        def update(t):
            if t - t0 < total + cfg.settle_s:
                return None
            z_now = float(self.sim.d.qpos[2])
            ref_kp = self.kp(goal.pos, goal.quat, goal.joints)
            return Outcome(
                True,
                info=dict(
                    pelvis_z=round(z_now, 3),
                    pelvis_z_ref=round(float(goal.pos[2]), 3),
                    torso_pitch_deg=round(float(np.degrees(tilt(self.sim.body("torso_link")[1]))), 1),
                    torso_pitch_ref_deg=round(float(np.degrees(tilt(ref_kp["torso_link"][1]))), 1),
                    pelvis_pitch_deg=round(float(np.degrees(tilt(self.sim.d.qpos[3:7]))), 1),
                    pelvis_pitch_ref_deg=round(float(np.degrees(tilt(goal.quat))), 1),
                    ik_hand_err_cm=[round(100 * float(e), 2) for e in ik_pose.hand_errors],
                ),
            )

        t0 = self.t
        return self._run("posture", update, total + cfg.settle_s + 1.0)

    @_logged
    def reach(
        self,
        side: str,
        pos,
        rot=None,
        *,
        via=(),
        speed: float | None = None,
        rotation_weight: float | None = None,
        servo: bool = True,
        pelvis_z: float | None = None,
        keep_squeeze: bool = False,
        start=None,
        start_rot=None,
        ik_from_ref: bool = False,
    ) -> Outcome:
        """Move `side`'s jaw centre to `pos` (world) with wrist rotation `rot` (3x3 world; None keeps the current one)
        through the points `via`, legs straight at `speed`, whole-body IK at waypoints reach_step_m apart, feet
        planted. side "both": `pos`, `rot` and each via point are pairs (left, right). The other jaw centre stays where it
        is in the world (IK weight 0.3). The servo integrates the jaw-centre
        error from the start and settles up to reach_settle_s at the end. A reach ends the moved hands' squeeze
        unless `keep_squeeze` (a two-hand lift by the arms: both hands up, still pressing; TODO 38.4).

        `start` / `start_rot` (one side only): the path starts at this point / attitude instead of the measured ones.
        ik_from_ref: the whole-body IK is built on the current reference pose and seeded there, not on the measured joints
        with its waist-forward seed. The reference is what the first leg blends from, so the plan stays in its posture
        family; from a measured posture far from the reference the IK can land in another family and the joint-space
        blend swings the reference jaw tens of cm (grasp_exp.py, top shelf, 2026-11-08)."""
        cfg = self.cfg
        sides = SIDES if side == "both" else (side,)
        solver = self._solver(pelvis_height_target=pelvis_z, at=self.ref(self.t) if ik_from_ref else None)
        cur = {s: self.jaw_centre(s) for s in SIDES}
        if start is not None or start_rot is not None:
            assert side != "both", "reach: start / start_rot are for one side"
            cur[side] = (
                cur[side][0] if start is None else np.asarray(start, float),
                cur[side][1] if start_rot is None else np.asarray(start_rot, float),
            )
        if side == "both":
            goal_p = {s: np.asarray(pos[i], float) for i, s in enumerate(SIDES)}
            goal_R = {s: cur[s][1] if rot is None else np.asarray(rot[i], float) for i, s in enumerate(SIDES)}
            legs = [{s: np.asarray(v[i], float) for i, s in enumerate(SIDES)} for v in via]
        else:
            goal_p, goal_R = (
                {side: np.asarray(pos, float)},
                {side: cur[side][1] if rot is None else np.asarray(rot, float)},
            )
            legs = [{side: np.asarray(v, float)} for v in via]
        points = [{s: cur[s][0] for s in sides}] + legs + [goal_p]
        pred = {}
        if self.reach_note is not None and side != "both":
            try:  # a prediction must never change or end a round
                pelvis_p, pelvis_q = self.pelvis()
                i = SIDES.index(side)
                jaw_open = abs(float(self.jaw[i]) - JAW_OPEN) < 1e-3
                pred = self.reach_note(side, goal_p[side], goal_R[side], pelvis_p, pelvis_q, pelvis_z, jaw_open) or {}
            except Exception as e:  # noqa: BLE001
                pred = {"pred_error": repr(e)[:160]}
        true_p = {s: np.asarray(goal_p[s], float).copy() for s in sides}  # what the skill asked; errors are against it
        if self.reach_offset is not None and side != "both":
            try:  # a compensation that fails leaves the command as it was
                pelvis_p, pelvis_q = self.pelvis()
                off = self.reach_offset(side, goal_p[side], goal_R[side], pelvis_p, pelvis_q, pelvis_z)
                if off is not None:
                    goal_p[side] = goal_p[side] - np.asarray(off, float)
                    pred["comp_offset_cm"] = [round(100 * float(v), 2) for v in off]
            except Exception as e:  # noqa: BLE001
                pred["comp_error"] = repr(e)[:160]
            points[-1] = {s: goal_p[s] for s in sides}
        yaw = self.yaw()
        weights = tuple(1.0 if s in sides else 0.3 for s in SIDES)
        poses, durations, errs, leg_ends = [self.ref(self.t)], [], [], []
        for leg, (a, b) in enumerate(zip(points[:-1], points[1:])):
            length = max(float(np.linalg.norm(b[s] - a[s])) for s in sides)
            n = max(1, int(np.ceil(length / cfg.reach_step_m)))
            dur = max(cfg.reach_min_s, length / (speed or cfg.reach_speed))
            for k in range(1, n + 1):
                u = k / n
                tp, tr = [], []
                for s in SIDES:
                    if s in sides:
                        tp.append(a[s] + u * (b[s] - a[s]))
                        frac = u if leg == 0 else 1.0  # the attitude turns over the first leg, then holds
                        tr.append(cur[s][1] @ expmap(frac * rotvec(cur[s][1].T @ goal_R[s])))
                    else:
                        tp.append(cur[s][0])
                        tr.append(cur[s][1])
                ik_pose = self._solve(
                    solver, np.array(tp), np.array(tr), weights=weights, rotation_weight=rotation_weight
                )
                poses.append(self._pose_of(ik_pose, yaw))
                durations.append(dur / n)
                errs.append([float(e) for e in ik_pose.hand_errors])
            leg_ends.append(len(poses) - 1)
        routed = self.reach_route is not None and side != "both"
        if routed and not via:  # TODO 40.4 "open": a reach whose jaw is open and holds nothing
            routed = getattr(cfg, "reach_route_scope", "via") == "open"
            routed = routed and abs(float(self.jaw[SIDES.index(side)]) - JAW_OPEN) < 1e-3
            routed = routed and not self.between_fingers(side)
        if routed:
            jaws = [(points[k + 1][side], goal_R[side]) for k in range(len(leg_ends))]
            at, total, pred["route"] = self._routed(poses, durations, leg_ends, jaws, approach=not via)
        else:
            at, total = self._blend(poses, durations)
        self._stand(poses[0])
        self.ref = at
        for s in sides:
            self.servo_on[s] = servo
            self.servo_target.pop(s, None)
            if not keep_squeeze:
                self.squeeze_offset.pop(s, None)  # a reach of this hand ends its squeeze
        t0 = self.t

        def update(t):
            tt = t - t0
            if tt < total:
                return None
            err = max(float(np.linalg.norm(self.jaw_centre(s)[0] - true_p[s])) for s in sides)
            if tt >= total + cfg.reach_settle_s or (tt >= total + 0.3 and err <= cfg.reach_tolerance):
                att = max(float(np.degrees(np.linalg.norm(rotvec(goal_R[s] @ self.jaw_centre(s)[1].T)))) for s in sides)
                return Outcome(
                    err <= max(cfg.reach_tolerance, 0.02),
                    "" if err <= 0.02 else f"jaw centre {100 * err:.1f} cm off",
                    info=dict(
                        jaw_err_cm=round(100 * err, 2),
                        attitude_err_deg=round(att, 1),
                        ik_err_cm=round(100 * max(max(e) for e in errs), 2),
                        legs=len(points) - 1,
                        **pred,
                    ),
                )
            return None

        return self._run("reach", update, total + cfg.reach_settle_s + 1.0)

    @_logged
    def adopt_hand(self, side: str) -> None:
        """Make the reference carry `side`'s wrist where `hand` drives it now, then release the override: a walk after
        a hand program then carries the hand with the body instead of snapping the wrist back to the last IK pose."""
        jaw, rot, _ = self.hand_override[side]
        solver = self._solver()
        cur = {s: self.jaw_centre(s) for s in SIDES}
        tp = np.array([jaw if s == side else cur[s][0] for s in SIDES])
        tr = np.array([rot if s == side else cur[s][1] for s in SIDES])
        ik_pose = self._solve(solver, tp, tr, weights=tuple(1.0 if s == side else 0.3 for s in SIDES))
        pose = self._pose_of(ik_pose, self.yaw())
        self._stand(pose)
        self.hand_override.pop(side, None)
        self.servo_on[side] = True
        self.servo_target.pop(side, None)
        self.servo.pop(("pos", side), None)
        self.servo.pop(("turn", side), None)

    # -- hands ------------------------------------------------------------------------------------------------------
    @_logged
    def grip(self, side: str, *, seconds: float | None = None, target: float | None = None) -> Outcome:
        """Close `side`'s jaw (ramp to `target`, default jaw_close) and report what is between the fingers. side
        "both" closes the two jaws together (TODO 38.4): ok when each holds something; `held` and `opening_cm` are
        then per side, left then right."""
        cfg = self.cfg
        sides = SIDES if side == "both" else (side,)
        idx = [SIDES.index(s) for s in sides]
        j0, j1 = [float(self.jaw[i]) for i in idx], cfg.jaw_close if target is None else float(target)
        T = seconds or cfg.grip_seconds
        t0 = self.t

        def update(t):
            u = (t - t0) / max(T, 1e-9)
            for i, a in zip(idx, j0):
                self.jaw[i] = a + min(1.0, u) * (j1 - a)
            if u < 1.0 + 0.25 / T:
                return None
            held = [sorted(self.sim.m.body(b).name for b in self.between_fingers(s)) for s in sides]
            opening = [round(100 * self.jaw_opening(s), 2) for s in sides]
            empty = [s for s, h in zip(sides, held) if not h]
            return Outcome(
                not empty,
                "" if not empty else f"nothing between the fingers ({', '.join(empty)})" if side == "both"
                else "nothing between the fingers",
                info=dict(held=held if side == "both" else held[0], opening_cm=opening if side == "both" else opening[0]),
            )  # fmt: skip

        return self._run("grip", update, T + 1.0)

    @_logged
    def release(self, side: str, *, seconds: float = 0.5) -> Outcome:
        """Open `side`'s jaw (side "both": the two together) and end its squeeze, if any."""
        sides = SIDES if side == "both" else (side,)
        idx = [SIDES.index(s) for s in sides]
        j0 = [float(self.jaw[i]) for i in idx]
        for s in sides:
            self.squeeze_offset.pop(s, None)
        t0 = self.t

        def update(t):
            u = (t - t0) / seconds
            for i, a in zip(idx, j0):
                self.jaw[i] = a + min(1.0, u) * (JAW_OPEN - a)
            if u < 1.0 + 0.25 / seconds:
                return None
            opening = [round(100 * self.jaw_opening(s), 2) for s in sides]
            return Outcome(True, info=dict(opening_cm=opening if side == "both" else opening[0]))

        return self._run("release", update, seconds + 1.0)

    @_logged
    def squeeze(self, inward_m: float = 0.03, seconds: float = 1.0) -> Outcome:
        """Push the two jaw centres towards each other by `inward_m` each, along the line between them, and keep
        pushing (TODO 38.4, D4): the offsets ride on each hand's servo reference in the pelvis frame, so they stay on
        through posture, hold and walk_to until `release` (or a `reach` of that hand) drops them; blocked by an object,
        each servo's lead winds up to servo_cap and the controller keeps pressing. Ticks `seconds` for the leads to build.
        ok when, at the end, the fingers of both hands touch one common body; info: that body (`common`), the jaw-centre
        distance, each side's lead."""
        left, right = self.jaw_centre("left")[0], self.jaw_centre("right")[0]
        axis = right - left
        n = float(np.linalg.norm(axis))
        if n < 1e-6:
            return Outcome(False, "the jaw centres coincide", self.t, self.t)
        axis = axis / n
        back = R.rot_z(-self.yaw())
        self.squeeze_offset = {
            **self.squeeze_offset,
            "left": back @ (inward_m * axis),
            "right": back @ (-inward_m * axis),
        }
        for s in SIDES:
            self.servo_on[s] = True
        t0 = self.t

        def update(t):
            if t - t0 < seconds:
                return None
            m, d = self.sim.m, self.sim.d
            touched = {s: set() for s in SIDES}
            for i in range(d.ncon):
                pair = (int(m.geom_bodyid[d.contact[i].geom1]), int(m.geom_bodyid[d.contact[i].geom2]))
                for s in SIDES:
                    for f in self.fingers[s]:
                        if f in pair:
                            touched[s].add(pair[1] if pair[0] == f else pair[0])
            common = sorted(
                m.body(b).name
                for b in touched["left"] & touched["right"]
                if b not in self.fingers["left"] + self.fingers["right"]
            )
            gap = float(np.linalg.norm(self.jaw_centre("right")[0] - self.jaw_centre("left")[0]))
            return Outcome(
                bool(common),
                "" if common else "the two hands touch no common body",
                info=dict(common=common, gap_cm=round(100 * gap, 2),
                          lead_cm=[round(100 * float(np.linalg.norm(self.servo.get(("pos", s), np.zeros(3)))), 2) for s in SIDES]),
            )  # fmt: skip

        return self._run("squeeze", update, seconds + 1.0)

    @_logged
    def hold(self, seconds: float) -> Outcome:
        """Keep the current reference for `seconds` (a walk's end state stays pinned in mode 4)."""
        t0 = self.t
        return self._run("hold", lambda t: Outcome(True) if t - t0 >= seconds else None, seconds + 1.0)

    # -- record -----------------------------------------------------------------------------------------------------
    def arrays(self) -> dict:
        """The trace as arrays (npz-ready): t, mode, robot (pelvis xyz + quat), joints, ref_pelvis, ref_wrists, wrists,
        jaw (the gripper command: past the finger joints' -0.02 limit while gripping), fingers (the four finger joints as
        measured, left 1, 2, right 1, 2: what a replay draws, TODO 27.14), primitive (index into `primitives`), one array
        per tracked object (xyz + quat)."""
        rows = self.trace
        names = sorted({r["primitive"] for r in rows})
        out = {
            k: np.array([r[k] for r in rows], dtype=np.float32)
            for k in ("t", "mode", "robot", "joints", "ref_pelvis", "ref_wrists", "wrists", "jaw", "fingers")
        }
        out["primitive"] = np.array([names.index(r["primitive"]) for r in rows], dtype=np.int16)
        out["primitives"] = np.array(names)
        for n in self.track:
            out[f"obj_{n}"] = np.array([r["objects"][n] for r in rows], dtype=np.float32)
        return out
