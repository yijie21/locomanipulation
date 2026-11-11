"""Cabinet grasp experiment (2026-11-02, after grasp_pilot.py): direct commands, two servos and the self model, with the
self model's plans checked against the cabinet.

Same cabinet scene, grasp procedure and success test as grasp_pilot.py v2 (pre-grasp outside the front, entry, close,
lift, hold 3 s; success = the box between the fingers and risen > 2 cm), the 8 cm box only. 72 placements (4 shelves x
3 depths x 3 lateral offsets x 2 hands), each from 3 seeded starts 1.5-2.5 m back = 216 setups; every method replays
the same reset and walk of a setup, so all comparisons are paired. Ground-truth target (the settled box).

Methods (each reach: the pre-grasp, the entry and the lift):
    direct       the goal point as the command, no feedback
    direct_vis   + a visual servo: the late position-only servo, integrating only while the head camera sees the box
                 (a ray from the camera to the box centre or a corner of its front face hits the box, inside the camera's
                 field of view); while it does not, the servo's target is the measured jaw, so the lead holds
    direct_prop  + a proprioceptive servo: the late position-only servo on the jaw centre. In simulation the joint
                 readings with the planted foot give the true hand position, so this is also the true-position servo
    inv          the self model's command: wrist_invert's position fixed point from the state the command is issued
                 from, for each pelvis-height candidate; the candidates whose plan the clearance check rejects are
                 dropped, the lowest predicted loss wins (changing the pelvis height from the last reach costs 0.5 cm);
                 none left = the robot declines the grasp (counted as a failure, reported apart); at the lift, with the box already
                 held, the best candidate is used anyway
    inv_prop     inv + the proprioceptive servo

v2 methods (2026-11-06, the three fixes, each a switch in Opt; the five methods above keep their behaviour):
    hold         the late servo keeps applying its stored lead while the reference moves (v1 dropped it, so a lead built
                 at the pre-grasp vanished for the whole entry) and keeps its target between reaches. The inverted command
                 is lowered by that lead: it already compensates, and the lead on top would count the error twice
    lift         the lift is the direct command with the proprioceptive servo (the model never saw a held box)
    check        strict: as in v1; off: nothing declined; cal: thresholds from GRASP_CAL, set on the calib placements
    direct_hold, inv_hold: every reach with the servo, hold; inv_lift: eyes closed up to the grip, then the servo lifts;
    inv_free, inv_cal, inv_hold_free, inv_hold_cal: the same with the check off or calibrated.
    cont         the entry's path starts this fraction of the way from the measured jaw to the last command. Motor.reach
                 paths from the measured jaw, which drops the last command's compensation at the start of the reach, and a
                 hand that holds a command off by c lags about c / 2 in motion: inv_half_free, inv_half_cal = 0.5. Compare
                 families, not runs: a 1 cm stance shift flips about 16 of 144 outcomes. Five stances each, calib: 0.5 vs 0 is
                 107.8 vs 105.4 at 15 cm/s (within noise), 110.4 vs 105.8 at 8 cm/s
    gate         stop GATE_M before the grasp point first. Worse (calib 86): the last short reach swings off at its start
    name@f15o10t0p30u5  the same method from another stance: 15 cm further forward, 10 cm towards the reaching hand's side, turned
                 0 deg that way (a direct walk there); the jaw pitched 30 deg down for the pre-grasp and the entry (it raises the
                 forearm and the elbow off the cabinet's top edge); the pre-grasp 5 cm above the grasp height; x8 (optional like p and u):
                 the pre-grasp 8 cm further out of the cabinet than PRE_OUT_M; s8 (optional): the entry at 8 cm/s (MotorConfig.reach_speed
                 is 15, the speed all the training data ran at). inv_half_cal@f0o0t0s8: five stances on the test set 164.0 of 216
                 against 157.2 at 15 cm/s. It halves the across offset at the box front (> 1.5 cm: 9.1 -> 5.0 %), the end offset
                 stays. Final setting: inv_half_cal@f0o0t0s8v9g100h35 with GRASP_PATH, 167.8 (see g and h below). w9 (optional): the same straight entry with a point on it where the jaw is 9 cm before the
                 grasp point: Motor.reach ends the wrist's turn at the first via point, so only the wrist changes; v9 (optional):
                 a via point there too, its command inverted by the self model like the entry's; a100 (optional): the entry's wrist
                 attitude starts that percentage of the way from the measured one to the last command (the attitude's cont);
                 e20 (optional, with v): that inverted via point 20 mm further out across, to the reaching hand's side;
                 g60 (optional, with v, needs GRASP_PATH): moved against 60 % of the swing path_model.py predicts for this entry;
                 h35 (optional, with g): no via point where the swing model's members disagree by more than 0.35 cm. Where they
                 agree it is right (correlation 0.80 and 0.69 in the lower two quarters of the spread, -0.13 in the top one)
                 r1 (optional): the entry's whole-body IK starts from the last reference pose instead of the measured joints
                 (on the top shelf the measured-seeded plan jumps posture family and the joint-space blend swings the reference
                 jaw ~28 cm, the hand dives under the top board); rN, N >= 2: only where that swing exceeds N cm. Not adopted:
                 r1 calib 111.8 -> 110.0 (other shelves land off, the self model learned the old plans); r10 short 86.8,
                 calib 113.6, test 167.8 -> 169.0 (35 wins 29 losses, 95 % -4.0..+7.4). Fix the planner and retrain instead
                 c1 (optional, with r1): the model and the clearance check also get the plan the entry really runs: reach_kw's start, speed and IK seed go to
                 plan_of. Until 11-08 they saw a plain reach's plan (measured jaw, 0.15 m/s, measured seed), so r1 fed the model a plan that was not the one executed.
                 The planner options are now Motor.reach's start / start_rot / ik_from_ref; the rows of r1 and rN replay the one-shot wrappers bit for bit (top shelf short, 48 of 48).
                 With the planner fixed and the data regenerated under it (data_grasp2, invert/wrist_grasp2, path/path_swing2.pt): inv_half_cal@f0o0t0s8v9g100r1c1, five
                 stances, short 84.0 -> 90.0, calib 111.8 -> 109.6, test 167.8 -> 169.2 (86 wins 79 losses, 95 % -8.2..+11.2: a tie). The jaw lands better (height off at
                 the grip, median, 0.95 m shelf 0.84 -> 0.24 cm, top 3.48 -> 0.75 cm); what is left is not landing. No gate: h only ever lowers the score here, always via.
                 n12 (optional, with p): the hand crosses the board at the height of the point 12 cm back along its own axis, then goes in along its fingers (leg "over"); k4 (optional): the
                 grasp point 4 cm above the box's centre. Not adopted (2026-11-09): a hand pitched down needs the wrist at 1.286 + 0.1515 sin p m (p 30: 1.36, p 90: 1.44), ScaleBFM's wrist tops out near
                 1.26 m. calib top shelf f15 p15 n12: 1 of 36, f15 p30 n12 k4: 1 of 36, against 10.6 of 36 for the level hand. Straight down is out of reach of the arm itself (8-21 cm short at every stance).
GRASP_SET=calib draws placements none of the 216 use (depths 0.15, 0.25, 0.30; laterals 0.10, 0.20, other seeds);
GRASP_SET=short the same for short entries (depths 0.10, 0.12; laterals 0.10, 0.20; other seeds again);
GRASP_OUT names the output directory, GRASP_MODEL the predictor file, GRASP_PATH the swing model (every entry then logs it),
GRASP_SHELF one shelf index to run alone.

Clearance check (the robot's own plan against the cabinet's geometry): reach()'s whole-body IK plan for the goal at a
candidate pelvis height (where the self model says the body will end; the inverted command's own plan is off by the
error it pre-compensates),
every waypoint pose and two points between neighbours (the joint-space interpolation the reference follows), set in a
copy of the model whose cabinet geoms carry a CLEAR_M margin; the upper arm or elbow within CLEAR_M of the cabinet,
or the wrist or hand inside it, rejects the candidate (the torso, legs, head and fingers are not judged; see Clear). The direct runs record the
check's verdict on their own plans without acting on it (does it predict the collisions?).

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    $PY scripts/experiments/reachwm_20261030/grasp_exp.py h --procs 6      # all setups x methods
    $PY scripts/experiments/reachwm_20261030/grasp_exp.py d                # report
    $PY scripts/experiments/reachwm_20261030/grasp_exp.py _w smoke 0 '[[17, 0, "inv"]]'   # one run in the foreground
Writes outputs/reachwm_20261030/grasp_exp/.
"""

from __future__ import annotations

import os

os.environ.setdefault("REACHWM_GOAL_REF", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grasp_pilot as GP  # noqa: E402

WI, PV, IE, PI, SG = GP.WI, GP.PV, GP.IE, GP.PI, GP.SG
OUT = IE.OUT.parent / os.environ.get("GRASP_OUT", "grasp_exp")
SET = os.environ.get("GRASP_SET", "test")
PATH = {"model": None}  # path_model.py's swing model, loaded by the worker from GRASP_PATH
ONLY_SHELF = os.environ.get("GRASP_SHELF", "")
CAL = json.loads(
    os.environ.get("GRASP_CAL", "{}")
)  # {"arm_cm": .., "hand_cm": ..}: the smallest clearance the check allows
if os.environ.get("GRASP_MODEL"):
    GP.MODEL = Path(os.environ["GRASP_MODEL"])
WIDTH = 0.08
LATERALS = (0.05, 0.15, 0.25)  # box centre this far to the reaching hand's side of the cabinet's centre line
STARTS = 3
SEED = 4300000
METHODS = ("direct", "direct_vis", "direct_prop", "inv", "inv_prop")


@dataclasses.dataclass(frozen=True)
class Opt:
    inv: bool = False  # the self model's command
    servo: str = ""  # "": none, "all": every reach, "lift": the lift only, "entry": the entry (late: after it stops) and the lift
    vis: bool = False  # the visual gate on the servo
    hold: bool = False  # see the v2 note above
    check: str = "strict"  # strict, off, cal
    lift_direct: bool = False  # the lift is the goal as the command, whatever the method
    conv: bool = (
        False  # hold only a lead that converged: the last reach ended within HOLD_TOL_M of its goal, else clear it
    )
    cont: float = (
        0.0  # the entry's path starts this fraction of the way from the measured jaw to the last command (see below)
    )
    gate: bool = (
        False  # stop GATE_M before the grasp point, then the last short reach: the end state is what the model knows
    )
    perp: bool = (
        False  # enter with the lead across the entry only: along it the hand overshoots into the box (direct_hold)
    )
    step: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )  # stance: forward m, towards the reaching hand's side m, turned that way deg
    pitch: float = 0.0  # the jaw pitched this many deg down, world frame
    up: float = 0.0  # the pre-grasp this far above the grasp height, m
    back: float = 0.0  # the pre-grasp this much further out of the cabinet, m
    entry_speed: float = 0.0  # the entry's speed, m/s; 0: MotorConfig.reach_speed
    via_line: float = 0.0  # a via point on the entry's straight line, this far before the grasp point, m; 0: none
    via_inv: float = 0.0  # a via point this far before the grasp point, its command inverted, m; 0: none
    via_out: float = 0.0  # the inverted via point this much further out across, to the reaching hand's side, m
    via_ff: float = (
        0.0  # the inverted via point moved against this fraction of the swing path_model predicts (GRASP_PATH)
    )
    via_gate: float = (
        0.0  # no via point at all where the swing model's members disagree by more than this, cm; 0: always
    )
    att: float = (
        0.0  # the entry's wrist attitude starts this fraction of the way from the measured one to the last command
    )
    seed_ref: int = 0  # 1: the entry's whole-body IK starts from the reference pose, not the measured one; N >= 2: only where the measured-seeded plan's first blend swings the jaw > N cm
    same_plan: bool = False  # c1: the model and the clearance check see the plan the entry really gets (its start, speed and IK seed), not the plan of a plain reach
    along: float = 0.0  # n12: the pre-grasp this far back from the grasp point along the jaw's own axis, m, so a pitched hand enters along its fingers; 0: outside the cabinet front
    grip_up: float = 0.0  # k4: the grasp point this far above the box's centre, m


OPTS = {
    "direct": Opt(), "direct_vis": Opt(servo="all", vis=True), "direct_prop": Opt(servo="all"),
    "inv": Opt(inv=True), "inv_prop": Opt(inv=True, servo="all"),
    "direct_hold": Opt(servo="all", hold=True),
    "inv_lift": Opt(inv=True, servo="lift", lift_direct=True),
    "inv_hold": Opt(inv=True, servo="all", hold=True, lift_direct=True),
    "inv_free": Opt(inv=True, servo="lift", lift_direct=True, check="off"),
    "inv_hold_free": Opt(inv=True, servo="all", hold=True, lift_direct=True, check="off"),
    "inv_cal": Opt(inv=True, servo="lift", lift_direct=True, check="cal"),
    "inv_cont_free": Opt(inv=True, servo="lift", lift_direct=True, check="off", cont=1.0),
    "inv_half_free": Opt(inv=True, servo="lift", lift_direct=True, check="off", cont=0.5),
    "inv_half_cal": Opt(inv=True, servo="lift", lift_direct=True, check="cal", cont=0.5),
    "inv_gate_free": Opt(inv=True, servo="lift", lift_direct=True, check="off", gate=True),
    "inv_gate_cal": Opt(inv=True, servo="lift", lift_direct=True, check="cal", gate=True),
    "inv_late_free": Opt(inv=True, servo="entry", lift_direct=True, check="off"),
    "inv_late_cal": Opt(inv=True, servo="entry", lift_direct=True, check="cal"),
    "direct_conv": Opt(servo="all", hold=True, conv=True),
    "direct_perp": Opt(servo="all", hold=True, conv=True, perp=True),
    "inv_perp": Opt(inv=True, servo="all", hold=True, lift_direct=True, conv=True, perp=True),
    "inv_perp_free": Opt(inv=True, servo="all", hold=True, lift_direct=True, conv=True, perp=True, check="off"),
    "inv_perp_cal": Opt(inv=True, servo="all", hold=True, lift_direct=True, conv=True, perp=True, check="cal"),
    "inv_conv": Opt(inv=True, servo="all", hold=True, lift_direct=True, conv=True),
    "inv_conv_free": Opt(inv=True, servo="all", hold=True, lift_direct=True, conv=True, check="off"),
    "inv_conv_cal": Opt(inv=True, servo="all", hold=True, lift_direct=True, conv=True, check="cal"),
    "inv_hold_cal": Opt(inv=True, servo="all", hold=True, lift_direct=True, check="cal"),
}  # fmt: skip


def opt_of(method: str) -> Opt:
    """`name`, or `name@f15o10t0` for that method from another stance (see the v2 note)."""
    base, _, at = method.partition("@")
    o = OPTS[base]
    if at:
        pat = r"f(-?\d+)o(-?\d+)t(-?\d+)(?:p(\d+))?(?:u(\d+))?(?:x(\d+))?(?:s(\d+))?(?:w(\d+))?(?:v(\d+))?(?:e(\d+))?(?:g(\d+))?(?:h(\d+))?(?:a(\d+))?(?:r(\d+))?(?:c(\d+))?(?:n(\d+))?(?:k(\d+))?"
        f, out, t, p, u, x, sp, w, v, e, g, h, a, r, c, n, k = (float(c or 0) for c in re.fullmatch(pat, at).groups())
        o = dataclasses.replace(o, step=(f / 100, out / 100, t), pitch=p, up=u / 100, back=x / 100, entry_speed=sp / 100,
                                via_line=w / 100, via_inv=v / 100, via_out=e / 1000, via_ff=g / 100, via_gate=h / 100, att=a / 100, seed_ref=int(r), same_plan=bool(c), along=n / 100, grip_up=k / 100)  # fmt: skip
    return o


CLEAR_M = 0.03
GATE_M = 0.10  # the fingertips meet the box's front with the jaw centre 7.7 cm before the grasp point (median of the first box motion)
HOLD_TOL_M = 0.02  # on the low shelf the hand stops 10-20 cm short of the pre-grasp and the lead winds up to the cap: no estimate of anything
PZ_STEP = 0.06
SWITCH_PZ_CM = 0.5


def placements() -> list[dict]:
    out = []
    sets = {"calib": ((0.15, 0.25, 0.30), (0.10, 0.20)), "short": ((0.10, 0.12), (0.10, 0.20))}
    depths, lats = sets.get(SET, (GP.DEPTHS, LATERALS))
    for si, z in enumerate(GP.SHELVES):
        for d in depths:
            for lat in lats:
                for side in ("right", "left"):
                    sd = 1.0 if side == "right" else -1.0
                    out.append(dict(pid=len(out), shelf=si, depth=d, lateral=lat, side=side, width=WIDTH,
                                    body=f"pilot_box{int(round(100 * WIDTH))}",
                                    box=[GP.FRONT_X + d, -sd * lat, z + GP.BOARD_T / 2 + GP.BOX_HALF_H + 0.001]))  # fmt: skip
    return out


def start_of(pid: int, si: int) -> tuple[list, float]:
    rng = np.random.default_rng(SEED + 10 * pid + si + {"calib": 500000, "short": 900000}.get(SET, 0))
    return [GP.FRONT_X - GP.STANCE_BACK_M - float(rng.uniform(1.5, 2.5)), float(rng.uniform(-0.2, 0.2))], \
        float(math.radians(rng.uniform(-15, 15)))  # fmt: skip


class Clear:
    """The clearance check: plan poses in a model copy whose cabinet geoms have a CLEAR_M margin."""

    def __init__(self, env):
        import mujoco

        self.mj = mujoco
        sm = env.sim.m
        self.M = copy.deepcopy(sm)
        cab = sm.body("cab").id
        self.cab = np.array([sm.geom_bodyid[g] == cab for g in range(sm.ngeom)])
        self.M.geom_margin[self.cab] = CLEAR_M
        root = sm.body_rootid[sm.body("pelvis").id]
        self.robot = np.array([sm.body_rootid[b] == root for b in range(sm.nbody)])
        self.finger = np.array(["finger" in sm.body(b).name for b in range(sm.nbody)])
        # what blocks a reach is the arm: the shoulder-to-elbow and elbow links keep CLEAR_M (on the top shelf the
        # elbow's plan went 5.2 cm into the cabinet and the hand stayed 15-25 cm short), the wrist and hand fail on
        # penetration only. The torso, legs, head and fingers are not judged: in the first full run the check declined
        # 27 setups for a plan with the torso 1-6 cm inside a board edge or the open fingers 1-3 cm off the front,
        # and the direct command with the servo grasped 19 of those (the torso brushes the edge and is pushed back)
        nm = [sm.body(b).name for b in range(sm.nbody)]
        self.thr = np.array([CLEAR_M if any(k in n for k in ("shoulder", "elbow")) else
                             (0.0 if any(k in n for k in ("wrist", "hand", "rubber")) and "finger" not in n else np.nan)
                             for n in nm])  # fmt: skip
        self.D = mujoco.MjData(self.M)
        self.qadr = np.asarray(env.m.kp.qadr)
        self.d0 = env.sim.d

    def poses(self, plan) -> list:
        poses = plan[0]
        out = [poses[0]]
        for a, b in zip(poses[:-1], poses[1:]):
            for s in (1 / 3, 2 / 3, 1.0):
                q = (1 - s) * a.quat + s * (b.quat if float(np.dot(a.quat, b.quat)) >= 0 else -b.quat)
                out.append((a.pos + s * (b.pos - a.pos), q / np.linalg.norm(q), a.joints + s * (b.joints - a.joints)))
        return [(p.pos, p.quat, p.joints) if hasattr(p, "pos") else p for p in out]

    def check(self, plan, fingers_ok: bool) -> dict:
        """The worst violation over the plan: how far a robot body comes inside its allowed clearance of the cabinet
        (CLEAR_M, 0 for the legs), the body, ok = no violation."""
        M, D, mj = self.M, self.D, self.mj
        worst, part, near = 0.0, "", CLEAR_M
        arm, hand = CLEAR_M, CLEAR_M  # the nearest the arm links and the wrist and hand come to the cabinet
        for pos, quat, joints in self.poses(plan):
            D.qpos[:] = self.d0.qpos
            D.qpos[:3], D.qpos[3:7], D.qpos[self.qadr] = pos, quat, joints
            mj.mj_forward(M, D)
            for i in range(D.ncon):
                c = D.contact[i]
                for ga, gb in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                    b = M.geom_bodyid[gb]
                    if not (self.cab[ga] and self.robot[b]) or np.isnan(self.thr[b]):
                        continue
                    near = min(near, float(c.dist))
                    if self.thr[b] > 0:
                        arm = min(arm, float(c.dist))
                    else:
                        hand = min(hand, float(c.dist))
                    v = self.thr[b] - float(c.dist)
                    if v > worst:
                        worst, part = v, M.body(b).name
        return dict(
            ok=bool(worst <= 0.0),
            clear_cm=round(100 * near, 2),
            part=part,
            arm_cm=round(100 * arm, 2),
            hand_cm=round(100 * hand, 2),
        )


class Eyes:
    """Can the head camera see the box: in the field of view and a ray reaching it first (box centre or a front-face
    corner). Also gates the visual servo: while the box is unseen the servo target is the measured jaw."""

    def __init__(self, env, body: str):
        import mujoco

        self.mj, self.env = mujoco, env
        sm = env.sim.m
        self.box_body = sm.body(body).id
        self.box_geom = sm.geom(f"{body}_geom").id
        self.cam = env.cam
        self.cam_body = sm.cam_bodyid[self.cam]
        self.half_v = math.radians(sm.cam_fovy[self.cam]) / 2
        self.half_h = math.atan(math.tan(self.half_v) * 4 / 3)
        self.hw = WIDTH / 2
        self.gate = False  # True: apply to the servo
        self.side, self.goal = "right", None
        self.seen_ticks, self.ticks, self.last_seen = 0, 0, -1
        self.groups = np.array([1, 1, 1, 0, 0, 0], np.uint8)

    def visible(self) -> bool:
        mj, sim = self.mj, self.env.sim
        d, sm = sim.d, sim.m
        cp = d.cam_xpos[self.cam].copy()
        Rc = d.cam_xmat[self.cam].reshape(3, 3)
        c = d.xpos[self.box_body]
        Rb = d.xmat[self.box_body].reshape(3, 3)
        pts = [c] + [
            c + Rb @ np.array([-0.03, sy * self.hw * 0.8, sz * GP.BOX_HALF_H * 0.8]) for sy in (-1, 1) for sz in (-1, 1)
        ]
        gid = np.zeros(1, np.int32)
        for p in pts:
            v = p - cp
            loc = Rc.T @ v  # camera frame: looks along -z
            if (
                loc[2] >= 0
                or abs(math.atan2(loc[0], -loc[2])) > self.half_h
                or abs(math.atan2(loc[1], -loc[2])) > self.half_v
            ):
                continue
            dist = mj.mj_ray(sm, d, cp, v / np.linalg.norm(v), self.groups, 1, self.cam_body, gid)
            if dist >= 0 and int(gid[0]) == self.box_geom:
                return True
        return False

    def tick(self) -> None:
        m = self.env.m
        seen = self.visible()
        self.ticks += 1
        if seen:
            self.seen_ticks += 1
            self.last_seen = self.ticks
        if self.gate and self.goal is not None:
            tgt = self.goal if seen else m.jaw_centre(self.side)[0]
            m.servo_target = IE.KeepTarget({self.side: np.asarray(tgt, float)})


def episode(env, rec, watch, gate, clear, eyes, model, pl: dict, si: int, method: str) -> dict:
    import gen_reach as G

    m, sim = env.m, env.sim
    side = pl["side"]
    start, yaw0 = start_of(pl["pid"], si)
    watch.clear()
    watch.mark("walk")
    eyes.gate, eyes.goal, eyes.side = False, None, side
    m.reset(tuple(start), yaw0)
    for w, b in env.boxes.items():
        park = np.array([3.0 + GP.WIDTHS.index(w), 3.0, 0.2])
        m.place(
            f"pilot_box{int(round(100 * w))}",
            np.array(pl["box"]) if w == WIDTH else park,
            np.array([1.0, 0.0, 0.0, 0.0]),
        )
    m.hold(1.0)
    o = opt_of(method)
    sd = 1.0 if side == "right" else -1.0
    stance = np.array([GP.FRONT_X - GP.STANCE_BACK_M, 0.0]) + np.array([o.step[0], -sd * o.step[1]])
    m.walk_to(stance, yaw=0.0 - sd * math.radians(o.step[2]))
    res = dict(
        pid=pl["pid"], start=si, method=method, shelf=pl["shelf"], depth=pl["depth"], lateral=pl["lateral"], side=side
    )
    if m.fell_at is not None:
        return dict(res, fell="walk", success=False)
    m.hold(0.5)
    ob = env.boxes[WIDTH]
    box0 = sim.d.xpos[ob].copy()
    pv = m.pelvis()[0]
    res["stance_cm"] = (100 * (pv[:2] - stance)).tolist()
    grasp = np.r_[box0[0] + 0.01, box0[1], box0[2] + o.grip_up]
    pre = np.array([GP.FRONT_X - GP.PRE_OUT_M - o.back, grasp[1], grasp[2] + o.up])
    rot_w = G.rot_y(math.radians(o.pitch)) if o.pitch else np.eye(3)
    over = (
        grasp - o.along * rot_w[:, 0]
    )  # n: the hand crosses the board at this height, then goes in along its own axis
    if o.along:
        pre = np.array([pre[0], pre[1], max(pre[2], over[2])])
    pz_rule = GP.pelvis_for(float(grasp[2]))
    cfg = m.cfg
    if o.servo:
        gate.mode, gate.radius, gate.hold = "late_pos", None, o.hold
        m.cfg = dataclasses.replace(cfg, servo_ki_rot=0.0)
    eyes.seen_ticks, eyes.ticks, eyes.last_seen = 0, 0, -1
    state = dict(pz=pz_rule)

    def passes(c: dict) -> bool:
        if o.check == "off":
            return True
        return c["arm_cm"] >= CAL["arm_cm"] and c["hand_cm"] >= CAL["hand_cm"] if o.check == "cal" else c["ok"]

    def reach_kw(stage: str) -> dict:
        """Motor.reach's planner options for this reach. The path of the entry starts o.cont of the way from the measured jaw
        to the last command: Motor.reach paths from the measured jaw, so at the start the reference drops the compensation
        the last command carried and the hand swings back by about half of it (it holds a command off by c, in motion it
        lags c / 2). o.att: the attitude as well (reach starts its turn from the measured wrist, a jump of the reference)."""
        kw: dict = {}
        if stage not in ("entry", "over"):
            return kw
        if o.entry_speed:
            kw["speed"] = o.entry_speed
        if o.cont and "last_cmd" in state:
            p_, R_ = m.jaw_centre(side)
            kw["start"] = p_ + o.cont * (state["last_cmd"] - p_)
            if o.att:
                kw["start_rot"] = R_ @ G.rotvec(o.att * WI.rotvec_of(R_.T @ rot_w))
        if o.seed_ref == 1:
            kw["ik_from_ref"] = True
        return kw

    def do(stage: str, goal_w, chain: int) -> dict | None:
        """One reach; None = declined (no candidate passes the clearance check). o.same_plan: every plan the model and the
        check look at during this reach is planned with the options the reach itself gets."""
        rec.start(side)
        m.hold(G.PRE_S)
        rec.on = False
        kw = reach_kw(stage)
        if not (o.same_plan and kw):
            return one_reach(stage, goal_w, chain, kw)
        plan_of0 = PI.plan_of
        PI.plan_of = lambda *a: plan_of0(*a, **kw)
        try:
            return one_reach(stage, goal_w, chain, kw)
        finally:
            PI.plan_of = plan_of0

    def one_reach(stage: str, goal_w, chain: int, kw: dict) -> dict | None:
        info = dict(goal_w=np.asarray(goal_w, float).tolist())
        fingers_ok = stage != "pregrasp"
        servo = (
            o.servo == "all" or (o.servo == "lift" and stage == "lift") or (o.servo == "entry" and stage != "pregrasp")
        )
        by_model = o.inv and not (stage == "lift" and o.lift_direct)
        if by_model:
            st = GP.state_here(env, rec, side, np.asarray(goal_w, float), rot_w, chain)
            cands = []
            base = state["pz"]
            pzs = sorted({base, *(float(np.clip(base + k * PZ_STEP, 0.52, 0.76)) for k in (-1, 1) if base > 0)} |
                         ({-1.0} if grasp[2] >= 0.5 else set()) | ({0.70} if base < 0 else set()))  # fmt: skip
            for pz in pzs:
                evs = WI.fixed_point(model, m, st, st["goal"], pz, None)
                best = min(evs, key=lambda ev: ev["l_pos"])
                tw = GP.to_w(st, best["t"])
                # the check runs on the plan for the goal itself, not for the inverted command: the model says the hand
                # ends at the goal, so that plan is where the arm will really be (the inverted command's plan sits off
                # by the very error it pre-compensates: on the bottom shelf 3.6 cm lower, 0.6 cm from the board)
                ck = clear.check(PI.plan_of(m, side, np.asarray(goal_w, float), rot_w, pz), fingers_ok)
                score = float(np.sqrt(best["l_pos"]) * 100) + (SWITCH_PZ_CM if pz != base else 0.0)
                cands.append(
                    dict(
                        pz=pz,
                        t=tw,
                        pred_err=float(best["pred_err"]),
                        pred_std=float(best["pred_std"]),
                        score=score,
                        **ck,
                    )
                )
                state.setdefault("ev", {})[pz] = best
            info["cands"] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()} for c in cands]
            ok = [c for c in cands if passes(c)]
            if not ok and stage == "lift":  # the box is already held: lift with the best candidate rather than let go
                ok = cands
            if not ok:
                state["declined"] = info
                return None
            pick = min(ok, key=lambda c: c["score"])
            target_w, pz = pick["t"], pick["pz"]
            info.update(pz=pz, pred_err=pick["pred_err"], pred_std=pick["pred_std"], clear_cm=pick["clear_cm"])
        else:
            target_w, pz = np.asarray(goal_w, float), state["pz"]
            ck = clear.check(PI.plan_of(m, side, target_w, rot_w, pz), fingers_ok)  # recorded only
            info.update(pz=pz, clear_cm=ck["clear_cm"], clear_ok=ck["ok"], clear_part=ck["part"])
        state["pz"] = pz
        if o.conv and state.get("last_err_m", 0.0) > HOLD_TOL_M:
            m.servo.pop(("pos", side), None)
        if o.perp and stage == "entry" and ("pos", side) in m.servo:
            u = np.asarray(goal_w, float) - m.jaw_centre(side)[0]
            u /= max(float(np.linalg.norm(u)), 1e-9)
            m.servo[("pos", side)] = m.servo[("pos", side)] - u * float(m.servo[("pos", side)] @ u)
        if o.hold and by_model:  # the command already compensates; the stored lead stays on top of it
            lead = m.servo.get(("pos", side), np.zeros(3))
            target_w = np.asarray(target_w, float) - lead
            info["lead_cm"] = (100 * lead).tolist()
        if servo:
            gate.reset()
            m.servo_target = IE.KeepTarget({side: np.asarray(goal_w, float)})
            eyes.gate, eyes.goal = o.vis, np.asarray(goal_w, float)
        via = ()
        if stage == "entry" and by_model and PATH["model"] is not None:
            import torch

            ev = state["ev"][pz]
            u_ = np.r_[st["sd"], ev["t"], st["rot6"], float(pz < 0), 0.0 if pz < 0 else pz]
            F = lambda a: torch.as_tensor(np.asarray(a, np.float32)[None])  # noqa: E731
            with torch.no_grad():
                sw, sw_std = PATH["model"].predict(F(ev["x0"]), F(u_))
            z = PATH["model"].scaled(F(ev["x0"]), F(u_))[0].abs()
            info.update(
                swing_cm=sw[0].tolist(), swing_std=sw_std[0].tolist(), swing_z=[float(z.max()), int(z.argmax())]
            )
        if stage == "entry" and o.via_line:
            a = m.jaw_centre(side)[0]
            a = a + o.cont * (state.get("last_cmd", a) - a)
            b = np.asarray(target_w, float)
            u = float(np.clip((goal_w[0] - o.via_line - a[0]) / (b[0] - a[0]), 0.05, 0.95))
            via = (a + u * (b - a),)
        if stage == "entry" and o.via_inv and (not o.via_gate or info["swing_std"][0] <= o.via_gate):
            st_v = GP.state_here(
                env, rec, side, np.asarray(goal_w, float) - np.array([o.via_inv, 0.0, 0.0]), rot_w, chain
            )
            ev = min(WI.fixed_point(model, m, st_v, st_v["goal"], pz, None), key=lambda e: e["l_pos"])
            shift = np.array([0.0, sd * o.via_out, 0.0])
            if o.via_ff:
                shift = shift + st["Rh"] @ np.array([0.0, o.via_ff * info["swing_cm"][0] / 100, 0.0])
            via = (GP.to_w(st_v, ev["t"]) - shift,)
        if via:
            info["via_w"] = np.asarray(via[0], float).tolist()
        gated = {}
        if o.seed_ref >= 2 and stage == "entry":
            # Motor.reach's whole-body IK starts from the measured joints with the waist pitched 0.45 rad forward (the
            # WholeBodyIK seed), while its first leg blends in joint space from the last reference pose. On the top shelf
            # ScaleBFM ends the pre-grasp in an arm posture far from that reference, the new solution lands in another
            # family (waist yaw 28 -> -56 deg, pitch -30 -> +30 deg at both limits) and the blend swings the reference
            # jaw 18 cm down within 0.3 s: the hand dives under the top board. r1 (reach_kw, Motor.reach's ik_from_ref): the
            # solver is built on the reference pose and seeded exactly there, so the reach continues the reference's family.
            # rN, N >= 2, keeps the measured-seeded plan unless its first blend swings the reference jaw more than N cm off
            # the straight segment.
            orig_solver, orig_solve, ref0 = m._solver, m._solve, m.ref(m.t)

            def make_solver(pelvis_height_target=None, at=None):
                m._solver = orig_solver
                gated["pz"] = pelvis_height_target
                return orig_solver(pelvis_height_target, at=at)

            m._solver = make_solver

            def solve_gated(solver, positions, rotations, **kws):
                if "sv" not in gated:
                    ik = orig_solve(solver, positions, rotations, **kws)
                    gated["swing"] = G.blend_swing(m, side, ref0, m._pose_of(ik, m.yaw()))
                    if 100 * gated["swing"] <= o.seed_ref:
                        gated["sv"] = solver
                        return ik
                    gated["sv"], gated["ref"] = orig_solver(gated.get("pz"), at=ref0), True
                return orig_solve(gated["sv"], positions, rotations, **kws)

            m._solve = solve_gated
        pelvis_z = None if pz < 0 else float(pz)
        rec.plan = None
        try:
            out = m.reach(side, np.asarray(target_w, float), rot_w, via=via, pelvis_z=pelvis_z, servo=servo, **kw)
        finally:
            m.__dict__.pop("_solve", None)
        if "swing" in gated:
            info.update(blend_swing_cm=round(100 * gated["swing"], 2), seeded_ref=bool(gated.get("ref")))
        elif o.seed_ref == 1 and stage == "entry":
            info.update(
                blend_swing_cm=round(100 * G.blend_swing(m, side, rec.plan[0][0], rec.plan[0][1]), 2), seeded_ref=True
            )
        state["last_cmd"] = np.asarray(target_w, float)
        if m.fell_at is None:
            m.hold(G.POST_S)
        eyes.gate, eyes.goal = False, None
        if not o.hold:
            m.servo_target = {}
        jaw = m.jaw_centre(side)[0]
        state["last_err_m"] = float(np.linalg.norm(jaw - np.asarray(goal_w)))
        info.update(target_w=np.asarray(target_w, float).tolist(), ok=bool(out.ok),
                    err_cm=float(100 * np.linalg.norm(jaw - np.asarray(goal_w))))  # fmt: skip
        return info

    def finish(r: dict) -> dict:
        m.cfg = cfg
        gate.mode, gate.hold = "orig", False
        r.update(contacts=dict(watch.contact), contact_parts=dict(sorted(watch.parts.items(), key=lambda kv: -kv[1])[:6]),
                 seen_frac=eyes.seen_ticks / max(eyes.ticks, 1))  # fmt: skip
        return r

    legs = [("pregrasp", pre, 0)] + ([("gate", np.r_[grasp[0] - GATE_M, grasp[1:]], 1)] if o.gate else []) + ([("over", over, 1)] if o.along else []) + [("entry", grasp, 1)]  # fmt: skip
    for stage, goal, chain in legs:
        watch.mark(stage)
        if stage == "entry":
            eyes.seen_ticks, eyes.ticks = 0, 0
        info = do(stage, goal, chain)
        if info is None:
            return finish(dict(res, declined=stage, declined_info=state.get("declined"), success=False))
        res[stage] = info
        if m.fell_at is not None:
            return finish(dict(res, fell=stage, success=False))
    res["seen_entry"] = eyes.seen_ticks / max(eyes.ticks, 1)  # share of the entry the box was in sight
    res["seen_at_grip"] = bool(eyes.visible())
    jaw = m.jaw_centre(side)[0]
    res["at_grip_cm"] = (100 * (jaw - grasp)).tolist()
    res["pushed_cm"] = float(100 * np.linalg.norm(sim.d.xpos[ob][:2] - box0[:2]))
    watch.mark("grip")
    res["grip_held"] = bool(m.grip(side).ok)
    watch.mark("lift")
    info = do("lift", m.jaw_centre(side)[0] + np.array([0.0, 0.0, GP.LIFT_M]), 1)  # the data's chain flag is 0 or 1
    if info is None:
        return finish(dict(res, declined="lift", declined_info=state.get("declined"), success=False))
    res["lift"] = info
    if m.fell_at is None:
        watch.mark("hold")
        m.hold(3.0)
    obj = sim.d.xpos[ob]
    gripped = side in m.gripping(ob)
    res.update(fell=None if m.fell_at is None else "grip/lift", gripped=bool(gripped), rise_cm=float(100 * (obj[2] - box0[2])),
               success=bool(gripped and obj[2] > box0[2] + 0.02 and m.fell_at is None))  # fmt: skip
    return finish(res)


def worker(wid: int, jobs: list) -> None:
    import gen_reach as G
    import torch

    torch.set_num_threads(1)
    env = GP.make_env()
    rec = G.Recorder(env)
    watch = GP.Watch(env)
    gate = SG.Gate(env.m)
    clear = Clear(env)
    eyes = Eyes(env, f"pilot_box{int(round(100 * WIDTH))}")
    orig = env.m.tick

    def tick():  # the eyes look (and gate the visual servo) before every motor tick
        eyes.tick()
        return orig()

    env.m.tick = tick
    model = torch.load(GP.MODEL, map_location="cpu", weights_only=False).eval()
    if os.environ.get("GRASP_PATH"):
        import path_model

        PATH["model"] = path_model.load(os.environ["GRASP_PATH"])
    P = {p["pid"]: p for p in placements()}
    res = []
    for pid, si, method in jobs:
        tic = time.time()
        r = episode(env, rec, watch, gate, clear, eyes, model, P[pid], si, method)
        np.save(OUT / "q" / f"{pid}_{si}_{method}.npy", np.array(watch.q, np.float32))
        r["marks"] = watch.marks
        res.append(r)
        print(f"w{wid} pid {pid} start {si} {method}: success {r['success']} declined {r.get('declined')} fell {r.get('fell')} "
              f"at_grip {np.round(r.get('at_grip_cm', [np.nan] * 3), 1)} pushed {r.get('pushed_cm', float('nan')):.1f} "
              f"seen {r.get('seen_entry', float('nan')):.2f} contacts {sum(r.get('contacts', {}).values())} | {time.time() - tic:.0f} s",
              flush=True)  # fmt: skip
        json.dump(res, open(OUT / f"h_w{wid}.json", "w"))


def launch(procs: int, gpus: str, methods: tuple[str, ...] = METHODS) -> None:
    jobs = [
        (p["pid"], si, mth)
        for p in placements()
        if ONLY_SHELF in ("", str(p["shelf"]))
        for si in range(STARTS)
        for mth in methods
    ]
    (OUT / "q").mkdir(parents=True, exist_ok=True)
    cards = gpus.split(",")
    ps = []
    for w in range(procs):
        envv = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=cards[w % len(cards)], REACHWM_GOAL_REF="1")
        log = open(OUT / f"h_w{w}.log", "w")
        ps.append(subprocess.Popen([sys.executable, __file__, "_w", "h", str(w), json.dumps(jobs[w::procs])], env=envv,
                                   stdout=log, stderr=subprocess.STDOUT))  # fmt: skip
    for p in ps:
        p.wait()
    print("done", [p.returncode for p in ps])


def report() -> None:
    R = []
    for f in sorted(OUT.glob("h_w*.json")):
        R += json.load(open(f))
    by = {}
    for r in R:
        by.setdefault((r["pid"], r["start"]), {})[r["method"]] = r
    keys = [k for k, v in by.items() if all(mm in v for mm in METHODS)]
    L = ["# Cabinet grasp experiment", "", f"{len(keys)} setups with all {len(METHODS)} methods (paired)", "",
         "| method | success | declined | pushed > 2 cm | falls | jaw across the jaw at the grip, median cm | > 2 cm across |",
         "|---|---|---|---|---|---|---|"]  # fmt: skip
    for mm in METHODS:
        rs = [by[k][mm] for k in keys]
        ac = [abs(r["at_grip_cm"][1]) for r in rs if "at_grip_cm" in r]
        L.append(f"| {mm} | {sum(r['success'] for r in rs)} | {sum(bool(r.get('declined')) for r in rs)} | "
                 f"{sum(r.get('pushed_cm', 0) > 2 for r in rs)} | {sum(bool(r.get('fell')) for r in rs)} | "
                 f"{np.median(ac) if ac else float('nan'):.2f} | {sum(a > 2 for a in ac)} |")  # fmt: skip
    L += ["", "paired against direct: " + "; ".join(
        f"{mm} wins {sum(by[k][mm]['success'] and not by[k]['direct']['success'] for k in keys)} loses "
        f"{sum(by[k]['direct']['success'] and not by[k][mm]['success'] for k in keys)}" for mm in METHODS[1:])]  # fmt: skip
    L += ["", "| shelf m | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for sh in range(len(GP.SHELVES)):
        ks = [k for k in keys if by[k]["direct"]["shelf"] == sh]
        L.append(
            f"| {GP.SHELVES[sh]:.2f} | "
            + " | ".join(f"{sum(by[k][mm]['success'] for k in ks)}/{len(ks)}" for mm in METHODS)
            + " |"
        )
    L += ["", "| depth m | " + " | ".join(METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for d in GP.DEPTHS:
        ks = [k for k in keys if abs(by[k]["direct"]["depth"] - d) < 1e-6]
        L.append(
            f"| {d:.2f} | " + " | ".join(f"{sum(by[k][mm]['success'] for k in ks)}/{len(ks)}" for mm in METHODS) + " |"
        )
    # sight: by whether the box was in the head camera's view for most of the direct run's entry
    L += [
        "",
        "| box in sight during the entry (direct run) | n | " + " | ".join(METHODS) + " |",
        "|---|---|" + "---|" * len(METHODS),
    ]
    for nm, f in (("mostly seen (>= 50 %)", lambda s: s >= 0.5), ("mostly unseen (< 50 %)", lambda s: s < 0.5)):
        ks = [k for k in keys if "seen_entry" in by[k]["direct"] and f(by[k]["direct"]["seen_entry"])]
        L.append(
            f"| {nm} | {len(ks)} | " + " | ".join(f"{sum(by[k][mm]['success'] for k in ks)}" for mm in METHODS) + " |"
        )
    # the clearance check on the direct runs: predicted contact vs actual contact during the entry
    tp = fp = fn = tn = 0
    for k in keys:
        r = by[k]["direct"]
        if "entry" not in r:
            continue
        pred = not (r["entry"].get("clear_ok", True) and r["pregrasp"].get("clear_ok", True))
        act = any(ph in r.get("contacts", {}) for ph in ("pregrasp", "entry"))
        tp, fp, fn, tn = (
            tp + (pred and act),
            fp + (pred and not act),
            fn + (act and not pred),
            tn + (not pred and not act),
        )
    L += ["", f"clearance check on the direct plans vs robot-cabinet contact in the pre-grasp / entry: predicted and happened {tp}, "
          f"predicted not happened {fp}, happened not predicted {fn}, neither {tn}"]  # fmt: skip
    (OUT / "results.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["h", "d", "_w"])
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--methods", default=",".join(METHODS))
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "q").mkdir(parents=True, exist_ok=True)
    if a.phase == "h":
        launch(a.procs, a.gpus, tuple(a.methods.split(",")))
    elif a.phase == "d":
        report()
    else:
        worker(int(a.rest[1]), [tuple(j) for j in json.loads(a.rest[2])])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
