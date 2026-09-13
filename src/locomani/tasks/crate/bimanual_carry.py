"""Two-handed crate pick: the scripted expert for SCENE_PRESET=crate_bimanual.

    INSTANCE=1 LOCOMANI_SIM_DIR=outputs/scene/crate_bimanual \\
        python src/locomani/tasks/crate/bimanual_carry.py --rounds 3 --manifest <jsonl>

Runs against a stack started with the same INSTANCE / LOCOMANI_SIM_DIR (and
SIM_VIDEO_DIR for the third-person video). Implements sections A-D of
docs/bimanual_grasp_design.md as a minimal closed loop -- no recovery
branch, no carry to surface B yet:

  A  heading aligned DURING the walk-in (face the crate the whole way), the
     residual after the brake is measured and logged, never turned out in
     place (7-8 deg dead zone); the arms absorb it (offline: 8 deg costs
     0.0 cm). Stand-off 0.40 m, feet 5 cm under the table edge like the
     single-hand expert (EDGE_MIN 0.17).
  B  both arms rise to a pre-grasp on the way in (UpperRamp, 2 deg/tick);
     at the stand the reach and the 45 deg palm pitch run as ONE servo, no
     phases; both arms and forward waist pitch are solved together.
  C  one capture region per rim bar, both must be inside before the two
     grippers close in the same frame.
  D  lift 0.10 m; crate roll/pitch over 8 deg or either hand losing contact
     is a failure; hold 2 s.

Everything talks to the stack through the existing pieces: Primitives for
state and locomotion, PlannerArm for the 17-joint upper-body channel (both
arms ride on it; the class is per-side only in what it guards, and the
guard is off here), ArmIK for both arms. Nothing in stream_reach /
grasp_gym is modified.

Split across three files (refactor step 4b family 5, pure extraction;
decision 9's file-size cap): this file keeps `BimanualArms`, the module-level
constants, `rot_z`/`rot_y`/`crate_frame`/`hand_txt`/`crate_tilt_deg`, and
`run_round` itself (now a thin orchestrator); `carry_phases.py` has
`run_round`'s first three phases (`_phase_reset`, `_phase_approach`,
`_phase_grasp`) behind a `CarryRoundContext`; `carry_phases_lift.py` has the
last two (`_phase_close`, `_phase_lift`) -- all five would not fit
carry_phases.py's own 600-line cap.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)  # reach src/ for `python src/locomani/tasks/crate/bimanual_carry.py`

from locomani import paths  # noqa: E402
from locomani.backends.sonic.agent_server import HAND_OPEN, LocomotionMode  # noqa: E402
from locomani.backends.sonic.primitives import Primitives  # noqa: E402
from locomani.backends.sonic.primitives_const import GRASP_CENTRE_OFFSET_CONST  # noqa: E402
from locomani.backends.sonic.streamed_arm import PlannerArm  # noqa: E402
from locomani.primitives.g1_joints import ARM_MJ  # noqa: E402
from locomani.sim import ports
from locomani.sim.privileged_pub import EpisodeControlClient  # noqa: E402
from locomani.tasks.crate import round_runner  # noqa: E402
from locomani.tasks.crate.bimanual_ik import BimanualIK  # noqa: E402
from locomani.tasks.crate.carry_phases import (  # noqa: E402
    CarryRoundContext,
    _phase_approach,
    _phase_grasp,
    _phase_reset,
)
from locomani.tasks.crate.carry_phases_lift import _phase_close, _phase_lift  # noqa: E402
from locomani.tasks.crate.config import CrateTaskConfig  # noqa: E402

OBJ = "blue_crate"
OFFSET = np.asarray(GRASP_CENTRE_OFFSET_CONST, dtype=float)  # palm -> grip centre, palm frame
# Rim-bar centres in the crate body frame (model.xml sites grip_px/grip_nx).
# Grip point: 1 cm BELOW the rim bar's centre (bar 0.055..0.085 in the crate
# frame), so the bar lands nearer the pad's root than its tip. 2 cm put the
# wrist_yaw link (gripper base, 62 mm thick) on the rim top at the 60 deg
# palm pitch (pilot 24: touch ['L.wrist_yaw'], hands parked 2 cm high).
BAR_LOCAL = {"+": np.array([0.1933, 0.0, 0.060]), "-": np.array([-0.1933, 0.0, 0.060])}  # 2/3 crate

# --- A: stand geometry (reach_pitch.py sweep) --------------------------------
STAND_OFF = 0.36  # pelvis to the bar line, m; the feet go 9 cm under the table edge
# (grasp_gym's EDGE_MIN = 0.17 has stood there all along)
STAND_TOL = 0.06  # walk_to stop distance
# Lateral stand: the walk-in scattered the pelvis -7..+16 cm along the bar
# line (pilots 34/35) and every "right hand 20-50 mm high" failure stood LEFT
# of the centre line (+4..+16 cm): the far bar went to the weaker arm. Bias
# the stand toward the right bar and gate the residual; one side-step first.
STAND_LAT_BIAS = -0.07  # m along y_hat (robot's left); negative = toward the right bar.
# -3 cm of bar bias plus the walk-in's systematic +4 cm (pilot 36:
# four stands at +6.4..+7.6 cm of a -3 cm line, 1 cm scatter)
LAT_TOL = 0.04  # lateral gate at the stand
# A side-step at the stand does not work (pilot 36): the slowest gait turns
# but does not translate, and the one that moved went the wrong way. Close the
# lateral error while there is still distance to walk: first a waypoint on
# the (biased) centre line, then straight in along the crate normal.
WAYPOINT_BACK = 0.35  # m behind the stand along the normal
WAYPOINT_TOL = 0.08
PALM_PITCH_DEG = 60  # 45 put wrist_yaw_link on the rim (reach_vert sweep 2026-09-07); 60 clears it at 0.30-0.36
# Forward lean participates in the shared two-hand solve on SONIC v1.1.
# This task cap is additionally bounded by the physical robot model. Old
# locked-waist trials remain historical evidence, not a runtime restriction.
WAIST_PITCH_MAX = 0.4  # radians (22.9 degrees); yaw and roll remain neutral

# --- C: capture region per bar, in the stand frame (x toward the crate) ------
# Capture region per hand, in the stand frame: n points from the robot ACROSS
# the bar line, i.e. ALONG each bar (the bars run fore-aft at y = +-0.29);
# y_hat runs from bar to bar, i.e. ACROSS each bar's 24 mm thickness. Pilots
# 1-12 had the two tolerances on the wrong axes (10 mm along a 0.4 m bar,
# 40 mm across a 24 mm one).
ALONG_TOL = (
    0.025  # along the bar (component on n): the 2/3 crate's 73 mm slot; the 29 mm pad tip keeps ~8 mm to the posts
)
ACROSS_TOL = 0.010  # across the bar (component on y_hat). 25 mm let the jaws close 2 cm off the
# bar: with both hands wide of their bars the two pushes on one rigid crate
# cancel and neither jaw clamps (pilot 26 round 0, no contact at all). The
# policy's 2-4 cm outward bias is taken out by ACROSS_INT below instead.
ACROSS_INT_RATE = 0.008  # per tick, m per m of error: 0.16 mm/tick at 2 cm, ~4 s to wind up 3 cm.
# 0.05 hit the cap in 1 s and flipped sign (pilot 27: the left hand thrown 20 cm up)
ACROSS_INT_CAP = 0.03  # m
HEIGHT_TOL = 0.025  # pad centre ABOVE the grip point. 15 mm made pilot 34 wait 17 s
# for the right hand to creep from 23 to 15 mm with its wrist
# resting on the rim; the 66 mm pad clamps the 16 mm bar 2 cm high.
HEIGHT_TOL_BELOW = 0.010  # and below it
# Stall close: with the wrist links on the rim the hands cannot descend and the
# residual only moves with the policy's slow wrist-pitch catch-up. If along and
# across are in the gate, the height is within the pad's reach, and nothing
# has moved for STALL_S, close instead of waiting for the height gate.
STALL_S = 3.0
STALL_MM = 0.002  # max change of any error component over STALL_S
STALL_HEIGHT_TOL = 0.035
# servo pacing
ADVANCE_PER_TICK = 0.003  # m of reference travel per 20 ms tick (0.15 m/s)
LEAD_CAP = 0.06  # reference may run this far ahead of the measured grip point
OVERDRIVE = 0.7  # fraction of the measured residual added to the command
WRIST_PITCH_COMP = 0.5  # fraction of the wrist-pitch tracking gap (last cmd - measured) added back
WRIST_PITCH_COMP_CAP = np.radians(15.0)
OVERDRIVE_CAP = 0.04  # m, per axis. The policy holds both hands 3-4 cm high of
# the reference (track_test 2026-09-07) and that is a
# SATURATION: 1.0 / 6 cm (pilot 13) only drove the wrist
# pitch and elbow 20-30 deg past what it tracks and the
# arms oscillated. Take the height up in HEIGHT_TOL instead.
# approach: the same shape as grasp_gym's do_approach -- slow down by distance
# remaining, stop on the edge floor, then brake in IDLE.
APPR_SPEED = (0.30, 0.25, 0.20)  # far / mid (< 0.35 m) / near (< 0.15 m); below 0.20 the gait does not move (pilot 6)
EDGE_FLOOR = 0.13  # pelvis-to-edge floor; grasp_gym stands at 0.17, the crate needs 4 cm more
UP_ABOVE, UP_SHORT = 0.08, 0.06  # staging point: 8 cm above the bar, 6 cm short, fingers LEVEL -- the
# tips clear the 0.79 m rim; pilot 10 swept the near wall on the way in.
# The palm pitches 0 -> PALM_PITCH_DEG as it descends (offline: elbow 22 ->
# 43, path error 0.2 cm, both hands).
SERVO_TIMEOUT = 25.0
CLOSE_RAMP = 1.6
# --- D --------------------------------------------------------------------
LIFT_HEIGHT = 0.10
LIFT_MIN = 0.06
TILT_MAX_DEG = 15.0
# Levelling during lift/hold: the crate rolls about the bar line when the two
# hands do not rise together (pilot 32 round 0: 8 deg, right hand 7 mm higher
# at the close). Each step the higher hand's target is lowered and the lower
# one raised by half the bar-end height difference, low-passed and capped.
LEVEL_GAIN = 1.0
LEVEL_CAP = 0.02
LEVEL_LP = 0.15
HOLD_S = 2.0
DUMP_DIR = ""  # set from --manifest's directory in main(); servo-end state dumps
# Closed target for the crate scene: bimanual_scene.py widens the finger travel
# to -0.023 so the pad faces meet; the shared HAND_CLOSED (-0.020) would stop
# 5.8 mm short of that and leave an off-centre bar touched by one pad only.
CRATE_HAND_CLOSED = np.array([-0.023, -0.023, 0, 0, 0, 0, 0], dtype=np.float32)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def crate_frame(p: Primitives, s):
    """(centre, bars{'left','right'}, x_hat toward the crate, y_hat left, psi).

    The stand frame is built from the crate, not from the robot: x points from
    the robot's side of the crate across the bar line, y along the bars from
    the right bar to the left one. 'left'/'right' are the robot's hands.
    """
    o = s["objects"][OBJ]
    c = np.asarray(o["pos"], dtype=float)
    R = Primitives._rot(o["quat"])
    b1, b2 = c + R @ BAR_LOCAL["+"], c + R @ BAR_LOCAL["-"]
    pel = np.asarray(s["pelvis"]["pos"][:2])
    # bar axis in the plane, and the normal pointing away from the robot
    axis = (b1 - b2)[:2]
    axis /= max(np.linalg.norm(axis), 1e-6)
    n = np.array([-axis[1], axis[0]])
    if np.dot(n, c[:2] - pel) < 0:
        n = -n
    psi = float(np.arctan2(n[1], n[0]))
    y_hat = np.array([-n[1], n[0]])  # robot's left when facing +n
    left, right = (b1, b2) if np.dot((b1 - b2)[:2], y_hat) > 0 else (b2, b1)
    return c, {"left": left, "right": right}, n, y_hat, psi


def hand_txt(p, s) -> str:
    """Finger joint positions (sim) and the open/close command (ours), mm."""
    hq = {sd: np.asarray(s["palms"][sd].get("hand_q", [np.nan, np.nan])[:2]) * 1000 for sd in ("left", "right")}
    cl = float(np.asarray(p.cmd.left_hand)[0]) * 1000
    cr = float(np.asarray(p.cmd.right_hand)[0]) * 1000
    return (
        f"fingers L [{hq['left'][0]:+.1f} {hq['left'][1]:+.1f}] R [{hq['right'][0]:+.1f} {hq['right'][1]:+.1f}]"
        f" cmd L {cl:+.1f} R {cr:+.1f}"
    )


def crate_tilt_deg(s) -> float:
    R = Primitives._rot(s["objects"][OBJ]["quat"])
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))


class BimanualArms:
    """Both arms + the shared waist on the planner upper-body channel."""

    def __init__(self, p: Primitives):
        self.p = p
        self.arm = PlannerArm(p, "right", send_vel=True)
        self.arm.guard = None  # no off-hand: both hands work
        ik = self.arm.ik
        self.solver = BimanualIK(ik, waist_pitch_max=WAIST_PITCH_MAX, tool_offset=OFFSET)
        # Keep yaw/roll neutral while allowing forward pitch within real limits.
        ik.lo[12:15] = self.solver.lo[12:15]
        ik.hi[12:15] = self.solver.hi[12:15]
        # Arm bounds that make the "arm overhead" branch impossible. Pilots
        # 7-9: one arm or the other flipped to a hyperextended elbow with the
        # wrist rolled 50-70 deg and shot 60-70 cm above the bar; the policy
        # then tracks nothing. The bar solution needs elbow 43, wrist roll 7,
        # shoulder yaw 8, wrist yaw 21 (offline), so these leave it room and
        # forbid the rest: elbow forearm-down only, wrist roll and shoulder
        # yaw within 45 deg.
        for sd in ("left", "right"):
            a = ARM_MJ[sd]
            ik.lo[a[3]] = np.radians(5.0)  # elbow: never above level
            ik.lo[a[2]], ik.hi[a[2]] = -np.radians(45), np.radians(45)  # shoulder yaw
            ik.lo[a[4]], ik.hi[a[4]] = -np.radians(45), np.radians(45)  # wrist roll
        self.hands = {"left": HAND_OPEN.copy(), "right": HAND_OPEN.copy()}

    def start(self, q17=None):
        self.arm.start(upper_reference=q17)
        self.arm.hands = {k: v.astype(np.float32) for k, v in self.hands.items()}

    def stop(self):
        self.arm.stop()

    def grip_point(self, s, side):
        palm = np.asarray(s["palms"][side]["pos"], dtype=float)
        R = Primitives._rot(s["palms"][side]["quat"])
        return palm + R @ OFFSET, R

    def reach_errors(self, s, targets_w, rots_w):
        """Both hand residuals at the current pelvis pose, without sending motion."""
        q = self.solve(s, targets_w, rots_w, iters=200, from_measured=True, rot_weight=0.25)
        points, _ = self.solver.tool_poses(q)
        targets = [self.p.world_to_root_full(targets_w[side], s) for side in self.solver.sides]
        return np.linalg.norm(points - targets, axis=1)

    def solve(
        self,
        s,
        targets_w,
        rots_w,
        iters=12,
        from_measured=False,
        rot_weight=0.5,
        posture=None,
        posture_weight=0.0,
        ghost_targets=None,
    ):
        """Solve both hands and their shared waist in one system.

        from_measured seeds the waist and both arms from encoder feedback;
        otherwise the previous command is the initial posture. Joint bounds
        constrain the solution; trajectory pacing belongs to the caller.
        """
        q = self.arm.q.copy()
        if from_measured:
            q[12:29] = np.asarray(s["joints"], dtype=float)[12:29]
        ghost = getattr(self, "ghost", None)
        if ghost is not None:
            # ghost_trace.GhostTargetTrace: the aimed wrist poses, per state seq.
            # A servo passes `ghost_targets` = its pure reference point (where
            # the hand should be), because `targets_w` is then the command --
            # reference + lead cap + overdrive -- which leads the hand and
            # jitters (user, 2026-09-10: "the ghost moves and leaves the rail").
            ghost.row(s.get("seq"), ghost_targets if ghost_targets is not None else targets_w, rots_w)
        Rp = Primitives._rot(s["pelvis"]["quat"])
        targets = [self.p.world_to_root_full(targets_w[side], s) for side in self.solver.sides]
        rotations = [Rp.T @ rots_w[side] for side in self.solver.sides]
        return self.solver.solve(
            q,
            targets,
            rotations,
            iterations=iters,
            posture=posture,
            rot_weight=rot_weight,
            posture_weight=posture_weight,
            # The wall expert can additionally constrain shoulder pitch.
            joint_limits=(self.arm.ik.lo, self.arm.ik.hi),
        ).joints

    def push(self, q):
        self.arm.q = q
        self.arm._push(q)

    def set_hands(self, alpha: float):
        for side, closed in (("left", CRATE_HAND_CLOSED), ("right", CRATE_HAND_CLOSED)):
            self.hands[side] = (HAND_OPEN + alpha * (closed - HAND_OPEN)).astype(np.float32)
        self.arm.hands = dict(self.hands)


def run_round(
    p: Primitives,
    reset: EpisodeControlClient,
    meta: dict,
    rnd: int,
    rng: np.random.Generator,
    log,
    video_label: str,
    record: bool = False,
) -> dict:
    """One round's five phases (carry_phases.py / carry_phases_lift.py),
    threaded through a CarryRoundContext.

    Body moved there (refactor step 4b, pure extraction) -- this function
    stays HERE, thin, so that `locomani.tasks.crate.__init__`'s
    `bimanual_carry.run_round(*args, **kwargs)` call keeps resolving to this
    same name.
    """
    ctx = CarryRoundContext(
        p=p,
        reset=reset,
        meta=meta,
        rnd=rnd,
        rng=rng,
        log=log,
        video_label=video_label,
        m={"round": rnd, "ok": False, "why": "", "produced": False},
    )
    for phase in (_phase_reset, _phase_approach, _phase_grasp, _phase_close, _phase_lift):
        result = phase(ctx)
        if result is not None:
            return result
        if phase is _phase_reset and record:
            # See bimanual_wall.run_round: after the settle, before the task.
            round_runner.start_recording(p, ctx.m)
    raise AssertionError("carry_phases: no phase returned a result")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", default="")
    ap.add_argument("--label", default="bimanual")
    ap.add_argument("--config", type=Path, help="CrateTaskConfig JSON; defaults if omitted")
    ap.add_argument("--set", dest="set_", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument(
        "--record",
        action="store_true",
        help="drive the data exporter too, keeping only the rounds that succeed (grasp_gym.py's flag, same "
        "protocol). Needs INSTANCE=0: start_stack.sh runs the exporter only there",
    )
    args = ap.parse_args()
    # Same CrateTaskConfig and the same --config/--set flags as bimanual_wall.py.
    # This file's ROUND logic still reads none of the task's geometry fields
    # (unlike the wall expert's wall_feature/wall_walk); deploy_self_heal_rounds
    # below is the first field it does read.
    cfg = CrateTaskConfig.load(args.config) if args.config else CrateTaskConfig()
    if args.set_:
        cfg = cfg.with_overrides(args.set_)
    # Script-vs-package double copy (see bimanual_wall.main): carry_phases*.py import this
    # module by its package name; make that name resolve to the running module so the
    # globals set below (DUMP_DIR) are the ones the phases read.
    sys.modules.setdefault("locomani.tasks.crate.bimanual_carry", sys.modules[__name__])
    inst = ports.instance()
    # One formula, in locomani.sim.ports: this used to carry its own copy of the
    # bases, a third one after start_stack.sh's and every module-level fallback's.
    for k in ("TF_SONIC_PORT", "TF_OUT_PORT", "TF_KEYBOARD_PORT", "TF_PRIVILEGED_PORT", "TF_EPISODE_PORT"):
        os.environ.setdefault(k, str(ports.port(k)))
    global DUMP_DIR
    DUMP_DIR = os.path.dirname(os.path.abspath(args.manifest)) if args.manifest else ""
    meta = json.load(open(paths.SCENE_META))
    if meta.get("target_object") != OBJ:
        raise SystemExit(
            f"{paths.SCENE_META} is not the crate scene (target "
            f"{meta.get('target_object')}); build it with bimanual_scene.py "
            f"and point LOCOMANI_SIM_DIR at it"
        )
    print(f"[bimanual] instance {inst}, scene {paths.SCENE_META}, task config {cfg.to_dict()}")
    if args.record:
        # The same refusal grasp_gym makes: the exporter only exists on instance 0,
        # so a --record run anywhere else finishes every round and writes nothing.
        ports.require_exporter("--record")
    p = Primitives(meta=meta)
    reset = EpisodeControlClient()
    round_runner.prime(p, reset)
    # Shared with bimanual_wall.py (round_runner.py). This loop used to be a
    # copy that had drifted: no deploy self-heal, so one LowState trip turned
    # every remaining round into "[stand] did not stabilise".
    round_runner.run_rounds(
        p,
        reset,
        meta,
        run_round=run_round,
        rng=np.random.default_rng(args.seed),
        rounds=args.rounds,
        label=args.label,
        tag="bimanual",
        manifest=args.manifest,
        record=args.record,
        self_heal_rounds=cfg.deploy_self_heal_rounds,
        instance=inst,
    )


if __name__ == "__main__":
    main()
