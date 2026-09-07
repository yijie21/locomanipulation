"""Two-handed crate pick: the scripted expert for SCENE_PRESET=crate_bimanual.

    INSTANCE=1 LOCOMANI_SIM_DIR=outputs/scene/crate_bimanual \\
        python src/teleop_free/bimanual_carry.py --rounds 3 --manifest <jsonl>

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
     phases, waist locked -- a commanded lean is not delivered (diag_lean).
  C  one capture region per rim bar, both must be inside before the two
     grippers close in the same frame.
  D  lift 0.10 m; crate roll/pitch over 8 deg or either hand losing contact
     is a failure; hold 2 s.

Everything talks to the stack through the existing pieces: Primitives for
state and locomotion, PlannerArm for the 17-joint upper-body channel (both
arms ride on it; the class is per-side only in what it guards, and the
guard is off here), ArmIK for both arms. Nothing in stream_reach /
grasp_gym is modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_server import HAND_OPEN, LocomotionMode, StreamMode  # noqa: E402
from g1_joints import ARM_MJ  # noqa: E402
from primitives import (DT, HAND_CLOSED_LEFT, HAND_CLOSED_RIGHT, Primitives,  # noqa: E402
                        upper_body_from_mj, wrap)
from primitives_const import GRASP_CENTRE_OFFSET_CONST  # noqa: E402
from privileged_pub import EpisodeControlClient  # noqa: E402
from stream_reach import PlannerArm  # noqa: E402
import paths  # noqa: E402
import upper_ref  # noqa: E402

OBJ = "blue_crate"
OFFSET = np.asarray(GRASP_CENTRE_OFFSET_CONST, dtype=float)   # palm -> grip centre, palm frame
# Rim-bar centres in the crate body frame (model.xml sites grip_px/grip_nx).
# Grip point: 1 cm BELOW the rim bar's centre (bar 0.055..0.085 in the crate
# frame), so the bar lands nearer the pad's root than its tip. 2 cm put the
# wrist_yaw link (gripper base, 62 mm thick) on the rim top at the 60 deg
# palm pitch (pilot 24: touch ['L.wrist_yaw'], hands parked 2 cm high).
BAR_LOCAL = {"+": np.array([0.1933, 0.0, 0.060]), "-": np.array([-0.1933, 0.0, 0.060])}  # 2/3 crate

# --- A: stand geometry (reach_pitch.py sweep) --------------------------------
STAND_OFF = 0.36          # pelvis to the bar line, m; the feet go 9 cm under the table edge
                          # (grasp_gym's EDGE_MIN = 0.17 has stood there all along)
STAND_TOL = 0.06          # walk_to stop distance
# Lateral stand: the walk-in scattered the pelvis -7..+16 cm along the bar
# line (pilots 34/35) and every "right hand 20-50 mm high" failure stood LEFT
# of the centre line (+4..+16 cm): the far bar went to the weaker arm. Bias
# the stand toward the right bar and gate the residual; one side-step first.
STAND_LAT_BIAS = -0.07    # m along y_hat (robot's left); negative = toward the right bar.
                          # -3 cm of bar bias plus the walk-in's systematic +4 cm (pilot 36:
                          # four stands at +6.4..+7.6 cm of a -3 cm line, 1 cm scatter)
LAT_TOL = 0.04            # lateral gate at the stand
# A side-step at the stand does not work (pilot 36): the slowest gait turns
# but does not translate, and the one that moved went the wrong way. Close the
# lateral error while there is still distance to walk: first a waypoint on
# the (biased) centre line, then straight in along the crate normal.
WAYPOINT_BACK = 0.35      # m behind the stand along the normal
WAYPOINT_TOL = 0.08
PALM_PITCH_DEG = 60  # 45 put wrist_yaw_link on the rim (reach_vert sweep 2026-09-07); 60 clears it at 0.30-0.36
# NO waist. diag_lean.py: 2-10% of a commanded lean is delivered on either
# channel, and pilot 4 confirmed it (34 deg commanded, 2.8 measured). The
# arms alone reach both bars at 0.40 m with the palms at 45 deg (arms-only
# sweep 2026-09-07: 1.0 cm standing, 0.0 with a 9 cm crouch), at 0.47 they
# do not (6.4 cm) -- so the stand-off is what changed, not the waist.
WAIST_PITCH_MAX = 0.0

# --- C: capture region per bar, in the stand frame (x toward the crate) ------
# Capture region per hand, in the stand frame: n points from the robot ACROSS
# the bar line, i.e. ALONG each bar (the bars run fore-aft at y = +-0.29);
# y_hat runs from bar to bar, i.e. ACROSS each bar's 24 mm thickness. Pilots
# 1-12 had the two tolerances on the wrong axes (10 mm along a 0.4 m bar,
# 40 mm across a 24 mm one).
ALONG_TOL = 0.025         # along the bar (component on n): the 2/3 crate's 73 mm slot; the 29 mm pad tip keeps ~8 mm to the posts
ACROSS_TOL = 0.010        # across the bar (component on y_hat). 25 mm let the jaws close 2 cm off the
                          # bar: with both hands wide of their bars the two pushes on one rigid crate
                          # cancel and neither jaw clamps (pilot 26 round 0, no contact at all). The
                          # policy's 2-4 cm outward bias is taken out by ACROSS_INT below instead.
ACROSS_INT_RATE = 0.008   # per tick, m per m of error: 0.16 mm/tick at 2 cm, ~4 s to wind up 3 cm.
                          # 0.05 hit the cap in 1 s and flipped sign (pilot 27: the left hand thrown 20 cm up)
ACROSS_INT_CAP = 0.03     # m
HEIGHT_TOL = 0.025        # pad centre ABOVE the grip point. 15 mm made pilot 34 wait 17 s
                          # for the right hand to creep from 23 to 15 mm with its wrist
                          # resting on the rim; the 66 mm pad clamps the 16 mm bar 2 cm high.
HEIGHT_TOL_BELOW = 0.010  # and below it
# Stall close: with the wrist links on the rim the hands cannot descend and the
# residual only moves with the policy's slow wrist-pitch catch-up. If along and
# across are in the gate, the height is within the pad's reach, and nothing
# has moved for STALL_S, close instead of waiting for the height gate.
STALL_S = 3.0
STALL_MM = 0.002          # max change of any error component over STALL_S
STALL_HEIGHT_TOL = 0.035
# servo pacing
ADVANCE_PER_TICK = 0.003  # m of reference travel per 20 ms tick (0.15 m/s)
LEAD_CAP = 0.06           # reference may run this far ahead of the measured grip point
OVERDRIVE = 0.7           # fraction of the measured residual added to the command
WRIST_PITCH_COMP = 0.5    # fraction of the wrist-pitch tracking gap (last cmd - measured) added back
WRIST_PITCH_COMP_CAP = np.radians(15.0)
OVERDRIVE_CAP = 0.04      # m, per axis. The policy holds both hands 3-4 cm high of
                          # the reference (track_test 2026-09-07) and that is a
                          # SATURATION: 1.0 / 6 cm (pilot 13) only drove the wrist
                          # pitch and elbow 20-30 deg past what it tracks and the
                          # arms oscillated. Take the height up in HEIGHT_TOL instead.
# approach: the same shape as grasp_gym's do_approach -- slow down by distance
# remaining, stop on the edge floor, then brake in IDLE.
APPR_SPEED = (0.30, 0.25, 0.20)   # far / mid (< 0.35 m) / near (< 0.15 m); below 0.20 the gait does not move (pilot 6)
EDGE_FLOOR = 0.13                 # pelvis-to-edge floor; grasp_gym stands at 0.17, the crate needs 4 cm more
UP_ABOVE, UP_SHORT = 0.08, 0.06   # staging point: 8 cm above the bar, 6 cm short, fingers LEVEL -- the
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
DUMP_DIR = ""             # set from --manifest's directory in main(); servo-end state dumps
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
    y_hat = np.array([-n[1], n[0]])           # robot's left when facing +n
    left, right = (b1, b2) if np.dot((b1 - b2)[:2], y_hat) > 0 else (b2, b1)
    return c, {"left": left, "right": right}, n, y_hat, psi


def hand_txt(p, s) -> str:
    """Finger joint positions (sim) and the open/close command (ours), mm."""
    hq = {sd: np.asarray(s["palms"][sd].get("hand_q", [np.nan, np.nan])[:2]) * 1000 for sd in ("left", "right")}
    cl = float(np.asarray(p.cmd.left_hand)[0]) * 1000
    cr = float(np.asarray(p.cmd.right_hand)[0]) * 1000
    return (f"fingers L [{hq['left'][0]:+.1f} {hq['left'][1]:+.1f}] R [{hq['right'][0]:+.1f} {hq['right'][1]:+.1f}]"
            f" cmd L {cl:+.1f} R {cr:+.1f}")


def crate_tilt_deg(s) -> float:
    R = Primitives._rot(s["objects"][OBJ]["quat"])
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))


class BimanualArms:
    """Both arms + the shared waist on the planner upper-body channel."""

    def __init__(self, p: Primitives):
        self.p = p
        self.arm = PlannerArm(p, "right", send_vel=True)
        self.arm.guard = None                     # no off-hand: both hands work
        ik = self.arm.ik
        # All three waist joints locked: what the IK put there was not
        # delivered (see WAIST_PITCH_MAX), and a reference the body does not
        # follow is a lie in the recorded data.
        ik.lo[12:15] = 0.0
        ik.hi[12:15] = 0.0
        # Arm bounds that make the "arm overhead" branch impossible. Pilots
        # 7-9: one arm or the other flipped to a hyperextended elbow with the
        # wrist rolled 50-70 deg and shot 60-70 cm above the bar; the policy
        # then tracks nothing. The bar solution needs elbow 43, wrist roll 7,
        # shoulder yaw 8, wrist yaw 21 (offline), so these leave it room and
        # forbid the rest: elbow forearm-down only, wrist roll and shoulder
        # yaw within 45 deg.
        for sd in ("left", "right"):
            a = ARM_MJ[sd]
            ik.lo[a[3]] = np.radians(5.0)                         # elbow: never above level
            ik.lo[a[2]], ik.hi[a[2]] = -np.radians(45), np.radians(45)   # shoulder yaw
            ik.lo[a[4]], ik.hi[a[4]] = -np.radians(45), np.radians(45)   # wrist roll
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

    def solve(self, s, targets_w, rots_w, iters=12, from_measured=False,
              rot_weight=0.5, posture=None, posture_weight=0.0):
        """One coupled step: each arm in turn. use_waist stays on so the
        solver's structure is unchanged, but the waist bounds are zero, so it
        cannot move.

        from_measured: warm-start from the MEASURED arm joints instead of the
        last command. Warm-starting from the command let the reference drift
        away from what the policy tracks -- pilot 8: right wrist roll 12 ->
        66 deg off, shoulder yaw 8 -> 47, the hand climbing 10 cm while the
        overdrive kept pushing. From the measurement the command can only
        ever be one small step from the body.
        """
        q = self.arm.q.copy()
        if from_measured:
            q[15:29] = np.asarray(s["joints"], dtype=float)[15:29]
        Rp = Primitives._rot(s["pelvis"]["quat"])
        for side in ("left", "right"):
            tgt = self.p.world_to_root_full(targets_w[side], s)
            rot = Rp.T @ rots_w[side]
            q, pe, re = self.arm.ik.solve(side, tgt, rot, q_mj29=q, iters=iters,
                                          use_waist=True, waist_cost=3.0,
                                          point_offset=OFFSET, rot_weight=rot_weight,
                                          posture=posture, posture_weight=posture_weight)
        return q

    def push(self, q):
        self.arm.q = q
        self.arm._push(q)

    def set_hands(self, alpha: float):
        for side, closed in (("left", CRATE_HAND_CLOSED), ("right", CRATE_HAND_CLOSED)):
            self.hands[side] = (HAND_OPEN + alpha * (closed - HAND_OPEN)).astype(np.float32)
        self.arm.hands = dict(self.hands)


def run_round(p: Primitives, reset: EpisodeControlClient, meta: dict, rnd: int,
              rng: np.random.Generator, log, video_label: str) -> dict:
    m = {"round": rnd, "ok": False, "why": ""}
    surf = meta["surfaces"]["A"]
    hx = float(surf["flat_half_extent"][0])
    crate0 = np.asarray(meta["objects"][OBJ]["init_pos"], dtype=float)

    # --- reset: crate yaw jitter, robot spawn 1.0-1.3 m behind the stand ----
    crate_dyaw = 0.0                                # bars parallel to the table edge (was +-15 deg)
    crate_dy = float(rng.uniform(-0.10, 0.10))
    n0 = np.array([1.0, 0.0])                       # crate near wall faces -x
    stand0 = crate0[:2] - n0 * STAND_OFF
    spawn_back = float(rng.uniform(1.0, 1.3))
    spawn_lat = float(rng.uniform(-0.15, 0.15))
    spawn_xy = stand0 - n0 * spawn_back + np.array([0.0, spawn_lat])
    robot_nom = np.asarray(meta["robot_start"]["xy"], dtype=float)
    psi0 = float(np.arctan2(*(crate0[:2] - spawn_xy)[::-1])) + float(rng.uniform(-0.17, 0.17))
    m.update(crate_dyaw_deg=np.degrees(crate_dyaw), crate_dy=crate_dy,
             spawn_back=spawn_back, spawn_lat=spawn_lat)

    # reset() takes an ABSOLUTE yaw (privileged_pub: qpos quat := yaw_quat),
    # so the crate's scene yaw (90 deg, handles at +-y) is the base, not zero.
    # The first pilot passed the jitter alone: the crate came up handles-fore,
    # the stand point landed beside the table and every walk-in timed out.
    w, x, y, z = meta["objects"][OBJ]["init_quat"]
    crate_yaw0 = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    p.cmd.reset_pose()
    p.cmd.set_hands(left=HAND_OPEN, right=HAND_OPEN)     # never carry a closed jaw into the walk-in
    reset.pin_arms()
    reset.set_ghost()
    reset.reset(objects={OBJ: {"dxy": [0.0, crate_dy], "yaw": crate_yaw0 + crate_dyaw}},
                robot={"dxy": [float(v) for v in (spawn_xy - robot_nom)], "yaw": psi0})
    time.sleep(0.3)
    p.cmd.stream_mode = StreamMode.PLANNER_VR_3PT
    p.cmd.upper_body = None
    p.cmd.start_control()
    p._gym_started = True
    upper_ref.shared().wait(3.0)
    p.cmd.heading_offset = float(p.pelvis_yaw(p.obs()))
    fv0 = (float(np.cos(psi0)), float(np.sin(psi0)), 0.0)
    p.cmd.hold(3.0, mode=LocomotionMode.IDLE, facing=fv0)
    s = p.obs()
    if s["fallen"]:
        m["why"] = "[stand] did not stabilise after reset"
        return m
    reset.start_video(video_label)
    log(f"[reset] {hand_txt(p, p.obs())}")
    t_video = time.time()

    # --- A+B: walk in facing the crate, both arms rising to the pre-grasp ----
    centre, bars, n, y_hat, psi = crate_frame(p, s)
    stand_xy = centre[:2] - n * STAND_OFF + y_hat * STAND_LAT_BIAS
    edge_x = float(surf["xy"][0]) - hx
    m["stand_target"] = [float(v) for v in stand_xy]
    q_meas = upper_body_from_mj(s["joints"])
    q_free = upper_ref.shared().upper17()
    q_free = q_meas if q_free is None else q_free
    ramp = upper_ref.UpperRamp(p.cmd, q_free)
    p.cmd.stream_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY
    # pre-grasp: both palms at the staging points, expressed for the FINAL stand
    arms = BimanualArms(p)
    R_stand = rot_z(psi)
    rot_w = {sd: R_stand for sd in ("left", "right")}          # fingers level while walking in
    up_w = {sd: bars[sd] + np.array([*(-n * UP_SHORT), UP_ABOVE]) for sd in ("left", "right")}
    # IK in the pelvis frame the robot WILL have at the stand: same yaw as psi,
    # pelvis over stand_xy at the standing height it has now.
    s_stand = dict(s)
    s_stand["pelvis"] = dict(s["pelvis"], pos=[float(stand_xy[0]), float(stand_xy[1]),
                                                float(s["pelvis"]["pos"][2])],
                             quat=[float(np.cos(psi / 2)), 0.0, 0.0, float(np.sin(psi / 2))])
    # Seed the pre-grasp IK from the zero posture, NOT the measured arms:
    # from the hanging arms the DLS lands in a straight-elbow, wrist-rolled
    # branch (elbow -6 deg, wrist roll 80) that the policy tracks with a 20 deg
    # gap, and the servo inherits it (pilot 6: right hand stuck 2.3 cm out).
    # From zero it finds the comfortable branch (elbow 43, wrist roll 7,
    # 61 deg from the nearest limit), and every later solve warm-starts there.
    arms.arm.q = np.zeros(29)
    q_pre = arms.solve(s_stand, up_w, rot_w, iters=200)
    q_pre17 = upper_body_from_mj(q_pre)
    for sd in ("left", "right"):
        pos, R = arms.arm.ik.fk(q_pre, sd)
        g = np.asarray(s_stand["pelvis"]["pos"]) + Primitives._rot(s_stand["pelvis"]["quat"]) @ (pos + R @ OFFSET)
        log(f"[pre-grasp] {sd}: ik residual {np.linalg.norm(g - up_w[sd]) * 100:.1f} cm, "
            f"arm q {np.round(np.degrees(q_pre[ARM_MJ[sd]]), 0).tolist()}")

    def on_tick(_s, dist):
        u = float(np.clip(1.0 - (dist - 0.10) / 0.90, 0.0, 1.0))
        ramp.push(q_free + (3 * u * u - 2 * u ** 3) * (q_pre17 - q_free))

    # Walk in facing the crate; speed by distance remaining; stop on the edge
    # floor or at the stand. walk_to alone scattered the stand by -18..+14 cm
    # (pilot 3): its stop_dist has no notion of the table, and the carry after
    # the last frame put the feet under the edge.
    t0, why = time.time(), "timeout"
    best_along, best_t = None, time.time()
    way_xy = stand_xy - n * WAYPOINT_BACK
    leg = "waypoint"
    m["waypoint"] = [float(v) for v in way_xy]
    while time.time() - t0 < 40.0:
        s = p.obs()
        if s["fallen"]:
            m["why"] = "[stand] fell during the walk-in"
            return m
        pel = np.asarray(p.pelvis_xy(s))
        to_stand = stand_xy - pel
        along = float(np.dot(to_stand, n))          # + = still short of the stand
        dist = float(np.linalg.norm(to_stand))
        on_tick(s, dist)
        if leg == "waypoint":
            to_way = way_xy - pel
            if float(np.dot(to_way, n)) <= 0.03 or float(np.linalg.norm(to_way)) < WAYPOINT_TOL:
                leg = "in"
                log(f"[walk] waypoint reached: lateral {float(np.dot(pel - stand_xy, y_hat)) * 100:+.1f} cm, "
                    f"{float(np.dot(to_stand, n)) * 100:.0f} cm to the stand")
        edge_clear = edge_x - float(pel[0])
        if along <= 0.04 or dist < STAND_TOL:
            why = "at the stand"
            break
        if edge_clear <= EDGE_FLOOR:
            why = "edge floor"
            break
        # no progress for 3 s: the gait has parked; take the stand it gives
        # (the band check below decides whether it is usable)
        if best_along is None or along < best_along - 0.015:
            best_along, best_t = along, time.time()
        elif time.time() - best_t > 3.0:
            why = "stalled"
            break
        centre_now, _, n_now, _, _ = crate_frame(p, s)
        look = centre_now[:2] - pel
        b = float(np.arctan2(look[1], look[0]))
        speed = APPR_SPEED[0] if along > 0.35 else APPR_SPEED[1] if along > 0.15 else APPR_SPEED[2]
        if leg == "waypoint":
            mv = to_way / max(float(np.linalg.norm(to_way)), 1e-3)
        else:
            mv = to_stand / max(dist, 1e-3)
            b = psi                                  # straight in, square to the bars
        if int((time.time() - t0) / 2.0) != int((time.time() - t0 - DT) / 2.0):
            log(f"[walk] t={time.time() - t0:4.1f} pelvis {np.round(pel, 2).tolist()} dist {dist:.2f} "
                f"along {along:+.2f} edge {edge_clear:.2f} speed {speed} facing_err "
                f"{np.degrees(wrap(b - p.pelvis_yaw(s))):+.0f} deg ramp_gap {np.degrees(np.abs(ramp.q - q_pre17).max()):.0f} deg  {hand_txt(p, s)}")
        p.cmd.frame(mode=LocomotionMode.SLOW_WALK, facing=(float(np.cos(b)), float(np.sin(b)), 0.0),
                    movement=(float(mv[0]), float(mv[1]), 0.0), speed=speed)
        time.sleep(DT)
    m["approach_exit"] = why
    if why == "timeout":
        m["why"] = "[stand] walk-in timed out"
        return m
    def brake():
        # pelvis speed under 0.06 m/s for 0.25 s, arms finishing their rise
        t0, still = time.time(), None
        prev = p.pelvis_xy(p.obs())
        while time.time() - t0 < 6.0:
            ramp.push(q_pre17)
            p.cmd.frame(mode=LocomotionMode.IDLE, facing=(float(np.cos(psi)), float(np.sin(psi)), 0.0))
            time.sleep(DT)
            cur = p.pelvis_xy(p.obs())
            v = float(np.linalg.norm(cur - prev)) / DT
            prev = cur
            still = time.time() if v > 0.06 else (still if still is not None else time.time())
            if still is not None and time.time() - still > 0.25 and time.time() - t0 > 1.0:
                break

    def stand_metrics(tag):
        s = p.obs()
        centre, bars, n, y_hat, psi = crate_frame(p, s)
        pel = np.asarray(p.pelvis_xy(s))
        yaw_err = float(np.degrees(wrap(psi - p.pelvis_yaw(s))))
        stand_err = float(np.dot(pel - centre[:2], n)) + STAND_OFF
        lateral = float(np.dot(pel - centre[:2], y_hat)) - STAND_LAT_BIAS   # + = left of the biased line
        edge_clear = edge_x - float(pel[0])
        log(f"[stand{tag}] yaw residual {yaw_err:+.1f} deg, along-track {stand_err:+.3f} m, "
            f"lateral {lateral:+.3f} m, pelvis {edge_clear:.2f} m outside the edge  {hand_txt(p, s)}")
        return s, yaw_err, stand_err, lateral, edge_clear

    brake()
    s, yaw_err, stand_err, lateral, edge_clear = stand_metrics("")
    m.update(yaw_err_deg=yaw_err, stand_err=stand_err, lateral=lateral, edge_clear=edge_clear)
    ramp.settle()
    s = p.obs()
    centre, bars, n, y_hat, psi = crate_frame(p, s)
    if abs(yaw_err) > 20 or stand_err < -0.08 or stand_err > 0.10:
        m["why"] = "[stand] outside the reach band"
        return m
    if abs(lateral) > LAT_TOL:
        m["why"] = f"[stand] lateral {lateral * 100:+.1f} cm off the centre line"
        return m

    # --- B+C: one servo: lean + reach + pitch, both hands, to the bars -------
    arms.start(q17=ramp.q)
    R_stand = rot_z(psi)
    ref = {sd: bars[sd] + np.array([*(-n * UP_SHORT), UP_ABOVE]) for sd in ("left", "right")}

    def pitched(sd):
        # palm pitch follows the descent: level at the staging height, the
        # full PALM_PITCH_DEG at the bar, so the fingertips never lead the jaw
        # into the rim.
        u = float(np.clip(1.0 - (ref[sd][2] - bars[sd][2]) / UP_ABOVE, 0.0, 1.0))
        return R_stand @ rot_y(np.radians(PALM_PITCH_DEG * u))
    crate_start = centre.copy()
    t0 = time.time()
    ready_since = None
    err_hist = []                          # (t, errs) for the stall test
    close_trigger = None
    q_last = None
    across_int = {"left": 0.0, "right": 0.0}
    phase = "servo"
    while time.time() - t0 < SERVO_TIMEOUT:
        s = p.obs()
        if s["fallen"]:
            m["why"] = "[grasp] fell during the reach"
            arms.stop()
            return m
        centre, bars, n, y_hat, psi = crate_frame(p, s)
        if float(np.linalg.norm(centre[:2] - crate_start[:2])) > 0.05:
            m["why"] = "[grasp] crate pushed > 5 cm before closing"
            arms.stop()
            return m
        errs, ready = {}, True
        for sd in ("left", "right"):
            g, _ = arms.grip_point(s, sd)
            e = g - bars[sd]
            along, across, height = float(e[:2] @ n), float(e[:2] @ y_hat), float(e[2])
            errs[sd] = (along, across, height)      # logged as [along, across, height]
            ready &= (abs(along) <= ALONG_TOL and abs(across) <= ACROSS_TOL
                      and -HEIGHT_TOL_BELOW <= height <= HEIGHT_TOL)
            # reference: advance toward the bar, capped ahead of the measured point
            want = bars[sd]
            step = want - ref[sd]
            d = float(np.linalg.norm(step))
            if d > ADVANCE_PER_TICK:
                step *= ADVANCE_PER_TICK / d
            nxt = ref[sd] + step
            lead = nxt - g
            ld = float(np.linalg.norm(lead))
            if ld > LEAD_CAP:
                nxt = g + lead * (LEAD_CAP / ld)
            ref[sd] = nxt
        # The planner channel tracks the reference with a 1-3 cm residual and
        # the reference alone never closes it (pilot 3: 9-50 mm left over after
        # 25 s). Same remedy as the single-hand servo: overdrive the command by
        # a fraction of the CURRENT residual, from the current measurement only
        # (accumulating corrections diverges on transients), capped.
        cmd = {}
        for sd in ("left", "right"):
            g, _ = arms.grip_point(s, sd)
            corr = np.clip(OVERDRIVE * (ref[sd] - g), -OVERDRIVE_CAP, OVERDRIVE_CAP)
            # slow integrator across the bar: the proportional overdrive leaves the
            # policy's constant outward offset in place, an integrator does not
            across_int[sd] = float(np.clip(across_int[sd] - ACROSS_INT_RATE * errs[sd][1],
                                           -ACROSS_INT_CAP, ACROSS_INT_CAP))
            cmd[sd] = ref[sd] + corr + np.array([*(y_hat * across_int[sd]), 0.0])
        rot_w = {sd: pitched(sd) for sd in ("left", "right")}
        q = arms.solve(s, cmd, rot_w, from_measured=True, rot_weight=0.25)
        # Joint-space compensation on the wrist pitch alone: the policy delivers
        # the right wrist pitch 20 deg short of the command (diagnostic dump
        # 2026-09-07: cmd 49 / meas 29, every other joint within 10), which is
        # 5 cm at the pad and the Cartesian overdrive cannot buy it back.
        # Pure proportional on the tracking error of the IK solution itself.
        # Using the previous (already compensated) command as the reference
        # made this an integrator that sat on its +-15 deg cap; pilots 25-28
        # then showed the right wrist 50-60 deg past the command and the hand
        # 13 cm high.
        for sd in ("left", "right"):
            j = ARM_MJ[sd][5]
            gap = float(q[j] - s["joints"][j])
            q[j] += float(np.clip(WRIST_PITCH_COMP * gap, -WRIST_PITCH_COMP_CAP, WRIST_PITCH_COMP_CAP))
        q_last = q.copy()
        arms.push(q)
        if int((time.time() - t0) / 1.0) != int((time.time() - t0 - DT) / 1.0):
            from g1_joints import MUJOCO_JOINT_NAMES
            dj = np.degrees(np.asarray(s["joints"])[15:29] - q[15:29])
            worst = np.argsort(-np.abs(dj))[:3]
            gap = float(np.abs(dj).max())
            gap_txt = " ".join(f"{MUJOCO_JOINT_NAMES[15 + i].replace('_joint', '')}{dj[i]:+.0f}" for i in worst)
            log(f"[servo] t={time.time() - t0:4.1f} err_mm L {np.round(np.array(errs['left']) * 1000, 0).tolist()} "
                f"R {np.round(np.array(errs['right']) * 1000, 0).tolist()} ref-lead L "
                f"{np.linalg.norm(ref['left'] - arms.grip_point(s, 'left')[0]) * 100:.1f} R "
                f"{np.linalg.norm(ref['right'] - arms.grip_point(s, 'right')[0]) * 100:.1f} cm  meas-cmd {gap_txt}"
                f"  touch {s['objects'][OBJ].get('touch_links')} across_int L {across_int['left'] * 100:+.1f} R {across_int['right'] * 100:+.1f} cm  {hand_txt(p, s)}")
        now = time.time()
        err_hist.append((now, {sd: np.array(errs[sd]) for sd in errs}))
        err_hist = [(tt, ee) for tt, ee in err_hist if now - tt <= STALL_S]
        stalled = False
        if now - err_hist[0][0] >= STALL_S - DT and all(
                abs(errs[sd][0]) <= ALONG_TOL and abs(errs[sd][1]) <= ACROSS_TOL
                and -HEIGHT_TOL_BELOW <= errs[sd][2] <= STALL_HEIGHT_TOL for sd in errs):
            span = max(float(np.max(np.abs(ee[sd] - err_hist[-1][1][sd])))
                       for _, ee in err_hist for sd in errs)
            stalled = span <= STALL_MM
        if ready or stalled:
            ready_since = ready_since or now
            if now - ready_since >= 0.1:
                phase = "close"
                close_trigger = "region" if ready else "stall"
                break
        else:
            ready_since = None
    m["capture_err_mm"] = {sd: [round(v * 1000, 1) for v in errs[sd]] for sd in errs}
    # diagnostic dump of the servo end state: measured joints + last command +
    # crate pose + contact links, for an offline FK / contact reconstruction
    if DUMP_DIR:
        dump = {"round": rnd, "joints_meas": [float(v) for v in s["joints"]],
                "q_cmd": [float(v) for v in arms.arm.q], "pelvis": s["pelvis"],
                "palms": s["palms"], "crate": s["objects"][OBJ],
                "bars": {sd: [float(v) for v in bars[sd]] for sd in bars},
                "ref": {sd: [float(v) for v in ref[sd]] for sd in ref},
                "n": [float(v) for v in n], "y_hat": [float(v) for v in y_hat],
                "errs_mm": m["capture_err_mm"]}
        path = os.path.join(DUMP_DIR, f"servo_end_round{rnd}.json")
        with open(path, "w") as f:
            json.dump(dump, f, indent=1)
        log(f"[dump] {path}  touch_links {s['objects'][OBJ].get('touch_links')}")
    m["waist_pitch_cmd_deg"] = float(np.degrees(arms.arm.q[14]))
    m["waist_pitch_meas_deg"] = float(np.degrees(s["joints"][14]))
    if phase != "close":
        m["why"] = f"[grasp] capture regions never both satisfied: {m['capture_err_mm']}"
        arms.stop()
        return m
    m["close_trigger"] = close_trigger
    log(f"[grasp] both in region ({close_trigger}) after {time.time() - t0:.1f} s: {m['capture_err_mm']} "
        f"waist pitch cmd {m['waist_pitch_cmd_deg']:.0f} / meas {m['waist_pitch_meas_deg']:.0f} deg")

    # --- C: same-frame double close ------------------------------------------
    nq = int(round(CLOSE_RAMP / DT))
    for k in range(nq + 1):
        arms.set_hands(k / nq)
        arms.push(arms.arm.q)
    for _ in range(int(0.6 / DT)):
        arms.push(arms.arm.q)
    s = p.obs()
    touching = list(s["objects"][OBJ]["touching"])
    m["touching"] = touching
    if not ("left" in touching and "right" in touching):
        m["why"] = f"[grasp] closed without bilateral contact: {touching}"
        arms.stop()
        return m

    # --- D: lift, level check, hold ------------------------------------------
    z0 = float(s["objects"][OBJ]["pos"][2])
    lift_ref = {sd: arms.grip_point(s, sd)[0] for sd in ("left", "right")}
    nl = int(round(2.0 / DT))
    nh = int(round(HOLD_S / DT))
    level = 0.0                                  # + means the left hand is lowered
    max_dz = 0.0
    for k in range(1, nl + nh + 1):
        s = p.obs()
        u = min(k / nl, 1.0)
        _, bars, _, _, _ = crate_frame(p, s)
        dz = float(bars["left"][2] - bars["right"][2])   # left bar end higher: +
        max_dz = max(max_dz, abs(dz))
        level += LEVEL_LP * (float(np.clip(LEVEL_GAIN * dz, -LEVEL_CAP, LEVEL_CAP)) - level)
        z_lift = LIFT_HEIGHT * (3 * u * u - 2 * u ** 3)
        tgt = {"left": lift_ref["left"] + np.array([0.0, 0.0, z_lift - 0.5 * level]),
               "right": lift_ref["right"] + np.array([0.0, 0.0, z_lift + 0.5 * level])}
        arms.push(arms.solve(s, tgt, rot_w))
        if k % int(round(0.5 / DT)) == 0:
            log(f"[lift] t={k * DT:.1f} rise {float(s['objects'][OBJ]['pos'][2]) - z0:+.3f} m "
                f"tilt {crate_tilt_deg(s):.1f} deg bar dz {dz * 1000:+.0f} mm level {level * 1000:+.0f} mm "
                f"touch {list(s['objects'][OBJ]['touching'])}")
        if crate_tilt_deg(s) > TILT_MAX_DEG:
            m["why"] = f"[{'lift' if k <= nl else 'hold'}] crate tilted {crate_tilt_deg(s):.1f} deg"
            m.update(rise=float(s["objects"][OBJ]["pos"][2]) - z0, max_bar_dz=max_dz)
            arms.stop()
            return m
    m["max_bar_dz"] = max_dz
    s = p.obs()
    rise = float(s["objects"][OBJ]["pos"][2]) - z0
    tilt = crate_tilt_deg(s)
    touching = list(s["objects"][OBJ]["touching"])
    m.update(rise=rise, tilt_deg=tilt, touching_after=touching,
             video_s=time.time() - t_video)
    if rise < LIFT_MIN:
        m["why"] = f"[lift] rose {rise * 100:.1f} cm (< {LIFT_MIN * 100:.0f})"
    elif tilt > TILT_MAX_DEG:
        m["why"] = f"[hold] crate tilted {tilt:.1f} deg"
    elif not ("left" in touching and "right" in touching):
        m["why"] = f"[hold] lost contact: {touching}"
    else:
        m["ok"] = True
    arms.stop()
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", default="")
    ap.add_argument("--label", default="bimanual")
    args = ap.parse_args()
    inst = int(os.environ.get("INSTANCE", "0"))
    for k, base in (("TF_SONIC_PORT", 5556), ("TF_OUT_PORT", 5557), ("TF_KEYBOARD_PORT", 5580),
                    ("TF_PRIVILEGED_PORT", 5581), ("TF_EPISODE_PORT", 5582)):
        os.environ.setdefault(k, str(base + 10 * inst))
    global DUMP_DIR
    DUMP_DIR = os.path.dirname(os.path.abspath(args.manifest)) if args.manifest else ""
    meta = json.load(open(paths.SCENE_META))
    if meta.get("target_object") != OBJ:
        raise SystemExit(f"{paths.SCENE_META} is not the crate scene (target "
                         f"{meta.get('target_object')}); build it with bimanual_scene.py "
                         f"and point LOCOMANI_SIM_DIR at it")
    print(f"[bimanual] instance {inst}, scene {paths.SCENE_META}")
    p = Primitives(meta=meta)
    reset = EpisodeControlClient()
    p.cmd.reset_pose()
    reset.pin_arms()
    reset.reset()
    time.sleep(0.3)
    p.cmd.start_control()
    p.cmd.hold(2.5, mode=LocomotionMode.IDLE)
    rng = np.random.default_rng(args.seed)
    wins = 0
    for rnd in range(args.rounds):
        t0 = time.time()
        label = f"{args.label}_{rnd:02d}"
        try:
            m = run_round(p, reset, meta, rnd, rng, p.log, label)
        finally:
            try:
                reset.end_video()
            except Exception:
                pass
        m["seconds"] = round(time.time() - t0, 1)
        m["video"] = label
        wins += int(m["ok"])
        print(f"[bimanual] round {rnd}: {'OK' if m['ok'] else 'FAIL'} {m['why']} ({m['seconds']} s)")
        if args.manifest:
            with open(args.manifest, "a") as f:
                f.write(json.dumps(m, default=float) + "\n")
    print(f"[bimanual] {wins}/{args.rounds} OK")


if __name__ == "__main__":
    main()
