"""Ready-stand pilot for the WALL-TOP crate grasp (design 2026-09-07).

The crate is rotated 90 deg from the carry scene so its two full-height long
walls are left/right of the robot, and each jaw straddles a wall top with the
fingers forward, pitched down. No walk-in: the robot is reset on a stand
sampled from the walk-in's measured scatter, so this measures the grasp alone.

    python src/locomani/tasks/crate/bimanual_wall.py --rounds 20 --manifest <jsonl>

Phases (each observable, no integrators):
  A hover    both jaws 6 cm above the wall tops, at a depth along the wall
             chosen from the measured stand-off (reach_wall sweep: stand-off +
             depth <= 0.40 at palm pitch 60)
  B descend  reference lowered at DESCEND_SPEED until each pad centre is
             SEAT_BELOW under the top with the wall between the pads, or the
             reference reaches its floor
  C close    same-frame double close, bilateral contact check
  D lift     10 cm with bar-end levelling, 2 s hold; success = rise >= 6 cm,
             both hands touching, tilt < TILT_OK_DEG
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)  # reach src/ for `python src/locomani/tasks/crate/bimanual_wall.py`

from locomani import paths  # noqa: E402
from locomani.backends.sonic import upper_ref  # noqa: E402
from locomani.backends.sonic.agent_server import HAND_OPEN, LocomotionMode, StreamMode  # noqa: E402
from locomani.backends.sonic.primitives import DT, Primitives, upper_body_from_mj, wrap  # noqa: E402
from locomani.backends.sonic.stream_reach import reanchor  # noqa: E402
from locomani.sim.privileged_pub import EpisodeControlClient  # noqa: E402
from locomani.tasks.crate.bimanual_carry import (
    APPR_SPEED,
    CLOSE_RAMP,
    EDGE_FLOOR,
    HOLD_S,
    LEVEL_CAP,
    LEVEL_GAIN,
    LEVEL_LP,
    LIFT_HEIGHT,
    LIFT_MIN,
    OBJ,
    OVERDRIVE,
    OVERDRIVE_CAP,
    STAND_TOL,
    WAYPOINT_BACK,
    WAYPOINT_TOL,
    BimanualArms,
    crate_tilt_deg,
    hand_txt,
    rot_y,
    rot_z,  # noqa: E402
)

# --- crate geometry (assets/crates/blue_crate/model.xml, 2/3 crate) ----------
# Two grasp features, chosen with WALL_FEATURE:
#   wall  crate rotated 90 deg: the full-height long walls (crate-local y = +-0.1267)
#         are left/right, depth runs along crate-local x from the rail side's outer
#         face (0.2093).
#   rail  crate as in the carry scene (handles left/right): the 30 mm rim bars
#         (crate-local x = +-0.1933) with the posts removed from the asset, so the
#         slot is open under the whole bar; depth runs along crate-local y from the
#         long wall's outer face (0.1347).
FEATURE = os.environ.get("WALL_FEATURE", "rail")
if FEATURE == "wall":
    DEPTH_AXIS, WALL_AXIS = 0, 1
    WALL_Y, NEAR_X = 0.1267, 0.2093
    CRATE_YAW_DELTA = -np.pi / 2
    GRIP_BELOW = 0.080  # descend floor: pad centre this far below the top (full wall)
else:
    DEPTH_AXIS, WALL_AXIS = 1, 0
    WALL_Y, NEAR_X = 0.1933, 0.1347
    CRATE_YAW_DELTA = 0.0
    GRIP_BELOW = 0.050  # the bar is 30 mm tall: pad centre 5 cm under its top still
    # spans it (pad +-33 mm) and clears the low wall 65 mm down
TOP_Z = 0.085  # wall / bar top above the crate body origin
# --- grasp design (outputs/reach/wall_grasp_sweep.txt) -----------------------
PALM_PITCH_DEG = 60
REACH_SUM = 0.38  # stand-off + depth budget; IK limit 0.40, 2 cm margin
DEPTH_MIN, DEPTH_MAX = 0.06, 0.11  # 0.04 puts the finger root on the near rail; the hands land
# +2.5 cm deeper than the target (rail pilot 13), so 0.11 puts the
# grasp line on the crate's centre line (0.135); deeper than the
# crate's centre line (0.135) the crate pitches on the lift
# (rail pilot 11: every 0.20 m grasp tilted > 20 deg, 0.09-0.15 gave 2-13)
HOVER_ABOVE = 0.06  # pad centre above the wall top at the hover
# (GRIP_BELOW per feature above; 0.03 was eaten whole by the policy's 3-5 cm height bias)
SEAT_BELOW = 0.015  # seated when the pad centre is at least this far below the top
ACROSS_TOL = 0.015  # wall centred between the pads (47 mm jaw, 16 mm wall)
# Feed-forward for the policy's systematic 2-3 cm outward hand bias (wall pilot 9:
# 14/20 stands ended with both jaws outside their walls, L +13..+26 / R -20..-36 mm,
# the inner pad landing on the wall top). Hover/descend targets sit this far INSIDE
# the wall centre plane; errors are still measured against the true wall.
ACROSS_BIAS = 0.012  # 0.020 overshot: hands 0.6-1.4 cm inside the bar (rail pilot 11)
# The policy holds both hands 3-4 cm OUTSIDE their walls (wall pilots 2/3: across
# +40/-30 mm, inner finger riding the wall top). The across axis gets its own
# overdrive; along and height keep bimanual_carry's 0.7 / 4 cm.
ACROSS_OD, ACROSS_OD_CAP = 1.0, 0.04
LEAD_CAP = 0.06  # the commanded point never runs further than this ahead of the
# measured one (bimanual_carry's cap): an unbounded jump to the
# hover point swung the hands +-10 cm (wall pilot 4)
DESCEND_SPEED = 0.04  # m/s reference descent
HOVER_S, HOVER_TOL = 2.5, 0.03  # cap only: the hover ends when the hands stop moving (was a 5 s open-loop wait)
SHOULDER_PITCH_MIN_DEG = -80.0  # hover/grasp solutions sit at -40..-55
DESCEND_TIMEOUT = 12.0
FLOOR_DWELL = 0.5  # at the floor without seating: close anyway after this (was 2.0)
# Contact close: the descent reference keeps going down but the measured
# height stops -- the hand is resting on the bar. Visible in the image and in
# proprioception, unlike the millimetre seat test, which stays as a secondary.
CONTACT_STALL_S = 0.3
CONTACT_STALL_MM = 3.0
SEAT_DWELL = 0.3  # both seated continuously this long before the close (full-episode
# pilot 17 r0: a one-tick seat during a 3 cm/0.5 s overshoot closed on air)
TILT_OK_DEG = 20.0
# Pitch follow on the lift: the rectangular bar is clamped rigidly in the flat
# jaw, so the crate pitches with the hand (rail pilot 14: 4 -> 14 -> 20 deg as it
# rose from a folded-arm stand). The palm pitch target is counter-rotated by the
# measured crate pitch about the bar axis.
PITCH_FOLLOW_GAIN = 1.5
PITCH_FOLLOW_CAP_DEG = 25.0
PITCH_FOLLOW_LP = 0.2
# --- ready-stand distribution (walk-in scatter measured in pilots 34-37) ------
STAND_OFF_RANGE = (0.14, 0.28)  # pelvis to the crate's near face
STAND_LAT_RANGE = (-0.05, 0.05)
STAND_YAW_DEG = 5.0
STAND_YAW_MAX_DEG = 10.0  # fail fast: nothing downstream survives a bigger heading error
CRATE_DY_RANGE = (-0.08, 0.08)
SPAWN_BACK = 0.45  # reset this far behind the stand, teleport in with the arms up
# --- walk-in (WALL_WALK=1): bimanual_carry's approach in front of this grasp ---
WALK = os.environ.get("WALL_WALK", "0") == "1"
STAND_OFF_WALK = 0.24  # 0.21 (pilots 18/19) parked 0.30-0.39 with yaw +-10 deg; back to pilot 17's value for the A/B
WALK_LAT_BIAS = -0.04  # the gait parks ~4 cm left of its target (pilots 36/37)
WALK_SPAWN_BACK = (1.0, 1.3)
WALK_SPAWN_LAT = 0.15
WALK_STAND_OFF_BAND = (0.13, 0.32)  # depth = 0.38 - stand-off must stay in [0.06, 0.25]
WALK_LAT_TOL = 0.08
# Carry pose while walking: pilot 18 traces replayed in MuJoCo showed the open
# jaws hitting the near wall's top in 20/20 rounds at 2.5-4 s, pelvis still
# 0.36 m from the wall -- pads commanded 12 cm above the bar tops, delivered
# 1-8 cm (right hand lowest), fingers pitched down, hands 24 cm ahead of the
# pelvis. The dragged fingers turned the crate 7-14 deg, which IS the stand
# yaw residual. So: hands high, close to the chest, fingers level, and the
# reach forward only happens in the post-stand hover re-solve.
# Walking pose: arms at the sides. The policy's default pose carries the
# finger tips 0.19 m ahead of the pelvis at 0.56 m, i.e. into the table-edge /
# crate-wall plane at every stand under 0.23 m; the chest-high carry pose of
# pilot 21 (20/20) hid the whole crate from the ego camera at 3.5 s. This pose
# keeps the tips 1 cm ahead of the pelvis (12-30 cm behind the plane at every
# stand in the band), 27 cm out to the side, and 0% of the crate occluded
# (kinematic pre-flight 2026-09-07 evening). Joint order sp, sr, sy, el, wr,
# wp, wy; the right arm is the mirror.
WALK_SIDE_LEFT_DEG = (10.0, 10.0, 0.0, 60.0, 0.0, 30.0, 0.0)
# Reach after the stop: ONE continuous Cartesian path for each pad, a cubic
# Bezier from the side pose up along the body with the elbows out, over the
# rim, into the IK hover -- then straight into the descent, no hold. The
# G1's elbow hangs at rim height with a 0.30 m forearm, so every joint-space
# route from a hanging arm (elbow flex, forward swing, chained fixed poses)
# swept the finger tips through the rim plane at stands < 0.30 m; the chain
# of side/chest poses of pilot 22 avoided it but read as staged. Kinematic
# pre-flight of this curve (2026-09-07 night): no contact at stand-offs
# 0.13-0.31, 2.7-4.4 cm over the bar on the way in. The IK follows the curve
# with a small posture prior toward the hover solution so the wrist does not
# drift into a twisted branch (arm_ik posture term).
REACH_S = 2.5
REACH_UP = 0.25  # first control point: this far above the side pad ...
REACH_OUT = 0.15  # ... and this far outward (elbows out)
REACH_OVER = 0.10  # second control point: this far above the hover point
REACH_PITCH0_DEG = 90.0  # palm pitch at the start (fingers hanging), blends to PALM_PITCH_DEG
REACH_POSTURE_W = 0.02
# Re-approach instead of rejecting an out-of-band stand: back up to the
# waypoint and walk in again, so the data holds "stopped wrong -> corrected"
# rather than only clean stands. The crate-disturbed check still rejects.
WALK_ATTEMPTS = 3
DUMP_DIR = ""


def wall_frame(s):
    """(centre, n toward the crate, y_hat left, psi, sign, R) for the rotated crate."""
    o = s["objects"][OBJ]
    c = np.asarray(o["pos"], dtype=float)
    R = Primitives._rot(o["quat"])
    ex = R[:2, DEPTH_AXIS] / max(np.linalg.norm(R[:2, DEPTH_AXIS]), 1e-6)
    pel = np.asarray(s["pelvis"]["pos"][:2])
    sign = 1.0 if np.dot(ex, c[:2] - pel) >= 0 else -1.0
    n = sign * ex
    y_hat = np.array([-n[1], n[0]])
    psi = float(np.arctan2(n[1], n[0]))
    return c, n, y_hat, psi, sign, R


def wall_point(s, side, depth, dz, inset=0.0):
    """World point on the wall's centre plane: `depth` in from the near face, `dz` above
    the top, `inset` toward the crate centre (feed-forward for the outward hand bias)."""
    c, n, y_hat, psi, sign, R = wall_frame(s)
    # which local wall is the robot's left
    e_wall = np.zeros(3)
    e_wall[WALL_AXIS] = WALL_Y
    yl = WALL_Y if np.dot((R @ e_wall)[:2], y_hat) > 0 else -WALL_Y
    if side == "right":
        yl = -yl
    yl -= np.sign(yl) * inset
    v = np.zeros(3)
    v[DEPTH_AXIS] = sign * (-NEAR_X + depth)
    v[WALL_AXIS] = yl
    v[2] = TOP_Z + dz
    return c + R @ v


def errs_of(s, arms, side, depth):
    """Measured pad centre minus the wall line at the grasp depth: [along, across, height]."""
    c, n, y_hat, psi, sign, R = wall_frame(s)
    g, _ = arms.grip_point(s, side)
    e = g - wall_point(s, side, depth, 0.0)
    return np.array([float(e[:2] @ n), float(e[:2] @ y_hat), float(e[2])])


def run_round(
    p: Primitives, reset: EpisodeControlClient, meta: dict, rnd: int, rng: np.random.Generator, log, video_label: str
) -> dict:
    m = {"round": rnd, "ok": False, "why": ""}
    surf = meta["surfaces"]["A"]
    hx = float(surf["flat_half_extent"][0])
    edge_x = float(surf["xy"][0]) - hx
    crate0 = np.asarray(meta["objects"][OBJ]["init_pos"], dtype=float)
    w, x, y, z = meta["objects"][OBJ]["init_quat"]
    crate_yaw0 = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    # --- reset: crate rotated, near face on the table edge; robot on a sampled stand
    crate_dy = float(rng.uniform(*CRATE_DY_RANGE))
    crate_x = edge_x + NEAR_X
    so = float(rng.uniform(*STAND_OFF_RANGE))
    lat = float(rng.uniform(*STAND_LAT_RANGE))
    yaw = float(np.radians(rng.uniform(-STAND_YAW_DEG, STAND_YAW_DEG)))
    stand_xy = np.array([crate_x - NEAR_X - so, crate_dy + lat])
    robot_nom = np.asarray(meta["robot_start"]["xy"], dtype=float)
    m.update(crate_dy=crate_dy, stand_off=so, stand_lat=lat, stand_yaw_deg=float(np.degrees(yaw)))
    p.cmd.reset_pose()
    p.cmd.set_hands(left=HAND_OPEN, right=HAND_OPEN)
    reset.pin_arms()
    reset.set_ghost()
    # The robot comes up SPAWN_BACK behind the stand: spawned on the stand, the
    # policy's start-up arm swing knocked the crate 84 deg round (wall pilot 1).
    # It is teleported onto the stand once the arms are up at the hover pose.
    if WALK:
        stand_xy = np.array([crate_x - NEAR_X - STAND_OFF_WALK, crate_dy + WALK_LAT_BIAS])
        spawn_back = float(rng.uniform(*WALK_SPAWN_BACK))
        spawn_lat = float(rng.uniform(-WALK_SPAWN_LAT, WALK_SPAWN_LAT))
        spawn_xy = stand_xy - np.array([spawn_back, -spawn_lat])
        yaw = float(np.arctan2(*(np.array([crate_x, crate_dy]) - spawn_xy)[::-1])) + float(rng.uniform(-0.17, 0.17))
        m.update(spawn_back=spawn_back, spawn_lat=spawn_lat, spawn_yaw_deg=float(np.degrees(yaw)))
    else:
        spawn_xy = stand_xy - np.array([SPAWN_BACK, 0.0])
    reset.reset(
        objects={OBJ: {"dxy": [float(crate_x - crate0[0]), crate_dy], "yaw": crate_yaw0 + CRATE_YAW_DELTA}},
        robot={"dxy": [float(v) for v in (spawn_xy - robot_nom)], "yaw": yaw},
    )
    time.sleep(0.3)
    p.cmd.stream_mode = StreamMode.PLANNER_VR_3PT
    p.cmd.upper_body = None
    p.cmd.start_control()
    p._gym_started = True
    upper_ref.shared().wait(3.0)
    p.cmd.hold(2.0, mode=LocomotionMode.IDLE)
    # The deploy keeps its heading anchor across resets; setting heading_offset
    # from the pelvis yaw alone left a residual that the IDLE facing command
    # then turned into a real rotation, growing ~5 deg per round with
    # alternating sign (wall pilot 8: +2.6 -> -84 deg by round 19). A streamed
    # toggle re-anchors the deploy at the current yaw and records it.
    reanchor(p)
    fv0 = (float(np.cos(yaw)), float(np.sin(yaw)), 0.0)
    p.cmd.hold(1.5, mode=LocomotionMode.IDLE, facing=fv0)
    s = p.obs()
    if s["fallen"]:
        m["why"] = "[stand] did not stabilise after reset"
        return m
    # --- A: arms up, robot onto the stand (walk-in or teleport) ---------------
    c, n, y_hat, psi, sign, R = wall_frame(s)
    edge_x = float(surf["xy"][0]) - hx
    q_free = upper_ref.shared().upper17()
    q_free = upper_body_from_mj(s["joints"]) if q_free is None else q_free
    ramp = upper_ref.UpperRamp(p.cmd, q_free)
    p.cmd.stream_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY
    arms = BimanualArms(p)
    for sd in ("left", "right"):  # no overhead branch (wall pilot 5: +65 cm)
        a = ARM_IDX[sd]
        arms.arm.ik.lo[a[0]] = np.radians(SHOULDER_PITCH_MIN_DEG)

    def solve_for(stand, dz, pitch_deg, depth_):
        """Arm pose for the pelvis over `stand` facing psi, pads `dz` above the bar tops."""
        rot = {sd: rot_z(psi) @ rot_y(np.radians(pitch_deg)) for sd in ("left", "right")}
        tg = {sd: wall_point(s, sd, depth_, dz, ACROSS_BIAS) for sd in ("left", "right")}
        s_stand = dict(s)
        s_stand["pelvis"] = dict(
            s["pelvis"],
            pos=[float(stand[0]), float(stand[1]), float(s["pelvis"]["pos"][2])],
            quat=[float(np.cos(psi / 2)), 0.0, 0.0, float(np.sin(psi / 2))],
        )
        arms.arm.q = np.zeros(29)
        return arms.solve(s_stand, tg, rot, iters=200)

    def mirror29(left_deg):
        q = np.zeros(29)
        q[ARM_IDX["left"]] = np.radians(left_deg)
        q[ARM_IDX["right"]] = np.radians(left_deg) * np.array([1, -1, -1, 1, -1, 1, -1], dtype=float)
        return q

    def measure_stand(s):
        c_, n_, y_, psi_, _, _ = wall_frame(s)
        pel_ = np.asarray(p.pelvis_xy(s))
        near_ = wall_point(s, "left", 0.0, 0.0)
        so_ = float(np.dot(near_[:2] - pel_, n_))
        lat_ = float(np.dot(pel_ - c_[:2], y_))
        yaw_ = float(np.degrees(wrap(psi_ - p.pelvis_yaw(s))))
        return so_, lat_, yaw_

    def stand_fault(so_, lat_, yaw_):
        if abs(yaw_) > STAND_YAW_MAX_DEG:
            return f"yaw residual {yaw_:+.1f} deg"
        if not (WALK_STAND_OFF_BAND[0] <= so_ <= WALK_STAND_OFF_BAND[1]):
            return f"stand-off {so_:.2f} m outside the reach band"
        if abs(lat_) > WALK_LAT_TOL:
            return f"lateral {lat_ * 100:+.1f} cm off the centre line"
        return ""

    if WALK:
        reset.start_video(video_label)
        t_video = time.time()
        q_walk17 = upper_body_from_mj(mirror29(WALK_SIDE_LEFT_DEG))
        way_xy = stand_xy - n * WAYPOINT_BACK
        m["stand_target"] = [float(v) for v in stand_xy]
        m["stand_attempts"] = []
        for attempt in range(WALK_ATTEMPTS):
            t0, why = time.time(), "timeout"
            best_along, best_t = None, time.time()
            leg = "waypoint"
            while time.time() - t0 < 40.0:
                s = p.obs()
                if s["fallen"]:
                    m["why"] = "[stand] fell during the walk-in"
                    return m
                pel = np.asarray(p.pelvis_xy(s))
                to_stand = stand_xy - pel
                along = float(np.dot(to_stand, n))
                dist = float(np.linalg.norm(to_stand))
                ramp.push(q_walk17)
                if leg == "waypoint":
                    to_way = way_xy - pel
                    if float(np.dot(to_way, n)) <= 0.03 or float(np.linalg.norm(to_way)) < WAYPOINT_TOL:
                        leg = "in"
                        log(
                            f"[walk] waypoint reached: lateral {float(np.dot(pel - stand_xy, y_hat)) * 100:+.1f} cm, "
                            f"{along * 100:.0f} cm to the stand"
                        )
                edge_clear = edge_x - float(pel[0])
                if along <= 0.04 or dist < STAND_TOL:
                    why = "at the stand"
                    break
                if edge_clear <= EDGE_FLOOR:
                    why = "edge floor"
                    break
                if best_along is None or along < best_along - 0.015:
                    best_along, best_t = along, time.time()
                elif time.time() - best_t > 3.0:
                    why = "stalled"
                    break
                look = c[:2] - pel
                b = float(np.arctan2(look[1], look[0]))
                speed = APPR_SPEED[0] if along > 0.35 else APPR_SPEED[1] if along > 0.15 else APPR_SPEED[2]
                if leg == "waypoint":
                    mv = to_way / max(float(np.linalg.norm(to_way)), 1e-3)
                else:
                    mv = to_stand / max(dist, 1e-3)
                    b = psi
                if int((time.time() - t0) / 2.0) != int((time.time() - t0 - DT) / 2.0):
                    log(
                        f"[walk] t={time.time() - t0:4.1f} pelvis {np.round(pel, 2).tolist()} dist {dist:.2f} along {along:+.2f} "
                        f"edge {edge_clear:.2f} speed {speed} facing_err {np.degrees(wrap(b - p.pelvis_yaw(s))):+.0f} deg  {hand_txt(p, s)}"
                    )
                p.cmd.frame(
                    mode=LocomotionMode.SLOW_WALK,
                    facing=(float(np.cos(b)), float(np.sin(b)), 0.0),
                    movement=(float(mv[0]), float(mv[1]), 0.0),
                    speed=speed,
                )
                time.sleep(DT)
            m["approach_exit"] = why
            if why == "timeout":
                m["why"] = "[stand] walk-in timed out"
                return m
            # brake: pelvis under 0.06 m/s for 0.25 s
            t0, still = time.time(), None
            prev = p.pelvis_xy(p.obs())
            while time.time() - t0 < 6.0:
                ramp.push(q_walk17)
                p.cmd.frame(mode=LocomotionMode.IDLE, facing=(float(np.cos(psi)), float(np.sin(psi)), 0.0))
                time.sleep(DT)
                cur = p.pelvis_xy(p.obs())
                v = float(np.linalg.norm(cur - prev)) / DT
                prev = cur
                still = time.time() if v > 0.06 else (still if still is not None else time.time())
                if still is not None and time.time() - still > 0.25 and time.time() - t0 > 1.0:
                    break
            ramp.settle()
            s = p.obs()
            if s["fallen"]:
                m["why"] = "[stand] fell at the stand"
                return m
            so_a, lat_a, yaw_a = measure_stand(s)
            fault = stand_fault(so_a, lat_a, yaw_a)
            m["stand_attempts"].append(dict(stand_off=so_a, lateral=lat_a, yaw_deg=yaw_a, fault=fault))
            if not fault:
                break
            log(
                f"[stand] attempt {attempt + 1}: {fault} (stand-off {so_a:.3f}, lateral {lat_a:+.3f}, yaw {yaw_a:+.1f})"
            )
            if attempt == WALK_ATTEMPTS - 1:
                m["why"] = f"[stand] {fault} after {WALK_ATTEMPTS} walk-ins"
                return m
            # back up to the waypoint, facing the crate, and walk in again
            t0 = time.time()
            while time.time() - t0 < 8.0:
                s = p.obs()
                if s["fallen"]:
                    m["why"] = "[stand] fell backing up"
                    return m
                pel = np.asarray(p.pelvis_xy(s))
                if float(np.dot(stand_xy - pel, n)) >= WAYPOINT_BACK:
                    break
                ramp.push(q_walk17)
                p.cmd.frame(
                    mode=LocomotionMode.SLOW_WALK,
                    facing=(float(np.cos(psi)), float(np.sin(psi)), 0.0),
                    movement=(float(-n[0]), float(-n[1]), 0.0),
                    speed=APPR_SPEED[2],
                )
                time.sleep(DT)
            log(
                f"[walk] backed up to {float(np.dot(stand_xy - np.asarray(p.pelvis_xy(p.obs())), n)):.2f} m for attempt {attempt + 2}"
            )
        m["stand_attempt_n"] = len(m["stand_attempts"])
    else:
        depth = float(np.clip(REACH_SUM - so, DEPTH_MIN, DEPTH_MAX))  # from the sampled stand; re-measured below
        q_hover = solve_for(stand_xy, HOVER_ABOVE, PALM_PITCH_DEG, depth)
        q_h17 = upper_body_from_mj(q_hover)
        n_ramp = int(2.0 / DT)
        for k in range(1, n_ramp + 1):
            u = k / n_ramp
            ramp.push(q_free + (3 * u * u - 2 * u**3) * (q_h17 - q_free))
            p.cmd.frame(mode=LocomotionMode.IDLE, facing=(float(np.cos(psi)), float(np.sin(psi)), 0.0))
            time.sleep(DT)
        ramp.settle()
        # teleport the standing robot onto the stand, arms already up
        reset.align_robot_xy(stand_xy)
        for _ in range(int(1.0 / DT)):
            ramp.push(q_h17)
            p.cmd.frame(mode=LocomotionMode.IDLE, facing=(float(np.cos(psi)), float(np.sin(psi)), 0.0))
            time.sleep(DT)
        ramp.settle()
        s = p.obs()
        if s["fallen"]:
            m["why"] = "[stand] fell after the teleport"
            return m
        reset.start_video(video_label)
        t_video = time.time()

    # measured stand -> depth along the wall
    c, n, y_hat, psi, sign, R = wall_frame(s)
    pel = np.asarray(p.pelvis_xy(s))
    near = wall_point(s, "left", 0.0, 0.0)
    so_meas = float(np.dot(near[:2] - pel, n))
    lat_meas = float(np.dot(pel - c[:2], y_hat))
    yaw_err = float(np.degrees(wrap(psi - p.pelvis_yaw(s))))
    depth = float(np.clip(REACH_SUM - so_meas, DEPTH_MIN, DEPTH_MAX))
    m.update(stand_off_meas=so_meas, stand_lat_meas=lat_meas, yaw_err_deg=yaw_err, depth=depth)
    log(
        f"[stand] stand-off {so_meas:.3f} m, lateral {lat_meas:+.3f} m, yaw {yaw_err:+.1f} deg -> depth {depth:.2f}  {hand_txt(p, s)}"
    )
    if abs(yaw_err) > STAND_YAW_MAX_DEG:
        m["why"] = f"[stand] yaw residual {yaw_err:+.1f} deg after the {'walk-in' if WALK else 'teleport'}"
        return m
    # hover pose for the stand we actually have (from zeros: the validated branch)
    q_hover = solve_for(pel, HOVER_ABOVE, PALM_PITCH_DEG, depth)
    for sd in ("left", "right"):
        pos, Rk = arms.arm.ik.fk(q_hover, sd)
        g = np.asarray(s["pelvis"]["pos"]) + Primitives._rot(s["pelvis"]["quat"]) @ (pos + Rk @ arms_offset())
        log(
            f"[hover] {sd}: ik residual {np.linalg.norm(g - wall_point(s, sd, depth, HOVER_ABOVE, ACROSS_BIAS)) * 100:.1f} cm, "
            f"arm q {np.round(np.degrees(q_hover[ARM_IDX[sd]]), 0).tolist()}"
        )
    rot_w = {sd: rot_z(psi) @ rot_y(np.radians(PALM_PITCH_DEG)) for sd in ("left", "right")}
    if not WALK:
        # teleport path keeps the old joint ramp onto the hover pose
        q_h17 = upper_body_from_mj(q_hover)
        q_from = ramp.q.copy()
        n_ramp = int(1.0 / DT)
        for k in range(1, n_ramp + 1):
            u = k / n_ramp
            ramp.push(q_from + (3 * u * u - 2 * u**3) * (q_h17 - q_from))
            p.cmd.frame(mode=LocomotionMode.IDLE, facing=(float(np.cos(psi)), float(np.sin(psi)), 0.0))
            time.sleep(DT)
        ramp.settle()
        arms.start(q17=ramp.q)
        arms.arm.q = q_hover.copy()
    else:
        # one continuous reach: side pads -> Bezier -> hover, streamed as IK
        # joint targets, warm-started from the previous command with the
        # posture prior toward q_hover
        arms.start(q17=ramp.q)
        arms.arm.q = np.asarray(s["joints"], dtype=float).copy()
        c, n, y_hat, psi, sign, R = wall_frame(s)
        p0 = {sd: arms.grip_point(s, sd)[0].copy() for sd in ("left", "right")}
        p3 = {sd: wall_point(s, sd, depth, HOVER_ABOVE, ACROSS_BIAS) for sd in ("left", "right")}
        out_w = {"left": np.array([y_hat[0], y_hat[1], 0.0]), "right": -np.array([y_hat[0], y_hat[1], 0.0])}
        p1 = {sd: p0[sd] + out_w[sd] * REACH_OUT + np.array([0.0, 0.0, REACH_UP]) for sd in p0}
        p2 = {sd: p3[sd] + np.array([0.0, 0.0, REACH_OVER]) for sd in p3}
        n_reach = int(REACH_S / DT)
        t0 = time.time()
        last_log = 0.0
        for k in range(1, n_reach + 1):
            u = k / n_reach
            s = p.obs()
            if s["fallen"]:
                m["why"] = "[reach] fell"
                arms.stop()
                return m
            tg = {
                sd: (1 - u) ** 3 * p0[sd] + 3 * (1 - u) ** 2 * u * p1[sd] + 3 * (1 - u) * u * u * p2[sd] + u**3 * p3[sd]
                for sd in p0
            }
            pitch = REACH_PITCH0_DEG + (PALM_PITCH_DEG - REACH_PITCH0_DEG) * u
            rot_u = {sd: rot_z(psi) @ rot_y(np.radians(pitch)) for sd in ("left", "right")}
            arms.push(arms.solve(s, tg, rot_u, iters=60, posture=q_hover, posture_weight=REACH_POSTURE_W))
            if time.time() - last_log > 0.5:
                last_log = time.time()
                res = {sd: float(np.linalg.norm(arms.grip_point(s, sd)[0] - tg[sd])) for sd in tg}
                log(
                    f"[reach] u={u:.2f} lag L {res['left'] * 100:.1f} R {res['right'] * 100:.1f} cm  "
                    f"arm q L {np.round(np.degrees(arms.arm.q[ARM_IDX['left']]), 0).tolist()}"
                )
            time.sleep(DT)
        m["reach_s"] = round(time.time() - t0, 1)

    def servo_tick(s, ref):
        cmd = {}
        _, n_, y_, _, _, _ = wall_frame(s)
        yh = np.array([y_[0], y_[1], 0.0])
        for sd in ("left", "right"):
            g, _ = arms.grip_point(s, sd)
            e = ref[sd] - g
            ld = float(np.linalg.norm(e))
            if ld > LEAD_CAP:  # rate-limit: lead capped, not the target
                e = e * (LEAD_CAP / ld)
            e_across = float(e @ yh)
            corr = np.clip(OVERDRIVE * (e - e_across * yh), -OVERDRIVE_CAP, OVERDRIVE_CAP)
            corr += yh * float(np.clip(ACROSS_OD * e_across, -ACROSS_OD_CAP, ACROSS_OD_CAP))
            cmd[sd] = g + e + corr
        arms.push(arms.solve(s, cmd, rot_w, from_measured=True))

    # No hover hold: the descent starts as the reach ends (the descend servo
    # takes up whatever lag the policy still carries). The teleport path
    # keeps a short settle so the old A/B numbers stay comparable.
    crate_start = c.copy()
    s = p.obs()
    ref = {sd: wall_point(s, sd, depth, HOVER_ABOVE, ACROSS_BIAS) for sd in ("left", "right")}
    if not WALK:
        t0 = time.time()
        while time.time() - t0 < HOVER_S:
            s = p.obs()
            arms.push(q_hover)
            time.sleep(DT)
    res = {sd: float(np.linalg.norm(arms.grip_point(s, sd)[0] - ref[sd])) for sd in ref}
    log(
        f"[hover] residual L {res['left'] * 100:.1f} R {res['right'] * 100:.1f} cm "
        f"errs L {np.round(errs_of(s, arms, 'left', depth) * 1000, 0).tolist()} R {np.round(errs_of(s, arms, 'right', depth) * 1000, 0).tolist()}"
    )
    m["hover_residual_mm"] = {sd: round(v * 1000, 1) for sd, v in res.items()}

    # --- B: descend until seated ----------------------------------------------
    t0 = time.time()
    dz = HOVER_ABOVE
    seated = {"left": False, "right": False}
    seated_since = None
    floor_since = None
    last_log = 0.0
    close_trigger = None
    hz_hist = []  # (t, mean measured pad height error)
    while time.time() - t0 < DESCEND_TIMEOUT:
        s = p.obs()
        if s["fallen"]:
            m["why"] = "[descend] fell"
            arms.stop()
            return m
        c, n, y_hat, psi, sign, R = wall_frame(s)
        if float(np.linalg.norm(c[:2] - crate_start[:2])) > 0.05:
            m["why"] = "[descend] crate pushed > 5 cm before closing"
            arms.stop()
            return m
        dz = max(dz - DESCEND_SPEED * DT, -GRIP_BELOW)
        ref = {sd: wall_point(s, sd, depth, dz, ACROSS_BIAS) for sd in ("left", "right")}
        servo_tick(s, ref)
        e = {sd: errs_of(s, arms, sd, depth) for sd in ("left", "right")}
        for sd in e:
            seated[sd] = bool(abs(e[sd][1]) <= ACROSS_TOL and e[sd][2] <= -SEAT_BELOW)
        if time.time() - last_log > 0.5:
            last_log = time.time()
            log(
                f"[descend] t={time.time() - t0:.1f} ref dz {dz * 100:+.1f} cm errs L {np.round(e['left'] * 1000, 0).tolist()} "
                f"R {np.round(e['right'] * 1000, 0).tolist()} seated {seated} touch {s['objects'][OBJ].get('touch_links')}  {hand_txt(p, s)}"
            )
        # primary: the reference is still descending but the hands are not --
        # they are resting on the bar (contact), and across is within tolerance
        hz_hist.append((time.time(), float(np.mean([e[sd][2] for sd in e]))))
        hz_hist = [(tt, h) for tt, h in hz_hist if time.time() - tt <= CONTACT_STALL_S]
        descending = dz > -GRIP_BELOW + 1e-6
        if (
            descending
            and hz_hist[0][0] <= time.time() - CONTACT_STALL_S + DT
            and abs(hz_hist[-1][1] - hz_hist[0][1]) * 1000 < CONTACT_STALL_MM
            and all(abs(e[sd][1]) <= ACROSS_TOL + 0.005 for sd in e)
            and all(e[sd][2] <= 0.0 for sd in e)
        ):
            close_trigger = "contact"
            break
        if seated["left"] and seated["right"]:
            seated_since = seated_since or time.time()
            if time.time() - seated_since >= SEAT_DWELL:
                close_trigger = "seated"
                break
        else:
            seated_since = None
        if dz <= -GRIP_BELOW + 1e-6:
            floor_since = floor_since or time.time()
            if time.time() - floor_since >= FLOOR_DWELL:
                # the hands must actually be at the bar top: pilot 23 r5 closed
                # on air with both pads 26-33 mm above it from a 0.30 m stand
                if all(abs(e[sd][1]) <= ACROSS_TOL + 0.005 and e[sd][2] <= 0.005 for sd in e):
                    close_trigger = "floor"
                    break
        time.sleep(DT)
    m["capture_err_mm"] = {sd: [round(v * 1000, 1) for v in e[sd]] for sd in e}
    m["close_trigger"] = close_trigger
    m["descend_s"] = round(time.time() - t0, 1)
    if DUMP_DIR:
        with open(os.path.join(DUMP_DIR, f"wall_end_round{rnd}.json"), "w") as f:
            json.dump(
                {
                    "round": rnd,
                    "joints_meas": [float(v) for v in s["joints"]],
                    "q_cmd": [float(v) for v in arms.arm.q],
                    "pelvis": s["pelvis"],
                    "palms": s["palms"],
                    "crate": s["objects"][OBJ],
                    "depth": depth,
                    "errs_mm": m["capture_err_mm"],
                },
                f,
                indent=1,
            )
    if close_trigger is None:
        m["why"] = f"[descend] never seated: {m['capture_err_mm']}"
        arms.stop()
        return m
    log(f"[grasp] closing ({close_trigger}) after {m['descend_s']} s: {m['capture_err_mm']}")

    # --- C: close -------------------------------------------------------------
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

    # --- D: lift with levelling, hold -----------------------------------------
    z0 = float(s["objects"][OBJ]["pos"][2])
    lift_ref = {sd: arms.grip_point(s, sd)[0] for sd in ("left", "right")}
    nl = int(round(2.0 / DT))
    nh = int(round(HOLD_S / DT))
    level, max_dz = 0.0, 0.0
    pitch_cmd, max_pitch = 0.0, 0.0
    for k in range(1, nl + nh + 1):
        s = p.obs()
        u = min(k / nl, 1.0)
        gl, gr = wall_point(s, "left", depth, 0.0), wall_point(s, "right", depth, 0.0)
        dzb = float(gl[2] - gr[2])
        # crate pitch about the bar axis: + when the far end is low (nose down),
        # the same sense as the palm pitching its fingers down
        _, _, _, _, sgn, Rc = wall_frame(s)
        e_depth = np.zeros(3)
        e_depth[DEPTH_AXIS] = sgn
        d_far = Rc @ e_depth
        pitch_down = float(np.degrees(np.arcsin(np.clip(-d_far[2], -1.0, 1.0))))
        max_pitch = max(max_pitch, abs(pitch_down))
        pitch_cmd += PITCH_FOLLOW_LP * (
            float(np.clip(PITCH_FOLLOW_GAIN * pitch_down, -PITCH_FOLLOW_CAP_DEG, PITCH_FOLLOW_CAP_DEG)) - pitch_cmd
        )
        rot_w = {sd: rot_z(psi) @ rot_y(np.radians(PALM_PITCH_DEG - pitch_cmd)) for sd in ("left", "right")}
        max_dz = max(max_dz, abs(dzb))
        level += LEVEL_LP * (float(np.clip(LEVEL_GAIN * dzb, -LEVEL_CAP, LEVEL_CAP)) - level)
        z_lift = LIFT_HEIGHT * (3 * u * u - 2 * u**3)
        tgt = {
            "left": lift_ref["left"] + np.array([0.0, 0.0, z_lift - 0.5 * level]),
            "right": lift_ref["right"] + np.array([0.0, 0.0, z_lift + 0.5 * level]),
        }
        # Open IK target on purpose: the closed-loop lift (servo_tick) capped
        # the rise at ~6.5 cm (lead cap + the policy's height bias, rail pilot
        # 15), while the open target rises 9-14 cm. The pitch follow above is
        # what keeps the crate level from a close stand.
        arms.push(arms.solve(s, tgt, rot_w))
        if k % int(round(0.5 / DT)) == 0:
            log(
                f"[lift] t={k * DT:.1f} rise {float(s['objects'][OBJ]['pos'][2]) - z0:+.3f} m tilt {crate_tilt_deg(s):.1f} deg "
                f"pitch_down {pitch_down:+.1f} palm {PALM_PITCH_DEG - pitch_cmd:.0f} "
                f"wall dz {dzb * 1000:+.0f} mm level {level * 1000:+.0f} mm touch {list(s['objects'][OBJ]['touching'])}"
            )
        if crate_tilt_deg(s) > TILT_OK_DEG:
            m["why"] = f"[{'lift' if k <= nl else 'hold'}] crate tilted {crate_tilt_deg(s):.1f} deg"
            m.update(rise=float(s["objects"][OBJ]["pos"][2]) - z0, max_wall_dz=max_dz)
            arms.stop()
            return m
    s = p.obs()
    rise = float(s["objects"][OBJ]["pos"][2]) - z0
    tilt = crate_tilt_deg(s)
    touching = list(s["objects"][OBJ]["touching"])
    m.update(
        rise=rise,
        tilt_deg=tilt,
        touching_after=touching,
        max_wall_dz=max_dz,
        max_pitch_deg=max_pitch,
        video_s=time.time() - t_video,
    )
    if rise < LIFT_MIN:
        m["why"] = f"[lift] rose {rise * 100:.1f} cm (< {LIFT_MIN * 100:.0f})"
    elif not ("left" in touching and "right" in touching):
        m["why"] = f"[hold] lost contact: {touching}"
    else:
        m["ok"] = True
    arms.stop()
    return m


def arms_offset():
    from locomani.tasks.crate.bimanual_carry import OFFSET

    return OFFSET


from locomani.primitives.g1_joints import ARM_MJ as ARM_IDX  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", default="")
    ap.add_argument("--label", default="wall")
    args = ap.parse_args()
    inst = int(os.environ.get("INSTANCE", "0"))
    for k, base in (
        ("TF_SONIC_PORT", 5556),
        ("TF_OUT_PORT", 5557),
        ("TF_KEYBOARD_PORT", 5580),
        ("TF_PRIVILEGED_PORT", 5581),
        ("TF_EPISODE_PORT", 5582),
    ):
        os.environ.setdefault(k, str(base + 10 * inst))
    global DUMP_DIR
    DUMP_DIR = os.path.dirname(os.path.abspath(args.manifest)) if args.manifest else ""
    meta = json.load(open(paths.SCENE_META))
    if meta.get("target_object") != OBJ:
        raise SystemExit(f"{paths.SCENE_META} is not the crate scene")
    print(f"[wall] instance {inst}, scene {paths.SCENE_META}")
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
    dead_rounds = 0  # consecutive "did not stabilise": the deploy is gone
    restarts = 0
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
        m["deploy_restarts"] = restarts
        wins += int(m["ok"])
        # Self-heal: the SONIC deploy stops itself on a one-loop LowState
        # timeout (rail pilot 12: another session's per-round deploy restarts
        # blip the sim clock), after which every reset "does not stabilise".
        # Two in a row -> restart our own deploy (port-scoped) and go on.
        dead_rounds = dead_rounds + 1 if m["why"].startswith("[stand] did not stabilise") else 0
        if dead_rounds >= 2:
            import subprocess

            print(f"[wall] deploy looks dead ({dead_rounds} rounds); restarting instance {inst} deploy")
            r = subprocess.run(
                ["bash", os.path.join(paths.REPO, "scripts", "restart_deploy.sh")],
                env=dict(os.environ, INSTANCE=str(inst)),
                cwd=paths.REPO,
            )
            print(f"[wall] restart_deploy.sh exit {r.returncode}")
            restarts += 1
            dead_rounds = 0
            p.cmd.reset_pose()
            reset.pin_arms()
            reset.reset()
            time.sleep(0.3)
            p.cmd.start_control()
            p.cmd.hold(2.5, mode=LocomotionMode.IDLE)
        print(f"[wall] round {rnd}: {'OK' if m['ok'] else 'FAIL'} {m['why']} ({m['seconds']} s)")
        if args.manifest:
            with open(args.manifest, "a") as f:
                f.write(json.dumps(m, default=float) + "\n")
    print(f"[wall] {wins}/{args.rounds} OK")


if __name__ == "__main__":
    main()
