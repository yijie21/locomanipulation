"""One scripted loco-manipulation episode, start to finish (P1.2).

Six stages, each with a privileged predicate that must hold before the next
begins. The predicates are the point: they are what turns a scripted sequence
into a *dataset*, because an episode that fails one is discarded rather than
silently teaching the policy a failed behaviour.

    approach_A   pelvis within reach of surface A, orange in front of it
    grasp        fingers in contact with the orange, and it lifts with the hand
    transport    orange stays put in the palm frame while the robot walks 2.5 m
    approach_B   pelvis within reach of surface B
    place        orange resting on the plate, hand no longer touching
    retreat      robot has stepped back, orange still on the plate

`transport` is checked in the *palm* frame, not the world frame: the whole
point of the stage is that the object moves with the hand, so a world-frame
displacement test would flag a successful carry as failure. A drop is a hard
fail — no partial credit for an episode where the object hits the floor.

Usage:
    python teleop_free/collect_episode.py [--no-record] [--out <dir>]

Assumes the stack is already up (teleop_free/start_stack.sh) with the sim
launched via run_sim_locomani.py so :5581 carries privileged state.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import adaptive_grasp  # noqa: E402
import reach_map  # noqa: E402
import grasping  # noqa: E402
import stream_reach  # noqa: E402

from primitives import (  # noqa: E402
    ARM_REACH_LIMIT,
    CROUCH_MARGIN,
    CROUCH_SKIP,
    PALM_PITCH_ACHIEVED,
    STANDING_PELVIS_Z,
    GRASP_BEARING,
    SHOULDER_OFFSET,
    GRASP_ABOVE,
    GRASP_CENTRE_OFFSET,
    PREGRASP_BACK,
    LIFT_HEIGHT,
    PREGRASP_ABOVE,
    LocomotionMode,
    StreamMode,
    Primitives,
    STAND_OFF,
    TRACK_STOP_SHORT,
    READY_WRISTS,
    wrap,
)
from privileged_pub import EpisodeControlClient  # noqa: E402

SCENE_META = "/workspace/code/locomanipulation/outputs/scene/sim/scene_meta.json"

# Predicate thresholds. Deliberately explicit rather than tuned-in-place, so a
# failed episode can be attributed to a specific one.
# Pelvis-to-object planar distance that counts as "at" a surface. The palm
# reaches about 0.50 m in front of the pelvis, but only descends to table height
# from around 0.35 m (teleop_free/diag_reach.py) — so the bound here is set by
# where the arm can still work, not by where it can barely touch.
# How close the pelvis has to get before the arm takes over — 0.36, from the
# reachability map, not from another round of guessing. Past 0.40 m the arm can
# reach the staging point but not the grip point, at any bearing; that gap is
# what every "reach stalled" has been.
# 0.45, was 0.40: the table's inflation ring parks the pelvis at ~0.44 m on
# this layout and five re-plans cannot do better. The streamed reach with
# waist lean + squat covers grip x up to ~0.42 in the pelvis frame
# (diag_stream --waist --squat), so let the grasp stage's own IK residual be
# the reachability judge instead of failing the approach.
# 0.65, was 0.45: with the can pulled 12 cm inboard (so passing arms stop
# sweeping it off the table) the legal stand ring bottoms out ~0.46-0.62 m.
# Coarse arrival is all the approach owes us now — the streamed grasp's own
# guarded walk-in (stream_reach._nudge_closer) closes the final gap to a
# pelvis-frame grip depth of 0.24-0.34 m.
# Stage-1 endpoint: a SAFE distance — greater than arm extent + object
# radius, so no contact is possible regardless of anything's height. The
# careful stage-2 approach (stream_reach.final_approach) closes the rest.
REACH_XY_TOL = 0.90
# The plate sits deep inside surface B's footprint: the legal stand ring is
# ~0.50 m out no matter how many times the approach re-plans. Dropping from
# above is tolerant, so let the place stages accept what the geometry allows.
PLACE_XY_REACH = 0.58
LIFT_MIN = 0.06            # the orange must actually leave the table
PALM_DRIFT_MAX = 0.03      # 3 cm of slip in the palm frame before it counts as dropped
PLACE_XY_TOL = 0.09        # orange centre within this of the plate centre
PLACE_Z_TOL = 0.06         # and resting at plate-top height
RETREAT_MIN = 0.30         # how far the pelvis must back off at the end
STAND_SETTLE = 4.0         # seconds of IDLE after a reset before any walking
OBJECT_NUDGE_MAX = 0.10    # how far the arm may shove the object before grasping
AIM_TURN_W = 0.05          # metres of reach a radian of in-place turn is worth
# Full task = grasp + transport + place + retreat. Off by default while the
# grasp is under iteration: a grasp-only episode is the same data at half the
# wall-clock. Turn the rest back on with FULL_TASK=1.
FULL_TASK = os.environ.get("FULL_TASK", "0") == "1"


class EpisodeFailed(Exception):
    def __init__(self, stage: str, why: str):
        super().__init__(f"[{stage}] {why}")
        self.stage = stage
        self.why = why


def obj_top(meta: dict, obj_key: str) -> float:
    """World z of the object's top, from its bounding box."""
    o = meta["objects"][obj_key]

    return float(o["init_pos"][2] + np.asarray(o["box"]["half_extent"])[2])


def palm_frame_offset(s: dict, obj_key: str, side: str) -> np.ndarray:
    """Object position expressed in the palm's frame."""
    palm = np.array(s["palms"][side]["pos"])
    w, x, y, z = s["palms"][side]["quat"]
    # Rotation matrix from a scalar-first quaternion.
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])

    return R.T @ (np.array(s["objects"][obj_key]["pos"]) - palm)


def run_episode(p: Primitives, meta: dict, side: str = "right",
                record: bool = True, reset: EpisodeControlClient | None = None,
                perturb: dict | None = None) -> dict:
    obj_key = meta["target_object"]
    goal_key = meta["goal_object"]
    surf_a = meta["surfaces"]["A"]
    surf_b = meta["surfaces"]["B"]
    result = {"stages": {}, "ok": False}

    def stage(name: str) -> None:
        result["stages"][name] = {"t": time.time()}
        print(f"  -- {name}")

    def require(name: str, cond: bool, why: str) -> None:
        result["stages"].setdefault(name, {})["ok"] = bool(cond)
        if not cond:
            raise EpisodeFailed(name, why)

    # Reset before control, not after: `start_control` makes the deploy latch
    # onto the current state, and teleporting the robot after that leaves its
    # odometry integrating from a pose that no longer exists.
    #
    # But the gap between the two has to be short. The reset restores qpos0,
    # which is the URDF's straight-legged pose, and with the elastic band off
    # nothing holds the robot up until the controller takes over — a 1.5 s gap
    # is 1.5 s of free fall, which is how the first attempt ended flat on the
    # floor during approach_A.
    # Before the reset, not after: the deploy is still streaming, and the first
    # frames it sees once control starts should already carry the rest pose.
    p.cmd.reset_pose()
    p.hold_yaw = None            # a failed episode must not leak its anchor
    p.crouch_hold = None
    if reset is not None:
        # Close whatever the previous episode left recording before this one's
        # reset teleports the robot into the frame.
        reset.end_video()
        reset.reset(objects=perturb.get("objects") if perturb else None,
                    robot=perturb.get("robot") if perturb else None,
                    label=perturb.get("label", "") if perturb else "")
        time.sleep(0.2)

    p.cmd.start_control()
    # Stand before walking: the controller needs a moment to settle the stance
    # out of the nominal pose, and a walk command issued into that transient is
    # what tips it over.
    p.cmd.hold(STAND_SETTLE, mode=LocomotionMode.IDLE)
    s = p.obs()
    require("stand", not s["fallen"],
            f"robot did not stabilise after reset (pelvis z {s['pelvis']['pos'][2]:.3f})")
    # The deploy's planner heading anchor may be stale from a previous
    # process (any streamed-mode exit re-anchors it at whatever yaw the robot
    # had — diag_heading.py). Re-anchor it HERE at a yaw this process knows,
    # so the commander's compensation is exact from the first step.
    stream_reach.reanchor(p, side=side)
    # FROZEN_UPPER_BODY: the one planner sub-mode that holds an arm pose
    # THROUGH a walk (VR-3pt wrist tracking dies the moment the gait starts).
    # tuck_arms then pins the READY pose — forearms level, hands above the
    # tabletop — from the first step to the table.
    p.cmd.stream_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY
    p.tuck_arms()
    if reset is not None:
        # The actuator-level pin is what actually HOLDS the level forearms
        # through the gait; the frozen reference alone under-delivers.
        # Safety near obstacles is DISTANCE-staged (stage 1 stops 0.9 m out)
        # and reactive (stage 2 backs off on any disturbance) — no altitude
        # assumptions about the world.
        from primitives import READY_ARM
        reset.pin_arms(left=READY_ARM["left"], right=READY_ARM["right"])
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE)

    # Both recordings start here, after the settle, so their first frame is the
    # first frame of the task rather than four seconds of a robot standing
    # still. The exporter was already wired this way; the video was not.
    if reset is not None:
        reset.start_video(perturb["video"] if perturb and perturb.get("video")
                          else f"adhoc_{int(time.time())}")
    if record:
        p.cmd.key("c")           # IDLE -> RECORDING; a second 'c' saves
        print("  recording started")

    start_xy = p.pelvis_xy()

    def show_ghost(pl) -> None:
        """Park the transparent hand at a plan, for the video.

        The tilt has to be in here too. The ghost is the only picture of what
        the reach is aiming at, and drawing a level hand while the arm goes for
        a tilted one would make every frame of every video misleading in the
        one place they are used. It is drawn at the pitch the palm ACHIEVES
        (~23 deg), not the 105 deg commanded — the command is a lever on a
        heavily attenuated channel, not an angle.
        """
        if reset is None or not pl.ok:
            return
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, np.asarray(pl.rot, dtype=float).flatten())
        q = q.tolist()
        reset.set_ghost(pos=np.asarray(pl.wrist, dtype=float).tolist(),
                        quat=q)

    def plan_for(pelvis_xy, yaw, state=None):
        state = state or p.obs()

        return grasping.grasp_pose(
            obj_pos=p.object_pos(obj_key, state),
            obj_quat=state["objects"][obj_key]["quat"],
            box=meta["objects"][obj_key]["box"], surface_top=surf_a["top"],
            pelvis_xy=pelvis_xy, pelvis_yaw=yaw,
            approach_dir=(np.cos(yaw), np.sin(yaw)), pitched=True)

    # Show the target from frame one, from a PREDICTED stand pose. The grip
    # point itself is a fact about the object and does not wait on the robot,
    # but the orientation now genuinely does — the whole point of the adaptive
    # plan is that it depends on where the hand is — so the frame-0 ghost is a
    # forecast: stand on the ring running outward from the table's centre
    # through the object, turned GRASP_BEARING past it, arms tucked. It is
    # replanned from the real pose on arrival, and the difference between the
    # two is exactly the thing the video is there to show.
    _o = p.object_pos(obj_key)[:2] - np.asarray(surf_a["xy"], dtype=float)
    _o = _o / max(float(np.linalg.norm(_o)), 1e-6)
    _stand = p.object_pos(obj_key)[:2] + _o * STAND_OFF
    _head = float(np.arctan2(*(p.object_pos(obj_key)[:2] - _stand)[::-1]))
    _head += GRASP_BEARING if side == "right" else -GRASP_BEARING
    _yawrot = Primitives._rot([np.cos(_head / 2), 0.0, 0.0, np.sin(_head / 2)])
    _wrist = (np.array([_stand[0], _stand[1], STANDING_PELVIS_Z])
              + _yawrot @ np.asarray(READY_WRISTS[side], dtype=float))
    show_ghost(adaptive_grasp.plan(
        np.array([p.object_pos(obj_key)[0], p.object_pos(obj_key)[1],
                  obj_top(meta, obj_key) - grasping.GRIP_BELOW_RIM]),
        _wrist, _yawrot, surf_a["top"]))

    # --- 1. approach surface A ------------------------------------------
    stage("approach_A")
    obj_xy = p.object_pos(obj_key)[:2]
    if not p.approach(obj_xy, surf_a["xy"], REACH_XY_TOL, attempts=5, side=side,
                      circumradius=surf_a.get("circumradius", 0.0),
                      half_extent=surf_a.get("half_extent")):
        raise EpisodeFailed("approach_A", p.aborted or "walk failed")
    s = p.obs()
    d = float(np.linalg.norm(p.object_pos(obj_key, s)[:2] - p.pelvis_xy(s)))
    require("approach_A", d < REACH_XY_TOL,
            f"orange {d:.2f} m from pelvis, need < {REACH_XY_TOL}")

    # --- 2. grasp --------------------------------------------------------
    stage("grasp")
    obj = p.object_pos(obj_key)
    z0 = float(obj[2])
    grip_z = (surf_a["top"] + grasping.FINGER_DROP + grasping.CLEARANCE_MARGIN
              + grasping.GRIP_LIFT)
    hold_side = "left" if side == "right" else "right"

    # Stand-pose selection from the reach map: put the GRIP POINT deep
    # inside the chosen arm's graspable region (margin-maximised), with the
    # facing within 40 deg of the object's bearing and the knees clear of
    # the table edge. Replaces aim()'s shoulder-distance scoring — the map
    # already encodes handedness, self-collision, grasp-quality orientation
    # and the delivered torso envelope, so the side bearing and the stand
    # depth fall out of one lookup instead of three tuned thresholds.
    grip_pre = np.array([p.object_pos(obj_key)[0], p.object_pos(obj_key)[1],
                         obj_top(meta, obj_key) - grasping.GRIP_BELOW_RIM])
    _half = np.asarray(surf_a.get("half_extent", [0.6, 0.6])[:2], dtype=float)
    _cxy = np.asarray(surf_a["xy"], dtype=float)

    def edge_dist(xy):
        dxy = np.abs(np.asarray(xy) - _cxy) - _half
        out = np.maximum(dxy, 0.0)
        return float(np.hypot(out[0], out[1])) if (dxy > 0).any() else \
            -float(max(dxy))

    # Stage-2 navigation: careful facing-the-object advance from the safe
    # ring, reactive back-off on any disturbance, stop at grasping depth or
    # the hip's table-edge limit, then the minimal side-turn that puts the
    # object inside the reach map (dead-centre is kinematically infeasible
    # past the hip-contact line, so this is "as centred as possible").
    rm = reach_map.ReachMap()
    z_pf_std = float(grip_pre[2] - STANDING_PELVIS_Z)
    ok2, side, why2 = stream_reach.final_approach(
        p, obj_key, rm, edge_dist, z_pf_std, side_pref=side)
    require("grasp", ok2, why2)
    hold_side = "left" if side == "right" else "right"

    # Plan first, then crouch only as far as THIS plan needs. The order used to
    # be the other way round because the grip height was a constant of the
    # table, so the squat could be computed without the plan. With the tilted
    # palm it cannot: the plan grips near the object's rim rather than its
    # middle, and the wrist rides 7.5 cm above the grip point, so how far the
    # body has to come down is a property of the pose, not of the furniture.
    # The grip point is the only fixed thing: the object's own axis, just under
    # its rim. Everything about the hand's orientation is decided from where
    # the hand actually IS, so the approach comes out as a straight line rather
    # than a pose the arm has to contort its way into.
    st = p.obs()
    grip = np.array([p.object_pos(obj_key, st)[0], p.object_pos(obj_key, st)[1],
                     obj_top(meta, obj_key) - grasping.GRIP_BELOW_RIM])
    # Plan from a VIRTUAL hand start, not the actual palm. The adaptive
    # planner points its travel from the given start to the grip, which made
    # sense when the planner-mode reach had to fly straight from wherever the
    # hand was — but the streamed reach goes over the top and descends, and a
    # robot that stands close leaves the real palm nearly ON the grip, which
    # degenerates the plan (1 cm of travel, jaw tilted to its 45 deg limit).
    # A start 25 cm back along the robot->can bearing and a little above
    # yields the same sane, near-horizontal-jaw pose every time.
    _d = grip[:2] - p.pelvis_xy(st)
    _d = _d / max(np.linalg.norm(_d), 1e-6)
    virtual_start = grip - 0.25 * np.array([_d[0], _d[1], 0.0]) + [0, 0, 0.08]
    obj_box = meta["objects"][obj_key]["box"]
    plan = adaptive_grasp.plan(grip, virtual_start,
                               Primitives._rot(st["palms"][side]["quat"]),
                               surf_a["top"],
                               box=obj_box,
                               obj_rot=Primitives._rot(
                                   st["objects"][obj_key]["quat"]))
    p.log(adaptive_grasp.describe(plan))
    require("grasp", plan.ok, plan.why)
    show_ghost(plan)

    def replan(wrist_now, palm_rot):
        """The live candidate: the shortest line from the REAL wrist to the
        LIVE can.

        Called from inside the streamed servo at ~5 Hz until it freezes. The
        grip point is recomputed from the can's CURRENT position — the close
        stands lean the hip on the table edge and shove it a few cm, and a
        plan anchored on where the can USED to be turns that nudge into a
        guaranteed miss. The hand follows the can the way a person's would.
        """
        st2 = p.obs()
        c = p.object_pos(obj_key, st2)
        grip_live = np.array([c[0], c[1],
                              obj_top(meta, obj_key) - grasping.GRIP_BELOW_RIM])
        return adaptive_grasp.plan(
            grip_live, wrist_now, palm_rot, surf_a["top"], box=obj_box,
            obj_rot=Primitives._rot(st2["objects"][obj_key]["quat"]))

    # Streamed motion from here to the lift: full joint references on the
    # `pose` topic — the channel Pico teleop actually drives — with IK on the
    # scene model. Crouch and lean are part of the reference now
    # (stream_reach squats the legs and leans the waist), so the planner-era
    # crouch machinery is bypassed entirely.

    # The map refinement above already placed the stand; the planner-era
    # walk-in band (grip x 0.24-0.34) is tuned for a farther stand and would
    # walk the robot back OUT of the map core, so it is skipped.
    ok, why = stream_reach.grasp_streamed(p, obj_key, grip, plan, side=side,
                                          surface_top=surf_a["top"],
                                          replan=replan, ghost=show_ghost,
                                          nudge=True,
                                          pin=(reset.pin_arms if reset
                                               else None))
    require("grasp", ok, why)

    s = p.obs()
    hold_ref = palm_frame_offset(s, obj_key, side)
    # Back up before walking: a crouched humanoid carrying something is not a
    # gait the controller was trained on.
    p.stand_up(hold_side)

    if not FULL_TASK:
        # Grasp-only episode: walk to the table, pick the can up, hold it.
        # The transport/place/retreat stages exist and still run with
        # FULL_TASK=1; while the grasp is the thing being iterated they only
        # double the episode time and hide the grasp statistics behind
        # unrelated place scatter.
        stage("hold")
        p.carry_pose(side)
        p.cmd.hold(2.0, mode=LocomotionMode.IDLE)
        s = p.obs()
        drift = float(np.linalg.norm(
            palm_frame_offset(s, obj_key, side) - hold_ref))
        require("hold", drift < PALM_DRIFT_MAX,
                f"orange slipped {drift * 100:.1f} cm in the palm frame "
                f"during the hold")
        require("hold", float(p.object_pos(obj_key, s)[2]) > surf_a["top"] + 0.03,
                "orange is back on the table after the hold")
        result["ok"] = True
        return result

    # --- 3. transport ----------------------------------------------------
    stage("transport")
    p.carry_pose(side)
    walked = p.approach(p.object_pos(goal_key)[:2], surf_b["xy"],
                        PLACE_XY_REACH, attempts=5, side=side,
                        circumradius=surf_b.get("circumradius", 0.0),
                        half_extent=surf_b.get("half_extent"), hold=side)
    s = p.obs()
    drift = float(np.linalg.norm(palm_frame_offset(s, obj_key, side) - hold_ref))
    require("transport", walked, p.aborted or "walk to B failed")
    require("transport", drift < PALM_DRIFT_MAX,
            f"orange slipped {drift * 100:.1f} cm in the palm frame (dropped)")

    # --- 4. approach surface B -------------------------------------------
    stage("approach_B")
    s = p.obs()
    d = float(np.linalg.norm(p.object_pos(goal_key, s)[:2] - p.pelvis_xy(s)))
    require("approach_B", d < PLACE_XY_REACH,
            f"plate {d:.2f} m from pelvis, need < {PLACE_XY_REACH}")

    # --- 5. place --------------------------------------------------------
    stage("place")
    plate = p.object_pos(goal_key)
    # Set the can DOWN, not drop it: released from PREGRASP_ABOVE (14 cm) a
    # cylinder tips on landing and rolls — clean off the 0.37 m counter and
    # 1.7 m across the floor, on video. Target the can's centre a couple of
    # centimetres above its seated height on the plate.
    # +6 cm, not +2: releasing at seat height leaves the can still between
    # the fingers, and the withdrawal drags it 20+ cm through the 1.3 cm of
    # total clearance an OPEN jaw has around it. From 6 cm the upright can
    # falls clear the instant the fingers open and lands standing (verified:
    # a dragged-then-dropped can still landed upright on the counter).
    drop_at = plate + [0, 0, meta["objects"][goal_key]["height"] / 2
                       + meta["objects"][obj_key]["height"] / 2 + 0.035]
    ok, why = stream_reach.place_streamed(p, obj_key, drop_at, side=side)
    require("place", ok, why)
    # Let it settle before judging where it landed.
    p.cmd.hold(1.5, mode=LocomotionMode.IDLE)
    s = p.obs()
    obj_now = p.object_pos(obj_key, s)
    plate_now = p.object_pos(goal_key, s)
    dxy = float(np.linalg.norm(obj_now[:2] - plate_now[:2]))
    dz = float(obj_now[2] - plate_now[2])
    require("place", dxy < PLACE_XY_TOL and abs(dz) < PLACE_Z_TOL + LIFT_HEIGHT,
            f"orange landed {dxy * 100:.1f} cm from the plate centre, dz {dz:.3f}")
    require("place", side not in s["objects"][obj_key]["touching"],
            "hand is still touching the orange after release")

    # --- 6. retreat ------------------------------------------------------
    stage("retreat")
    # The streamed retreat already left the hand high and back; the planner-
    # mode "raise" that used to sit here went through the 25 cm-warp wrist
    # channel and could swing the arm anywhere — with it, clean 0.5 cm places
    # were still swept off during retreat. Just tuck, and log what (if
    # anything) touches the can while doing it.
    # No tuck_arms here: the arms were tucked exactly, in streamed mode,
    # before place_streamed handed control back — the planner-channel tuck
    # was what kept hitting the can (RETREAT-TUCK forensics). Just verify.
    s = p.obs()
    _can0 = p.object_pos(obj_key, s)
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
    s = p.obs()
    _moved = float(np.linalg.norm(p.object_pos(obj_key, s) - _can0))
    if _moved > 0.02:
        print(f"    RETREAT idle moved the can {_moved * 100:.1f} cm "
              f"(touch {s['objects'][obj_key].get('touch_links')})")
    back = p.pelvis_xy() - np.array([RETREAT_MIN + 0.2, 0.0])
    p.walk_to(back, stop_dist=0.15, timeout=20.0, face_target=False)
    s = p.obs()
    moved = float(np.linalg.norm(p.pelvis_xy(s) - plate_now[:2]))
    still_there = float(np.linalg.norm(
        p.object_pos(obj_key, s)[:2] - p.object_pos(goal_key, s)[:2]))
    require("retreat", still_there < PLACE_XY_TOL,
            f"orange left the plate during retreat ({still_there * 100:.1f} cm)")
    # XY alone let knocked-over cans through: tipped during the walk-away but
    # still inside the plate radius -> "OK" with a lying can in the last
    # frames (batch recbatch1 eps 2/5/13). Re-check uprightness at the very
    # end too.
    w, x, y, z = p.obs()["objects"][obj_key]["quat"]
    up_z = 1 - 2 * (x * x + y * y)
    require("retreat", up_z > 0.8,
            f"orange tipped over during retreat (up.z {up_z:.2f})")
    result["retreat_distance"] = moved

    result["ok"] = True
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--out", default="/workspace/code/locomanipulation/outputs/logs")
    args = ap.parse_args()

    meta = json.load(open(SCENE_META))
    p = Primitives(meta=meta)
    reset = EpisodeControlClient()
    record = not args.no_record
    t0 = time.time()
    try:
        result = run_episode(p, meta, side=args.side, record=record, reset=reset)
        print(f"\nEPISODE OK in {time.time() - t0:.1f}s")
    except EpisodeFailed as e:
        result = {"ok": False, "stage": e.stage, "why": e.why}
        print(f"\nEPISODE FAILED {e}")
    finally:
        if record:
            # 'c' again -> NEED_TO_SAVE; 'x' discards. A failed episode must not
            # reach the dataset.
            p.cmd.key("c" if result.get("ok") else "x")
            time.sleep(2.0)
        p.cmd.pause_control()
        reset.end_video()
        reset.close()
        p.close()

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"episode_{int(t0)}.json")
    result["duration_s"] = time.time() - t0
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {path}")
    sys.exit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
