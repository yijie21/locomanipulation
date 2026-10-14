"""Limits for the crate RGB expert; independent of the legacy wall/carry experts."""

import math
from dataclasses import dataclass

RANGE_FIELDS = (
    "start_yaw_range_deg",
    "start_lateral_range",
    "object_setback_range",
    "object_yaw_range_deg",
    "object_along_range",
)
EDGE_SIDES = ("xm", "xp", "ym", "yp")


@dataclass(frozen=True)
class CrateVisualConfig:
    pose_source: str = "markers"  # markers: RGB fitting; assisted: visible target with simulation pose/feedback
    waist_pitch_max: float = 0.40  # radians; additionally bounded by the robot model
    palm_pitch_deg: float = 60.0  # downward grasp orientation relative to the observed basket
    start_edge_min: float = 1.2  # initial pelvis distance from the near table edge, metres
    start_edge_max: float = 1.5  # upper end of initial table-edge distance sampling
    start_pose: str = "sampled"  # sampled: start_edge_min..max before the near edge; scene: the scene's robot_start, advanced by start_advance (the logistics-room box line starts where the box-rim reference round dy0.225_r00 did)
    start_advance: float = 0.0  # metres along robot_start heading (negative = back); start_pose scene only
    start_edge_tolerance: float = 0.0  # sampled start: metres the measured initial edge distance may fall outside start_edge_min..max (the reset lands a few cm off) before the round is refused as invalid_initial_distance
    # Per-episode layout ranges, uniform (low, high); (0, 0) draws nothing, so a config without them samples as before.
    # Built for the scene capability sweep (2026-09-26): the robot's start and the target's placement relative to the edge.
    start_yaw_range_deg: tuple[float, float] = (
        0.0,
        0.0,
    )  # robot start heading off the walk-in axis, degrees (+ = turned left); the walk still commands the walk-in axis
    start_lateral_range: tuple[float, float] = (
        0.0,
        0.0,
    )  # robot start offset across the walk-in axis, metres (+ = left of it); the guard walk goes straight on and does not correct it
    object_setback_range: tuple[float, float] = (
        0.0,
        0.0,
    )  # target moved away from the near edge along the walk-in axis, metres (>= 0), on top of position_jitter
    object_yaw_range_deg: tuple[float, float] = (
        0.0,
        0.0,
    )  # target yaw offset, degrees (+ = counter-clockwise from above), on top of yaw_jitter_deg
    # The four-edge line (2026-09-28): each round walks in toward one side of the target surface. Needs a scene whose
    # surface lists its `edges` (locomani.tasks.crate.warehouse_box_scene) and start_pose sampled.
    start_edges: tuple[
        str, ...
    ] = ()  # sides to draw from, uniformly (xm/xp/ym/yp: the -x/+x/-y/+y side); the target is carried to the drawn edge, its offset from its own edge and its yaw turned with the walk-in. () = the scene's robot_start walk-in, nothing drawn
    start_clearance: float = 0.25  # start_edges: the start distance is capped at the edge's free floor (`clear_m`, to the room's racks and walls, which do not collide) minus this, m
    object_along_range: tuple[float, float] = (
        0.0,
        0.0,
    )  # target moved along the near edge, as a fraction of the room the edge leaves it (edge half length - the box's half width; -1..1, + = left of the walk-in); the robot start moves with it, so start_lateral_range stays relative to the target
    walk_speed: float = 0.35  # maximum SONIC approach speed command, metres/second
    walk_speed_min: float = (
        0.35  # smallest approach command; the deploy does not step below ~0.3, so a lower floor is a dead command
    )
    walk_brake_margin: float = 0.07  # stop commanding steps before the final basket distance
    walk_timeout: float = 30.0  # maximum walking and stance-settling duration
    settle_seconds: float = 1.0  # continuously quiet visual/IMU interval before reaching
    stand_lateral_tolerance: float = 0.04  # metres; final body IK still owns low-reach admission
    stand_lateral_gate: str = "walk"  # walk: keep walking until |lateral| is inside the tolerance; turn: stop on depth, then turn in place to face the target (ledger crate.md 2026-09-22: walking on for lateral is what overshoots depth)
    approach_stop: str = "depth"  # depth: stop commanding steps at stand_off + walk_brake_margin; guard: the target surface's near edge is a line nothing below 0.37 m may cross -- brake on the foot landing where one more stride would (primitives/foot_watch.py; walk scene 50/50, docs/walk.md). Guard stances are accepted as they land (no depth re-walks). Guard also runs the walk scene's sequence: walk with the planner's own arms (no ApproachPreparation), ready the arms in place after the brake, no reach admission; the line is the nearer of the surface edge and a `box` target's front face
    guard_extra: float = (
        0.20  # guard: worst reach of the frontmost low-body point past the front planted ankle after IDLE, m
    )
    guard_margin: float = 0.03  # guard: clearance kept before the line, m
    guard_stride0: float = 0.14  # guard: floor on the predicted next stride, m
    guard_stride_mode: str = "floor"  # guard: next-stride prediction (primitives/foot_watch.GuardBrake): floor = max(last stride, guard_stride0); alternate = 0.08 after a stride of 0.15 or more, else 0.23 -- the 0.25 gait alternates a ~0.21 step and a <= 0.07 one, and floor over-predicts the step after a long one by 0.12-0.17 m (TODO 2.4.10.23.3)
    box_grasp_below_rim: float = 0.01  # grasp point depth under the rim for scenes whose target carries a `box` (the cardboard box); the crate's bar uses grasp_region.GRASP_BELOW_RIM_M (0.025) instead. 0.025 put the Dex1 palm on the 1 cm wall top (2026-09-23); 0.03 pushed the box 6/10 (2026-09-24), 0.02 lost to 0.01 15/30 vs 10/30 (2026-09-27); default 0.01 since 2026-10-01, which every warehouse file already set (TODO 2.4.9)
    retreat_distance: float = 0.0  # after a completed hold: walk this far backward holding the load, m; 0 = end at the hold. Success then requires the load still held after the retreat
    retreat_speed: float = 0.35  # SLOW_WALK speed command for the retreat
    retreat_timeout: float = 10.0  # simulated seconds allowed for the retreat
    walk_steer: str = "target"  # target: steer the movement at the planned stance (depth and lateral); depth: walk straight ahead on the body heading and stop on depth alone, the lateral residual left to the reach (2026-09-23: target steering turned the last 0.1 s of the walk up to 50 deg sideways and scattered the stance)
    stand_depth_tolerance: float = 0.04  # metres past stand_off a quiet stance may rest before re-walking
    stand_depth_retries: int = 2  # re-walks allowed when the quiet stance is farther than the tolerance (approach_stop=depth) or than guard_rewalk_depth (guard)
    guard_brake: str = "landing"  # approach_stop=guard: landing = brake at the foot landing after which one more stride would carry the body past the line (GuardBrake); phase = brake on any tick once the pelvis plus guard_phase_band(gait phase) reaches the line minus guard_margin -- needs a steady gait, measured at 0.55 with >= 1.0 m run-up (TODO 3.14.1)
    guard_phase_band: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )  # guard_brake=phase: c0, a, b of the settled frontmost low-body point ahead of the pelvis at the brake, c0 + a cos(2 pi phase) + b sin(2 pi phase), m; phase = time since the last foot landing / last step period. Fit from diag_brake's walk-scene trials (docs/walk.md)
    guard_rewalk_depth: float = 0.0  # approach_stop=guard: a quiet stance farther than this (pelvis to box centre along the walk-in, m) backs off guard_rewalk_back and walks in again, up to stand_depth_retries times; 0 = off. The guard walk cannot be steered closer, but each walk-in lands at an independent depth, and the warehouse yield falls from 0.41 at < 0.44 m to 0.20 at 0.48-0.52 and 0.03 beyond (1013 rounds, 2026-09-25; TODO 2.4.10.23.2)
    guard_rewalk_back: float = (
        0.4  # guard re-walk: how far to back off before walking in again, m (by the target's measured depth)
    )
    lift_seconds: float = 4.0  # lift ramp duration; assisted standing overlaps after the measured probe
    stand_seconds: float = 5.0  # attached recovery duration after the measured probe, seconds
    upright_tolerance_deg: float = 12.0  # measured torso lean allowed at successful completion
    stand_off: float = 0.36  # final pelvis distance from the visible basket centre, metres
    position_jitter: float = 0.02  # visible basket x/y randomisation at reset, metres
    yaw_jitter_deg: float = 5.0  # visible basket yaw randomisation at reset
    pregrasp_above: float = 0.09  # metres above each grasp point
    reach_path: str = "direct"  # assisted CoordinatedReach: direct = one straight blend from the ready hands to the grasp points; staged = first to a pregrasp point pregrasp_above over each grasp point and pregrasp_back toward the body, then straight down (2026-09-23: direct swept the hands through the warehouse box, 8 of 10 displaced-box rounds had a hand on it)
    reach_route: str = "straight"  # staged reach, how the hands get to the pregrasp points: straight = a straight Cartesian blend from where they are at the handoff (after the ready pose); planned = along one collision-free joint path for the waist and both arms from the CuRobo route service (primitives/curobo_service.py, must be running; the runner sets CoordinatedReach.route_planner), whose goal is this reach's own whole-body solve at the pregrasp points, tracked by the same closed loop -- the runner then skips the ready pose and the hands rise from their hang in one motion (user, 2026-10-14: natural motion; the straight blend from the hang sweeps the hands through the box in 9 of 10 rounds)
    reach_route_speed: float = 0.8  # reach_route planned: peak speed of the most moved joint along the planned path, rad/s (smoothstep timing, primitives/arm_route.retime_path)
    reach_route_clearance: float = 0.015  # reach_route planned: metres every box obstacle grows on each side before planning (tasks/crate/reach_route.py). The hands track the path with a few centimetres of error; at 0 the planned hands came 1.1 cm (median) from the box and the Dex1 fingers brushed its near rim in 11 of 49 rounds, at 0.015 the planned clearance is 3.0 cm with 49 of 50 planned, at 0.025 4.2 cm with 48 of 50 (diagnostic pool, 2026-10-14)
    descend_speed: float = 0.0  # staged reach: Cartesian speed limit of the final straight-down segment, m/s; 0 = approach_speed. The descent undershot the grasp point by 1-3 cm and put the palm on the box rim (2026-09-23)
    descend_z_guard: bool = False  # staged reach, descent only: once a measured grip point is at or below its target height, its z position feedback may no longer grow downward and backs off by the undershoot. The feedback had to cover a ~8 cm model offset (cap 0.15) and then drove the palm 1-3 cm past the grasp point into the box rim (2026-09-23)
    pregrasp_back: float = 0.04  # staged reach: metres the pregrasp point sits toward the body from the grasp point
    descend_align_tolerance: float = 0.0  # staged reach: hold at the pregrasp point until both measured grip points are within this of the grasp line across the wall (assisted: simulation truth), then descend; 0 = descend at once. The across-wall error was 2-6 cm at the start of the descend and jammed a pad on the wall face (2026-09-24)
    descend_align_timeout: float = 3.0  # staged reach: seconds the alignment hold may take before descending anyway
    reach_feedforward_forward: float = 0.0  # metres added to every reach/align/descend hand command along the body-to-object axis, outside the capped tracking feedback. The controller delivered the hands 2-5 cm short of the reference in the warehouse descend (2026-09-24); negative pulls the command back
    reach_feedforward_up: float = 0.0  # metres added to the same commands in world z. The hands arrived 2-9 cm above the reference (waist pitch delivered at ~2/3 of the command); a negative value commands lower from the first tick instead of waiting for the feedback to wind up
    descend_retries: int = 0  # staged reach: after a descend stall, back up to the pregrasp point (the align phase), re-centre and descend again, this many times; 0 = the stall ends the round. 278/835 sweep rounds (2026-09-27) stalled 1-5 cm off the grasp points with no second attempt
    reseat_retries: int = 0  # after an empty close, a failed probe lift or a lift-phase loss: open the jaws where they are (reopen phase, close_seconds), back up to the pregrasp point, re-centre and descend again, this many times; 0 = those end the round
    attachment_gain: float = 0.0  # stand/hold: fraction of each hand's measured VERTICAL drift from its closed hand-box offset (the pinch creeping up the wall) subtracted from that hand's target, low-passed 0.3 s and capped by attachment_cap; 0 = off. Lost grips in the sweep (2026-09-27) were the hands rising 1.4-3.5 cm relative to the box at 0.2-0.4 cm/s over the 6-8 s under load (pads already hardened); this re-seats the pinch instead of watching it leave the rim
    attachment_cap: float = 0.03  # largest vertical attachment correction per hand, m
    seat_depth: float = 0.0  # seat check at the end of the close, before the lift: both grip points at least this far under the rim top (m) and the two hands' along-wall and height offsets within seat_skew, else a reseat retry (visual.reseat_retries) is spent -- reopen, back up, re-centre, descend again; when none is left the lift goes ahead as before. 0 = no check. 59/835 sweep rounds tilted at lift-off from uneven seating, and every lost grip was the pinch leaving the rim (2026-09-27)
    seat_skew: float = (
        0.01  # seat check: largest allowed difference between the two grip points along the wall and in height, m
    )
    face_after_brake: str = "none"  # approach_stop=guard: turn on the spot after the brake, before the arms are readied -- none: keep the walk-in heading; centre: face the target's centre (a lateral start offset becomes a heading change, and heading is free while lateral costs, sweep 2026-09-26); box_yaw: square the body to the gripped walls (the grasp line's normal nearest to straight ahead; a few degrees of box yaw halved the success rate). turn_to in SLOW_WALK, then the new heading is held through the arm preparation, the reach and the retreat
    face_tolerance_deg: float = 2.0  # face_after_brake: turn_to tolerance (v1.1 reaches +-3-20 deg within ~3 deg at a 2 deg tolerance, 2026-09-23)
    face_timeout: float = (
        6.0  # face_after_brake: seconds allowed for the turn; a timed-out turn holds the heading it reached
    )
    ready_after_brake: str = "full"  # approach_stop=guard, arm preparation after the brake: full = the episode-start sequence (rest arms, planner re-entry, 1.5 s + 0.4 s + 1.5 s holds, heading re-anchor, 3 s hold, then the READY blend), 13.5 s per round; quick = the robot is already standing, so only the heading re-anchor the POSE stream needs (+1 s to let its twitch die), then the same READY blend and quiet admission, ~7 s. The 6 s of holds were written for a teleported reset, not for a stance the walk just delivered (2026-09-27); compact = quick without its fixed waits (2026-09-28: the hands stood still 3.9 s before the blend and 2.4 s after it, 8.7 s brake-to-reach): the planner arm takes over the walking arms at once instead of a 0.6 s seed hold, no 0.3 s post-placement sleep, and the blend/hold/re-anchor durations are the knobs below. The bimanual collection default since 2026-09-28 is compact with blend 1.5, hold 0, re-anchor 0.3 (box_platform_default_20260928); the dataclass stays full because 53 historical guard experiment files leave this unset
    ready_blend_seconds: float = 2.5  # arm preparation: seconds of the smooth blend from the walking arms to the READY pose (125 frames at 50 Hz historically)
    ready_hold_seconds: float = 1.2  # arm preparation: seconds the READY pose is held before the POSE stream takes over
    reanchor_hold_seconds: float = (
        1.0  # ready_after_brake=quick: IDLE seconds after the heading re-anchor, for its twitch to die
    )
    reanchor_after_brake: bool = True  # ready_after_brake quick/compact: re-anchor the heading after the brake (a POSE toggle, ~1.6 s + reanchor_hold_seconds). The episode start already re-anchors (stand_up) and the guard walk never leaves the planner, so whether the second one is needed is what false tests (2026-09-28)
    lateral_hold_common_mode: bool = False  # with lateral_hold_below_rim: the hold removes only the DIFFERENTIAL across-wall change (a hand's minus the pair's mean), so the pair may still shift together. In 278 stalled sweep rounds (2026-09-27) the pair had drifted 3-5 cm to one side during the descend (0.5 cm at its start) with the hold freezing the correction; a common shift keeps the wall between each hand's pads (jaw 9.5 cm, wall 1 cm) and moves the approaching pads AWAY from the faces
    lateral_hold_below_rim: bool = False  # staged reach, descent only: once a measured grip point is within 1 cm of the rim top or below it, its across-wall feedback stops changing. A pad against a wall face is blocked; the correction otherwise winds up to tracking_cap and shoves the wall until the box tips (2026-09-24)
    ready_pose: str = "forward"  # arms readied in place after a guard brake (prepare_stationary_ready): forward = the historical pose, tool points 0.38 m ahead at pelvis height (fingertips 0.42 m: on the warehouse box's near wall at stances under 0.51 m, 2026-09-24); lifted = tool points at ready_tool with the palm pitched ready_pitch_deg, solved by IK from the walking arms with the waist locked -- the forearms fold up, so it needs elbow_min_deg below 0
    ready_tool: tuple[float, float, float] = (
        0.16,
        0.16,
        0.05,
    )  # lifted ready: tool point forward / lateral (mirrored) / up from the pelvis, metres; fingertips stay within 0.20 m of the pelvis through the blend from the walking arms
    ready_pitch_deg: float = 30.0  # lifted ready: palm pitch below level; the reach slerps it to palm_pitch_deg
    elbow_min_deg: float = 5.0  # elbow floor in both reach solvers (BimanualIK, WholeBodyIK); G1 elbow 0 = forearm level, negative folds it up (range -60). 5 since 2026-09-12; the lifted ready pose needs -60 (its solve misses by 5-10 cm at 5)
    reach_arm_posture_weight: float = 0.0  # assisted CoordinatedReach: posture-prior weight of the 14 arm joints toward the admission solve's pose (the other 15 keep the solver's 0.04); 0 = the solver's 0.04 on all. The commanded elbow and wrist pitch trade off along the null space through the descend, sd 15 deg at correlation -0.9 (TODO 2.4.10.20; the continuity damping did not touch it, 2026-10-01)
    reach_continuity_weight: float = 0.0  # assisted CoordinatedReach: whole-body IK residual weight x (x - previous solution) / dt, a damping prior on every base and joint coordinate; 0 = off. Each tick re-solves from a seed carried forward at the last velocity in 6 iterations, and the null space (elbow vs wrist pitch, which leave the hand where it is) swapped anti-phase: elbow reference sd 9-15 deg, correlation -0.95..-0.98, while the hand target moved 0.2 cm per tick (user saw the arms shake, 2026-09-24; TODO 2.4.10.20)
    reach_rotation_weight_transit: float = 0.2  # assisted CoordinatedReach: the hand orientation weight while the hands travel to the pregrasp point (phase reach); -1 = reach_rotation_weight. The user's question (2026-09-24): the wrist cameras need the box only in descend, so the transit may give up palm attitude for position (TODO 2.4.10.13). Single-edge warehouse, 20 rounds x 2 seeds: 0.2 19/40 vs -1 10/40 (discordant 13 : 4, p 0.049); the default since 2026-10-01 (user). Not measured on the crate basket or box_platform
    reach_rotation_weight: float = 0.6  # assisted CoordinatedReach: whole-body IK weight on hand orientation (rad) against hand position (10 per metre); 0.6 trades ~10 deg for 1 cm, and at long reach gave up 10-20 deg of palm pitch (2026-09-23)
    orientation_gain: float = 0.0  # assisted CoordinatedReach: fraction of the filtered predicted-vs-measured palm rotation added to the commanded attitude; 0 = none. SONIC tracked the palm pitch 8-14 deg flat at long reach (2026-09-23)
    orientation_cap_deg: float = 20.0  # largest palm-attitude correction the orientation feedback may apply, degrees
    close_target: float = -0.023  # closed-hand finger setpoint sent to both Dex1 fingers, m (joint range -0.023..0.03; 0 gap at -0.023). The sim's finger force is kp 1.5 x gear 250 x (setpoint - q): a 1 cm wall stops the jaw at -0.0177, so -0.023 gives 2 N per finger (9.5 N of friction for four pads, under the 1.2 kg box's 11.8 N); -0.06 gives 16 N, under the 20 N motor clamp. The deploy only clips finger targets to the Dex3 limits (-1.05/-0.724 rad) (2026-09-24)
    stand_forward_min: float = 0.0  # stand/hold: smallest forward distance (m, along the body-to-object axis from the CURRENT pelvis, enforced every tick -- the pelvis itself advances 11-17 cm during the stand) of the stand hand targets' midpoint; 0 = the historical targets, which pull the hands back to the ready pose's position (0.16 m ahead for the lifted ready). With the 0.32 m deep warehouse box that rams its near wall into the torso at the end of the stand: box wall 6-9 cm from the pelvis, tilt to 14 deg and one jaw slips off in 15/15 lifted rounds (2026-09-24). The box needs its half depth plus clearance, ~0.35
    lift_pitch_follow_gain: float = 0.0  # lift/stand/hold: rotate the commanded palm attitudes about the hand-hand axis by -gain x the box's measured pitch since closing (assisted truth; low-pass 0.2 s, capped by lift_pitch_follow_cap_deg); 0 = off. The rim pinch is a hinge and the box followed the palms' 10-25 deg attitude drift to the 15 deg tilt gate in 5/10 (2026-09-24); the crate line's bar grasp needed the same follow at 1.5 (bimanual_wall, 2026-09-13)
    lift_pitch_follow_cap_deg: float = 25.0  # largest palm rotation the pitch follow may command, degrees
    close_trigger: str = "gate"  # when the coordinated reach closes: gate = both measured grip points inside RimRegion (across/along/up box + 25 deg palm alignment + insertion, the crate bar's rule); contact = the jaw is physically around the wall wherever the hands landed: |across| <= contact_across (the open Dex1 jaw is 9.5 cm wide) and the grip point at least contact_below_rim under the rim top, OR that hand touching the object (privileged contact) with the grip point no more than 1.5 cm above the rim -- no palm-attitude gate. Adapts the grasp to what SONIC delivered instead of stalling 2-5 cm away (8/20 reach stalls, 2026-09-24)
    contact_across: float = (
        0.045  # contact trigger: largest across-wall error, m, at which the wall is still between the open pads
    )
    contact_along: float = 0.13  # contact trigger: largest along-wall offset of a grip point from the wall's midpoint, m, at which the pads are still on the wall (warehouse box wall 0.32 m long, pad 0.03: 0.13). The crate-era RimRegion window is +-0.09 and the hands slide 7-10 cm toward the robot on this wall (2026-09-24)
    contact_overdrive: float = 0.0  # close_trigger=contact: the staged descent aims this far BELOW the grasp point, m, so the hands keep going until the trigger fires (pads under the rim, or the pad/palm touching the wall) instead of parking 1-4 cm above the rim where the delivery stopped short (7/20 stalls, 2026-09-24). 0 = descend to the grasp point as before
    contact_below_rim: float = 0.005  # contact trigger: the grip point must be at least this far under the rim top, m, unless the hand already touches the object
    capture_across: float = 0.0  # closing gate: allowed across-wall error of each measured grip point, m; 0 = RimRegion's 0.018 (set for the crate's 16 mm bar). The Dex1 jaw opens 9.5 cm, so a 1 cm box wall can sit +-4.2 cm off the jaw centre and still be closed on; the warehouse descend arrived 2-5.6 cm off across in 28/39 stalled rounds (2026-09-24)
    approach_speed: float = 0.055  # maximum Cartesian reference speed, metres/second
    joint_speed: float = 0.7  # maximum reference joint speed, radians/second
    joint_acceleration: float = 2.0  # maximum reference velocity change, radians/second squared
    palm_rotation_speed: float = 0.6  # maximum palm reference rotation speed, radians/second
    palm_align_tolerance_deg: float = 10.0  # both measured wrists must align before closing
    tracking_gain: float = 0.5  # proportional correction from observed tool-point error
    tracking_cap: float = 0.025  # maximum Cartesian tracking correction, metres
    align_tolerance: float = 0.018  # both measured grasp points must align within this distance
    close_seconds: float = 1.6  # coordinated gripper closing duration
    probe_height: float = 0.025  # initial lift used to observe bilateral attachment
    lift_height: float = 0.10  # total desired rise after the probe
    hold_seconds: float = 2.0  # stable hold before ending the attempt
    hold_position_tolerance: float = 0.025  # maximum load-position radius over the final hold window, metres
    hold_frame: str = "world"  # frame of the hold-window load position: world (the crate line's rule: the load must not move in the room) or pelvis (the load must not move IN THE HANDS; the body's own IDLE drift under load, 2.5-8 cm back and up to 5 cm sideways in 4 s on the warehouse box, is left to the retreat). Success = lifted, torso upright, retreat_distance walked with the load still held (user's definition, 2026-09-24)
    timeout_seconds: float = 35.0  # bounded expert attempt
    vision_timeout: float = 0.7  # stop advancing when the last valid RGB estimate is older
    reprojection_px: float = 4.0  # maximum RMS marker reprojection residual
    max_tilt_deg: float = 15.0  # observed basket tilt limit during lifting
    collision_guard: str = "off"  # assisted CoordinatedReach, align and descend: off = the intended grip targets go to the whole-body solve as they are; mink = a kinematic shadow of the waist and arms (collision_guard.py, needs mink + qpsolvers[daqp]) first tracks them under a collision-avoidance limit against every non-robot collision geom, and its grip points and palm attitudes are what the solve gets. The descend's intended path put a finger or the palm into a carton wall in 33/119 warehouse_mars_50 rounds (2026-10-08, TODO 20)
    collision_guard_clearance: float = 0.003  # collision_guard mink: smallest distance, m, between the hand's bounding boxes (each 0.4-11 mm outside the real shape) and an obstacle

    def __post_init__(self):
        if self.pose_source not in ("markers", "assisted"):
            raise ValueError("visual.pose_source must be markers or assisted")
        if self.stand_lateral_gate not in ("walk", "turn"):
            raise ValueError("visual.stand_lateral_gate must be walk or turn")
        if self.walk_steer not in ("target", "depth"):
            raise ValueError("visual.walk_steer must be target or depth")
        if self.approach_stop not in ("depth", "guard"):
            raise ValueError("visual.approach_stop must be depth or guard")
        if self.reach_path not in ("direct", "staged"):
            raise ValueError("visual.reach_path must be direct or staged")
        if self.reach_route not in ("straight", "planned"):
            raise ValueError("visual.reach_route must be straight or planned")
        if self.reach_route == "planned" and self.reach_path != "staged":
            raise ValueError("visual.reach_route planned plans to the pregrasp points; it needs reach_path staged")
        if not 0.2 <= self.reach_route_speed <= 2.0:
            raise ValueError("visual.reach_route_speed must be in [0.2, 2.0] rad/s")
        if not 0.0 <= self.reach_route_clearance <= 0.1:
            raise ValueError("visual.reach_route_clearance must be in [0, 0.1] m")
        if self.ready_pose not in ("forward", "lifted"):
            raise ValueError("visual.ready_pose must be forward or lifted")
        if self.ready_pose == "lifted" and self.elbow_min_deg >= 0:
            raise ValueError("visual.ready_pose lifted needs visual.elbow_min_deg below 0 (the forearms fold up)")
        if self.close_trigger not in ("gate", "contact"):
            raise ValueError("visual.close_trigger must be gate or contact")
        if self.hold_frame not in ("world", "pelvis"):
            raise ValueError("visual.hold_frame must be world or pelvis")
        if self.start_pose not in ("sampled", "scene"):
            raise ValueError("visual.start_pose must be sampled or scene")
        if self.face_after_brake not in ("none", "centre", "box_yaw"):
            raise ValueError("visual.face_after_brake must be none, centre or box_yaw")
        if self.ready_after_brake not in ("full", "quick", "compact"):
            raise ValueError("visual.ready_after_brake must be full, quick or compact")
        if not self.reanchor_after_brake and self.ready_after_brake == "full":
            raise ValueError(
                "visual.reanchor_after_brake false needs ready_after_brake quick or compact (full re-anchors)"
            )
        if self.guard_stride_mode not in ("floor", "recent2", "alternate"):
            raise ValueError("visual.guard_stride_mode must be floor, recent2 or alternate")
        if self.guard_brake not in ("landing", "phase"):
            raise ValueError("visual.guard_brake must be landing or phase")
        if self.collision_guard not in ("off", "mink"):
            raise ValueError("visual.collision_guard must be off or mink")
        if self.start_edges:
            if any(side not in EDGE_SIDES for side in self.start_edges) or len(set(self.start_edges)) != len(
                self.start_edges
            ):
                raise ValueError(f"visual.start_edges must be distinct sides from {EDGE_SIDES}")
            if self.start_pose != "sampled":
                raise ValueError("visual.start_edges needs start_pose sampled (the scene's robot_start faces one side)")
            if max(abs(v) for v in self.start_yaw_range_deg) >= 45:
                # render_trace_videos and the Isaac tracking shot recover a round's edge from its start yaw
                raise ValueError("visual.start_edges needs start_yaw_range_deg within (-45, 45)")
        for name, value in vars(self).items():
            if name in (
                "pose_source",
                "start_edges",
                "close_trigger",
                "hold_frame",
                "stand_lateral_gate",
                "start_pose",
                "walk_steer",
                "reach_path",
                "reach_route",
                "approach_stop",
                "ready_pose",
                "face_after_brake",
                "ready_after_brake",
                "guard_stride_mode",
                "guard_brake",
                "collision_guard",
            ):
                continue
            if name in (
                "start_advance",
                "reach_feedforward_forward",
                "reach_feedforward_up",
                "close_target",
                "reach_rotation_weight_transit",
            ):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"visual.{name} must be a finite number")
                continue
            if name in (
                "descend_z_guard",
                "lateral_hold_below_rim",
                "lateral_hold_common_mode",
                "reanchor_after_brake",
            ):
                if not isinstance(value, bool):
                    raise ValueError(f"visual.{name} must be a boolean")
                continue
            if name in RANGE_FIELDS:
                if (
                    isinstance(value, (str, bytes))
                    or len(value) != 2
                    or any(
                        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value
                    )
                    or value[0] > value[1]
                ):
                    raise ValueError(f"visual.{name} must be two finite numbers (low, high) with low <= high")
                continue
            if name == "guard_phase_band":
                if (
                    isinstance(value, (str, bytes))
                    or len(value) != 3
                    or any(
                        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value
                    )
                ):
                    raise ValueError("visual.guard_phase_band must be three finite numbers (c0, a, b)")
                continue
            if name == "ready_tool":
                if (
                    isinstance(value, (str, bytes))
                    or len(value) != 3
                    or any(
                        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value
                    )
                ):
                    raise ValueError("visual.ready_tool must be three finite numbers (forward, lateral, up)")
                continue
            if name == "elbow_min_deg":
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not -60 <= value <= 90:
                    raise ValueError("visual.elbow_min_deg must be a number in [-60, 90] (G1 elbow range)")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"visual.{name} must be a finite non-negative number")
        if not 0 <= self.waist_pitch_max <= 0.52:
            raise ValueError("visual.waist_pitch_max must be in [0, 0.52]")
        if self.reach_continuity_weight > 0.2:  # the bottle line's bound on its own velocity prior
            raise ValueError("visual.reach_continuity_weight must be within [0, 0.2]")
        for name in ("stand_depth_retries", "descend_retries", "reseat_retries"):
            if not isinstance(getattr(self, name), int) or isinstance(getattr(self, name), bool):
                raise ValueError(f"visual.{name} must be an integer")
        if not 0 < self.upright_tolerance_deg <= 20:
            raise ValueError("visual.upright_tolerance_deg must be in (0, 20]")
        if not 0 < self.walk_speed_min <= self.walk_speed:
            raise ValueError("visual.walk_speed_min must be positive and at most walk_speed")
        if not 0 < self.probe_height < self.lift_height:
            raise ValueError("visual.probe_height must be smaller than lift_height")
        if not 0.5 <= self.start_edge_min <= self.start_edge_max <= 2.5:
            raise ValueError("visual.start_edge_min/max must define an interval within [0.5, 2.5] metres")
        if self.object_setback_range[0] < 0:
            raise ValueError("visual.object_setback_range must not move the target past the near edge (low >= 0)")
        if not -1 <= self.object_along_range[0] <= self.object_along_range[1] <= 1:
            raise ValueError("visual.object_along_range is a fraction of the edge's free half length, within [-1, 1]")
        for name in (
            "stand_off",
            "stand_lateral_tolerance",
            "stand_depth_tolerance",
            "lift_seconds",
            "stand_seconds",
            "upright_tolerance_deg",
            "walk_speed",
            "walk_timeout",
            "settle_seconds",
            "pregrasp_above",
            "align_tolerance",
            "reprojection_px",
            "hold_seconds",
            "hold_position_tolerance",
            "max_tilt_deg",
            "joint_speed",
            "joint_acceleration",
            "palm_rotation_speed",
            "palm_align_tolerance_deg",
            "approach_speed",
            "close_seconds",
            "timeout_seconds",
            "vision_timeout",
            "ready_blend_seconds",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"visual.{name} must be positive")
