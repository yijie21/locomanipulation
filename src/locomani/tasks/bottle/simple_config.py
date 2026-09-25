"""Simple single-arm bottle expert: open-loop phase durations and hard measured gates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields


@dataclass(frozen=True)
class SimpleGraspConfig:
    hang_shoulder_pitch_rad: float = 0.0  # walking arm pose (both arms, left mirrored in roll/yaw): the deploy's own hang (shoulder yaw 0.79, elbow 1.08) points each wrist camera at its hip from 2-6 cm
    hang_shoulder_roll_rad: float = 0.45  # outboard magnitude (right negative, left positive): the whole arm 8 cm further from the body with an untwisted wrist keeps the camera cones >= 35 cm from own body across the gait (offline ray sweep on stage4_wrap8n walks; 0.25 hits the thigh at 12 cm, 0.35 leaves 23 cm)
    hang_shoulder_yaw_rad: float = 0.0  # right; mirrored for the left
    hang_elbow_rad: float = 1.2  # forearm hanging as in the deploy's pose: a forward forearm (0.8, pitch 0.25) changed the gait and the stops overshot to 0.08-0.10 m (stage4_wrap8h)
    hang_wrist_roll_rad: float = 0.0  # right, mirrored: no wrist turn in the hang (the earlier roll 1.2 / pitch -0.6 / yaw -0.4 cleared the camera by twisting the hand, which looked wrong); the shoulder roll does it instead
    hang_wrist_pitch_rad: float = 0.0
    hang_wrist_yaw_rad: float = 0.0  # right; mirrored
    hang_ramp_rad_s: float = (
        1.0  # joint speed of the ramp from the deploy's pose to the walking hang, taken standing before the first step
    )
    hang_reached_rad: float = 0.15  # measured arm joints within this of the hang before the walk starts; the deploy tracks the hang with a 0.08-0.14 rad steady-state offset (stage4_wrap8n), so 0.08 ran every round to the 3 s timeout
    hang_prepare_timeout_s: float = 3.0  # give up waiting for the arms and walk anyway after this long
    raise_outboard_m: float = 0.0  # the raise ends this far outboard of the hanging tool point; 0.06 was needed when the hang sat 22 cm from the pelvis (camera 1-2 cm from the hip going straight up), the 0.45 shoulder roll hang starts 30 cm out
    stand_settle_s: float = 0.0  # legs and base below the rest thresholds for this long before the takeover; 0 = take over the tick the gait has ended (a still hanging-arm phase is what a policy learns as "stop at the table")
    stand_settle_timeout_s: float = 4.0  # take over anyway after this long; the stance is recorded, not failed
    stand_leg_speed_rad_s: float = 0.50  # leg joint speed below which the gait counts as ended
    stand_pelvis_speed_m_s: float = 0.10  # base planar speed below which the gait counts as ended
    target_grace_s: float = 8.0  # the cameras must have admitted the bottle within this long (the ego frame loses it below 0.22 m of depth at the stand); otherwise wait this long for a fresh admission before failing
    wrap_yaw_per_m: float = (
        1.2  # fingers turn left across the body by this much per metre of the bottle's rightward offset
    )
    wrap_yaw_max_rad: float = 0.35  # cap on that turn (20 deg): the offline IK sweep on eight recorded stands gives the arm its largest joint margins (65-70 deg) with the fingers forward to 20 deg inward, 42-55 deg on the old outward bearing and 17-27 deg at 60 deg inward
    raise_above_grip_m: float = 0.08  # tool height above the grip at the end of the raise
    raise_table_margin_m: float = 0.12  # and at least this far above the table top: the levelled fingers hang 5.8 cm below the tool point, the hand sags 4-5 cm on the way
    raise_forward_m: float = 0.0  # the raise ends this far ahead of the hanging (or pulled-back) tool point along the heading; the slide to the pre-grasp does the forward travel
    raise_seconds: float = 2.0  # arc-length share of the reach for the raise (the reach is one minimum-jerk profile)
    retract_m: float = 0.10  # pull the hand back along the heading by this much before rising when it sits in the table-top band at the edge
    retract_edge_margin_m: float = 0.08  # ... "at the edge" = the hand's forward extent within this of the table edge
    retract_speed_m_s: float = 0.15  # constant tool speed of the pull-back (it starts moving on the first tick)
    pregrasp_back_m: float = 0.18  # tool point behind the grip along the entry line at the end of the pre-grasp; the measured hand overruns the end of the slide by 6-8 cm toward the bottle (stage4_wrap8c), 12 cm put the fingers on it
    pregrasp_up_m: float = 0.0  # tool point above the grip height at the end of the pre-grasp
    pregrasp_seconds: float = 2.5  # arc-length share of the reach for the slide from the raised point to the pre-grasp
    settle_tolerance_m: float = 0.02  # measured tool within this distance of the pre-grasp command before entering
    settle_timeout_s: float = 1.5  # give up waiting for the pre-grasp settle after this long and enter anyway
    dash_speed_m_s: float = 0.12  # tool command speed for the entry from the pre-grasp to the grip
    lateral_correction_speed_m_s: float = (
        0.03  # the entry command follows the measured jaw-axis offset at up to this speed
    )
    lateral_correction_max_m: float = 0.04  # ... and by at most this much in total
    entry_overshoot_m: float = (
        0.02  # the entry command runs this far past the grip so tracking lag still centres the bottle
    )
    entry_capture_depth_m: float = 0.01  # measured tool within this distance of the grip along the line closes the jaw
    entry_timeout_s: float = 6.0  # entry without capture or finger contact fails
    entry_lateral_tolerance_m: float = 0.025  # bottle offset along the jaw axis allowed before closing; the jaw has 2 mm of clearance but the hand wanders 4-6 cm p-p during the entry, and the low-CoM bottle slides to the centre when one pad lands first (that is what the scene is for), so the bilateral contact after the close is the gate that counts
    entry_height_tolerance_m: float = 0.03  # bottle offset below/above the tool point allowed before closing
    entry_attempts: int = (
        2  # entries per round; a misaligned arrival retreats, shifts by the measured offset and re-enters (last resort)
    )
    retreat_m: float = 0.08  # retreat along the entry line before a re-entry
    retreat_seconds: float = 1.0  # duration of the retreat move
    close_seconds: float = 1.0  # jaw closure ramp
    contact_timeout_s: float = 0.8  # bilateral pad contact must appear within this time after the closure ramp
    lift_height_m: float = 0.08  # world rise of the held tool during the lift, torso lean held
    lift_seconds: float = 2.0  # rise duration
    recover_seconds: float = 1.5  # torso lean returns to upright; the held hand rides the torso
    min_rise_m: float = 0.04  # measured bottle rise required at the end of the lift
    disturb_limit_m: float = 0.05  # bottle knocked further than this before the close fails the round
    lean_max_rad: float = 0.55  # torso lean cap for the supported reach
    approach_stop: str = field(
        default="guard", metadata={"choices": ("visible", "guard")}
    )  # the default since 2026-09-25 (user decision; pilot bottle_simple_guard_lifted_20260925 10/10). visible = walk with the arms in the hang above, the checkpoint's speed bands and the visible depth/edge brake, and pull the hand back if it stands at the edge; guard = the warehouse walk (user, 2026-09-25): the walking policy's own arms, one constant walk_speed held in one direction (straight at the planned stand), GuardBrake at the table edge (tasks/bottle/edge_guard.py), then both hands straight up in place before the right one reaches; the hang_* and retract_* fields are not read
    walk_speed: float = 0.25  # guard: SLOW_WALK command from the first step to the brake; v1.1 carries 4.5 +- 2.5 cm after IDLE at 0.25, the least of any speed measured (walk scene, 2026-09-23)
    guard_extra_m: float = 0.20  # guard: how far past the front planted ankle the table-band front ends after IDLE; 0.20 is the warehouse calibration for the feet, stats.guard.realised_extra_m measures it on this line
    guard_margin_m: float = 0.03  # guard: the band front is to stop at least this far behind the edge
    guard_stride0_m: float = (
        0.14  # guard: floor of the predicted next stride (the measured last strides at 0.25 are 2-22 cm)
    )
    guard_band_below_m: float = 0.12  # guard: robot geometry from this far below the table top ...; the dining table's top and apron reach 0.12 m down at the edge, its legs are 9 cm further in
    guard_band_above_m: float = 0.03  # ... to this far above it must stay behind the edge
    ready_tool_forward_m: float = 0.16  # guard: the warehouse's lifted ready (crate visual.ready_tool, BimanualIK.lifted_ready): both tool points this far ahead of the pelvis ...
    ready_tool_lateral_m: float = 0.16  # ... this far to each side (the walking policy's hang is 0.21) ...
    ready_tool_up_m: float = 0.05  # ... and this far above it, reached by a joint-space blend with the waist locked; the straight Cartesian rise that kept the hands where they hung raised the shoulders out to 53-68 deg with the elbows 13-16 cm outboard (bottle_simple_guard_straight_20260925, user review)
    ready_pitch_deg: float = 30.0  # guard: palms pitched this far below level in the ready pose
    ready_elbow_min_deg: float = -60.0  # guard: elbow floor of the ready solve; the forearms fold above level, which the 5 deg floor of the reach solvers forbids (test_crate_lifted_ready)
    ready_seconds: float = 2.5  # guard: duration of the smoothstep joint blend from the walking arms to the ready pose (the warehouse's 125 frames)
    ready_hold_s: float = 0.3  # guard: both hands held in the ready pose this long before the right hand slides to the pre-grasp; the left one stays there for the whole grasp

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if "choices" in f.metadata:
                if value not in f.metadata["choices"]:
                    raise ValueError(f"simple.{f.name} must be one of {f.metadata['choices']}, not {value!r}")
            elif type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"simple.{f.name} must be a finite number")
        if not (
            0.0 <= self.stand_settle_s <= 3.0
            and 0.5 <= self.stand_settle_timeout_s <= 10.0
            and 0.01 <= self.stand_leg_speed_rad_s <= 1.0
            and 0.005 <= self.stand_pelvis_speed_m_s <= 0.3
            and 0.0 <= self.target_grace_s <= 20.0
        ):
            raise ValueError("simple stand settle must be 0-3 s within a 0.5-10 s timeout with sane rest thresholds")
        if not (
            all(
                abs(getattr(self, f"hang_{n}_rad")) <= 2.0
                for n in (
                    "shoulder_pitch",
                    "shoulder_roll",
                    "shoulder_yaw",
                    "elbow",
                    "wrist_roll",
                    "wrist_pitch",
                    "wrist_yaw",
                )
            )
            and 0.1 <= self.hang_ramp_rad_s <= 3.0
            and 0.01 <= self.hang_reached_rad <= 0.5
            and 0.5 <= self.hang_prepare_timeout_s <= 10.0
            and 0.0 <= self.raise_outboard_m <= 0.15
        ):
            raise ValueError(
                "simple hang pose must be within 2 rad per joint, ramp 0.1-3 rad/s, raise outboard 0-15 cm"
            )
        if not (0.0 <= self.wrap_yaw_per_m <= 10.0 and 0.0 <= self.wrap_yaw_max_rad <= 1.5):
            raise ValueError("simple wrap yaw must be 0-10 rad/m capped within [0, 1.5] rad")
        if not (
            0.03 <= self.raise_above_grip_m <= 0.20
            and 0.05 <= self.raise_table_margin_m <= 0.30
            and 0.0 <= self.raise_forward_m <= 0.25
            and 1.0 <= self.raise_seconds <= 6.0
        ):
            raise ValueError(
                "simple raise must end 3-20 cm above the grip, 5-30 cm above the table, 0-25 cm ahead, over 1-6 s"
            )
        if not (
            0.03 <= self.retract_m <= 0.20
            and 0.0 <= self.retract_edge_margin_m <= 0.20
            and 0.03 <= self.retract_speed_m_s <= 0.5
        ):
            raise ValueError("simple retract must be 3-20 cm within a 0-20 cm edge margin at 3-50 cm/s")
        if not (0.05 <= self.pregrasp_back_m <= 0.30 and 0.0 <= self.pregrasp_up_m <= 0.10):
            raise ValueError("simple pre-grasp point must be 5-30 cm behind and 0-10 cm above the grip")
        if not (0.005 <= self.settle_tolerance_m <= 0.05 and 0.5 <= self.settle_timeout_s <= 6.0):
            raise ValueError("simple settle tolerance must be 0.5-5 cm and its timeout 0.5-6 s")
        if not (1.0 <= self.pregrasp_seconds <= 8.0 and 0.02 <= self.dash_speed_m_s <= 0.2):
            raise ValueError("simple pre-grasp duration must be 1-8 s and the entry speed 2-20 cm/s")
        if not (0.0 <= self.lateral_correction_speed_m_s <= 0.10 and 0.0 <= self.lateral_correction_max_m <= 0.08):
            raise ValueError("simple lateral correction must be 0-10 cm/s and at most 0-8 cm")
        if not (0.0 <= self.entry_overshoot_m <= 0.04 and 0.0 <= self.entry_capture_depth_m <= 0.03):
            raise ValueError("simple entry overshoot must be 0-4 cm and the capture depth 0-3 cm")
        if not (0.0 <= self.entry_lateral_tolerance_m <= 0.03 and 0.0 <= self.entry_height_tolerance_m <= 0.06):
            raise ValueError("simple entry alignment tolerances must be 0-3 cm lateral and 0-6 cm vertical")
        if not (
            1 <= int(self.entry_attempts) <= 6 and 0.03 <= self.retreat_m <= 0.15 and 0.3 <= self.retreat_seconds <= 3.0
        ):
            raise ValueError("simple entry attempts must be 1-6, the retreat 3-15 cm over 0.3-3 s")
        if not (2.0 <= self.entry_timeout_s <= 15.0 and 0.5 <= self.close_seconds <= 3.0):
            raise ValueError("simple entry timeout must be 2-15 s and the closure 0.5-3 s")
        if not (0.2 <= self.contact_timeout_s <= 3.0 and 0.03 <= self.lift_height_m <= 0.20):
            raise ValueError("simple contact timeout must be 0.2-3 s and the lift 3-20 cm")
        if not (1.0 <= self.lift_seconds <= 8.0 and 0.5 <= self.recover_seconds <= 6.0):
            raise ValueError("simple lift must take 1-8 s and the recovery 0.5-6 s")
        if not (0.01 <= self.min_rise_m <= self.lift_height_m and 0.02 <= self.disturb_limit_m <= 0.10):
            raise ValueError("simple minimum rise must be within the lift and the disturbance limit 2-10 cm")
        if not 0.0 <= self.lean_max_rad <= 1.22:
            raise ValueError("simple lean cap must be within [0, 1.22] rad")
        if not 0.20 <= self.walk_speed <= 0.80:
            raise ValueError("simple guard walk speed must be 0.20-0.80 (v1.1 does not move below 0.20)")
        if not (
            0.0 <= self.guard_extra_m <= 0.40
            and 0.0 <= self.guard_margin_m <= 0.10
            and 0.05 <= self.guard_stride0_m <= 0.30
            and 0.0 <= self.guard_band_below_m <= 0.40
            and 0.0 <= self.guard_band_above_m <= 0.10
            and 0.0 <= self.ready_hold_s <= 2.0
            and 0.0 <= self.ready_tool_forward_m <= 0.40
            and 0.05 <= self.ready_tool_lateral_m <= 0.40
            and -0.20 <= self.ready_tool_up_m <= 0.40
            and -10.0 <= self.ready_pitch_deg <= 90.0
            and -60.0 <= self.ready_elbow_min_deg <= 5.0
            and 1.0 <= self.ready_seconds <= 6.0
        ):
            raise ValueError(
                "simple guard must be extra 0-40 cm, margin 0-10 cm, stride floor 5-30 cm, band 0-40 cm below "
                "and 0-10 cm above the top, ready hold 0-2 s; the ready pose 0-40 cm ahead, 5-40 cm out, -20..40 cm "
                "up, palm -10..90 deg, elbow floor -60..5 deg, blend 1-6 s"
            )
