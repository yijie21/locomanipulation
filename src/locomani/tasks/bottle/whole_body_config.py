"""Limits for continuous supported bottle reach; geometry stays in the task."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class WholeBodyConfig:
    waist_pitch_max: float = 0.52
    pelvis_pitch_max: float = 0.70
    pelvis_height_min: float = 0.65
    joint_speed: float = 0.70
    joint_acceleration: float = 2.0
    hand_speed: float = 0.16
    close_seconds: float = 1.2
    capture_depth_m: float = 0.0012  # installed bilateral pad axial midpoint relative to calibrated tool point
    capture_depth_tolerance_m: float = 0.005  # centred window keeps bottle cylinder within both installed hard pads
    capture_lateral_tolerance_m: float = 0.012  # whole-body close admission half-width in palm-frame lateral metres
    capture_height_tolerance_m: float = 0.015  # whole-body close admission half-height in world metres
    capture_quiet_seconds: float = 0.10  # bounded capture window duration on advancing physical time
    capture_min_samples: int = 6  # distinct physical-time samples required across the capture window
    capture_position_diameter_max: float = 0.004  # maximum pairwise palm-frame position spread, metres
    capture_drift_speed_max: float = 0.03  # maximum least-squares palm-frame position drift, metres/second
    capture_rotation_diameter_max: float = 0.035  # maximum pairwise measured palm rotation spread, radians
    capture_angular_drift_max: float = 0.20  # maximum least-squares measured palm rotation drift, radians/second
    capture_brake_acceleration: float = 0.20  # terminal measured-speed braking acceleration, metres/second squared
    capture_brake_lookahead_seconds: float = 0.10  # terminal advance lookahead on measured forward speed
    contact_confirm_seconds: float = (
        0.10  # continuous bilateral contact and quiet palm-frame motion required before loading
    )
    contact_loss_grace_seconds: float = 0.12  # smoothly decelerate loading across brief contact loss
    contact_recovery_timeout: float = 0.80  # bounded wait for initial contact or stable reacquisition
    contact_speed_max: float = 0.08  # filtered bottle speed in the rotating palm frame, metres/second
    tracking_gain: float = (
        1.0  # gain on the measured-vs-reference hand discrepancy fed back into the supported targets (0.4 s filter)
    )
    tracking_correction_dynamics: bool = (
        False  # rate/acceleration-limit residual tracking changes while preserving takeover bias
    )
    tracking_translation_deadband: float = 0.002  # radial deadband around the takeover translation bias, metres
    tracking_translation_speed: float = 0.04  # maximum translation-correction rate, metres/second
    tracking_translation_acceleration: float = (
        0.20  # maximum translation-correction acceleration, metres/second squared
    )
    tracking_rotation_deadband: float = 0.015  # radial deadband around the takeover rotation bias, radians
    tracking_rotation_speed: float = 0.15  # maximum rotation-correction rate, radians/second
    tracking_rotation_acceleration: float = 0.60  # maximum rotation-correction acceleration, radians/second squared
    ik_velocity_continuity_weight: float = 0.0  # 0 = legacy solve; velocity residual weight around x + v*dt
    comfort_posture: bool = False  # blend the natural walking elbow/wrist prior and neutral waist into the reach
    comfort_waist_weight: float = 0.12  # posture residual weight for waist yaw and roll
    comfort_elbow_wrist_weight: float = 0.08  # posture residual weight for the grasp-side elbow and wrist
    pose_lowpass_hz: float = 0.0  # 0 = off; second-order low-pass on the sent 29-joint POSE reference and its velocity (smoothness test A, 2026-09-15)
    reach_seconds: float = 0.0  # 0 = per-tick entry servo; > 0 = minimum-jerk time-parameterised reach path of this duration, servo finishes if the capture is not ready at its end (test C)
    vision_timeout: float = 1.0
    walk_plan_hz: float = 5.0
    walk_filter_seconds: float = 0.45
    walk_clearance: float = 0.06  # tool point above the grip point during the walk; the tips stay above the grip even with the fingers down (dex1 fingers reach 4.3 cm past the tool point); 0.04 left the fingers-forward pose 2-3 cm out of reach
    raise_trigger_px: float = 36.0  # the arm starts turning/raising when the bottle subtends this many pixels in the ego view (0 = from the first plan); 36 px ~ 1.5 m, so the wrist turn (2 s at 0.7 rad/s) is done before the hand is over the table edge
    natural_walk_prep: bool = (
        False  # opt in to bounded stance/pregrasp search and a relaxed, comfortable right-arm posture
    )
    wrist_roi_supervisor: bool = False  # use camera-admitted wrist ROI for acquisition and sparse centering corrections
    raise_release_px: float = 28.0  # lower edge of the ego-size trigger hysteresis
    raise_late_depth: float = 0.78  # force preparation by this admitted pelvis-frame depth
    vision_confirm_frames: int = 2  # distinct camera frames required for ego trigger and wrist acquisition
    stance_search_span: float = 0.06  # candidate stop depths on either side of visible_brake_depth
    stance_candidates: int = 3  # maximum expensive swept-clearance checks per stance selection
    pregrasp_gap_min: float = 0.10
    pregrasp_gap_max: float = 0.16
    pregrasp_lateral_span: float = 0.04
    stance_ik_tolerance: float = 0.025
    walk_stop_acceleration: float = 0.8  # conservative measured gait deceleration used for stop lead, m/s^2
    walk_stop_latency: float = 0.10  # command/observation allowance in the stopping envelope, seconds
    walk_stop_lead_max: float = 0.10
    walk_tracking_gap: float = 0.02  # extra raw-target tool gap for gait/reference tracking error
    find_raise_rate: float = 0.035  # upward wrist-camera search rate before stable acquisition, m/s
    find_raise_max: float = 0.07
    roi_deadband: float = 0.10  # normalized image-coordinate half width
    roi_gain: float = 0.04  # metres per normalized image error outside the deadband
    roi_correction_max: float = 0.035
    roi_correction_hz: float = 2.0
    lean_max: float = 0.55  # torso lean target cap in the supported reach; the height term or the distance term (0 at 0.30 m, full at 0.60 m pelvis-to-grip) asks for it
    walk_joint_speed: float = 0.7  # reference joint speed cap while walking (the crate's walking preparation uses 0.7); at 0.45 the wrist turn took the whole 3.5 s walk
    lift_height: float = 0.06  # explicit vertical lift after capture, metres; the torso recovery adds ~0.2 m of its own
    lift_seconds: float = (
        5.0  # lift ramp duration; 0.16 m in 3 s (13 cm/s with the torso rise) slid the bottle out of the jaws
    )

    @property
    def waist_dofs(self):
        return "all"

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name in (
                "natural_walk_prep",
                "wrist_roi_supervisor",
                "tracking_correction_dynamics",
                "comfort_posture",
            ):
                if type(value) is not bool:
                    raise ValueError(f"whole_body.{name} must be boolean")
                continue
            if name == "capture_depth_m":
                if type(value) not in (float, int) or not math.isfinite(value):
                    raise ValueError("whole_body.capture_depth_m must be finite")
                continue
            optional = name in (
                "pose_lowpass_hz",
                "reach_seconds",
                "raise_trigger_px",
                "pregrasp_lateral_span",
                "find_raise_max",
                "walk_stop_latency",
                "walk_tracking_gap",
                "ik_velocity_continuity_weight",
            )  # 0 selects the default law
            if (
                type(value) not in (float, int)
                or not math.isfinite(value)
                or value < 0
                or (value == 0 and not optional)
            ):
                raise ValueError(f"whole_body.{name} must be positive and finite")
        if not (self.pose_lowpass_hz <= 12 and self.reach_seconds <= 10 and self.lean_max <= 1.22):
            raise ValueError("pose_lowpass_hz above 12 Hz or reach_seconds above 10 s is outside the supported range")
        if not (self.waist_pitch_max <= 0.52 and self.pelvis_pitch_max <= 0.7):
            raise ValueError("whole-body posture exceeds supported limits")
        if not (0.55 <= self.pelvis_height_min <= 0.8 and self.joint_speed <= 1.0 and self.hand_speed <= 0.25):
            raise ValueError("whole-body height or speed outside supported range")
        if not (self.tracking_gain <= 1 and self.vision_timeout <= 1 and self.close_seconds >= 0.5):
            raise ValueError("whole-body feedback, freshness or closure outside supported range")
        if not (
            -0.01 <= self.capture_depth_m <= 0.01
            and 0.002 <= self.capture_depth_tolerance_m <= 0.01
            and 0.005 <= self.capture_lateral_tolerance_m < 0.026
            and 0.005 <= self.capture_height_tolerance_m < 0.020
        ):
            raise ValueError("whole-body capture geometry is outside the supported range")
        if not (
            0.04 <= self.contact_confirm_seconds <= 0.5
            and 0.04 <= self.contact_loss_grace_seconds <= 0.5
            and self.contact_loss_grace_seconds < self.contact_recovery_timeout <= 3.0
            and 0.02 <= self.contact_speed_max <= 0.20
        ):
            raise ValueError("whole-body contact loading hysteresis is outside the supported range")
        if not (
            0.04 <= self.capture_quiet_seconds <= 0.5
            and type(self.capture_min_samples) is int
            and 2 <= self.capture_min_samples <= 25
            and 0.001 <= self.capture_position_diameter_max <= 0.02
            and 0.005 <= self.capture_drift_speed_max <= 0.08
            and 0.005 <= self.capture_rotation_diameter_max <= 0.2
            and 0.05 <= self.capture_angular_drift_max <= 0.8
            and 0.05 <= self.capture_brake_acceleration <= 1.0
            and 0.02 <= self.capture_brake_lookahead_seconds <= 0.3
        ):
            raise ValueError("whole-body centred capture window or terminal braking outside supported range")
        if not (
            0 <= self.tracking_translation_deadband <= 0.01
            and 0.01 <= self.tracking_translation_speed <= 0.15
            and 0.05 <= self.tracking_translation_acceleration <= 1.0
            and 0 <= self.tracking_rotation_deadband <= 0.10
            and 0.05 <= self.tracking_rotation_speed <= 0.5
            and 0.1 <= self.tracking_rotation_acceleration <= 2.0
        ):
            raise ValueError("whole-body tracking correction dynamics are outside the supported range")
        if not 0 <= self.ik_velocity_continuity_weight <= 0.2:
            raise ValueError("whole_body.ik_velocity_continuity_weight must be within [0, 0.2]")
        if not (0.04 <= self.comfort_waist_weight <= 0.3 and 0.04 <= self.comfort_elbow_wrist_weight <= 0.3):
            raise ValueError("whole-body comfort posture weights are outside the supported range")
        if not (1 <= self.walk_plan_hz <= 10 and 0.2 <= self.walk_filter_seconds <= 1):
            raise ValueError("walking reach requires slow planning and gait filtering")
        if not (self.walk_clearance <= 0.2 and self.walk_joint_speed <= 1.0):
            raise ValueError("walking clearance or joint speed outside supported range")
        if self.natural_walk_prep and not (
            type(self.vision_confirm_frames) is int
            and 1 <= self.vision_confirm_frames <= 5
            and 0.0 < self.stance_search_span <= 0.15
            and type(self.stance_candidates) is int
            and 1 <= self.stance_candidates <= 9
            and 0.06 <= self.pregrasp_gap_min <= self.pregrasp_gap_max <= 0.22
            and 0 <= self.pregrasp_lateral_span <= 0.10
            and 0.005 <= self.stance_ik_tolerance <= 0.06
            and 0.3 <= self.walk_stop_acceleration <= 2.0
            and 0 <= self.walk_stop_latency <= 0.3
            and 0.04 <= self.walk_stop_lead_max <= 0.25
            and 0 <= self.walk_tracking_gap <= 0.08
        ):
            raise ValueError("natural walking stance search is outside the supported range")
        if self.wrist_roi_supervisor and not (
            (self.raise_trigger_px == 0 or 0 <= self.raise_release_px <= self.raise_trigger_px)
            and 0.4 <= self.raise_late_depth <= 1.5
            and type(self.vision_confirm_frames) is int
            and 1 <= self.vision_confirm_frames <= 5
            and 0 < self.find_raise_rate <= 0.10
            and 0 <= self.find_raise_max <= 0.12
            and 0.02 <= self.roi_deadband <= 0.30
            and 0 < self.roi_gain <= 0.15
            and 0 < self.roi_correction_max <= 0.08
            and 0.5 <= self.roi_correction_hz <= self.walk_plan_hz
        ):
            raise ValueError("wrist ROI supervision is outside the supported range")
        if self.lift_height > 0.30:
            raise ValueError("whole_body.lift_height above 0.30 m is outside the supported lift")
        if not 1.0 <= self.lift_seconds <= 10.0:
            raise ValueError("whole_body.lift_seconds must be within [1, 10] s")
