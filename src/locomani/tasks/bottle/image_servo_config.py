"""Serializable RGB calibration and task feedback limits for the bottle pilot."""

import math
from dataclasses import asdict, dataclass


def positive_finite(config):
    for key, value in asdict(config).items():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")


@dataclass(frozen=True)
class BottleCalibration:
    cap_radius: float = 0.009498  # bottled_water_1 mesh radius, metres.
    cap_bottom: float = 0.069377  # cap bottom relative to bottle centre.
    cap_top: float = 0.080000  # cap top relative to bottle centre.
    min_cap_pixels: int = 8  # reject silhouettes too small to fit.
    precise_wrist_pixels: int = 250  # minimum monocular near-field cap support.
    max_fit_error_px: float = 1.8  # reject inconsistent silhouette fits.
    max_image_age_s: float = 0.15  # no motion from stale images.
    edge_noise_px: float = 2.0  # segmentation error used in local sensitivity.
    close_uncertainty_m: float = 0.010  # maximum position sensitivity for CLOSE.
    approach_uncertainty_m: float = 0.15  # coarse approach only; CLOSE retains 10mm gate.
    bright_value: int = 180  # cap highlights distinguish it from the green label.
    bright_fraction: float = 0.10  # EGL calibration and actual JPEG cap support.

    def __post_init__(self):
        positive_finite(self)
        if self.cap_top <= self.cap_bottom:
            raise ValueError("cap_top must exceed cap_bottom")
        if type(self.min_cap_pixels) is not int or type(self.precise_wrist_pixels) is not int:
            raise ValueError("pixel counts must be integers")


@dataclass(frozen=True)
class ServoLimits:
    entry_gap: float = 0.075  # clear long fingers before lowering.
    alignment_tolerance: float = 0.018  # cross-axis tolerance before entry.
    close_depth: float = 0.015  # capture-region depth tolerance.
    close_lateral: float = 0.019  # capture-region lateral tolerance.
    close_height: float = 0.016  # capture-region height tolerance.
    approach_start_depth: float = 0.90  # start visual preparation here.
    approach_end_depth: float = 0.32  # brake before squat's measured passive retreat.
    approach_lateral: float = 0.195  # comfortable right-hand offset, template 4.
    lateral_tolerance: float = 0.120  # let the arm resolve reachable lateral offsets.
    prealign_depth: float = 0.44  # start safe alignment before base braking.
    view_elbow: float = 0.30  # negative elbow reference pitches the wrist view down.
    cartesian_lead: float = 0.035  # bound reference lead over measured palm.
    tracking_gain: float = 0.7  # proportional sag compensation from SONIC servo.
    far_yaw_lead: float = 0.40  # turn toward the RGB bearing outside near field.
    joint_step: float = 0.025  # maximum reference change per tick, radians.
    body_step: float = 0.001  # screened squat proposal, metres.
    walking_floor: float = 0.20  # final braking floor from visible expert calibration.
    walking_mid: float = 0.215  # avoid carrying a coarse stride into the fingers.
    walking_mid_depth: float = 0.65  # finish coarse gait before near-field alignment.
    walking_speed: float = 0.30  # far-field gait command.
    edge_margin: float = 0.17  # independent table-edge safety veto.
    edge_lookahead: float = 0.04  # screen prospective base displacement.
    ik_position_tolerance: float = 0.035  # reject unreachable bounded goals.
    ik_rotation_tolerance: float = 0.30  # reject large orientation residuals.

    def __post_init__(self):
        positive_finite(self)
        if self.approach_start_depth <= self.approach_end_depth or self.walking_speed < self.walking_floor:
            raise ValueError("image servo approach bounds are reversed")
