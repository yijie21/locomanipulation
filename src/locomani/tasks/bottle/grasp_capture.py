"""Acceptance region for the calibrated upright bottle / Dex1 grasp.

The preferred point guides planning; delivered capture is a region. Furniture
clearance, object disturbance, and stable closure are checked by the caller.
"""

import numpy as np

from locomani.backends.sonic.primitives import Primitives
from locomani.backends.sonic.primitives_const import GRASP_CENTRE_OFFSET_CONST

DEPTH_TOL = 0.020
LATERAL_TOL = 0.026
HEIGHT_TOL = 0.020
JAW_TILT_MAX = 0.30
# Template pose residual is distinct from measured jaw tilt. Planning and
# the level-bottle executor share these bounds; actual capture/collision
# checks still decide whether a candidate may close or move.
IK_POSITION_TOL = 0.020
IK_ROTATION_TOL = 0.30
OFFSET = np.asarray(GRASP_CENTRE_OFFSET_CONST)


def bilateral_dex1_contact(obj, side):
    prefix = "R" if side == "right" else "L"
    links = set(obj.get("touch_links", []))
    return {f"{prefix}.dex1_finger_1", f"{prefix}.dex1_finger_2"} <= links


def closure_geometry(state, side, target):
    palm = np.asarray(state["palms"][side]["pos"])
    rot = Primitives._rot(state["palms"][side]["quat"])
    error = rot.T @ (np.asarray(target) - palm) - OFFSET
    height_error = float((np.asarray(target) - palm - rot @ OFFSET)[2])
    depth, lateral = float(error[0]), float(error[1])
    jaw_tilt = abs(float(rot[2, 1]))
    reasons = []
    for failed, name in (
        (abs(depth) >= DEPTH_TOL, "depth"),
        (abs(lateral) >= LATERAL_TOL, "lateral"),
        (abs(height_error) >= HEIGHT_TOL, "height"),
        (jaw_tilt >= JAW_TILT_MAX, "jaw_tilt"),
    ):
        if failed:
            reasons.append(name)
    return dict(
        ready=not reasons,
        depth_m=depth,
        lateral_m=float(np.linalg.norm(error[1:])),
        vertical_m=float(error[2]),
        world_height_error_m=height_error,
        lateral_signed_m=lateral,
        jaw_tilt=jaw_tilt,
        rejection_reasons=reasons,
    )


def planning_points(preferred):
    """Prefer the low point, then interior alternatives (leave tracking room)."""
    return [np.asarray(preferred) + [0.0, 0.0, dz] for dz in (0.0, 0.006, -0.006, 0.012, -0.012)]


def height_correction(geometry):
    """Correct measured height only outside the same capture region."""
    return geometry["world_height_error_m"] if abs(geometry["world_height_error_m"]) >= HEIGHT_TOL else 0.0
