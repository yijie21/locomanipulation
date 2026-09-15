"""Acceptance region for the calibrated upright bottle / Dex1 grasp.

The preferred point guides planning; delivered capture is a region. Furniture
clearance, object disturbance, and stable closure are checked by the caller.
"""

import numpy as np

from locomani.backends.sonic.primitives import Primitives
from locomani.backends.sonic.primitives_const import GRASP_CENTRE_OFFSET_CONST
from locomani.primitives.contacts import bilateral_dex1_contact  # noqa: F401

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

# Measured from the installed hard-pad collision meshes in
# tests/fixtures/geometry/scene_locomani.xml.  Both pads cover this axial
# interval; the palm/base collision geometry ends at x=0.1108 m.
DEX1_PAD_X_MIN = 0.120588
DEX1_PAD_X_MAX = 0.184759
DEX1_PALM_X_MAX = 0.110800


def closure_geometry(
    state,
    side,
    target,
    *,
    preferred_depth_m=0.0,
    depth_tolerance_m=DEPTH_TOL,
    lateral_tolerance_m=LATERAL_TOL,
    height_tolerance_m=HEIGHT_TOL,
):
    """Measure the target in the palm frame and admit the requested depth window.

    The optional window lets a task ask for a deeper capture without changing
    the globally calibrated tool offset or the legacy symmetric tolerance.
    """
    palm = np.asarray(state["palms"][side]["pos"])
    rot = Primitives._rot(state["palms"][side]["quat"])
    error = rot.T @ (np.asarray(target) - palm) - OFFSET
    height_error = float((np.asarray(target) - palm - rot @ OFFSET)[2])
    depth, lateral = float(error[0]), float(error[1])
    depth_error = depth - float(preferred_depth_m)
    jaw_tilt = abs(float(rot[2, 1]))
    reasons = []
    for failed, name in (
        (abs(depth_error) >= depth_tolerance_m, "depth"),
        (abs(lateral) >= lateral_tolerance_m, "lateral"),
        (abs(height_error) >= height_tolerance_m, "height"),
        (jaw_tilt >= JAW_TILT_MAX, "jaw_tilt"),
    ):
        if failed:
            reasons.append(name)
    return dict(
        ready=not reasons,
        depth_m=depth,
        preferred_depth_m=float(preferred_depth_m),
        depth_error_m=depth_error,
        depth_tolerance_m=float(depth_tolerance_m),
        lateral_tolerance_m=float(lateral_tolerance_m),
        height_tolerance_m=float(height_tolerance_m),
        lateral_m=float(np.linalg.norm(error[1:])),
        vertical_m=float(error[2]),
        world_height_error_m=height_error,
        lateral_signed_m=lateral,
        jaw_tilt=jaw_tilt,
        rejection_reasons=reasons,
    )


def capture_tool_goal(target, entry_axis, depth_m):
    """Return the tool point that realizes ``depth_m`` in the entry frame."""
    axis = np.asarray(entry_axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("capture entry axis must be finite and nonzero")
    return np.asarray(target, dtype=float) - float(depth_m) * axis / norm


def capture_axial_clearance(preferred_depth_m, depth_tolerance_m, bottle_radius_m):
    """Worst-case axial hard-pad and palm clearances for a capture window."""
    proximal = OFFSET[0] + preferred_depth_m - depth_tolerance_m - bottle_radius_m
    distal = OFFSET[0] + preferred_depth_m + depth_tolerance_m + bottle_radius_m
    clearances = dict(
        pad_proximal_m=float(proximal - DEX1_PAD_X_MIN),
        pad_distal_m=float(DEX1_PAD_X_MAX - distal),
        palm_m=float(proximal - DEX1_PALM_X_MAX),
    )
    clearances["ready"] = all(value >= 0 for value in clearances.values())
    clearances["bottle_axial_bounds_m"] = [float(proximal), float(distal)]
    clearances["pad_axial_bounds_m"] = [DEX1_PAD_X_MIN, DEX1_PAD_X_MAX]
    return clearances


def planning_points(preferred):
    """Prefer the low point, then interior alternatives (leave tracking room)."""
    return [np.asarray(preferred) + [0.0, 0.0, dz] for dz in (0.0, 0.006, -0.006, 0.012, -0.012)]


def height_correction(geometry):
    """Correct measured height only outside the same capture region."""
    return geometry["world_height_error_m"] if abs(geometry["world_height_error_m"]) >= HEIGHT_TOL else 0.0
