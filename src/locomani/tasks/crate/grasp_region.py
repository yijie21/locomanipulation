"""Grasp the open rim segment; clearance is distinct from a preferred target pose."""

from dataclasses import dataclass

import numpy as np

GRASP_POINTS = ((0.1933, 0.0, 0.060), (-0.1933, 0.0, 0.060))

# The two points above, generalised: one per end wall (object-frame +-x), 8 mm inside the
# outer face, 25 mm below the rim top. Written so the blue crate reproduces GRASP_POINTS.
GRASP_WALL_INSET_M = 0.008
GRASP_BELOW_RIM_M = 0.025


def target_key(meta: dict) -> str:
    """The scene's bimanual target; scenes built before 2026-09-22 carry only the crate."""
    return str(meta.get("target_object") or "blue_crate")


def grasp_points_for(obj: dict, below_rim: float = GRASP_BELOW_RIM_M) -> tuple[tuple[float, float, float], ...]:
    """Object-frame grasp pair from scene_meta.objects[key].box (half_extent, centre), `below_rim` under its top."""
    box = obj.get("box")
    if not box:
        return GRASP_POINTS
    hx, _, hz = (float(v) for v in box["half_extent"])
    cz = float(box.get("centre", (0.0, 0.0, 0.0))[2])
    x, z = hx - GRASP_WALL_INSET_M, cz + hz - below_rim
    return ((x, 0.0, z), (-x, 0.0, z))


@dataclass(frozen=True)
class RimRegion:
    # Current no-post crate: 269.4 mm bar, end walls begin at +/-118.7 mm.
    # A 29 mm finger pad plus 14.2 mm end clearance leaves +/-90 mm travel.
    half_length: float = 0.09
    midpoint_tolerance: float = 0.015  # keep the grip line near the empty basket centre
    across: float = 0.018
    below: float = 0.010
    above: float = 0.030

    def nearest(self, centres, tangent, tools):
        """Nearest pair whose midpoint supports the basket, allowing opposite offsets."""
        along = (np.asarray(tools) - centres) @ tangent
        along = np.clip(along, -self.half_length, self.half_length)
        # Aim inside the acceptance band so decoder tracking error does not
        # leave the measured pair perpetually just outside its boundary.
        along -= np.mean(along)
        return np.asarray(centres) + along[:, None] * tangent

    def contains(self, centres, rotation, tools, tool_rotations):
        """Both open jaws surround the bar with insertion and end-wall clearance.

        Basket x is jaw-normal, y runs along its open end rims, z is up.
        Wrist pitch is not required to equal the IK's preferred angle.
        """
        error = (np.asarray(tools) - centres) @ rotation
        jaws = np.asarray(tool_rotations)[:, :, 1]
        aligned = np.abs(jaws @ rotation[:, 0]) >= np.cos(np.radians(25))
        insertion = np.asarray(tool_rotations)[:, :, 0] @ rotation[:, 2] < -0.45
        ready = (
            (np.abs(error[:, 0]) <= self.across)
            & (np.abs(error[:, 1]) <= self.half_length)
            & (error[:, 2] >= -self.below)
            & (error[:, 2] <= self.above)
            & aligned
            & insertion
        )
        balanced = abs(np.mean(error[:, 1])) <= self.midpoint_tolerance
        return bool(np.all(ready) and balanced), error
