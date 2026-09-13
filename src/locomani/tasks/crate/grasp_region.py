"""Grasp the open rim segment; clearance is distinct from a preferred target pose."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RimRegion:
    # Current no-post crate: 269.4 mm bar, end walls begin at +/-118.7 mm.
    # A 29 mm finger pad plus 14.2 mm end clearance leaves +/-90 mm travel.
    half_length: float = 0.09
    across: float = 0.018
    below: float = 0.010
    above: float = 0.030

    def nearest(self, centres, tangent, tools):
        """Choose one common along-rim offset, preserving bilateral spacing."""
        along = np.mean((np.asarray(tools) - centres) @ tangent)
        return np.asarray(centres) + np.clip(along, -self.half_length, self.half_length) * tangent

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
        return bool(np.all(ready)), error
