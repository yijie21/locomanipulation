"""Retain a camera-admitted world target between images, without refreshing its age."""

from dataclasses import replace

import numpy as np

from locomani.tasks.crate.visual_low_reach import rotation


class VisibleTarget:
    def __init__(self):
        self.world = None

    def update(self, estimate, state):
        if estimate is None:
            self.world = None
            return
        base = rotation(state["pelvis"]["quat"])
        self.world = replace(
            estimate,
            position=np.asarray(state["pelvis"]["pos"]) + base @ estimate.position,
            rotation=base @ estimate.rotation,
        )

    def at(self, state):
        if self.world is None:
            return None
        base = rotation(state["pelvis"]["quat"])
        return replace(
            self.world,
            position=base.T @ (self.world.position - state["pelvis"]["pos"]),
            rotation=base.T @ self.world.rotation,
        )
