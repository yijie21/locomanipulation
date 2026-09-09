"""Bounded object-relative palm stabilization during seating and closure."""

import numpy as np
from grasp_capture import OFFSET, closure_geometry
from primitives import Primitives


class CaptureFeedback:
    def __init__(self, state, side, grip):
        self.side, self.grip = side, grip
        self.origin = np.asarray(grip(state)).copy()
        self.palm = np.asarray(state["palms"][side]["pos"]).copy()
        self.rot = Primitives._rot(state["palms"][side]["quat"])
        self.error = self.origin - self.palm - self.rot @ OFFSET
        # Prefer a little interior margin without making precision a gate.
        self.error -= np.clip(0.5 * self.error, -0.006, 0.006)
        self.previous = None
        self.last_seq = None
        self.last_wall = None
        self.paused_s = 0.0

    def update(self, state, dt):
        target = np.asarray(self.grip(state))
        palm = np.asarray(state["palms"][self.side]["pos"])
        actual_rot = Primitives._rot(state["palms"][self.side]["quat"])
        # The same jaw frame used by closure_geometry. A palm translation
        # compensating wrist rotation must not look like a moving grasp.
        relative = actual_rot.T @ (target - palm) - OFFSET
        seq, wall = state.get("seq"), state.get("wall")
        fresh = seq is None or seq != self.last_seq
        sample_dt = wall - self.last_wall if wall is not None and self.last_wall is not None else dt
        speed = (
            float("inf")
            if self.previous is None or not fresh
            else float(np.linalg.norm(relative - self.previous) / max(sample_dt, 0.001))
        )
        if fresh:
            self.previous = relative.copy()
            self.last_seq, self.last_wall = seq, wall
        moved = target - self.origin
        # Compensate body sway by holding a world pose, not fixed joints.
        # Follow only small object motion; never chase a knocked-away bottle.
        followed = self.origin + moved * min(1.0, 0.01 / max(np.linalg.norm(moved), 1e-9))
        aim = followed - actual_rot @ OFFSET - self.error
        geometry = closure_geometry(state, self.side, target)
        advance = (
            fresh and 0 < sample_dt <= 0.1 and geometry["ready"] and speed < 0.045 and np.linalg.norm(moved) < 0.02
        )
        self.blocked_reasons = [
            name
            for condition, name in (
                (not fresh or not 0 < sample_dt <= 0.1, "observation"),
                (not geometry["ready"], "geometry"),
                (speed >= 0.045, "relative_speed"),
                (np.linalg.norm(moved) >= 0.02, "object_displacement"),
            )
            if condition
        ]
        self.paused_s = 0.0 if advance else self.paused_s + dt
        return aim, advance, self.paused_s > 0.8
