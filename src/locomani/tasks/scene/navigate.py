"""Walking around furniture (TODO 27.2.3): the motor layer only walks straight lines, so a walk across the room is a
path on the occupancy map (primitives/navigator.py: its fast-marching field and ring follower) cut into straight legs
where the line of sight on the traversable cells breaks, then `Motor.walk_to` per leg. A goal inside the obstacle
inflation (a stance at a table) is reached by a last straight leg from the nearest free cell; the motor check walked
straight from the scene start into the utility cart and fell (2026-10-16)."""

from __future__ import annotations

import numpy as np

from locomani.backends.scalebfm.motor import Outcome

STEP_M = 0.25  # follower ring, metres
MAX_STEPS = 400


def line_clear(trav: np.ndarray, occ, a, b) -> bool:
    """Every cell under the segment a-b traversable (sampled at half a cell)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / (occ.res / 2))) + 1)
    for u in np.linspace(0.0, 1.0, n):
        c = occ.cell_of(a + u * (b - a))
        if not occ.in_bounds(c) or not trav[c[0], c[1]]:
            return False
    return True


def path(occ, nav, start, goal) -> list[np.ndarray] | None:
    """Corner points from `start` to `goal` (both included), or None when the goal is unreachable."""
    start, goal = np.asarray(start, float)[:2], np.asarray(goal, float)[:2]
    if not nav.set_goal_point(goal, here_xy=start) or not nav.reachable(start):
        return None
    reach = occ.xy_of(occ.nearest_free(goal)) if not occ.is_free(goal) else goal
    pts, here = [start], start
    for _ in range(MAX_STEPS):
        if np.linalg.norm(here - reach) <= STEP_M:
            break
        w = nav.waypoint(here, STEP_M)
        if w is None:
            return None
        pts.append(np.asarray(w, float))
        here = pts[-1]
    pts.append(reach)
    corners, i = [pts[0]], 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not line_clear(nav.trav, occ, pts[i], pts[j]):
            j -= 1
        corners.append(pts[j])
        i = j
    if np.linalg.norm(corners[-1] - goal) > 1e-6:
        corners.append(goal)
    return corners


def walk_path(motor, occ, nav, goal_xy, goal_yaw: float):
    """Walk to (goal_xy, goal_yaw) around obstacles. Returns the last walk's Outcome (or the failing one)."""
    corners = path(occ, nav, motor.pelvis()[0][:2], goal_xy)
    if corners is None:
        return Outcome(False, "no path on the occupancy map", motor.t, motor.t)
    out = None
    for k, c in enumerate(corners[1:], start=1):
        last = k == len(corners) - 1
        out = motor.walk_to(c, goal_yaw if last else None)
        if not out.ok:
            return out
    out.info["legs"] = len(corners) - 1
    return out
