"""The simple expert's guard walk to the table edge (`reach.simple.approach_stop: guard`, 2026-09-25).

The warehouse walk moved to the bottle table: one constant slow gait command -- speed and direction -- from the
first step, the walking policy's own arms, and the walk-scene GuardBrake -- at a foot landing, brake if one more stride would carry the
body past the edge less `margin`. What must stay behind the line is not the feet, as in front of the warehouse
pallet, but whatever reaches the table-top band (top and apron), measured on the robot's meshes each tick:
the feet may pass under the top. The per-round record in `stats.guard` is what `guard_extra_m` is calibrated from:
`realised_extra_m` is how far the band front ended past the front planted ankle of the braking landing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from locomani.primitives.foot_watch import FootWatch, GuardBrake


@dataclass(frozen=True)
class TableEdge:
    """A straight edge of a table footprint: `axis` (a unit vector) points across it into the table, `line` is its
    coordinate along `axis` from the world origin. clearance = line - axis . p, positive on the robot's side."""

    axis: np.ndarray
    line: float

    def clearance(self, xy) -> float:
        return float(self.line - self.axis @ np.asarray(xy, dtype=float)[:2])


def facing_edge(centre_xy, half_extent, heading, yaw: float = 0.0) -> TableEdge:
    """The edge of the footprint the heading walks into: the one whose outward normal is most opposed to it.

    `yaw` turns the footprint (scene_discovery surfaces); the hand-built scenes' tables are axis-aligned, yaw 0."""
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    local = rot.T @ np.asarray(heading, dtype=float)[:2]
    k = int(np.argmax(np.abs(local)))
    axis_local = np.zeros(2)
    axis_local[k] = np.sign(local[k])
    axis = rot @ axis_local
    return TableEdge(axis, float(axis @ np.asarray(centre_xy, dtype=float)[:2]) - float(half_extent[k]))


def straight_walk(start_xy, aim_xy, heading) -> np.ndarray:
    """Unit walking direction from the start straight at the planned stand, held for the whole walk.

    Held, not re-aimed: the deploy replans on any change of the movement direction (an exact comparison,
    g1_deploy_onnx_ref.cpp), so a direction steered every frame replans at the planner's 10 Hz; with it the
    0.25 gait stepped 15 cm, stood 1.3-1.9 s and stepped 34 cm (bottle_simple_guard_20260925, 2 rounds),
    and the brake had to allow for that step. The diagonal takes out the spawn's lateral offset. A stand
    beside or behind the start (under a fifth of the way forward) walks along the heading instead.
    """
    h = np.asarray(heading, dtype=float)[:2]
    h = h / np.linalg.norm(h)
    d = np.asarray(aim_xy, dtype=float)[:2] - np.asarray(start_xy, dtype=float)[:2]
    length = float(np.linalg.norm(d))
    if length < 1e-6 or float(d @ h) < 0.2 * length:
        return h
    return d / length


class EdgeGuard:
    """Brake decision and record for one walk to one table edge."""

    def __init__(self, model, edge: TableEdge, table_top: float, cfg):
        self.watch = FootWatch(model=model)
        self.rule = GuardBrake(extra=cfg.guard_extra_m, margin=cfg.guard_margin_m, stride0=cfg.guard_stride0_m)
        self.edge = edge
        self.band = (float(table_top) - cfg.guard_band_below_m, float(table_top) + cfg.guard_band_above_m)
        self.walk_speed = float(cfg.walk_speed)
        self.walk_direction = None  # set by the first direction() call and then held
        self.brake = None
        self.closest = (np.inf, None)  # smallest band clearance seen, and its body
        self.final = None

    def direction(self, pelvis_xy, aim_xy, heading) -> np.ndarray:
        """The walk's one direction, fixed on the first call (the first walking tick)."""
        if self.walk_direction is None:
            self.walk_direction = straight_walk(pelvis_xy, aim_xy, heading)
        return self.walk_direction

    def band_gap(self, state) -> tuple[float, str | None]:
        """Clearance of the band's frontmost robot point to the edge, and its body; tracks the closest."""
        front, body = self.watch.frontmost_band(state, np.zeros(2), self.edge.axis, *self.band)
        gap = float(self.edge.line - front)
        if gap < self.closest[0]:
            self.closest = (gap, body)
        return gap, body

    def step(self, state) -> str | None:
        """One walking tick: 'guard brake' at the deciding landing (every later tick too), 'guard crossed' if the
        band already reaches past the edge, else None."""
        landed, front, stride = self.watch.update(state, np.zeros(2), self.edge.axis)
        gap, body = self.band_gap(state)
        if gap < 0:
            return "guard crossed"
        brake, used = self.rule.should_brake(landed, front, stride, self.watch.last_strides, self.edge.line)
        if brake and self.brake is None:
            self.brake = dict(
                t=float(state["t"]),
                seq=state.get("seq"),
                front_ankle=float(front),
                front_ankle_gap_m=float(self.edge.line - front),
                stride_m=float(used),
                band_gap_m=gap,
                band_body=body,
                pelvis_gap_m=self.edge.clearance(state["pelvis"]["pos"]),
            )
        return "guard brake" if self.brake is not None else None

    def finish(self, state) -> float:
        """The stand the arms start from: final band clearance, recorded with the pelvis's."""
        gap, body = self.band_gap(state)
        self.final = dict(band_gap_m=gap, band_body=body, pelvis_gap_m=self.edge.clearance(state["pelvis"]["pos"]))
        return gap

    def record(self) -> dict:
        realised = None
        if self.brake is not None and self.final is not None:
            realised = float(self.edge.line - self.final["band_gap_m"] - self.brake["front_ankle"])
        return dict(
            axis=self.edge.axis.tolist(),
            line=self.edge.line,
            band=list(self.band),
            walk_direction=None if self.walk_direction is None else self.walk_direction.tolist(),
            extra_m=self.rule.extra,
            margin_m=self.rule.margin,
            stride0_m=self.rule.stride0,
            landings=[float(v) for v in self.watch.landings],
            brake=self.brake,
            closest_band_gap_m=float(self.closest[0]),
            closest_band_body=self.closest[1],
            final=self.final,
            realised_extra_m=realised,
        )
