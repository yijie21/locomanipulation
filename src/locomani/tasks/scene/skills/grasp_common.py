"""Stage computations the one-hand (pick / grasp_program) and two-hand (bimanual) grasp skills share: one file, two
sections, pure functions. Decided with the user on 2026-11-23 after the first alternating run (gso-alt-a; rules.md
41): the two families reached the same two principles by different code, so the computations behind them live here,
where a change by either family's Engineers reaches the other through the code sync and is exercised by both.

1. The lift against the goal. A task whose goal asks lifted(X, c, t) is met only when X's lowest box corner is c above
   the top of the surface it rested on and X is upright under t; a lift by a fixed height can be short of that (a box
   hangs from the line between two hands and swings about it; a one-hand grasp sits where the pads caught it). So:
   `lift_goal` reads the goal, `clearance_now` measures the two numbers, `goal_check` judges them with a margin, and
   `top_up` says how much higher to send the hands when the object is held but short.
2. The collision hull's cross-section. The simulator collides each collision geom's convex hull; what a jaw or a pair of
   hands can take is read from those hulls at the grasp height, not from the object's box or its name (rules.md 1,
   graspability): `pieces` gives the hulls at the pose now, `cut` / `band` their points at a height or over a band, and
   `extent` / `width` their reach along a horizontal axis, within a strip along another axis when asked (the depth the
   fingers cover). The one-hand skill measures a width across the jaw over the pads' height; the two-hand skill scans
   heights for the band where both walls are at their full width. Both are these functions.

Policy lives with the family: every tolerance, margin, scan step and retry count is a parameter here and a constant in
pick.py, grasp_program.py or bimanual.py (rules.md 41, prompt). This module reads no task-family configuration and no
candidate overrides, and holds no constants.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from locomani.scene_discovery.predicates import parse
from locomani.tasks.scene import graspability

Piece = tuple[np.ndarray, np.ndarray]  # (n, 3) world vertices of one convex collision piece, (k, 2) its hull's edges

# -- 1. the lift against the goal -------------------------------------------------------------------------------------


def lift_goal(ctx, key: str) -> tuple[float, float] | None:
    """The task's lifted(key, clearance, tilt) as (clearance m, tilt deg), or None when its goal asks for none (a
    held() alone, a put-in's pick of the carried object)."""
    for text in getattr(getattr(ctx, "task", None), "goal", None) or ():
        p = parse(text)
        if p.name == "lifted" and p.args[0] == key:
            return float(p.args[1]), float(p.args[2])
    return None


def quat_mat(quat) -> np.ndarray:
    """The rotation matrix of a (w, x, y, z) quaternion."""
    w, x, y, z = (float(c) for c in quat)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


_CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)


def low_offset(quat, box: dict) -> float:
    """Depth (m) of the object's lowest box corner below its origin at `quat` (the scene file's `box`: half_extent and
    an optional centre)."""
    corners = np.asarray(box.get("centre") or (0, 0, 0), float) + _CORNERS * np.asarray(box["half_extent"], float)
    return float(-(corners @ quat_mat(quat).T)[:, 2].min())


def underside(world, key: str) -> float:
    """The object's lowest box corner (m) at its pose now."""
    pos, quat = world.object_pose(key)
    return float(pos[2]) - low_offset(quat, world.meta["objects"][key]["box"])


def clearance_now(world, key: str) -> tuple[float, float]:
    """lifted()'s two numbers now: (the lowest box corner above the top of the surface the scene file rests `key` on,
    m; its upright axis off the vertical, deg)."""
    meta = world.meta["objects"][key]
    surface = world.meta["surfaces"].get(meta.get("surface")) or {}
    axis = np.asarray((meta.get("semantic") or {}).get("upright_axis") or (0.0, 0.0, 1.0), float)
    up = quat_mat(world.object_pose(key)[1]) @ (axis / np.linalg.norm(axis))
    tilt = math.degrees(math.acos(float(np.clip(up[2], -1.0, 1.0))))
    return underside(world, key) - float(surface.get("top", 0.0)), tilt


def goal_check(world, key: str, goal: tuple[float, float] | None, margin_m: float, ok: bool, values: dict):
    """A lift stage's check extended by the goal: `ok` and `values` are the family's own check (the rise, the hands on);
    with a goal (clearance, tilt) the lowest corner must be clearance + margin_m above the support top and the tilt
    under the limit. Values add low_cm, need_cm, tilt_deg; without a goal they are returned as they came."""
    if goal is None:
        return ok, values
    low, tilt = clearance_now(world, key)
    need = goal[0] + margin_m
    values.update(low_cm=round(100 * low, 2), need_cm=round(100 * need, 1), tilt_deg=round(tilt, 1))
    return ok and low >= need and tilt < goal[1], values


def top_up(values: dict, goal: tuple[float, float], z_last: Sequence[float], z_cap: float,
           min_dz_m: float = 0.005) -> list[float] | None:  # fmt: skip
    """Where to send each hand (one or two) when the object is held but short of the goal's clearance after a lift
    judged by `goal_check`: each `z_last` raised by the shortfall, no higher than `z_cap`. None when the object is
    tipped to the goal's limit (higher will not right it) or no hand would rise min_dz_m."""
    if float(values["tilt_deg"]) >= goal[1]:
        return None
    short = max(float(values["need_cm"]) - float(values["low_cm"]), 0.0) / 100
    new = [min(float(z) + short, float(z_cap)) for z in z_last]
    if max(n - float(z) for n, z in zip(new, z_last)) < min_dz_m:
        return None
    return new


# -- 2. the collision hull's cross-section --------------------------------------------------------------------------------


def pieces(world, key: str) -> list[Piece]:
    """The object's collision pieces at its pose now (graspability's body parts: a mesh's vertices, a box's corners,
    a round geom's sampled surface, each with its hull's edges); [] without a simulator model (a test's world), so
    callers fall back to the scene file's box."""
    if getattr(getattr(world, "sim", None), "m", None) is None:
        return []
    p, q = world.object_pose(key)
    Rb = graspability._quat_mat(q)
    return [(np.asarray(p, float) + part.verts @ Rb.T, part.edges) for part in graspability.body_parts(world, key)]


def prepare(verts_list: Sequence) -> list[Piece]:
    """Pieces from plain vertex arrays (a scene file's box corners, a test's blocks, one mesh's vertices): each array's
    convex hull edges (every pair when the hull is flat or the array tiny)."""
    out = []
    for v in verts_list:
        if isinstance(v, tuple) and len(v) == 2:  # already a piece (from `pieces`)
            out.append(v)
            continue
        v = np.asarray(v, float)
        if len(v) >= 2:
            out.append((v, graspability._hull_edges(v)))
    return out


def cut(piece: Piece, z: float) -> np.ndarray:
    """(k, 3) points of the piece's hull on the horizontal plane `z`: its edges' crossings and its vertices on it
    (empty when the plane misses it)."""
    v, e = piece
    a, b = v[e[:, 0]], v[e[:, 1]]
    da, db = a[:, 2] - z, b[:, 2] - z
    k = da * db < 0
    return np.concatenate([a[k] + (da[k] / (da[k] - db[k]))[:, None] * (b[k] - a[k]), v[np.abs(v[:, 2] - z) < 1e-9]])


def band(piece: Piece, lo: float, hi: float) -> np.ndarray:
    """(k, 3) points of the piece's hull between the heights `lo` and `hi`: its vertices there and the cuts at both."""
    v, _ = piece
    return np.concatenate([v[(v[:, 2] >= lo) & (v[:, 2] <= hi)], cut(piece, lo), cut(piece, hi)])


def extent(points: np.ndarray, origin_xy, axis_xy, strip: tuple | None = None) -> tuple[float, float] | None:
    """(lowest, highest) coordinate along the horizontal `axis_xy` from `origin_xy` of one convex piece's section
    `points` -- or, with `strip` = (other horizontal axis, t0, dt), of the part of it within |t - t0| <= dt along that
    axis: the points inside the strip and the crossings of the strip's two lines by the chords between the points (a
    convex section's extreme within a strip lies on a vertex or on its boundary). None without any."""
    if len(points) == 0:
        return None
    rel = np.asarray(points, float)[:, :2] - np.asarray(origin_xy, float)[:2]
    s = rel @ np.asarray(axis_xy, float)[:2]
    if strip is None:
        return float(s.min()), float(s.max())
    t_axis, t0, dt = strip
    r = rel @ np.asarray(t_axis, float)[:2] - float(t0)
    out = [s[np.abs(r) <= dt]]
    i, j = np.triu_indices(len(s), 1)
    for line in (-dt, dt):
        ri, rj = r[i] - line, r[j] - line
        x = ri * rj < 0
        out.append(s[i[x]] + (ri[x] / (ri[x] - rj[x])) * (s[j[x]] - s[i[x]]))
    s = np.concatenate(out)
    return (float(s.min()), float(s.max())) if len(s) else None


def section_extent(pieces_: Sequence[Piece], z: float, origin_xy, axis_xy, strip: tuple | None = None):
    """`extent` of the pieces' sections at height `z`, piece by piece (a chord never spans two pieces), as one
    (lowest, highest) pair; None when no piece has a section there (within the strip)."""
    lo, hi = np.inf, -np.inf
    for piece in pieces_:
        v = piece[0]
        if not (v[:, 2].min() <= z <= v[:, 2].max()):
            continue
        e = extent(cut(piece, z), origin_xy, axis_xy, strip)
        if e is not None:
            lo, hi = min(lo, e[0]), max(hi, e[1])
    return (lo, hi) if hi >= lo else None


def width(points: np.ndarray, axis_xy) -> float:
    """The span (m) of `points` along the horizontal `axis_xy` (0 without points)."""
    e = extent(points, (0.0, 0.0), axis_xy)
    return 0.0 if e is None else e[1] - e[0]


def band_width(pieces_: Sequence[Piece], lo: float, hi: float, axis_xy) -> float:
    """The span of the pieces' hulls between the heights `lo` and `hi` along `axis_xy` (the one-hand side grasp's
    width across the jaw over the pads' height; 0 without any)."""
    pts = [band(p, lo, hi) for p in pieces_]
    return width(np.concatenate(pts), axis_xy) if pts else 0.0


def cuts_width(pieces_: Sequence[Piece], levels: Sequence[float], axis_xy) -> float | None:
    """The span along `axis_xy` of the pieces' sections at every height in `levels` together (graspability.section's
    cuts stacked); None when no level cuts a piece."""
    pts = [cut(p, float(z)) for p in pieces_ for z in levels]
    pts = [p for p in pts if len(p)]
    return width(np.concatenate(pts), axis_xy) if pts else None
