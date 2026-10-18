"""Collision and inertia numbers for a scene file's MuJoCo build (TODO 19.6.5); the XML is written by the script.

The build follows `scripts/build_mujoco_scene.py`: the room is scenery (no contact), and contact comes from
simple shapes. A support surface is a 2 cm slab whose top is the surface top, so whatever rests on it rests at
the measured height. A container is a floor and four walls (a convex hull would be a solid block nothing can be
put into); other objects collide with their convex hull. Inertia is a solid box of the object's mass, which is
all the estimate (Claude's mass over the ADT box) supports.
"""

from __future__ import annotations

import numpy as np

SLAB = 0.02  # m


def yaw_quat(yaw: float) -> list[float]:
    return [round(float(np.cos(yaw / 2)), 6), 0.0, 0.0, round(float(np.sin(yaw / 2)), 6)]


def slab(surface: dict) -> dict:
    """A box geom (pos, quat, size as half extents) whose top face is the surface."""
    x, y = surface["xy"]
    hx, hy = surface["half_extent"]
    return {
        "pos": [round(float(x), 4), round(float(y), 4), round(float(surface["top"]) - SLAB / 2, 4)],
        "quat": yaw_quat(float(surface.get("yaw", 0.0))),
        "size": [round(float(hx), 4), round(float(hy), 4), SLAB / 2],
    }


def container_boxes(box: dict) -> list[dict]:
    """Floor + four walls inside the object's box (body frame): [{name, pos, size}], size = half extents."""
    hx, hy, hz = box["half_extent"]
    cx, cy, cz = box["centre"]
    w = float(box.get("wall", 0.01)) / 2
    return [
        {"name": "floor", "pos": [cx, cy, cz - hz + w], "size": [hx, hy, w]},
        {"name": "wall_px", "pos": [cx + hx - w, cy, cz], "size": [w, hy, hz]},
        {"name": "wall_nx", "pos": [cx - hx + w, cy, cz], "size": [w, hy, hz]},
        {"name": "wall_py", "pos": [cx, cy + hy - w, cz], "size": [hx - 2 * w, w, hz]},
        {"name": "wall_ny", "pos": [cx, cy - hy + w, cz], "size": [hx - 2 * w, w, hz]},
    ]


def box_inertia(mass: float, box: dict) -> dict:
    """Solid-box inertial at the box centre: {mass, pos, diaginertia}."""
    lx, ly, lz = (2 * float(v) for v in box["half_extent"])
    k = mass / 12.0
    return {
        "mass": round(float(mass), 5),
        "pos": [round(float(v), 4) for v in box["centre"]],
        "diaginertia": [
            max(k * (ly * ly + lz * lz), 1e-7),
            max(k * (lx * lx + lz * lz), 1e-7),
            max(k * (lx * lx + ly * ly), 1e-7),
        ],
    }


PANEL_PROBE = (0.01, 0.04)  # m: an end's probe runs from this far inside the slab's end to this far beyond it
PANEL_MAX_H = 0.25  # m above the surface (or up to the surface above it, when that is lower)
PANEL_COVERAGE = 0.5  # share of the probe's face the furniture's surface must fill to count as a side panel
PANEL_T = 0.01  # m: the wall's thickness


def ceiling(surfaces: dict, key: str) -> float | None:
    """The top of the lowest surface of the same furniture over `key` (above it by more than 5 cm, its centre within
    half of `key`'s larger half extent), None when open above."""
    s = surfaces[key]
    tops = [
        float(t["top"])
        for t in surfaces.values()
        if t.get("of_object") == s.get("of_object")
        and float(t["top"]) > float(s["top"]) + 0.05
        and np.hypot(t["xy"][0] - s["xy"][0], t["xy"][1] - s["xy"][1]) < 0.5 * max(s["half_extent"])
    ]
    return min(tops) if tops else None


def side_panels(
    surface: dict, points: np.ndarray, area_per_point: float, ceiling: float | None, open_edges=()
) -> list[dict]:
    """The panels around a surface that the furniture's own mesh shows (2026-10-18): a row of books rests against the
    shelf's sides and back, but the furniture's collider is the robot's only and stops under its lowest surface, so
    the end book of a row stood out over the slab's end and the row fell end to end, and books crept off the back.
    `points`: the furniture's mesh surface sampled uniformly (world frame), each worth `area_per_point` m^2;
    `ceiling`: the top of the surface above (None: open above); `open_edges`: edges ("+x", "-y", ...: the slab's local
    frame, as approach edges) the robot works from, which keep no panel along the long sides (the front of a cabinet
    is open or a door, and the scene treats doors as open); the short ends ("+x", "-x") get theirs wherever the mesh
    has one. Returns box geoms (edge, cover, pos, quat, size) on the edges where the mesh fills PANEL_COVERAGE of the
    probe's face."""
    x, y = surface["xy"]
    half = np.asarray(surface["half_extent"][:2], float)
    top, yaw = float(surface["top"]), float(surface.get("yaw", 0.0))
    high = top + PANEL_MAX_H if ceiling is None else min(top + PANEL_MAX_H, float(ceiling) - SLAB - 0.01)
    if high <= top + 0.03:
        return []
    c, s = np.cos(-yaw), np.sin(-yaw)
    d = np.asarray(points, float)
    loc = np.stack([c * (d[:, 0] - x) - s * (d[:, 1] - y), s * (d[:, 0] - x) + c * (d[:, 1] - y)], axis=1)
    band = (d[:, 2] >= top + 0.01) & (d[:, 2] <= high)
    out = []
    for axis, name in ((0, "x"), (1, "y")):
        other = 1 - axis
        for sign in (1.0, -1.0):
            edge = f"{'+' if sign > 0 else '-'}{name}"
            if axis == 1 and edge in open_edges:
                continue
            probe = band & (np.abs(loc[:, other]) <= half[other])
            probe &= (sign * loc[:, axis] >= half[axis] - PANEL_PROBE[0]) & (sign * loc[:, axis] <= half[axis] + PANEL_PROBE[1])  # fmt: skip
            cover = float(probe.sum()) * area_per_point / (2 * half[other] * (high - top - 0.01))
            if cover < PANEL_COVERAGE:
                continue
            off = np.zeros(2)
            off[axis] = sign * (half[axis] + PANEL_T / 2)
            size = [0.0, 0.0, round((high - top) / 2, 4)]
            size[axis], size[other] = PANEL_T / 2, round(float(half[other]), 4)
            out.append(dict(edge=edge, cover=round(cover, 2), quat=yaw_quat(yaw), size=size,
                            pos=[round(float(x + np.cos(yaw) * off[0] - np.sin(yaw) * off[1]), 4),
                                 round(float(y + np.sin(yaw) * off[0] + np.cos(yaw) * off[1]), 4), round((top + high) / 2, 4)]))  # fmt: skip
    return out


def fmt(values) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)
