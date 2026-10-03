"""Collision and inertia numbers for a scene file's MuJoCo build (TODO 9.6.5); the XML is written by the script.

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


def fmt(values) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)
