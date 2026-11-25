"""Which surfaces held(X) reads X as resting on, and which of them cover X (r1_held_single loop iterations 0-1).

The goal's held(X) reads X as still resting on any surface whose flat top covers X's box centre and whose top is at most
RESTING_GAP_M below X's underside, with no lower bound: a wolf figurine lifted 10 cm on the corner display shelf's
middle shelf failed as "resting on" the shelf 0.40 m above it. `resting_on` re-implements that rule from the object's
pose (the predicates stay frozen); `overhead` is its list less X's own support: the shelves above X. held.py withdraws
a lifted X from under them; pick.py keeps a covered X's side grasp level and reaches its pre-grasp from farther out.
`past_edge` is how far out of the shelves X's pull has brought it."""

from __future__ import annotations

import numpy as np

from locomani.scene_discovery.scene_file import to_surface_frame

RESTING_GAP_M = 0.02  # held(X)'s rule: X rests on a surface over its centre with its top at most this below X


def resting_on(world, key: str) -> list[str]:
    """The surfaces held(X) reads X as resting on, from its pose now: X's box centre xy (rotation ignored) inside the
    surface's flat top and X's underside at most RESTING_GAP_M above the top, or anywhere below it."""
    box = world.meta["objects"][key]["box"]
    half, centre = np.asarray(box["half_extent"], float), np.asarray(box.get("centre") or (0.0, 0.0, 0.0), float)
    pos, _ = world.object_pose(key)
    xy, bottom = pos[:2] + centre[:2], float(pos[2] + centre[2] - half[2])
    out = []
    for k, s in world.meta["surfaces"].items():
        flat = np.asarray(s.get("flat_half_extent") or s["half_extent"], float)
        if np.all(np.abs(to_surface_frame(xy, s)) <= flat) and bottom - float(s["top"]) <= RESTING_GAP_M:
            out.append(k)
    return out


def overhead(world, key: str) -> list[str]:
    """`resting_on` less X's own support (its scene-file surface): the shelves above X."""
    own = world.meta["objects"][key].get("surface")
    return [k for k in resting_on(world, key) if k != own]


def past_edge(world, key: str, surfaces, yaw: float) -> float:
    """How far (m) X's box centre xy (rotation ignored, as `resting_on`) lies outside the flat top of each of `surfaces`
    across its edge nearest a robot facing `yaw` (the edge whose outward normal points most against the facing), the
    least over them; negative inside."""
    box = world.meta["objects"][key]["box"]
    pos, _ = world.object_pose(key)
    xy = pos[:2] + np.asarray(box.get("centre") or (0.0, 0.0, 0.0), float)[:2]
    out = np.inf
    for k in surfaces:
        s = world.meta["surfaces"][k]
        flat = np.asarray(s.get("flat_half_extent") or s["half_extent"], float)
        a = yaw - float(s.get("yaw", 0.0))
        back = -np.array([np.cos(a), np.sin(a)])  # against the facing, in the surface's frame
        axis = int(np.argmax(np.abs(back)))
        out = min(out, float(np.sign(back[axis]) * to_surface_frame(xy, s)[axis] - flat[axis]))
    return float(out)
