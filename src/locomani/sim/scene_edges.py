"""Which edge of the target surface a recorded round walked in toward, for scenes that list per-edge walk-ins.

A four-edge scene (`locomani.tasks.crate.warehouse_box_scene`) writes `surfaces.<s>.edges` -- per side the walk-in
heading, the free floor and a third-person camera -- and the expert draws one side per round (visual.start_edges).
A round starts within 45 deg of its edge's heading (visual_config refuses a wider start yaw), so the robot's yaw in
the first recorded state names the edge; the renderers read it from there rather than from the manifest, which a
closed-loop evaluation's traces and a collection's do not share. Plain math, importable from any venv.
"""

from __future__ import annotations

import math


def scene_edges(meta: dict) -> dict:
    """`{side: {heading_deg, clear_m, video_camera}}` of the target's surface; {} for a scene without them."""
    obj = (meta.get("objects") or {}).get(meta.get("target_object") or "") or {}
    return ((meta.get("surfaces") or {}).get(obj.get("surface")) or {}).get("edges") or {}


def root_yaw(qpos) -> float:
    """World yaw, radians, of the free root's w-x-y-z quaternion at qpos[3:7]."""
    w, x, y, z = (float(v) for v in qpos[3:7])
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def round_edge(meta: dict, qpos) -> dict | None:
    """The edge (its `edges` entry plus `side`) whose walk-in heading is nearest the robot yaw in `qpos`, the
    round's first recorded state; None when the scene lists no edges."""
    edges = scene_edges(meta)
    if not edges:
        return None
    yaw = root_yaw(qpos)

    def off(edge):
        return abs((math.radians(edge["heading_deg"]) - yaw + math.pi) % (2 * math.pi) - math.pi)

    side = min(edges, key=lambda s: off(edges[s]))
    return {"side": side, **edges[side]}
