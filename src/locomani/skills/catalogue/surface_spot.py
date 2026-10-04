"""Where a held object goes on a surface: the free spot `approach(surface)` aims the placing band at.

Pure geometry over the scene file and the privileged state (docs/handoff/put-down-single-20261003.md); the loaded
approach (`catalogue/bottle.py`) and `put_down_single` (`catalogue/put_down_single.py`) are its two callers.
"""

from __future__ import annotations

import numpy as np

SPOT_INBOARD = 0.12  # m in from the approach edge: the placing band is 0.26-0.44 m ahead of the pelvis, which stops >= 0.21 m short of an edge
SPOT_MARGIN = 0.03  # m kept clear around the held object's footprint, to the outline and to other objects
SPOT_STEP = 0.05  # m between candidate spots along the edge
ROUND_FILL = 0.83  # a square-boxed surface whose outline fills less of the box than this is a disc (pi/4 = 0.785)


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def is_surface(scene, key: str) -> bool:
    return key in scene.surfaces and key not in scene.objects


def _radius(box) -> float:
    half = (box or {}).get("half_extent")
    return float(max(half[0], half[1])) if half is not None else 0.05


def surface_occupants(meta: dict, surface: str, state: dict) -> dict:
    """{key: (xy, radius)} of every object standing on `surface` now: free bodies at their state position, the
    scenery (absent from the state) at its scene position; a footprint over the outline within 0.6 m of the top."""
    s = meta["surfaces"][surface]
    half = np.asarray(s.get("flat_half_extent") or s["half_extent"], dtype=float)[:2]
    c, sn = np.cos(-float(s.get("yaw", 0.0))), np.sin(-float(s.get("yaw", 0.0)))
    out = {}
    for key, o in meta["objects"].items():
        live = (state.get("objects") or {}).get(key)
        if live is not None:
            pos = np.asarray(live["pos"], dtype=float)
        elif o.get("init_pos") is not None:
            pos = np.asarray(o["init_pos"], dtype=float) + np.asarray((o.get("box") or {}).get("centre") or (0, 0, 0))
        else:
            continue
        r = _radius(o.get("box"))
        d = pos[:2] - np.asarray(s["xy"], dtype=float)[:2]
        local = np.array([c * d[0] - sn * d[1], sn * d[0] + c * d[1]])
        if np.all(np.abs(local) <= half + r) and float(s["top"]) - 0.05 <= pos[2] <= float(s["top"]) + 0.6:
            out[key] = (pos[:2], r)
    return out


def surface_spot(meta: dict, surface: str, here_xy, held_key: str, occupants: dict) -> dict | None:
    """Where `approach(surface)` aims the held object: on the open edge nearest `here_xy`, SPOT_INBOARD in (less on a
    shallow surface), along the edge as near `here_xy`'s projection as the outline (a disc for a round table) and the
    `occupants` ({key: (xy, radius)}) allow, SPOT_STEP apart. -> {"xy", "psi", "edge", "inboard"}, or None."""
    from locomani.tasks.bottle.plan import nearest_edge

    s = meta["surfaces"][surface]
    ax, sgn = nearest_edge(here_xy, s)
    along = 1 - ax
    half = np.asarray(s.get("flat_half_extent") or s["half_extent"], dtype=float)[:2]
    yaw = float(s.get("yaw", 0.0))
    c, sn = np.cos(yaw), np.sin(yaw)
    centre = np.asarray(s["xy"], dtype=float)[:2]
    d = np.asarray(here_xy, dtype=float)[:2] - centre
    here_local = np.array([c * d[0] + sn * d[1], -sn * d[0] + c * d[1]])
    r = _radius(meta["objects"][held_key].get("box")) + SPOT_MARGIN
    lo, hi = -half[along] + r, half[along] - r
    if 2 * half[ax] < 2 * r or lo > hi:
        return None
    inboard = float(np.clip(SPOT_INBOARD, r, 2 * half[ax] - r))
    disc = float(s.get("fill", 1.0)) < ROUND_FILL and abs(half[0] - half[1]) < 0.02
    others = [(np.asarray(xy, dtype=float), rad) for key, (xy, rad) in occupants.items() if key != held_key]
    start = float(np.clip(here_local[along], lo, hi))
    n = int(np.ceil((hi - lo) / SPOT_STEP)) + 1
    for k in range(2 * n + 1):
        a = start + ((k + 1) // 2) * SPOT_STEP * (1.0 if k % 2 else -1.0)  # start, +step, -step, +2 step, ...
        if not lo - 1e-9 <= a <= hi + 1e-9:
            continue
        local = np.zeros(2)
        local[ax], local[along] = sgn * (half[ax] - inboard), a
        if disc and float(np.hypot(*local)) > float(min(half)) - r:
            continue
        xy = centre + np.array([c * local[0] - sn * local[1], sn * local[0] + c * local[1]])
        if any(float(np.linalg.norm(xy - oxy)) < r + orad for oxy, orad in others):
            continue
        normal = np.zeros(2)
        normal[ax] = -sgn  # into the surface: the way the robot faces
        psi = _wrap(float(np.arctan2(normal[1], normal[0])) + yaw)
        return {"xy": xy, "psi": psi, "edge": [int(ax), float(sgn)], "inboard": inboard}
    return None


def reachable_spot(meta: dict, surface: str, held_key: str, occupants: dict, aim_xy, step: float = 0.02):
    """The free point on `surface` nearest the world point `aim_xy` (where the hand reaches best from where the robot
    stands now): inside the outline (a disc for a round table) by the held object's clear radius, clear of the
    `occupants`, on a `step` grid. None when the surface has no free point. put_down_single re-aims here after the
    base settles: the hand-over to the arm shifted the base 0.23-0.26 m in 3 of 4 v2/v3 put-downs."""
    s = meta["surfaces"][surface]
    half = np.asarray(s.get("flat_half_extent") or s["half_extent"], dtype=float)[:2]
    yaw = float(s.get("yaw", 0.0))
    c, sn = np.cos(yaw), np.sin(yaw)
    centre = np.asarray(s["xy"], dtype=float)[:2]
    r = _radius(meta["objects"][held_key].get("box")) + SPOT_MARGIN
    lim = half - r
    if np.any(lim < 0):
        return None
    xs = np.arange(-lim[0], lim[0] + 1e-9, step) if lim[0] > 0 else np.zeros(1)
    ys = np.arange(-lim[1], lim[1] + 1e-9, step) if lim[1] > 0 else np.zeros(1)
    local = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
    disc = float(s.get("fill", 1.0)) < ROUND_FILL and abs(half[0] - half[1]) < 0.02
    if disc:
        local = local[np.hypot(local[:, 0], local[:, 1]) <= float(min(half)) - r + 1e-9]
    world = centre + local @ np.array([[c, sn], [-sn, c]])
    for key, (xy, rad) in occupants.items():
        if key != held_key:
            world = world[np.linalg.norm(world - np.asarray(xy, dtype=float), axis=1) >= r + rad]
    if len(world) == 0:
        return None
    return world[int(np.argmin(np.linalg.norm(world - np.asarray(aim_xy, dtype=float)[:2], axis=1)))]
