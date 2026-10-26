"""Vision RSI D2 (docs/handoff/vision-rsi-design-20261023.md section 17): the pre-round mapping tour. Viewpoints are
chosen greedily on the navigation grid so that every surface of a room (tops at 0.3-1.3 m) is seen from 0.6-1.5 m with
a clear line of sight; each faces the surfaces it covers; they are visited in a short closed order from the start (so
the route ends where it began: one loop for the pose graph). At each viewpoint the robot stands still while the head
camera scans (`SCAN`: pan and tilt offsets from its mount). The sim has no head joints, so the scan is a render-time
camera pose, the stand-in for the real robot's D455 bracket (design doc section 17.5; whether the real bracket tilts
is section 11's open question).

Step 2a plans the route on the scene file's own map (truth); step 2e replaces it with frontier exploration on the
perceived map."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from locomani.primitives.navigator import UNREACHABLE

# head scan at a viewpoint: (pan, tilt) in degrees, tilt measured down from the horizon like the mount's 52; each pose
# is held `scan_dwell_s`
SCAN = ((0, 52), (-35, 52), (35, 52), (35, 30), (0, 30), (-35, 30), (-35, 10), (0, 10), (35, 10))
MOUNT = (0, 52)  # the head camera's own pose while walking


@dataclass(frozen=True)
class TourConfig:
    room: str = "living_dining_kitchen"
    top_m: tuple[float, float] = (0.3, 1.3)  # surface tops the tour must see
    min_area_m2: float = 0.02
    view_m: tuple[float, float] = (0.6, 1.5)  # robot to the surface's nearest point
    best_view_m: float = 1.0
    grid_m: float = 0.25  # candidate spacing
    scan: tuple[tuple[int, int], ...] = SCAN
    scan_dwell_s: float = 0.4
    must: tuple[str, ...] = field(default=())  # surfaces that must be covered (a family's targets)


@dataclass
class Viewpoint:
    xy: np.ndarray
    yaw: float
    surfaces: list[str]


def surfaces_of(meta: dict, cfg: TourConfig) -> dict[str, dict]:
    """The room's surfaces the tour has to see."""
    out = {}
    for k, s in meta.get("surfaces", {}).items():
        if s.get("half_extent") is None or s.get("room") != cfg.room:
            continue
        if not cfg.top_m[0] <= float(s["top"]) <= cfg.top_m[1] and k not in cfg.must:
            continue
        if float(s.get("area", 4 * s["half_extent"][0] * s["half_extent"][1])) < cfg.min_area_m2 and k not in cfg.must:
            continue
        out[k] = s
    return out


def nearest_on_box(xy, s: dict) -> np.ndarray:
    """The point of surface `s`'s footprint (an oriented rectangle) nearest to `xy`."""
    c, he, yaw = np.asarray(s["xy"], float), np.asarray(s["half_extent"], float), float(s.get("yaw", 0.0))
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    local = np.clip(R.T @ (np.asarray(xy, float) - c), -he, he)
    return c + R @ local


def in_box(xy, s: dict, pad: float) -> bool:
    c, he, yaw = np.asarray(s["xy"], float), np.asarray(s["half_extent"], float), float(s.get("yaw", 0.0))
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return bool(np.all(np.abs(R.T @ (np.asarray(xy, float) - c)) <= he + pad))


def line_of_sight(occ, a, b, s: dict, pad: float = 0.12) -> bool:
    """No occupied cell between `a` and the surface point `b`, except inside the surface's own footprint (+pad)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / (occ.res / 2))) + 1)
    for u in np.linspace(0.0, 1.0, n):
        p = a + u * (b - a)
        if in_box(p, s, pad):
            continue
        c = occ.cell_of(p)
        if not occ.in_bounds(c) or occ.occupied[c[0], c[1]]:
            return False
    return True


def candidates(occ, nav, start_xy, cfg: TourConfig) -> np.ndarray:
    """Free cells on a `grid_m` lattice that the robot can walk to from `start_xy`."""
    if not nav.set_goal_point(start_xy):
        raise ValueError(f"the start {start_xy} has no legal cell near it")
    step = max(1, round(cfg.grid_m / occ.res))
    out = []
    for i in range(0, occ.nx, step):
        for j in range(0, occ.ny, step):
            if occ.free[i, j] and nav.dist[i, j] < UNREACHABLE * 0.5:
                out.append(occ.xy_of(np.array([i, j])))
    return np.array(out)


def coverage(occ, cands: np.ndarray, surfaces: dict, cfg: TourConfig) -> np.ndarray:
    """score[c, s] > 0 when candidate c sees surface s: 1 at `best_view_m`, falling to 0.2 at the range's ends."""
    keys = list(surfaces)
    score = np.zeros((len(cands), len(keys)))
    lo, hi = cfg.view_m
    for si, k in enumerate(keys):
        s = surfaces[k]
        for ci, c in enumerate(cands):
            p = nearest_on_box(c, s)
            d = float(np.linalg.norm(p - c))
            if lo <= d <= hi and line_of_sight(occ, c, p, s):
                score[ci, si] = 1.0 - 0.8 * abs(d - cfg.best_view_m) / max(hi - cfg.best_view_m, cfg.best_view_m - lo)
    return score


def choose(cands: np.ndarray, score: np.ndarray, keys: list[str], cfg: TourConfig) -> tuple[list[int], list[str]]:
    """Greedy set cover: the candidate that adds the most uncovered score (must-surfaces weigh 3), until every surface
    some candidate sees is covered. Returns (chosen candidate indices, surfaces no candidate sees)."""
    weight = np.array([3.0 if k in cfg.must else 1.0 for k in keys])
    seen = score.max(axis=0) > 0
    left, chosen = seen.copy(), []
    while left.any():
        gain = (score * (weight * left)).sum(axis=1)
        best = int(np.argmax(gain))
        if gain[best] <= 0:
            break
        chosen.append(best)
        left &= ~(score[best] > 0)
    return chosen, [k for k, ok in zip(keys, seen, strict=True) if not ok]


def facing(xy, surfaces: list[dict]) -> float:
    """Heading toward the mean of the covered surfaces' nearest points."""
    v = np.mean([nearest_on_box(xy, s) - xy for s in surfaces], axis=0)
    return float(np.arctan2(v[1], v[0]))


def order(nav, start_xy, points: list[np.ndarray]) -> list[int]:
    """A short closed route start -> every point -> start on geodesic distances: nearest neighbour, then 2-opt."""
    pts = [np.asarray(start_xy, float)] + [np.asarray(p, float) for p in points]
    n = len(pts)
    D = np.zeros((n, n))
    for i, p in enumerate(pts):
        nav.set_goal_point(p)
        for j, q in enumerate(pts):
            D[i, j] = nav.distance_at(q)
    D = np.minimum(D, D.T)
    route, todo = [0], set(range(1, n))
    while todo:
        nxt = min(todo, key=lambda j: D[route[-1], j])
        route.append(nxt)
        todo.remove(nxt)
    route.append(0)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                a, b, c, d = route[i - 1], route[i], route[j], route[j + 1]
                if D[a, c] + D[b, d] < D[a, b] + D[c, d] - 1e-9:
                    route[i : j + 1] = route[i : j + 1][::-1]
                    improved = True
    return [k - 1 for k in route[1:-1]]


def plan(meta: dict, occ, nav, start_xy, cfg: TourConfig | None = None) -> tuple[list[Viewpoint], dict]:
    """The tour's viewpoints in visiting order, and a summary (surfaces, the ones no viewpoint sees, the length)."""
    cfg = cfg or TourConfig()
    surfaces = surfaces_of(meta, cfg)
    keys = list(surfaces)
    cands = candidates(occ, nav, start_xy, cfg)
    score = coverage(occ, cands, surfaces, cfg)
    chosen, unseen = choose(cands, score, keys, cfg)
    # each surface goes to the chosen viewpoint that sees it best
    best = {k: chosen[int(np.argmax(score[chosen, si]))] for si, k in enumerate(keys) if k not in unseen}
    vps = []
    for c in chosen:
        mine = [k for k, b in best.items() if b == c]
        if mine:
            vps.append(Viewpoint(cands[c], facing(cands[c], [surfaces[k] for k in mine]), mine))
    route = order(nav, start_xy, [v.xy for v in vps])
    vps = [vps[k] for k in route]
    legs = [np.asarray(start_xy, float)] + [v.xy for v in vps] + [np.asarray(start_xy, float)]
    length = 0.0
    for a, b in zip(legs[:-1], legs[1:], strict=True):
        nav.set_goal_point(b)
        length += nav.distance_at(a)
    summary = dict(surfaces=keys, unseen=unseen, must_unseen=[k for k in cfg.must if k in unseen],
                   candidates=len(cands), viewpoints=len(vps), route_m=round(length, 2))  # fmt: skip
    return vps, summary
