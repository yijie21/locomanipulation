"""Top-down occupancy map + FMM navigation for the scripted collector.

Replaces `orbit_to`. The orbit was a hand-rolled special case — keep the one
obstacle at a constant radius — and it failed in two ways the videos made
obvious: it has no notion of *where the robot's body is*, only where its
pelvis is, and it has no notion of a goal being unreachable, so a stand point
that landed inside the table produced a robot walking into it at full speed.

Four parts, in the order the episode uses them:

  1. rasterise   the surfaces (axis-aligned boxes from scene_meta's
                 `half_extent`) plus the room walls onto a 5 cm grid
  2. inflate     by the robot's body radius, via one distance transform that
                 also yields the clearance field for free
  3. field       fast-marching from a *set* of acceptable standing cells, so
                 one wavefront answers "which of these is nearest, and how do
                 I get there"
  4. follow      per tick, pick the cell on a ring of `step_size` that
                 minimises `dist - w*clearance`, turn towards it, walk

The three things that make this work rather than nearly work, all learned the
hard way elsewhere and worth stating up front:

  * The goal *will* be inside the inflated obstacle. A standing pose 0.45 m
    from an object sitting on a 1 m table is inside 0.30 m of inflation for
    most bearings. So the goal set is filtered by traversability and, if that
    leaves nothing, snapped outward — and the snap distance is added to the
    arrival tolerance, otherwise the follower waits forever at a goal it has
    already reached.
  * An unreachable goal does not error. FMM just leaves the field at its fill
    value everywhere, the gradient is flat, and the robot walks off in a
    straight line indefinitely. Test `dist(robot) < UNREACHABLE` explicitly.
  * The robot's own cell is often occupied — it starts pressed against a
    surface, or a perturbation put it there. Force a 3x3 patch open at the
    robot's cell before marching, or the wavefront never reaches it.
"""

from __future__ import annotations

import base64

import numpy as np
import skfmm
from scipy.ndimage import distance_transform_edt

RESOLUTION = 0.05  # metres per cell
ROOM_XLIM = (-3.0, 4.0)  # the scene spans -1.5 (robot start) to 2.5 (counter)
ROOM_YLIM = (-3.0, 3.0)

# Radius the robot's *body* sweeps while navigating, measured from the pelvis.
# With the arms tucked (primitives.TUCK_WRISTS) it is about 0.25 m.
#
# 0.26 rather than the 0.30 this started at, because the standing ring is now
# the binding constraint rather than the fall rate. Raising the table to the
# arm's working height scaled the asset up with it — the footprint went from
# 0.503 to 0.617 — so a 0.30 inflation puts the ring 0.92 m from the table
# centre, leaving only 0.34 m to an object 0.58 m out. With the drift the robot
# picks up turning, it landed at 0.39-0.58 and approach_A failed 9 times in 10.
BODY_RADIUS = 0.26

STEP_SIZE = 0.45  # how far ahead the follower looks, in metres
CLEARANCE_W = 0.35  # weight on hugging the middle of free space
CLEARANCE_CAP = 0.50  # beyond this much clearance, more does not help
UNREACHABLE = 1e5  # fill value for cells the wavefront never reached


class OccupancyMap:
    """Static top-down map of the scene, plus its C-space inflation."""

    def __init__(
        self,
        meta: dict,
        resolution: float = RESOLUTION,
        xlim=None,
        ylim=None,
        body_radius: float = BODY_RADIUS,
    ):
        # A v2 scene file (scene_discovery) carries its own bounds, walls and furniture footprints; the
        # hand-built scenes keep the fixed ROOM_XLIM / ROOM_YLIM box with walls on its border only.
        room = meta.get("room") or {}
        bounds = room.get("bounds") or {}
        xlim = xlim if xlim is not None else bounds.get("x", ROOM_XLIM)
        ylim = ylim if ylim is not None else bounds.get("y", ROOM_YLIM)
        self.res = resolution
        self.origin = np.array([xlim[0], ylim[0]], dtype=float)
        self.nx = int(np.ceil((xlim[1] - xlim[0]) / resolution))
        self.ny = int(np.ceil((ylim[1] - ylim[0]) / resolution))
        self.body_radius = body_radius

        self.occupied = np.zeros((self.nx, self.ny), dtype=bool)
        # Walls. Not decoration: without them an unreachable goal sends the
        # follower off the edge of the world, and the snap-to-nearest-free
        # fallback can pick a cell outside the room.
        self.occupied[0, :] = self.occupied[-1, :] = True
        self.occupied[:, 0] = self.occupied[:, -1] = True

        for a, b in room.get("walls") or ():
            self.add_segment(a, b)
        for poly in room.get("obstacles") or ():
            self.add_polygon(poly)
        if room.get("blocked"):  # a raster of blocked cells (vision RSI D2's perceived map, scene/mapped.py)
            self.add_raster(room["blocked"])

        self.surfaces = {}
        for key, surf in meta.get("surfaces", {}).items():
            he = surf.get("half_extent")
            if he is None:
                continue
            self.add_box(surf["xy"], he, surf.get("yaw", 0.0))
            self.surfaces[key] = surf

        self.recompute()

    def recompute(self) -> None:
        """Refresh the clearance field after obstacles were added to `occupied`.

        One transform gives both answers: the hard C-space (anything nearer
        than the body radius is illegal) and the soft clearance field the
        follower uses to prefer the middle of a gap over grazing an edge."""
        self.clearance = distance_transform_edt(~self.occupied, sampling=self.res).astype(float)
        self.free = self.clearance >= self.body_radius

    # -- geometry ----------------------------------------------------------

    def add_box(self, centre, half_extent, yaw: float = 0.0) -> None:
        """Mark a footprint occupied: axis-aligned (the hand-built scenes attach surfaces with no yaw, so
        this is exact) or turned by `yaw` (scene_discovery surfaces). Half a cell is added on each side so a
        footprint never lands between samples."""
        c = np.asarray(centre, dtype=float)[:2]
        h = np.asarray(half_extent, dtype=float)[:2] + self.res * 0.5
        if yaw == 0.0:
            lo = self.clip_cell(self.cell_of(c - h))
            hi = self.clip_cell(self.cell_of(c + h))
            self.occupied[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1] = True
            return
        co, si = np.cos(yaw), np.sin(yaw)
        corners = np.array([[-h[0], -h[1]], [h[0], -h[1]], [h[0], h[1]], [-h[0], h[1]]]) @ np.array(
            [[co, si], [-si, co]]
        )
        self.add_polygon(corners + c)

    def add_polygon(self, polygon) -> None:
        """Mark the cells whose centres fall inside a world-xy polygon (even-odd rule)."""
        poly = np.asarray(polygon, dtype=float)[:, :2]
        lo = self.clip_cell(self.cell_of(poly.min(axis=0)))
        hi = self.clip_cell(self.cell_of(poly.max(axis=0)))
        gx, gy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing="ij")
        px, py = self.origin[0] + (gx + 0.5) * self.res, self.origin[1] + (gy + 0.5) * self.res
        inside = np.zeros(gx.shape, dtype=bool)
        for (x0, y0), (x1, y1) in zip(poly, np.roll(poly, -1, axis=0)):
            crosses = (y0 > py) != (y1 > py)
            with np.errstate(divide="ignore", invalid="ignore"):
                x_at = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
            inside ^= crosses & (px < x_at)
        self.occupied[gx[inside], gy[inside]] = True

    def add_segment(self, a, b, thickness: float = 0.10) -> None:
        """Mark a wall segment occupied, `thickness` wide."""
        a, b = np.asarray(a, dtype=float)[:2], np.asarray(b, dtype=float)[:2]
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / (self.res / 2))) + 1)
        pts = a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)
        r = int(np.ceil(thickness / 2 / self.res))
        cells = np.floor((pts - self.origin) / self.res).astype(int)  # cell_of takes one point
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                c = cells + [dx, dy]
                ok = (c[:, 0] >= 0) & (c[:, 0] < self.nx) & (c[:, 1] >= 0) & (c[:, 1] < self.ny)
                self.occupied[c[ok, 0], c[ok, 1]] = True

    def add_raster(self, raster: dict) -> None:
        """Mark the cells under a raster's blocked cells ({origin, res, shape [nx, ny], bits: base64 of
        np.packbits over the [x, y]-ordered mask}), each by its centre."""
        nx, ny = (int(v) for v in raster["shape"])
        mask = np.unpackbits(np.frombuffer(base64.b64decode(raster["bits"]), np.uint8), count=nx * ny)
        ij = np.argwhere(mask.reshape(nx, ny).astype(bool))
        xy = np.asarray(raster["origin"], float) + (ij + 0.5) * float(raster["res"])
        c = np.floor((xy - self.origin) / self.res).astype(int)
        ok = (c[:, 0] >= 0) & (c[:, 0] < self.nx) & (c[:, 1] >= 0) & (c[:, 1] < self.ny)
        self.occupied[c[ok, 0], c[ok, 1]] = True

    def cell_of(self, xy) -> np.ndarray:
        return np.floor((np.asarray(xy, dtype=float)[:2] - self.origin) / self.res).astype(int)

    def clip_cell(self, cell) -> np.ndarray:
        return np.clip(np.asarray(cell, dtype=int), [0, 0], [self.nx - 1, self.ny - 1])

    def xy_of(self, cell) -> np.ndarray:
        return self.origin + (np.asarray(cell, dtype=float) + 0.5) * self.res

    def in_bounds(self, cell) -> bool:
        return bool(0 <= cell[0] < self.nx and 0 <= cell[1] < self.ny)

    def is_free(self, xy) -> bool:
        c = self.cell_of(xy)

        return bool(self.in_bounds(c) and self.free[c[0], c[1]])

    # -- goal sets ---------------------------------------------------------

    def ring_mask(self, centre, r_lo: float, r_hi: float) -> np.ndarray:
        """Free cells whose distance from `centre` is in [r_lo, r_hi]."""
        gx, gy = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="ij")
        wx = self.origin[0] + (gx + 0.5) * self.res
        wy = self.origin[1] + (gy + 0.5) * self.res
        d = np.hypot(wx - centre[0], wy - centre[1])

        return (d >= r_lo) & (d <= r_hi) & self.free

    def nearest_free(self, xy) -> np.ndarray:
        """Cell of the closest free cell to `xy`. Used to snap an illegal goal."""
        gx, gy = np.nonzero(self.free)
        w = self.xy_of(np.stack([gx, gy], axis=1))
        i = int(np.argmin(np.linalg.norm(w - np.asarray(xy, dtype=float)[:2], axis=1)))

        return np.array([gx[i], gy[i]])


class Navigator:
    """A geodesic distance field over the map, and a follower for it."""

    def __init__(self, occ: OccupancyMap):
        self.occ = occ
        self.dist: np.ndarray | None = None
        self.trav: np.ndarray = occ.free.copy()
        self.mask: np.ndarray | None = None
        self.goal_xy: np.ndarray | None = None
        self.snap: float = 0.0

    def replan(self, here_xy) -> bool:
        """Re-march the same goal set with the escape corridor moved.

        The robot drifts while it turns, and it can drift into a cell the
        previous wavefront never reached — its own corridor has moved out from
        under it. Without this the follower reads a flat field and heads off in
        whatever direction the last waypoint pointed.
        """
        if self.mask is None:
            return False

        return self.set_goal_mask(self.mask, here_xy)

    # -- planning ----------------------------------------------------------

    def _open_escape(self, trav: np.ndarray, here_xy) -> None:
        """Carve a corridor from the robot's cell out to free space.

        The robot spends most of the episode *inside* the inflated region —
        that is what standing at a table means — so the wavefront has to be
        able to reach it. A 3x3 patch is not enough: the inflation is 0.30 m
        deep and the patch is 0.15 m across, so it opens a pocket that is
        still walled off. Opening the segment to the nearest free cell is both
        sufficient and minimal: it is the shortest way out, it is the way the
        robot would step anyway, and it never invites the planner to take a
        shortcut through an obstacle because it dead-ends in free space.
        """
        c = self.occ.clip_cell(self.occ.cell_of(here_xy))
        hi_lim = np.array([self.occ.nx - 1, self.occ.ny - 1])
        cells = [c]
        if not self.occ.free[c[0], c[1]]:
            tgt = self.occ.nearest_free(here_xy)
            n = int(max(np.abs(tgt - c).max(), 1))
            cells = [np.round(c + (tgt - c) * k / n).astype(int) for k in range(n + 1)]
        for p in cells:
            lo = np.maximum(p - 1, 0)
            hi = np.minimum(p + 1, hi_lim)
            patch = (slice(lo[0], hi[0] + 1), slice(lo[1], hi[1] + 1))
            # The corridor may cross the inflation band but never the obstacle
            # itself — otherwise the 1-cell dilation punches a hole in the
            # table and the follower cheerfully walks through it.
            trav[patch] |= ~self.occ.occupied[patch]
        # The robot is where it is; refusing to mark its own cell traversable
        # only guarantees the field cannot reach it.
        trav[c[0], c[1]] = True

    def set_goal_mask(self, mask: np.ndarray, here_xy=None) -> bool:
        """Fast-march from every cell in `mask`. False if the mask is empty."""
        mask = mask & self.occ.free
        if not mask.any():
            self.dist = None
            return False
        self.mask = mask

        trav = self.occ.free.copy()
        if here_xy is not None:
            self._open_escape(trav, here_xy)
        self.trav = trav

        phi = np.ma.masked_array(np.ones(trav.shape, dtype=float), ~trav)
        phi[mask] = 0.0
        self.dist = np.ma.filled(skfmm.distance(phi, dx=self.occ.res), UNREACHABLE)

        return True

    def set_goal_ring(self, centre, r_lo: float, r_hi: float, here_xy=None) -> bool:
        """Stand somewhere in an annulus around `centre`, e.g. beside a table.

        A ring rather than a point, because the exact spot does not matter and
        marching from the whole set in one wavefront picks the nearest legal
        one for free. If the annulus is entirely inside the inflated obstacle
        it widens outward — a plate in the middle of a 1 m counter simply
        cannot be stood 0.45 m from.
        """
        centre = np.asarray(centre, dtype=float)[:2]
        self.goal_xy = centre
        self.snap = 0.0
        for grow in (0.0, 0.10, 0.20, 0.35, 0.50):
            mask = self.occ.ring_mask(centre, max(r_lo - grow * 0.5, 0.0), r_hi + grow)
            if mask.any():
                self.snap = grow
                return self.set_goal_mask(mask, here_xy)

        self.dist = None
        return False

    def set_goal_point(self, xy, here_xy=None) -> bool:
        """Go to one spot, snapping to the nearest legal cell if it is illegal.

        `self.snap` records how far the goal moved, so the caller can widen its
        arrival tolerance by the same amount instead of waiting out a timeout
        at a goal it is already standing on.
        """
        xy = np.asarray(xy, dtype=float)[:2]
        self.goal_xy = xy
        cell = self.occ.clip_cell(self.occ.cell_of(xy))
        if not self.occ.free[cell[0], cell[1]]:
            cell = self.occ.nearest_free(xy)
            self.snap = float(np.linalg.norm(self.occ.xy_of(cell) - xy))
        else:
            self.snap = 0.0

        mask = np.zeros_like(self.occ.free)
        mask[cell[0], cell[1]] = True

        return self.set_goal_mask(mask, here_xy)

    # -- following ---------------------------------------------------------

    def distance_at(self, xy) -> float:
        """Geodesic distance from `xy` to the goal set, in metres."""
        if self.dist is None:
            return UNREACHABLE
        c = self.occ.clip_cell(self.occ.cell_of(xy))

        return float(self.dist[c[0], c[1]])

    def reachable(self, xy) -> bool:
        return self.distance_at(xy) < UNREACHABLE * 0.5

    def waypoint(self, xy, step_size: float = STEP_SIZE):
        """The next point to steer at: best cell on a ring of `step_size`.

        Following the field's gradient is what replaces a planner plus a path
        smoother — the wavefront has already done the smoothing. Sampling a
        ring rather than the 8 neighbours keeps the bearing from quantising to
        45 degrees, which at a 5 cm resolution would make the robot weave.

        The clearance term is what keeps it off the corners: the shortest path
        to a spot beside a table grazes the table, and the pelvis has margin
        but the arms do not.
        """
        if self.dist is None:
            return None
        here = np.asarray(xy, dtype=float)[:2]
        best, best_cost = None, float("inf")
        for r in (step_size, step_size * 0.6, step_size * 0.3):
            for a in np.linspace(-np.pi, np.pi, 48, endpoint=False):
                p = here + r * np.array([np.cos(a), np.sin(a)])
                c = self.occ.cell_of(p)
                # `trav`, not `free`: the escape corridor is the only way out
                # when the robot starts inside the inflated region, so the
                # follower has to be allowed to step along it.
                if not self.occ.in_bounds(c) or not self.trav[c[0], c[1]]:
                    continue
                d = self.dist[c[0], c[1]]
                if d >= UNREACHABLE * 0.5:
                    continue
                clear = min(self.occ.clearance[c[0], c[1]], CLEARANCE_CAP)
                cost = d - CLEARANCE_W * clear
                if cost < best_cost:
                    best, best_cost = p, cost
            if best is not None:
                break

        return best


def build(meta: dict, **kw) -> tuple[OccupancyMap, Navigator]:
    occ = OccupancyMap(meta, **kw)

    return occ, Navigator(occ)


def ascii_map(occ: OccupancyMap, marks: dict | None = None) -> str:
    """Render the map for eyeballing: '#' obstacle, ':' inflated, '.' free."""
    grid = np.where(occ.occupied, "#", np.where(occ.free, ".", ":"))
    for ch, xy in (marks or {}).items():
        c = occ.clip_cell(occ.cell_of(xy))
        grid[c[0], c[1]] = ch
    # rows are y descending so the picture reads like a top-down view with +x
    # to the right
    return "\n".join("".join(grid[:, j]) for j in range(occ.ny - 1, -1, -1))


__all__ = ["OccupancyMap", "Navigator", "build", "ascii_map", "RESOLUTION", "BODY_RADIUS", "STEP_SIZE", "UNREACHABLE"]
