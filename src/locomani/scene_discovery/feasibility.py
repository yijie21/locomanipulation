"""Sort a task into the three buckets of plan decision 8, check by check, against the measured envelope.

A task is a goal (predicates, `scene_discovery.predicates`) over a v2 scene file. Each predicate expands into the
steps a robot would take -- `in(X, C)`: pick X, carry it to C, put it in -- and every step is checked against
`configs/scene_discovery/capability.yaml`. A check ends one of three ways:

    ok        inside a measured setting
    untested  no hard limit is crossed, but nothing like it was ever run; names a gap (`grasp_single.support_top`)
    blocked   a hard limit is crossed (too big for the jaw, no room to stand, outside the hand's reach); names why

A single-hand grasp is matched against the measured settings whole (`reach_single_arm.grasp_settings`): its checks
come from the setting it misses on fewest dimensions, and they all pass only inside one setting. A grasp at least as
hard as a failed pilot (`grasp_failures`) carries that failure's skill gap. Every measured grasp and put-down had
nothing above it; a spot under another surface's slab closer than `overhead.open_above` (a bookcase or cabinet cell)
is outside them all. Only the extracted surfaces count: they are the furniture's collision geometry in MuJoCo.

Bucket 3 = any blocked check; bucket 2 = any untested (the strict rule the user approved: an extrapolation never
counts as composable); bucket 1 = everything ok. `bucket(..., policy="pilot")` is the alternative where untested
tasks are attempted and the pilot decides (decision 10); both are reported, the choice is the user's.
`gap_ranking` turns bucket-2 tasks into the ranking decision 11 reads: per gap, how many tasks it alone blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from locomani.scene_discovery import predicates
from locomani.scene_discovery.mjcf import SLAB
from locomani.scene_discovery.scene_file import to_surface_frame

OK, UNTESTED, BLOCKED = "ok", "untested", "blocked"
TOL = 0.02  # m; a measured point setting (one table height) counts within this
UPRIGHT_DEG = 15.0
FLOOR_OBSTACLE_TOP = 0.3  # m; a static object whose underside is below this stands on the floor and blocks walking
STAND_OUT = 0.10  # m past the body radius from an edge: where a stand is taken for route checks
COVER_PAD = 0.05  # m round a slab above: about half the hand's width


@dataclass
class Check:
    step: str  # e.g. "grasp_single(Banana_A)"
    name: str  # what was compared, e.g. "support_top"
    outcome: str  # ok | untested | blocked
    measured: object = None
    bound: object = None
    gap: str = ""  # untested: the capability that would have to be measured or built
    why: str = ""  # blocked: the hard limit
    setting: str = ""  # the measured setting the value was compared with (single-hand grasps)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in ("", None)}


@dataclass
class Result:
    task: dict
    checks: list = field(default_factory=list)
    invalid: str = ""  # the goal cannot be compiled, or is already true

    @property
    def gaps(self) -> list[str]:
        return sorted({c.gap for c in self.checks if c.outcome == UNTESTED})

    @property
    def blocked(self) -> list[str]:
        return sorted({f"{c.step}: {c.why}" for c in self.checks if c.outcome == BLOCKED})

    def bucket(self, policy: str = "strict") -> int:
        if self.invalid or self.blocked:
            return 3
        if self.gaps and policy == "strict":
            return 2
        return 1

    def to_dict(self) -> dict:
        return {
            **self.task,
            "bucket": self.bucket("strict"),
            "bucket_pilot_policy": self.bucket("pilot"),
            "gaps": self.gaps,
            "blocked": self.blocked,
            "invalid": self.invalid,
            "checks": [c.to_dict() for c in self.checks],
        }


def _range(value: float, lo_hi, tol: float = TOL) -> bool:
    return lo_hi[0] - tol <= value <= lo_hi[1] + tol


class Checker:
    """One scene file + one capability spec; `check(task)` per task. Routes come from the scene's navigator."""

    def __init__(self, meta: dict, capability: dict, navigator_factory=None):
        self.meta, self.cap = meta, capability
        self.objects, self.surfaces = meta["objects"], meta["surfaces"]
        self.rooms = (meta.get("room") or {}).get("rooms") or {}
        self._navs: dict = {}
        self._navigator_factory = navigator_factory
        self._routes: dict = {}
        # occupancy inflation and stand offset: SONIC's navigator radius in the 10-03 table; the scene runner's 0.35 m
        # in the mars + ScaleBFM table (tasks/scene/world.py BODY_RADIUS_M)
        self.radius = float((capability.get("robot") or {}).get("body_radius", 0.26))

    # -- scene lookups --------------------------------------------------------------------------
    def box(self, key: str):
        b = self.objects[key]["box"]
        return np.asarray(b["half_extent"], dtype=float), np.asarray(b.get("centre", (0, 0, 0)), dtype=float)

    def underside(self, key: str) -> float:
        half, centre = self.box(key)
        return float(self.objects[key]["init_pos"][2] + centre[2] - half[2])

    def top(self, key: str) -> float:
        half, centre = self.box(key)
        return float(self.objects[key]["init_pos"][2] + centre[2] + half[2])

    def xy(self, key: str) -> np.ndarray:
        return np.asarray(self.objects[key]["init_pos"][:2], dtype=float)

    def room_of(self, key: str):
        if key in self.objects:
            return (self.objects[key].get("semantic") or {}).get("room")
        return self.surfaces[key].get("room")

    def edge_stand(self, surface: str, xy) -> tuple[np.ndarray | None, float]:
        """Stand in front of the open edge of `surface` nearest to `xy`, and `xy`'s distance in from that edge."""
        from locomani.tasks.bottle.plan import nearest_edge

        s = self.surfaces[surface]
        if not s.get("approach_edges"):
            return None, float("nan")
        ax, sgn = nearest_edge(xy, s)
        local = to_surface_frame(xy, s)
        half = np.asarray(s["half_extent"], dtype=float)
        inboard = float(half[ax] - sgn * local[ax])
        out = np.zeros(2)
        out[ax] = sgn * (half[ax] + self.radius + STAND_OUT)
        out[1 - ax] = np.clip(local[1 - ax], -half[1 - ax], half[1 - ax])
        c, sn = np.cos(s.get("yaw", 0.0)), np.sin(s.get("yaw", 0.0))
        world = np.asarray(s["xy"], dtype=float) + np.array([c * out[0] - sn * out[1], sn * out[0] + c * out[1]])
        return world, inboard

    def ring_stand(self, key: str) -> tuple[np.ndarray, float]:
        """A floor-standing object (a bin): its centre and a ring radius just outside its footprint."""
        half, _ = self.box(key)
        return self.xy(key), float(np.hypot(*half[:2]) + self.radius + STAND_OUT)

    def headroom(self, surface: str, points) -> float:
        """Least clearance from `surface`'s top up to the underside of another surface's slab over any of `points`
        (world xy); inf when nothing is above them."""
        top, least = self.surfaces[surface]["top"], np.inf
        for key, other in self.surfaces.items():
            gap = float(other["top"]) - SLAB - top
            if key == surface or gap <= 0 or gap >= least:
                continue
            half = np.asarray(other["half_extent"], dtype=float) + COVER_PAD  # the slab's size (mjcf.slab)
            if any(np.all(np.abs(to_surface_frame(p, other)) <= half) for p in points):
                least = gap
        return least

    def spots(self, surface: str, n: int = 5) -> list:
        """An n x n grid over `surface` in world xy: where a put-down could land."""
        s = self.surfaces[surface]
        half = np.asarray(s["half_extent"], dtype=float)
        c, sn = np.cos(s.get("yaw", 0.0)), np.sin(s.get("yaw", 0.0))
        grid = [(u * half[0], v * half[1]) for u in np.linspace(-1, 1, n) for v in np.linspace(-1, 1, n)]
        return [np.asarray(s["xy"], dtype=float) + np.array([c * a - sn * b, sn * a + c * b]) for a, b in grid]

    def _overhead(self, step, surface: str, points, gap: str) -> Check:
        h, bound = self.headroom(surface, points), self.cap["overhead"]["open_above"]
        return Check(step, "headroom", OK if h >= bound else UNTESTED, None if np.isinf(h) else round(h, 3), bound,
                     "" if h >= bound else gap)  # fmt: skip

    # -- routes ---------------------------------------------------------------------------------
    def navigator(self, radius: float | None = None):
        radius = self.radius if radius is None else radius
        if radius not in self._navs:
            if self._navigator_factory is not None:
                self._navs[radius] = self._navigator_factory(self.meta)
            else:
                from locomani.primitives.navigator import Navigator, OccupancyMap

                occ = OccupancyMap(self.meta, body_radius=radius)
                for key, o in self.objects.items():  # static floor objects collide but are not in room.obstacles
                    if not o.get("free") and o.get("surface") is None and self.underside(key) < FLOOR_OBSTACLE_TOP:
                        half, centre = self.box(key)
                        if (o.get("semantic") or {}).get("movable"):
                            occ.add_box(self.xy(key) + centre[:2], half[:2])
                occ.recompute()
                self._navs[radius] = Navigator(occ)
        return self._navs[radius]

    def route(self, start, goal, goal_ring: float | None = None, radius: float | None = None) -> float:
        """Geodesic length start -> goal (point, or ring of that radius around it) for a body of `radius` (default the
        table's); inf when unreachable."""
        key = (tuple(np.round(start, 2)), tuple(np.round(goal, 2)), goal_ring, radius)
        if key not in self._routes:
            nav = self.navigator(radius)
            # here_xy opens the escape corridor the follower uses when it starts inside the inflated region
            ok = (
                nav.set_goal_ring(goal, goal_ring - 0.07, goal_ring + 0.07, here_xy=start)
                if goal_ring
                else nav.set_goal_point(goal, here_xy=start)
            )
            d = nav.distance_at(start) if ok else float("inf")
            self._routes[key] = float("inf") if not np.isfinite(d) or d >= 1e4 else d
        return self._routes[key]

    # -- steps ----------------------------------------------------------------------------------
    def grasp(self, x: str, out: list) -> np.ndarray | None:
        """Pick X from where it rests; returns the stand the pick happens from (None when blocked)."""
        o = self.objects[x]
        sem = o.get("semantic") or {}
        hands = sem.get("graspable", "none")
        step = f"grasp_{'bimanual' if hands == 'bimanual' else 'single'}({x})"
        if not sem.get("movable") or not o.get("free"):
            out.append(
                Check(step, "free_body", BLOCKED, why="fixed in the scene (furniture, or not resting on a surface)")
            )
            return None
        if not (sem.get("drop_test") or {}).get("ok", False):
            out.append(Check(step, "drop_test", BLOCKED, sem.get("drop_test"), why="does not rest still in the sim"))
            return None
        half, _ = self.box(x)
        if hands == "none":
            out.append(
                Check(
                    step,
                    "graspable",
                    BLOCKED,
                    (2 * half[:2]).round(3).tolist(),
                    why="neither the jaw nor the box grip fits",
                )
            )
            return None
        s_key = o.get("surface")
        support = self.surfaces[s_key]["top"]
        stand, inboard = self.edge_stand(s_key, self.xy(x))
        if hands == "single":  # a grasp measured on this very object where it rests: the measurement decides
            for s in self.cap["reach_single_arm"].get("grasp_settings") or ():
                if x in s.get("objects", ()) and s.get("surface") == s_key:
                    out.append(Check(step, "measured_task", OK, s.get("success"), s["name"]))
                    if stand is None and s.get("stance_xy") is not None:
                        stand = np.asarray(s["stance_xy"], dtype=float)
                    return stand
        if stand is None:
            out.append(Check(step, "approach_edge", BLOCKED, s_key, why="no edge of its surface has room to stand"))
            return None
        kind = "single" if hands == "single" else "bimanual"
        out.append(self._overhead(step, s_key, [self.xy(x)], f"grasp_{kind}.covered"))
        if hands == "single":
            c = self.cap["reach_single_arm"]
            grip = float(self.objects[x]["init_pos"][2] + self.box(x)[1][2])  # the bottle line grips at the centre
            lo, hi = c["grip_height_accepted"]
            if not lo <= grip <= hi and c.get("grip_height_blocked_outside"):  # a measured hand envelope, not a skill's
                out.append(Check(step, "grip_height", BLOCKED, round(grip, 3), [lo, hi], why="outside the hand's measured reach"))  # fmt: skip
            elif not lo <= grip <= hi:
                out.append(Check(step, "grip_height", UNTESTED, round(grip, 3), [lo, hi], "grasp_single.grip_height"))
            mass = float(sem.get("mass_kg", 0.0))
            limits = []
            limit = 2 * 20.0 * float(sem.get("friction", 0.5)) / 9.81  # both pads at the 20 N clamp
            if mass > limit:
                limits.append(Check(step, "mass", BLOCKED, round(mass, 3), round(limit, 3), why="heavier than the jaw can hold at its force clamp"))  # fmt: skip
            elif mass > c.get("mass_lifted_max", np.inf):
                limits.append(Check(step, "mass", BLOCKED, round(mass, 3), c["mass_lifted_max"], why="heavier than any mass the hand lifted"))  # fmt: skip
            if inboard > c["inboard_max"]:
                limits.append(Check(step, "inboard", BLOCKED, round(inboard, 3), c["inboard_max"], why="beyond the hand's reach from the edge"))  # fmt: skip
            for f in c.get("grasp_failures") or ():
                if "objects" in f:  # a measured task below the bucket-1 rate: that object, as it rests, is a gap
                    if x in f["objects"]:
                        out.append(Check(step, f["name"], UNTESTED, f.get("success"), f.get("pilots"), f["gap"]))
                    continue
                if inboard >= f["inboard_min"] and grip - support <= f.get("grip_above_support_max", np.inf):
                    out.append(Check(step, f["name"], UNTESTED, [round(inboard, 3), round(grip - support, 3)],
                                     f["pilots"], f["gap"]))  # fmt: skip
            dims = {
                "support_top": support,
                "object_size": (float(2 * min(half[:2])), float(2 * half[2])),
                "mass": mass,
                "inboard": inboard,
            }
            blocked = {k.name for k in limits}
            out.extend(
                limits + [k for k in self._grasp_setting(step, dims, c["grasp_settings"]) if k.name not in blocked]
            )
        elif self.cap["reach_bimanual"]["box_grasp"].get("status") == "untested":
            out.append(Check(step, "skill", UNTESTED, gap="grasp_bimanual.skill"))
        else:
            c = self.cap["reach_bimanual"]["box_grasp"]
            dims = sorted((2 * half[:2]).tolist())
            out.append(self._range(step, "width", dims[1] if c["width"][0] <= dims[1] <= c["width"][1] else dims[0], c["width"], "grasp_bimanual.width"))  # fmt: skip
            out.append(self._max(step, "length", min(dims), c["length_max"], "grasp_bimanual.length"))
            grip = self.top(x) - 0.025
            out.append(self._range(step, "grip_height", grip, c["grip_height"], "grasp_bimanual.grip_height"))
            out.append(self._max(step, "setback", max(inboard - float(half[0]), 0.0), c["setback_from_edge"] + TOL, "grasp_bimanual.setback"))  # fmt: skip
            out.append(
                self._max(
                    step, "mass", float(sem.get("mass_kg", 0.0)), c["mass_measured"], "grasp_bimanual.mass", tol=0.0
                )
            )
        low = self.cap["posture"]["squat"]["hands_lowest"]
        if support < low:
            out.append(
                Check(
                    step,
                    "hand_height",
                    BLOCKED,
                    round(support, 3),
                    low,
                    why="below the lowest hand position ever reached",
                )
            )
        return stand

    def carry(self, x: str, start, target: str, out: list) -> np.ndarray | None:
        """Walk from the pick stand to the stand at `target` (surface or floor object) with X held."""
        hands = (self.objects[x].get("semantic") or {}).get("graspable")
        step = f"carry({x} -> {target})"
        on = target if target in self.surfaces else self.objects[target].get("surface")
        if on is not None:  # a surface, or an object resting on one: stand at that surface's nearest open edge
            at = self.surfaces[on]["xy"] if target in self.surfaces else self.xy(target)
            goal, _ = self.edge_stand(on, at)
            if goal is None:
                out.append(Check(step, "approach_edge", BLOCKED, on, why="no edge of the target has room to stand"))
                return None
            ring = None
            d = self.route(start, goal)
        elif self.underside(target) < FLOOR_OBSTACLE_TOP:  # a floor object (a bin): a ring just outside it
            centre, ring = self.ring_stand(target)
            goal, d = centre, self.route(start, centre, ring)
        else:  # on a wall shelf, a hook, inside a cabinet: nothing to stand at
            out.append(Check(step, "target_support", BLOCKED, round(self.underside(target), 2),
                             why="rests on no surface the robot can stand at and is off the floor"))  # fmt: skip
            return None
        tight = (self.cap.get("robot") or {}).get("body_radius_tight")
        if not np.isfinite(d) and tight is not None and tight < self.radius:  # the planner's margin, not the body
            d_tight = self.route(start, goal, ring, tight)
            if np.isfinite(d_tight):
                out.append(Check(step, "route", UNTESTED, round(d_tight, 2), tight, gap="navigate.narrow_passage"))
                d = d_tight
        if not np.isfinite(d):
            out.append(Check(step, "route", BLOCKED, why="no route in the occupancy map"))
            return None
        hit = (
            self.placed(x, target)[0]
            if target in self.surfaces
            else self.placed(x, target, "into_container_single_arm")[0]
        )
        if hit is not None:  # this very carry was part of a measured put-down or put-in (room change, distance)
            out.append(Check(step, "measured_task", OK, round(d, 2), hit["name"]))
            return goal
        if self.room_of(x) != self.room_of(target):
            out.append(
                Check(step, "room", UNTESTED, [self.room_of(x), self.room_of(target)], gap="navigate.cross_room")
            )
        out.append(
            self._max(step, "route", d, self.cap["locomotion"]["navigation"]["route_max_measured"], "navigate.distance")
        )
        if hands == "bimanual":
            usable = self.cap["carry"]["bimanual_box"]["usable_distance"]
            if d > usable:
                out.append(
                    Check(
                        step, "carry_distance", BLOCKED, round(d, 2), usable, why="the held box slides out within ~1 m"
                    )
                )
        else:
            c = self.cap["carry"]["single_arm"]
            longest = max([*c["straight_or_backward"]["distance"], c.get("route_measured_max") or 0.0])
            out.append(self._max(step, "carry_distance", d, longest, "carry_single.distance"))
        return goal

    def place_into(self, x: str, c: str, out: list) -> None:
        step = f"place({x} in {c})"
        hx, cx = self.box(x)[0], self.box(c)[0]
        wall = float(self.objects[c]["box"].get("wall", 0.01))
        opening = np.sort(2 * (cx[:2] - wall))
        size = np.sort(2 * hx[:2])
        if np.any(size > opening):
            out.append(
                Check(
                    step,
                    "fits",
                    BLOCKED,
                    size.round(3).tolist(),
                    opening.round(3).tolist(),
                    why="does not fit the opening",
                )
            )
            return
        hands = (self.objects[x].get("semantic") or {}).get("graspable")
        if hands == "bimanual":
            out.append(Check(step, "skill", UNTESTED, gap="place_into.bimanual"))
            return
        hit, miss = self.placed(x, c, "into_container_single_arm")
        if hit is not None:  # measured on this object into this container: the measurement decides
            out.append(Check(step, "measured_task", OK, hit.get("success"), hit["name"]))
            return
        if miss is not None:
            out.append(Check(step, miss["name"], UNTESTED, miss.get("success"), miss.get("pilots"), miss["gap"]))
            return
        if self.objects[c].get("surface") is not None:
            out.append(self._overhead(step, self.objects[c]["surface"], [self.xy(c)], "place_into.covered"))
        p = self.cap["place"]["into_container_single_arm"]
        if p.get("status") == "untested":
            out.append(Check(step, "skill", UNTESTED, gap="place_into.single"))
            return
        out.append(self._range(step, "rim_height", self.top(c), p["rim_height_measured"], "place_into.rim_height"))
        measured = np.sort(np.asarray(p["opening_measured"], dtype=float))
        ok = bool(np.all(opening >= measured - TOL))
        out.append(Check(step, "opening", OK if ok else UNTESTED, opening.round(3).tolist(), measured.tolist(), "" if ok else "place_into.opening"))  # fmt: skip
        # the jaw lets the object go holding it at its centre, at least half its height over the rim (as put_on)
        low = self.cap["posture"]["squat"]["hands_lowest"]
        if self.top(c) + hx[2] < low:
            out.append(Check(step, "hand_height", BLOCKED, round(self.top(c) + hx[2], 3), low, why="the hand would let the object go below the lowest hand position ever reached"))  # fmt: skip

    def placed(self, x: str, s: str, kind: str = "on_surface_single") -> tuple[dict | None, dict | None]:
        """(setting, failure) of the single-hand put-down (`kind` "into_container_single_arm": put-in) measured on
        this very object onto this very surface (into this container): an inner loop's sealed result
        (capability_write.py), or None for each."""
        p = (self.cap.get("place") or {}).get(kind) or {}
        hit = [e for e in p.get("measured_tasks") or () if x in e.get("objects", ()) and e.get("surface") == s]
        miss = [e for e in p.get("failed_tasks") or () if x in e.get("objects", ()) and e.get("surface") == s]
        return (hit[0] if hit else None), (miss[0] if miss else None)

    def put_on(self, x: str, s: str, out: list) -> None:
        step = f"put_on({x} on {s})"
        half = np.asarray(self.surfaces[s]["half_extent"], dtype=float)
        if np.any(np.sort(2 * self.box(x)[0][:2]) > np.sort(2 * half) + TOL):
            out.append(Check(step, "fits", BLOCKED, why="the surface is smaller than the object"))
            return
        hands = (self.objects[x].get("semantic") or {}).get("graspable")
        hit, miss = self.placed(x, s) if hands != "bimanual" else (None, None)
        if hit is not None:  # measured on this object onto this surface: the measurement decides (as grasps do)
            out.append(Check(step, "measured_task", OK, hit.get("success"), hit["name"]))
            return
        out.append(self._overhead(step, s, self.spots(s), f"{'put_down_bimanual' if hands == 'bimanual' else 'place_on_surface'}.covered"))  # fmt: skip
        if hands == "bimanual" and self.cap["place"]["put_down_bimanual"].get("status") == "untested":
            out.append(Check(step, "skill", UNTESTED, gap="put_down_bimanual.skill"))
        elif hands == "bimanual":
            p = self.cap["place"]["put_down_bimanual"]
            out.append(
                self._range(
                    step,
                    "support_top",
                    self.surfaces[s]["top"],
                    p["support_top_measured"],
                    "put_down_bimanual.support_top",
                )
            )
        else:  # put_down_single (docs/handoff/put-down-single-20261003.md); without a measurement it is a skill gap
            p = self.cap["place"].get("on_surface_single")
            if miss is not None:
                out.append(Check(step, miss["name"], UNTESTED, miss.get("success"), miss.get("pilots"), miss["gap"]))
                return
            if p is None or p.get("status") == "untested":
                out.append(Check(step, "skill", UNTESTED, gap="place_on_surface.single"))
                return
            top = self.surfaces[s]["top"]
            out.append(self._range(step, "support_top", top, p["support_top_measured"], "place_on_surface.support_top"))
            height = float(2 * self.box(x)[0][2])
            out.append(self._range(step, "object_height", height, p["object_height_measured"], "place_on_surface.object_height", 0.01))  # fmt: skip
            # the jaw sets the object down holding it at its centre, half its height over the top (a pick grips there)
            low = self.cap["posture"]["squat"]["hands_lowest"]
            if top + height / 2 < low:
                out.append(Check(step, "hand_height", BLOCKED, round(top + height / 2, 3), low, why="the hand would set the object down below the lowest hand position ever reached"))  # fmt: skip

    # -- tasks ----------------------------------------------------------------------------------
    def check(self, task: dict) -> Result:
        r = Result(task)
        try:
            predicates.compile_goal(task["goal"], self.meta)
        except ValueError as exc:
            r.invalid = str(exc)
            return r
        # upright(X) beside a move of X is a constraint on that move ("set it down standing"), not a reorientation
        moves = {p.args[0]: p.name for p in map(predicates.parse, task["goal"]) if p.name in ("in", "on", "held")}
        for text in task["goal"]:
            p = predicates.parse(text)
            if p.name in ("in", "on", "held"):
                x = p.args[0]
                stand = self.grasp(x, r.checks)
                if p.name == "held" or stand is None:
                    continue
                target = p.args[1]
                same_surface = p.name == "in" and self.objects[target].get("surface") == self.objects[x].get("surface")
                if p.name == "on" and target == self.objects[x].get("surface"):
                    r.invalid = f"{text}: already on it"
                    return r
                if not same_surface and self.carry(x, stand, target, r.checks) is None:
                    continue
                (self.place_into if p.name == "in" else self.put_on)(x, target, r.checks)
            elif p.name == "upright":
                x = p.args[0]
                q = self.objects[x]["init_quat"]
                axis = (self.objects[x].get("semantic") or {}).get("upright_axis") or (0, 0, 1)
                up = predicates._rotate(q, np.asarray(axis, dtype=float) / np.linalg.norm(axis))
                standing = np.degrees(np.arccos(np.clip(up[2], -1, 1))) < UPRIGHT_DEG
                moved = moves.get(x)
                if not standing:  # nothing measured stands an object up; a put-down keeps the grasp's tilt
                    r.checks.append(Check(f"reorient({x})", "skill", UNTESTED, gap="reorient"))
                elif moved is None:  # standing and nothing moves it: the goal already holds
                    r.invalid = f"{text}: already upright"
                    return r
                elif moved == "in":  # the container release was never checked for tilt
                    r.checks.append(Check(f"place({x})", "keeps_upright", UNTESTED, gap="place_into.upright"))
                else:  # held, or set down by a put-down (put_down_single lands at 0.3-0.4 deg, put_down_bimanual <= 2.5)
                    r.checks.append(Check(f"{moved}({x})", "keeps_upright", OK))
            else:  # near / in_room: the robot (or a carried object) going somewhere
                r.checks.append(Check(f"{p.name}{p.args}", "skill", UNTESTED, gap=f"goal.{p.name}"))
        return r

    def _grasp_setting(self, step, dims: dict, settings: list) -> list[Check]:
        """Checks against the measured setting the grasp misses on fewest dimensions (ties: the first listed)."""
        best = None
        for s in settings:
            width, height = dims["object_size"]
            size_ok = _range(width, s["object_width"], 0.005) and _range(height, s["object_height"], 0.01)
            checks = [
                self._range(step, "support_top", dims["support_top"], s["support_top"], "grasp_single.support_top"),
                Check(
                    step,
                    "object_size",
                    OK if size_ok else UNTESTED,
                    [round(width, 3), round(height, 3)],
                    [s["object_width"], s["object_height"]],
                    "" if size_ok else "grasp_single.object_size",
                ),  # fmt: skip
                self._max(step, "mass", dims["mass"], s["mass_max"], "grasp_single.mass", tol=0.0),
                self._range(step, "inboard", dims["inboard"], s["inboard"], "grasp_single.inboard"),
            ]
            for k in checks:
                k.setting = s["name"]
            misses = sum(k.outcome != OK for k in checks)
            if best is None or misses < best[0]:
                best = (misses, checks)
        return best[1]

    def _range(self, step, name, value, lo_hi, gap, tol: float = TOL) -> Check:
        ok = lo_hi is not None and _range(float(value), lo_hi, tol)  # None: the setting has no such measurement
        bound = None if lo_hi is None else list(lo_hi)
        return Check(step, name, OK if ok else UNTESTED, round(float(value), 3), bound, "" if ok else gap)

    def _max(self, step, name, value, bound, gap, tol: float = TOL) -> Check:
        ok = float(value) <= float(bound) + tol
        return Check(step, name, OK if ok else UNTESTED, round(float(value), 3), bound, "" if ok else gap)


def templates(meta: dict) -> list[dict]:
    """Every in / on / held goal over the scene's graspable objects: the template space, before Claude proposes."""
    out = []
    objects, surfaces = meta["objects"], meta["surfaces"]
    pickable = sorted(
        k
        for k, o in objects.items()
        if o.get("free")
        and (o.get("semantic") or {}).get("graspable") != "none"
        and ((o.get("semantic") or {}).get("drop_test") or {}).get("ok")  # unstable objects are no task's object
    )
    containers = sorted(k for k, o in objects.items() if o.get("container"))
    for x in pickable:
        out.append({"id": f"pick_{x}", "goal": [f"held({x})"]})
        for c in containers:
            if c != x:
                out.append({"id": f"put_{x}_in_{c}", "goal": [f"in({x}, {c})"]})
        for s in sorted(surfaces):
            if s != objects[x].get("surface") and surfaces[s].get("approach_edges"):
                out.append({"id": f"put_{x}_on_{s}", "goal": [f"on({x}, {s})"]})
    return out


def gap_ranking(results: list[Result]) -> list[dict]:
    """Per gap: tasks it blocks alone (closing it moves them to bucket 1), and tasks it blocks with others."""
    alone, shared = {}, {}
    for r in results:
        if r.bucket("strict") != 2:
            continue
        gaps = r.gaps
        for g in gaps:
            shared[g] = shared.get(g, 0) + 1
        if len(gaps) == 1:
            alone[gaps[0]] = alone.get(gaps[0], 0) + 1
    return sorted(
        ({"gap": g, "unlocks_alone": alone.get(g, 0), "in_tasks": n} for g, n in shared.items()),
        key=lambda d: (-d["unlocks_alone"], -d["in_tasks"], d["gap"]),
    )
