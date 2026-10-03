"""Sort a task into the three buckets of plan decision 8, check by check, against the measured envelope.

A task is a goal (predicates, `scene_discovery.predicates`) over a v2 scene file. Each predicate expands into the
steps a robot would take -- `in(X, C)`: pick X, carry it to C, put it in -- and every step is checked against
`configs/scene_discovery/capability.yaml`. A check ends one of three ways:

    ok        inside a measured setting
    untested  no hard limit is crossed, but nothing like it was ever run; names a gap (`grasp_single.support_top`)
    blocked   a hard limit is crossed (too big for the jaw, no room to stand, outside the hand's reach); names why

Bucket 3 = any blocked check; bucket 2 = any untested (the strict rule the user approved: an extrapolation never
counts as composable); bucket 1 = everything ok. `bucket(..., policy="pilot")` is the alternative where untested
tasks are attempted and the pilot decides (decision 10); both are reported, the choice is the user's.
`gap_ranking` turns bucket-2 tasks into the ranking decision 11 reads: per gap, how many tasks it alone blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from locomani.scene_discovery import predicates
from locomani.scene_discovery.scene_file import to_surface_frame

OK, UNTESTED, BLOCKED = "ok", "untested", "blocked"
TOL = 0.02  # m; a measured point setting (one table height) counts within this
UPRIGHT_DEG = 15.0
FLOOR_OBSTACLE_TOP = 0.3  # m; a static object whose underside is below this stands on the floor and blocks walking
STAND_OUT = 0.10  # m past the body radius from an edge: where a stand is taken for route checks


@dataclass
class Check:
    step: str  # e.g. "grasp_single(Banana_A)"
    name: str  # what was compared, e.g. "support_top"
    outcome: str  # ok | untested | blocked
    measured: object = None
    bound: object = None
    gap: str = ""  # untested: the capability that would have to be measured or built
    why: str = ""  # blocked: the hard limit

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
        self._nav = None
        self._navigator_factory = navigator_factory
        self._routes: dict = {}

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
        out[ax] = sgn * (half[ax] + 0.26 + STAND_OUT)
        out[1 - ax] = np.clip(local[1 - ax], -half[1 - ax], half[1 - ax])
        c, sn = np.cos(s.get("yaw", 0.0)), np.sin(s.get("yaw", 0.0))
        world = np.asarray(s["xy"], dtype=float) + np.array([c * out[0] - sn * out[1], sn * out[0] + c * out[1]])
        return world, inboard

    def ring_stand(self, key: str) -> tuple[np.ndarray, float]:
        """A floor-standing object (a bin): its centre and a ring radius just outside its footprint."""
        half, _ = self.box(key)
        return self.xy(key), float(np.hypot(*half[:2]) + 0.26 + STAND_OUT)

    # -- routes ---------------------------------------------------------------------------------
    def navigator(self):
        if self._nav is None:
            if self._navigator_factory is not None:
                self._nav = self._navigator_factory(self.meta)
            else:
                from locomani.primitives.navigator import Navigator, OccupancyMap

                occ = OccupancyMap(self.meta)
                for key, o in self.objects.items():  # static floor objects collide but are not in room.obstacles
                    if not o.get("free") and o.get("surface") is None and self.underside(key) < FLOOR_OBSTACLE_TOP:
                        half, centre = self.box(key)
                        if (o.get("semantic") or {}).get("movable"):
                            occ.add_box(self.xy(key) + centre[:2], half[:2])
                occ.recompute()
                self._nav = Navigator(occ)
        return self._nav

    def route(self, start, goal, goal_ring: float | None = None) -> float:
        """Geodesic length start -> goal (point, or ring of that radius around it); inf when unreachable."""
        key = (tuple(np.round(start, 2)), tuple(np.round(goal, 2)), goal_ring)
        if key not in self._routes:
            nav = self.navigator()
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
        if stand is None:
            out.append(Check(step, "approach_edge", BLOCKED, s_key, why="no edge of its surface has room to stand"))
            return None
        if hands == "single":
            c = self.cap["reach_single_arm"]
            grip = float(self.objects[x]["init_pos"][2] + self.box(x)[1][2])  # the bottle line grips at the centre
            lo, hi = c["grip_height_accepted"]
            if not lo <= grip <= hi:
                out.append(Check(step, "grip_height", UNTESTED, round(grip, 3), [lo, hi], "grasp_single.grip_height"))
            width, height = float(2 * min(half[:2])), float(2 * half[2])
            out.append(self._range(step, "support_top", support, c["support_top_measured"], "grasp_single.support_top"))
            size_ok = _range(width, c["object_width_measured"], 0.005) and _range(
                height, c["object_height_measured"], 0.01
            )
            out.append(Check(step, "object_size", OK if size_ok else UNTESTED, [round(width, 3), round(height, 3)],
                             [c["object_width_measured"], c["object_height_measured"]], "" if size_ok else "grasp_single.object_size"))  # fmt: skip
            mass = float(sem.get("mass_kg", 0.0))
            limit = 2 * 20.0 * float(sem.get("friction", 0.5)) / 9.81  # both pads at the 20 N clamp
            if mass > limit:
                out.append(Check(step, "mass", BLOCKED, round(mass, 3), round(limit, 3), why="heavier than the jaw can hold at its force clamp"))  # fmt: skip
            else:
                out.append(self._max(step, "mass", mass, c["object_mass_measured"], "grasp_single.mass", tol=0.0))
            if inboard > c["inboard_max"]:
                out.append(Check(step, "inboard", BLOCKED, round(inboard, 3), c["inboard_max"], why="beyond the hand's reach from the edge"))  # fmt: skip
            else:
                out.append(self._range(step, "inboard", inboard, c["inboard_measured"], "grasp_single.inboard"))
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
            d = self.route(start, goal)
        elif self.underside(target) < FLOOR_OBSTACLE_TOP:  # a floor object (a bin): a ring just outside it
            centre, ring = self.ring_stand(target)
            goal, d = centre, self.route(start, centre, ring)
        else:  # on a wall shelf, a hook, inside a cabinet: nothing to stand at
            out.append(Check(step, "target_support", BLOCKED, round(self.underside(target), 2),
                             why="rests on no surface the robot can stand at and is off the floor"))  # fmt: skip
            return None
        if not np.isfinite(d):
            out.append(Check(step, "route", BLOCKED, why="no route in the occupancy map"))
            return None
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
            out.append(self._max(step, "carry_distance", d, max(self.cap["carry"]["single_arm"]["straight_or_backward"]["distance"]),
                                 "carry_single.distance"))  # fmt: skip
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
        p = self.cap["place"]["into_container_single_arm"]
        out.append(self._range(step, "rim_height", self.top(c), p["rim_height_measured"], "place_into.rim_height"))
        measured = np.sort(np.asarray(p["opening_measured"], dtype=float))
        ok = bool(np.all(opening >= measured - TOL))
        out.append(Check(step, "opening", OK if ok else UNTESTED, opening.round(3).tolist(), measured.tolist(), "" if ok else "place_into.opening"))  # fmt: skip
        low = self.cap["posture"]["squat"]["hands_lowest"]
        if self.top(c) < low:
            out.append(Check(step, "hand_height", BLOCKED, round(self.top(c), 3), low, why="the rim is below the lowest hand position ever reached"))  # fmt: skip

    def put_on(self, x: str, s: str, out: list) -> None:
        step = f"put_on({x} on {s})"
        half = np.asarray(self.surfaces[s]["half_extent"], dtype=float)
        if np.any(np.sort(2 * self.box(x)[0][:2]) > np.sort(2 * half) + TOL):
            out.append(Check(step, "fits", BLOCKED, why="the surface is smaller than the object"))
            return
        hands = (self.objects[x].get("semantic") or {}).get("graspable")
        if hands == "bimanual":
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
        else:
            out.append(
                Check(step, "skill", UNTESTED, gap="place_on_surface.single")
            )  # `place` only puts into containers

    # -- tasks ----------------------------------------------------------------------------------
    def check(self, task: dict) -> Result:
        r = Result(task)
        try:
            predicates.compile_goal(task["goal"], self.meta)
        except ValueError as exc:
            r.invalid = str(exc)
            return r
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
                q = self.objects[p.args[0]]["init_quat"]
                axis = (self.objects[p.args[0]].get("semantic") or {}).get("upright_axis") or (0, 0, 1)
                up = predicates._rotate(q, np.asarray(axis, dtype=float) / np.linalg.norm(axis))
                if np.degrees(np.arccos(np.clip(up[2], -1, 1))) < UPRIGHT_DEG:
                    r.invalid = f"{text}: already upright"
                    return r
                r.checks.append(Check(f"reorient({p.args[0]})", "skill", UNTESTED, gap="reorient"))
            else:  # near / in_room: the robot (or a carried object) going somewhere
                r.checks.append(Check(f"{p.name}{p.args}", "skill", UNTESTED, gap=f"goal.{p.name}"))
        return r

    def _range(self, step, name, value, lo_hi, gap, tol: float = TOL) -> Check:
        ok = _range(float(value), lo_hi, tol)
        return Check(step, name, OK if ok else UNTESTED, round(float(value), 3), list(lo_hi), "" if ok else gap)

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
