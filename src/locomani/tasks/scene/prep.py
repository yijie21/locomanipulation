"""TODO 33.5 (plan section 2, D2, D5, D9; user 2026-10-23 "先做 33.5"): the preparatory-action layer -- what a skill does
when the chooser finds no stance from which the hand is delivered: turn the object so a side that fits faces a stance,
pull or push it across its support toward an edge, move a neighbour out of the hand's corridor. The user's order
(2026-10-22): nudge the object so the narrow side faces you; stand where it is nearest. This module is the interface
and the selector, written by hand once (D2); the actions themselves -- the motions that turn, pull and clear -- are
skills the loop writes and registers here.

    from locomani.tasks.scene import prep
    plans = prep.propose(world, key, start_xy)      # every goal that would give the chooser a standing option, best first
    plan = prep.select(world, key, start_xy)        # the first one an action implements, or None
    prep.wanted(plans)                               # the kinds that would help but no action implements: a gap to report

    @prep.register("turn")                           # in the skills package: the loop's implementation
    def turn(ctx, plan: prep.PrepPlan) -> Outcome: ...

A plan is a goal for the object (a pose it should end in) plus the chooser's option after it: the selector predicts
by placing the object at the goal pose in a view of the scene and asking the chooser again, so a plan is listed only
when, after it, some standing stance has odds of a miss under choose.P_MISS_OK. The action's job is to bring the
object within the goal's tolerance; the skill then re-chooses on the measured state (the prediction is for choosing
what to try; only the round is evidence, rules.md 7). Goals keep the object on its support (footprint inside the flat
top, SURFACE_MARGIN_M in) and clear of the other objects there."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from locomani.tasks.scene import choose as C
from locomani.tasks.scene import graspability as G
from locomani.tasks.scene.start import SURFACE_MARGIN_M, footprint, inside_top, overlap

KINDS = ("turn", "pull", "clear")
TURN_STEP_DEG, TURN_MAX_DEG = 30.0, 90.0  # turns tried about the vertical, smallest first, either way
PULL_STEP_M, PULL_MAX_M = 0.10, 0.50  # pulls tried toward each approach edge
EDGE_IN_M = 0.12  # a pulled object stops this far inside the edge
CLEAR_ASIDE_M = (0.10, 0.15, 0.20)  # a neighbour is moved this far out of the corridor, either way ...
CLEAR_BACK_M = (0.15, 0.20)  # ... or this far along the approach, behind the object
SECOND_STEP = 0.30  # score for a clear that leaves one more neighbour to move
COST = {"turn": 0.10, "pull": 0.15, "clear": 0.20}  # the user's order, as score: turn before pull before clear
TURN_W = 0.10 / 90.0  # score per degree turned
PULL_W = 0.3  # score per metre moved
TOL_TURN_DEG = 10.0  # what counts as the goal met, for the action and the check after it
TOL_MOVE_M = 0.03
NORMALS = {"-x": (-1.0, 0.0), "+x": (1.0, 0.0), "-y": (0.0, -1.0), "+y": (0.0, 1.0)}  # surface frame


@dataclass
class PrepGoal:
    kind: str  # turn | pull | clear
    key: str  # the object to move: the task object, or the neighbour for "clear"
    pos: tuple  # its target position, world xyz
    quat: tuple  # its target orientation, world wxyz
    turn_deg: float = 0.0  # turn: about the vertical, + counter-clockwise seen from above
    move_m: float = 0.0  # pull / clear: how far
    direction_deg: float | None = None  # pull / clear: the world yaw it moves along
    tol_deg: float = TOL_TURN_DEG
    tol_m: float = TOL_MOVE_M
    note: str = ""


@dataclass
class PrepPlan:
    goal: PrepGoal
    after: dict  # the chooser's best standing option once the goal is met (choose.Option.as_dict)
    before_p: float | None  # the chooser's odds of a miss before (None: it listed nothing)
    score: float  # the option's score after, plus the action's cost
    implemented: bool  # an action for this kind is registered
    remaining: str | None = None  # a clear that leaves this neighbour in the corridor for a second clear
    why: list = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["goal"]["pos"] = [round(float(v), 3) for v in self.goal.pos]
        d["goal"]["quat"] = [round(float(v), 4) for v in self.goal.quat]
        d["score"] = round(float(self.score), 3)
        return d


ACTIONS: dict = {}  # kind -> run(ctx, plan) -> Outcome, registered by the skills package


def register(kind: str):
    """Decorator: the skills package's implementation of a prep action kind."""
    if kind not in KINDS:
        raise KeyError(f"prep action kind {kind!r}: one of {KINDS}")

    def deco(fn):
        ACTIONS[kind] = fn
        return fn

    return deco


def implemented() -> list[str]:
    return [k for k in KINDS if k in ACTIONS]


def run(ctx, plan: PrepPlan):
    """The registered action for the plan's kind; KeyError when none is registered."""
    return ACTIONS[plan.goal.kind](ctx, plan)


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def _qmul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])  # fmt: skip


def turned(quat, deg: float) -> np.ndarray:
    """`quat` turned by `deg` about the world vertical."""
    return _qmul(_yaw_quat(math.radians(deg)), np.asarray(quat, float))


def edges(surf: dict) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(name, outward normal world xy, a point on the edge world xy) for the surface's approach edges (all four when
    the scene file lists none)."""
    hx, hy = np.asarray(surf.get("flat_half_extent") or surf["half_extent"], float)[:2]
    yaw = float(surf.get("yaw", 0.0))
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    centre = np.asarray(surf["xy"], float)
    out = []
    for name in surf.get("approach_edges") or list(NORMALS):
        n_local = np.asarray(NORMALS[name], float)
        half = hx if name[1] == "x" else hy
        out.append((name, rot @ n_local, centre + rot @ (n_local * half)))
    return out


class Shifted:
    """A view of a world with some objects moved: `object_pose` answers from `poses`, everything else is the world's.
    The navigation map is shared, so the chooser does not rebuild it."""

    def __init__(self, world, poses: dict):
        self._world, self._poses = world, poses
        self.occ, self.nav = getattr(world, "occ", None), getattr(world, "nav", None)

    def __getattr__(self, name):
        return getattr(self._world, name)

    def object_pose(self, key: str):
        return self._poses[key] if key in self._poses else self._world.object_pose(key)


def on_support(world, key: str, pos, quat) -> bool:
    """The object's footprint at (pos, quat) inside its surface's flat top and clear of the other free objects there."""
    o = world.meta["objects"][key]
    surf = world.meta["surfaces"][o["surface"]]
    flat = np.asarray(surf.get("flat_half_extent") or surf["half_extent"], float)[:2]
    top = (np.asarray(surf["xy"], float), flat - SURFACE_MARGIN_M, float(surf.get("yaw", 0.0)))
    fp = footprint(o, np.asarray(pos, float)[:2], quat)
    if not inside_top(fp, top):
        return False
    for other, oo in world.meta["objects"].items():
        if other == key or oo.get("surface") != o["surface"] or "box" not in oo:
            continue
        p2, q2 = world.object_pose(other)
        if overlap(fp, footprint(oo, np.asarray(p2, float)[:2], q2), 0.0):
            return False
    return True


def blockers(world, key: str, u: np.ndarray, z: float) -> list[str]:
    """Every free object in the hand's corridor along `u` (choose.corridor_blocker's strip), nearest the object first."""
    p, _ = world.object_pose(key)
    p = np.asarray(p, float)
    half = np.asarray(world.meta["objects"][key]["box"]["half_extent"], float)
    near = float(np.hypot(*half[:2]))
    left = C._left(u)
    found = []
    for other, o in world.meta["objects"].items():
        if other == key or not o.get("free") or "box" not in o:
            continue
        q, _ = world.object_pose(other)
        q = np.asarray(q, float)
        hq = np.asarray(o["box"]["half_extent"], float)
        r = float(np.hypot(hq[0], hq[1]))
        if abs(q[2] - z) > hq[2] + 0.05:
            continue
        d = q[:2] - p[:2]
        along, lat = float(d @ u), float(d @ left)
        if -(near + C.PRE_BACK_M + r) <= along <= -max(near - r, 0.0) and abs(lat) <= C.CORRIDOR_HALF_W_M + r:
            found.append((-along, other))
    return [k for _, k in sorted(found)]


def goals(world, key: str, before: C.Option | None) -> list[PrepGoal]:
    """The candidate goals: turns, pulls toward each approach edge, the corridor blocker moved aside."""
    p, q = world.object_pose(key)
    p, q = np.asarray(p, float), np.asarray(q, float)
    out = []
    steps = sorted({d for k in range(1, int(TURN_MAX_DEG / TURN_STEP_DEG) + 1) for d in (k * TURN_STEP_DEG, -k * TURN_STEP_DEG)},
                   key=abs)  # fmt: skip
    for deg in steps:
        out.append(PrepGoal("turn", key, tuple(p), tuple(turned(q, deg)), turn_deg=float(deg),
                            note=f"turn {deg:+.0f} deg about the vertical"))  # fmt: skip
    surf = world.meta["surfaces"][world.meta["objects"][key]["surface"]]
    for name, n, pt in edges(surf):
        along = float((pt - p[:2]) @ n) - EDGE_IN_M  # how far it may move toward this edge
        for s in np.arange(PULL_STEP_M, min(along, PULL_MAX_M) + 1e-9, PULL_STEP_M):
            pos = (float(p[0] + s * n[0]), float(p[1] + s * n[1]), float(p[2]))
            out.append(PrepGoal("pull", key, pos, tuple(q), move_m=float(s),
                                direction_deg=math.degrees(math.atan2(n[1], n[0])),
                                note=f"move {s * 100:.0f} cm toward the {name} edge"))  # fmt: skip
    if before is not None and before.corridor:
        u = C._unit(before.approach_deg)
        left = C._left(u)
        for bk in blockers(world, key, u, float(before.grasp_xyz[2])):
            bp, bq = world.object_pose(bk)
            bp = np.asarray(bp, float)
            moves = [(s * dist, 0.0, f"{dist * 100:.0f} cm aside") for dist in CLEAR_ASIDE_M for s in (1.0, -1.0)]
            moves += [(0.0, dist, f"{dist * 100:.0f} cm back, behind the object") for dist in CLEAR_BACK_M]
            for lat, back, how in moves:
                pos = bp[:2] + lat * left + back * u
                out.append(PrepGoal("clear", bk, (float(pos[0]), float(pos[1]), float(bp[2])), tuple(np.asarray(bq, float)),
                                    move_m=float(math.hypot(lat, back)),
                                    direction_deg=math.degrees(math.atan2(lat * left[1] + back * u[1], lat * left[0] + back * u[0])),
                                    note=f"move {bk} {how}, out of the hand's corridor"))  # fmt: skip
    return out


def propose(world, key: str, start_xy=None, kinds=KINDS, table: str | None = None) -> list[PrepPlan]:
    """Every goal after which the chooser finds a standing option with odds of a miss under P_MISS_OK, best first.
    Empty when the chooser already has one (nothing to prepare) or when no goal helps."""
    before = C.choose(world, key, start_xy, table=table)
    if before is not None and not before.needs_prep:
        return []
    before_p = None if before is None else before.p_miss
    plans = []
    for g in goals(world, key, before):
        if g.kind not in kinds or not on_support(world, g.key, g.pos, g.quat):
            continue
        view = Shifted(world, {g.key: (np.asarray(g.pos, float), np.asarray(g.quat, float))})
        opts = C.options(view, key, start_xy, postures=(None,), table=table)
        good = [o for o in opts if o.p_miss <= C.P_MISS_OK and o.corridor is None]  # a free corridor, delivered
        remaining = None
        if not good and g.kind == "clear":  # one neighbour moved, another still in the way: a second step
            good = [o for o in opts if o.p_miss <= C.P_MISS_OK and o.corridor not in (None, g.key)]
            remaining = good[0].corridor if good else None
        if not good:
            continue
        best = good[0]
        cost = COST[g.kind] + TURN_W * abs(g.turn_deg) + PULL_W * g.move_m + (SECOND_STEP if remaining else 0.0)
        why = [g.note, f"then a {best.kind} grasp, {best.hand} hand, wrist {best.att}, {best.distance_m:.2f} m ahead, "
                       f"miss odds {best.p_miss:.0%}"]  # fmt: skip
        if remaining:
            why.append(f"after that {remaining} still stands in the corridor: a second clear")
        plans.append(PrepPlan(g, best.as_dict(), before_p, best.score + cost, g.kind in ACTIONS, remaining, why=why))
    plans.sort(key=lambda x: x.score)
    return plans


def select(world, key: str, start_xy=None, **kw) -> PrepPlan | None:
    """The first proposed plan whose kind an action implements."""
    return next((x for x in propose(world, key, start_xy, **kw) if x.implemented), None)


def wanted(plans: list[PrepPlan]) -> list[str]:
    """The kinds that would help here but no action implements (the gap to report, in the plans' order)."""
    out = []
    for x in plans:
        if not x.implemented and x.goal.kind not in out:
            out.append(x.goal.kind)
    return out


def met(goal: PrepGoal, pos, quat) -> bool:
    """Whether a measured pose meets the goal within its tolerances (position, and yaw for a turn)."""
    d = float(np.linalg.norm(np.asarray(pos, float)[:2] - np.asarray(goal.pos, float)[:2]))
    if d > goal.tol_m:
        return False
    if goal.kind != "turn":
        return True
    w, x, y, z = (float(v) for v in quat)
    gw, gx, gy, gz = (float(v) for v in goal.quat)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    gyaw = math.atan2(2 * (gw * gz + gx * gy), 1 - 2 * (gy * gy + gz * gz))
    return abs(math.degrees((yaw - gyaw + math.pi) % (2 * math.pi) - math.pi)) <= goal.tol_deg


def explain(plans: list[PrepPlan]) -> str:
    lines = []
    for i, x in enumerate(plans):
        a = x.after
        lines.append(f"{i + 1}. {x.goal.kind}{'' if x.implemented else ' (no action implements it)'}: {x.goal.note}; "
                     f"before {'-' if x.before_p is None else f'{x.before_p:.0%}'} -> after {a['p_miss']:.0%} "
                     f"({a['kind']}, {a['hand']}, wrist {a['att']}, {a['distance_m']:.2f} m); score {x.score:.2f}")  # fmt: skip
    return "\n".join(lines)
