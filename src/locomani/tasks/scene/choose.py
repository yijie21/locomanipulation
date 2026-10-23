"""TODO 33.12 (plan section 2.2, D9; user 2026-10-22's four rules, 2026-10-23 "接着做 33.12"): choose the stance, the hand
and the grasp together, scored before anything moves, and fall back in the order a person would.

The user's order: take the narrow side when the object is wider than the jaw; when that side does not face the
shoulder, move the stance so the hand comes in along it; failing that nudge the object so the narrow side faces you;
when the hand cannot reach, stand where the object is nearest -- the other side of the table, the nearest point of
its edge. The skills of the 32.12 line picked a stance from the support's edge normals alone, closed the jaw across
the shoulder-to-object line whatever the object's shape, used the right hand only and never lowered the pelvis.

What this module does, by hand (infrastructure like graspability.py and body.py; the loop calls it and may not change
it, D2 / D9):
1. grasps: every side approach whose section fits the open jaw (graspability.section_widths, APPROACH_STEP_DEG apart,
   with the margin a side), a rim pinch when the container's wall allows one, a top pinch across a flat object's
   narrow axis (approaches within the wrist's twist window of that axis) -- pinches with the wrist straight down and
   tilted 45 degrees both, because the envelope grid says a straight-down wrist at table height is not delivered
   (83-100 % misses at 0.3-0.6 m ahead) while the tilted one is (0 %); the option carries the attitude to command;
2. stances: for each grasp, each hand and each forward distance D_MIN..D_MAX, the pelvis SHOULDER_Y_M to the hand's
   inner side of the approach line, kept STANCE_CLEAR_M from the map's obstacles (the support, other furniture, walls)
   and with a path from where the robot is; other free objects in the hand's corridor count against it;
3. score: the body model's odds of landing >= 3 cm off (body.reach, from the envelope grid), a thin fit margin, a
   blocked corridor, a lowered pelvis, the distance beyond the grid's good band and the walk;
4. fallback order: standing options first; when none predicts fewer misses than P_MISS_OK, the same with the pelvis
   lowered (POSTURES); when none still, the best there is, marked `needs_prep` with graspability's advice (turn the
   object, pull it to the edge: the preparatory actions of TODO 33.5, which the skills write); when no stance is
   clear at all, the same list with the clearance relaxed, marked the same way.
The in-run checks (re-walk when the stop is off, re-approach when the pre-grasp is off) are stage()'s (rules.md 31).

    from locomani.tasks.scene import choose
    best = choose.choose(world, "Mug_2", start_xy)          # an Option, or None when nothing can be listed
    opts = choose.options(world, "Mug_2", start_xy)         # every option, best first
    print(choose.explain(opts[:5]))

`world` is a SceneWorld or a graspability.StaticScene (a navigation map is built for it when it has none). A
prediction is for choosing what to try; only a closed-loop round is evidence (rules.md 7)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from locomani.primitives.navigator import build as build_navigation
from locomani.tasks.scene import body as B
from locomani.tasks.scene import graspability as G
from locomani.tasks.scene.navigate import path as nav_path

BODY_RADIUS_M = 0.35  # world.BODY_RADIUS_M (not imported: world pulls in the simulator)
STANCE_CLEAR_M = 0.15  # pelvis to the map's obstacles, the least a stance may have (the skills stood at 0.17)
STANCE_CLEAR_GOOD_M = 0.20  # ... and what it should have (A800 2026-10-22: <= 0.16 m 3/25 succeeded, >= 0.29 m 13/32)
TIGHT_W = 2.0  # score per metre of clearance short of STANCE_CLEAR_GOOD_M (0.05 m short costs what a 10 % miss does)
STANCE_CLEAR_RELAXED_M = 0.12  # the relaxed listing when nothing stands clear
RAISE_STEP_M, RAISE_MAX_M = 0.04, 0.12  # a tall object is also sectioned this far above the grasp height, in steps
SHOULDER_Y_M = 0.10  # the pelvis this far to the hand's inner side of the approach line (the skills' convention)
D_MIN, D_MAX, D_STEP = 0.30, 0.70, 0.05  # pelvis-to-object forward distances tried
D_MAX_RELAXED = 0.80
D_SWEET_M = 0.50  # beyond this the score grows DIST_W a metre: the grid is thinner and the tasks miss more past it
DIST_W = 0.4
POSTURES = (None, 0.68, 0.62)  # standing, then the pelvis lowered (envelope rows; 0.60 m is the controller's floor)
P_MISS_OK = 0.25  # the fallback order moves on when no option predicts fewer misses than this
APPROACH_STEP_DEG = 15.0
MIN_SECTION_M = 0.005  # a side grasp needs this much object across the jaw at the jaw height
TWIST_MAX_DEG = 60.0  # a top pinch's closing axis may be this far from square to the facing (the skills' twist window)
PINCH_ATTS = ("tilt45", "down")  # pinches are offered tilted and straight down; the body model ranks them
TOP_Z_ABOVE_M = 0.035  # a top pinch's jaw centre over the object's top
RIM_Z_ABOVE_M = 0.02  # a rim pinch's jaw centre over the rim
PRE_BACK_M = 0.12  # the hand's corridor starts this far before the object's near face ...
CORRIDOR_HALF_W_M = 0.06  # ... and is this wide either side of the approach line
CORRIDOR_PENALTY = 0.30  # score added when another free object sits in the corridor
WALK_W = 0.05  # score per metre walked (a 2 m detour round the table costs what a 10 % miss does)
HAND_PREF = {"right": 0.0, "left": 0.03}  # the skills grew right-handed; the left hand needs a reason
MARGIN_W = 2.0  # score per metre of fit margin short of FIT_WANT_M
FIT_WANT_M = 0.010
POSTURE_PENALTY = 0.05  # score for a lowered pelvis (it costs time and the grid rows are thinner)
MAX_OPTIONS = 200


@dataclass
class Option:
    kind: str  # side | rim | top
    hand: str  # left | right
    att: str  # the wrist attitude to command: level | tilt45 | down
    approach_deg: float  # world yaw of the approach (robot toward object), degrees
    stance_xy: tuple  # pelvis xy
    stance_yaw: float  # facing, radians (= the approach)
    distance_m: float  # pelvis to object, forward
    grasp_xyz: tuple  # jaw-centre target, world
    pelvis_z: float | None  # None: standing
    fit_margin_m: float | None  # a side, for side / rim / top
    clear_m: float  # pelvis clearance on the map
    walk_m: float | None  # path length from the start (None: no start given)
    corridor: str | None  # another free object in the hand's corridor
    p_miss: float
    err_cm: float
    inside: bool  # the query lies within the envelope grid
    score: float
    needs_prep: bool = False
    why: list = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["stance_xy"] = [round(float(v), 3) for v in self.stance_xy]
        d["grasp_xyz"] = [round(float(v), 3) for v in self.grasp_xyz]
        for k in ("approach_deg", "stance_yaw", "distance_m", "clear_m", "p_miss", "err_cm", "score"):
            d[k] = round(float(d[k]), 3)
        d["walk_m"] = None if self.walk_m is None else round(float(self.walk_m), 2)
        d["fit_margin_m"] = None if self.fit_margin_m is None else round(float(self.fit_margin_m), 4)
        return d


def _unit(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([math.cos(a), math.sin(a)])


def _left(u: np.ndarray) -> np.ndarray:
    return np.array([-u[1], u[0]])


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def navigation(world):
    """The world's occupancy map and navigator, built once for a StaticScene."""
    if getattr(world, "occ", None) is None or getattr(world, "nav", None) is None:
        world.occ, world.nav = build_navigation(world.meta, body_radius=BODY_RADIUS_M)
    return world.occ, world.nav


FLOOR_BOTTOM_M, FLOOR_TOP_M, FLOOR_MAPPED_FRAC = (
    0.10,
    0.60,
    0.60,
)  # floor_statics (the skills' floor_blocked, TODO 33.16)


def floor_statics(world) -> list:
    """(centre xy, half extent xy) of the low fixed objects the navigation map leaves out -- chairs, stools, low
    cabinets: fixed scene objects with a box whose bottom is at most FLOOR_BOTTOM_M and top at most FLOOR_TOP_M, of
    the inner half of whose footprint the map covers less than FLOOR_MAPPED_FRAC. The robot fell twice in run C
    starting 4 cm from a step stool the map did not have. Built once per world."""
    got = getattr(world, "_choose_floor_statics", None)
    if got is not None:
        return got
    occ = navigation(world)[0]
    out = []
    for key, o in world.meta["objects"].items():
        if o.get("free") or "box" not in o:
            continue
        p, q = world.object_pose(key)
        R = G._quat_mat(q)
        c3 = np.asarray(p, float) + R @ np.asarray(o["box"].get("centre", (0.0, 0.0, 0.0)), float)
        e = np.abs(R) @ np.asarray(o["box"]["half_extent"], float)
        if c3[2] - e[2] > FLOOR_BOTTOM_M or c3[2] + e[2] > FLOOR_TOP_M:
            continue
        lo, hi = occ.clip_cell(occ.cell_of(c3[:2] - e[:2] / 2)), occ.clip_cell(occ.cell_of(c3[:2] + e[:2] / 2))
        if float(occ.occupied[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1].mean()) < FLOOR_MAPPED_FRAC:
            out.append((c3[:2].copy(), e[:2].copy()))
    try:
        world._choose_floor_statics = out
    except AttributeError:
        pass
    return out


def clearance(world, xy) -> float:
    """The pelvis's clearance at `xy`: the map's, and the gap to every low fixed object the map leaves out."""
    occ = navigation(world)[0]
    c = occ.cell_of(xy)
    d = float(occ.clearance[c[0], c[1]]) if occ.in_bounds(c) else 0.0
    xy = np.asarray(xy, float)[:2]
    for centre, e in floor_statics(world):
        d = min(d, float(np.linalg.norm(np.maximum(np.abs(xy - centre) - e, 0.0))))
    return d


def approachable(surf: dict, pel_xy) -> bool:
    """Whether a pelvis at `pel_xy` stands off one of the surface's listed approach edges (the scene file's judgement
    of where the robot can stand: a display unit's back is free on the map but reaching through its panel is not a
    grasp, TODO 27.13); every side when the scene lists none or the object has no surface."""
    edges = surf.get("approach_edges") if surf else None
    if not edges:
        return True
    hx, hy = np.asarray(surf.get("flat_half_extent") or surf["half_extent"], float)[:2]
    yaw = float(surf.get("yaw", 0.0))
    c, s = math.cos(-yaw), math.sin(-yaw)
    d = np.asarray(pel_xy, float)[:2] - np.asarray(surf["xy"], float)
    lx, ly = c * d[0] - s * d[1], s * d[0] + c * d[1]
    name = ("+x" if lx > 0 else "-x") if abs(lx) / hx >= abs(ly) / hy else ("+y" if ly > 0 else "-y")
    return name in edges


def object_top(world, key: str) -> float:
    z = np.concatenate([v[:, 2] for v in G.world_parts(world, key)])
    return float(z.max()) if len(z) else float(world.object_pose(key)[0][2])


def grasps(world, key: str) -> list[dict]:
    """The grasp candidates: kind, approach yaw (deg), attitude for the body model, jaw height, fit margin, a note."""
    out = []
    z0 = G.grasp_height(world, key)
    top = object_top(world, key)
    heights = [z0] + [z0 + k * RAISE_STEP_M for k in range(1, int(RAISE_MAX_M / RAISE_STEP_M) + 1)
                      if z0 + k * RAISE_STEP_M <= top - 0.03]  # fmt: skip
    side: dict[float, dict] = {}  # per approach, the height with the widest margin
    for z in heights:
        for d, w in G.section_widths(world, key, z, step_deg=APPROACH_STEP_DEG).items():
            v, margin = G.verdict(w)
            if v == "fits" and w >= MIN_SECTION_M and (float(d) not in side or margin > side[float(d)]["margin_m"]):
                raised = f", jaw raised {100 * (z - z0):.0f} cm" if z > z0 else ""
                side[float(d)] = dict(kind="side", approach_deg=float(d), att="level", z=z, margin_m=float(margin),
                                      note=f"side grasp, {w * 100:.1f} cm across the jaw{raised}")  # fmt: skip
    out += [side[d] for d in sorted(side)]
    rm = G.rim(world, key)
    if rm and rm.get("available"):
        for att in PINCH_ATTS:
            for d in np.arange(-180.0, 180.0, 45.0):
                out.append(dict(kind="rim", approach_deg=float(d), att=att, z=top + RIM_Z_ABOVE_M,
                                margin_m=float(rm["margin_m"]), note=f"{rm['reason']}, wrist {att}"))  # fmt: skip
    tp = G.top_pinch(world, key)
    if tp and tp["fits"]:
        square = tp["closing_deg"] + 90.0  # the facing square to the closing axis
        for att in PINCH_ATTS:
            for base in (square, square + 180.0):
                for off in np.arange(-TWIST_MAX_DEG, TWIST_MAX_DEG + 1e-9, 30.0):
                    out.append(dict(kind="top", approach_deg=_wrap(base + off), att=att, z=top + TOP_Z_ABOVE_M,
                                    margin_m=float(tp["margin_m"]), note=f"{tp['reason']}, wrist {att}"))  # fmt: skip
    return out


def corridor_blocker(world, key: str, u: np.ndarray, z: float) -> str | None:
    """Another free object whose box meets the strip the open hand sweeps along `u` into the object at height `z`."""
    p, _ = world.object_pose(key)
    p = np.asarray(p, float)
    half = np.asarray(world.meta["objects"][key]["box"]["half_extent"], float)
    near = float(np.hypot(*half[:2]))  # a bound on the object's own near face
    left = _left(u)
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
        if -(near + PRE_BACK_M + r) <= along <= -max(near - r, 0.0) and abs(lat) <= CORRIDOR_HALF_W_M + r:
            return other
    return None


def _score(hand: str, p_miss: float, d: float, margin: float, blocker, pel_z, walk: float | None, cl: float = 1.0) -> float:  # fmt: skip
    s = p_miss + MARGIN_W * max(0.0, FIT_WANT_M - margin) + DIST_W * max(0.0, d - D_SWEET_M) + HAND_PREF[hand]
    s += TIGHT_W * max(0.0, STANCE_CLEAR_GOOD_M - cl)
    s += CORRIDOR_PENALTY if blocker else 0.0
    s += POSTURE_PENALTY if pel_z is not None else 0.0
    return s + WALK_W * (walk or 0.0)


def options(world, key: str, start_xy=None, hands=("right", "left"), postures=POSTURES, table: str | None = None,
            limit: int = MAX_OPTIONS, relax: bool = False) -> list[Option]:  # fmt: skip
    """Every (grasp, hand, posture) combination at its best stance distance, standing clear and reachable, best first
    (score ascending). `relax` lowers the clearance and lengthens the reach for the diagnosis of a hopeless spot."""
    occ, nav = navigation(world)
    clear_min = STANCE_CLEAR_RELAXED_M if relax else STANCE_CLEAR_M
    d_max = D_MAX_RELAXED if relax else D_MAX
    p, _ = world.object_pose(key)
    p = np.asarray(p, float)
    surf = world.meta["surfaces"].get(world.meta["objects"][key].get("surface")) or {}
    out: list[Option] = []
    for g in grasps(world, key):
        u = _unit(g["approach_deg"])
        blocker = corridor_blocker(world, key, u, g["z"])
        for hand in hands:
            sign = 1.0 if hand == "right" else -1.0
            for pel_z in postures:
                cands = []
                for d in np.arange(D_MIN, d_max + 1e-9, D_STEP):
                    pel = p[:2] - d * u + sign * SHOULDER_Y_M * _left(u)
                    if not approachable(surf, pel):
                        continue
                    cl = clearance(world, pel)
                    if cl < clear_min:
                        continue
                    r = B.reach(float(d), SHOULDER_Y_M, float(g["z"]), g["att"], pel_z, hand, path=table)
                    cands.append(
                        (
                            _score(hand, r["p_miss"], float(d), g["margin_m"], blocker, pel_z, None, cl),
                            float(d),
                            pel,
                            cl,
                            r,
                        )
                    )
                for _, d, pel, cl, r in sorted(cands, key=lambda c: (c[0], c[1])):
                    walk = None
                    if start_xy is not None:
                        pts = nav_path(occ, nav, start_xy, pel)
                        if pts is None:
                            continue
                        walk = float(sum(np.linalg.norm(b - a) for a, b in zip(pts[:-1], pts[1:])))
                    why = [g["note"], f"body model: {r['p_miss']:.0%} odds of >= 3 cm off, about {r['err_cm']:.1f} cm"]
                    if blocker:
                        why.append(f"{blocker} in the hand's corridor")
                    if not r["inside"]:
                        why.append("outside the measured envelope: the odds are a guess")
                    if cl < STANCE_CLEAR_GOOD_M:
                        why.append(f"a tight stance, {cl:.2f} m from the furniture")
                    if relax:
                        why.append(f"clearance relaxed to {clear_min:.2f} m, reach to {d_max:.2f} m: a guess")
                    out.append(Option(g["kind"], hand, g["att"], g["approach_deg"], (float(pel[0]), float(pel[1])),
                                      math.radians(g["approach_deg"]), d, (float(p[0]), float(p[1]), float(g["z"])),
                                      pel_z, g["margin_m"], cl, walk, blocker, r["p_miss"], r["err_cm"], r["inside"],
                                      _score(hand, r["p_miss"], d, g["margin_m"], blocker, pel_z, walk, cl), why=why))  # fmt: skip
                    break
    out.sort(key=lambda o: (o.score, o.distance_m))
    return out[:limit]


def choose(world, key: str, start_xy=None, **kw) -> Option | None:
    """The option to try first, by the fallback order: standing with odds of a miss under P_MISS_OK, else a lowered
    pelvis under it, else the best there is marked `needs_prep` with graspability's advice; with no clear stance at
    all, the relaxed listing's best, marked the same way. None when nothing can be listed even relaxed."""
    opts = options(world, key, start_xy, **kw)
    relaxed = False
    if not opts:
        opts = options(world, key, start_xy, relax=True, **kw)
        relaxed = True
    if not opts:
        return None
    if not relaxed:
        for lowered in (False, True):
            tier = [o for o in opts if (o.pelvis_z is not None) == lowered and o.p_miss <= P_MISS_OK
                    and o.corridor is None]  # fmt: skip
            if tier:
                return tier[0]
    best = opts[0]
    best.needs_prep = True
    reason = ("no clear stance within reach" if relaxed
              else f"{best.corridor} stands in the hand's corridor from every stance that delivers" if best.corridor
              else "no stance predicts a delivered reach")  # fmt: skip
    best.why = best.why + [reason, "a preparatory action first"] + (G.assess(world, key).get("recommend") or [])
    return best


def explain(opts: list[Option]) -> str:
    lines = []
    for i, o in enumerate(opts):
        pz = "standing" if o.pelvis_z is None else f"pelvis {o.pelvis_z:.2f} m"
        lines.append(f"{i + 1}. {o.kind} grasp, {o.hand} hand, wrist {o.att}, approach {o.approach_deg:+.0f} deg, "
                     f"{o.distance_m:.2f} m ahead, {pz}: miss odds {o.p_miss:.0%}, err {o.err_cm:.1f} cm, clear "
                     f"{o.clear_m:.2f} m, walk {'-' if o.walk_m is None else f'{o.walk_m:.1f} m'}, score {o.score:.2f}"
                     f"{' NEEDS PREP' if o.needs_prep else ''} -- " + "; ".join(o.why))  # fmt: skip
    return "\n".join(lines)
