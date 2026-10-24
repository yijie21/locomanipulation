"""TODO 33.9 (plan section 2.1, D8; RoboHarness arXiv 2610.14018): grasps written as keypoint constraints.

The 32.12 / grasp_atomic lines grew one geometric rule per object -- side grasp above 3.5 cm, pinch the flat ones from
above, the mug across its wall. Here the person writes, once, the kinds of constraint, the solver and the scoring; a
skill only says which points of an object to use and which constraints hold between them, by the object's geometry,
and gets back ranked, executable grasps:

    from locomani.tasks.scene import keypoints as K
    spec = dict(pinch=("hull:-y", "hull:+y"))              # the fingertips on the hull's two sides across its y axis
    spec = dict(pinch=("hull:-y", "hull:+y"), approach="level")      # ... coming in level
    spec = dict(centre="top", approach="down", closing="horizontal", depth=-0.01)
    opts = K.plan(world, key, spec, start_xy)              # choose.Option list, best first, each with its `rot`
    out = K.pick(ctx, key, spec)                           # plan from where the robot stands and execute the best

Points (`points`), from the scene file's box and the collision hull, all world coordinates with an outward normal:
`centre`, `top` (the hull's highest point over the centre), `face:+x` ... `face:-z` (the box's faces in its own
frame), `hull:+x` ... `hull:-y` (where the hull section at height `z` ends along the box's own horizontal axes, on the
line through the section's middle). A spec may also give a point as an (x, y, z) triple.

Constraints (a spec's keys):
  pinch=(A, B)        the fingertips on A and B: the jaw centre at their midpoint, the closing axis along B - A
  centre=A            the jaw centre at A (when no pinch)
  approach=           "down", "level" (horizontal, its angle free), ("normal", A) -- into A's face, ("in_plane", A) --
                      within A's face's plane, angle free -- or an (x, y, z) direction; None: free
  closing=            "horizontal", "vertical" or an (x, y, z) direction; None: free (fixed by a pinch)
  depth=0.0           the jaw centre moved this far along the approach (negative: stop short of the centre)
  z=None              the height of the `hull:` points (default: the side-jaw height for a standing object, the
                      object's middle for a flat one)
  twist=0.0           with a pinch, the closing axis may turn about the vertical by up to this many degrees (the
                      wrist's twist window: a flat object pinched a little askew), the width measured across it
The support is a hard constraint: a pose whose fingertips (FINGERTIP_M beyond the jaw centre along the approach) or
housing (HOUSING_BELOW_M under the jaw axis) would go below the support's top + TIP_CLEAR_M is raised until they do
not, and a pose with nothing (or under choose.MIN_SECTION_M) between the fingers over their span, or whose fingers'
height range overlaps the object's by less than MIN_OVERLAP_M (half a flat object's height), is dropped.
A free angle is sampled every STEP_DEG; an approach from below (up more than UP_MAX) is never offered. The wrist frame
is the motor's: column 0 the approach, column 1 the closing axis, column 2 their cross product -- up for a side grasp,
toward the far side for a pinch from above (the seed skills' rot_z(psi) rot_y(pi/2)).

Scoring is the chooser's (choose.options with `grasp_list`): each solved grasp becomes a candidate with its approach
yaw, the body model's attitude class of its rotation (body.attitude), its jaw height and its fit margin; the chooser
adds the stance (clear of the furniture, with a path), the hand, the posture and the body model's odds -- no inverse
kinematics (warehouse line: IK feasibility AUC 0.56). A grasp whose closing axis is horizontal is checked against the
hull section at the jaw height (graspability.width_across); others against the pinch points' distance or the box.

Execution (`execute`, `pick`) goes through stage() (rules.md 31): walk to the stance (re-walk once when the stop is
off), the posture when the option lowers the pelvis, the pre-grasp PRE_BACK_M back along the approach over a raised
via point, the insert, the grip, the lift; the grasp follows the object if it moved since the plan. It returns the
lift's Outcome with `held`. The person writes this module once; the loop calls it and does not change it (D8)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from locomani.backends.scalebfm.motor import Outcome
from locomani.tasks.scene import body as B
from locomani.tasks.scene import choose as C
from locomani.tasks.scene import graspability as G
from locomani.tasks.scene.stage import grip_check, lift_check, reach_check, stage, stance_check

STEP_DEG = 15.0  # a free approach or roll angle is sampled this often
UP_MAX = 0.10  # an approach whose z component is above this comes from below: never offered
VERTICAL = 0.30  # an approach whose horizontal part is under this is "from above": the stance's facing is sampled
FACING_STEP_DEG = 30.0  # ... every this many degrees within the wrist's twist window (choose.TWIST_MAX_DEG)
ORTHO_TOL_DEG = 10.0  # a given approach this far from square to the closing axis is squared; farther, refused
FLAT_HALF_M = G.FLAT_HALF_M
FINGERTIP_M = 0.033  # the jaw centre (the pads' centre, runtime.OFF) to the fingertips, along the approach
HOUSING_BELOW_M = 0.034  # the gripper housing below the jaw axis, along the wrist's -z (graspability.JAW_ABOVE_TOP_M)
TIP_CLEAR_M = 0.005  # the fingertips and the housing kept this far above the support's top
FINGER_HALF_W_M = 0.010  # the fingers' half width across the closing plane (graspability.SECTION_BAND_M)
FINGER_REACH_M = (
    FINGERTIP_M  # what lies this far either side of the jaw centre along the approach is between the fingers
)
MIN_OVERLAP_M = 0.010  # the object's height range and the fingers' must overlap this much (half a flat object's)
PRE_BACK_M = 0.10  # the pre-grasp stands this far back along the approach
VIA_UP_M = 0.10  # the hand passes this high over the support (or the pre-grasp + VIA_OVER_M) on its way
VIA_OVER_M = 0.05
VIA_BACK_M = 0.10  # ... after drawing back this far against the approach
INSERT_SPEED = 0.05  # m/s (the seed skills' entry speed)
LIFT_M = 0.10
LIFT_SPEED = 0.08
PRE_TOL_M = 0.03  # the pre-grasp's reach check
INSERT_TOL_M = 0.02
REWALKS = 1
REAPPROACHES = 1


@dataclass
class Point:
    xyz: np.ndarray
    normal: np.ndarray | None = None


@dataclass
class GraspPose:
    centre: np.ndarray  # the jaw centre, world
    rot: np.ndarray  # 3x3: approach, closing, approach x closing
    width: float | None  # the object across the jaw (m), None when unknown
    margin: float  # (open jaw - width) / 2, a side
    att: str  # body.attitude(rot)
    note: str

    @property
    def approach(self) -> np.ndarray:
        return self.rot[:, 0]

    @property
    def closing(self) -> np.ndarray:
        return self.rot[:, 1]


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError("a zero direction")
    return v / n


def _frame(world, key: str):
    o = world.meta["objects"][key]
    p, q = world.object_pose(key)
    Rb = G._quat_mat(q)
    centre = np.asarray(p, float) + Rb @ np.asarray(o["box"].get("centre", (0.0, 0.0, 0.0)), float)
    return centre, Rb, np.asarray(o["box"]["half_extent"], float)


def default_z(world, key: str) -> float:
    """The `hull:` points' height: the side jaw's for a standing object, the middle of a flat one."""
    if G.resting_half_height(world, key) < FLAT_HALF_M:
        z = np.concatenate([v[:, 2] for v in G.world_parts(world, key)])
        return float(0.5 * (z.max() + z.min())) if len(z) else float(_frame(world, key)[0][2])
    return G.grasp_height(world, key)


def points(world, key: str, z: float | None = None) -> dict[str, Point]:
    """The named points of an object at its pose now (module docstring)."""
    centre, Rb, half = _frame(world, key)
    out = {"centre": Point(centre.copy(), None)}
    parts = G.world_parts(world, key)
    zs = np.concatenate([v[:, 2] for v in parts]) if parts else np.array([centre[2] + half[2]])
    out["top"] = Point(np.array([centre[0], centre[1], float(zs.max())]), np.array([0.0, 0.0, 1.0]))
    for i, ax in enumerate("xyz"):
        for s, sign in (("+", 1.0), ("-", -1.0)):
            n = sign * Rb[:, i]
            out[f"face:{s}{ax}"] = Point(centre + sign * half[i] * Rb[:, i], n)
    zz = default_z(world, key) if z is None else float(z)
    sec = G.section(world, key, zz)
    if len(sec):
        mid = 0.5 * (sec.min(0) + sec.max(0))
        for i, ax in enumerate("xy"):
            h = Rb[:2, i]
            if float(np.linalg.norm(h)) < 0.3:  # the box lies on its side: this axis is not horizontal
                continue
            h = h / np.linalg.norm(h)
            s = sec @ h
            for sgn, ext in (("+", float(s.max())), ("-", float(s.min()))):
                xy = mid + (ext - float(mid @ h)) * h
                out[f"hull:{sgn}{ax}"] = Point(np.array([xy[0], xy[1], zz]), np.r_[h if sgn == "+" else -h, 0.0])
    return out


def _point(pts: dict[str, Point], ref) -> Point:
    if isinstance(ref, Point):
        return ref
    if isinstance(ref, str):
        if ref not in pts:
            raise KeyError(f"no point {ref!r}; known: {', '.join(sorted(pts))}")
        return pts[ref]
    return Point(np.asarray(ref, float), None)


def _circle(k: np.ndarray) -> list[np.ndarray]:
    """Unit vectors square to `k`, every STEP_DEG."""
    e1 = np.cross(k, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.array([1.0, 0.0, 0.0])
    e1 = _unit(e1)
    e2 = np.cross(k, e1)
    return [math.cos(t) * e1 + math.sin(t) * e2 for t in np.radians(np.arange(0.0, 360.0, STEP_DEG))]


def _approaches(spec_a, pts, k: np.ndarray | None) -> list[np.ndarray]:
    """The approach directions a spec allows, square to the closing axis `k` when a pinch fixed it: a free or in-plane
    approach is taken square to `k` directly; a given direction within ORTHO_TOL_DEG of square is squared, farther
    refused."""
    z = np.array([0.0, 0.0, 1.0])
    level = [np.array([math.cos(t), math.sin(t), 0.0]) for t in np.radians(np.arange(0.0, 360.0, STEP_DEG))]
    fixed = False
    if spec_a is None:
        cands = _circle(k) if k is not None else _sphere()
    elif isinstance(spec_a, str) and spec_a == "down":
        cands, fixed = [-z], True
    elif isinstance(spec_a, str) and spec_a == "level":
        if k is not None and abs(float(k[2])) < 1 - 1e-6:
            h = _unit(np.cross(k, z))
            cands = [h, -h]
        else:
            cands = level
    elif isinstance(spec_a, (tuple, list)) and len(spec_a) == 2 and spec_a[0] in ("normal", "in_plane"):
        n = _point(pts, spec_a[1]).normal
        if n is None:
            raise ValueError(f"approach {spec_a[0]} of a point without a normal")
        n = _unit(n)
        if spec_a[0] == "normal":
            cands, fixed = [-n], True
        elif k is not None and np.linalg.norm(np.cross(n, k)) > 1e-6:
            h = _unit(np.cross(n, k))
            cands = [h, -h]
        else:
            cands = _circle(n)
    else:
        cands, fixed = [_unit(spec_a)], True
    out = []
    for a in cands:
        if k is not None:
            off = math.degrees(math.asin(min(1.0, abs(float(a @ k)))))
            if fixed and off > ORTHO_TOL_DEG:
                continue
            a = a - float(a @ k) * k
            if np.linalg.norm(a) < 1e-6:
                continue
            a = _unit(a)
        if a[2] > UP_MAX:
            continue
        if not any(float(a @ b) > 1 - 1e-6 for b in out):
            out.append(a)
    return out


def _sphere() -> list[np.ndarray]:
    out = []
    for el in np.radians(np.arange(-90.0, 0.0 + 1e-9, STEP_DEG)):
        for az in np.radians(np.arange(0.0, 360.0, STEP_DEG)):
            out.append(np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)]))
            if abs(el + math.pi / 2) < 1e-9:
                break
    return out


def _closings(spec_k, a: np.ndarray) -> list[np.ndarray]:
    """Closing axes square to the approach `a` that the spec allows (one of each +- pair: a pinch is symmetric)."""
    z = np.array([0.0, 0.0, 1.0])
    if spec_k is None:
        cands = _circle(a)
    elif spec_k == "horizontal":
        h = np.cross(a, z)
        cands = [_unit(h)] if np.linalg.norm(h) > 1e-6 else [c for c in _circle(a)]
    elif spec_k == "vertical":
        v = z - float(z @ a) * a
        cands = [_unit(v)] if np.linalg.norm(v) > 1e-6 else []
    else:
        v = _unit(spec_k)
        v = v - float(v @ a) * a
        cands = [_unit(v)] if np.linalg.norm(v) > 1e-6 else []
    out = []
    for c in cands:
        if not any(abs(float(c @ b)) > 1 - 1e-6 for b in out):
            out.append(c)
    return out


def _agrees(spec_k, k: np.ndarray) -> bool:
    """A pinch's closing axis against a `closing` the spec also gives, within ORTHO_TOL_DEG."""
    tol = math.sin(math.radians(ORTHO_TOL_DEG))
    if spec_k is None:
        return True
    if spec_k == "horizontal":
        return abs(float(k[2])) <= tol
    if spec_k == "vertical":
        return abs(float(k[2])) >= math.cos(math.radians(ORTHO_TOL_DEG))
    return abs(float(_unit(spec_k) @ k)) >= math.cos(math.radians(ORTHO_TOL_DEG))


def _signed(a: np.ndarray, k: np.ndarray) -> np.ndarray:
    """The rotation [a, k, a x k] with k's sign chosen so the third column points up (a side or tilted approach); for
    a straight-down approach the facing decides it later (`_facings`)."""
    z = np.cross(a, k)
    if abs(a[2]) < 1 - 1e-6 and z[2] < 0:
        k = -k
    return np.stack([a, k, np.cross(a, k)], axis=1)


def _hull2d(pts: np.ndarray) -> np.ndarray:
    """The convex hull of 2-D points, counter-clockwise (monotone chain)."""
    p = np.unique(np.round(pts, 6), axis=0)
    if len(p) < 3:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(seq):
        out = []
        for q in seq:
            while len(out) >= 2 and np.cross(out[-1] - out[-2], q - out[-2]) <= 0:
                out.pop()
            out.append(q)
        return out

    lo, hi = half(p), half(p[::-1])
    return np.array(lo[:-1] + hi[:-1])


def _clip(poly: np.ndarray, n: np.ndarray, b: float) -> np.ndarray:
    """The part of a convex polygon with n . x <= b (Sutherland-Hodgman, one edge)."""
    if len(poly) == 0:
        return poly
    out = []
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        fp, fq = float(n @ p) - b, float(n @ q) - b
        if fp <= 0:
            out.append(p)
        if fp * fq < 0:
            out.append(p + (q - p) * fp / (fp - fq))
    return np.array(out) if out else np.zeros((0, 2))


def _width(world, key: str, centre: np.ndarray, rot: np.ndarray, pinch) -> float | None:
    """The object between the fingers, across the closing axis. With a horizontal closing axis: at heights over the
    fingers' span (the jaw centre to FINGERTIP_M along the approach -- a flat object is held low in the fingers of a
    pinch from above), the hull section's convex outline clipped to the space between the fingers (FINGER_HALF_W_M
    either side of the closing plane, FINGER_REACH_M either side of the jaw centre along the approach), its extent
    across the closing axis; None when nothing, or under choose.MIN_SECTION_M, lies there (the pose is dropped). A
    twisted pinch of a long object meets only the strip under the fingers, not the whole length. Otherwise the pinch
    points' distance or the box's extent."""
    k, a = rot[:, 1], rot[:, 0]
    e = np.cross(a, k)
    if abs(float(k[2])) < 1e-6:
        widths = []
        for f in (0.0, 0.25, 0.5, 0.75, 1.0):
            z = float(centre[2]) + f * FINGERTIP_M * min(float(a[2]), 0.0)
            sec = G.section(world, key, z)
            if len(sec) < 3:
                continue
            poly = _hull2d(sec - centre[:2])
            dz = z - float(centre[2])
            for n3, b in ((e, FINGER_HALF_W_M), (-e, FINGER_HALF_W_M), (a, FINGER_REACH_M), (-a, FINGER_REACH_M)):
                poly = _clip(poly, n3[:2], b - float(n3[2]) * dz)
            if len(poly):
                s = poly @ k[:2]
                widths.append(float(s.max() - s.min()))
        w = max(widths) if widths else 0.0
        return w if w >= C.MIN_SECTION_M else None
    if pinch is not None:
        return float(np.linalg.norm(pinch[1].xyz - pinch[0].xyz))
    _, Rb, half = _frame(world, key)
    return float(2.0 * (np.abs(Rb.T @ k) @ half))


def solve(world, key: str, spec: dict) -> list[GraspPose]:
    """Every grasp pose the spec allows, the free angles sampled (module docstring); the too-wide left out."""
    pts = points(world, key, spec.get("z"))
    pinch = None
    if spec.get("pinch") is not None:
        pinch = (_point(pts, spec["pinch"][0]), _point(pts, spec["pinch"][1]))
        centre = 0.5 * (pinch[0].xyz + pinch[1].xyz)
        k_fixed = _unit(pinch[1].xyz - pinch[0].xyz)
    elif spec.get("centre") is not None:
        centre, k_fixed = _point(pts, spec["centre"]).xyz.copy(), None
    else:
        raise ValueError("a spec needs `pinch` or `centre`")
    if k_fixed is not None and not _agrees(spec.get("closing"), k_fixed):
        return []
    sup = world.meta["surfaces"].get(world.meta["objects"][key].get("surface")) or {}
    floor = float(sup["top"]) + TIP_CLEAR_M if "top" in sup else -np.inf
    twist = float(spec.get("twist", 0.0))
    parts = G.world_parts(world, key)
    zs = np.concatenate([v[:, 2] for v in parts]) if parts else np.array([centre[2]])
    obj_lo, obj_hi = float(zs.min()), float(zs.max())
    ks = None
    if k_fixed is not None:
        ks = [k_fixed]
        if twist > 0 and abs(float(k_fixed[2])) < 1 - 1e-6:
            for d in np.radians(np.arange(STEP_DEG, twist + 1e-9, STEP_DEG)):
                for s in (d, -d):
                    c_, s_ = math.cos(s), math.sin(s)
                    ks.append(
                        _unit(
                            np.array([c_ * k_fixed[0] - s_ * k_fixed[1], s_ * k_fixed[0] + c_ * k_fixed[1], k_fixed[2]])
                        )
                    )
    out, seen = [], []
    for kf in ks or [None]:
        for a in _approaches(spec.get("approach"), pts, kf):
            for k in [kf] if kf is not None else _closings(spec.get("closing"), a):
                rot = _signed(a, k)
                if any(np.allclose(rot, r, atol=1e-6) for r in seen):
                    continue
                seen.append(rot)
                c = centre + float(spec.get("depth", 0.0)) * a
                low = min(
                    float(c[2]) + FINGERTIP_M * min(float(a[2]), 0.0),
                    float(c[2]) - HOUSING_BELOW_M * max(float(rot[2, 2]), 0.0),
                )
                raised = max(0.0, floor - low)
                c = c + np.array([0.0, 0.0, raised])
                e = np.cross(rot[:, 0], rot[:, 1])
                span = FINGER_REACH_M * abs(float(rot[2, 0])) + FINGER_HALF_W_M * abs(float(e[2]))
                overlap = min(obj_hi, float(c[2]) + span) - max(obj_lo, float(c[2]) - span)
                if overlap < min(MIN_OVERLAP_M, 0.5 * (obj_hi - obj_lo)):
                    continue  # the fingers pass over the object: a level jaw raised off the support over a flat one
                w = _width(world, key, c, rot, pinch)
                if w is None and abs(float(rot[2, 1])) < 1e-6:
                    continue  # nothing between the pads at this height
                margin = 0.5 * (G.JAW_OPEN_M - w) if w is not None else 0.0
                if margin < 0:
                    continue
                pitch = math.degrees(math.asin(max(-1.0, min(1.0, -float(a[2])))))
                off = math.degrees(math.asin(min(1.0, abs(float(rot[2, 1])))))
                note = (f"keypoint grasp: approach pitched {pitch:.0f} deg down, closing "
                        f"{'horizontal' if off < 1e-3 else f'{off:.0f} deg off level'}"
                        + ("" if w is None else f", {100 * w:.1f} cm across the jaw")
                        + (f", raised {100 * raised:.1f} cm off the support" if raised > 1e-4 else ""))  # fmt: skip
                out.append(GraspPose(c, rot, w, margin, B.attitude(rot), note))
    return out


def _facings(g: GraspPose) -> list[tuple[float, np.ndarray]]:
    """(stance facing in degrees, the rotation for it): the approach's own heading, or for a grasp from above every
    facing within the wrist's twist window of square to the closing axis, the rotation's third column turned along it."""
    a = g.approach
    if float(np.hypot(a[0], a[1])) >= VERTICAL:
        return [(math.degrees(math.atan2(a[1], a[0])), g.rot)]
    k = g.closing
    square = math.degrees(math.atan2(k[1], k[0])) + 90.0
    out = []
    for base in (square, square + 180.0):
        for off in np.arange(-C.TWIST_MAX_DEG, C.TWIST_MAX_DEG + 1e-9, FACING_STEP_DEG):
            f = C._wrap(base + off)
            u = np.array([math.cos(math.radians(f)), math.sin(math.radians(f)), 0.0])
            kk = k if float(np.cross(a, k) @ u) >= 0 else -k
            out.append((f, np.stack([a, kk, np.cross(a, kk)], axis=1)))
    return out


def plan(world, key: str, spec: dict, start_xy=None, **kw) -> list:
    """The spec's grasps, each with a stance, a hand and a posture, scored by the chooser and the body model, best
    first: choose.Option with kind "keypoint" and `rot` set (module docstring). `kw` goes to choose.options."""
    grasps = []
    for g in solve(world, key, spec):
        for facing, rot in _facings(g):
            grasps.append(dict(kind="keypoint", approach_deg=float(facing), att=B.attitude(rot), z=float(g.centre[2]),
                               margin_m=float(g.margin), note=g.note, xy=g.centre[:2].tolist(),
                               rot=np.round(rot, 6).tolist()))  # fmt: skip
    out, seen = [], set()
    for o in C.options(world, key, start_xy, grasp_list=grasps, **kw):  # several near-vertical poses rank alike
        k = (o.att, o.hand, round(o.approach_deg), round(o.distance_m, 2), o.pelvis_z, tuple(np.round(o.grasp_xyz, 3)))
        if k not in seen:
            seen.add(k)
            out.append(o)
    return out


def _note(ctx, name: str, **info) -> None:
    m = ctx.motor
    ctx.note(name, Outcome(True, "", m.t, m.t, info))


def execute(
    ctx,
    key: str,
    opt,
    planned_xyz=None,
    *,
    pre_back: float = PRE_BACK_M,
    insert_speed: float = INSERT_SPEED,
    lift_m: float = LIFT_M,
    lift_speed: float = LIFT_SPEED,
) -> Outcome:
    """Carry out one planned option (module docstring); the grasp follows the object by however far it moved since
    `planned_xyz` (its position when planned). The step parameters are the skill's to set (rules.md 36)."""
    w, m = ctx.world, ctx.motor
    side = opt.hand
    pel, yaw = np.asarray(opt.stance_xy, float), float(opt.stance_yaw)
    rot = np.asarray(opt.rot, float)
    a = rot[:, 0]
    _note(ctx, "keypoint_plan", hand=side, kind=opt.kind, att=opt.att, stance=list(opt.stance_xy),
          facing_deg=round(math.degrees(yaw), 1), grasp=list(opt.grasp_xyz), p_miss=opt.p_miss,
          margin_cm=round(100 * float(opt.fit_margin_m or 0.0), 2), reasons=opt.why[:3])  # fmt: skip
    from locomani.tasks.scene.navigate import walk_path

    out = stage(ctx, "walk", lambda: walk_path(m, w.occ, w.nav, pel, yaw), check=lambda o: stance_check(m, pel, yaw),
                on_fail=lambda k, v: (lambda: m.walk_to(pel, yaw, face="fixed")) if k < REWALKS else None)  # fmt: skip
    if not out.ok and m.fell_at is not None:
        return out
    if opt.pelvis_z is not None:
        out = stage(ctx, "posture", lambda: m.posture(float(opt.pelvis_z)))
        if m.fell_at is not None:
            return out
    g = np.asarray(opt.grasp_xyz, float)
    if planned_xyz is not None:
        g = g + (np.asarray(w.object_pose(key)[0], float) - np.asarray(planned_xyz, float))
    pre = g - float(pre_back) * a
    sup = w.meta["surfaces"].get(w.meta["objects"][key].get("surface")) or {}
    top = float(sup.get("top", g[2]))
    jaw0, _ = m.jaw_centre(side)
    ah = np.r_[a[:2], 0.0] / (float(np.hypot(a[0], a[1])) or 1.0) if float(np.hypot(a[0], a[1])) > 1e-6 else np.zeros(3)
    back = jaw0 - VIA_BACK_M * ah  # draw the hand back off the support's edge before it rises over it
    via = [np.r_[back[:2], max(float(jaw0[2]), top + VIA_UP_M, float(pre[2]) + VIA_OVER_M)],
           np.r_[pre[:2], float(pre[2]) + VIA_OVER_M]]  # fmt: skip
    out = stage(ctx, "pregrasp", lambda: m.reach(side, pre, rot, via=via), check=lambda o: reach_check(m, side, pre, PRE_TOL_M),
                on_fail=lambda k, v: (lambda: m.reach(side, pre, rot)) if k < REAPPROACHES else None)  # fmt: skip
    if not out.ok:
        return out
    out = stage(ctx, "insert", lambda: m.reach(side, g, rot, speed=float(insert_speed)),
                check=lambda o: reach_check(m, side, g, INSERT_TOL_M))  # fmt: skip
    if not out.ok:
        return out
    body_id = w.free[key]
    out = stage(ctx, "grip", lambda: m.grip(side), check=lambda o: grip_check(m, side, body_id))
    if not out.ok:
        return out
    z0 = float(w.object_pose(key)[0][2])
    up = g + np.array([0.0, 0.0, float(lift_m)])
    out = stage(
        ctx, "lift", lambda: m.reach(side, up, rot, speed=float(lift_speed)), check=lambda o: lift_check(w, key, z0)
    )
    held = bool(m.gripping(body_id))
    return Outcome(out.ok and held, out.why or ("" if held else "not between the fingers after the lift"), out.t0,
                   out.t1, dict(out.info, held=held))  # fmt: skip


def pick(ctx, key: str, spec: dict, *, option: int = 0, steps: dict | None = None, **kw) -> Outcome:
    """Plan from where the robot stands now and execute the best option; a failed Outcome naming why when the spec
    gives no grasp or no option can stand clear. `option` takes another than the best (0); `steps` goes to execute
    (pre_back, insert_speed, lift_m, lift_speed); `kw` to choose.options."""
    w, m = ctx.world, ctx.motor
    start = m.sim.d.qpos[:2].copy()
    opts = plan(w, key, spec, start, **kw)
    if not opts:
        out = Outcome(False, "keypoints: no grasp with a clear stance for this spec", m.t, m.t)
        return ctx.note("keypoint_plan", out, spec={k: str(v) for k, v in spec.items()})
    opt = opts[min(int(option), len(opts) - 1)]
    return execute(ctx, key, opt, planned_xyz=w.object_pose(key)[0].copy(), **(steps or {}))
