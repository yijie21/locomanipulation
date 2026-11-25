"""The feasibility probe's hand-written two-hand pick (TODO 38.7, D5): not a seed and not in any sandbox. It exists
to show, before the loop spends anything, that the motor layer's two-hand actions lift the logistics room's carton
from a stance the skills could choose: walk to the platform edge nearest the box, lower the pelvis, put both jaw
centres beside the box's two side walls with the jaws closed and the wrists pointing forward (a hug), squeeze, lift both hands
(squeeze kept), hold. Everything a skill would decide is a
module constant here so `--set` can move it (apply_set), and every step is logged through ctx.note.

Loaded by scripts/rsi/scene_round.py like any skills package (`plan_for`, `apply_set`, `parameters`)."""

from __future__ import annotations

import types

import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import Outcome
from locomani.tasks.scene import navigate

STAND_FROM_EDGE_M = 0.38  # pelvis to the platform edge the box is nearest to (body radius 0.35 keeps the map free)
PELVIS_Z = 0.72  # lowered before the reach
GRASP_Z_FRAC = 0.55  # grasp height as a fraction of the box's height above its bottom
MODE = "wall"  # "wall": the human-defined grasp, each side wall pinched near the rim with the palm pitched down
# (default since 2026-10-24, user); "hug": open jaws pressing the walls with the outer finger flanks + squeeze;
# "rim": a pinch on each wall's top edge with the wrist straight down
WALL_INSET_M = 0.008  # grasp_region.GRASP_WALL_INSET_M: jaw centre this far inside the wall's outer face
WALL_BELOW_RIM_M = 0.01  # visual_config.box_grasp_below_rim: jaw centre under the rim top (the cardboard box)
PALM_PITCH_DEG = 60.0  # visual_config.palm_pitch_deg: the palm pitched this far below level
WALL_PRE_ABOVE_M = 0.09  # visual_config.pregrasp_above
WALL_PRE_BACK_M = 0.04  # visual_config.pregrasp_back
WALL_PELVIS_Z = 0.76  # pelvis before the reach: the rim is at 0.70 m, the hands work at about pelvis height
FLANK_OPEN_M = 0.0635  # the open jaw's outer finger flank from the jaw axis (robot.xml: 0.039 at 0 + 0.0245 travel)
FLANK_GAP_M = 0.005  # ... plus this before the squeeze
RIM_BELOW_M = 0.025  # rim pinch: jaw centre below the wall's top edge
RIM_ABOVE_M = 0.08  # rim pinch: the pre-grasp above the grasp
PRE_OUT_M = 0.06  # the pre-grasp: this much further out from each wall
PRE_BACK_M = 0.12  # ... and this much short of the walls' centre (the hands come in from the robot's side)
INWARD_M = 0.06  # squeeze offset per hand (the push is at most twice this, motor._servo; 0.04: 1 / 4, 0.06 and 0.08: 4 / 4 on 2026-10-23)
LIFT_M = 0.32  # the lift: both jaw centres this much up (the strict goal needs the lowest corner 20 cm up; 0.25 fell a few mm short, 2026-10-24)
LIFT_SPEED = 0.08  # m/s
LIFT_PELVIS_Z = (
    0.79  # the lift is a whole-body IK with the pelvis back up (standing): the legs lift, the arms keep the hold
)
SQUEEZE_S = 1.0
LIFT_HOLD_S = 2.0
NORMALS = {"-x": (-1.0, 0.0), "+x": (1.0, 0.0), "-y": (0.0, -1.0), "+y": (0.0, 1.0)}


def _quat_mat(q) -> np.ndarray:
    return R.mat(np.asarray(q, float))


def _nearest_edge(meta: dict, surf_key: str, xy) -> tuple[str, np.ndarray, float]:
    """(edge name, outward unit normal (world), the edge's distance from the object's centre) of the box's surface."""
    s = meta["surfaces"][surf_key]
    yaw = float(s.get("yaw", 0.0))
    Rz = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    flat = np.asarray(s.get("flat_half_extent") or s["half_extent"], float)[:2]
    local = Rz.T @ (np.asarray(xy, float)[:2] - np.asarray(s["xy"], float))
    best = None
    for name in s.get("approach_edges") or list(NORMALS):
        axis = 0 if name[1] == "x" else 1
        depth = flat[axis] - local[axis] if name[0] == "+" else local[axis] + flat[axis]
        if best is None or depth < best[2]:
            best = (name, Rz @ np.asarray(NORMALS[name], float), float(depth))
    return best


def plan_for(task):
    return _two_hand_pick if any(g.startswith("held(") for g in task.goal) else None


def _two_hand_pick(ctx) -> Outcome:
    m, w = ctx.motor, ctx.world
    key = ctx.task.objects[0]
    obj = w.meta["objects"][key]
    pos, quat = w.object_pose(key)
    Rb = _quat_mat(quat)
    half = np.asarray(obj["box"]["half_extent"], float)
    centre = np.asarray(pos, float) + Rb @ np.asarray(obj["box"].get("centre") or (0, 0, 0), float)
    edge, n_out, depth = _nearest_edge(w.meta, obj["surface"], centre[:2])
    # the stance: on the outward normal of the nearest edge, STAND_FROM_EDGE_M past it, facing the box
    edge_pt = centre[:2] + n_out * depth  # the box centre projected onto the edge
    stance = edge_pt + n_out * STAND_FROM_EDGE_M
    face = float(np.arctan2(-n_out[1], -n_out[0]))
    out = ctx.note(
        "stance", navigate.walk_path(m, w.occ, w.nav, stance, face), edge=edge, depth_cm=round(100 * depth, 1)
    )
    if not out.ok:
        return out
    # the two walls the hands press: the box axis most across the robot's facing direction
    fwd = -n_out
    axes = [Rb[:2, 0], Rb[:2, 1]]
    j = int(np.argmax([abs(float(np.cross(fwd, a))) for a in axes]))
    u = np.r_[axes[j], 0.0] / np.linalg.norm(axes[j])
    if float(np.cross(fwd, u[:2])) < 0:  # u points to the robot's left
        u = -u
    up = np.array([0.0, 0.0, 1.0])
    xf = np.r_[fwd, 0.0]
    out = ctx.note("lower", m.posture(pelvis_z=WALL_PELVIS_Z if MODE == "wall" else PELVIS_Z))
    if not out.ok:
        return out
    if MODE == "wall":
        # the human-defined two-arm grasp of the logistics room's box (the crate line's expert, DECISIONS 2026-09-22;
        # tasks/crate/grasp_region.py, visual_config.py): one grasp point on each side wall -- the two walls left and
        # right of the robot -- WALL_INSET_M inside its outer face (the open jaw straddles the 1 cm wall: one finger
        # in the box, one outside) and WALL_BELOW_RIM_M under the rim; the jaw closes across the wall (wrist y = u);
        # the palm pitched PALM_PITCH_DEG below level; pre-grasp WALL_PRE_ABOVE_M above and WALL_PRE_BACK_M toward
        # the body, then straight down; both jaws close; the arms lift
        z = float(pos[2]) + 2 * float(half[2]) - WALL_BELOW_RIM_M
        gl = np.r_[centre[:2], z] + u * (half[j] - WALL_INSET_M)
        gr = np.r_[centre[:2], z] - u * (half[j] - WALL_INSET_M)
        th = np.radians(PALM_PITCH_DEG)
        xw = np.cos(th) * xf - np.sin(th) * up
        Rl = Rr = np.stack([xw, u, np.cross(xw, u)], axis=1)
        pre = (gl + up * WALL_PRE_ABOVE_M - xf * WALL_PRE_BACK_M, gr + up * WALL_PRE_ABOVE_M - xf * WALL_PRE_BACK_M)
        via = []
    elif MODE == "hug":
        # open jaws, wrist x forward, z up: each hand's outer finger flank (FLANK_OPEN_M from the jaw axis) presses a
        # wall; the gripper base is narrower than that, so the fingers touch first (held() counts finger contacts)
        z = float(pos[2]) + GRASP_Z_FRAC * 2 * float(half[2])
        gl = np.r_[centre[:2], z] + u * (half[j] + FLANK_OPEN_M + FLANK_GAP_M)
        gr = np.r_[centre[:2], z] - u * (half[j] + FLANK_OPEN_M + FLANK_GAP_M)
        Rl = Rr = np.stack([xf, np.cross(up, xf), up], axis=1)
        pre = (gl + u * PRE_OUT_M - xf * PRE_BACK_M, gr - u * PRE_OUT_M - xf * PRE_BACK_M)
        via = [(gl + u * PRE_OUT_M, gr - u * PRE_OUT_M)]
    else:
        # rim pinch: wrist x down, the jaw closes across the wall (y = u), the jaw centre on the wall's plane a little
        # below its top edge; the crate line's grasp (DECISIONS 2026-09-22), one hand per end wall
        z = float(pos[2]) + 2 * float(half[2]) - RIM_BELOW_M
        gl = np.r_[centre[:2], z] + u * half[j]
        gr = np.r_[centre[:2], z] - u * half[j]
        xd = -up
        Rl = np.stack([xd, u, np.cross(xd, u)], axis=1)
        Rr = np.stack([xd, -u, np.cross(xd, -u)], axis=1)
        pre = (gl + up * RIM_ABOVE_M, gr + up * RIM_ABOVE_M)
        via = []
    out = ctx.note("pregrasp", m.reach("both", pre, (Rl, Rr)), grasp_left=gl, grasp_right=gr, mode=MODE)
    if m.fell_at is not None:
        return out
    out = ctx.note("entry", m.reach("both", (gl, gr), (Rl, Rr), via=via))
    if m.fell_at is not None:
        return out
    if MODE != "hug":
        out = ctx.note("pinch", m.grip("both"))
        if m.fell_at is not None:
            return out
    if MODE != "wall":  # the wall pinch holds by the closed jaws; a squeeze would crush the box's walls inward
        out = ctx.note("squeeze", m.squeeze(INWARD_M, SQUEEZE_S))
        if m.fell_at is not None:
            return out
    z0 = float(w.object_pose(key)[0][2])
    # the lift: both hands straight up in the world, still squeezing. posture() (stand up) keeps the arms in joint
    # space, so from a forward lean it swings the hands up and back and lets go (probe 2026-10-23)
    pl, pr = m.jaw_centre("left")[0] + up * LIFT_M, m.jaw_centre("right")[0] + up * LIFT_M
    out = ctx.note("lift", m.reach("both", (pl, pr), keep_squeeze=True, speed=LIFT_SPEED, pelvis_z=LIFT_PELVIS_Z))
    if m.fell_at is not None:
        return out
    rise = float(w.object_pose(key)[0][2]) - z0
    out = ctx.note("carry_hold", m.hold(LIFT_HOLD_S), rise_cm=round(100 * rise, 1),
                   touching=sorted(m.touching(w.free[key])))  # fmt: skip
    return out


def apply_set(package, overrides: dict) -> list[str]:
    done = []
    for key, value in (overrides or {}).items():
        parts = key.split(".")
        target = package
        if len(parts) > 1 and hasattr(package, parts[0]) and not hasattr(package, key):
            target, parts = getattr(package, parts[0]), parts[1:]
        if not hasattr(target, parts[0]):
            raise KeyError(f"{key}: no parameter {parts[0]!r} in {getattr(target, '__name__', target)}")
        if len(parts) == 1:
            setattr(target, parts[0], value)
        elif len(parts) == 2 and isinstance(getattr(target, parts[0]), dict):
            getattr(target, parts[0])[parts[1]] = value
        else:
            raise KeyError(f"{key}: expected NAME, module.NAME or module.NAME.key")
        done.append(key)
    return done


def _overridable(v) -> bool:
    return isinstance(v, (int, float, str, dict)) and not isinstance(v, bool)


def parameters(package) -> dict:
    out = {k: v for k, v in vars(package).items() if k.isupper() and _overridable(v)}
    for name, mod in vars(package).items():
        if isinstance(mod, types.ModuleType) and mod.__name__.startswith(package.__name__ + "."):
            out.update({f"{name}.{k}": v for k, v in vars(mod).items() if k.isupper() and _overridable(v)})
    return out
