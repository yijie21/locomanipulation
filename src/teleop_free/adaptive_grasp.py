"""Choose the grasp orientation from where the hand actually is.

The fixed grasp pose was the wrong shape of answer. It pinned the palm to a
heading derived from the pelvis and a tilt derived from a calibration, and then
the arm had to contort its way from wherever it happened to be into that pose —
which is most of what the videos show. But the robot never stands in quite the
same place twice, so the pose it should aim for is not the same twice either.

What IS fixed is the grip point: the object's own axis, at a chosen height.
Everything about the hand's orientation is free, and the useful way to spend
that freedom is to make the approach a straight line.

The trick is that GRASP_CENTRE_OFFSET is a vector in the palm frame from the
wrist to the point that has to land on the object. Write the final wrist as

    wrist_end = grip - R @ offset

and ask for the hand to travel in a straight line from where it is now, i.e.
wrist_end = wrist_now + t * d for the unit direction d. Substituting,

    R @ offset = grip - wrist_now - t * d = (|grip - wrist_now| - t) * d

so R @ offset must be PARALLEL TO d. That is the whole constraint: orient the
palm so its grip offset points along the line of travel, and the hand arrives
grip-centre first, in a straight line, with nothing to unwind at the end.

Two of the three rotational degrees of freedom go into that. The third — spin
about the offset axis — is free, and is spent on the things that actually
decide whether the grasp works: the jaw has to close across the object rather
than along it, the fingers have to clear the table, and among what is left the
pose closest to the hand's current orientation is the one it can reach without
turning itself inside out.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from primitives_const import GRASP_CENTRE_OFFSET_CONST

HAND_POINTS_PATH = "/workspace/code/locomanipulation/outputs/scene/hand_points.npy"

OFFSET = np.asarray(GRASP_CENTRE_OFFSET_CONST, dtype=float)
OFFSET_DIR = OFFSET / np.linalg.norm(OFFSET)
OFFSET_LEN = float(np.linalg.norm(OFFSET))

# How many spins about the offset axis to score. 5 deg is finer than the arm
# tracks and the whole sweep is one matrix multiply per candidate.
SPIN_STEPS = 72
# Clearance the fingers must keep over the surface. The staging point sits
# above the object and the hand comes down into place, so this is checked at
# the END pose, where it is tightest.
TABLE_CLEARANCE = 0.012
# How far off horizontal the pinch axis may sit. A vertical cylinder can be
# gripped across any horizontal chord, and a tilted jaw still catches the
# barrel, but past this the grip is riding up the side rather than around it.
JAW_TILT_MAX = np.deg2rad(12.0)   # was 45, then 25; tilt drops a fingertip
                                  # into the can before the jaw arrived
MAX_DESCEND = np.deg2rad(8.0)     # steepest allowed approach dive. 25 deg
                                  # let the fingertips lead ~10 cm below the
                                  # palm: with the grip 8.5 cm over the table
                                  # the plan required fingertips BELOW the
                                  # tabletop — the z-floor held the palm 2.4 cm
                                  # above the grasp line (v-24..-28 telemetry)
                                  # while the fingertips still scraped, and
                                  # every entry ended in a wedge/flinch loop
# Weights for the spin score. Closeness to the current palm dominates, because
# every degree of it is arm travel; clearance is a hard gate rather than a cost.
W_CLOSENESS = 1.0
W_JAW = 0.6
# Widest span the jaw can usefully close over, minus a pinch margin. Kept
# numerically in sync with grasping.HAND_APERTURE / PINCH_MARGIN without
# importing that module (it drags in the planner-era pose machinery).
APERTURE_MAX = 0.061 - 0.010


@dataclass
class Grasp:
    ok: bool = False
    why: str = ""
    rot: np.ndarray | None = None        # palm rotation, world
    wrist: np.ndarray | None = None      # where the wrist ends up
    grip: np.ndarray | None = None       # where the grasp centre ends up
    approach: np.ndarray | None = None   # unit, the straight line it travels
    travel: float = 0.0                  # metres from the current wrist
    jaw_tilt: float = 0.0                # radians off horizontal
    turn: float = 0.0                    # radians the palm has to rotate
    clearance: float = 0.0               # fingers above the surface at the end


_POINTS: np.ndarray | None = None


def hand_points() -> np.ndarray:
    """The hand's shape about the grasp centre, in the palm frame."""
    global _POINTS
    if _POINTS is None:
        if not os.path.exists(HAND_POINTS_PATH):
            raise FileNotFoundError(
                f"{HAND_POINTS_PATH} missing — run teleop_free/make_hand_points.py")
        _POINTS = np.load(HAND_POINTS_PATH).astype(float)

    return _POINTS


def _align(a, b) -> np.ndarray:
    """Minimal rotation taking unit a onto unit b (Rodrigues, no trig)."""
    v = np.cross(a, b)
    c = float(a @ b)
    if c < -1.0 + 1e-9:                      # antiparallel: any perpendicular
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return -np.eye(3) + 2 * np.outer(axis, axis)
    k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])

    return np.eye(3) + k + k @ k / (1.0 + c)


def _spin(axis, angle) -> np.ndarray:
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])

    return (np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k))


def plan(grip, wrist_now, palm_now, surface_top: float,
         obj_axis=(0.0, 0.0, 1.0), box=None, obj_rot=None) -> Grasp:
    """Orient the palm so the grip offset points along the line to the object.

    `grip` is where the grasp centre must end up — on the object's own axis, and
    the one thing that does not depend on the robot. `wrist_now`/`palm_now` are
    the hand's live pose, which is what makes the answer specific to this stand
    rather than to a nominal one.
    """
    grip = np.asarray(grip, dtype=float)
    wrist_now = np.asarray(wrist_now, dtype=float)
    palm_now = np.asarray(palm_now, dtype=float).reshape(3, 3)
    axis = np.asarray(obj_axis, dtype=float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)

    to_grip = grip - wrist_now
    travel = float(np.linalg.norm(to_grip))
    if travel < OFFSET_LEN + 1e-3:
        return Grasp(False, why=(
            f"already inside the grasp offset: the grip point is {travel:.3f} m "
            f"from the wrist and the offset alone is {OFFSET_LEN:.3f} m"))
    direction = to_grip / travel

    # Cap the dive. The azimuth follows the real wrist — that is the whole
    # point of planning from where the hand is — but a hand that happens to
    # hover above the grip must not turn into a top-down strike: past ~25 deg
    # the fingertips lead the descent and press the object over before the
    # jaw arrives (watched happening: palm flat over the can, can rolling).
    dh = float(np.hypot(direction[0], direction[1]))
    if -direction[2] > np.tan(MAX_DESCEND) * dh:
        if dh < 1e-6:
            return Grasp(False, why=(
                "directly above the grip — no approach azimuth to flatten "
                "the dive along"))
        direction = np.array([direction[0], direction[1],
                              -np.tan(MAX_DESCEND) * dh])
        direction /= np.linalg.norm(direction)

    # Base rotation: the smallest turn from the CURRENT palm that puts its grip
    # offset on the line. Starting from the current pose rather than from an
    # arbitrary frame is what keeps `turn` small.
    base = _align(palm_now @ OFFSET_DIR, direction) @ palm_now

    points = hand_points()
    best = None
    for i in range(SPIN_STEPS):
        phi = 2.0 * np.pi * i / SPIN_STEPS
        rot = base @ _spin(OFFSET_DIR, phi)

        # The jaw closes along the palm's y. Across the object's axis is what
        # grips it; along the axis just slides off.
        # tilt is how far the jaw leans out of the plane perpendicular to the
        # object's axis: 0 when it closes straight across the barrel, 90 deg
        # when it closes along it and grips nothing.
        jaw = rot[:, 1]
        tilt = abs(float(np.arcsin(np.clip(abs(float(jaw @ axis)), -1.0, 1.0))))
        if tilt > JAW_TILT_MAX:
            continue

        # The continuous envelope gate: the span the jaw would have to close
        # over, projected onto the jaw axis. A raw AABB projection
        # overestimates a cylinder by up to sqrt(2) (corner-to-corner where
        # the barrel is round), which refused every heading on the can — so
        # when two half-extents tie the shape is treated as the cylinder it
        # is, with its exact support width. Elongated objects keep the box
        # projection and get their infeasible headings cut here instead of
        # by a pick-the-shortest-edge rule.
        if box is not None and obj_rot is not None:
            # A pinch closes over a local cross-section, not over the whole
            # solid, so the span that must fit in the jaw is measured in the
            # plane perpendicular to the object's axis — a jaw tilted 20 deg
            # on a can still only spans the barrel's diameter, not diameter-
            # plus-height. (JAW_TILT_MAX separately caps how far the jaw may
            # lean before it stops gripping at all.)
            j_in = jaw - float(jaw @ axis) * axis
            nj = float(np.linalg.norm(j_in))
            if nj > 1e-6:
                j_in /= nj
                half = np.asarray(box["half_extent"], dtype=float)
                ax = np.abs(np.asarray(obj_rot, dtype=float).T @ j_in)
                pairs = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
                i, j, k = min(pairs,
                              key=lambda t: abs(half[t[0]] - half[t[1]]))
                if abs(half[i] - half[j]) < 0.005:
                    # Symmetric footprint: the cross-section is a circle and
                    # the box's diagonal would overestimate it by sqrt(2).
                    r = 0.5 * (half[i] + half[j])
                    width = 2.0 * (r * float(np.hypot(ax[i], ax[j]))
                                   + half[k] * float(ax[k]))
                else:
                    width = 2.0 * float(ax @ half)
                if width > APERTURE_MAX:
                    continue

        # Where the whole hand sits once it is there.
        world = points @ rot.T + grip
        clear = float(world[:, 2].min()) - surface_top
        if clear < TABLE_CLEARANCE:
            continue

        # How far the palm has to turn to get here, as an angle.
        turn = float(np.arccos(np.clip(
            (np.trace(rot @ palm_now.T) - 1.0) / 2.0, -1.0, 1.0)))
        cost = W_CLOSENESS * turn + W_JAW * tilt
        if best is None or cost < best[0]:
            best = (cost, rot, tilt, turn, clear)

    if best is None:
        return Grasp(False, why=(
            "no spin about the approach line both clears the surface and puts "
            "the jaw across the object — the hand is coming from a direction "
            "this object cannot be gripped from"))

    _, rot, tilt, turn, clear = best

    return Grasp(True, rot=rot, wrist=grip - rot @ OFFSET, grip=grip,
                 approach=direction, travel=travel - OFFSET_LEN,
                 jaw_tilt=tilt, turn=turn, clearance=clear)


def describe(g: Grasp) -> str:
    if not g.ok:
        return f"no grasp: {g.why}"

    return (f"grip {np.round(g.grip, 3)}, straight in along "
            f"{np.round(g.approach, 2)} for {g.travel * 100:.0f} cm, palm turns "
            f"{np.rad2deg(g.turn):.0f} deg, jaw {np.rad2deg(g.jaw_tilt):.0f} deg "
            f"off horizontal, fingers clear the surface by "
            f"{g.clearance * 100:.1f} cm")


__all__ = ["Grasp", "plan", "describe", "hand_points", "OFFSET", "OFFSET_LEN"]
