"""Where to put the hand, from the object's size and 6-D pose.

Replaces the three fixed offsets the grasp used to run on — stage 16 cm short
and 22 cm high, move directly over, descend to 2 cm above centre. That plan
assumed a hand that closes downwards. This one does not:

  * the palm's +x points forward and its z points up, whatever quaternion the
    wrist is sent (teleop_free/diag_wrist.py). The controller will yaw the palm
    but will not roll or pitch it, so the approach is horizontal and the pinch
    axis is horizontal too.
  * the pinch spans 6.1 cm at most, thumb surface to finger surface
  * the fingers hang 5.6 cm below the grip centre, so the grip point cannot
    come nearer than that to whatever the object is standing on

Those three turn "grasp the object" into a small geometry problem with an
honest failure case, which is the point: an object that cannot be grasped
should say so before the episode is spent finding out.

All measurements come from teleop_free/diag_hand.py and are re-derivable by
running it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Measured, teleop_free/diag_hand.py.
HAND_APERTURE = 0.061          # widest thumb-to-finger opening
FINGER_DROP = 0.056            # how far the fingers hang below the grip centre
FINGER_LEAD = 0.110            # how far the fingertips reach past the grip centre

# --- the tilted grasp -------------------------------------------------------
#
# Holding the palm level forces the crouch: level, the grip centre sits at the
# wrist's own height, so reaching a 0.62 m table means putting the pelvis at
# 0.65 — 14 cm of squat, held for the whole approach by parking the free hand
# at the floor, and the single most fragile thing in the episode.
#
# A person does not do that. The hand comes down at an angle, the web of the
# thumb rides the shoulder of the object, the fingers close on the side, and
# the wrist stays high. That is one rotation: pitch the palm nose-down about
# its own y, which IS the pinch axis, so the jaw still straddles the cylinder.
#
# PALM_PITCH_CMD is a command, not an angle. The orientation channel is heavily
# attenuated — measured on the live stack (teleop_free/diag_pitch.py), a 105 deg
# command yields 23 deg of actual palm pitch, and the response saturates there:
#
#     cmd   0d  45d  60d  75d  90d 105d 120d
#     got   1d   4d   9d  14d  20d  23d  24d
#
# Past 105 the extra pitch is nil and the parasitic roll runs away (-26d at 90,
# -40d at 105, -48d at 120), so 105 is the knee of the curve rather than a
# preference. It buys 7.5 cm of lift between the wrist and the grip point,
# which takes the crouch from 14 cm to 2.
PALM_PITCH_CMD = 105.0
# What the hand hangs below the grip centre once pitched — the tilted
# equivalent of FINGER_DROP, measured the same way, by placing the ghost at the
# pose and taking the lowest vertex. Nearly half the level figure, because the
# fingers come in over the object instead of under it.
PITCHED_FINGER_DROP = 0.030
# Where on the barrel to close, below the rim. High on the object used to be
# the point (it minimised the crouch), but 1.5 cm of rim is thinner than the
# streamed channel's ~1 cm delivery noise: the video shows the fingers closing
# at rim height, reporting contact, and lifting air. 4.5 cm buries the jaw in
# the barrel with 3 cm of finger clearance left above the table (can 12 cm,
# fingers reach 4.5 cm below the grip centre).
GRIP_BELOW_RIM = 0.035
# The palm's axes, in the pelvis frame, once PALM_PITCH_CMD has been commanded
# and the hand has settled — measured, not derived (diag_pitch.py). Columns are
# +x (along the fingers), +y (the thumb side, which is the direction the object
# slides in along) and +z (the palm normal).
#
# The +y column is the reason this constant has to exist. Asking for pitch also
# rolls the hand 40 deg, so the thumb-side opening no longer points sideways —
# it points 40 deg DOWN. The level planner computes the staging offset as
# cross(up, approach), a strictly horizontal vector, and staging 14 cm along
# that while the jaw actually opens downward leaves the object 21 cm out along
# the palm's real y instead of the 10 cm intended. The alignment servo then
# tries to correct a 40 deg frame error inside its own palm frame, sweeps the
# command 0.57 m across, and knocks the can over on the way — which is what
# "can shoved 5 cm while closing in (touched R:index_1)" was.
PALM_PITCHED_ROT = np.array([
    [0.9197, -0.2905, 0.2641],
    [0.0184, 0.7039, 0.7101],
    [-0.3922, -0.6482, 0.6527],
])

# Margins on top of the measurements. The pinch one matters most: closing on
# something the exact width of the aperture jams rather than grips.
PINCH_MARGIN = 0.010
CLEARANCE_MARGIN = 0.015
# How far back along the ENTRY axis to stage, beyond the object's own
# half-width. Three terms, all measured:
#
#   thumb-side block  0.015  where the open hand first fouls the object
#   reach residual    0.08   what reach() actually lands within
#   object radius     0.02
#
# The 0.18 this used to be carried FINGER_LEAD (0.11) instead of the first
# term. That was right when the hand went in nose-first along its +x, where
# the middle fingertip does lead the grip centre by 11 cm — but the approach
# now slides in sideways from the thumb side, which diag_grasp_pose measured
# as clear to within 1.5 cm. So 7 cm of the old staging distance was clearance
# for an axis the hand no longer travels along, and it was spent where the arm
# can least afford it: sideways.
#
# That 7 cm is not cosmetic. Live traces put the staging point 0.55 m to the
# robot's own right at 0.45 m forward — |xy| 0.71 m against a measured 0.50 m
# palm envelope — because GRASP_BEARING offsets the object to that side and
# the staging offsets it further to the same side. The two add.
PREGRASP_GAP = 0.11
# An object axis counts as horizontal if it tilts less than this from the
# ground plane; the pinch has to straddle a horizontal span.
HORIZONTAL_TOL = 0.5           # |axis . z|
# Two spans within this of each other count as equal, so a symmetric object is
# approached head-on rather than from an arbitrary side.
SPAN_TIE = 0.005
GRIP_LIFT = 0.010              # keep the grip this far off its lower limit


@dataclass
class GraspPose:
    """A plan the arm can execute, or a refusal with a reason."""

    ok: bool
    why: str = ""
    grip: np.ndarray | None = None        # world point for the grip centre
    pregrasp: np.ndarray | None = None    # world point to stage at
    approach: np.ndarray | None = None    # unit horizontal, robot -> object
    entry: np.ndarray | None = None       # unit world, how the hand slides in
    wrist_quat: np.ndarray | None = None  # pelvis-frame wrist orientation
    width: float = 0.0                    # the span being pinched
    grip_height: float = 0.0              # above the supporting surface


def _rot(quat) -> np.ndarray:
    w, x, y, z = quat

    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)],
                    dtype=np.float32)


def _pitch_quat(deg: float) -> np.ndarray:
    """Nose-down about y. The COMMAND, not the angle it produces."""
    a = np.deg2rad(deg) / 2.0

    return np.array([np.cos(a), 0.0, np.sin(a), 0.0], dtype=np.float32)


def _qmul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b

    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], dtype=np.float32)


def grasp_pose(obj_pos, obj_quat, box: dict, surface_top: float,
               pelvis_xy, pelvis_yaw: float,
               approach_dir=None, pitched: bool = False) -> GraspPose:
    """Plan a horizontal pinch of `box` at `obj_pos`/`obj_quat`.

    `box` is the body-frame bounding box from scene_meta — half extents and
    centre — so the live quaternion is all that is needed to know which way the
    object is currently long.

    `approach_dir` fixes the palm's facing instead of deriving it from the
    robot-to-object line. It exists because the two cannot be independent on
    this stack: the wrist's orientation channel is poisoned (commanding yaw
    also pitches the palm ~25 deg down, teleop_free/diag_wrist.py), so the palm
    faces wherever the PELVIS faces, full stop. Meanwhile the arm's workspace
    wants the object 20-40 deg to its own side, not dead ahead — facing the
    object square-on cost 5 of 10 episodes to staging stalls. Passing the
    pelvis-forward direction here lets the body stand in the arm's sweet spot
    while the plan honestly describes the palm the controller will produce.
    The object does not need to be faced: it enters the jaw from the side.
    """
    obj_pos = np.asarray(obj_pos, dtype=float)
    rot = _rot(obj_quat)
    half = np.asarray(box["half_extent"], dtype=float)
    centre = obj_pos + rot @ np.asarray(box.get("centre", [0, 0, 0]), dtype=float)

    # --- which span can the fingers straddle? -----------------------------
    # Each of the object's own axes, in the world. Only the near-horizontal
    # ones are candidates: the pinch axis is the palm's y, which stays level.
    if approach_dir is not None:
        towards = np.array([approach_dir[0], approach_dir[1], 0.0], dtype=float)
    else:
        towards = np.array([centre[0] - pelvis_xy[0],
                            centre[1] - pelvis_xy[1], 0.0])
    n = float(np.linalg.norm(towards))
    towards = towards / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])

    cands = [(rot[:, i] / np.linalg.norm(rot[:, i]), 2 * half[i])
             for i in range(3)
             if abs(rot[2, i]) <= HORIZONTAL_TOL
             and 2 * half[i] <= HAND_APERTURE - PINCH_MARGIN]
    if not cands:
        spans = ", ".join(f"{2 * h * 100:.1f}" for h in half)
        return GraspPose(False, why=(
            f"no horizontal span fits the {HAND_APERTURE * 100:.1f} cm hand "
            f"(object spans {spans} cm)"))

    # Among spans that fit, prefer the narrowest — but treat near-ties as ties.
    # A cylinder's two horizontal spans are exactly equal, and picking whichever
    # comes first in the model sends the robot round to grasp it side-on for no
    # reason. Break the tie by taking the axis most perpendicular to the line
    # from the robot, which is the one it can walk straight up to.
    narrowest = min(w for _, w in cands)
    tied = [(a, w) for a, w in cands if w <= narrowest + SPAN_TIE]
    pinch_axis, width = min(tied, key=lambda aw: abs(float(aw[0] @ towards)))

    # --- how high can the grip sit? ---------------------------------------
    # The fingers hang below the grip centre, so the grip point has a floor.
    # An object shorter than that floor simply cannot be taken this way — a
    # sphere on a table is the standard example, and saying so here is much
    # cheaper than discovering it as "no contact after closing".
    drop = PITCHED_FINGER_DROP if pitched else FINGER_DROP
    floor = surface_top + drop + CLEARANCE_MARGIN
    top = centre[2] + float(np.abs(rot[2, :] * half).sum())
    if floor > top - 0.01:
        return GraspPose(False, why=(
            f"too short to grip: the fingers need the grip point at "
            f"{floor - surface_top:.3f} m above the surface but the object "
            f"only reaches {top - surface_top:.3f} m"))
    if pitched:
        # High on the barrel, deliberately. The level grasp takes the object's
        # mid-height because a level hand is equally happy anywhere up the
        # side; the tilted one is not, because grip height is what sets how far
        # the shoulder must come down, and every centimetre higher is a
        # centimetre less crouch. Measured at the 105 deg command: gripping
        # 1.5 cm under the rim needs the pelvis at 0.773 against a standing
        # 0.793, while mid-barrel needs 0.743.
        grip_z = float(np.clip(top - GRIP_BELOW_RIM, floor + GRIP_LIFT, top - 0.010))
    else:
        # Not right on the floor: keep a centimetre in hand so a little sag in
        # the arm does not put the fingertips through the table.
        grip_z = float(np.clip(centre[2], floor + GRIP_LIFT, top - 0.015))

    # --- approach, perpendicular to the pinch, pointing away from the robot -
    if approach_dir is not None:
        # The palm's facing is fixed by the body; the only question is whether
        # the object's graspable span lies across the jaw, i.e. roughly along
        # the palm's y. For a cylinder any horizontal axis does. For an
        # elongated object this is where an ungraspable heading gets refused —
        # the fix being to walk to a different bearing, since the wrist cannot
        # be yawed to compensate.
        approach = towards
        jaw = np.cross([0.0, 0.0, 1.0], approach)
        if len(tied) >= 2:
            # Two horizontal spans tie: the footprint is rotationally
            # symmetric (a cylinder), so ANY horizontal direction pinches at
            # the same width and the box axes' directions are meaningless.
            # Take the jaw direction itself — asking a symmetric object's
            # bounding box which way it points refused a perfectly graspable
            # can because both box axes sat 40-50 deg off the jaw.
            pinch_axis = jaw
        elif abs(float(pinch_axis @ jaw)) < 0.8:
            better = [a for a, w in tied if abs(float(a @ jaw)) >= 0.8]
            if not better:
                return GraspPose(False, why=(
                    "no graspable span lies across the jaw at this heading — "
                    "stand at a different bearing"))
            pinch_axis = better[0]
    else:
        perp = np.cross([0.0, 0.0, 1.0], pinch_axis)
        perp[2] = 0.0
        n = float(np.linalg.norm(perp))
        if n < 1e-6:                   # pinch axis was vertical after all
            return GraspPose(False, why="pinch axis is not horizontal")
        perp = perp / n
        approach = perp if float(perp @ towards) > 0 else -perp

    grip = np.array([centre[0], centre[1], grip_z])

    # The hand slides in SIDEWAYS, along the palm's +y, not along the fingers.
    #
    # Sliding the object along each palm axis with the hand open and finding
    # the first contact (teleop_free/diag_grasp_pose.py) gives:
    #
    #     +x  straight in, along the fingers   blocked 0.105 m out, by middle_1
    #     -y  from the finger side             blocked 0.120 m out
    #     +z  from above                       blocked 0.085 m out
    #     +y  from the thumb side              clear to within 0.015 m
    #
    # Only the thumb side is open. Pushing along the fingers — which is what
    # every batch did — puts the middle fingertip on the object 10.8 cm before
    # the grip centre arrives, so the last 10.8 cm of every approach was
    # shoving the object across the table rather than closing around it. The
    # traces show exactly that: perfect lateral alignment, 1 mm, while the
    # displacement climbs 0 -> 1 -> 3 -> 4 cm.
    #
    # With no wrist roll the palm's +y is the approach rotated a quarter turn
    # to the left, which is also how a person takes a cup off a table: the hand
    # arrives beside it and closes across, never nose-first.
    if pitched:
        # Take the entry axis from the palm the controller will actually
        # produce, not from an idealised level one. cross(up, approach) is only
        # the thumb side when the palm is flat; tilted, the hand is rolled 40
        # deg and its opening faces down as much as sideways, so staging along
        # the horizontal puts the object nowhere near the jaw.
        entry = (_rot(_yaw_quat(pelvis_yaw)) @ PALM_PITCHED_ROT)[:, 1]
    else:
        entry = np.cross([0.0, 0.0, 1.0], approach)
    entry /= max(float(np.linalg.norm(entry)), 1e-6)
    half_along = float(np.abs(rot.T @ entry) @ half)
    pregrasp = grip - entry * (half_along + PREGRASP_GAP)

    # The palm's +x follows the wrist yaw, so ask for the yaw that lines it up
    # with the approach. Expressed in the pelvis frame, which is the frame the
    # wrist command is interpreted in.
    heading = float(np.arctan2(approach[1], approach[0]))

    # Yaw stays zero in the pitched case for the same reason it always did —
    # commanding wrist yaw drags the palm nose-down as a side effect, and here
    # the nose-down is being asked for deliberately, so mixing the two would
    # make the achieved pitch depend on the heading. The body already faces
    # where the palm should face; this quaternion only carries the tilt.
    quat = (_pitch_quat(PALM_PITCH_CMD) if pitched
            else _yaw_quat(heading - pelvis_yaw))

    return GraspPose(
        True, grip=grip, pregrasp=pregrasp, approach=approach, entry=entry,
        wrist_quat=quat, width=float(width),
        grip_height=float(grip_z - surface_top))


def describe(g: GraspPose) -> str:
    if not g.ok:
        return f"no grasp: {g.why}"

    return (f"grip {np.round(g.grip, 3)} ({g.grip_height * 100:.1f} cm above "
            f"the surface), face {np.round(g.approach, 2)}, slide in along "
            f"{np.round(g.entry, 2)}, pinch {g.width * 100:.1f} cm, stage "
            f"{np.linalg.norm(g.grip - g.pregrasp) * 100:.0f} cm to the side")


__all__ = ["GraspPose", "grasp_pose", "describe", "HAND_APERTURE",
           "FINGER_DROP"]
