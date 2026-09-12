"""`_servo_continuous`'s shared vocabulary, split out of stream_reach.py
(refactor step 4b family 2, pure extraction; decision 9's file-size cap).
This is the leaf of that split -- the continuous-servo tuning constants,
`_clipn`, and the `ServoState` dataclass that carries every local the
original 631-line loop closed over (fields documented with what they
replace) -- with no dependency on the other two pieces, so both can import
from here without a cycle:
  - `servo_phases.py` has the six per-phase functions (up / down /
    in-with-backoff / in / close / lift), each taking a `ServoState` instead
    of being a branch of one big if/elif chain;
  - `servo_ticks.py` has `_servo_setup` (the loop's pre-`while` setup),
    `_tick_prelude` / `_dispatch_phase` / `_tick_advance` (the per-tick
    telemetry+replan, phase routing, and common ref/IK/push/log tail that
    used to sit around that if/elif chain), and `_servo_continuous` itself.

Every phase/prelude/tail function across the split returns one of three
things, mirroring what the original branch did at that point in the loop:
  - `None` -- fall through to whatever runs next (the original branch's
    plain fall-off-the-end into the common tail, or the prelude's fall
    into the phase dispatch, or the dispatch's fall into the tail);
  - `"retick"` -- skip the rest of this iteration and loop again (the
    original branch's bare `continue`);
  - `(False, why)` or `(True, "")` -- end `_servo_continuous` now with that
    result (the original branch's `return ...`).
`state.tgt` / `state.v` are how a phase function hands its computed target
and speed to `_tick_advance`, replacing the original's `tgt, v = ...`.

`stream_reach.py` imports `_servo_continuous` from `servo_ticks.py` and the
tuning constants from here, and re-exports both (so
`from locomani.tasks.bottle.execution import _servo_continuous` and
`patch("locomani.tasks.bottle.execution._servo_continuous", ...)` keep
working unchanged) -- see that module's own docstring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from locomani.backends.sonic.primitives_const import HAND as _HAND

LIFT_MIN = 0.06  # the object must actually leave the table

# Fraction of servo ticks with the object inside the ego camera's central 90%
# (legacy callers only). New task code reads GraspResult.visibility so a
# failed or camera-less attempt cannot inherit a preceding episode's value.
LAST_EGO_VIS = None

# Continuous-servo tuning. One 50 Hz loop replaces the walk-stop-look-walk
# discrete servo: every tick reads the freshest privileged state, advances a
# cartesian reference a fraction of a millimetre, and re-solves warm-started
# IK — the way a teleoperator's hand moves, not the way a surveyor works.
V_TRAVEL = 0.15  # m/s toward the staging point (was 0.30: the WBC arm lags
# the reference by ~0.2 s x speed -- 6 cm at 0.30, measured 2026-09-03)
V_DESCEND = 0.10  # m/s dropping to the pregrasp (was 0.18, same lag argument)
V_APPROACH = 0.10  # m/s inside the jaw-alignment approach
# Pre-narrow the jaw while sliding in. Calibrated kinematically: fingertip
# gap ~ 6.1 cm at alpha 0, closing ~0.6 cm per 0.1 alpha, so 0.10 leaves
# ~5.6 cm over the 4.8 cm can — 8 mm of total margin instead of 13, but the
# first contact now lands on the closing envelope (which wraps) instead of a
# flat finger (which shoves): the graze-and-roll that killed 8 of 15
# episodes happens at the OPEN aperture's finger tips.
PRE_NARROW = 0.10
SAG_COMP = 0.018  # free-air sag is ~1.6-2 cm; 0.035 was measured
# against a WEDGED hand (the table held it down) and
# overcompensated: signed telemetry showed the palm
# riding 15-19 mm HIGH, closing the jaw at the rim
V_LIFT = 0.22
LEAD_MAX = 0.08  # m the reference may run ahead of the real palm
KP_TRACK = 0.7  # extra command past the reference, fights the sag
CLOSE_RAMP_S = 2.0  # 1.0 s fingers punched the can off the table

# hold the object this far out along the entry axis. dex1's fingertips
# LEAD the grasp centre by 3.3 cm along the approach (tips at wrist+18.5,
# grasp centre at wrist+15.15) — the whole slide-in geometry needs that
# lead budgeted or the tips reach the can while the servo still thinks
# it is 4-6 cm out.
SAFE_GAP = 0.17 if _HAND == "dex1" else 0.13
# (0.10 left only 5 cm of margin over a bad pregrasp, and 5 cm pregrasp
# errors do happen — the fingertips grazed the can twice)

# How much entry travel may remain when a CONTACT is allowed to trigger the
# close. Geometry, measured off the dex1 collision meshes: the pads span
# wrist+0.1206..0.1848 along the entry axis and the grasp centre sits at
# wrist+0.1515, so with e_ent still to travel the can centre sits at
# 0.1515+e_ent from the wrist. It is inside the pad span only for
# e_ent <= 0.033, while the fingertips (0.1848) first touch a 24 mm-radius
# can at e_ent ~ 0.0575. The old 0.06 gate therefore fired at the exact
# moment the TIPS grazed the can, with the can centre still 27 mm beyond
# them — closing there dealt the can a one-sided shove. Mined over 158
# labelled rounds: reaching the close phase through the servo's tolerance
# gate succeeded 70/70, while this contact shortcut succeeded 26/64, and
# every "toppled in the close phase" in the corpus came through it.
CLOSE_ENT_MAX = 0.02

# How far the can centre may sit from the grasp centre ALONG THE ENTRY AXIS
# and still be between the pads. Straight off the dex1 collision meshes: the
# pads span wrist+0.1206..0.1848 and the grasp centre is wrist+0.1515, so the
# can is seated for -0.031..+0.033; 0.030 keeps 3 mm at the near end and
# 3 mm at the far end.
#
# The tolerance gate used to ask for 0.012 (0.018 once riding or patient),
# which was tuned when the sim-side pin executed the IK solution exactly and
# the arm really could reach 0.012. Without the pin the arm stalls 2-3.7 cm
# short -- measured palm->ref pinned at LEAD_MAX with res 0.000, i.e. the
# joints solve and the policy does not deliver, which is the p90 of the
# 216-sample pin-off error. The can was then SEATED and the gate still
# refused, so the servo hovered until the 6 s wedge timer lifted it off, twice
# (one 118 s episode against 66-71 s for the successes). This is not the old
# 0.06 contact shortcut that succeeded 26/64: that one fired with the can
# centre 2.7 cm BEYOND the fingertips, where closing is a one-sided shove.
CLOSE_SEATED = 0.030

FREEZE_TRAVEL = 0.10  # residual travel below which the plan is frozen
STAGE_UP = 0.15  # staging point height above the pregrasp (was 0.10: only ~5 cm
# over the bottle top, eaten by the 6-7 cm tracking lag -> 'up' topples)
# Deliverable staging box, pelvis frame. The scene-model IK reaches further
# than the policy delivers (res 0.000 while the palm stalls): a pregrasp
# 0.5 m to the side solves fine and then never arrives — and so does one at
# the chest (x 0.10, z +0.08), where retracting the palm is equally
# undelivered. The box is the region the planner-era reach demonstrably
# worked in; candidates whose staging point leaves it are rejected before
# the arm ever chases them.
PRE_X_MIN, PRE_X_MAX = 0.17, 0.38
PRE_Z_MIN, PRE_Z_MAX = -0.24, 0.04
PRE_XY_MAX = 0.46  # horizontal reach from the pelvis


def _clipn(v, m):
    n = float(np.linalg.norm(v))

    return v if n <= m else v * (m / n)


@dataclass
class ServoState:
    """Every local `_servo_continuous`'s 631-line loop used to close over,
    now threaded through the phase functions below as one object instead.
    Grouped the way the original locals were introduced: the call's own
    parameters, `_servo_setup`'s derived/one-time values, the state that
    evolves tick to tick, and the handful recomputed fresh every tick but
    still read across a function boundary (`state.tgt`/`state.v` are how a
    phase function hands `_tick_advance` what the original branch's
    `tgt, v = ...` would have)."""

    # -- _servo_continuous's own parameters, unchanged ---------------------
    p: Any = None
    arm: Any = None
    obj_key: str = ""
    plan: Any = None
    side: str = "right"
    replan: Callable | None = None
    ghost: Callable | None = None
    pin: Callable | None = None
    surface_top: float | None = None
    execution: Any = None
    transport_guard: Any = None
    start_at_pregrasp: bool = False
    rotation_bias: Any = None
    reference_guard: Callable | None = None
    pregrasp_gap: float | None = None
    close_from_current: bool = False
    grip_target: Callable | None = None
    require_bilateral: bool = False
    capture_feedback: Any = None
    grasp_point_control: bool = False
    preserve_capture_reference: bool = False

    # -- derived once by _servo_setup, then read-only -----------------------
    gap: float = 0.0  # pregrasp_gap resolved against SAFE_GAP
    ENTRY_COL: int = 0  # 0 = palm +x (dex1 entry axis), 1 = palm +y (dex3)
    offset: np.ndarray = None  # GRASP_CENTRE_OFFSET, palm frame
    closed: np.ndarray = None  # this side's HAND_CLOSED_*
    obj0: np.ndarray = None  # object position when the servo call started
    z0: float = 0.0  # object height at servo start (the lift-height gate)
    deadline: float = 0.0  # wall-clock timeout for the whole call

    # -- evolve tick to tick (the loop's own mutable locals) -----------------
    rot_t: np.ndarray = None  # planned/replanned palm rotation
    entry: np.ndarray = None  # rot_t's entry-axis column
    pre_w: np.ndarray = None  # pregrasp point, world
    above_w: np.ndarray = None  # staging point above the pregrasp, world
    frozen: bool = True  # True once replanning has stopped (committed)
    last_replan: float = 0.0  # wall-clock time of the last accepted replan
    phase: str = "up"  # up|down|in|close|lift
    ok_ticks: int = 0  # consecutive in-tolerance ticks, gates "in"->"close"
    alpha: float = 0.0  # hand-closure ramp, 0 (open) .. 1 (closed)
    contact_alpha: float | None = None  # alpha at first sustained contact
    capture_bias: np.ndarray = None  # legacy capture-feedback offset, always
    # zero on every live path (kept so the tail's `cmd` sum is unchanged)
    full_f: np.ndarray | None = None  # low-passed palm-frame error (in phase)
    obj_f: np.ndarray | None = None  # low-passed can position
    close_tgt: np.ndarray | None = None  # frozen close-phase reference
    obj_in0: np.ndarray | None = None  # can position when the slide-in began
    backoff_until: float = 0.0  # wall-clock time a contact-flinch backoff ends
    touch_ticks: int = 0  # consecutive close-phase contact ticks
    trim_w: np.ndarray = None  # slow world-frame trim absorbing delivery bias
    lift_target: np.ndarray | None = None  # world target for the lift phase
    settle_until: float | None = None  # wall-clock time the post-lift settle ends
    t_prev: float = 0.0  # previous tick's wall-clock time (for dt)
    last_log: float = 0.0  # wall-clock time of the last status line
    res: float = 0.0  # latest IK residual
    in_since: float | None = None  # wall-clock time the "in" phase began
    ent_mark: tuple | None = None  # (entry-progress, wall-clock) wedge tracker
    stall_palm: np.ndarray | None = None  # palm position at the last stall reset
    stall_t: float = 0.0  # wall-clock time of the last stall reset
    vis_n: int = 0  # ego-camera ticks with the object visible
    tot_n: int = 0  # ego-camera ticks total
    in_lift: float = 0.0  # slide-in extra height, set in the in phase
    dbg_z: str = ""  # z-chain telemetry (line/cmd/palm/floor)
    touch_seen: set = field(default_factory=set)  # object-contact links
    # since the last status line
    recorded_phase: str | None = None  # last phase execution.enter() saw

    # -- recomputed every tick, read across a function boundary --------------
    s: dict = None  # this tick's observation snapshot
    now: float = 0.0
    dt: float = 0.0
    palm: np.ndarray = None
    rot_p: np.ndarray = None  # measured palm rotation, world
    controlled: np.ndarray = None  # palm, or the grasp-centre point if grasp_point_control
    obj: np.ndarray = None  # this tick's object position
    transport_phase: str | None = None  # "up"/"down" if the transport guard applies this tick
    tgt: np.ndarray = None  # this tick's phase-computed target, world
    v: float = 0.0  # this tick's phase-computed speed, m/s
