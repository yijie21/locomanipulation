"""The planner's `speed` field is not m/s, and its meaning belongs to the CHECKPOINT.

Every walking constant in this repo used to be a bare command value chosen off
one checkpoint's behaviour. That is the bug this module exists to prevent: on
2026-09-11 the default moved to policy/sonic_v1_1 and `APPR_COARSE = 0.30`,
which delivers 0.203 m/s on policy/release, delivered 0.044 m/s -- a 4.6x
slower approach that spent 37.8 s of do_approach's 45 s budget and left the
stall rule (1.5 cm in 5 s, round_phases_approach.py) a 2.9x margin. Nothing
failed loudly; the rounds simply parked 12 cm short and were discarded as
"arrival left the object out of reach" (ledger bottle.md 17, 21).

THE SHAPE OF THE MAP -- read this before picking a number

It is a STEP, not a slope, and the step sits in a different place on each
checkpoint (measured 2026-09-11, both checkpoints in one session, 11 commands
x 3 repeats, `python -m locomani.backends.sonic.diag_shuffle`; full table in
ledger bottle.md 21.1):

    command   0.185  0.20   0.215  0.26   0.28   0.30   0.32   0.35   0.40   0.45
    v1.1      0.008  0.009  0.014  0.009  0.007  0.044  0.067  0.083  0.187  0.273
    release   0.013  0.018  0.016  0.123  0.179  0.203  0.239  0.245  0.348  0.400

release steps 0.016 -> 0.123 m/s between 0.215 and 0.26 (7.9x); v1.1 steps
0.007 -> 0.044 between 0.28 and 0.30 (6.6x). So:

  * pick by DELIVERED velocity, never by carrying a command value across;
  * never interpolate INSIDE a step -- that interval is bistable. release's own
    0.215 sits on its step edge and returned 0.007 / 0.016 / 0.039 on three
    identical legs, which is why the 0.040 recorded for `APPR_MID` in
    primitives_tuning.py never reproduced;
  * below the step the two checkpoints are the same order (0.007-0.018 m/s),
    above it v1.1 delivers 28-68% of release. v1.1's usable approach range is
    therefore 3x (0.067-0.203) against release's 11x: it has NO equivalent slow
    crawl, and that is a property of the checkpoint, not something a constant
    here can fix.

WHAT IS DELIBERATELY NOT CHANGED

`APPR_MID` / `APPR_SHUFFLE` stay at release's command values on v1.1 even
though they deliver 0.85x / 0.48x of release's speed, because no command
delivers release's 0.016-0.018 -- the step jumps straight over it. Only the one
constant Stage A showed to be badly wrong (COARSE) moves, so a pilot reads one
variable. primitives_tuning.NAV_SPEED / NAV_SPEED_MIN are also left alone:
matching release's 0.400 m/s would need a command above 0.45, which is
unmeasured, and both clear the stall rule by 91x / 28x on v1.1 anyway.
"""

from __future__ import annotations

from locomani import paths

# Delivered velocity (m/s, median of 3 legs) per commanded speed, per checkpoint.
# Provenance: ledger bottle.md 21.1; raw legs in
# outputs/diagnostics/gait_speed_v11_20260911{,_knee}/<arm>/legs.json.
MEASURED: dict[str, dict[float, float]] = {
    "sonic_v1_1": {
        0.185: 0.008,
        0.20: 0.009,
        0.215: 0.014,
        0.26: 0.009,
        0.28: 0.007,
        0.30: 0.044,
        0.32: 0.067,
        0.35: 0.083,
        0.37: 0.151,
        0.40: 0.187,
        0.45: 0.273,
    },
    "release": {
        0.185: 0.013,
        0.20: 0.018,
        0.215: 0.016,
        0.26: 0.123,
        0.28: 0.179,
        0.30: 0.203,
        0.32: 0.239,
        0.35: 0.245,
        0.37: 0.260,
        0.40: 0.348,
        0.45: 0.400,
    },
}

# (coarse, mid, shuffle) for the tabletop approach. release's row is the
# historical production triple, unchanged, so a release run is byte-identical to
# every dataset collected before the switch. v1.1's COARSE is 0.40 -- a MEASURED
# point (0.187 m/s, +-8% over repeats, 92% of release's 0.203), not the 0.41 the
# inverse lookup suggests, because 0.41 is interpolation inside a 0.05 gap and
# the 0.1 s of budget it would buy is not worth an unmeasured command.
APPROACH_BANDS: dict[str, tuple[float, float, float]] = {
    "sonic_v1_1": (0.40, 0.215, 0.20),
    "release": (0.30, 0.215, 0.20),
}


def active_policy() -> str:
    """The checkpoint name the deploy is running, e.g. ``sonic_v1_1``."""
    return paths.SONIC_POLICY.rsplit("/", 1)[-1]


def approach_bands(policy: str | None = None) -> tuple[float, float, float]:
    """(coarse, mid, shuffle) commands for this checkpoint's tabletop approach.

    Raises for a checkpoint nobody has measured: a silent fallback to another
    checkpoint's numbers is the exact failure this module exists to stop, and
    the caller is a phase that is about to walk the robot at that speed.
    """
    name = policy or active_policy()
    try:
        return APPROACH_BANDS[name]
    except KeyError:
        raise KeyError(
            f"no measured gait bands for checkpoint {name!r} (have: {sorted(APPROACH_BANDS)}). "
            f"Measure it first -- python -m locomani.backends.sonic.diag_shuffle "
            f"--speeds 0.20,0.26,0.30,0.35,0.40,0.45 --repeats 3 -- then add a row here "
            f"and record it in docs/ledger/bottle.md."
        ) from None


def delivered(command: float, policy: str | None = None) -> float | None:
    """Measured m/s for a commanded speed, or None if that command was not measured."""
    return MEASURED.get(policy or active_policy(), {}).get(command)
