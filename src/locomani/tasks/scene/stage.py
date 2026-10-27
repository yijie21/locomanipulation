"""TODO 33.16 (plan section 2.4, D12): a grasp step with a check after it and a bounded retry.

The contract the rules cite: every grasp step of a skill goes through `stage(ctx, name, run, check, on_fail)`. `run()`
is a motor call (walk_to, reach, grip, ...) returning its Outcome; `check(out) -> (ok, values)` judges the step from
the measured state (the standard checks below, or the skill's own); `on_fail(attempt, values)` returns a callable that
tries again (re-walk, re-approach, open and re-grip) or None to give up. Every attempt is logged in the round record
under the step's name with its check values (`check`) and attempt number (`attempt`), so the failure classes and the
Reviewer read what each check saw. One `Budget` per round (ctx.stage_budget, made on first use) caps the retries and
the simulated seconds they take; a fall ends the step at once. The thresholds and the on_fail actions are the skill's
(the loop writes them); the defaults here are starting points.

The stall check (`track_ratio`) follows MobiAgent (arXiv 2610.03476): over the step, the measured wrist's
displacement against the commanded wrist's; well under 1 means the arm was told to move and did not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from locomani.backends.scalebfm.motor import Outcome

SIDES = ("left", "right")


@dataclass
class Budget:
    """Retries left for the whole round, shared by every stage."""

    max_retries: int = 3
    max_extra_s: float = 40.0
    retries: int = 0
    extra_s: float = 0.0

    def allows(self) -> bool:
        return self.retries < self.max_retries and self.extra_s < self.max_extra_s

    def spend(self, seconds: float) -> None:
        self.retries += 1
        self.extra_s += max(0.0, seconds)


def budget_of(ctx) -> Budget:
    if getattr(ctx, "stage_budget", None) is None:
        ctx.stage_budget = Budget()
    return ctx.stage_budget


def _log(ctx, name: str, out: Outcome, values: dict | None, attempt: int) -> None:
    info = {k: v for k, v in out.info.items() if k not in ("check", "attempt")}
    ctx.note(name, Outcome(out.ok, out.why, out.t0, out.t1, info), check=values or {}, attempt=attempt)


def stage(ctx, name: str, run, check=None, on_fail=None, budget: Budget | None = None) -> Outcome:
    """Run a step, check it, retry while `on_fail` offers a retry and the budget lasts. Returns the last attempt's
    Outcome; its `ok` is the check's verdict (the motor's own `ok` when there is no check)."""
    m, budget = ctx.motor, budget or budget_of(ctx)
    attempt, step = 0, run
    while True:
        out = step()
        ok, values = (out.ok, {}) if check is None else check(out)
        _log(ctx, name, out, values, attempt)
        if m.fell_at is not None:
            return Outcome(False, f"fell at {m.fell_at:.2f} s", out.t0, out.t1, dict(values))
        if ok:
            return Outcome(True, "", out.t0, out.t1, dict(values))
        retry = on_fail(attempt, values) if on_fail is not None else None
        if retry is None:
            return Outcome(False, f"{name}: check failed {_brief(values)}, no retry", out.t0, out.t1, dict(values))
        if not budget.allows():
            return Outcome(False, f"{name}: check failed {_brief(values)}, retry budget spent", out.t0, out.t1,
                           dict(values))  # fmt: skip
        t_before = m.t
        step = lambda retry=retry, t_before=t_before: _timed(retry, budget, m, t_before)  # noqa: E731
        attempt += 1


def _timed(retry, budget: Budget, m, t_before: float) -> Outcome:
    out = retry()
    budget.spend(m.t - t_before)
    return out


def _brief(values: dict) -> str:
    return "(" + ", ".join(f"{k} {v}" for k, v in values.items() if not isinstance(v, (list, dict))) + ")"


# -- standard checks: pure functions of the measured state -> (ok, values) --------------------------------------------


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def stance_check(m, pel_xy, yaw: float, tol_xy: float = 0.05, tol_yaw_deg: float = 8.0):
    p, _ = m.pelvis()
    err = float(np.linalg.norm(np.asarray(p[:2], float) - np.asarray(pel_xy, float)))
    yaw_err = math.degrees(_wrap(m.yaw() - float(yaw)))
    values = dict(stop_err_cm=round(100 * err, 2), yaw_err_deg=round(yaw_err, 2))
    return err <= tol_xy and abs(yaw_err) <= tol_yaw_deg, values


def track_ratio(m, side: str, since: float | None = None) -> float | None:
    """Measured wrist displacement / commanded wrist displacement over the trace from `since` (default: the last run
    of the last primitive). None when the command moved less than 1 cm."""
    rows = m.trace
    if not rows:
        return None
    if since is None:
        name, k = rows[-1]["primitive"], len(rows) - 1
        while k > 0 and rows[k - 1]["primitive"] == name:
            k -= 1
        rows = rows[k:]
    else:
        rows = [r for r in rows if r["t"] >= since - 1e-9]
    if len(rows) < 2:
        return None
    i = SIDES.index(side)
    cmd = float(np.linalg.norm(np.asarray(rows[-1]["ref_wrists"][i]) - np.asarray(rows[0]["ref_wrists"][i])))
    act = float(np.linalg.norm(np.asarray(rows[-1]["wrists"][i]) - np.asarray(rows[0]["wrists"][i])))
    return None if cmd < 0.01 else round(act / cmd, 3)


def reach_check(m, side: str, target_xyz, tol_m: float = 0.03, min_track: float = 0.12, since: float | None = None):
    jaw, _ = m.jaw_centre(side)
    err = float(np.linalg.norm(np.asarray(jaw, float) - np.asarray(target_xyz, float)))
    ratio = track_ratio(m, side, since)
    values = dict(jaw_err_cm=round(100 * err, 2), track_ratio=ratio)
    return err <= tol_m and (ratio is None or ratio >= min_track), values


def grip_check(m, side: str, body: int, min_open_m: float = 0.002):
    opening = float(m.jaw_opening(side))
    both = body in m.between_fingers(side)
    values = dict(opening_cm=round(100 * opening, 2), both_fingers=bool(both))
    return both and opening >= min_open_m, values


def lift_check(world, key: str, z0: float, min_rise_m: float = 0.03):
    p, _ = world.object_pose(key)
    rise = float(p[2]) - float(z0)
    return rise >= min_rise_m, dict(rise_cm=round(100 * rise, 2))


def bimanual_grip_check(m, body: int):
    """Both hands on `body` (TODO 38.5): the fingers of each side touch it (what `held(X)`'s two-hand branch needs;
    a box wider than a jaw is never between one hand's fingers). Values: the touching sides, each jaw's opening."""
    touching = sorted(m.touching(body))
    values = dict(touching=touching, opening_cm=[round(100 * float(m.jaw_opening(s)), 2) for s in SIDES])
    return touching == list(SIDES), values


def bimanual_lift_check(world, m, key: str, z0: float, min_rise_m: float = 0.03):
    """The object rose `min_rise_m` and both hands are still on it (TODO 38.5): a lift where one hand slid off or the
    object tipped out is a failed step even when it left the surface."""
    ok_rise, values = lift_check(world, key, z0, min_rise_m)
    touching = sorted(m.touching(world.free[key]))
    values["touching"] = touching
    return ok_rise and touching == list(SIDES), values
