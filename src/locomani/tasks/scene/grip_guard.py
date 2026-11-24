"""TODO 41.2: before a jaw closes, check that the hand got where the last reach sent it; if not, back off and try once
from a stance moved by the miss.

Why here and not in the skills: the pattern is the same for every object and grasp kind, and the skills the loop writes
close the jaw without looking (outputs/scene_rsi_20261016/stage_checks_20261027/README.md: a jaw closed 3 cm or more off
after an entry, 5 cm after a descend, won 1 round of 18). `install(world)` wraps the motor's `grip`, so skill code runs
unchanged. Off unless RSI_GRIP_GUARD=1 (scripts/rsi/scene_round.make_world; batch.motor_signature carries it while on).

What a retry does. The miss `e` = commanded jaw centre - measured. The arm did reach the measured point from this
stance, so the stance is moved by the horizontal part of `e` (shortened until the pelvis stays on free floor): the hand
first pulls back to the pre-grasp point less that step, the robot steps keeping its heading, and the last two reaches
run again as the skill commanded them. One retry a round. A second miss closes the jaw anyway: not closing is a lost
round for certain.

It leaves the round alone when it cannot know better: the last arm command was not a one-hand reach of this side (a
hand program, a two-hand reach), no earlier reach of this side gives a point to pull back to, the hand is held up
rather than short (jaw above the target with almost no horizontal miss: it rests on something), or no free step exists.
Every decision is appended to `world.grip_guard_log`, which the round record keeps.
"""

from __future__ import annotations

import os

import numpy as np

ENV = "RSI_GRIP_GUARD"
VERSION = 1  # part of the motor signature while on: bump when a retry's behaviour changes
ENTRY_TOL_M = 0.03  # jaw centre this far from the last reach's target or more: do not close yet ...
DESCEND_TOL_M = 0.05  # ... this far when that reach was a descent
DESCENT_SHARE = 0.7  # a reach is a descent when this share of its commanded move is downward
RETRIES = 1  # per round
MAX_STEP_M = 0.30
MIN_STEP_M = 0.03  # a shorter step is inside the walk's own stop error
HELD_UP_XY_M = 0.02  # jaw above the target and less than this off sideways: resting on something, a step cannot help
STEP_SHRINK = (1.0, 0.75, 0.5)  # shares of the wanted step tried until the pelvis lands on free floor


def enabled() -> bool:
    return os.environ.get(ENV) == "1"


def _reaches(calls: list, side: str) -> list[dict]:
    """The trailing run of one-hand reaches of `side`: the commands since the last walk or other primitive."""
    run = []
    for row in reversed(calls):
        if row.get("name") != "reach" or not row.get("args") or row["args"][0] != side:
            break
        run.append(row)
    return run[::-1]


def _args(row: dict):
    a = row["args"]
    pos = np.asarray(a[1], float)
    rot = np.asarray(a[2], float) if len(a) > 2 and a[2] is not None else None
    kw = dict(row.get("kwargs") or {})
    if "via" in kw:
        kw["via"] = [np.asarray(v, float) for v in kw["via"]]
    return pos, rot, kw


def decide(calls: list, side: str, jaw, pelvis_xy, is_free) -> dict:
    """What to do before closing `side`: dict(act="close", why=...) or dict(act="retry", step=xy, ...). Pure."""
    run = _reaches(calls, side)
    if not run:
        return dict(act="close", why="last command was not a reach of this hand")
    goal, _, _ = _args(run[-1])
    e = goal - np.asarray(jaw, float)
    err = float(np.linalg.norm(e))
    descent = False
    if len(run) > 1:
        move = goal - _args(run[-2])[0]
        n = float(np.linalg.norm(move))
        descent = n > 1e-6 and -move[2] / n >= DESCENT_SHARE
    tol = DESCEND_TOL_M if descent else ENTRY_TOL_M
    facts = dict(err_cm=round(100 * err, 2), tol_cm=round(100 * tol, 1), descent=bool(descent),
                 miss=[round(float(v), 4) for v in e])  # fmt: skip
    if err < tol:
        return dict(act="close", why="on target", **facts)
    if len(run) < 2:
        return dict(act="close", why="no pre-grasp point to pull back to", **facts)
    exy = e[:2]
    if e[2] < 0 and float(np.linalg.norm(exy)) < HELD_UP_XY_M:
        return dict(act="close", why="held up, not short", **facts)
    n = float(np.linalg.norm(exy))
    want = exy * min(1.0, MAX_STEP_M / max(n, 1e-9))
    for share in STEP_SHRINK:
        step = want * share
        if float(np.linalg.norm(step)) < MIN_STEP_M:
            break
        if is_free(np.asarray(pelvis_xy, float) + step):
            return dict(act="retry", step=[round(float(v), 4) for v in step], **facts)
    return dict(act="close", why="no free step toward the target", **facts)


def install(world) -> None:
    """Wrap `world.motor.grip` with the check. The retry's own commands are ordinary top-level motor calls, so the
    round's `calls` and trace show them."""
    m = world.motor
    grip = m.grip
    world.grip_guard_log = []
    state = dict(round_calls=None, used=0)

    def guarded(side: str, **grip_kw):
        if state["round_calls"] is not m.calls:  # `reset` starts a new list: a new round
            state.update(round_calls=m.calls, used=0)
            world.grip_guard_log = []
        if side in ("left", "right") and state["used"] < RETRIES and m.fell_at is None and not m._call_depth:
            d = decide(m.calls, side, m.jaw_centre(side)[0], m.pelvis()[0][:2], world.occ.is_free)
            row = dict(t=round(m.t, 2), side=side, **d)
            world.grip_guard_log.append(row)
            if d["act"] == "retry":
                state["used"] += 1
                run = _reaches(m.calls, side)
                (pre, pre_rot, pre_kw), (goal, rot, kw) = _args(run[-2]), _args(run[-1])
                step = np.asarray(d["step"], float)
                back = pre.copy()
                back[:2] -= step
                m.reach(side, back, pre_rot)
                if m.fell_at is None:
                    m.walk_to(m.pelvis()[0][:2] + step, m.yaw(), face="fixed")
                if m.fell_at is None:
                    m.reach(side, pre, pre_rot, **pre_kw)
                if m.fell_at is None:
                    out = m.reach(side, goal, rot, **kw)
                    row["after_cm"] = out.info.get("jaw_err_cm")
                row["t_end"] = round(m.t, 2)
        return grip(side, **grip_kw)

    m.grip = guarded
