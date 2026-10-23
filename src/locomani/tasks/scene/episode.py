"""One scene-task round (TODO 27.2.3): reset the robot at the start's spawn, move the task objects to their spots, let
them settle, run the plan (skills over the motor layer), hold, and judge the goal predicates on the privileged state.
Returns the repository's `RoundResult` (execution = the round record) and the motor trace arrays."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from locomani.backends.scalebfm.motor import Outcome
from locomani.pipeline.execution_record import RoundResult
from locomani.scene_discovery.predicates import parse
from locomani.tasks.scene.start import Start
from locomani.tasks.scene.view import TruthView
from locomani.tasks.scene.world import SceneWorld, Task

SETTLE_S = 1.0  # standing still after the objects are placed
HOLD_S = 2.0  # the goal is judged after this long standing still at the end of the plan
# TODO 36 (review 2026-10-22): simulated seconds a whole episode (settle, plan, hold) may take; the motor primitives time
# out one by one but a skill may retry for ever. The longest recorded round is 98 s.
EPISODE_LIMIT_S = 300.0


class EpisodeTimeLimit(BaseException):
    """Raised by the limited tick; a BaseException so a skill's own `except Exception` cannot swallow it."""


@dataclass
class SkillContext:
    """What a skill sees: the world (privileged scene state, or a scene.view world view for a package that sets
    WORLD_VIEW), the motor (through that view's RobotView), the task, a random generator, and the log of every step's
    outcome (the round record keeps it)."""

    world: SceneWorld | TruthView
    task: Task
    rng: np.random.Generator
    log: list = field(default_factory=list)

    @property
    def motor(self):
        return self.world.motor

    def note(self, name: str, out: Outcome, **extra) -> Outcome:
        row = dict(step=name, ok=out.ok, why=out.why, t0=round(out.t0, 2), t1=round(out.t1, 2), **out.info, **extra)
        self.log.append({k: (v if not isinstance(v, np.ndarray) else v.round(4).tolist()) for k, v in row.items()})
        return out


def goal_keys(goal) -> list[str]:
    keys = []
    for g in goal:
        for a in parse(g).args:
            if isinstance(a, str) and a not in keys:
                keys.append(a)
    return keys


def run_episode(world: SceneWorld, task: Task, start: Start, plan, *, hold_s: float = HOLD_S,
                limit_s: float = EPISODE_LIMIT_S, view: bool = False):  # fmt: skip
    """`plan(ctx) -> Outcome`. Returns (RoundResult, trace arrays). A plan still stepping at `limit_s` simulated seconds
    ends the round as a failure, why "episode time limit" (TODO 36, review 2026-10-22). `view`: the plan sees the world
    through scene.view.TruthView (the package set WORLD_VIEW) instead of the SceneWorld; the judge reads the SceneWorld."""
    m, wall = world.motor, time.time()
    m.track = {world.meta["objects"][k]["body"]: world.free[k] for k in task.objects}
    m.reset(start.robot_xy, start.robot_yaw)
    for k, (pos, quat) in start.placements.items():
        m.place(world.meta["objects"][k]["body"], pos, quat)
    seated = {k: round(world.seat(k), 4) for k in start.placements}  # placements drawn on older surface heights
    seated = {k: v for k, v in seated.items() if v > 0}
    ctx = SkillContext(TruthView(world) if view else world, task, np.random.default_rng(start.seed))
    # the stepping layer's cap: every primitive ticks through m.tick (hand programs included)
    had_tick, tick = "tick" in vars(m), m.tick

    def limited() -> bool:
        if m.t >= limit_s:
            raise EpisodeTimeLimit
        return tick()

    m.tick, timed_out, error = limited, False, ""
    try:
        ctx.note("settle", m.hold(SETTLE_S))
        try:
            out = plan(ctx)
        except Exception as e:  # a skill's bug is the round's failure, recorded, not the batch's crash
            out, error = Outcome(False, f"skill error: {type(e).__name__}: {e}", m.t, m.t), repr(e)
        if m.fell_at is None:
            ctx.note("hold", m.hold(hold_s))
    except EpisodeTimeLimit:
        timed_out, out = True, Outcome(False, "episode time limit", m.t, m.t)
        ctx.note("episode_time_limit", out)
    finally:
        if had_tick:
            m.tick = tick
        else:
            del m.tick
    keys = [k for k in goal_keys(task.goal) if k in world.meta["objects"]]
    ok, why = world.check(task.goal)(world.state(keys))
    fell = m.fell_at is not None
    execution = dict(
        task=task.id,
        goal=list(task.goal),
        start=start.to_dict(),
        plan_ok=bool(out.ok),
        plan_why=out.why,
        error=error,
        steps=ctx.log,
        fell=fell,
        seated=seated,
        t_end=round(m.t, 2),
        wall_s=round(time.time() - wall, 1),
    )
    if timed_out:
        return RoundResult(False, "episode time limit", False, execution), m.arrays()
    return RoundResult(bool(ok and not fell), why if not ok else ("fell" if fell else ""), False, execution), m.arrays()
