"""The row one episode leaves (TODO 49.5): what the loops read, with the run's switches and the judge's verdict."""

from __future__ import annotations

from locomani.bench.bigym2 import judge


def row_fields(eid: str, task: str, pool: str, seed: int) -> dict:
    return dict(id=eid, task=task, pool=pool, seed=seed)


def make_row(base: dict, outcome: dict, verdict: dict, info: dict, marks: dict, wall_s: float, extra: dict) -> dict:
    """`success` is the compliant success (the benchmark's check passed, the hands did it, nobody fell); `bench_success` BiGym's own."""
    return dict(
        base,
        success=bool(verdict["ok"]),
        bench_success=bool(verdict["official"]),
        why=verdict["why"],
        fell=bool(outcome["fell"]),
        steps=int(outcome["steps"]),
        termination=outcome.get("termination"),
        hand_steps=verdict["hand_steps"],
        body_steps=verdict["body_steps"],
        body_parts=verdict["body_parts"],
        marks=marks,
        flow=_plain(info),
        wall_s=round(wall_s, 2),
        error=None,
        **extra,
    )


def error_row(base: dict, err: str, wall_s: float, extra: dict) -> dict:
    return dict(base, success=False, bench_success=False, why="error", fell=False, steps=0, hand_steps=0, body_steps=0,
                body_parts=[], marks={}, flow={}, wall_s=round(wall_s, 2), error=err[-600:], **extra)  # fmt: skip


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, float | int | str | bool) or x is None:
        return x
    return str(x)


def summary(rows: list[dict]) -> dict:
    n = len(rows)
    return dict(n=n, compliant=sum(r["success"] for r in rows), bench=sum(r["bench_success"] for r in rows),
                errors=sum(r["error"] is not None for r in rows), why=sorted({r["why"] for r in rows if r["why"]}),
                wall_s=round(sum(r["wall_s"] for r in rows), 1)) | dict(judge_rules=dict(hand_steps=judge.HAND_STEPS, body_steps=judge.BODY_STEPS))  # fmt: skip
