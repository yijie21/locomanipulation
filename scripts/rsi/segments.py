"""Segments of a task (TODO 48.17.6, user 2026-11-23: "长任务…把不容易失败和容易失败的段区分开来然后再进行分别的处理").

A round record's `steps` are the task's phases in order, each with `ok` and its seconds `t0`..`t1`; the step a
failure is blamed on (the backend's BLAME map from the failure class) has `ok` False. Over the incumbent's
diagnostic rounds this gives, per task and per step: how many rounds reached it, how many of them failed there,
what share of the task's losses that is, and when the step starts. A step that holds most of the losses is `hot`; a
step that many rounds reached and none failed at is `stable`. Pure functions of round dicts (batch.outcomes): the
cards, the role contexts and the Composer's view of a draft read them."""

from __future__ import annotations

import statistics

HOT_SHARE = 0.4  # a hot step holds at least this share of the task's losses ...
HOT_MIN = 2  # ... and at least this many of them
STABLE_MIN = 5  # a stable step was reached by at least this many rounds and failed in none


def blamed(r: dict) -> str | None:
    """The step a failed round's failure is blamed on (the first step with ok False), None for a win or no steps."""
    if r.get("ok"):
        return None
    return next((s.get("step") for s in r.get("steps") or [] if not s.get("ok", True)), None)


def segment_table(rounds) -> dict:
    """{task: [row per step, in the order the rounds run them]}; a row: step, reached, failed (blamed there),
    pass_rate (of the rounds that reached it), loss_share (of the task's losses), t_reach (median start, s), kind
    ("hot", "stable" or "")."""
    by: dict[str, list[dict]] = {}
    for r in rounds:
        by.setdefault(r.get("task") or "", []).append(r)
    out: dict = {}
    for task, rs in sorted(by.items()):
        order: dict[str, float] = {}
        reached: dict[str, int] = {}
        starts: dict[str, list[float]] = {}
        for r in rs:
            for i, s in enumerate(r.get("steps") or []):
                name = s.get("step")
                order.setdefault(name, i)
                order[name] = min(order[name], i)
                reached[name] = reached.get(name, 0) + 1
                if isinstance(s.get("t0"), (int, float)):
                    starts.setdefault(name, []).append(float(s["t0"]))
        losses = [r for r in rs if not r.get("ok")]
        failed: dict[str, int] = {}
        for r in losses:
            b = blamed(r)
            if b is not None:
                failed[b] = failed.get(b, 0) + 1
        top = max(failed.values(), default=0)
        rows = []
        for name in sorted(order, key=lambda k: order[k]):
            n, f = reached[name], failed.get(name, 0)
            share = f / len(losses) if losses else 0.0
            kind = ("hot" if f == top and f >= HOT_MIN and share >= HOT_SHARE
                    else "stable" if f == 0 and n >= STABLE_MIN else "")  # fmt: skip
            rows.append(dict(step=name, reached=n, failed=f, pass_rate=round(1 - f / n, 2) if n else None,
                             loss_share=round(share, 2), t_reach=round(statistics.median(starts[name]), 1)
                             if starts.get(name) else None, kind=kind))  # fmt: skip
        out[task] = rows
    return out


def kinds(table: dict) -> dict:
    """{task: {step: kind}} for the steps that are hot or stable."""
    return {t: {r["step"]: r["kind"] for r in rows if r["kind"]} for t, rows in table.items()}


def table_lines(table: dict, rounds) -> list[str]:
    """The table as the cards' header has it: one line per task with losses, its steps in order with the failures
    blamed there, the hot step marked, and the seconds a lost round spent before its blamed step."""
    by: dict[str, list[dict]] = {}
    for r in rounds:
        by.setdefault(r.get("task") or "", []).append(r)
    lines = []
    for task, rows in table.items():
        losses = [r for r in by.get(task, []) if not r.get("ok")]
        if not losses:
            continue
        parts = []
        for x in rows:
            mark = " HOT" if x["kind"] == "hot" else " stable" if x["kind"] == "stable" else ""
            parts.append(f"{x['step']} {x['failed']}/{x['reached']}{mark}")
        before, total = 0.0, 0.0
        for r in losses:
            b = blamed(r)
            steps = r.get("steps") or []
            t_end = next((s.get("t1") for s in reversed(steps) if isinstance(s.get("t1"), (int, float))), None)
            t_b = next(
                (s.get("t0") for s in steps if s.get("step") == b and isinstance(s.get("t0"), (int, float))), None
            )
            if t_end and t_b is not None:
                before, total = before + t_b, total + t_end
        spent = (
            f"; lost rounds spent {round(100 * before / total)} % of their time before the blamed step" if total else ""
        )
        lines.append(f"- {task} ({len(losses)} lost): " + " > ".join(parts) + spent)
    return lines


def touched(changed: dict, table: dict) -> dict:
    """Where a draft's changed rounds first left the incumbent's path ({label: (task, step)}), counted by the kind of
    that step in the task: {"stable": {...}, "hot": {...}, "other": {...}} with "task:step" -> rounds."""
    k = kinds(table)
    out: dict = {"stable": {}, "hot": {}, "other": {}}
    for task, step in changed.values():
        kind = k.get(task, {}).get(step) or "other"
        key = f"{task}:{step}"
        out[kind][key] = out[kind].get(key, 0) + 1
    return out
