"""Claude proposes a scene's tasks; the feasibility checker answers; Claude revises (plan decision 8, TODO 19.10).

`discover` is the loop. Claude reads the scene (rooms, surfaces, the objects a task can use) and the measured
capability (`configs/scene_discovery/capability.yaml`, written for this reader) and proposes tasks as goal
predicates with 3-5 English instructions (decision 7). Every task goes through `feasibility.Checker`; the
verdicts -- bucket, the untested checks with their measured value against the measured range, the hard limits,
an invalid goal -- go back, and Claude returns the whole list again: tasks kept, revised toward the envelope,
dropped, or added, and for a task that stays out of reach the missing skill (bucket 2) or the reason (bucket 3).
The loop ends when a revision changes no goal, or after `iterations` revisions.

`ask(prompt, schema, system) -> dict` is the one Claude call (`properties.ask` bound to a model and a log in the
script); tests pass a fake. Output rows follow the task file format (docs/handoff/scene-file-format-20261003.md §6).
"""

from __future__ import annotations

import hashlib
import json

from locomani.scene_discovery import feasibility, predicates

TASK = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "room": {"type": "string"},
        "goal": {"type": "array", "items": {"type": "string"}},
        "prompts": {"type": "array", "items": {"type": "string"}},
        "from": {"type": "string"},  # the earlier id this task keeps or revises; "" for a new one
        "missing": {"type": "string"},  # still bucket 2: the skill or predicate that would make it composable
        "reason": {"type": "string"},  # still bucket 3: why it is out of reach, citing capability.yaml keys
    },
    "required": ["id", "room", "goal", "prompts", "from", "missing", "reason"],
}
SCHEMA = {
    "type": "object",
    "properties": {"tasks": {"type": "array", "items": TASK}, "notes": {"type": "string"}},
    "required": ["tasks", "notes"],
}
SYSTEM = """You discover manipulation tasks for a Unitree G1 humanoid (two Dex1 two-finger grippers, a whole-body
walking controller) in one simulated apartment. A task is a goal: a conjunction of predicates over the scene's keys,
judged from simulator state, never by you:
  held(X)            X lifted off its surface in the robot's hand
  on(X, S)           X resting on surface S (a key from `surfaces`)
  in(X, C)           X inside the open container C
  upright(X)         X standing within 15 deg of its upright axis
  near(X, Y, d)      X within d metres of Y (xy)
  in_room(robot, R)  the robot inside room R
Use only keys that appear in the request. X must be an object with "pickable": true.

The `capability` is what the robot has been measured to do; each entry says measured / measured_one / untested /
unavailable. Propose tasks a person in this apartment would plausibly ask for, and spread them over objects,
surfaces and containers -- breadth matters, not twenty variations of one task. Keep each task inside one room.
Prefer tasks inside measured settings, but also propose useful tasks just outside them: a checker sorts every
task into 1 (composable now), 2 (plausible, a skill or measurement is missing) or 3 (out of capability), and the
bucket-2 tasks decide which skill is built next.

Each task has a short snake_case id, the room, the goal, 3-5 different English instructions a person would
say (natural wording, using the objects' phrases, not their keys), and `from`, `missing`, `reason` (empty strings
when proposing). When the request carries `checked`, it is your previous list with the checker's verdict per task:
return the whole list again. Keep bucket-1 tasks unchanged. For bucket 2 and 3, revise the goal when a nearby
variant would land inside the measured envelope and still be a sensible task (another object, surface or
container), or keep it and fill `missing` (bucket 2: the skill or measurement it needs, as a one-line spec) or
`reason` (bucket 3: why, citing the capability keys). Drop a task only when it is invalid and has no sensible
variant. You may add new tasks. `from` names the id a returned task came from ("" for a new one)."""


def prompt_sha() -> str:
    return hashlib.sha256((SYSTEM + json.dumps(SCHEMA, sort_keys=True)).encode()).hexdigest()[:12]


def scene_view(meta: dict) -> dict:
    """What Claude reads about the scene: rooms, surfaces, and the objects a task can name."""
    objects = {}
    for key, o in sorted(meta["objects"].items()):
        sem = o.get("semantic") or {}
        pickable = bool(
            o.get("free") and sem.get("graspable", "none") != "none" and (sem.get("drop_test") or {}).get("ok")
        )
        if not (pickable or o.get("container")):
            continue
        objects[key] = {
            "phrase": sem.get("phrase", key),
            "category": (sem.get("adt") or {}).get("category"),
            "size_m": [round(2 * v, 3) for v in o["box"]["half_extent"]],
            "mass_kg": sem.get("mass_kg"),
            "room": sem.get("room"),
            "on_surface": o.get("surface"),
            "pickable": pickable,
            "hands": sem.get("graspable"),
            "container": bool(o.get("container")),
        }
    surfaces = {
        key: {
            "phrase": s.get("phrase", key),
            "room": s.get("room"),
            "top_m": round(float(s["top"]), 2),
            "size_m": [round(2 * v, 2) for v in s["half_extent"]],
            "robot_can_stand_at": bool(s.get("approach_edges")),
        }
        for key, s in sorted(meta["surfaces"].items())
    }
    return {"rooms": sorted((meta.get("room") or {}).get("rooms") or {}), "surfaces": surfaces, "objects": objects}


def verdict(result: feasibility.Result) -> dict:
    """The checker's answer for one task, as Claude reads it back."""
    untested = [
        {k: c.to_dict()[k] for k in ("step", "name", "measured", "bound", "gap", "setting") if k in c.to_dict()}
        for c in result.checks
        if c.outcome == feasibility.UNTESTED
    ]
    return {"bucket": result.bucket(), "untested": untested, "blocked": result.blocked, "invalid": result.invalid}


def _row(task: dict, result: feasibility.Result, iteration: int, model: str) -> dict:
    return {
        "id": task["id"],
        "room": task["room"],
        "goal": list(task["goal"]),
        "prompts": list(task["prompts"]),
        "bucket": result.bucket(),
        "plan": None,  # bucket 1: written in the next step (decision 10), not by the proposal loop
        "missing": task.get("missing", "") if result.bucket() == 2 else "",
        "reason": task.get("reason", "") if result.bucket() == 3 else "",
        "gaps": result.gaps,
        "blocked": result.blocked,
        "invalid": result.invalid,
        "checks": [c.to_dict() for c in result.checks],
        "provenance": {
            "proposed_by": "claude",
            "model": model,
            "prompt_sha": prompt_sha(),
            "iteration": iteration,
            "from": task.get("from", ""),
        },
    }


def _unique(tasks: list[dict]) -> list[dict]:
    """Claude's ids made unique (a repeated id gets a suffix), so verdicts map back one to one."""
    seen, out = {}, []
    for t in tasks:
        n = seen.get(t["id"], 0)
        seen[t["id"]] = n + 1
        out.append(t if n == 0 else dict(t, id=f"{t['id']}_{n + 1}"))
    return out


def discover(
    meta: dict, capability: dict, ask, *, iterations: int = 2, per_room: int = 30, model: str = "", log=print
) -> tuple[list[dict], list[dict]]:
    """(final task rows, per-iteration history). History rows: {iteration, tasks: [rows], buckets, changed}."""
    checker = feasibility.Checker(meta, capability)
    base = {"scene": scene_view(meta), "capability": capability, "tasks_per_room": per_room}
    reply = ask(base, SCHEMA, SYSTEM)
    history, previous = [], None
    for iteration in range(iterations + 1):
        tasks = _unique(reply["tasks"])
        results = [checker.check({"id": t["id"], "goal": t["goal"]}) for t in tasks]
        rows = [_row(t, r, iteration, model) for t, r in zip(tasks, results)]
        goals = sorted(json.dumps(sorted(t["goal"])) for t in tasks)
        changed = previous is None or goals != previous
        buckets = {b: sum(r["bucket"] == b for r in rows) for b in (1, 2, 3)}
        history.append({"iteration": iteration, "tasks": rows, "buckets": buckets, "changed": changed,
                        "notes": reply.get("notes", "")})  # fmt: skip
        log(f"[discover] iteration {iteration}: {len(rows)} tasks, buckets {buckets}, changed {changed}")
        if not changed or iteration == iterations:
            return rows, history
        previous = goals
        checked = [
            {**{k: t[k] for k in ("id", "room", "goal", "prompts", "missing", "reason") if k in t}, **verdict(r)}
            for t, r in zip(tasks, results)
        ]
        reply = ask({**base, "checked": checked}, SCHEMA, SYSTEM)
    return rows, history


def report(rows: list[dict], templates: list[dict]) -> dict:
    """Counts for decision 1, and how the proposals relate to the template space the checker enumerates."""

    def key(goal):  # upright(X) beside a move of X is a constraint on the move, not another task (feasibility)
        preds = [predicates.parse(g) for g in goal]
        moved = {p.args[0] for p in preds if p.name in ("in", "on", "held")}
        return json.dumps(sorted(str(p) for p in preds if not (p.name == "upright" and p.args[0] in moved)))

    template_bucket = {key(t["goal"]): t.get("bucket") for t in templates}
    proposed = {key(r["goal"]) for r in rows}
    gaps: dict = {}
    for r in rows:
        if r["bucket"] == 2:
            for g in r["gaps"]:
                gaps[g] = gaps.get(g, 0) + 1
    return {
        "tasks": len(rows),
        "buckets": {b: sum(r["bucket"] == b for r in rows) for b in (1, 2, 3)},
        "invalid": sum(bool(r["invalid"]) for r in rows),
        "outside_template_space": sum(k not in template_bucket for k in proposed),
        "template_bucket1_found": sum(1 for k, b in template_bucket.items() if b == 1 and k in proposed),
        "template_bucket1": sum(1 for b in template_bucket.values() if b == 1),
        "gaps": dict(sorted(gaps.items(), key=lambda kv: -kv[1])),
    }
