"""Did the robot's hands do what the task says? Beside BiGym's own success check (TODO 49.5).

BiGym's checks are lenient (docs/tasks.md of the clone: put_cups "any contact counts", sandwich_remove "does not test that the
spatula was used", a drawer, door or tray is just a joint position): a knee, a hip or a forearm that pushes the drawer passes. The
benchmark's own verdict is `official(outcome)` (success and never fell, bigym/loco/eval/protocol.py:is_success); `verdict` adds
what the hands did, from the environment server's contact record: every robot body touching a scene body that has joints of its
own (a drawer, a door, a free object), per control step.

A hand is a body of the Dex1 gripper (a body name containing `dex1`: the base link and the two finger links). Every other robot body
is the body: wrist, forearm, elbow, shoulder, torso, pelvis, legs. Rules (`outcome["contacts"]` is {"<robot body>|<object>":
[step, ...]}):

    reach       no scene body is to be touched; the benchmark checks the hand's pinch centre against the ball. ok = official.
    contact     (drawers, doors, trays, objects) ok = official, a hand on a task body on at least HAND_STEPS steps, and the body on
                a task body on fewer than BODY_STEPS steps.
    why         ""  or one of: not_success, fell, no_hand, body_push.

The server's contact record has no floor and no robot-on-robot pairs. A body that touches only its own fixtures is not a task body.
"""

from __future__ import annotations

HAND_STEPS = 5  # a hand on a task body on at least this many control steps (0.1 s)
BODY_STEPS = 3  # the body may touch a task body on fewer steps than this (a graze, 0.06 s)
REACH_TASKS = ("reach_target_single", "reach_target_multi_modal", "reach_target_dual")


def is_hand(robot_body: str) -> bool:
    """A body of the Dex1 gripper. The model's prefix (`g1_29dof_with_dex1_1/`) names the gripper too, so only the body's own name counts."""
    return "dex1" in robot_body.rsplit("/", 1)[-1]


def official(outcome: dict) -> bool:
    """BiGym's protocol: the task reported success and the robot never fell (bigym.loco.eval.protocol.is_success)."""
    return bool(outcome.get("success")) and not bool(outcome.get("fell"))


def contact_steps(contacts: dict) -> tuple[int, int, list[str]]:
    """(steps with a hand on a task body, steps with the body on one, the body parts that touched), steps counted once each."""
    hand, body, parts = set(), set(), set()
    for key, steps in (contacts or {}).items():
        robot, _, _obj = key.partition("|")
        if is_hand(robot):
            hand |= set(steps)
        else:
            body |= set(steps)
            parts.add(robot.split("/")[-1])
    return len(hand), len(body), sorted(parts)


def verdict(task: str, outcome: dict) -> dict:
    """{ok, why, official, hand_steps, body_steps, body_parts} for one finished episode."""
    off = official(outcome)
    hand, body, parts = contact_steps(outcome.get("contacts") or {})
    out = dict(official=off, hand_steps=hand, body_steps=body, body_parts=parts, ok=False, why="")
    if not outcome.get("success"):
        out["why"] = "not_success"
    elif outcome.get("fell"):
        out["why"] = "fell"
    elif task in REACH_TASKS:
        out["ok"] = True
    elif hand < HAND_STEPS:
        out["why"] = "no_hand"
    elif body >= BODY_STEPS:
        out["why"] = "body_push"
    else:
        out["ok"] = True
    return out
