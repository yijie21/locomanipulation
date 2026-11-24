"""Did the robot's hands do what the task says? Beside HumanoidArena's own success check (TODO 49.5b).

The benchmark's checks need no hand contact (survey, 2026-11-24): P&PBox is "the box centre lies over a shelf surface at the target
layer's height", DoubleDesk "the hammer's centre of mass is inside the bin", OpenDoor (strict) "root past the door plane, latch
unlocked, leaf >= 60 degrees". A shove, a kick or a body push passes. The benchmark's own verdict is `official(outcome)` (the reward
reached 1 before a fall); `verdict` adds what the hands did, from the environment server's record: every step on which a robot link
pressed on the task object (the contact sensor, filtered to that object, force > 1 N) and how far the object travelled.

A hand is a link of the Dex3 hand (a name containing `hand`: base, palm, finger links). Every other link is the body: wrist, forearm,
elbow, shoulder, torso, pelvis, legs. Rules (`outcome["contacts"]` is {"<link>": [step, ...]}):

    ok = official, a hand on the object on at least HAND_STEPS steps, the body on it on fewer than BODY_STEPS steps, and the
         object moved at least MOVE_MIN metres (it did not just start where the check wants it).
    why  "" or one of: not_success, fell, no_record (no contact record for the task: OpenDoor), no_hand, body_push, not_moved.

The record has the task's one movable object: no floor and no robot-on-robot pairs. OpenDoor has no object record (the door leaf is
part of the scene articulation); a compliant verdict there is not available, so its `ok` stays False with `no_record`.
"""

from __future__ import annotations

HAND_STEPS = 5  # a hand on the object on at least this many control steps (0.1 s)
BODY_STEPS = 3  # the body may touch it on fewer steps than this (a graze, 0.06 s)
MOVE_MIN = 0.10  # the object travelled at least this far from where it started (m)
NO_RECORD = ("open_door",)


def is_hand(link: str) -> bool:
    return "hand" in link


def official(outcome: dict) -> bool:
    """The benchmark's rule: the task reward reached 1 and the fall detector had not confirmed a fall."""
    return bool(outcome.get("success")) and not bool(outcome.get("fell"))


def contact_steps(contacts: dict) -> tuple[int, int, list[str]]:
    """(steps with a hand on the object, steps with the body on it, the body links that touched), steps counted once each."""
    hand, body, parts = set(), set(), set()
    for link, steps in (contacts or {}).items():
        if is_hand(link):
            hand |= set(steps)
        else:
            body |= set(steps)
            parts.add(link)
    return len(hand), len(body), sorted(parts)


def verdict(task: str, outcome: dict) -> dict:
    """{ok, why, official, hand_steps, body_steps, body_parts, obj_move} for one finished episode."""
    off = official(outcome)
    hand, body, parts = contact_steps(outcome.get("contacts") or {})
    move = float(outcome.get("obj_move") or 0.0)
    out = dict(official=off, hand_steps=hand, body_steps=body, body_parts=parts, obj_move=move, ok=False, why="")
    if not outcome.get("success"):
        out["why"] = "not_success"
    elif outcome.get("fell"):
        out["why"] = "fell"
    elif task in NO_RECORD:
        out["why"] = "no_record"
    elif hand < HAND_STEPS:
        out["why"] = "no_hand"
    elif body >= BODY_STEPS:
        out["why"] = "body_push"
    elif move < MOVE_MIN:
        out["why"] = "not_moved"
    else:
        out["ok"] = True
    return out
