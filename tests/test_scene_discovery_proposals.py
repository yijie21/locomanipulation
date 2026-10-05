"""The proposal loop with a scripted Claude: verdicts go back, revisions are re-checked, the loop stops when stable."""

from __future__ import annotations

import copy
import json

from test_scene_discovery_feasibility import CAP, META

from locomani.scene_discovery import feasibility, proposals


def task(id_, goal, frm="", missing=""):
    return {"id": id_, "room": "kitchen", "goal": goal, "prompts": ["Pick it up.", "Grab it.", "Take it."],
            "from": frm, "missing": missing, "reason": ""}  # fmt: skip


class ScriptedClaude:
    """Returns the replies in order and keeps every request."""

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []

    def __call__(self, prompt, schema, system):
        assert schema is proposals.SCHEMA and system is proposals.SYSTEM
        self.requests.append(json.loads(json.dumps(prompt, default=float)))
        return {"tasks": self.replies.pop(0), "notes": ""}


def test_verdicts_go_back_and_each_revision_is_checked_again():
    claude = ScriptedClaude(
        [task("pick_book", ["held(book)"]), task("pick_bottle", ["held(bottle)"]), task("typo", ["held(teapot)"])],
        # the book task becomes a basket task, the typo is dropped, a repeated id is renamed
        [
            task("pick_bottle", ["held(bottle)"], "pick_bottle"),
            task("bottle_in_basket", ["in(bottle, basket)"], "pick_book"),
            task("pick_bottle", ["held(high_bottle)"], "", "grasp from a 0.90 m top"),
        ],  # fmt: skip
    )
    rows, history = proposals.discover(copy.deepcopy(META), CAP, claude, iterations=1, log=lambda *_: None)
    first, revision = claude.requests
    assert "checked" not in first and set(first["scene"]["objects"]) >= {"bottle", "book", "basket"}
    assert not first["scene"]["objects"]["basket"]["pickable"] and first["scene"]["objects"]["basket"]["container"]
    by = {t["id"]: t for t in revision["checked"]}
    assert by["pick_bottle"]["bucket"] == 1 and by["pick_bottle"]["untested"] == []
    assert by["pick_book"]["bucket"] == 2 and by["pick_book"]["untested"][0]["gap"] == "grasp_single.object_size"
    assert by["typo"]["invalid"].startswith("held(teapot)")
    assert [r["id"] for r in rows] == ["pick_bottle", "bottle_in_basket", "pick_bottle_2"]
    assert [r["bucket"] for r in rows] == [1, 1, 2]
    assert rows[2]["missing"] == "grasp from a 0.90 m top" and rows[0]["missing"] == ""
    assert rows[1]["provenance"] == {"proposed_by": "claude", "model": "", "prompt_sha": proposals.prompt_sha(),
                                     "iteration": 1, "from": "pick_book"}  # fmt: skip
    assert [h["buckets"] for h in history] == [{1: 1, 2: 1, 3: 1}, {1: 2, 2: 1, 3: 0}]


def test_the_loop_stops_when_a_revision_changes_no_goal():
    same = [task("pick_bottle", ["held(bottle)"])]
    claude = ScriptedClaude(same, same, same)
    rows, history = proposals.discover(copy.deepcopy(META), CAP, claude, iterations=5, log=lambda *_: None)
    assert len(claude.requests) == 2 and [h["changed"] for h in history] == [True, False]


def test_the_report_relates_proposals_to_the_template_space():
    checker = feasibility.Checker(copy.deepcopy(META), CAP)
    templates = [checker.check(t).to_dict() for t in feasibility.templates(META)]
    claude = ScriptedClaude(
        [
            task("a", ["in(bottle, basket)", "upright(bottle)"]),  # the template, set down standing
            task("b", ["held(bottle)", "upright(book)"]),  # a conjunction over two objects is not a template
        ]
    )
    rows, _ = proposals.discover(copy.deepcopy(META), CAP, claude, iterations=0, log=lambda *_: None)
    rep = proposals.report(rows, templates)
    assert rep["tasks"] == 2 and rep["outside_template_space"] == 1
    assert rep["template_bucket1_found"] == 1 and rep["template_bucket1"] >= 2
