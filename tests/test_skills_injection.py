"""Failure injection over fakes: when it fires, what it pins, what it traces; and the paired summary."""

from __future__ import annotations

import json

import numpy as np
import pytest

from locomani.skills import ArmOwner, Executor, HeldObject, Session
from locomani.skills.contract import Outcome
from locomani.skills.injection import (
    Injection,
    Injector,
    load_injection,
    paired_summary,
    recovery_segment,
    regrasped,
    render_report,
    sign_test_p,
)

ALL = frozenset(ArmOwner)


class FakeReset:
    def __init__(self, stack):
        self.stack, self.calls = stack, []

    def pin_object(self, name, pos=None, quat=None):
        self.calls.append((name, pos, quat))
        if pos is not None:
            self.stack.objects[name]["pos"] = list(pos)
            if quat is not None:
                self.stack.objects[name]["quat"] = list(quat)


class FakeStack:
    def __init__(self):
        self.objects = {
            "soup_can": {"pos": [-0.4, -0.28, 0.692], "quat": [1.0, 0.0, 0.0, 0.0], "touching": ["dining_table"]},
            "basket": {"pos": [-0.34, 0.0, 0.698], "quat": [0.7, 0.0, 0.0, 0.7], "touching": []},
        }

    def obs(self):
        return {"fallen": False, "objects": {k: dict(v) for k, v in self.objects.items()}}


class Scene:
    meta = {
        "objects": {
            "soup_can": {"init_pos": [-0.4, -0.28, 0.692], "init_quat": [1.0, 0.0, 0.0, 0.0]},
            "basket": {"init_pos": [-0.34, 0.0, 0.698]},
        }
    }


class Prim:
    def __init__(self, name, ok=True, mutate=None):
        self.name, self.requires, self.yields, self.ok, self.mutate = name, ALL, None, ok, mutate

    def preconditions(self, session, *, target: str = "", profile: str = ""):
        return []

    def run(self, session, *, target: str = "", profile: str = ""):
        if self.mutate:
            self.mutate(session)
        return Outcome(self.ok, "" if self.ok else "[x] no")


def grasp(session):
    session.held = HeldObject("soup_can", ("right",), {"right": np.zeros(3)})
    session.hands["right"] = 1.0
    session.stack.objects["soup_can"]["pos"] = [-0.8, -0.28, 0.85]  # lifted, in the hand
    session.stack.objects["soup_can"]["touching"] = ["right_palm"]


def make(tmp_path, injection):
    stack = FakeStack()
    session = Session(stack=stack, scene=Scene(), reset=FakeReset(stack))
    catalogue = {
        "approach": Prim("approach"),
        "grasp_single": Prim("grasp_single", mutate=grasp),
        "place": Prim("place", ok=False),
    }
    executor = Executor(session, catalogue, trace_path=tmp_path / "trace.jsonl")
    injector = Injector(injection, log=lambda *_: None, sleep=lambda _s: None)
    return session, executor, injector


def test_move_target_fires_once_after_the_named_approach_and_traces_the_event(tmp_path):
    session, executor, injector = make(tmp_path, Injection("move_target", "approach", "soup_can", (0.0, -0.17)))
    e = executor.step("approach", {"target": "basket", "profile": "p"})
    assert injector.after(e, executor) is None  # approach to the goal is not the trigger
    e = executor.step("approach", {"target": "soup_can", "profile": "p"})
    event = injector.after(e, executor)
    assert event["kind"] == "move_target" and event["after_call"] == 2
    assert event["before"] == [-0.4, -0.28, 0.692] and event["commanded"] == [-0.4, -0.45, 0.692]
    assert event["moved_m"] == pytest.approx(0.17) and event["landed_err_m"] == 0.0
    pin, release = session.reset.calls
    assert pin[0] == "soup_can" and pin[1] == [-0.4, -0.45, 0.692] and pin[2] == [1.0, 0.0, 0.0, 0.0]
    assert release == ("soup_can", None, None)
    # fires once
    e = executor.step("approach", {"target": "soup_can", "profile": "p"})
    assert injector.after(e, executor) is None and len(session.reset.calls) == 2
    lines = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [line.get("event") for line in lines] == [None, None, "injection", None]
    assert "session" in lines[2] and lines[2]["target"] == "soup_can"


def test_drop_held_returns_the_held_object_to_its_init_pose_and_leaves_held_stale(tmp_path):
    session, executor, injector = make(tmp_path, Injection("drop_held", "grasp_single", offset_m=(0.1, 0.0)))
    e = executor.step("approach", {"target": "soup_can", "profile": "p"})
    assert injector.after(e, executor) is None
    e = executor.step("grasp_single", {"target": "soup_can", "profile": "p"})
    event = injector.after(e, executor)
    assert event["target"] == "soup_can" and event["commanded"] == [-0.3, -0.28, 0.692]
    assert event["before"] == [-0.8, -0.28, 0.85] and event["moved_m"] == pytest.approx(0.5)
    assert event["moved_z_m"] == pytest.approx(-0.158)
    assert session.held is not None and session.held.key == "soup_can"  # nothing told the session


def test_failed_or_rejected_calls_never_fire(tmp_path):
    session, executor, injector = make(tmp_path, Injection("move_target", "place", "soup_can", (0.1, 0.0)))
    e = executor.step("place", {"target": "soup_can", "profile": "p"})
    assert not e["outcome"]["ok"] and injector.after(e, executor) is None
    e = executor.step("nope")
    assert "rejected" in e and injector.after(e, executor) is None
    assert session.reset.calls == [] and injector.fired is None


def test_load_injection_validates(tmp_path):
    path = tmp_path / "i.json"
    path.write_text(json.dumps({"kind": "move_target", "after": "approach", "target": "x", "offset_m": [0, 0.2]}))
    inj = load_injection(path)
    assert inj.offset_m == (0.0, 0.2) and inj.settle_s == 0.5
    with pytest.raises(ValueError):
        Injection("push_robot", "approach")
    with pytest.raises(ValueError):
        Injection("move_target", "approach", offset_m=(float("nan"), 0.0))


def test_recovery_segment_reads_calls_after_the_event():
    trace = [
        {"primitive": "approach", "outcome": {"ok": True}},
        {"event": "injection"},
        {"primitive": "grasp_single", "outcome": {"ok": False}},
        {"primitive": "approach", "rejected": []},
        {"primitive": "approach", "outcome": {"ok": True}},
        {"primitive": "abort", "reason": "done: x"},
    ]
    assert recovery_segment(trace) == ["grasp_single!", "approach?", "approach", "abort"]
    assert recovery_segment(trace[:1]) is None


def test_sign_test_and_paired_summary():
    assert sign_test_p(0, 0) == 1.0
    assert sign_test_p(5, 0) == pytest.approx(2 / 32)
    assert sign_test_p(3, 3) == 1.0
    plan = [
        {"round": i, "seed": 300 + i, "ok": ok, "injected": True, "why": "[x]"} for i, ok in enumerate([0, 0, 1, 0])
    ]
    agent = [
        {"round": i, "seed": 300 + i, "ok": ok, "injected": inj, "why": "", "agent_calls": 5}
        for i, (ok, inj) in enumerate([(1, True), (1, True), (1, True), (0, False)])
    ]
    traces = {
        0: [{"event": "injection"}, {"primitive": "release", "outcome": {"ok": True}}],
        1: [{"event": "injection"}],
    }
    pairs, summary = paired_summary(plan, agent, traces)
    assert summary["pairs"] == 4 and summary["pairs_both_injected"] == 3
    assert summary["plan_ok"] == 1 and summary["agent_ok"] == 3
    assert summary["agent_only_ok"] == 2 and summary["plan_only_ok"] == 0
    assert summary["sign_test_p"] == pytest.approx(0.5)
    assert summary["recovery_classes"]["release"] == {"n": 1, "ok": 1, "segments": ["release"]}
    assert summary["agent_regrasped"] == 0 and pairs[0]["regrasped"] is False
    assert regrasped(["approach", "grasp_single", "approach!"]) and not regrasped(None)
    text = render_report(pairs, summary, plan_run="p", agent_run="a", injection={"kind": "move_target"})
    assert "| 300 | PA | FAIL | OK |" in text and "agent-only OK 2" in text
