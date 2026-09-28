"""Executor gate, owner bookkeeping, trace file and plan loop over stub primitives (no stack, no scene)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from locomani.skills.contract import Outcome, Violation
from locomani.skills.executor import Executor, _jsonable
from locomani.skills.session import ArmOwner, HeldObject, Session, Stance

ALL = frozenset(ArmOwner)


class Stub:
    def __init__(self, name, requires=ALL, yields=None, ok=True, reason="", violations=(), mutate=None):
        self.name, self.requires, self.yields = name, requires, yields
        self.ok, self.reason, self.violations, self.mutate = ok, reason, list(violations), mutate
        self.calls = []

    def preconditions(self, session, *, gain: float = 1.0):
        return list(self.violations)

    def run(self, session, *, gain: float = 1.0):
        self.calls.append(gain)
        if self.mutate is not None:
            self.mutate(session)
        return Outcome(self.ok, self.reason, metrics={"gain": gain})


def session(**kw):
    return Session(stack=object(), scene=None, **kw)


def test_unknown_primitive_and_unknown_arg_are_rejected_and_traced():
    s = session()
    ex = Executor(s, {"noop": Stub("noop")})
    assert ex.step("fly", {})["rejected"][0]["check"] == "unknown_primitive"
    entry = ex.step("noop", {"speed": 2})
    assert entry["rejected"][0]["check"] == "unknown_arg" and "gain" in entry["rejected"][0]["note"]
    assert len(s.trace) == 2


def test_wrong_owner_and_precondition_violations_reject_before_run():
    s = session(arm_owner=ArmOwner.PLANNER_SWING)
    only_stream = Stub("grasp", requires=frozenset({ArmOwner.PLANNER_FROZEN}))
    entry = Executor(s, {"grasp": only_stream}).step("grasp")
    assert entry["rejected"][0]["check"] == "arm_owner" and only_stream.calls == []
    picky = Stub("turn", violations=[Violation("heading", None, note="anchor first")])
    entry = Executor(s, {"turn": picky}).step("turn")
    assert entry["rejected"] == [{"check": "heading", "measured": None, "bound": [None, None], "note": "anchor first"}]
    assert picky.calls == []


def test_ok_step_applies_declared_yields_and_writes_one_json_line(tmp_path):
    s = session(arm_owner=ArmOwner.PLANNER_FROZEN)

    def take(sess):
        sess.arm_owner = ArmOwner.POSE_STREAM
        sess.arm = type("StreamedArm", (), {"_stream_released": False})()

    prim = Stub("take_over", requires=frozenset({ArmOwner.PLANNER_FROZEN}), yields=ArmOwner.POSE_STREAM, mutate=take)
    ex = Executor(s, {"take_over": prim}, trace_path=tmp_path / "t" / "agent_trace.jsonl")
    entry = ex.step("take_over", {"gain": 2.0})
    assert entry["outcome"]["ok"] and entry["outcome"]["metrics"] == {"gain": 2.0}
    assert entry["session"]["arm_owner"] == "pose_stream" and prim.calls == [2.0]
    lines = (tmp_path / "t" / "agent_trace.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["primitive"] == "take_over"


def test_yields_none_means_unchanged_and_a_silent_owner_change_raises():
    s = session()
    ok = Executor(s, {"turn": Stub("turn")}).step("turn")
    assert ok["outcome"]["ok"] and s.arm_owner is ArmOwner.PLANNER_SWING

    def sneak(sess):
        sess.arm_owner = ArmOwner.PLANNER_FROZEN

    with pytest.raises(RuntimeError, match="declared planner_swing"):
        Executor(session(), {"turn": Stub("turn", mutate=sneak)}).step("turn")


def test_ok_step_that_breaks_an_invariant_raises():
    def hold_without_closing(sess):
        sess.held = HeldObject(key="can", sides=("right",), grip_in_palm={})

    with pytest.raises(RuntimeError, match="invariants"):
        Executor(session(), {"grasp": Stub("grasp", mutate=hold_without_closing)}).step("grasp")


def test_failed_outcome_is_traced_not_raised():
    s = session()
    entry = Executor(s, {"walk": Stub("walk", ok=False, reason="[walk] stalled")}).step("walk")
    assert entry["outcome"] == {**entry["outcome"], "ok": False, "tag": "walk"} and s.trace[-1] is entry


def test_run_plan_stops_at_the_first_rejected_or_failed_step():
    cat = {"a": Stub("a"), "b": Stub("b", ok=False, reason="[b] no"), "c": Stub("c")}
    entries = Executor(session(), cat).run_plan([{"primitive": "a"}, {"primitive": "b"}, {"primitive": "c"}])
    assert [e["primitive"] for e in entries] == ["a", "b"] and cat["c"].calls == []
    entries = Executor(session(), cat).run_plan([{"primitive": "zzz"}, {"primitive": "a"}])
    assert len(entries) == 1 and "rejected" in entries[0]


def test_abort_is_traced_with_a_snapshot():
    s = session()
    entry = Executor(s, {}).abort("[agent] gave up")
    assert entry["primitive"] == "abort" and entry["session"]["arm_owner"] == "planner_swing" and s.trace == [entry]


def test_jsonable_handles_numpy_enums_and_dataclasses():
    stance = Stance(target="x", profile="p", depth_m=0.3, lateral_m=0.0, yaw_err_rad=0.1, quiet=True)
    out = _jsonable({"a": np.arange(2), "b": np.float32(1.5), "o": ArmOwner.POSE_STREAM, "s": stance})
    assert json.dumps(out) and out["a"] == [0, 1] and out["o"] == "pose_stream" and out["s"]["target"] == "x"
