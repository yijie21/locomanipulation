"""TODO 41.2: the check before a jaw closes -- when it retries, when it leaves the round alone, and that it is off
unless asked for."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from locomani.tasks.scene import grip_guard as G  # noqa: E402

ROT = np.eye(3).tolist()


def reach(side, pos, **kw):
    return dict(name="reach", args=[side, list(pos), ROT], kwargs=kw)


ENTRY = [reach("right", (0.40, 0.0, 0.9), via=[[0.3, 0.0, 1.0]]), reach("right", (0.52, 0.0, 0.9), speed=0.08)]
FREE = lambda xy: True  # noqa: E731


def test_on_target_closes():
    d = G.decide(ENTRY, "right", (0.50, 0.0, 0.9), (0.0, 0.0), FREE)
    assert d["act"] == "close" and d["why"] == "on target" and d["err_cm"] == 2.0


def test_a_short_entry_steps_by_the_miss():
    d = G.decide(ENTRY, "right", (0.42, 0.02, 0.9), (0.0, 0.0), FREE)
    assert d["act"] == "retry" and not d["descent"] and d["tol_cm"] == 3.0
    assert np.allclose(d["step"], [0.10, -0.02])


def test_the_step_is_capped_and_shortened_onto_free_floor():
    far = G.decide(ENTRY, "right", (0.0, 0.0, 0.9), (0.0, 0.0), FREE)
    assert np.isclose(np.linalg.norm(far["step"]), G.MAX_STEP_M)
    half = G.decide(ENTRY, "right", (0.42, 0.0, 0.9), (0.0, 0.0), lambda xy: xy[0] < 0.06)
    assert half["act"] == "retry" and np.allclose(half["step"], [0.05, 0.0])
    none = G.decide(ENTRY, "right", (0.42, 0.0, 0.9), (0.0, 0.0), lambda xy: False)
    assert none["act"] == "close" and none["why"] == "no free step toward the target"


def test_a_descent_has_its_own_tolerance_and_a_hand_held_up_is_left_alone():
    calls = [reach("right", (0.5, 0.0, 1.0)), reach("right", (0.5, 0.0, 0.9), speed=0.05)]
    assert G.decide(calls, "right", (0.5, 0.04, 0.9), (0, 0), FREE)["act"] == "close"  # 4 cm: inside 5
    assert G.decide(calls, "right", (0.5, 0.07, 0.9), (0, 0), FREE)["act"] == "retry"
    up = G.decide(calls, "right", (0.5, 0.01, 0.96), (0, 0), FREE)
    assert up["act"] == "close" and up["why"] == "held up, not short"


def test_it_leaves_alone_what_it_cannot_judge():
    walk = dict(name="walk_to", args=[[0, 0], 0.0], kwargs={})
    assert (
        G.decide([*ENTRY, walk], "right", (0, 0, 0), (0, 0), FREE)["why"] == "last command was not a reach of this hand"
    )
    assert G.decide(ENTRY, "left", (0, 0, 0), (0, 0), FREE)["act"] == "close"
    assert G.decide([walk, ENTRY[1]], "right", (0, 0, 0.9), (0, 0), FREE)["why"] == "no pre-grasp point to pull back to"


class FakeMotor:
    """Reaches land `short` before their target along x until the pelvis has stepped; then on target."""

    def __init__(self):
        self.calls, self.fell_at, self._call_depth, self.t = [], None, 0, 0.0
        self.jaw, self.pel, self.grips = np.zeros(3), np.zeros(3), 0

    def jaw_centre(self, side):
        return self.jaw.copy(), np.eye(3)

    def pelvis(self):
        return self.pel.copy(), np.array([1.0, 0, 0, 0])

    def yaw(self):
        return 0.0

    def reach(self, side, pos, rot=None, **kw):
        self.calls.append(
            reach(side, np.asarray(pos, float).tolist(), **{k: np.asarray(v).tolist() for k, v in kw.items()})
        )
        self.jaw = np.asarray(pos, float) - [max(0.0, 0.10 - self.pel[0]), 0, 0]
        return SimpleNamespace(ok=True, info=dict(jaw_err_cm=round(100 * max(0.0, 0.10 - self.pel[0]), 2)))

    def walk_to(self, xy, yaw=None, **kw):
        self.calls.append(dict(name="walk_to", args=[list(map(float, xy)), yaw], kwargs=kw))
        self.pel[:2] = xy

    def grip(self, side, **kw):
        self.grips += 1
        self.calls.append(dict(name="grip", args=[side], kwargs=kw))
        return SimpleNamespace(ok=True, info={})


def test_installed_it_retries_once_then_closes_and_passes_grip_arguments_through():
    m = FakeMotor()
    w = SimpleNamespace(motor=m, occ=SimpleNamespace(is_free=FREE))
    G.install(w)
    m.reach("right", (0.40, 0.0, 0.9), via=[[0.3, 0.0, 1.0]])
    m.reach("right", (0.52, 0.0, 0.9), speed=0.08)
    m.grip("right", seconds=0.5)
    names = [c["name"] for c in m.calls]
    assert names == ["reach", "reach", "reach", "walk_to", "reach", "reach", "grip"]
    assert np.allclose(m.calls[2]["args"][1], [0.30, 0.0, 0.9])  # pulled back by the step before walking
    assert m.calls[3]["kwargs"] == dict(face="fixed") and np.allclose(m.calls[3]["args"][0], [0.10, 0.0])
    assert m.calls[4]["kwargs"]["via"] == [[0.3, 0.0, 1.0]] and m.calls[5]["kwargs"] == dict(speed=0.08)
    assert m.calls[6]["kwargs"] == dict(seconds=0.5)
    row = w.grip_guard_log[0]
    assert row["act"] == "retry" and row["after_cm"] == 0.0
    m.reach("right", (0.9, 0.0, 0.9))
    m.reach("right", (1.2, 0.0, 0.9))
    m.jaw = np.array([0.9, 0.0, 0.9])
    m.grip("right")  # the round's one retry is spent
    assert [c["name"] for c in m.calls[-3:]] == ["reach", "reach", "grip"] and m.grips == 2
    m.calls = []  # the motor's reset starts a new list: a new round, a new retry and a new log
    m.pel[:] = 0
    m.reach("right", (0.40, 0.0, 0.9))
    m.reach("right", (0.52, 0.0, 0.9))
    m.grip("right")
    assert "walk_to" in [c["name"] for c in m.calls] and len(w.grip_guard_log) == 1


def test_off_unless_asked_for_and_in_the_signature_only_while_on():
    import importlib.util

    saved = sys.modules.pop("batch", None)
    sys.path.insert(0, str(ROOT / "scripts" / "rsi"))
    try:
        spec = importlib.util.spec_from_file_location("grip_guard_batch", ROOT / "scripts" / "rsi" / "batch.py")
        B = importlib.util.module_from_spec(spec)
        sys.modules["grip_guard_batch"] = B  # dataclasses look the module up by name
        spec.loader.exec_module(B)
    finally:
        sys.path.pop(0)
        if saved is not None:
            sys.modules["batch"] = saved
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(G.ENV, None)
        assert not G.enabled()
        base = B.motor_signature()
        for v in ("", "0"):
            os.environ[G.ENV] = v
            assert not G.enabled() and B.motor_signature() == base
        os.environ[G.ENV] = "1"
        assert G.enabled() and B.motor_signature() != base
