"""TODO 38.19 (user 2026-10-24): the strict box goal -- `lifted(X, clearance, tilt)` and a judging window over the
round's last seconds (episode.judge_window) -- and the failure classes that read them."""

from __future__ import annotations

import math

import pytest

from locomani.scene_discovery.predicates import compile_goal, parse
from locomani.tasks.scene.episode import JUDGE_DT, judge_window

TOP = 0.371
META = {
    "surfaces": {"A": {"xy": [0.0, 0.0], "top": TOP, "half_extent": [0.6, 0.45], "yaw": 0.0}},
    "objects": {
        "box": {"surface": "A", "free": True, "box": {"half_extent": [0.225, 0.16, 0.165], "centre": [0, 0, 0.165]}}
    },
}


def tilt_x(deg: float) -> list[float]:
    a = math.radians(deg) / 2
    return [math.cos(a), math.sin(a), 0.0, 0.0]


def state(z: float, quat=(1.0, 0.0, 0.0, 0.0), touching=("left", "right")) -> dict:
    return {"objects": {"box": {"pos": [0.0, 0.0, z], "quat": list(quat), "touching": list(touching), "gripped": []}}}


def test_lifted_parses_its_numbers_and_needs_an_object():
    assert parse("lifted(box, 0.20, 30)").args == ("box", 0.2, 30.0)
    with pytest.raises(ValueError):
        compile_goal(["lifted(A, 0.2, 30)"], META)
    with pytest.raises(ValueError):
        parse("lifted(box, 0.2)")


def test_lifted_measures_the_lowest_corner_above_the_platform_top():
    check = compile_goal(["held(box)", "lifted(box, 0.20, 30)"], META)
    assert check(state(TOP + 0.21))[0]
    ok, why = check(state(TOP + 0.19))
    assert not ok and "lowest corner 19.0 cm" in why
    # tilted 20 deg about x: the low edge drops by 0.16 sin 20 = 5.5 cm, so the origin needs about 25.5 cm
    assert not check(state(TOP + 0.24, tilt_x(20)))[0]
    assert check(state(TOP + 0.27, tilt_x(20)))[0]


def test_lifted_refuses_a_tilt_past_the_limit_and_held_still_needs_both_hands():
    check = compile_goal(["held(box)", "lifted(box, 0.20, 30)"], META)
    ok, why = check(state(TOP + 0.40, tilt_x(31)))
    assert not ok and "tilted 31 deg" in why
    ok, why = check(state(TOP + 0.30, touching=("left",)))
    assert not ok and why.startswith("[goal] held(box)")


def test_the_window_needs_every_sample_of_the_last_seconds():
    good = [(1.0 + k * JUDGE_DT, True, "") for k in range(100)]  # 1.0 .. 10.9 s, all hold
    v = judge_window(good, 10.9, 5.0)
    assert v["ok"] and v["n"] == v["n_ok"] == 51 and v["ok_since"] == 1.0
    bad = [(t, not (7.0 <= t < 7.25), "" if not (7.0 <= t < 7.25) else "lowest corner 15 cm") for t, _, _ in good]
    v = judge_window(bad, 10.9, 5.0)
    assert not v["ok"] and v["first_fail_t"] == 7.0 and v["ever_ok"] and v["ok_since"] == pytest.approx(7.3)
    # a failure before the window does not count
    early = [(t, t >= 5.0, "") for t, _, _ in good]
    assert judge_window(early, 10.9, 5.0)["ok"]
    assert not judge_window([], 3.0, 5.0)["ok"]
