"""The SONIC expert's CuRobo route (TODO 26.5.6): the plan runs off the control thread while the held reference keeps
streaming, a planner error reaches the caller, and a planned reach without a fallback setting is refused."""

import time
from types import SimpleNamespace

import pytest

from locomani.tasks.crate.expert import StreamedRoute, route_planner_for


class SlowPlanner:
    def __init__(self, seconds, error=None):
        self.seconds, self.error, self.last = seconds, error, {"ok": True}

    def __call__(self, start, goal, state):
        time.sleep(self.seconds)
        if self.error:
            raise self.error
        return [start, goal]


def test_the_held_reference_keeps_streaming_while_the_route_is_planned():
    sent = []
    route = StreamedRoute(SlowPlanner(0.3), lambda state: sent.append((time.time(), state["t"])))
    assert route([0.0], [1.0], {"t": 12.34}) == [[0.0], [1.0]]
    assert len(sent) >= 10  # 50 Hz over 0.3 s
    assert max(b[0] - a[0] for a, b in zip(sent, sent[1:])) < 0.1
    assert {t for _, t in sent} == {12.34}  # the planning tick's reference time, never a later one
    assert route.last == {"ok": True}


def test_a_planner_error_is_raised_on_the_control_thread():
    with pytest.raises(RuntimeError, match="no service"):
        StreamedRoute(SlowPlanner(0.05, RuntimeError("no service")), lambda state: None)([0.0], [1.0], {"t": 0.0})


def test_sonic_refuses_a_planned_reach_with_a_straight_fallback():
    visual = SimpleNamespace(reach_route="planned", reach_route_fallback="straight", pose_source="assisted")
    config = SimpleNamespace(carry=SimpleNamespace(reach_control="whole_body"))
    with pytest.raises(ValueError, match="reach_route_fallback none"):
        route_planner_for(visual, config, "scene.xml", None, {"body": "box"}, "box")
    assert route_planner_for(SimpleNamespace(reach_route="straight"), config, "scene.xml", None, {}, "box") is None
