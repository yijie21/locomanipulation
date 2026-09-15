from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from locomani.backends.sonic.primitives import LocomotionMode
from locomani.tasks.bottle import round_phases_approach as approach
from locomani.tasks.bottle.whole_body_policy import WholeBodyGraspPolicy


def _state(seq, depth, *, x=0.0, t=None, fallen=False, speed=0.0):
    return dict(
        seq=seq,
        t=float(seq if t is None else t),
        depth=depth,
        fallen=fallen,
        pelvis=dict(pos=[x, 0.0, 0.8], vel=[speed, 0.0, 0.0]),
        objects={"bottle": {"cam": [0.0, 0.0]}},
    )


class _Coordinated:
    def __init__(self, events):
        self.events = events
        self.error = None
        self.walking = SimpleNamespace(
            nominal_stop_depth=0.31,
            nominal_stop_edge=None,
            chosen_stop_depth=None,
            chosen_stop_edge=None,
            stop_lead=0.0,
            stop_speed=0.0,
        )
        self.settlement_observer = None

    def walk_tick(self, state, *, edge_distance=None):
        self.events.append(("plan", state["seq"], edge_distance))
        return True

    def set_approach_settlement_observer(self, observer):
        self.settlement_observer = observer


def _context(states, events):
    current = iter(states)
    last = states[-1]

    def observe():
        nonlocal last
        try:
            last = next(current)
        except StopIteration:
            pass
        return last

    cmd = SimpleNamespace(frame=Mock(side_effect=lambda **kw: events.append(("frame", kw["mode"]))))
    robot = SimpleNamespace(
        obs=Mock(side_effect=observe),
        cmd=cmd,
        pelvis_xy=lambda state: np.asarray(state["pelvis"]["pos"][:2]),
        log=Mock(),
    )
    coordinated = _Coordinated(events)
    return SimpleNamespace(
        p=robot,
        rm=SimpleNamespace(margin=Mock(return_value=True), graspable=Mock(return_value=True)),
        reach_config=SimpleNamespace(
            image_servo=False,
            visible_expert=True,
            visible_brake_depth=0.31,
            visible_brake_edge=0.0,
            visible_decel_band=0.24,
            arrival_advance=0.0,
        ),
        coordinated=coordinated,
        spawn_xy=np.array([0.0, 0.0]),
        fv=(1.0, 0.0, 0.0),
        raise_lead=0.3,
        arm_ramp=Mock(),
        q_free=np.zeros(17),
        mid_band=0.3,
        plan_r=0.25,
        edge_dist=lambda xy: 0.5 - float(xy[0]),
        edge_min=0.06,
        one_shot=True,
        execution=SimpleNamespace(metrics={}),
        obj_key="bottle",
        view_gate_x=0.8,
        view_gate_y=0.9,
    )


@pytest.fixture(autouse=True)
def _simple_approach_geometry(monkeypatch):
    monkeypatch.setattr(approach, "_rel_now", lambda _ctx, state: (state["depth"], -0.1, 0.0))
    monkeypatch.setattr(approach, "_rel_dbg", lambda *_args: None)
    monkeypatch.setattr(approach.stream_reach, "toppled", lambda *_args: False)


def test_approach_stops_on_first_fresh_observation_before_planning():
    events = []
    state = _state(17, 0.30, x=0.1, t=2.4)
    ctx = _context([state], events)

    assert approach._phase_approach(ctx) is None

    assert events == [("frame", LocomotionMode.IDLE)]
    request = ctx.execution.metrics["approach_stop_requests"][0]["request"]
    assert request["observation_sequence"] == 17
    assert request["observation_physical_time_s"] == 2.4
    assert request["pelvis_position_m"] == [0.1, 0.0, 0.8]
    assert request["object_depth_m"] == 0.30
    assert request["table_edge_distance_m"] == 0.4
    assert request["reason"] == "visible brake depth"
    assert request["forward_direction_xy"] == [1.0, 0.0]
    assert request["observation_received_wall_time_s"] <= request["idle_call_wall_before_s"]
    assert request["idle_call_wall_before_s"] <= request["idle_call_wall_after_s"]
    assert request["idle_call_monotonic_before_s"] <= request["idle_call_monotonic_after_s"]
    assert request["nominal_stop_depth_m"] == 0.31
    assert request["chosen_stop_depth_m"] == 0.31
    assert request["nominal_stop_edge_m"] is None
    assert request["chosen_stop_edge_m"] is None
    assert request["stop_lead_m"] == 0.0
    assert request["stop_speed_m_s"] == 0.0
    assert request["last_slow_walk_command_speed"] is None


def test_approach_rechecks_a_new_observation_after_planning_before_walking():
    events = []
    before_plan = _state(20, 0.40, x=0.0, t=3.0)
    after_plan = _state(21, 0.30, x=0.1, t=3.1)
    ctx = _context([before_plan, after_plan], events)

    assert approach._phase_approach(ctx) is None

    assert events == [
        ("plan", 20, 0.5),
        ("frame", LocomotionMode.IDLE),
    ]
    request = ctx.execution.metrics["approach_stop_requests"][0]["request"]
    assert request["observation_sequence"] == 21
    assert all(event != ("frame", LocomotionMode.SLOW_WALK) for event in events)

    settled = _state(22, 0.18, x=0.22, t=3.4)
    ctx.coordinated.settlement_observer(settled, complete=True)
    settling = ctx.execution.metrics["approach_stop_requests"][0]["settling"]
    assert settling == dict(
        maximum_forward_displacement_m=pytest.approx(0.12),
        minimum_table_edge_distance_m=pytest.approx(0.28),
        final_observation_sequence=22,
        final_physical_time_s=3.4,
        final_forward_displacement_m=pytest.approx(0.12),
        final_object_depth_m=0.18,
        final_table_edge_distance_m=pytest.approx(0.28),
        complete=True,
        settled=False,
        completion_reason=None,
    )


def test_approach_emits_idle_before_returning_from_a_fall():
    events = []
    fallen = _state(25, 0.40, t=4.0, fallen=True)
    ctx = _context([fallen], events)

    result = approach._phase_approach(ctx)

    assert result[0] is False
    assert "fell during" in result[1]
    assert events == [("frame", LocomotionMode.IDLE)]
    trace = ctx.execution.metrics["approach_stop_requests"][0]
    assert trace["request"]["reason"] == "fell"
    assert trace["settling"]["complete"] is True
    assert trace["settling"]["settled"] is False
    assert trace["settling"]["completion_reason"] == "fell"


def test_whole_body_settles_before_stand_admission_uses_the_final_state():
    events = []
    moving = _state(27, 0.30, x=0.1, t=4.2)
    settled = _state(28, 0.20, x=0.2, t=4.6)
    ctx = _context([moving], events)

    def settle_for_admission():
        events.append(("settle", 27))
        ctx.p.obs.side_effect = lambda: settled
        return True

    ctx.coordinated.settle_for_admission = settle_for_admission
    ctx.rm.graspable.side_effect = lambda *_args: events.append(("admit", 28)) or True

    assert approach._phase_approach(ctx) is None

    assert events.index(("settle", 27)) < events.index(("admit", 28))
    assert ctx.execution.metrics["first_stop"] == pytest.approx(np.hypot(0.20, 0.10), abs=0.001)


def test_whole_body_brake_holds_idle_before_each_planner_tick_and_finalizes_trace():
    events = []
    first = _state(30, 0.3, x=0.1, t=5.0)
    quiet_start = _state(31, 0.25, x=0.15, t=5.1)
    quiet = _state(32, 0.2, x=0.2, t=5.4)
    policy = WholeBodyGraspPolicy.__new__(WholeBodyGraspPolicy)
    policy.p = SimpleNamespace(obs=Mock(side_effect=[first, quiet_start, quiet]))
    policy._hold_planner = Mock(side_effect=lambda state: events.append(("hold", state["seq"])))
    policy.walk_tick = Mock(side_effect=lambda state: events.append(("plan", state["seq"])) or True)
    observed = []
    policy._approach_settlement_observer = lambda state, complete=False, settled=False, completion_reason=None: (
        observed.append((state["seq"], complete, settled, completion_reason))
    )

    assert policy._brake()

    assert events == [("hold", 30), ("plan", 30), ("hold", 31), ("plan", 31)]
    assert observed[-1] == (32, True, True, "brake quiet")


def test_settle_for_admission_is_reused_by_execute_without_a_second_brake():
    policy = WholeBodyGraspPolicy.__new__(WholeBodyGraspPolicy)
    policy._base_settled = False
    policy.execution = Mock()
    policy._brake = Mock(return_value=True)

    assert policy.settle_for_admission()
    assert policy.settle_for_admission()

    policy._brake.assert_called_once_with()
    policy.execution.enter.assert_called_once_with("whole_body.brake")
