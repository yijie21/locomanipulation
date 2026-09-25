from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from locomani.backends.sonic.primitives import LocomotionMode
from locomani.tasks.bottle import round_phases_approach as approach
from locomani.tasks.bottle.whole_body_policy import WholeBodyGraspPolicy


def _state(seq, depth, *, x=0.0, yaw=0.0, t=None, fallen=False, speed=0.0):
    return dict(
        seq=seq,
        t=float(seq if t is None else t),
        depth=depth,
        fallen=fallen,
        pelvis=dict(
            pos=[x, 0.0, 0.8],
            quat=[np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)],
            vel=[speed, 0.0, 0.0],
        ),
        objects={"bottle": {"cam": [0.0, 0.0], "pos": [x + depth, 0.0, 0.7]}},
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

    def walk_tick(
        self,
        state,
        *,
        edge_distance=None,
        navigation_heading=None,
        navigation_target=None,
    ):
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
        object_pos=lambda key, state: np.asarray(state["objects"][key]["pos"]),
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


def test_coordinated_visible_stop_and_speed_ignore_pelvis_yaw_at_fixed_world_pose():
    first = _state(1, 0.40, yaw=0.0)
    turned_yaw = 0.70
    turned = _state(2, 0.40, yaw=turned_yaw)
    ctx = _context([first], [])
    cfg = ctx.reach_config
    turned_rel = (0.40 * np.cos(turned_yaw), 0.40 * np.sin(turned_yaw), 0.0)

    first_depth, first_frame = approach._visible_depth_measurement(ctx, first, (0.40, -0.1, 0.0))
    turned_depth, turned_frame = approach._visible_depth_measurement(ctx, turned, turned_rel)

    assert first_depth == pytest.approx(0.40)
    assert turned_depth == pytest.approx(first_depth)
    assert first_frame == turned_frame == "fixed_navigation_heading"
    assert approach.visible_stop(first_depth, 0.5, cfg) == approach.visible_stop(turned_depth, 0.5, cfg)
    assert approach.visible_speed_scale(first_depth, 0.5, cfg) == pytest.approx(
        approach.visible_speed_scale(turned_depth, 0.5, cfg)
    )
    speed_args = dict(
        plan_depth=0.25,
        mid_band=0.10,
        coarse_speed=0.40,
        mid_speed=0.25,
        shuffle_speed=0.20,
    )
    first_speed = approach._visible_command_speed(first_depth, 0.5, cfg, **speed_args)
    turned_speed = approach._visible_command_speed(turned_depth, 0.5, cfg, **speed_args)
    assert turned_speed == pytest.approx(first_speed)

    ctx.coordinated = None
    legacy_depth, legacy_frame = approach._visible_depth_measurement(ctx, turned, turned_rel)
    assert legacy_depth == pytest.approx(turned_rel[0])
    assert legacy_frame == "current_pelvis_yaw"
    assert approach.visible_stop(legacy_depth, 0.5, cfg)
    assert approach.visible_speed_scale(legacy_depth, 0.5, cfg) == pytest.approx(0.0)
    assert approach._visible_command_speed(legacy_depth, 0.5, cfg, **speed_args) == pytest.approx(0.20)


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
    assert request["object_depth_m"] == pytest.approx(0.30)
    assert request["object_depth_frame"] == "fixed_navigation_heading"
    assert request["ego_object_depth_m"] == 0.30
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
        final_object_depth_m=pytest.approx(0.18),
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
    policy._preparation_hold_tick = Mock(
        side_effect=lambda state: events.append(("prepare_hold", state["seq"])) or True
    )
    observed = []
    policy._approach_settlement_observer = lambda state, complete=False, settled=False, completion_reason=None: (
        observed.append((state["seq"], complete, settled, completion_reason))
    )

    assert policy._brake()

    assert events == [
        ("hold", 30),
        ("prepare_hold", 30),
        ("hold", 31),
        ("prepare_hold", 31),
    ]
    assert observed[-1] == (32, True, True, "brake quiet")


def test_whole_body_brake_samples_preparation_every_tick_and_restarts_support_dwell():
    samples = [
        _state(40, 0.2, t=8.00, speed=0.1),
        _state(41, 0.2, t=8.05),
        _state(42, 0.2, t=8.10),
        _state(43, 0.2, t=8.15, speed=0.1),
        _state(44, 0.2, t=8.20),
        _state(45, 0.2, t=8.25),
        _state(46, 0.2, t=8.30),
        _state(47, 0.2, t=8.35),
        _state(48, 0.2, t=8.40),
        _state(49, 0.2, t=8.45),
    ]
    policy = WholeBodyGraspPolicy.__new__(WholeBodyGraspPolicy)
    policy.p = SimpleNamespace(obs=Mock(side_effect=samples))
    policy._hold_planner = Mock()
    policy._preparation_hold_tick = Mock(return_value=True)
    policy._approach_settlement_observer = lambda *_args, **_kwargs: None
    policy._settled_request = None
    calls = []

    def settled_tick(state, *, support_quiet, allow_submission):
        calls.append((state["t"], support_quiet, allow_submission))
        return {"status": "legacy" if allow_submission else "preparing"}

    policy._settled_stance_tick = settled_tick

    assert policy._brake()
    assert [stamp for stamp, _, _ in calls] == pytest.approx([sample["t"] for sample in samples[1:]])
    assert calls[2][1:] == (False, False)
    assert not any(admitted for _, _, admitted in calls[:-1])
    assert calls[-1][2]


def test_settle_for_admission_is_reused_by_execute_without_a_second_brake():
    policy = WholeBodyGraspPolicy.__new__(WholeBodyGraspPolicy)
    policy._base_settled = False
    policy.execution = Mock()
    policy._brake = Mock(return_value=True)

    assert policy.settle_for_admission()
    assert policy.settle_for_admission()

    policy._brake.assert_called_once_with()
    policy.execution.enter.assert_called_once_with("whole_body.brake")


def test_visible_shuffle_speed_overrides_the_checkpoint_floor():
    assert approach.approach_shuffle_speed(SimpleNamespace(visible_shuffle_speed=0.35), 0.20) == 0.35
    assert approach.approach_shuffle_speed(SimpleNamespace(visible_shuffle_speed=0.0), 0.20) == 0.20
    assert approach.approach_shuffle_speed(SimpleNamespace(), 0.20) == 0.20
    cfg = SimpleNamespace(visible_brake_depth=0.31, visible_decel_band=0.50, visible_brake_edge=0.27)
    common = dict(plan_depth=0.3, mid_band=0.0, coarse_speed=0.40, mid_speed=0.215, shuffle_speed=0.35)
    # At the brake edge the command is the crawl floor, half a band out it is halfway to coarse.
    assert approach._visible_command_speed(0.6, 0.27, cfg, **common) == pytest.approx(0.35)
    assert approach._visible_command_speed(0.6, 0.52, cfg, **common) == pytest.approx(0.375)
    assert approach._visible_command_speed(0.9, 0.9, cfg, **common) == pytest.approx(0.40)


def test_simple_task_accepts_the_crawl_band_and_rejects_a_floor_above_bounds():
    from locomani import profiles

    task = profiles.load("task", "bottle_simple")
    # A crawl floor above v1.1's 0.28-0.30 speed step and a band the gait can respond within.
    assert 0.30 <= task.reach.visible_shuffle_speed <= 0.35 and task.reach.visible_decel_band >= 0.50
    with pytest.raises(ValueError):
        task.with_overrides(["reach.visible_shuffle_speed=0.7"])
    with pytest.raises(ValueError):
        task.with_overrides(["reach.visible_decel_band=0.7"])


class _EdgeGuard:
    """The guard walk's interface: scripted stop reasons, a fixed steering direction and speed."""

    walk_speed = 0.25

    def __init__(self, reasons):
        self.reasons = iter(reasons)
        self.seen = []

    def step(self, state):
        self.seen.append(state["seq"])
        return next(self.reasons)

    def direction(self, pelvis_xy, aim_xy, heading):
        return np.array([0.6, 0.8])


def test_guard_walk_holds_one_speed_steers_by_the_guard_and_stops_only_on_its_brake():
    events, frames = [], []
    # Depth 0.30 is already past the visible brake depth (0.31) and the edge floor: only the guard may stop.
    states = [_state(30, 0.30, t=1.0), _state(31, 0.30, t=1.1), _state(32, 0.30, t=1.2)]
    ctx = _context(states, events)
    ctx.coordinated.edge_guard = _EdgeGuard([None, None, "guard brake"])
    ctx.p.cmd.frame = Mock(side_effect=lambda **kw: frames.append(kw))

    assert approach._phase_approach(ctx) is None

    assert ctx.coordinated.edge_guard.seen == [30, 31, 32]
    walk, idle = frames
    assert walk["mode"] == LocomotionMode.SLOW_WALK and walk["speed"] == 0.25
    np.testing.assert_allclose(walk["movement"], [0.6, 0.8, 0.0])
    assert idle["mode"] == LocomotionMode.IDLE
    request = ctx.execution.metrics["approach_stop_requests"][0]["request"]
    assert request["reason"] == "guard brake" and request["observation_sequence"] == 32
    assert request["last_slow_walk_command_speed"] == 0.25
