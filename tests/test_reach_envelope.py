"""Boundary-search planning logic and the floor-scene strip, without a simulator."""

import json
import xml.etree.ElementTree as ET

import pytest

from locomani.tasks.crate.reach_envelope import boundary_search as B
from locomani.tasks.crate.reach_envelope.floor_scene import strip_to_floor


def rec(x, pos, reason="target_timeout"):
    return {"x": x, "best_pos": pos, "reason": reason, "status": "failed"}


def test_next_x_walks_in_from_the_ik_bound_then_brackets():
    k = ("k",)
    assert B.next_x(k, 0.65, {}) == pytest.approx(0.55)  # 10 cm inside the optimistic bound
    assert B.next_x(k, None, {}) is None  # never admitted offline: nothing to try
    obs = {k: {0.55: rec(0.55, 0.045)}}
    assert B.next_x(k, 0.65, obs) == pytest.approx(0.50)  # fail -> step in
    obs[k][0.50] = rec(0.50, 0.012)
    assert B.next_x(k, 0.65, obs) is None  # pass below fail: bracketed
    obs = {k: {0.55: rec(0.55, 0.012)}}
    assert B.next_x(k, 0.65, obs) == pytest.approx(0.60)  # pass -> step out
    obs[k][0.60] = rec(0.60, 0.015)
    assert B.next_x(k, 0.65, obs) == pytest.approx(0.65)  # still inside the IK bound
    obs[k][0.65] = rec(0.65, 0.010)
    assert B.next_x(k, 0.65, obs) is None  # would exceed the bound: stop


def test_next_x_retries_infrastructure_failures_and_floors_at_x_min():
    k = ("k",)
    obs = {k: {0.55: rec(0.55, None, "quiet_ready_state_lost_before_task_recording")}}
    assert B.next_x(k, 0.65, obs) == pytest.approx(0.55)
    obs = {k: {x: rec(x, 0.1) for x in (0.25, 0.20)}}
    assert B.next_x(k, 0.65, obs) is None  # nothing passes down to X_MIN: reported, not retried


def test_verdict_uses_position_only():
    assert B.verdict(rec(0.5, 0.019)) == "pass"
    assert B.verdict(rec(0.5, 0.021)) == "fail"
    assert B.verdict(rec(0.5, 0.3, "controller_unsupported_assisted_reach")) == "reject"


def test_strip_to_floor_keeps_robot_and_floor_only(tmp_path):
    xml = """<mujoco><compiler meshdir="/nowhere"/><worldbody>
      <geom name="floor" type="plane" size="0 0 0.05"/>
      <light pos="0 0 3"/>
      <body name="pelvis" pos="0 0 0.8"><freejoint/><geom type="sphere" size="0.1"/></body>
      <body name="table"><geom type="box" size="0.5 0.5 0.02"/></body>
      <body name="crate"><freejoint/><geom type="box" size="0.2 0.2 0.1"/></body>
    </worldbody><contact><exclude body1="table" body2="crate"/></contact>
    <keyframe><key qpos="0 0 0.8 1 0 0 0 0 0 0 1 0 0 0"/></keyframe></mujoco>"""
    (tmp_path / "scene_locomani.xml").write_text(xml)
    (tmp_path / "scene_meta.json").write_text(json.dumps({"objects": {"crate": {}}, "robot_start": {"xy": [-1.5, 0]}}))
    result = strip_to_floor(tmp_path)
    names = [(c.tag, c.get("name")) for c in ET.parse(tmp_path / "scene_locomani.xml").getroot().find("worldbody")]
    assert ("body", "table") not in names and ("body", "crate") not in names
    assert ("body", "pelvis") in names and ("geom", "floor") in names
    assert ("camera", "reach_overview") in names
    meta = json.loads((tmp_path / "scene_meta.json").read_text())
    assert meta["objects"] == {} and meta["robot_start"]["xy"] == [0.0, 0.0]
    assert result["nq"] == 7


def test_envelope_overrides_apply_below_the_height_threshold():
    B._state.update(
        ENVELOPE={"tracking_cap": 0.15},
        OVERRIDES=[{"h_below": 0.50, "envelope": {"pelvis_height_target": 0.65, "base_anchor": "measured_xy"}}],
    )
    assert B.envelope_for({"h": 0.30}) == {
        "tracking_cap": 0.15,
        "pelvis_height_target": 0.65,
        "base_anchor": "measured_xy",
    }
    assert B.envelope_for({"h": 0.70}) == {"tracking_cap": 0.15}
    assert B.group_of({"h": 0.30}) == 0 and B.group_of({"h": 0.70}) == 1


def test_query_interpolates_boundary_and_flags_margin(tmp_path):
    from locomani.tasks.crate.reach_envelope.query import ReachEnvelope

    rows = []
    for y in (-0.1, 0.1):
        for h in (0.5, 0.7):
            for d in (0.3, 0.5):
                for yaw in (-10.0, 10.0):
                    rows.append(dict(y=y, h=h, d=d, yaw=yaw, trials=2, x_pass=0.5 if h > 0.6 else 0.4, x_fail=None))
    rows[0]["x_pass"] = None  # never passed: scored just below X_MIN
    (tmp_path / "boundary_table.json").write_text(json.dumps(rows))
    env = ReachEnvelope([tmp_path])
    assert env.query(0.30, 0.0, 0.7, 0.4, 0.0)["verdict"] == "reach"  # boundary 0.525, margin 0.225
    assert env.query(0.50, 0.0, 0.7, 0.4, 0.0)["verdict"] == "marginal"
    assert env.query(0.60, 0.0, 0.7, 0.4, 0.0)["verdict"] == "no_reach"
    assert env.query(0.30, 0.0, 0.9, 0.4, 0.0)["verdict"] == "unknown"  # h outside the measured box
    assert env.query(0.30, -0.1, 0.5, 0.3, -10.0)["boundary_x_m"] == pytest.approx(0.175)
