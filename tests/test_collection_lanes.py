"""Collection in lanes: how many fit, where they go, and how the lanes share one run's batches.

A lane is one stack plus the driver process that runs batches on it (`locomani.pipeline.lanes`). The arithmetic
is pure; the batch bookkeeping is the file several lane processes share (`batches.claim_batch` / `finish_batch`).
"""

from __future__ import annotations

import json
import os
import socket
from unittest.mock import Mock

import pytest

from locomani.pipeline import batches, lanes
from locomani.stack import StackSpec, manager

LIVE = dict(cores=3.0, quota=15.36, pids=1860, pids_max=3840)
WAREHOUSE = lanes.lane_cost(True, "offline", 4)


def test_a_recording_lane_costs_more_threads_the_longer_its_batches():
    assert lanes.lane_cost(True, "offline", 4)["pids"] == 760
    assert lanes.lane_cost(True, "offline", 10)["pids"] == 1600  # the old single-stack gate's ~1800
    assert lanes.lane_cost(False, "offline", 10)["pids"] == lanes.NO_RECORD_PIDS  # no exporter, no growth


def test_the_smallest_resource_decides_and_each_limit_is_reported():
    limits, slots = lanes.fit(LIVE, WAREHOUSE, {"0": 30000, "1": 30000}, per_gpu=4, instances=8, want=None)
    assert limits["cpu"] == 3  # (15.36 - 3.0) / 4
    assert limits["pids"] == 2  # (3840 - 200 - 1860) / 760
    assert limits["gpu"] == 8 and slots == {"0": 4, "1": 4}
    assert min(limits.values()) == 2


def test_a_requested_count_caps_and_a_busy_machine_gives_none():
    limits, _ = lanes.fit(LIVE, WAREHOUSE, {"0": 30000}, per_gpu=4, instances=8, want=1)
    assert min(limits.values()) == 1
    full = dict(LIVE, cores=13.0)
    limits, _ = lanes.fit(full, WAREHOUSE, {"0": 30000}, per_gpu=4, instances=8, want=None)
    assert limits["cpu"] == 0 and min(limits.values()) == 0


def test_a_card_takes_as_many_lanes_as_its_memory_and_render_mode_allow():
    _, slots = lanes.fit(LIVE, WAREHOUSE, {"0": 5000, "1": 500}, per_gpu=4, instances=8, want=None)
    assert slots == {"0": 2, "1": 0}
    _, live = lanes.fit(LIVE, lanes.lane_cost(True, "live", 4), {"0": 30000}, per_gpu=1, instances=8, want=None)
    assert live == {"0": 1}  # rendering sims never share a card


def test_lanes_alternate_over_the_cards_and_take_one_instance_each():
    placed = lanes.spread(3, {"0": 4, "1": 4}, [2, 3, 4, 5])
    assert [(lane.index, lane.instance, lane.gpu) for lane in placed] == [(0, 2, "0"), (1, 3, "1"), (2, 4, "0")]
    assert lanes.spread(3, {"0": 1, "1": 0}, [2, 3, 4]) == [lanes.Lane(0, 2, "0")]


def test_lanes_asked_for_come_from_the_command_line_then_the_experiment():
    assert lanes.wanted(None, {"collect": {}}) is None
    assert lanes.wanted(None, {"collect": {"lanes": 1}}) == 1
    assert lanes.wanted("3", {"collect": {"lanes": 1}}) == 3
    assert lanes.wanted("auto", {"collect": {"lanes": 1}}) is None
    with pytest.raises(SystemExit, match="positive number or auto"):
        lanes.wanted("0", {"collect": {}})


def test_the_plan_is_written_and_read_back(tmp_path):
    plan = lanes.Plan([lanes.Lane(0, 2, "0"), lanes.Lane(1, 3, "1")], {"cpu": 2}, dict(LIVE), WAREHOUSE, {0: "busy"})
    lanes.write(tmp_path, plan)
    again = lanes.read(tmp_path)
    assert again.lanes == plan.lanes and again.busy == {0: "busy"} and again.cost == WAREHOUSE
    assert "2 lanes in parallel" in again.describe()[0] and "--lanes 1" in again.describe()[-1]
    assert lanes.read(tmp_path / "elsewhere") is None


def test_a_bound_port_makes_an_instance_busy(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "session_root", lambda instance: tmp_path)
    monkeypatch.setattr(manager.paths, "TMP", str(tmp_path))
    instance = 9
    assert manager.instance_busy(instance) is None
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 5555 + 10 * instance))
        server.listen()
        assert "in use" in manager.instance_busy(instance)
    (tmp_path / "stack.json").write_text(json.dumps({"pids": {"sim": os.getpid()}}))
    assert "runs a stack" in manager.instance_busy(instance)


# -- the batches the lanes share -----------------------------------------------------------------


def _claim(inputs, lane, target=10, max_batches=5):
    return batches.claim_batch(
        inputs, lane, target=target, max_batches=max_batches, manifest_for=lambda b: inputs.parent / f"m_{b:02d}"
    )


def test_lanes_claim_distinct_batches_lowest_first_and_only_finished_ones_are_listed(tmp_path):
    inputs = tmp_path / "collect" / "inputs.json"
    assert [_claim(inputs, lane) for lane in (0, 1, 2)] == [0, 1, 2]
    state = batches.finish_batch(inputs, 1, "s1", tmp_path / "m_01", kept=2)  # lane 1 finishes first
    assert state["sessions"] == ["s1"] and state["numbers"] == [1] and state["batches"] == 1 and state["kept"] == 2
    assert set(state["running"]) == {"0", "2"}
    assert _claim(inputs, 1) == 3


def test_no_claim_once_the_target_is_kept_or_the_budget_is_spent(tmp_path):
    inputs = tmp_path / "inputs.json"
    assert _claim(inputs, 0, target=2) == 0
    batches.finish_batch(inputs, 0, "s0", tmp_path / "m_00", kept=2)
    assert _claim(inputs, 0, target=2) is None
    inputs2 = tmp_path / "other" / "inputs.json"
    assert [_claim(inputs2, 0, max_batches=2) for _ in range(3)] == [0, 1, None]


def test_a_stop_decision_is_taken_under_the_lock_and_ends_the_claims(tmp_path):
    inputs = tmp_path / "inputs.json"
    _claim(inputs, 0)
    state = batches.finish_batch(inputs, 0, "s0", tmp_path / "m_00", kept=1, decide=lambda s: {"verdict": "above"})
    assert state["stop"] == {"verdict": "above"}
    assert _claim(inputs, 0) is None


def test_an_interrupted_lane_run_reruns_its_claims_from_the_lowest_number(tmp_path):
    inputs = tmp_path / "inputs.json"
    for lane in (0, 1):
        _claim(inputs, lane)
    (tmp_path / "m_00").write_text('{"ok": true}\n')  # batch 0 got a round in before the interruption
    batches.finish_batch(inputs, 1, "s1", tmp_path / "m_01", kept=1)

    state = batches.resume_state(inputs)

    assert "running" not in state and state["manifests"] == [str(tmp_path / "m_01")]
    assert not (tmp_path / "m_00").exists() and list(tmp_path.glob("interrupted_m_00_*"))
    assert _claim(inputs, 0) == 0  # the unfinished batch runs again, same number, same seed


def test_a_single_lane_state_file_reads_as_finished_batches_in_order(tmp_path):
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps(dict(sessions=["s0", "s1"], manifests=["m0", "m1"], kept=3, batches=2)))
    assert _claim(inputs, 0) == 2


def test_a_lanes_first_batch_takes_over_the_session_its_exporter_opened_for_it(tmp_path):
    stack = Mock(spec=["spec", "session", "pids", "env", "restart_exporter"])
    stack.spec, stack.session, stack.pids, stack.env = (
        StackSpec(record=True),
        tmp_path / "s",
        {"exporter": os.getpid()},
        {},
    )
    assert batches.open_exporter_session(stack, tmp_path, 3, starter_batch=3) == str(tmp_path / "s")
    stack.restart_exporter.assert_not_called()

    def restart():
        stack.session = tmp_path / "fresh"

    stack.restart_exporter.side_effect = restart
    assert batches.open_exporter_session(stack, tmp_path, 4, starter_batch=3) == str(tmp_path / "fresh")
    assert stack.env["TF_FRAME_AUDIT_PATH"].endswith("frame_audit_04.jsonl")
