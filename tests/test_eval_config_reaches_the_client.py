"""Resolved evaluation settings reach real consumers without shared mutable flags."""

from dataclasses import replace

import numpy as np
import pytest

from locomani.backends.sonic.vla_client import _ensemble, _grip_gain, _stalled
from locomani.backends.sonic.vla_obs import PolicyLink
from locomani.pipeline.eval_config import EvalConfig


def test_observation_dump_and_snap_are_independent_per_link(tmp_path, monkeypatch):
    config = EvalConfig().with_overrides([f"backend.obs_dump={tmp_path}", "snap=false"])
    configured = PolicyLink(config=config)
    default = PolicyLink()
    raw = np.full((2, 64), 0.123456)
    answer = {"action.motion_token": raw, "action.right_gripper": np.zeros(2)}
    monkeypatch.setattr(PolicyLink, "_call", lambda *a: answer)
    try:
        np.testing.assert_array_equal(configured.get_chunk({"state.test": [2]})[0], raw)
        assert not np.array_equal(default.get_chunk({})[0], raw)
        files = list(tmp_path.glob("*.npz"))
        assert len(files) == 1
        with np.load(files[0]) as saved:
            np.testing.assert_array_equal(saved["state.test"], [2])
            np.testing.assert_array_equal(saved["tok"], raw)
    finally:
        for link in (configured, default):
            link.sock.close(0)
            link.ctx.term()


def test_gain_ensemble_and_stall_settings_change_decisions_without_leaking():
    config = EvalConfig().with_overrides(["grip_gain=3.0", "ensemble_m=0.0", "stall_abort_s=10.0"])
    assert _grip_gain(0.018, config=config) < 0.018
    assert _grip_gain(0.018) == 0.018
    buf = [(np.zeros((1, 64)), [0.0], 0), (np.ones((1, 64)), [1.0], 0)]
    assert _ensemble(buf, 0, 25, config=config)[1] == pytest.approx(0.5)
    assert _ensemble(buf, 0, 25)[1] > 0.5
    trace = [{"t": t, "pelvis_xy": [0.0, 0.0]} for t in range(11)]
    assert _stalled(trace, False, config=config)
    assert not _stalled(trace, False)
    assert not _stalled(trace, True, config=config)


def test_clean_clock_uses_the_supplied_dataset_rate(monkeypatch):
    import threading

    from locomani.backends.sonic import initial_pose
    from locomani.backends.sonic.clean_clock import CleanClock

    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    monkeypatch.setattr(initial_pose, "initial_token", lambda: np.zeros(64))
    config = EvalConfig().with_overrides(["backend.dataset_fps=25.0"])
    assert CleanClock(config=config).dt == pytest.approx(1 / 25)
    assert CleanClock(config=replace(config, backend=replace(config.backend, dataset_fps=30))).dt == pytest.approx(
        1 / 30
    )


def test_clock_commanders_do_not_change_other_instances(monkeypatch):
    import threading
    from types import SimpleNamespace

    from locomani.backends.sonic.agent_server import AgentCommander
    from locomani.backends.sonic.clean_clock import clean_commander_class
    from locomani.backends.sonic.lockstep_client import lockstep_commander_class

    sent, first, second = [], [], []
    monkeypatch.setattr(AgentCommander, "__init__", lambda self: None)
    monkeypatch.setattr(AgentCommander, "command", lambda self, **kw: sent.append(kw))
    original = AgentCommander.command
    clock = SimpleNamespace(cmd=object(), lock=threading.RLock(), hold=first.append)
    other = SimpleNamespace(hold_frames=second.append)
    clean = clean_commander_class(clock)()
    lockstep = lockstep_commander_class(other)()
    clean.hold(0.2)
    lockstep.hold(0.2)
    assert first == [0.2] and second == [10]
    clean.command(start=True, stop=False, planner=True)
    lockstep.command(start=True, stop=False, planner=True)
    assert sent == []
    assert AgentCommander.command is original
    AgentCommander().command(start=True, stop=False, planner=True)
    assert len(sent) == 1
