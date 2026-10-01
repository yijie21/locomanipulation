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


def test_the_async_client_engages_the_deploy_it_then_refuses_to_command():
    """`client=async` and `backend.clean_deploy` are one decision.

    The async client installs CleanClock's commander, which drops every planner
    command -- including the `start_control` that starts the deploy's control
    loop. SONIC_AUTO_START is what stands in for it, and the stack exports that
    only for `clean_deploy`. Split apart (which is what the config groups did to
    the pre-refactor `wall) export CLEAN_DEPLOY=1; CL="--client async"`), the
    deploy takes the mode switch, receives the standing token, never starts
    control, and the robot falls where the reset dropped it: measured on
    2026-09-17 as 245 received tokens and zero `Loop timing` lines, reported as
    `[stand] robot did not stabilise after reset`.
    """
    from locomani.pipeline.eval_config import EvalConfig

    assert EvalConfig.from_dict({"client": "async"}).backend.clean_deploy is True
    # An explicit false in the file cannot re-open the gap either.
    assert EvalConfig.from_dict({"client": "async", "backend": {"clean_deploy": False}}).backend.clean_deploy is True
    for other in ("lockstep", "blocking"):
        assert EvalConfig.from_dict({"client": other}).backend.clean_deploy is False


def test_the_image_source_is_a_closed_choice_and_the_mirror_needs_a_wall_clock_client():
    assert EvalConfig().image_source == "mujoco"
    assert EvalConfig().with_overrides(["image_source=isaac"]).image_source == "isaac"
    with pytest.raises(ValueError, match="not in"):
        EvalConfig().with_overrides(["image_source=unreal"])
    with pytest.raises(ValueError, match="wall-clock client"):
        EvalConfig.from_dict({"client": "lockstep", "image_source": "isaac"})


def test_the_image_source_reaches_every_wall_clock_observation_builder():
    from pathlib import Path

    from locomani import paths

    src = Path(paths.REPO) / "src/locomani"
    for client in ("backends/sonic/vla_client.py", "backends/sonic/vla_round.py"):
        assert "ObsBuilder(task, image_source=config.image_source)" in (src / client).read_text()
    rounds = (src / "tasks/crate/vla_rounds.py").read_text()
    assert "ObsBuilder(config.prompt, schema=SONIC_CRATE, image_source=image_source)" in rounds
