"""Settings reach clocks, controllers and the real evaluation parser."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from locomani.backends.sonic.config import LockstepConfig, SonicBackendConfig


def test_upper_slew_controls_the_reference_sent_to_the_commander():
    from locomani.backends.sonic.upper_ref import UpperRamp

    cmd = Mock()
    ramp = UpperRamp(cmd, np.zeros(17), config=SonicBackendConfig(upper_slew_deg=7.0))
    ramp.push(np.ones(17))
    np.testing.assert_allclose(cmd.set_upper_body.call_args.args[0], np.radians(7.0))


def test_reference_tap_selects_the_instances_actual_debug_port(monkeypatch):
    from locomani.backends.sonic.upper_ref import RefTap
    from locomani.pipeline.stack import resolve_env

    env = resolve_env(SonicBackendConfig(), instance=1, inherit={})
    monkeypatch.setenv("INSTANCE", env["INSTANCE"])
    monkeypatch.delenv("TF_OUT_PORT", raising=False)
    assert RefTap().port == int(env["TF_OUT_PORT"]) == 5567


def test_lockstep_runner_passes_config_to_the_barrier_and_jitter_state(monkeypatch):
    import threading

    import zmq

    from locomani.backends.sonic.lockstep_sim import LockstepRunner

    monkeypatch.syspath_prepend(str(Path("src/teleop_free").resolve()))
    monkeypatch.setattr(zmq.Context, "instance", lambda: Mock())
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    config = LockstepConfig(ack_timeout=2.5, warmup_ticks=9, cmd_delay=2, jitter=True, jitter_seed=17)
    sim = SimpleNamespace(sim_env=object(), unitree_bridge=object(), sim_dt=0.005, image_dt=0.04)
    runner = LockstepRunner(sim, 5556, 5557, config=config)
    assert runner.barrier.ack_timeout == 2.5
    assert (runner._warmup, runner._cmd_delay, runner._jitter) == (9, 2, True)
    np.testing.assert_array_equal(runner._jrng.integers(100, size=5), np.random.default_rng(17).integers(100, size=5))


@pytest.mark.parametrize("capture", [False, True])
def test_ready_pilot_command_is_accepted_by_the_actual_eval_parser(tmp_path, capture):
    from locomani.pipeline.eval_closed_loop import parse_args
    from locomani.pipeline.eval_config import EvalConfig
    from locomani.pipeline.run_ready_pilot import evaluation_command

    inputs = SimpleNamespace(eval_rounds=7, stand_retries=0, capture_execution=capture)
    command = evaluation_command(inputs, tmp_path, tmp_path / "reach.json", None)
    argv = [str(value) for value in command[3 if capture else 2 :]]
    parsed = parse_args(argv)
    config = EvalConfig.load(parsed.config)
    assert (config.rounds, config.stand_retries, config.seed) == (7, 0, 16800)
    assert config.backend.dataset_fps == 25.0
    assert config.client == "async"
    assert parsed.cases == str(tmp_path / "eval_cases.json")
