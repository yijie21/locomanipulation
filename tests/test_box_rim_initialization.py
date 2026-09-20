"""Consumer checks for preparing stance before the one recorded attempt."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from locomani.backends.sonic.agent_server import HAND_OPEN
from locomani.tasks.crate import box_rim_execution_cli as cli


@pytest.mark.parametrize("mode, reset_count", [("coordinated", 2), ("legacy_low_reach", 1)])
def test_recorded_attempt_starts_after_prepared_stance_reset(tmp_path, monkeypatch, mode, reset_count):
    (tmp_path / "scene_meta.json").write_text(json.dumps({"robot_start": {"xy": [0, -1.205]}}))
    events = []
    state = {"t": 0.0, "joints": np.zeros(29), "pelvis": {"yaw": np.pi / 2, "pos": [0, -0.83, 0.78]}}
    hand_commands = []

    class Command:
        def reset_pose(self):
            pass

        def set_hands(self, *, left, right):
            hand_commands.append((np.asarray(left).copy(), np.asarray(right).copy()))

        def exit_streamed(self):
            pass

        def start_control(self):
            pass

        def hold(self, seconds, **_kwargs):
            state["t"] += seconds

    class Primitives:
        cmd = Command()

        def __init__(self, **_kwargs):
            pass

        def obs(self):
            return state

        def _facing(self, _state):
            return (0, 1, 0)

        def close(self):
            pass

    class Reset:
        def pin_arms(self, **kwargs):
            events.append(("pins", kwargs))

        def reset(self, *, robot, **_kwargs):
            events.append(("reset", robot))
            state["pelvis"]["pos"] = [0, -0.83, 0.78]
            state["t"] += 0.02

        def start_video(self, _label):
            events.append(("video", state["pelvis"]["pos"][1]))

        def end_video(self):
            events.append(("video_end", None))

        def close(self):
            pass

    class PlannerArm:
        def hold(self, seconds):
            events.append(("ready_hold", seconds))
            state["t"] += seconds
            if seconds == 1.2:
                state["pelvis"]["pos"][1] -= 0.09

        def start(self, *, upper_reference, settle_seconds):
            events.append(("reseed", np.asarray(upper_reference).copy()))
            assert settle_seconds == 0

    class Runtime:
        def __init__(self, *_args, **kwargs):
            self.planner_arm = PlannerArm()
            self.ready_hands = np.asarray(kwargs["initial_hands"])
            events.append(("runtime_hands", self.ready_hands.copy()))

        def observe(self):
            return state

        def close(self):
            pass

    class Executor:
        def __init__(self, *_args):
            pass

        def execute(self):
            events.append(("execute", state["pelvis"]["pos"][1]))
            return SimpleNamespace(ok=False, why="fixture_stop", produced=False, execution={})

    monkeypatch.setattr(cli, "require_pose_ownership", lambda: None)
    monkeypatch.setattr(cli, "Primitives", Primitives)
    monkeypatch.setattr(cli, "EpisodeControlClient", Reset)
    monkeypatch.setattr(cli, "ArmIK", lambda _path: object())
    monkeypatch.setattr(cli.box_rim_cli, "_parse_robot", lambda *_args: None)
    monkeypatch.setattr(cli, "BoxRimPlanner", lambda *_args: object())
    monkeypatch.setattr(cli, "SonicBoxRimRuntime", Runtime)
    monkeypatch.setattr(cli, "BoxRimExecutor", Executor)
    monkeypatch.setattr(cli, "reanchor", lambda _p: None)
    monkeypatch.setattr(cli.upper_ref, "shared", lambda: SimpleNamespace(wait=lambda _timeout: None))
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(cli, "_initialization_report", lambda *_args: {})
    pose = {
        "waist": [0.0, 0.0, 0.0],
        "left_arm": [-0.35, 0.2, 0.0, 0.8, 0.0, 0.0, 0.0],
        "right_arm": [-0.35, -0.2, 0.0, 0.8, 0.0, 0.0, 0.0],
        "hands": HAND_OPEN.tolist(),
    }
    request = {
        "path": tmp_path / "request.json",
        "scene_xml": tmp_path / "scene.xml",
        "robot_raw": {},
        "target": {"body": "box"},
        "target_key": "box",
        "box": object(),
        "planner": object(),
        "execution": SimpleNamespace(approach_mode=mode),
        "runtime": {
            "scene_dir": tmp_path,
            "initial_ready_pose": pose,
            "reset_before_episode": True,
            "robot_dxy": [0, 0.375],
            "robot_yaw": np.pi / 2,
            "video_label": "fixture",
        },
    }
    result = cli.run_live(request)
    names = [name for name, _value in events]
    resets = [value for name, value in events if name == "reset"]
    assert len(resets) == reset_count
    assert all(robot["dof_from"] == 12 for robot in resets)
    assert all(np.allclose(left, HAND_OPEN) and np.allclose(right, HAND_OPEN) for left, right in hand_commands)
    assert np.allclose(next(value for name, value in events if name == "runtime_hands"), HAND_OPEN)
    if mode == "coordinated":
        assert names.index("ready_hold") < max(i for i, name in enumerate(names) if name == "reset")
        assert names.index("reseed") < names.index("video") < names.index("execute")
        assert next(value for name, value in events if name == "execute") == pytest.approx(-0.83)
    else:
        assert "reseed" not in names
        assert next(value for name, value in events if name == "execute") == pytest.approx(-0.92)
    assert result["execution"]["initialization"]["initialization_resets"] == reset_count
    assert result["execution"]["initialization"]["video_start_simulation_time"] > 0
