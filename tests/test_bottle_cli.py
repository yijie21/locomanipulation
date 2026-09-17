"""Exercise the resolver and its actual runtime entry without starting a robot."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from locomani import profiles
from locomani.tasks.bottle.cli_args import GymOptions, parse_args, resolve_options
from locomani.tasks.bottle.config import BottleTaskConfig
from locomani.tasks.bottle.reach_config import DEFAULT_REACH

ROOT = Path(__file__).resolve().parents[1]
# task=bottle_visible as a standalone JSON file, for exercising --task-config's own path-loading
# (BottleTaskConfig.load reads JSON only); configs/tasks/bottle_visible.json is retired.
PROFILE = str(Path(tempfile.mkdtemp()) / "bottle_visible.json")
profiles.load("task", "bottle_visible").dump(PROFILE)


def test_no_profile_preserves_legacy_gym_defaults():
    args = parse_args([])
    assert args == GymOptions()
    assert (args.spawn_off, args.spawn_yaw_err, args.spawn_lateral, args.face_object) == ((1.0, 3.0), 90.0, 0.35, False)


def test_resolution_uses_passed_values_without_reading_their_source_paths():
    profile = BottleTaskConfig.load(PROFILE)
    explicit = {"task_config": "/does-not-exist/profile.json", "spawn_yaw_err": 90.0}
    first = resolve_options(explicit, profile=profile)
    assert first == resolve_options(explicit, profile=profile)
    assert first.spawn_yaw_err == 90.0
    assert first.reach_config_obj is profile.reach
    assert explicit == {"task_config": "/does-not-exist/profile.json", "spawn_yaw_err": 90.0}


def test_explicit_values_equal_to_old_defaults_override_the_profile():
    args = parse_args(
        [
            "--task-config",
            PROFILE,
            "--spawn-yaw-err",
            "90",
            "--spawn-lateral",
            "0.35",
            "--edge-jitter",
            "0.10",
            "--stand-retries",
            "1",
            "--targets",
            "pool",
            "--no-face-object",
        ]
    )
    assert (
        args.spawn_yaw_err,
        args.spawn_lateral,
        args.edge_jitter,
        args.stand_retries,
        args.targets,
        args.face_object,
    ) == (90.0, 0.35, 0.10, 1, "pool", False)
    assert args.spawn_edge == (1.1, 1.3)
    assert args.sources["spawn_yaw_err"] == "cli"
    assert args.sources["spawn_edge"].startswith("task:")


def test_explicit_reach_is_loaded_once_and_overrides_profile_even_for_vla():
    path = str(ROOT / "configs/collection/ready_grasp_baseline.json")
    for vla in (False, True):
        args = parse_args(["--task-config", PROFILE, "--reach-config", path], vla=vla)
        assert args.reach_config_obj.strategy == "ready"
        assert args.sources["reach"] == f"reach:{path}"
    assert parse_args(["--task-config", PROFILE], vla=True).reach_config_obj == DEFAULT_REACH


def test_explicit_offset_selects_offset_sampling_in_scene_setup(tmp_path):
    from types import SimpleNamespace

    from geometry_fixture import DIRECTORY

    from locomani.tasks.bottle import grasp_gym

    args = parse_args(
        ["--task-config", PROFILE, "--spawn-off", "3", "--sample-only", "--write-cases", str(tmp_path / "cases.json")]
    )
    with (
        patch.object(grasp_gym, "Primitives") as robot,
        patch.object(grasp_gym, "EpisodeControlClient") as control,
        patch.object(grasp_gym, "SCENE_META", str(DIRECTORY / "scene_meta.json")),
        patch.object(grasp_gym.reach_map, "ReachMap"),
        patch.object(
            grasp_gym,
            "plan_for_target",
            return_value=SimpleNamespace(grip_z=0.72, rel_cands=[], squat_hint=None, sgn=1, ax=0),
        ),
    ):
        setup = grasp_gym._build_scene_and_plan(args)
    robot.assert_not_called()
    control.assert_not_called()
    assert setup.edge_range is None
    assert (setup.off_lo, setup.off_hi) == (3.0, 3.0)
    assert args.sources["spawn_edge"] == "cli (--spawn-off selects offset sampling)"


@pytest.mark.parametrize("vla", [False, True])
def test_profile_boolean_options_and_explicit_negative_overrides(tmp_path, vla):
    path = tmp_path / "task.json"
    data = json.loads(Path(PROFILE).read_text())
    data.update(one_shot=True, video_include_init=True)
    path.write_text(json.dumps(data))
    args = parse_args(["--task-config", str(path)], vla=vla, video_include_init=False)
    assert args.one_shot and args.video_include_init
    assert args.sources["one_shot"] == args.sources["video_include_init"] == f"task:{path}"
    override = parse_args(["--task-config", str(path), "--no-one-shot", "--no-video-include-init"], vla=vla)
    assert not override.one_shot and not override.video_include_init
    assert override.sources["one_shot"] == override.sources["video_include_init"] == "cli"


def test_profile_is_loaded_once_and_kept_for_place_consumer(tmp_path):
    path = tmp_path / "task.json"
    data = json.loads(Path(PROFILE).read_text())
    data["goal"] = "basket"
    path.write_text(json.dumps(data))
    args = parse_args(["--task-config", str(path)])
    path.unlink()
    assert args.task_config_obj.goal == "basket"
    assert args.reach_config_obj is args.task_config_obj.reach


def test_diagnostic_video_override_is_explicit_and_parsing_has_no_runtime_effect():
    from locomani.tasks.bottle import grasp_gym

    before = grasp_gym.GymRuntime()
    args = parse_args(["--no-video-include-init"], video_include_init=True)
    assert not args.video_include_init
    assert grasp_gym.GymRuntime() == before
    assert parse_args([], video_include_init=True).video_include_init


def test_gym_main_delivers_the_resolved_options_to_the_round_driver():
    from locomani.tasks.bottle import grasp_gym

    with (
        patch.object(sys, "argv", ["gym", "--task-config", PROFILE, "--spawn-yaw-err", "90"]),
        patch.object(grasp_gym, "_build_scene_and_plan", return_value="scene") as setup,
        patch.object(grasp_gym.round_loop, "round_loop") as run,
    ):
        grasp_gym.main(runtime=grasp_gym.GymRuntime(policy=object()))
    options = setup.call_args.args[0]
    assert options.spawn_yaw_err == 90.0
    assert options.reach_config_obj == DEFAULT_REACH
    run.assert_called_once_with(options, "scene", runtime=setup.call_args.kwargs["runtime"])


def test_report_mode_does_not_start_runtime(capsys):
    from locomani.tasks.bottle import grasp_gym

    with (
        patch.object(sys, "argv", ["gym", "--task-config", PROFILE, "--print-config"]),
        patch.object(grasp_gym, "_build_scene_and_plan") as setup,
    ):
        grasp_gym.main()
    setup.assert_not_called()
    assert json.loads(capsys.readouterr().out)["options"]["reach"]["strategy"] == "coupled"


def test_resolver_import_does_not_load_simulator_or_gym():
    code = "import sys; sys.path.insert(0, 'src'); from locomani.tasks.bottle.cli_args import parse_args; parse_args([]); assert 'mujoco' not in sys.modules; assert 'locomani.tasks.bottle.grasp_gym' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True, capture_output=True)


@pytest.mark.parametrize(
    "argv",
    [
        ["--sample-only"],
        ["--cases", "cases.json", "--until-ok", "1"],
        ["--spawn-edge", "1", "2", "3"],
        ["--unknown-flag"],
    ],
)
def test_invalid_requests_fail_before_runtime(argv):
    with pytest.raises(SystemExit) as caught:
        parse_args(argv)
    assert caught.value.code == 2
