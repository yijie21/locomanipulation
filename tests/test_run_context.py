"""The checks every stage entry runs before it starts anything.

The admission thresholds are the expensive part to get wrong in either direction. Too strict and a
run refuses on a machine that could carry it; too loose and the container's CPU quota saturates,
the simulation misses its wall clock, and every timed expert phase fails as a control error -- a
failure that reads like a robot problem for hours. `shortfall` is pure so the thresholds can be
examined with numbers instead of with a GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from locomani.pipeline import run_context as run_common  # noqa: E402

IDLE = dict(cores=1.0, pids=1000, quota=15.36, pids_max=3840, gpu_free_mib=32000)


def test_an_idle_machine_admits_every_stage():
    assert all(run_common.shortfall(stage, IDLE) is None for stage in run_common.NEEDS)


def test_a_busy_quota_refuses_the_stages_that_need_cores():
    live = dict(IDLE, cores=9.0)  # 9 + 8 > 15.36
    assert "cpu" in run_common.shortfall("collect", live)
    assert run_common.shortfall("convert", live) is None  # 9 + 4 still fits


def test_a_stack_without_the_exporter_is_charged_its_own_cores():
    live = dict(IDLE, cores=9.2)  # 9.2 + 8 > 15.36, 9.2 + 5 fits
    assert "cpu" in run_common.shortfall("collect", live)
    assert run_common.shortfall("collect", live, record=False) is None
    assert "cpu" in run_common.shortfall("collect", dict(live, cores=11.0), record=False)


def test_a_driver_attaching_to_a_live_stack_is_not_charged_the_stack_again():
    """The stack's ~5 cores are already inside `cores`; the driver only needs its own."""
    live = dict(IDLE, cores=9.5)  # others 4.5 + our stack 5: a fresh stack would not fit
    assert "cpu" in run_common.shortfall("collect", live)
    assert run_common.shortfall("collect", live, record=False, already=313) is None  # 9.5 + 2 fits
    assert "cpu" in run_common.shortfall("collect", dict(live, cores=14.0), record=False, already=313)


def test_the_pid_ceiling_counts_the_margin():
    """A stack that passes the check still has to fork; the margin is what it forks into."""
    live = dict(IDLE, pids=3840 - 200 - 1800)  # exactly at the line
    assert run_common.shortfall("collect", live) is None
    assert "pids" in run_common.shortfall("collect", dict(live, pids=live["pids"] + 1))


def test_a_card_someone_else_is_using_refuses_training():
    live = dict(IDLE, gpu_free_mib=20000)
    assert "gpu" in run_common.shortfall("train", live)
    assert run_common.shortfall("eval", live) is None  # eval needs less than training does
    assert run_common.shortfall("convert", live) is None  # and convert needs no card at all


def test_a_measurement_that_failed_does_not_refuse():
    """Refusing on a number we could not read would make the scripts unusable off this container."""
    assert run_common.shortfall("collect", {}) is None


def test_require_headroom_names_the_reason(monkeypatch):
    monkeypatch.setattr(run_common, "measure", lambda gpu=None: dict(IDLE, cores=15.0))
    with pytest.raises(SystemExit, match="cpu"):
        run_common.require_headroom("collect", gpu="0")


def test_open_run_copies_the_experiment_verbatim(tmp_path, monkeypatch):
    monkeypatch.setattr(run_common, "RUNS", tmp_path / "runs")
    experiment = tmp_path / "exp.yaml"
    experiment.write_text("# a comment the copy keeps\nname: demo\ntask:\n  plugin: bottle\n")
    cfg, run_dir = run_common.open_run(experiment)
    assert cfg["name"] == "demo" and run_dir.name == "demo"
    assert (run_dir / "config.yaml").read_text() == experiment.read_text()  # comments and order survive


def test_open_run_takes_a_name_override(tmp_path, monkeypatch):
    monkeypatch.setattr(run_common, "RUNS", tmp_path / "runs")
    experiment = tmp_path / "exp.yaml"
    experiment.write_text("name: demo\n")
    cfg, run_dir = run_common.open_run(experiment, "tonight")
    assert cfg["name"] == "tonight" and run_dir.name == "tonight"


def test_open_run_refuses_to_change_a_run_in_flight(tmp_path, monkeypatch):
    monkeypatch.setattr(run_common, "RUNS", tmp_path / "runs")
    experiment = tmp_path / "exp.yaml"
    experiment.write_text("name: demo\nseed: 1\n")
    run_common.open_run(experiment)
    experiment.write_text("name: demo\nseed: 2\n")
    with pytest.raises(SystemExit, match="differs"):
        run_common.open_run(experiment)


def test_a_holdout_that_takes_every_success_is_refused_before_collecting():
    collect = {"record": True, "successes": 10, "batch_rounds": 20, "max_batches": 1}
    run_common.require_train_split({"train": {"holdout": 9}, "collect": collect})
    with pytest.raises(SystemExit, match="training set would be empty"):
        run_common.require_train_split({"train": {"holdout": 10}, "collect": collect})


def test_a_holdout_that_takes_every_round_of_the_budget_is_refused_before_collecting():
    """A copy cut to 10 rounds kept `holdout: 10`: it passed the successes check and every line's convert
    refused it afterwards (docs reproduction 2026-09-25, P7)."""
    collect = {"record": True, "successes": 20, "batch_rounds": 10, "max_batches": 1}
    run_common.require_train_split({"train": {"holdout": 9}, "collect": collect})
    with pytest.raises(SystemExit, match="10 rounds"):
        run_common.require_train_split({"train": {"holdout": 10}, "collect": collect})


def test_a_run_that_records_nothing_has_nothing_to_split():
    collect = {"record": False, "successes": 10, "batch_rounds": 10, "max_batches": 1}
    run_common.require_train_split({"train": {"holdout": 10}, "collect": collect})


def test_summary_accumulates_one_entry_per_stage(tmp_path):
    run_common.summarise(tmp_path, "collect", kept=200, attempts=299)
    path = run_common.summarise(tmp_path, "convert", episodes=190)
    summary = yaml.safe_load(path.read_text())
    assert summary["collect"]["kept"] == 200 and summary["convert"]["episodes"] == 190
    assert all("finished" in entry for entry in summary.values())


def test_show_prints_the_command_and_its_own_settings(capsys):
    run_common.show("collect_00", ["python", "grasp_gym.py", "--rounds", 10], {"SIM_VIDEO_DIR": "/tmp/v"})
    out = capsys.readouterr().out
    assert "grasp_gym.py --rounds 10" in out and "SIM_VIDEO_DIR=/tmp/v" in out


def test_a_run_that_records_nothing_is_not_charged_for_the_exporter():
    """Measured 2026-09-21: sim + deploy moved pids.current 2269 -> 2441, peak 2445.

    The 1800 in NEEDS is the exporter. Charging it to a run that starts no exporter -- every
    evaluation, and the dry pilots that sample and score rounds without writing episodes -- refuses
    work the machine can carry, which is how a whole class of checks went unrun on a loaded box.
    """
    loaded = dict(IDLE, pids=2264)
    assert run_common.shortfall("collect", loaded) is not None  # recording still refused
    assert run_common.shortfall("collect", loaded, record=False) is None
    assert run_common.shortfall("eval", loaded, record=False) is None


def test_the_exporter_free_budget_still_refuses_a_full_container():
    """Lower is not absent: a container with no room left must still be refused."""
    full = dict(IDLE, pids=3500)
    assert run_common.shortfall("collect", full, record=False) is not None
    assert run_common.shortfall("eval", full, record=False) is not None


def test_a_stage_needing_less_than_the_exporter_free_budget_keeps_its_own_number():
    """convert asks for 200; `record=False` must not raise it to NO_EXPORTER_PIDS."""
    assert run_common.NEEDS["convert"]["pids"] < run_common.NO_EXPORTER_PIDS
    tight = dict(IDLE, pids=3400)
    assert run_common.shortfall("convert", tight, record=False) is None


def test_a_scene_name_resolves_to_its_directory_in_the_run_and_its_copy(tmp_path, monkeypatch):
    """The copy keeps the name verbatim; both readers hand consumers the directory."""
    monkeypatch.setattr(run_common, "RUNS", tmp_path / "runs")
    experiment = tmp_path / "exp.yaml"
    experiment.write_text("name: r\nscene:\n  name: warehouse\n  cases: null\n")
    cfg, run_dir = run_common.open_run(experiment)
    assert cfg["scene"]["dir"] == "outputs/box_rim_execution_20260920/scene_reverse_camera"
    assert "dir" not in yaml.safe_load((run_dir / "config.yaml").read_text())["scene"]
    assert run_common.load_run_config(run_dir)["scene"] == cfg["scene"]


def test_every_named_scene_exists_and_a_dir_without_a_name_is_left_alone():
    names = yaml.safe_load(run_common.SCENES.read_text())
    boxes = {f"warehouse_box_45x{length}x{height}" for length in (20, 32) for height in (18, 33)}
    stock = {"bottle", "crate", "warehouse", "warehouse_textured", "walk", "box_platform", *boxes}
    mars = {"bottle_mars", "warehouse_textured_mars", "box_platform_mars", *(f"{box}_mars" for box in boxes)}
    assert set(names) == stock | mars
    for entry in names.values():
        assert "example" in entry or "build" in entry
        if "example" in entry:
            example = yaml.safe_load((run_common.ROOT / entry["example"]).read_text())
            assert run_common.resolve_scene(example)["scene"]["dir"] == entry["dir"]
    cfg = {"scene": {"dir": "outputs/scene/sim_lowcom"}}
    assert run_common.resolve_scene(cfg) == {"scene": {"dir": "outputs/scene/sim_lowcom"}}


def test_an_unknown_or_doubly_given_scene_is_refused():
    with pytest.raises(
        SystemExit, match="bottle, bottle_mars, box_platform, box_platform_mars, crate, walk, warehouse"
    ):
        run_common.resolve_scene({"scene": {"name": "blue"}})
    with pytest.raises(SystemExit, match="give one"):
        run_common.resolve_scene({"scene": {"name": "crate", "dir": "outputs/scene/sim"}})


def test_isaac_scene_reads_the_stage_and_look_from_the_scene_registry(tmp_path, monkeypatch):
    registry = tmp_path / "scenes.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "room": {"dir": "outputs/scene/room", "isaac": {"dir": "room_usd", "rtx_settings": ["/rtx/a=1"]}},
                "floor": {"dir": "outputs/scene/floor"},
            }
        )
    )
    monkeypatch.setattr(run_common.paths, "ISAAC_ASSETS", str(tmp_path / "isaac"))
    with pytest.raises(SystemExit, match="by name"):
        run_common.isaac_scene({"scene": {"dir": "outputs/scene/room"}}, registry)
    with pytest.raises(SystemExit, match="has no Isaac stage"):
        run_common.isaac_scene({"scene": {"name": "floor"}}, registry)
    with pytest.raises(SystemExit, match="scene.usda is missing"):
        run_common.isaac_scene({"scene": {"name": "room"}}, registry)
    stage = tmp_path / "isaac" / "scene" / "room_usd"
    stage.mkdir(parents=True)
    (stage / "scene.usda").write_text("#usda 1.0\n")
    assert run_common.isaac_scene({"scene": {"name": "room"}}, registry) == {
        "dir": str(stage),
        "rtx_settings": ["/rtx/a=1"],
        "supersample": 1,
    }
    for bad in (0, 1.5, True):
        registry.write_text(yaml.safe_dump({"room": {"isaac": {"dir": "room_usd", "supersample": bad}}}))
        with pytest.raises(SystemExit, match="supersample"):
            run_common.isaac_scene({"scene": {"name": "room"}}, registry)


def test_the_warehouse_isaac_entry_is_the_look_its_renders_were_calibrated_with():
    # 2026-09-29: matched to the Blender Cycles stills -- ISO 80 (was 30 against MuJoCo's robot torso), Blender's world
    # light on (stage warehouse_textured_env), indirect diffuse x2 and the Hejl-Hable tonemap for the bounce-lit
    # ceiling, docs/ledger/blender_scenes.md sections 4 and 6
    entry = yaml.safe_load(run_common.SCENES.read_text())["warehouse_textured"]["isaac"]
    assert entry["dir"] == "warehouse_textured_env"
    assert entry["rtx_settings"] == [
        "/rtx/post/histogram/enabled=false",
        "/rtx/post/tonemap/filmIso=80",
        "/rtx/indirectDiffuse/scalingFactor=2.0",
        "/rtx/post/tonemap/op=4",
    ]
    # 2x supersampling over FXAA, docs/ledger/blender_scenes.md section 5 (user 2026-09-29)
    assert entry["supersample"] == 2


def test_the_four_edge_warehouse_scenes_render_with_the_warehouse_look():
    # the same room with a resized carton: one look, and a stage built with Blender's world light on (label *_env; the
    # 2026-09-28 stages without it stay for the runs that rendered with them)
    scenes = yaml.safe_load(run_common.SCENES.read_text())
    look = {k: scenes["warehouse_textured"]["isaac"][k] for k in ("rtx_settings", "supersample")}
    boxes = [name for name in scenes if name.startswith("warehouse_box_")]
    assert len(boxes) == 8  # four carton sizes, on the stock and on the mars robot
    for name in boxes:
        entry = scenes[name]["isaac"]
        assert {k: entry[k] for k in look} == look, name
        assert entry["dir"] == f"{name}_env"


def _rounds(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _traces(directory: Path, *labels: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for label in labels:
        (directory / f"{label}_state_trace.npz").write_bytes(b"")


def test_only_successful_collection_rounds_are_picked_for_the_isaac_rerender(tmp_path):
    # two batches; a round interrupted mid-batch has its manifest renamed and its trace left behind
    _rounds(
        tmp_path / "manifest_00.jsonl",
        [{"label": "visual_crate_s1_00", "ok": True}, {"label": "visual_crate_s1_01", "ok": False}],
    )
    _rounds(tmp_path / "manifest_01.jsonl", [{"label": "visual_crate_s2_00", "ok": True}])
    _rounds(tmp_path / "interrupted_manifest_02_x.jsonl", [{"label": "visual_crate_s3_00", "ok": True}])
    _traces(tmp_path / "trace", "visual_crate_s1_00", "visual_crate_s1_01", "visual_crate_s2_00", "visual_crate_s3_00")
    assert run_common.successful_traces(tmp_path, tmp_path / "trace") == ["visual_crate_s1_00", "visual_crate_s2_00"]


def test_an_evaluation_is_judged_by_the_manifest_beside_its_traces(tmp_path):
    # a moved-aside evaluation keeps its own manifest; the collection manifests do not judge crate_vla_* rounds
    _rounds(tmp_path / "manifest_00.jsonl", [{"label": "visual_crate_s1_00", "ok": True}])
    ev = tmp_path / "eval_smoke10"
    _rounds(
        ev / "closed_loop.manifest.jsonl",
        [{"round": 0, "video": "crate_vla_000", "ok": True}, {"round": 1, "video": "crate_vla_001", "ok": False}],
    )
    _traces(ev / "state_trace", "crate_vla_000", "crate_vla_001")
    assert run_common.successful_traces(tmp_path, ev / "state_trace") == ["crate_vla_000"]


def test_single_arm_manifests_under_collect_are_read_too(tmp_path):
    manifest = tmp_path / "collect" / "manifest_00.jsonl"
    _rounds(manifest, [{"label": "bottle_00", "ok": True}, {"label": "bottle_01", "ok": False}])
    _traces(tmp_path / "collect" / "state_trace", "bottle_00", "bottle_01")
    assert run_common.successful_traces(tmp_path, tmp_path / "collect" / "state_trace") == ["bottle_00"]
