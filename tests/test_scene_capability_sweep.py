"""The scene capability sweep's pieces: layout ranges at reset, the stop rule, the cell queue, the layout scene."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from locomani.pipeline import batches
from locomani.tasks.crate import sweep as sweep_module
from locomani.tasks.crate.visual_config import CrateVisualConfig

META = {  # the warehouse's layout: walk-in +y, near edge at y = -0.555
    "robot_start": {"xy": [0.0, -1.205], "yaw_deg": 90.0},
    "target_object": "nor_cal_box",
    "objects": {"nor_cal_box": {"surface": "A", "init_quat": [1, 0, 0, 0]}},
    "surfaces": {"A": {"xy": [0.0, -0.105], "flat_half_extent": [0.6, 0.45]}},
}


class RecordingReset:
    def __init__(self):
        self.calls = []

    def reset(self, **kwargs):
        self.calls.append(kwargs)


def _place(cfg, seed=3):
    from locomani.tasks.crate.expert import start_placement

    reset, rng = RecordingReset(), np.random.default_rng(seed)
    placement, record = start_placement(reset, META, cfg, rng)
    placement("initial_placement")
    return reset.calls[0], record, rng


def test_closed_ranges_place_exactly_as_before():
    cfg = CrateVisualConfig()
    call, record, rng = _place(cfg)
    reference = np.random.default_rng(3)
    jitter = reference.uniform(-cfg.position_jitter, cfg.position_jitter, size=2)
    yaw = reference.uniform(-cfg.yaw_jitter_deg, cfg.yaw_jitter_deg)
    distance = reference.uniform(cfg.start_edge_min, cfg.start_edge_max)
    np.testing.assert_allclose(call["objects"]["nor_cal_box"]["dxy"], jitter)
    assert call["objects"]["nor_cal_box"]["yaw"] == pytest.approx(np.radians(yaw))
    assert call["robot"]["yaw"] == pytest.approx(np.pi / 2)
    assert record["start_edge_distance"] == pytest.approx(distance)
    assert rng.random() == reference.random()  # no extra draws: the rest of the episode sees the same stream
    assert (record["start_lateral_m"], record["object_setback_m"]) == (0.0, 0.0)


def test_open_ranges_move_the_robot_across_and_the_box_along_the_walk_in():
    cfg = replace(
        CrateVisualConfig(position_jitter=0.0, yaw_jitter_deg=0.0, start_edge_min=1.0, start_edge_max=1.0),
        start_yaw_range_deg=(20.0, 20.0),
        start_lateral_range=(0.1, 0.1),
        object_setback_range=(0.05, 0.05),
        object_yaw_range_deg=(-10.0, -10.0),
    )
    call, record, _ = _place(cfg)
    # walk-in +y: left of it is -x; the start sits 1.0 m before the edge at y = -0.555
    np.testing.assert_allclose(record["start_xy"], [-0.1, -1.555], atol=1e-9)
    assert call["robot"]["yaw"] == pytest.approx(np.radians(110))
    np.testing.assert_allclose(call["objects"]["nor_cal_box"]["dxy"], [0.0, 0.05], atol=1e-9)
    assert call["objects"]["nor_cal_box"]["yaw"] == pytest.approx(np.radians(-10))
    assert record["start_yaw_offset_deg"] == 20.0 and record["object_yaw_offset_deg"] == -10.0


def test_range_fields_are_validated():
    with pytest.raises(ValueError, match="low <= high"):
        CrateVisualConfig(start_lateral_range=(0.2, -0.2))
    with pytest.raises(ValueError, match="near edge"):
        CrateVisualConfig(object_setback_range=(-0.05, 0.1))
    CrateVisualConfig(start_edge_min=0.6, start_edge_max=2.0)  # the sweep's start range
    with pytest.raises(ValueError, match="start_edge"):
        CrateVisualConfig(start_edge_min=0.3)


RULE = {"threshold": 0.5, "min_rounds": 10, "max_rounds": 30, "min_rtf": 0.9}


@pytest.mark.parametrize(
    ("k", "n", "expected"),
    [(1, 10, "below"), (10, 10, "above"), (5, 10, None), (15, 30, "undecided"), (0, 9, None), (9, 12, None)],
)
def test_stop_verdict(k, n, expected):
    assert batches.stop_verdict(k, n, RULE) == expected


def test_stop_tally_leaves_out_rounds_that_measured_the_stack(tmp_path):
    rows = [
        dict(ok=True, why="completed", real_time_factor=0.95),
        dict(ok=False, why="reach_stalled", real_time_factor=0.96),
        dict(ok=False, why="initial_stance_unstable"),
        dict(ok=False, why="basket_tilt", control_reason="runtime_error", real_time_factor=0.95),
        dict(ok=True, why="completed", real_time_factor=0.6),
    ]
    manifest = tmp_path / "manifest_00.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    from locomani.tasks.crate.collect import stop_tally

    assert stop_tally([str(manifest)], RULE) == dict(successes=1, rounds=2, excluded=3)


SWEEP = {
    "name": "t",
    "base": "configs/experiments/box_room_grasp_200_cap120_20260925.yaml",
    "platform_top": 0.10,
    "stand_forward_clearance": 0.19,
    "set": {"collect.record": False, "collect.stop_rule": RULE, "task.visual.retreat_distance": 0.0},
    "centre": {
        "width": 0.45,
        "length": 0.32,
        "height": 0.45,
        "start_edge": [1.2, 1.2],
        "start_yaw_deg": [0, 0],
        "start_lateral": [0, 0],
        "object_setback": [0, 0],
        "object_yaw_deg": [0, 0],
    },
    "grid": {"height": [0.3, 0.6]},
    "chains": {"start_yaw_deg": [[[0, 10], [10, 20]]]},
}


def test_cells_expand_the_grid_and_chain_outward():
    cells = sweep_module.cells(SWEEP)
    assert [c["id"] for c in cells] == ["height0.3", "height0.6", "start_yaw_deg+0_+10", "start_yaw_deg+10_+20"]
    assert cells[3]["after"] == "start_yaw_deg+0_+10" and cells[3]["values"]["start_yaw_deg"] == [10, 20]
    assert cells[3]["values"]["height"] == 0.45


@pytest.mark.skipif(
    not (sweep_module.ROOT / SWEEP["base"]).exists(), reason="the base experiment is not in this checkout"
)
def test_cell_experiment_carries_the_layout():
    cell = sweep_module.cells(SWEEP)[3]
    cfg = sweep_module.experiment(SWEEP, cell, instance=2)
    visual = cfg["task"]["visual"]
    assert visual["start_yaw_range_deg"] == [10.0, 20.0] and visual["start_edge_min"] == 1.2
    assert visual["retreat_distance"] == 0.0 and visual["stand_forward_min"] == pytest.approx(0.35)
    assert cfg["collect"]["instance"] == 2 and cfg["collect"]["stop_rule"] == RULE
    assert cfg["scene"]["dir"].endswith("scenes/W0.45_L0.32_H0.45")


def test_a_below_level_prunes_the_rest_of_its_chain(monkeypatch, tmp_path):
    monkeypatch.setattr(sweep_module, "ROOT", tmp_path)
    finished = {"height0.3": "above", "start_yaw_deg+0_+10": "below"}
    monkeypatch.setattr(
        sweep_module, "verdict", lambda _s, cell: {"verdict": finished[cell["id"]]} if cell["id"] in finished else None
    )
    state = sweep_module.status(SWEEP, sweep_module.cells(SWEEP))
    assert state == {
        "height0.3": "done:above",
        "height0.6": "pending",
        "start_yaw_deg+0_+10": "done:below",
        "start_yaw_deg+10_+20": "pruned",
    }


def test_layout_scene_puts_the_box_front_on_the_near_edge(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    from locomani.tasks.crate import box_platform_scene

    if not (box_platform_scene.SOURCE / "scene_meta.json").exists():
        pytest.skip("warehouse_textured scene not built on this machine")
    meta = box_platform_scene.build(tmp_path / "scene", width=0.38, length=0.44, height=0.6)
    model = mujoco.MjModel.from_xml_path(str(Path(meta["scene_xml"])))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    front = model.geom("nor_cal_ym").id
    assert data.geom_xpos[front][1] - model.geom_size[front][1] == pytest.approx(-0.555, abs=1e-6)
    assert data.geom_xpos[model.geom("nor_cal_xp").id][0] == pytest.approx(0.19 - 0.005)
    platform = model.geom("platform_col").id
    assert data.geom_xpos[platform][2] + model.geom_size[platform][2] == pytest.approx(0.10)
    assert meta["objects"]["nor_cal_box"]["box"]["half_extent"] == [0.19, 0.22, 0.3]


def test_grasp_points_move_to_mid_wall_when_the_box_states_a_thick_wall():
    from locomani.tasks.crate.grasp_region import grasp_points_for

    thin = {"box": {"half_extent": [0.225, 0.16, 0.3], "centre": [0, 0, 0.3], "wall": 0.01}}
    thick = {"box": {"half_extent": [0.225, 0.16, 0.3], "centre": [0, 0, 0.3], "wall": 0.03}}
    assert grasp_points_for(thin, 0.01)[0][0] == pytest.approx(0.225 - 0.008)  # the calibrated inset stays
    assert grasp_points_for(thick, 0.01)[0][0] == pytest.approx(0.225 - 0.015)  # mid-wall
    plain = {"box": {"half_extent": [0.225, 0.16, 0.3], "centre": [0, 0, 0.3]}}
    assert grasp_points_for(plain, 0.01)[0][0] == pytest.approx(0.217)
