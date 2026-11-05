"""The two task families of a logistics_room_1030 instance (scripts/rsi/gso_family.py, 2026-11-03): one `held` +
`lifted` row per object of each kind from the scene file, the carton among the two-hand tasks, every row placed on
the reach band; and -- when instance s0 is built under outputs/ -- the written families load and draw starts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "rsi"))

import family as F  # noqa: E402
import gso_family as GF  # noqa: E402

META = {
    "room": {"rooms": {"logistics_room": {"bounds": {"x": [-4, 4], "y": [-3.5, 3.5]}}}},
    "gso": {"seed": 7, "reach_depth_m": 0.35},
    "objects": {
        "nor_cal_box": {"free": True, "semantic": {"phrase": "open cardboard box", "graspable": "bimanual"}},
        "gso_0001": {"free": True, "semantic": {"phrase": "red mug", "graspable": "single"}},
        "gso_0002": {"free": True, "semantic": {"phrase": "big shoe box", "graspable": "bimanual"}},
        "gso_0003": {"free": True, "semantic": {"phrase": "flat coin", "graspable": "none"}},
        "platform": {"free": False},
    },
}
S0 = ROOT / "outputs/scene_discovery/logistics_room_1030_s0"


def test_one_row_per_object_of_the_kind_with_the_goal_prompts_and_the_reach_band():
    single = GF.rows_for(META, "single", 0.35)
    both = GF.rows_for(META, "bimanual", 0.35)
    assert [r["id"] for r in single] == ["pick_gso_0001"]
    assert [r["id"] for r in both] == ["lift_nor_cal_box", "lift_gso_0002"]
    assert single[0]["goal"] == ["held(gso_0001)", "lifted(gso_0001, 0.10, 60)"]
    assert both[1]["goal"] == ["held(gso_0002)", "lifted(gso_0002, 0.20, 60)"]
    assert "red mug" in single[0]["prompts"][0] and "both hands" in both[1]["prompts"][0]
    for r in single + both:
        assert r["room"] == "logistics_room" and r["placement"] == {"depth_m": [0.0, 0.35]}


def test_the_writer_puts_both_families_and_the_tasks_file_in_the_given_directory(tmp_path):
    scene = tmp_path / "logistics_room_1030_s7"
    scene.mkdir()
    (scene / "scene_meta.json").write_text(json.dumps(META))
    out = GF.write(scene, tmp_path / "families")
    assert out == {"gso_grasp_s7": 1, "gso_bimanual_s7": 2}
    grasp = json.loads((tmp_path / "families/gso_grasp_s7.json").read_text())
    both = json.loads((tmp_path / "families/gso_bimanual_s7.json").read_text())
    rows = [json.loads(x) for x in (tmp_path / "families/gso_s7_tasks.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["pick_gso_0001", "lift_nor_cal_box", "lift_gso_0002"]
    assert grasp["tasks"] == ["pick_gso_0001"] and both["tasks"] == ["lift_nor_cal_box", "lift_gso_0002"]
    assert grasp["tasks_file"] == both["tasks_file"] == "scripts/rsi/families/gso_s7_tasks.jsonl"
    assert "hold_s" not in grasp and (both["hold_s"], both["judge_s"]) == (2.0, 5.0)
    seeds = [p["seed"] for fam in (grasp, both) for p in fam["pools"].values()]
    assert len(set(seeds)) == 6 and {p["n"] for p in grasp["pools"].values()} == {30, 40, 50}
    for fam in (grasp, both):
        assert (ROOT / fam["plan"]).is_file() and (ROOT / fam["sandbox"]["seed"]).is_dir()
        assert fam["tasks_file"] in fam["sandbox"]["keep"] and fam["plan"] in fam["sandbox"]["keep"]
        assert f"scripts/rsi/families/{fam['name']}.json" in fam["sandbox"]["keep"]
        assert "red mug" in fam["about"] or "shoe box" in fam["about"]
    assert grasp["sandbox"]["seed"] == "scripts/rsi/seeds/atomic"
    assert both["sandbox"]["seed"] == "scripts/rsi/seeds/box_strict_start"


def test_a_scene_without_the_gso_block_or_a_kind_is_refused(tmp_path):
    scene = tmp_path / "plain"
    scene.mkdir()
    (scene / "scene_meta.json").write_text(json.dumps({k: v for k, v in META.items() if k != "gso"}))
    with pytest.raises(SystemExit, match="not a GSO scene"):
        GF.write(scene, tmp_path / "families")
    meta = json.loads(json.dumps(META))
    meta["objects"]["gso_0001"]["semantic"]["graspable"] = "none"
    (scene / "scene_meta.json").write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="no single object"):
        GF.write(scene, tmp_path / "families")


@pytest.mark.skipif(not (S0 / "scene_meta.json").exists(), reason="instance s0 is not built")
def test_the_tracked_s0_families_match_the_built_scene_and_draw_starts_on_the_reach_band():
    from locomani.primitives.navigator import build
    from locomani.tasks.scene.start import edge_depths, sample_start
    from locomani.tasks.scene.world import BODY_RADIUS_M

    meta = json.loads((S0 / "scene_meta.json").read_text())
    for name in ("gso_grasp_s0", "gso_bimanual_s0"):
        fam = F.load(name)
        assert fam["scene"] == "outputs/scene_discovery/logistics_room_1030_s0"
        kind = "single" if name.startswith("gso_grasp") else "bimanual"
        keys = [k for k, o in meta["objects"].items() if o.get("free") and o["semantic"]["graspable"] == kind]
        assert [t.split("_", 1)[1] for t in fam["tasks"]] == keys
        occ, nav = build(fam["meta"], body_radius=BODY_RADIUS_M)
        surf = fam["meta"]["surfaces"]["A"]
        for i, t in enumerate(fam["tasks"][:2]):
            task = F.task(fam, t)
            st = sample_start(fam["meta"], occ, nav, list(task.objects), np.random.default_rng(i), seed=i,
                              spawn=fam["spawn"], placement=fam["rows"][t]["placement"])  # fmt: skip
            key = t.split("_", 1)[1]
            p = st.placements[key][0]
            assert min(edge_depths(surf, p[:2]).values()) <= 0.35 + 1e-6
            assert np.hypot(st.robot_xy[0] - p[0], st.robot_xy[1] - p[1]) >= 0.5
