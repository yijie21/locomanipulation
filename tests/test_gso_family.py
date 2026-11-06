"""The four task families of a logistics_room_1030 instance (scripts/rsi/gso_family.py, 2026-11-03; the pack line
2026-11-05): one `held` + `lifted` row per object of each kind from the scene file, the carton among the two-hand
tasks, one pack row (`in` + the carton's `held` + `lifted`) per object but the carton, every row placed on the reach
band and the pack rows' carton on the hug's band; and -- when instance s0 is built under outputs/ -- the written
families load and draw starts."""

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
    "gso": {"seed": 7, "reach_depth_m": 0.35, "carton_placement": {"depth_m": [0.18, 0.31], "yaw_deg": [85, 90]}},
    "objects": {
        "nor_cal_box": {"free": True, "semantic": {"phrase": "open cardboard box", "graspable": "bimanual"},
                        "box": {"half_extent": [0.225, 0.16, 0.165]}},
        "gso_0001": {"free": True, "semantic": {"phrase": "red mug", "graspable": "single"},
                     "box": {"half_extent": [0.04, 0.04, 0.05]}},
        "gso_0002": {"free": True, "semantic": {"phrase": "big shoe box", "graspable": "bimanual"},
                     "box": {"half_extent": [0.1, 0.17, 0.06]}},  # the wide side is y
        "gso_0003": {"free": True, "semantic": {"phrase": "flat coin", "graspable": "none"}},
        "platform": {"free": False},
    },
}  # fmt: skip
PACK_GOAL = ["held(nor_cal_box)", "lifted(nor_cal_box, 0.20, 60)"]
NAMES = ("gso_grasp_s0", "gso_bimanual_s0", "gso_pack_s0", "gso_pack_bimanual_s0")
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


def test_a_pack_row_per_object_but_the_carton_puts_the_carton_on_its_band_and_a_two_hand_object_along_its_edge():
    pack = GF.rows_for(META, "pack", 0.35)
    both = GF.rows_for(META, "pack_bimanual", 0.35)
    assert [r["id"] for r in pack] == ["pack_gso_0001"] and [r["id"] for r in both] == ["pack_gso_0002"]
    assert pack[0]["goal"] == ["in(gso_0001, nor_cal_box)", *PACK_GOAL]
    assert both[0]["goal"] == ["in(gso_0002, nor_cal_box)", *PACK_GOAL]
    assert "into the cardboard box" in pack[0]["prompts"][0] and "both hands" in both[0]["prompts"][0]
    carton = {"depth_m": [0.18, 0.31], "yaw_deg": [85, 90]}
    assert pack[0]["placement"] == {"depth_m": [0.0, 0.35], "objects": {"nor_cal_box": carton}}
    assert both[0]["placement"] == {"depth_m": [0.0, 0.35], "objects": {
        "nor_cal_box": carton, "gso_0002": {"depth_m": [0.0, 0.35], "yaw_deg": [0, 5]}}}  # fmt: skip
    meta = json.loads(json.dumps(META))
    meta["objects"]["gso_0002"]["box"]["half_extent"] = [0.17, 0.1, 0.06]  # the wide side is x: x along the edge
    del meta["gso"]["carton_placement"]  # an instance built before the band was recorded: the same band
    row = GF.rows_for(meta, "pack_bimanual", 0.35)[0]
    assert row["placement"]["objects"] == {"nor_cal_box": GF.CARTON_BAND, "gso_0002": {"depth_m": [0.0, 0.35],
                                                                                         "yaw_deg": [85, 90]}}  # fmt: skip
    sys.path.insert(0, str(ROOT / "scripts"))
    import prepare_logistics_scene as P

    assert GF.CARTON_BAND == P.CARTON_PLACEMENT


def test_the_writer_puts_both_families_and_the_tasks_file_in_the_given_directory(tmp_path):
    scene = tmp_path / "logistics_room_1030_s7"
    scene.mkdir()
    (scene / "scene_meta.json").write_text(json.dumps(META))
    out = GF.write(scene, tmp_path / "families")
    assert out == {"gso_grasp_s7": 1, "gso_bimanual_s7": 2, "gso_pack_s7": 1, "gso_pack_bimanual_s7": 1}
    fams = {k: json.loads((tmp_path / f"families/{k}.json").read_text()) for k in out}
    grasp, both, pack, pack2 = (fams[k] for k in out)
    rows = [json.loads(x) for x in (tmp_path / "families/gso_s7_tasks.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["pick_gso_0001", "lift_nor_cal_box", "lift_gso_0002", "pack_gso_0001",
                                       "pack_gso_0002"]  # fmt: skip
    assert grasp["tasks"] == ["pick_gso_0001"] and both["tasks"] == ["lift_nor_cal_box", "lift_gso_0002"]
    assert pack["tasks"] == ["pack_gso_0001"] and pack2["tasks"] == ["pack_gso_0002"]
    assert len({f["tasks_file"] for f in fams.values()}) == 1 == len({f["scene"] for f in fams.values()})
    assert grasp["tasks_file"] == "scripts/rsi/families/gso_s7_tasks.jsonl"
    assert "hold_s" not in grasp
    assert all((f["hold_s"], f["judge_s"]) == (2.0, 5.0) for f in (both, pack, pack2))
    seeds = [p["seed"] for fam in fams.values() for p in fam["pools"].values()]
    assert len(set(seeds)) == 12 and {p["n"] for p in grasp["pools"].values()} == {30, 40, 50}
    assert pack["pools"]["diag"]["seed"] == 460_000 and pack2["pools"]["sealed"]["seed"] == 670_000
    for fam in fams.values():
        assert (ROOT / fam["plan"]).is_file()
        assert fam["tasks_file"] in fam["sandbox"]["keep"] and fam["plan"] in fam["sandbox"]["keep"]
        assert f"scripts/rsi/families/{fam['name']}.json" in fam["sandbox"]["keep"]
        assert "red mug" in fam["about"] or "shoe box" in fam["about"]
    assert grasp["sandbox"]["seed"] == "scripts/rsi/seeds/atomic" and (ROOT / grasp["sandbox"]["seed"]).is_dir()
    assert all(f["sandbox"]["seed"] is None for f in (both, pack, pack2))  # the main library as it stands
    assert pack["plan"] == pack2["plan"] == "scripts/rsi/plans/gso_pack.md"
    assert "fits the carton" in pack["about"] and "wide side along it" in pack2["about"]


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
    meta = json.loads(json.dumps(META))
    del meta["objects"]["gso_0002"]  # the carton alone is a two-hand object: nothing to pack with both hands
    (scene / "scene_meta.json").write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="no bimanual object for the gso_pack_bimanual"):
        GF.write(scene, tmp_path / "families")


@pytest.mark.skipif(not (S0 / "scene_meta.json").exists(), reason="instance s0 is not built")
def test_the_tracked_s0_families_match_the_built_scene_and_draw_starts_on_the_reach_band():
    from locomani.primitives.navigator import build
    from locomani.tasks.scene.start import edge_depths, placement_ok, sample_start
    from locomani.tasks.scene.world import BODY_RADIUS_M

    meta = json.loads((S0 / "scene_meta.json").read_text())
    for name in NAMES:
        fam = F.load(name)
        assert fam["scene"] == "outputs/scene_discovery/logistics_room_1030_s0"
        kind = "single" if name.split("_s0")[0] in ("gso_grasp", "gso_pack") else "bimanual"
        pack = "pack" in name
        keys = [k for k, o in meta["objects"].items()
                if o.get("free") and o["semantic"]["graspable"] == kind and not (pack and k == GF.CARTON)]  # fmt: skip
        assert [t.split("_", 1)[1] for t in fam["tasks"]] == keys
        occ, nav = build(fam["meta"], body_radius=BODY_RADIUS_M)
        surf = fam["meta"]["surfaces"]["A"]
        for t in fam["tasks"][:2]:
            task = F.task(fam, t)
            # a seed whose draw cannot meet the placement raises and the pool skips it (sample_start); a big object
            # among ten (s0's 35 cm motherboard) fails about half the seeds, so take the first seed that starts
            for i in range(20):
                try:
                    st = sample_start(fam["meta"], occ, nav, list(task.objects), np.random.default_rng(i), seed=i,
                                      spawn=fam["spawn"], placement=fam["rows"][t]["placement"])  # fmt: skip
                    break
                except RuntimeError:
                    continue
            else:
                pytest.fail(f"{t}: no start in 20 seeds")
            key = t.split("_", 1)[1]
            p = st.placements[key][0]
            assert min(edge_depths(surf, p[:2]).values()) <= 0.35 + 1e-6
            assert np.hypot(st.robot_xy[0] - p[0], st.robot_xy[1] - p[1]) >= 0.5
            if pack:  # the carton moved onto its band too, the object X first (it anchors the spawn)
                assert list(task.objects) == [key, GF.CARTON]
                c, cq = st.placements[GF.CARTON]
                yaw = 2 * np.arctan2(cq[3], cq[0])
                assert placement_ok(surf, c[:2], yaw, fam["rows"][t]["placement"]["objects"][GF.CARTON])
