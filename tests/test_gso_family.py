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
NAMES = ("gso_grasp_s0", "gso_bimanual_s0", "gso_pack_s0", "gso_pack_bimanual_s0")  # gso_all_s0 is their union
S0 = ROOT / "outputs/scene_discovery/logistics_room_1030_s0"


def test_one_row_per_object_of_the_kind_with_the_goal_prompts_and_the_reach_band():
    single = GF.rows_for(META, "single", 0.35)
    both = GF.rows_for(META, "bimanual", 0.35)
    assert [r["id"] for r in single] == ["pick_gso_0001"]
    assert [r["id"] for r in both] == ["lift_nor_cal_box", "lift_gso_0002"]
    assert single[0]["goal"] == ["held(gso_0001)", "lifted(gso_0001, 0.10, 60)"]
    assert both[1]["goal"] == ["held(gso_0002)", "lifted(gso_0002, 0.20, 60)"]
    assert "red mug" in single[0]["prompts"][0] and "both hands" in both[1]["prompts"][0]
    for r in single:
        assert r["room"] == "logistics_room" and r["placement"] == {"depth_m": [0.0, 0.35], "shuffle_others": True}
    for r in both:  # a hugged object keeps its hands' corridors free (user 2026-11-06)
        assert r["room"] == "logistics_room" and r["placement"] == {"depth_m": [0.0, 0.35], "shuffle_others": True,
                                                                    "hands_clear_m": GF.HANDS_CLEAR_M}  # fmt: skip


def test_a_pack_row_per_object_but_the_carton_puts_the_carton_on_its_band_and_a_two_hand_object_along_its_edge():
    pack = GF.rows_for(META, "pack", 0.35)
    both = GF.rows_for(META, "pack_bimanual", 0.35)
    assert [r["id"] for r in pack] == ["pack_gso_0001"] and [r["id"] for r in both] == ["pack_gso_0002"]
    assert pack[0]["goal"] == ["in(gso_0001, nor_cal_box)", *PACK_GOAL]
    assert both[0]["goal"] == ["in(gso_0002, nor_cal_box)", *PACK_GOAL]
    assert "into the cardboard box" in pack[0]["prompts"][0] and "both hands" in both[0]["prompts"][0]
    carton = {"depth_m": [0.18, 0.31], "yaw_deg": [85, 90], "hands_clear_m": GF.HANDS_CLEAR_M}
    assert pack[0]["placement"] == {"depth_m": [0.0, 0.35], "shuffle_others": True, "objects": {"nor_cal_box": carton}}
    assert both[0]["placement"] == {"depth_m": [0.0, 0.35], "shuffle_others": True, "objects": {
        "nor_cal_box": carton, "gso_0002": {"depth_m": [0.0, 0.35], "yaw_deg": [0, 5], "hands_clear_m": GF.HANDS_CLEAR_M}}}  # fmt: skip
    meta = json.loads(json.dumps(META))
    meta["objects"]["gso_0002"]["box"]["half_extent"] = [0.17, 0.1, 0.06]  # the wide side is x: x along the edge
    del meta["gso"]["carton_placement"]  # an instance built before the band was recorded: the same band
    row = GF.rows_for(meta, "pack_bimanual", 0.35)[0]
    hug = {"hands_clear_m": GF.HANDS_CLEAR_M}
    assert row["placement"]["objects"] == {"nor_cal_box": {**GF.CARTON_BAND, **hug},
                                           "gso_0002": {"depth_m": [0.0, 0.35], "yaw_deg": [85, 90], **hug}}  # fmt: skip
    sys.path.insert(0, str(ROOT / "scripts"))
    import prepare_logistics_scene as P

    assert GF.CARTON_BAND == P.CARTON_PLACEMENT


def test_the_writer_puts_both_families_and_the_tasks_file_in_the_given_directory(tmp_path):
    scene = tmp_path / "logistics_room_1030_s7"
    scene.mkdir()
    (scene / "scene_meta.json").write_text(json.dumps(META))
    out = GF.write(scene, tmp_path / "families")
    assert out == {"gso_grasp_s7": 1, "gso_bimanual_s7": 2, "gso_pack_s7": 1, "gso_pack_bimanual_s7": 1,
                   "gso_all_s7": 5, "gso_pack_all_s7": 2}  # fmt: skip
    every = json.loads((tmp_path / "families/gso_all_s7.json").read_text())
    packs = json.loads((tmp_path / "families/gso_pack_all_s7.json").read_text())
    fams = {k: json.loads((tmp_path / f"families/{k}.json").read_text()) for k in out
            if k not in ("gso_all_s7", "gso_pack_all_s7")}  # fmt: skip
    grasp, both, pack, pack2 = fams.values()
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
    assert len(set(seeds)) == 12 and {k: p["n"] for k, p in grasp["pools"].items()} == {
        "diag": 3,
        "accept": 31,
        "sealed": 10,
    }
    assert grasp["pool_per_task"] == GF.PER_TASK == {"diag": 3, "accept": 10, "sealed": 10}  # user 2026-11-24
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
    # gso_all (user 2026-11-06): every row of the four, one rotation, PER_TASK starts per task, timing per row
    assert every["tasks"] == [r["id"] for r in rows] and every["tasks_file"] == grasp["tasks_file"]
    assert {k: p["n"] for k, p in every["pools"].items()} == {"diag": 15, "accept": 50, "sealed": 50}
    assert not set(p["seed"] for p in every["pools"].values()) & set(seeds)
    assert "hold_s" not in every and every["sandbox"]["seed"] is None and (ROOT / every["plan"]).is_file()
    assert all(p in every["sandbox"]["keep"] for p in set(GF.PLANS.values()))
    assert (every["stop_rate"], every["max_wall_h"], every["max_run_iters"]) == (0.95, 20.0, 5)  # user 2026-11-24
    by_id = {r["id"]: r for r in rows}
    assert "hold_s" not in by_id["pick_gso_0001"]
    assert all((by_id[t]["hold_s"], by_id[t]["judge_s"]) == (2.0, 5.0) for t in every["tasks"][1:])
    fam = dict(every, rows=by_id)
    assert F.timing(fam, "pick_gso_0001") == {} and F.timing(fam, "pack_gso_0002") == GF.TIMING
    assert F.timing(dict(grasp, rows=by_id, judge_s=3.0), "pick_gso_0001") == {"judge_s": 3.0}  # the family's
    # gso_pack_all (user 2026-11-07): the two pack kinds' rows, PER_TASK starts per task, the pack timing, its own plan
    assert packs["tasks"] == pack["tasks"] + pack2["tasks"] and packs["tasks_file"] == grasp["tasks_file"]
    assert {k: p["n"] for k, p in packs["pools"].items()} == {"diag": 6, "accept": 31, "sealed": 20}
    assert [p["seed"] for p in packs["pools"].values()] == [490_000, 590_000, 690_000]
    assert not set(p["seed"] for p in packs["pools"].values()) & (
        set(seeds) | {p["seed"] for p in every["pools"].values()}
    )
    assert (packs["hold_s"], packs["judge_s"]) == (2.0, 5.0) and packs["sandbox"]["seed"] is None
    assert packs["plan"] == "scripts/rsi/plans/gso_pack_all.md" and (ROOT / packs["plan"]).is_file()
    assert (
        packs["plan"] in packs["sandbox"]["keep"]
        and "scripts/rsi/families/gso_pack_all_s7.json" in packs["sandbox"]["keep"]
    )
    assert (packs["stop_rate"], packs["max_wall_h"], packs["max_run_iters"]) == (0.95, 20.0, 5)
    assert "progress stage" in packs["about"] and "red mug" in packs["about"]


def test_every_tracked_gso_all_family_is_the_four_families_of_its_instance():
    for seed in range(10):
        fams = [json.loads((ROOT / f"scripts/rsi/families/{n.replace('_s0', f'_s{seed}')}.json").read_text())
                for n in NAMES]  # fmt: skip
        every = json.loads((ROOT / f"scripts/rsi/families/gso_all_s{seed}.json").read_text())
        assert every["tasks"] == [t for f in fams for t in f["tasks"]]
        assert {k: p["n"] for k, p in every["pools"].items()} == {
            k: n * len(every["tasks"]) for k, n in GF.PER_TASK.items()
        }
        packs = json.loads((ROOT / f"scripts/rsi/families/gso_pack_all_s{seed}.json").read_text())
        assert packs["tasks"] == [t for f in fams[2:] for t in f["tasks"]] and len(packs["tasks"]) == 10
        assert {k: p["n"] for k, p in packs["pools"].items()} == {"diag": 30, "accept": 100, "sealed": 100}
        for f in fams:  # every drawn family: PER_TASK by its task count, the acceptance floor, the 11-24 stops
            assert {k: p["n"] for k, p in f["pools"].items()} == {
                "diag": 3 * len(f["tasks"]), "accept": max(10 * len(f["tasks"]), 31), "sealed": 10 * len(f["tasks"])
            }  # fmt: skip
            assert (f["stop_rate"], f["max_wall_h"], f["max_run_iters"]) == (0.95, 20.0, 5)


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


def test_a_hand_picked_scene_gets_one_family_lifted_and_judged_over_the_last_5_s(tmp_path):
    """User 2026-11-17: logistics_room_gso_single / _bimanual each get one family, every object lifted PICKED_LIFT_M
    (0.20 m since 2026-11-18; --lift another height) and held, judged over the last 5 s for one hand too, from the main
    skill library."""
    for which, name, ids in (("single", "gso_single_grasp", ["pick_gso_0001"]),
                             ("bimanual", "gso_bimanual_lift", ["lift_nor_cal_box", "lift_gso_0002"])):  # fmt: skip
        meta = json.loads(json.dumps(META))
        meta["gso"].update(picks=which, scene=f"logistics_room_gso_{which}", seed=3)
        scene = tmp_path / which
        scene.mkdir()
        (scene / "scene_meta.json").write_text(json.dumps(meta))
        assert GF.write(scene, tmp_path / "families") == {name: len(ids)}
        fam = json.loads((tmp_path / f"families/{name}.json").read_text())
        rows = [json.loads(x) for x in (tmp_path / f"families/{name}_tasks.jsonl").read_text().splitlines()]
        assert fam["tasks"] == [r["id"] for r in rows] == ids
        assert GF.PICKED_LIFT_M == 0.20  # user 2026-11-18
        assert all(r["goal"][1].endswith(f", {GF.PICKED_LIFT_M:.2f}, 60)") and (r["hold_s"], r["judge_s"]) == (2.0, 5.0)
                   for r in rows)  # fmt: skip
        assert (fam["hold_s"], fam["judge_s"]) == (2.0, 5.0) and fam["sandbox"]["seed"] is None
        per = GF.PER_TASK  # pools by the task count (user 2026-11-18); acceptance at least gate.STAGE1 + 1
        assert fam["pool_per_task"] == per
        assert {k: v["n"] for k, v in fam["pools"].items()} == {
            "diag": per["diag"] * len(ids),
            "accept": max(per["accept"] * len(ids), GF.gate_stage1() + 1),
            "sealed": per["sealed"] * len(ids),
        }
        assert fam["spawn"] == {"rooms": "all", "distance_m": [0.5, 4.0], "face_target_deg": [-80, 80]}  # user 11-17
        assert fam["plan"] in fam["sandbox"]["keep"] and (ROOT / fam["plan"]).is_file()
        assert {"scripts/rsi/knowledge.py", "scripts/rsi/knowledge"} <= set(fam["sandbox"]["keep"])  # shared knowledge
        assert all(p in fam["sandbox"]["keep"] for p in (fam["tasks_file"], f"scripts/rsi/families/{name}.json"))
        assert ("hands_clear_m" in rows[0]["placement"]) == (which == "bimanual")
    assert not (tmp_path / "families/gso_s3_tasks.jsonl").exists()  # none of the drawn instance's families
    GF.write(tmp_path / "single", tmp_path / "low", lift=0.25)  # --lift: another height
    low = (tmp_path / "low/gso_single_grasp_tasks.jsonl").read_text()
    assert "lifted(gso_0001, 0.25, 60)" in low and "25 cm" in low
    with pytest.raises(SystemExit):  # a drawn instance keeps its lifts
        (tmp_path / "s7").mkdir()
        (tmp_path / "s7/scene_meta.json").write_text(json.dumps(META))
        GF.write(tmp_path / "s7", tmp_path / "families", lift=0.2)


def test_the_picked_scenes_start_the_robot_within_4_m_turned_towards_the_task_object():
    """User 2026-11-17: in logistics_room_gso_single / _bimanual the robot starts 0.5-4 m from the task object,
    turned towards it within +-80 deg, the angle drawn uniformly (start.sample_start's `face_target_deg`)."""
    from locomani.primitives.navigator import build
    from locomani.tasks.scene.start import sample_start
    from locomani.tasks.scene.world import BODY_RADIUS_M

    for name in ("gso_single_grasp", "gso_bimanual_lift"):
        if not (ROOT / f"scripts/rsi/families/{name}.json").is_file():
            pytest.skip(f"no {name} family")
        fam = json.loads((ROOT / f"scripts/rsi/families/{name}.json").read_text())
        scene = ROOT / fam["scene"]
        if not (scene / "scene_meta.json").is_file():
            pytest.skip(f"{fam['scene']} not built (docs/rsi_logistic_room_1030_single_bimanual.md section 3)")
        fam = F.load(name)
        occ, nav = build(fam["meta"], body_radius=BODY_RADIUS_M)
        got, offs = 0, []
        for t in fam["tasks"][:3]:
            task = F.task(fam, t)
            for i in range(10):
                try:
                    st = sample_start(fam["meta"], occ, nav, list(task.objects), np.random.default_rng(i), seed=i,
                                      spawn=fam["spawn"], placement=fam["rows"][t]["placement"])  # fmt: skip
                except RuntimeError:
                    continue
                p = np.asarray(st.placements[t.split("_", 1)[1]][0][:2])
                d = p - np.asarray(st.robot_xy)
                assert 0.5 <= float(np.hypot(*d)) <= 4.0
                heading = np.array([np.cos(st.robot_yaw), np.sin(st.robot_yaw)])
                off = np.degrees(np.arccos(np.clip(float(heading @ d) / float(np.hypot(*d)), -1.0, 1.0)))
                assert off <= 80.0 + 1e-6
                offs.append(off)
                got += 1
        assert got and max(offs) > 20.0  # a spread, not always straight at it


def test_a_picked_familys_starts_per_task_come_from_the_command_line_not_the_code(tmp_path):
    """User 2026-11-18: the diagnostic, acceptance and sealed pools are sized by --per-task (default PER_TASK),
    recorded in the family file; the acceptance floor is gate.py's STAGE1 + 1, read from gate.py."""
    import re

    stage1 = int(re.search(r"^STAGE1 = (\d+)", (ROOT / "scripts/rsi/gate.py").read_text(), re.M).group(1))
    assert GF.gate_stage1() == stage1
    meta = json.loads(json.dumps(META))
    meta["gso"].update(picks="single", scene="logistics_room_gso_single", seed=3)
    scene = tmp_path / "single"
    scene.mkdir()
    (scene / "scene_meta.json").write_text(json.dumps(meta))
    per = GF.parse_per_task(["diag=7", "sealed=2"])
    assert per == dict(GF.PER_TASK, diag=7, sealed=2)
    GF.write(scene, tmp_path / "families", per_task=per)
    fam = json.loads((tmp_path / "families/gso_single_grasp.json").read_text())
    n = len(fam["tasks"])
    assert fam["pool_per_task"] == per
    assert {k: v["n"] for k, v in fam["pools"].items()} == {
        "diag": 7 * n, "accept": max(per["accept"] * n, stage1 + 1), "sealed": 2 * n,
    }  # fmt: skip
    assert GF.main([str(scene), "--families-dir", str(tmp_path / "cli"), "--per-task", "accept=40"]) == 0
    cli = json.loads((tmp_path / "cli/gso_single_grasp.json").read_text())
    assert cli["pools"]["accept"]["n"] == 40 * n
    for bad in (["test=3"], ["diag=0"], ["diag=x"]):
        with pytest.raises(SystemExit):
            GF.parse_per_task(bad)
    drawn = tmp_path / "s7"
    drawn.mkdir()
    (drawn / "scene_meta.json").write_text(json.dumps(META))
    GF.write(drawn, tmp_path / "families", per_task=per)  # a drawn instance too (user 2026-11-24: every family)
    grasp = json.loads((tmp_path / "families/gso_grasp_s7.json").read_text())
    assert grasp["pool_per_task"] == per and {k: v["n"] for k, v in grasp["pools"].items()} == {
        "diag": 7, "accept": stage1 + 1, "sealed": 2,
    }  # fmt: skip
    with pytest.raises(SystemExit):  # the drawn instances keep LIFT_M
        GF.write(drawn, tmp_path / "families", lift=0.3)
