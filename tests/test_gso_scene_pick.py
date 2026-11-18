"""The hand-picked test scenes (user 2026-11-17): the picks file is checked against the catalog (each scene's kind,
the training split, the count when building), the set is placed on the platform's reach band with the carton on its
band, a set too large for the platform is refused naming what found no spot, and the picking page carries each
scene's pool and saves what the builder reads."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from locomani.sim import gso_objects as G  # noqa: E402


def _obj(name, kind, split="train", ext=(0.08, 0.08, 0.12)):
    half = [e / 2 for e in ext]
    return {"name": name, "phrase": name.lower().replace("_", " "), "category": None, "graspable": kind,
            "split": split, "mass_kg": 0.3, "extent_m": list(ext), "bbox_min_m": [-half[0], -half[1], 0.0],
            "bbox_max_m": [half[0], half[1], ext[2]], "fits_carton": True}  # fmt: skip


CATALOG = {"objects": {
    **{f"gso_{i:04d}": _obj(f"Single_{i}", "single") for i in range(22)},
    "gso_0100": _obj("Held_Out_Mug", "single", split="held_out"),
    **{f"gso_{200 + i:04d}": _obj(f"Wide_{i}", "bimanual", ext=(0.30, 0.22, 0.20)) for i in range(10)},
    "gso_0300": _obj("Huge_Tray", "bimanual", ext=(0.45, 0.44, 0.15)),
    "gso_0400": _obj("Shoe", "none"),
}}  # fmt: skip
SURFACE = {"xy": [0.0, -0.105], "top": 0.371, "half_extent": [0.6, 0.45], "flat_half_extent": [0.6, 0.45], "yaw": 0.0,
           "approach_edges": ["-x", "+x", "-y", "+y"]}  # fmt: skip
CARTON = {"nor_cal_box": {"depth_m": [0.18, 0.31], "yaw_deg": [85, 90]}}


def test_each_scene_offers_its_kind_of_the_training_split_and_the_picks_are_checked():
    assert G.pick_pool(CATALOG, "single") == [f"gso_{i:04d}" for i in range(22)]  # not the held-out mug
    assert len(G.pick_pool(CATALOG, "bimanual")) == 11
    twenty = [f"Single_{i}" for i in range(20)]
    doc = G.check_picks({"scenes": {"single": {"seed": 4, "objects": twenty}}}, CATALOG, complete=True)
    assert G.picked_keys(CATALOG, doc, "single") == [f"gso_{i:04d}" for i in range(20)]
    assert G.check_picks({"scenes": {"bimanual": {"objects": ["Wide_0"]}}}, CATALOG)["scenes"]["bimanual"]["seed"] == 0
    for bad in ({"scenes": {"single": {"objects": ["Held_Out_Mug"]}}}, {"scenes": {"single": {"objects": ["Wide_0"]}}},
                {"scenes": {"single": {"objects": ["Single_1", "Single_1"]}}}, {"scenes": {"pack": {}}},
                {"scenes": {"single": {"seed": -1, "objects": []}}}, {"scenes": []}, []):  # fmt: skip
        with pytest.raises(ValueError):
            G.check_picks(bad, CATALOG)
    for n, which in ((19, "single"), (21, "single"), (4, "bimanual"), (10, "bimanual")):  # 20 single; 5-9 two-hand
        names = [f"{'Single' if which == 'single' else 'Wide'}_{i}" for i in range(n)]
        with pytest.raises(ValueError, match="objects picked"):
            G.check_picks({"scenes": {which: {"objects": names}}}, CATALOG, complete=True)
    with pytest.raises(ValueError, match="no objects picked"):
        G.picked_keys(CATALOG, doc, "bimanual")


def test_a_picked_set_is_placed_on_the_reach_band_or_refused_naming_what_found_no_spot():
    from locomani.tasks.scene.start import edge_depths

    boxes = {k: G.object_box(CATALOG["objects"][k])["half_extent"] for k in G.pick_pool(CATALOG, "bimanual")[:5]}
    boxes["nor_cal_box"] = [0.225, 0.16, 0.165]
    placed, attempt = G.place_picked(boxes, SURFACE, 0, 0.35, constraints=CARTON)
    assert set(placed) == set(boxes) and attempt >= 0
    assert all(min(edge_depths(SURFACE, p["xy"]).values()) <= 0.35 + 1e-9 for p in placed.values())
    again, _ = G.place_picked(boxes, SURFACE, 0, 0.35, constraints=CARTON)
    assert again == placed  # the seed fixes the scene
    crowd = {f"big_{i}": [0.22, 0.22, 0.1] for i in range(8)} | {"nor_cal_box": [0.225, 0.16, 0.165]}
    with pytest.raises(RuntimeError, match="no spot left for big_"):
        G.place_picked(crowd, SURFACE, 0, 0.35, constraints=CARTON, attempts=3)


def test_the_one_hand_scene_places_its_twenty_objects_without_the_carton():
    """User 2026-11-17: no carton in logistics_room_gso_single, so twenty one-hand objects take the band."""
    import gso_scene_pick as SP

    out = SP.place({"scene": "single", "seed": 0, "objects": [f"Single_{i}" for i in range(20)]}, CATALOG, SURFACE)
    assert out["ok"] and len(out["items"]) == 20 and not any(it["carton"] for it in out["items"])
    two = SP.place({"scene": "bimanual", "seed": 0, "objects": [f"Wide_{i}" for i in range(5)]}, CATALOG, SURFACE)
    assert sum(it["carton"] for it in two["items"]) == 1


def test_the_picking_page_carries_each_pool_and_saves_what_the_builder_reads(tmp_path):
    import gso_scene_pick as SP

    page = SP.page(CATALOG, {"scenes": {"single": {"seed": 2, "objects": ["Single_3"]}}})
    data = json.loads(page.split("const D=", 1)[1].split(";\nconst $", 1)[0])
    assert {r["key"] for r in data["rows"]["single"]} == set(G.pick_pool(CATALOG, "single"))
    assert data["rules"]["bimanual"]["count"] == [5, 9] and data["picks"]["single"]["objects"] == ["Single_3"]
    assert data["rules"]["single"] == {"count": [20, 20], "scene": "logistics_room_gso_single", "kind": "single",
                                       "carton": False}  # fmt: skip
    assert "+'\\n']" in page  # the download's newline stays a JS escape
    path = tmp_path / "scene_picks.json"
    SP.save(path, {"scenes": {"single": {"seed": 1, "objects": ["Single_0", "Single_5"]}}}, CATALOG)
    assert G.load_picks(path, CATALOG)["scenes"] == {"single": {"seed": 1, "objects": ["Single_0", "Single_5"]}}
    with pytest.raises(ValueError):
        SP.save(path, {"scenes": {"single": {"objects": ["Nope"]}}}, CATALOG)
    assert G.load_picks(path, CATALOG)["scenes"]["single"]["objects"] == ["Single_0", "Single_5"]
