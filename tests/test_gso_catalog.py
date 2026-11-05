"""The Google Scanned Objects catalog and the scene builder's draw (locomani.sim.gso_objects, 2026-11-03): phrases
from directory names, the graspability rule, the dataset-category exclusion (no shoes, 2026-11-04), the stable train /
held-out split, the per-instance draw and the non-overlapping placement on the platform's reach band."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from locomani.sim import gso_objects as G
from locomani.tasks.scene.start import edge_depths, footprint, inside_top, overlap

TH = G.Thresholds()
PLATFORM = {"xy": [0.0, -0.105], "top": 0.371, "half_extent": [0.6, 0.45], "flat_half_extent": [0.6, 0.45],
            "yaw": 0.0, "approach_edges": ["-x", "+x", "-y", "+y"]}  # fmt: skip
TETRA = "v 0 0 0\nv 0.05 0 0\nv 0 0.05 0\nv 0 0 0.05\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"


def test_a_phrase_drops_the_scan_id_and_keeps_names_without_one():
    assert G.phrase("Lovable_Huggable_Teddy_Bear_Beige_NJnjGna4u1a") == "lovable huggable teddy bear beige"
    assert G.phrase("Android_Lego") == "android lego"
    assert G.phrase("FISHING_GAME") == "fishing game"
    assert G.phrase("Nintendo_Mario_Action_Figure") == "nintendo mario action figure"


@pytest.mark.parametrize(
    "extent, rest, kind, why",
    [
        ((0.06, 0.10, 0.15), None, "single", "fits the 9.5 cm jaw"),
        ((0.10, 0.06, 0.15), None, "single", "fits the 9.5 cm jaw"),  # the narrow side is whichever is narrower
        ((0.06, 0.10, 0.02), None, "none", "flatter than 3 cm"),
        ((0.06, 0.10, 0.05), None, "none", "is under 0.4 m"),  # grasp point 0.371 + 0.025
        ((0.15, 0.30, 0.12), None, "bimanual", "between two hands"),
        ((0.15, 0.30, 0.05), None, "none", "outside the two-hand range"),
        ((0.15, 0.60, 0.30), None, "none", "outside the two-hand range"),
        ((0.06, 0.10, 0.15), {"stands": False, "tilt_deg": 90.0, "slid_mm": 3.0}, "none", "does not stand at rest"),
        ((0.06, 0.10, 0.15), {"stands": True, "tilt_deg": 0.1, "slid_mm": 0.2}, "single", "fits"),
    ],
)
def test_the_graspability_rule_sorts_by_jaw_width_two_hand_band_height_and_rest(extent, rest, kind, why):
    got, reason = G.graspable(extent, TH, rest)
    assert got == kind and why in reason


def test_the_split_is_a_stable_name_hash_near_the_asked_fraction():
    names = [f"Object_{i}_Variant" for i in range(2000)]
    splits = [G.split_of(n, 0.10) for n in names]
    assert splits == [G.split_of(n, 0.10) for n in names]
    held = splits.count("held_out") / len(names)
    assert 0.07 <= held <= 0.13 and set(splits) == {"train", "held_out"}
    assert all(G.split_of(n, 0.0) == "train" for n in names[:50])


def catalog(n_single=20, n_bimanual=10, n_none=5):
    objs = {}
    i = 0
    for kind, n in (("single", n_single), ("bimanual", n_bimanual), ("none", n_none)):
        for j in range(n):
            key = G.KEY_FORMAT.format(index=i)
            objs[key] = {"name": f"{kind}_{j}", "graspable": kind, "split": "held_out" if j % 5 == 0 else "train"}
            i += 1
    return {"version": G.CATALOG_VERSION, "objects": objs}


def test_the_draw_takes_the_asked_mix_from_the_asked_pool_without_repeats_and_repeats_by_seed():
    cat = catalog()
    keys = G.sample_keys(cat, seed=3, n=10, n_bimanual=4, pool="train")
    kinds = [cat["objects"][k]["graspable"] for k in keys]
    assert len(set(keys)) == 10 and kinds.count("single") == 6 and kinds.count("bimanual") == 4
    assert all(cat["objects"][k]["split"] == "train" for k in keys)
    assert keys == G.sample_keys(cat, seed=3, n=10, n_bimanual=4, pool="train")
    assert keys != G.sample_keys(cat, seed=4, n=10, n_bimanual=4, pool="train")
    unseen = G.sample_keys(cat, seed=3, n=4, n_bimanual=2, pool="eval_unseen")
    assert all(cat["objects"][k]["split"] == "held_out" for k in unseen)
    assert len(G.pool_keys(cat, "single", "all")) == 20 and len(G.pool_keys(cat, "single", "eval_unseen")) == 4
    with pytest.raises(ValueError, match="holds 4 single"):
        G.sample_keys(cat, seed=0, n=10, n_bimanual=4, pool="eval_unseen")
    with pytest.raises(ValueError, match="between 0"):
        G.sample_keys(cat, seed=0, n=3, n_bimanual=4)
    with pytest.raises(ValueError, match="pool"):
        G.pool_keys(cat, "single", "test")


def test_the_placement_keeps_every_footprint_inside_the_top_apart_and_on_the_reach_band_when_it_can():
    boxes = {"nor_cal_box": [0.225, 0.16, 0.165]}
    rng = np.random.default_rng(1)
    for i in range(10):
        boxes[f"gso_{i:04d}"] = list(rng.uniform(0.03, 0.12, size=2)) + [0.1]
    placed = G.place_objects(boxes, PLATFORM, seed=5, reach_depth_m=0.35)
    assert set(placed) == set(boxes) and placed == G.place_objects(boxes, PLATFORM, seed=5, reach_depth_m=0.35)
    top = (np.asarray(PLATFORM["xy"]), np.asarray(PLATFORM["flat_half_extent"]) - G.SURFACE_MARGIN_M, 0.0)
    fps = {}
    for key, p in placed.items():
        q = np.array(G.yaw_quat(p["yaw"]))
        fps[key] = footprint({"box": {"half_extent": boxes[key][:2]}}, p["xy"], q)
        assert inside_top(fps[key], top), key
        depth = min(edge_depths(PLATFORM, p["xy"]).values())
        assert p["in_reach_band"] == (depth <= 0.35 + 1e-9), key
    assert all(p["in_reach_band"] for p in placed.values())  # 11 objects fit on the band of the 1.2 x 0.9 m platform
    keys = list(fps)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            assert not overlap(fps[keys[a]], fps[keys[b]], 0.03), (keys[a], keys[b])


def test_a_platform_too_small_for_the_objects_is_refused():
    with pytest.raises(RuntimeError, match="no spot"):
        G.place_objects({f"o{i}": [0.3, 0.3, 0.1] for i in range(6)}, PLATFORM, seed=0, reach_depth_m=0.35, tries=200)


def test_mass_density_and_hull_volume(tmp_path):
    (tmp_path / "hull.obj").write_text(TETRA)
    assert G.hull_volume(tmp_path / "hull.obj") == pytest.approx(0.05**3 / 6, rel=1e-6)
    assert G.density_of(1.2, 0.004) == pytest.approx(300.0)
    q = G.yaw_quat(np.pi / 2)
    assert np.allclose(q, [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])


def test_an_object_of_an_excluded_dataset_category_is_none_whatever_its_shape():
    kind, why = G.graspable((0.06, 0.10, 0.15), TH, None, category="Shoe")  # a single-hand shape
    assert kind == "none" and "category Shoe" in why
    assert G.graspable((0.15, 0.30, 0.12), TH, None, category="Shoe")[0] == "none"  # a two-hand shape
    assert G.graspable((0.06, 0.10, 0.15), TH, None, category="Toys")[0] == "single"
    assert G.graspable((0.06, 0.10, 0.15), TH, None, category=None)[0] == "single"  # unlabelled: the rule alone
    lenient = G.Thresholds(excluded_categories=())
    assert G.graspable((0.06, 0.10, 0.15), lenient, None, category="Shoe")[0] == "single"
    assert "Shoe" in G.EXCLUDED_CATEGORIES and TH.excluded_categories == G.EXCLUDED_CATEGORIES


def test_the_catalog_entry_records_the_category_and_the_builder_refuses_unlabelled_models(tmp_path):
    d = tmp_path / "models" / "Some_Sneaker"
    d.mkdir(parents=True)
    (d / "model.obj").write_text("v -0.03 -0.05 0\nv 0.03 -0.05 0\nv 0.03 0.05 0.15\nv -0.03 0.05 0.15\n")
    (d / "model_collision_0.obj").write_text(TETRA)
    shoe = G.catalog_entry(0, d, TH, rest=False, category="Shoe")
    assert shoe["category"] == "Shoe" and shoe["graspable"] == "none" and "category Shoe" in shoe["why"]
    toy = G.catalog_entry(0, d, TH, rest=False, category="Toys")
    assert toy["category"] == "Toys" and toy["graspable"] == "single"
    with pytest.raises(ValueError, match="no entry in categories.json"):
        G.build_catalog(tmp_path / "models", TH, workers=1, categories={})
    (tmp_path / "categories.json").write_text('{"categories": {"Some_Sneaker": "Shoe", "Other": null}}')
    assert G.load_categories(tmp_path / "categories.json") == {"Some_Sneaker": "Shoe", "Other": None}


def test_the_tracked_catalog_holds_no_drawable_shoe_and_every_name_has_a_label():
    assets = Path(__file__).resolve().parents[1] / "assets/mujoco_scanned_objects"
    if not (assets / "catalog.json").is_file() or not (assets / "categories.json").is_file():
        pytest.skip("no tracked catalog / categories")
    cat = G.load_catalog(assets / "catalog.json")
    labels = G.load_categories(assets / "categories.json")
    assert cat["thresholds"]["excluded_categories"] == ["Shoe"]
    assert all(o["name"] in labels and o["category"] == labels[o["name"]] for o in cat["objects"].values())
    shoes = [o for o in cat["objects"].values() if o["category"] == "Shoe"]
    assert len(shoes) == cat["counts"]["excluded"] > 0
    assert all(o["graspable"] == "none" for o in shoes)
    for kind in ("single", "bimanual"):
        for pool in G.POOLS:
            assert all(cat["objects"][k]["category"] != "Shoe" for k in G.pool_keys(cat, kind, pool))
