"""The scanned-object gallery page (scripts/gso_gallery.py, 2026-11-05): the camera frames the object's bounding box,
the page carries every catalog object with its kind, category and exclusion, and one object renders when the meshes
and an offscreen GL are there."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gso_gallery as G  # noqa: E402

CATALOG = {
    "counts": {"objects": 2, "single": 1, "bimanual": 0, "none": 1, "held_out": 0, "excluded": 1},
    "thresholds": {"used_categories": ["(none)"]},
    "not_used_categories": ["Shoe"],
    "objects": {
        "gso_0000": {"name": "Some_Sneaker", "phrase": "some sneaker", "category": "Shoe", "graspable": "none",
                     "why": "category Shoe: kept out", "split": "train", "mass_kg": 0.4, "extent_m": [0.1, 0.29, 0.12],
                     "grasp_height_m": 0.431, "rest": {"stands": True}},
        "gso_0001": {"name": "Red_Mug</script>", "phrase": "red mug", "category": None, "graspable": "single",
                     "why": "fits", "split": "train", "mass_kg": 0.2, "extent_m": [0.08, 0.08, 0.1],
                     "grasp_height_m": 0.421, "rest": None, "fits_carton": True},
    },
}  # fmt: skip


def test_the_camera_frames_the_bounding_sphere_at_the_fixed_view():
    cam = G.camera_for([-0.1, -0.2, 0.0], [0.1, 0.2, 0.3])
    assert cam["lookat"] == pytest.approx([0.0, 0.0, 0.15])
    radius = 0.5 * math.dist([-0.1, -0.2, 0.0], [0.1, 0.2, 0.3])
    assert math.asin(radius / cam["distance"]) == pytest.approx(math.radians(G.FOVY) / 2 / G.FRAME, rel=0.05)
    assert (cam["azimuth"], cam["elevation"]) == (G.AZIMUTH, G.ELEVATION)


def test_the_page_carries_every_object_and_marks_the_excluded_category():
    rows = G.entries(CATALOG)
    assert [r["key"] for r in rows] == ["gso_0000", "gso_0001"]
    assert rows[0]["excluded"] and not rows[1]["excluded"] and rows[1]["category"] == "(none)"
    assert rows[0]["extent"] == [10.0, 29.0, 12.0] and rows[0]["img"] == "img/gso_0000.jpg"
    page = G.page(CATALOG)
    assert "__DATA__" not in page and "Red_Mug</script>" not in page  # a name cannot close the data script
    data = json.loads(page.split("const D=", 1)[1].split(";\nconst $", 1)[0])
    assert {d["key"] for d in data} == {"gso_0000", "gso_0001"} and "excluded categories 1: Shoe" in page


def test_the_page_carries_the_shape_scores_the_kind_by_the_rules_and_the_choices_by_hand():
    rows = G.entries(CATALOG, {"gso_0001": {"fill": 0.42, "solid": 0.9}})
    assert rows[1]["fill"] == 0.42 and rows[0]["fill"] is None
    # `kind` is by the shape rules alone (the shoe's shape is two-hand); `auto` adds the category (the shoe is none)
    assert [r["kind"] for r in rows] == ["bimanual", "single"] and [r["auto"] for r in rows] == ["none", "single"]
    assert [r["excluded"] for r in rows] == [True, False] and [r["fits"] for r in rows] == [False, True]
    cat = json.loads(json.dumps(CATALOG))
    cat["objects"]["gso_0001"].update(graspable="none", by_hand={"excluded": True, "exclude_why": "toy or figurine"})
    assert G.entries(cat)[1]["auto"] == "single"  # a rebuilt catalog's hand choice does not hide the rules' kind
    page = G.page(CATALOG, choices={"Red_Mug</script>": {"excluded": True, "exclude_why": "irregular shape"}})
    assert "let CH={" in page and "Red_Mug</script>" not in page and "<option>checked by eye</option>" in page
    assert "+'\\n']" in page  # the download handler's newline stays a JS escape, not a break inside the string


def test_the_choices_are_saved_sorted_and_only_for_catalog_objects_in_the_entry_shape(tmp_path):
    path = tmp_path / "hand_choices.json"
    names = {"B_Thing", "A_Thing"}
    G.save_marks(path, {"objects": {"B_Thing": {"excluded": True, "exclude_why": "irregular shape"},
                                    "A_Thing": {"shape": "bimanual", "shape_why": "checked by eye", "junk": 1}}},
                 names)  # fmt: skip
    doc = json.loads(path.read_text())
    assert list(doc["objects"]) == ["A_Thing", "B_Thing"]
    assert doc["objects"]["A_Thing"] == {"shape": "bimanual", "shape_why": "checked by eye"}  # unknown keys dropped
    for bad in ({"objects": {"Nope": {"excluded": True}}}, {"objects": {"A_Thing": {"shape": "both"}}},
                {"objects": {"A_Thing": {}}}, {"objects": {"A_Thing": "irregular shape"}}, {"objects": []}, []):  # fmt: skip
        with pytest.raises(ValueError):
            G.save_marks(path, bad, names)
    assert json.loads(path.read_text()) == doc  # a refused body leaves the file alone


def test_one_object_renders_when_the_meshes_and_offscreen_gl_are_there(tmp_path):
    cat_path = G.ASSETS / "catalog.json"
    if os.environ.get("MUJOCO_GL") not in ("egl", "osmesa") or not cat_path.is_file():
        pytest.skip("needs MUJOCO_GL=egl or osmesa and the tracked catalog")
    rec = json.loads(cat_path.read_text())["objects"]["gso_0001"]
    model_dir = G.ASSETS / "models" / rec["name"]
    if not (model_dir / "model.obj").is_file():
        pytest.skip("no meshes: bash scripts/install_gso.sh")
    out = tmp_path / "gso_0001.jpg"
    assert G.render_one(("gso_0001", str(model_dir), rec, str(out), 128)) == ("gso_0001", None)
    import numpy as np
    from PIL import Image

    im = Image.open(out)
    assert im.size == (128, 128) and len(np.unique(np.asarray(im.convert("L")))) > 20  # an object, not blank
