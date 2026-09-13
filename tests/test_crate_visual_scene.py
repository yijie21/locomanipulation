import json
import xml.etree.ElementTree as ET

from locomani.tasks.crate.visual_scene import build


def test_wall_and_floor_paint_adds_no_mass_or_contact_and_preserves_source(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "painted"
    source.mkdir()
    original = '<mujoco><worldbody><body name="blue_crate_object"><geom name="floor" type="box" size=".2 .13 .005" mass=".2"/></body></worldbody></mujoco>'
    (source / "scene_locomani.xml").write_text(original)
    (source / "scene_meta.json").write_text("{}")
    build(source, destination)
    assert (source / "scene_locomani.xml").read_text() == original
    root = ET.parse(destination / "scene_locomani.xml")
    paint = root.findall('.//geom[@mass="0"]')
    assert len(paint) == 10
    assert all(geom.get("contype") == geom.get("conaffinity") == "0" for geom in paint)
    assert root.find('.//geom[@name="floor"]').get("mass") == ".2"
    metadata = json.loads((destination / "scene_meta.json").read_text())["crate_visual"]
    assert metadata["version"] == 3 and len(metadata["markers"]) == 8
