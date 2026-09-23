"""The walk scene's follow camera: placed from the pelvis the XML has, and made the video camera."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from locomani.tasks.walk import scene as walk_scene  # noqa: E402

XML = """<mujoco><worldbody><geom name="floor" type="plane" size="0 0 0.05"/>
<body name="pelvis" pos="-1.5 0.2 0.793"><freejoint/><geom type="sphere" size="0.1" mass="1"/></body>
</worldbody></mujoco>"""


def test_follow_camera_is_offset_from_the_pelvis_and_becomes_the_video_camera(tmp_path):
    (tmp_path / "scene_locomani.xml").write_text(XML)
    (tmp_path / "scene_meta.json").write_text(json.dumps({"video_camera": "reach_overview"}))
    walk_scene.add_follow_camera(tmp_path)
    walk_scene.add_follow_camera(tmp_path)  # a rebuild over the same directory adds nothing
    cams = ET.parse(tmp_path / "scene_locomani.xml").getroot().findall("worldbody/camera")
    assert [c.get("name") for c in cams] == ["walk_follow"]
    assert cams[0].get("mode") == "trackcom"
    assert [float(v) for v in cams[0].get("pos").split()] == [-1.5, 0.2 - 3.0, 1.0]
    assert json.loads((tmp_path / "scene_meta.json").read_text())["video_camera"] == "walk_follow"
