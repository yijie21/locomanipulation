"""Explicit portable geometry inputs for CPU control tests; no generated output paths."""

import json
import tempfile
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from pathlib import Path

from locomani.primitives.arm_ik import ArmIK

DIRECTORY = Path(__file__).parent / "fixtures/geometry"


def scene_meta():
    return json.loads((DIRECTORY / "scene_meta.json").read_text())


def scene_ik():
    return ArmIK(str(DIRECTORY / "scene_locomani.xml"))


@contextmanager
def open_box_scene(box, obstacle=None):
    """Use the tiny CPU scene robot with its target replaced by a matched box."""
    from locomani.tasks.crate.box_rim_cli import prepared_scene

    target = {"body": "water_bottle_object", "collider_mode": "replace_with_open_box_proxy"}
    with prepared_scene(DIRECTORY / "scene_locomani.xml", target, box) as path:
        if obstacle is None:
            yield path
            return
        tree = ET.parse(path)
        world = tree.getroot().find("worldbody")
        ET.SubElement(
            world,
            "geom",
            {
                "name": "box_rim_test_obstacle",
                "type": "box",
                "pos": " ".join(map(str, obstacle[0])),
                "size": " ".join(map(str, obstacle[1])),
            },
        )
        handle = tempfile.NamedTemporaryFile(dir=DIRECTORY, prefix="box_rim_obstacle_", suffix=".xml", delete=False)
        obstacle_path = Path(handle.name)
        handle.close()
        tree.write(obstacle_path, encoding="unicode")
        try:
            yield obstacle_path
        finally:
            obstacle_path.unlink(missing_ok=True)
