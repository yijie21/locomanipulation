"""Add visible paint patches to an existing crate scene without changing its contacts."""

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from locomani.tasks.crate.grasp_region import GRASP_POINTS

MARKERS = {
    "cyan": {"position": [0.075, -0.1178, 0.065], "rgba": [0.03, 1, 1, 1]},
    "green": {"position": [0.075, -0.1178, -0.035], "rgba": [0.03, 1, 0.03, 1]},
    "magenta": {"position": [-0.075, -0.1178, 0.065], "rgba": [1, 0.03, 1, 1]},
    "yellow": {"position": [-0.075, -0.1178, -0.035], "rgba": [1, 1, 0.03, 1]},
}


FLOOR_MARKERS = {
    "red": {"position": [0.060, -0.050, -0.074], "rgba": [1, 0.03, 0.03, 1]},
    "orange": {"position": [0.060, 0.050, -0.074], "rgba": [1, 0.40, 0.03, 1]},
    "blue": {"position": [-0.060, -0.050, -0.074], "rgba": [0.03, 0.15, 1, 1]},
    "violet": {"position": [-0.060, 0.050, -0.074], "rgba": [0.25, 0.03, 1, 1]},
}


def build(source: Path, destination: Path):
    """Create a distinct scene; patches have zero mass and no collision response."""
    if destination.exists():
        raise FileExistsError(f"scene already exists: {destination}")
    shutil.copytree(source, destination)
    xml = destination / "scene_locomani.xml"
    tree = ET.parse(xml)
    body = tree.find('.//body[@name="blue_crate_object"]')
    if body is None:
        raise ValueError("source scene has no blue_crate_object")
    # A label on the already solid inner far wall stays visible when grippers
    # cover the rim. It changes appearance only, not mass or contacts.
    ET.SubElement(
        body,
        "geom",
        name="crate_visual_label",
        type="box",
        size="0.10 0.0002 0.065",
        pos="0 -0.1183 0.015",
        rgba="0.025 0.04 0.06 1",
        contype="0",
        conaffinity="0",
        group="1",
        mass="0",
    )
    ET.SubElement(
        body,
        "geom",
        name="crate_visual_floor_label",
        type="box",
        size="0.085 0.075 0.0002",
        pos="0 0 -0.0747",
        rgba="0.025 0.04 0.06 1",
        contype="0",
        conaffinity="0",
        group="1",
        mass="0",
    )
    markers = {**MARKERS, **FLOOR_MARKERS}
    for name, marker in markers.items():
        ET.SubElement(
            body,
            "geom",
            name=f"crate_visual_{name}",
            type="cylinder",
            size="0.009 0.0003" if name in FLOOR_MARKERS else "0.0125 0.0003",
            quat="1 0 0 0" if name in FLOOR_MARKERS else "0.707106781 -0.707106781 0 0",
            pos=" ".join(map(str, marker["position"])),
            rgba=" ".join(map(str, marker["rgba"])),
            contype="0",
            conaffinity="0",
            group="1",
            mass="0",
        )
    tree.write(xml)
    meta_path = destination / "scene_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["scene_xml"] = str(xml.resolve())
    meta["crate_visual"] = {
        "version": 3,
        "markers": markers,
        "marker_groups": [list(MARKERS), list(FLOOR_MARKERS)],
        "grasp_points": GRASP_POINTS,
    }
    meta_path.write_text(json.dumps(meta, indent=2))


def remove_paint(scene: Path):
    """Restore the original appearance in a newly copied run scene, preserving physics."""
    xml = scene / "scene_locomani.xml"
    tree = ET.parse(xml)
    body = tree.find('.//body[@name="blue_crate_object"]')
    if body is not None:
        for geom in list(body.findall("geom")):
            if geom.get("name", "").startswith("crate_visual_"):
                body.remove(geom)
        tree.write(xml)
    meta_path = scene / "scene_meta.json"
    meta = json.loads(meta_path.read_text())
    meta.pop("crate_visual", None)
    meta["scene_xml"] = str(xml.resolve())
    meta_path.write_text(json.dumps(meta, indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    build(args.source, args.destination)


if __name__ == "__main__":
    main()
