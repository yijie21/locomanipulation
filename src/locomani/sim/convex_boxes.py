"""Represent box colliders as equal-volume convex meshes for generic collision detection.

MuJoCo's analytic box/box collider can fabricate deep contacts between nearly
coplanar resting faces (upstream issue 3476). Eight vertices preserve each box's
surface, mass and inertia while selecting the generic convex collider.
"""

import xml.etree.ElementTree as ET
from itertools import product


def convert_body_boxes(tree, body_name):
    """Convert explicit box sizes in one body; leave visual meshes and other bodies alone."""
    body = tree.find(f'.//body[@name="{body_name}"]')
    if body is None:
        raise ValueError(f"missing collision body {body_name}")
    asset = tree.find("asset")
    if asset is None:
        asset = ET.SubElement(tree.getroot(), "asset")
    for index, geom in enumerate(body.findall("geom")):
        size = geom.get("size", "").split()
        if geom.get("type", "box") != "box" or len(size) != 3:
            continue
        name = f"{body_name}_convex_box_{index}"
        half = list(map(float, size))
        vertices = " ".join(
            format(sign * extent, ".17g") for signs in product((-1, 1), repeat=3) for sign, extent in zip(signs, half)
        )
        ET.SubElement(asset, "mesh", name=name, vertex=vertices)
        geom.set("type", "mesh")
        geom.set("mesh", name)
        geom.attrib.pop("size")
