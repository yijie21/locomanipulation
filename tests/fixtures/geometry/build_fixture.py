import argparse
import hashlib
import json
import xml.etree.ElementTree as E
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

parser = argparse.ArgumentParser(description="Extract a portable convex collision fixture from an explicit scene")
parser.add_argument("scene", type=Path)
parser.add_argument("out", type=Path)
args = parser.parse_args()
src, out = args.scene, args.out
out.mkdir(parents=True, exist_ok=True)
r = E.parse(src).getroot()
for asset in list(r.find("asset")):
    if asset.tag != "mesh":
        r.find("asset").remove(asset)
        continue
    p = Path(asset.attrib.pop("file"))
    if p.suffix.lower() == ".obj":
        verts = np.array(
            [[float(x) for x in line.split()[1:4]] for line in p.read_text().splitlines() if line.startswith("v ")]
        )
    else:
        raw = p.read_bytes()
        n = int.from_bytes(raw[80:84], "little")
        if len(raw) == 84 + 50 * n:
            dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
            verts = np.frombuffer(raw[84:], dtype=dtype)["vertices"].reshape(-1, 3).copy()
        else:
            verts = np.array(
                [
                    [float(x) for x in line.strip().split()[1:]]
                    for line in raw.decode().splitlines()
                    if line.strip().startswith("vertex ")
                ]
            )
    verts = np.unique(verts, axis=0)
    hull = verts[ConvexHull(verts).vertices]
    asset.attrib.pop("content_type", None)
    asset.set("vertex", " ".join(f"{v:.9g}" for v in hull.flat))
for node in r.iter():
    node.attrib.pop("material", None)
    if node.tag == "texture":
        raise RuntimeError("texture remained")
E.indent(r)
E.ElementTree(r).write(out / "scene_locomani.xml", encoding="unicode")
meta = json.loads(src.with_name("scene_meta.json").read_text())
(out / "scene_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
(out / "source.json").write_text(
    json.dumps(
        {
            "source_scene_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "transformation": "Inline each mesh convex hull, remove textures/materials. Retain bodies, joints, collisions and inertial values. Kinematic fixture, not a rendering or dynamics reference.",
        },
        indent=2,
    )
    + "\n"
)
print("fixture bytes", sum(p.stat().st_size for p in out.iterdir()))
