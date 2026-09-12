"""Portable source assets and cache identity for the offline Isaac scene builder.

No Kit dependency: validate a fresh installation before starting the renderer.
Generated EmbodiedGen meshes are Y-up; the shipped MuJoCo meshes are Z-up.
"""

import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MeshSource:
    path: Path
    z_up: bool = False
    texture: Path | None = None


def mesh_source(scene_root: Path, name: str) -> MeshSource | None:
    for sub in ("layout", "surface_b"):
        path = scene_root / sub / "asset3d" / name / "result/mesh" / f"{name}.obj"
        if path.is_file():
            return MeshSource(path)
    path = scene_root / "mjcf" / name / "mesh" / f"{name}.obj"
    xml = path.parent.parent / f"{name}.xml"
    if not path.is_file() or not xml.is_file():
        return None
    texture = ET.parse(xml).find("./asset/texture")
    image = xml.parent / texture.attrib["file"] if texture is not None else None
    if image is not None and not image.is_file():
        raise FileNotFoundError(f"Missing texture: {image}; rerun scripts/install_scene.sh")
    return MeshSource(path, z_up=True, texture=image)


def prepare_obj(source: MeshSource, destination: Path) -> Path:
    """Give the existing Y-up converter a textured OBJ from either source format."""
    if not source.z_up:
        return source.path
    destination.mkdir(parents=True, exist_ok=True)
    obj = destination / source.path.name
    with source.path.open() as reader, obj.open("w") as writer:
        for line in reader:
            fields = line.split()
            if fields and fields[0] in ("v", "vn"):
                x, y, z = map(float, fields[1:4])
                line = f"{fields[0]} {x:.9g} {z:.9g} {-y:.9g}\n"
            elif fields and fields[0] == "mtllib":
                continue
            writer.write(line)
        # The shipped MJCF meshes retain UVs but omit the OBJ's material.mtl.
        # MuJoCo supplies its texture through XML; restore that binding for Isaac.
    if source.texture is not None:
        shutil.copyfile(source.texture, destination / source.texture.name)
        material = destination / "material.mtl"
        material.write_text(f"newmtl material_0\nKd 1 1 1\nmap_Kd {source.texture.name}\n")
        content = obj.read_text()
        obj.write_text("mtllib material.mtl\nusemtl material_0\n" + content)
    return obj


def background_source(scene_root: Path, repo: Path) -> tuple[Path, bool]:
    """Return the room path and whether its world transform is already baked."""
    generated = scene_root / "bg_empty/scene_0000/mesh_model.ply"
    if generated.is_file():
        return generated, False
    shipped = repo / "assets/scene/empty_room_world.ply.gz"
    if not shipped.is_file():
        raise FileNotFoundError(f"Missing tracked room asset: {shipped}; update the checkout")
    return shipped, True


def build_key(scene_root: Path, repo: Path, meta_path: Path, scene_xml: Path) -> str:
    """Validate required geometry and hash sources, builder and destination root.

    Including paths invalidates USD caches copied from another checkout: generated
    USD references are local build products and can contain absolute paths.
    """
    meta = json.loads(meta_path.read_text())
    sources = {meta_path, scene_xml, scene_root / "layout/layout.json", Path(__file__)}
    sources.add(repo / "src/locomani/sim/isaac_scene.py")
    sources.add(background_source(scene_root, repo)[0])
    for key, default in (("A", "dining_table"), ("B", "counter")):
        name = meta.get("surfaces", {}).get(key, {}).get("name", default)
        if mesh_source(scene_root, name) is None:
            raise FileNotFoundError(f"Missing surface {key}: {name}; run bash scripts/install_scene.sh")
    # Source textures/meshes affect conversion too, even when scene_meta is equal.
    for sub in ("mjcf", "layout/asset3d", "surface_b/asset3d"):
        sources.update(path for path in (scene_root / sub).rglob("*") if path.is_file())
    digest = hashlib.sha256(str(repo.resolve()).encode())
    for path in sorted(sources):
        digest.update(str(path.resolve()).encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()[:32]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scene-root", "repo", "meta", "scene-xml"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(build_key(args.scene_root, args.repo, args.meta, args.scene_xml))
