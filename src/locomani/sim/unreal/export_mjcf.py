"""A scene's moving part as a self-contained MJCF folder for URLab's importer: the robot, the target object and
the three dataset cameras, nothing the Unreal level brings itself.

URLab builds its Blueprint from an MJCF on disk and resolves meshes relative to the XML, so every mesh and texture
the scene references is copied into <out>/assets/ and the paths rewritten. Dropped: the world geoms (the room and
the platform are the level's baked meshes), the MJCF lights (the level has the Blender lights), invisible
collision-only geoms, every world camera (a camera URLab imports renders each frame until it is first captured;
the four-edge scenes carry four) and the com_marker debug site (URLab draws it as a large white sphere). The
dataset cameras get their `CAMERA_CONFIGS` resolution explicitly, which URLab's render targets take.

The XML is named after the level (`<out>/<level>.xml`), since URLab names the Blueprint after the file:
/Game/MuJoCoImports/<level>. The joints are written to <level>_joints.json for a reader who maps by name.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mujoco

from locomani.sim.dataset_cameras import CAMERA_CONFIGS

ROBOT_ROOT = "pelvis"


def _subtree(spec, root: str) -> set[str]:
    names, stack = set(), [spec.body(root)]
    while stack:
        b = stack.pop()
        names.add(b.name)
        stack.extend(b.bodies)
    return names


def _drop_unused_assets(spec) -> None:
    used_meshes = {g.meshname for g in spec.geoms if g.meshname}
    for m in list(spec.meshes):
        if m.name not in used_meshes:
            spec.delete(m)
    used_materials = {g.material for g in spec.geoms if g.material}
    for m in list(spec.materials):
        if m.name not in used_materials:
            spec.delete(m)
    used_textures = {t for m in spec.materials for t in m.textures if t}
    for t in list(spec.textures):
        if t.file and t.name not in used_textures:
            spec.delete(t)


def strip_to_dynamic(spec, target_body: str) -> None:
    """Keep the robot and target subtrees with the dataset cameras; drop world geoms and cameras, lights,
    invisible geoms and the com_marker site."""
    keep = _subtree(spec, ROBOT_ROOT) | _subtree(spec, target_body)
    for b in list(spec.worldbody.bodies):
        if b.name not in keep:
            spec.delete(b)
    for g in list(spec.worldbody.geoms):
        spec.delete(g)
    for c in list(spec.worldbody.cameras):
        spec.delete(c)
    for light in list(spec.lights):
        spec.delete(light)
    for g in list(spec.geoms):
        if g.rgba[3] == 0:
            spec.delete(g)
    for site in list(spec.sites):
        if site.name == "com_marker":
            spec.delete(site)
    for cfg in CAMERA_CONFIGS.values():
        cam = spec.camera(cfg["mjcf_name"])
        if cam is None:
            raise SystemExit(f"the scene has no camera {cfg['mjcf_name']!r} under the robot")
        cam.resolution = [cfg["width"], cfg["height"]]
    _drop_unused_assets(spec)


def localise_assets(spec, src_dir: Path, asset_dir: Path) -> None:
    """Copy each referenced mesh/texture file into asset_dir (unique names) and point the spec at it."""
    asset_dir.mkdir(parents=True, exist_ok=True)
    spec.meshdir = spec.texturedir = spec.compiler.meshdir = spec.compiler.texturedir = ""
    for kind, items in (("mesh", spec.meshes), ("tex", spec.textures)):
        for item in items:
            if not item.file:
                continue
            src = Path(item.file)
            if not src.is_absolute():
                src = src_dir / src
            dst_name = f"{kind}_{item.name}{src.suffix}" if item.name else f"{kind}_{src.name}"
            shutil.copy2(src, asset_dir / dst_name)
            item.file = f"{asset_dir.name}/{dst_name}"


def export(scene_dir: Path, out: Path, level: str) -> dict:
    """Write `<out>/<level>.xml` (+ assets/, joints) from the scene directory; returns a summary."""
    meta = json.loads((scene_dir / "scene_meta.json").read_text())
    target = meta["objects"][meta["target_object"]]["body"]
    scene_xml = scene_dir / "scene_locomani.xml"
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    strip_to_dynamic(spec, target)
    if out.exists():
        shutil.rmtree(out)
    localise_assets(spec, scene_xml.parent, out / "assets")
    spec.modelfiledir = str(out)  # the rewritten asset paths are relative to the exported XML
    spec.compile()
    xml_path = out / f"{level}.xml"
    xml_path.write_text(spec.to_xml())
    model = mujoco.MjModel.from_xml_path(str(xml_path))  # the written file must load on its own
    joints = [
        {"name": model.joint(j).name, "type": int(model.jnt_type[j]), "qposadr": int(model.jnt_qposadr[j])}
        for j in range(model.njnt)
    ]
    (out / f"{level}_joints.json").write_text(json.dumps({"nq": model.nq, "joints": joints}, indent=1))
    cams = [model.camera(i).name for i in range(model.ncam)]
    return {"xml": str(xml_path), "nq": model.nq, "nmesh": model.nmesh, "cameras": cams, "target": target}
