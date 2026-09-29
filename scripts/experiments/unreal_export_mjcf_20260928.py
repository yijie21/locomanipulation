"""Export a run's saved scene as a self-contained MJCF folder for URLab (Unreal) to import (TODO: Unreal smoke test).

URLab builds its Unreal Blueprint from an MJCF on disk and resolves meshes relative to the XML, so this copies
every mesh and texture the scene references into <out>/assets/ and rewrites the paths. Three variants:

    robot    the G1 + dex1 with the three dataset cameras, a floor and a light; nq 40 (robot joints only)
    scene    the whole saved scene (warehouse room, carton, platform); nq equals the trace's, so a state-trace row
             is a puppet qpos as it is
    dynamic  the moving part of the scene for an Unreal level that brings its own room and lights
             (unreal_warehouse_level_20260928.py): robot + carton + the world cameras, no room geoms, no MJCF
             lights, no invisible collision-only geoms; nq equals the trace's

All drop the com_marker debug site, which URLab draws as a large white sphere (the IMU sites stay: sensors use them).

The joint order of each variant is written to <out>/<variant>_joints.json so a replay can map trace columns by
name rather than by position.

    PYTHONPATH=src "$VENV_SIM_PY" scripts/experiments/unreal_export_mjcf_20260928.py \
        outputs/runs/box_room_grasp_200_cap120_20260925/eval/warehouse_textured/scene_locomani.xml \
        --out outputs/unreal_smoke_20260928/mjcf
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mujoco

ROBOT_ROOT = "pelvis"
CARTON_ROOT = "nor_cal_box_object"
KEEP_CAMERAS = ("head_camera", "left_wrist", "right_wrist")


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


def strip_to_robot(spec) -> None:
    """Delete every world body, world geom and world camera that is not the robot; keep lights and the floor."""
    keep = _subtree(spec, ROBOT_ROOT)
    for b in list(spec.worldbody.bodies):
        if b.name not in keep:
            spec.delete(b)
    for g in list(spec.worldbody.geoms):
        if g.name != "floor":
            spec.delete(g)
    for c in list(spec.worldbody.cameras):
        if c.name not in KEEP_CAMERAS:
            spec.delete(c)
    _drop_unused_assets(spec)


def strip_to_dynamic(spec) -> None:
    """Keep the robot and the carton subtrees and the world cameras; drop world geoms, lights, invisible geoms."""
    keep = _subtree(spec, ROBOT_ROOT) | _subtree(spec, CARTON_ROOT)
    for b in list(spec.worldbody.bodies):
        if b.name not in keep:
            spec.delete(b)
    for g in list(spec.worldbody.geoms):
        spec.delete(g)
    for light in list(spec.lights):
        spec.delete(light)
    for g in list(spec.geoms):
        if g.rgba[3] == 0:
            spec.delete(g)
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


def export(scene_xml: Path, out: Path, variant: str) -> dict:
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    if variant == "robot":
        strip_to_robot(spec)
    elif variant == "dynamic":
        strip_to_dynamic(spec)
    for site in list(spec.sites):
        if site.name == "com_marker":
            spec.delete(site)
    localise_assets(spec, scene_xml.parent, out / "assets")
    spec.modelfiledir = str(out)  # the rewritten asset paths are relative to the exported XML
    spec.compile()
    xml_path = out / f"{variant}.xml"
    xml_path.write_text(spec.to_xml())
    model = mujoco.MjModel.from_xml_path(str(xml_path))  # the written file must load on its own
    joints = [
        {"name": model.joint(j).name, "type": int(model.jnt_type[j]), "qposadr": int(model.jnt_qposadr[j])}
        for j in range(model.njnt)
    ]
    (out / f"{variant}_joints.json").write_text(json.dumps({"nq": model.nq, "joints": joints}, indent=1))
    cams = [model.camera(i).name for i in range(model.ncam)]
    return {"variant": variant, "xml": str(xml_path), "nq": model.nq, "nmesh": model.nmesh, "cameras": cams}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_xml", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--variants", nargs="+", default=["robot", "scene", "dynamic"], choices=["robot", "scene", "dynamic"]
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for v in args.variants:
        print(json.dumps(export(args.scene_xml.resolve(), args.out, v)))


if __name__ == "__main__":
    main()
