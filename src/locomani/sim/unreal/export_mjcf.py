"""A scene's moving part as a self-contained MJCF folder for URLab's importer: the robot, every free body and the
three dataset cameras, nothing the Unreal level brings itself -- plus what the third-person video needs.

The free bodies are what moves besides the robot: the warehouse carton, and in a scene_discovery apartment every
object that can be picked up (`moving_bodies`); the rest of the room is the level's baked meshes.

URLab builds its Blueprint from an MJCF on disk and resolves meshes relative to the XML, so every mesh and texture
the scene references is copied into <out>/assets/ and the paths rewritten. Dropped: the world geoms (the room and
the platform are the level's baked meshes), the MJCF lights (the level has the Blender lights), invisible
collision-only geoms, every world camera (a camera URLab imports renders each frame until it is first captured;
the four-edge scenes carry four) and the com_marker debug site (URLab draws it as a large white sphere). The
dataset cameras get their `CAMERA_CONFIGS` resolution explicitly, which URLab's render targets take.

The third-person video (`view_render.film_rounds`) is a moving world camera plus ghost grippers, and URLab moves only
what the import's qpos holds, so three free-joint bodies are added, parked 5 m under the floor at qpos0:
`RIG` carries the camera `FILM_CAMERA` at the scene's shot fovy (`third_person_shot.scene_fovy`) and the video's
size, and each `GHOSTS` body carries its gripper's visual meshes (`ghost.hand_parts`, the Isaac ghost's own
offsets) on copies of the meshes under the material `GHOST_MATERIAL`, so the level can make them translucent
magenta without touching the robot's (`in_editor/robot_materials.py`). The views map these joints themselves.

The XML is named after the level (`<out>/<level>.xml`), since URLab names the Blueprint after the file:
/Game/MuJoCoImports/<level>. The joints are written to <level>_joints.json for a reader who maps by name.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mujoco

from locomani.pipeline import sim_video
from locomani.sim import third_person_shot
from locomani.sim.dataset_cameras import CAMERA_CONFIGS
from locomani.sim.isaac import ghost

ROBOT_ROOT = "pelvis"
RIG = "loco_third_person_rig"  # body and free joint
FILM_CAMERA = "loco_third_person"  # not the body's name: URLab renames a camera that shares one ("..._rig2")
GHOSTS = {"left": "loco_ghost_left", "right": "loco_ghost_right"}  # body and free joint, per ghost.SIDES
GHOST_MATERIAL = "loco_ghost"
PARK = (0.0, 0.0, -5.0)


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


def moving_bodies(spec) -> list[str]:
    """The world's free bodies other than the robot: the ones URLab must move."""
    return [
        b.name
        for b in spec.worldbody.bodies
        if b.name != ROBOT_ROOT and any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in b.joints)
    ]


def bake_inertia(spec, model, moving: list[str]) -> list[str]:
    """Give every body of the `moving` subtrees that takes its mass from its geoms an explicit inertial, read from
    the compiled `model` of the whole scene, so that dropping its collision geoms (strip_to_dynamic) leaves its mass
    and inertia as they were. A scanned object (locomani.sim.gso_objects) carries its mass on its convex hulls and
    nothing on its visual mesh: without this it would compile with no mass at all (2026-11-04). The robot's links
    carry their own inertials and are left alone. Returns the bodies baked."""
    baked = []
    for name in sorted(set().union(*(_subtree(spec, b) for b in moving)) if moving else ()):
        body = spec.body(name)
        if body.explicitinertial or not any(g.contype or g.conaffinity for g in body.geoms):
            continue
        mb = model.body(body.name)
        if mb.mass[0] <= 0:
            continue
        body.mass, body.ipos, body.iquat, body.inertia = (
            float(mb.mass[0]),
            mb.ipos.tolist(),
            mb.iquat.tolist(),
            mb.inertia.tolist(),
        )
        body.explicitinertial = True
        baked.append(body.name)
    return baked


def strip_to_dynamic(spec, moving: list[str], model=None) -> None:
    """Keep the robot and the `moving` subtrees with the dataset cameras; drop world geoms and cameras, lights,
    invisible geoms and the com_marker site. With `model` (the compiled scene), the mass a body took from the geoms
    that go is baked in first (bake_inertia)."""
    if model is not None:
        bake_inertia(spec, model, moving)
    keep = _subtree(spec, ROBOT_ROOT).union(*(_subtree(spec, b) for b in moving))
    for b in list(spec.worldbody.bodies):
        if b.name not in keep:
            spec.delete(b)
    for g in list(spec.worldbody.geoms):
        spec.delete(g)
    for c in list(spec.worldbody.cameras):
        spec.delete(c)
    for light in list(spec.lights):
        spec.delete(light)
    # Coloured collision geoms (the mars robot's wear their link's colour so MuJoCo's own cameras draw them; the stock
    # G1 has a few shoulder and ankle primitives) go too where the body has visuals: URLab draws a primitive geom as
    # an engine shape in its default grey checker material, and the mars wrist's D405 collision box boxed the
    # camera in (2026-10-03).
    for g in list(spec.geoms):
        if g.rgba[3] == 0:
            spec.delete(g)
    for body in spec.bodies:
        geoms = list(body.geoms)
        if any(not (g.contype or g.conaffinity) for g in geoms):
            for g in geoms:
                if g.contype or g.conaffinity:
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


def _parked_body(spec, name: str):
    body = spec.worldbody.add_body(name=name, pos=list(PARK))
    body.add_freejoint(name=name)
    body.explicitinertial = True
    body.mass, body.inertia = 1e-3, [1e-6, 1e-6, 1e-6]
    return body


def add_film_bodies(spec, scene_model, fovy: float) -> None:
    """The third-person camera rig and the two ghost grippers, as free bodies parked under the floor."""
    cam = _parked_body(spec, RIG).add_camera(name=FILM_CAMERA, fovy=fovy)
    cam.resolution = [sim_video.WIDTH, sim_video.HEIGHT]
    rgba = [*ghost.GHOST_RGB, ghost.GHOST_ALPHA]
    spec.add_material(name=GHOST_MATERIAL, rgba=rgba)
    for side in ghost.SIDES:
        parts = ghost.hand_parts(scene_model, side)
        if not parts:
            raise SystemExit(f"the scene has no {side} gripper meshes to make a ghost of")
        body = _parked_body(spec, GHOSTS[side])
        for k, (mesh, rel_pos, rel_quat) in enumerate(parts):
            src = spec.mesh(mesh)
            copy = spec.add_mesh(name=f"{GHOSTS[side]}_{mesh}", file=src.file)
            copy.scale, copy.refpos, copy.refquat = src.scale, src.refpos, src.refquat
            g = body.add_geom(name=f"{GHOSTS[side]}_{k}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=copy.name)
            g.pos, g.quat, g.material, g.rgba = rel_pos.tolist(), rel_quat.tolist(), GHOST_MATERIAL, rgba
            g.contype = g.conaffinity = 0


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
    scene_xml = scene_dir / "scene_locomani.xml"
    scene_model = mujoco.MjModel.from_xml_path(str(scene_xml))
    fovy = third_person_shot.scene_fovy(scene_model, meta)
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    moving = moving_bodies(spec)
    strip_to_dynamic(spec, moving, scene_model)
    add_film_bodies(spec, scene_model, fovy)
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
    film = {"camera": FILM_CAMERA, "fovy": fovy, "size": [sim_video.WIDTH, sim_video.HEIGHT]}
    return {"xml": str(xml_path), "nq": model.nq, "nmesh": model.nmesh, "cameras": cams, "moving": moving, "film": film}
