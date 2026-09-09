"""Build the two-surface scene in Isaac Sim and render it with the real background.

This is the P3 renderer's ancestor: it proves the assets, the generated room and
the G1 can all be represented in USD with correct scale and pose. It does NOT
do the MuJoCo <-> Isaac per-frame state sync, which is P3's actual work.

Why not EmbodiedGen's USD converter: `cvt_embodiedgen_asset_to_anysim(...,
AssetType.USD, ...)` needs both isaacsim and embodied_gen importable in one
interpreter, and embodied_gen is not installed in the sonic env (it only ever
imported there because the cwd happened to be its repo). Isaac ships everything
needed anyway:

  * textured object OBJs -> omni.kit.asset_converter (handles the materials)
  * vertex-coloured background PLY -> a hand-built UsdGeom.Mesh, because the
    room has no texture, only per-vertex colours, which map to the
    `displayColor` primvar

Usage:
    OMNI_KIT_ACCEPT_EULA=YES python src/locomani/sim/isaac_scene.py <scene_root> <out_dir>
"""

from __future__ import annotations

import asyncio
import os
import sys

from locomani import paths  # noqa: E402

# abspath is not cosmetic. A relative OUT_DIR produces relative `references =`
# in scene.usda, which USD then resolves against the layer's own directory, so
# every asset silently fails to load and only the background renders. It also
# sends replicator's output to ~/omni.replicator_out/<relative path>.
SCENE_ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else (f"{paths.REPO}/outputs/scene"))
OUT_DIR = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else (f"{paths.REPO}/outputs/renders_isaac"))
os.makedirs(OUT_DIR, exist_ok=True)

BUILD_LOG = os.path.join(OUT_DIR, "build.log")
open(BUILD_LOG, "w").close()


def log(msg: str) -> None:
    """Isaac captures Python's stdout into its own logger and prints() vanish,
    so progress goes straight to a file instead."""
    with open(BUILD_LOG, "a") as f:
        f.write(f"{msg}\n")
        f.flush()


print = log  # noqa: A001 - deliberately redirect every print in this module

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

import numpy as np  # noqa: E402
import omni.kit.asset_converter as asset_converter  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import trimesh  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt  # noqa: E402

G1_USD = f"{paths.REPO}/third_party/GR00T-WholeBodyControl/gear_sonic/data/robots/g1/g1_29dof_with_hand_rev_1_0.usd"

# Same arrangement as locomani/sim/compose_scene.py — keep these in sync.
# Fallbacks only. The scene MuJoCo actually runs is rebuilt per batch from
# SCENE_SEED, and its surface tops move with it (0.62-0.65 m across seeds), so
# the render must take them from that batch's scene_meta.json or it puts the
# table at a different height from the physics. These constants are what gets
# used when no meta is available -- and they are already stale: the meta at the
# time of writing says 0.6277, this says 0.62, an 8 mm discrepancy that was
# silently in every render before 2026-09-05.
SURFACE_A_XY = (0.0, 0.0)
SURFACE_B_XY = (2.5, 0.0)
SURFACE_TARGET_TOP = {"dining_table": 0.62, "counter": 0.74}


def load_scene_meta(path: str = "", emit=print) -> dict:
    """Per-seed surface geometry from a scene_meta.json, {} if unavailable.

    Accepts either the meta file itself or a batch snapshot directory
    ($OUT/scene_<seed>/). Empty dict means "fall back to the constants", which
    is correct for the very first build and wrong for anything reproducible.
    """
    import json

    cand = path or os.environ.get("SCENE_META") or paths.SCENE_META
    if os.path.isdir(cand):
        cand = os.path.join(cand, "scene_meta.json")
    if not os.path.exists(cand):
        emit(f"  scene_meta: NOT FOUND at {cand} -- using built-in constants")
        return {}
    meta = json.load(open(cand))
    emit(f"  scene_meta: {cand}")
    return meta


def surface_plan(meta: dict):
    """-> ((key, name, xy), ...) and {name: top}, meta first, constants after."""
    tops = dict(SURFACE_TARGET_TOP)
    plan = [("A", "dining_table", SURFACE_A_XY), ("B", "counter", SURFACE_B_XY)]
    surf = meta.get("surfaces") or {}
    if not surf:
        return tuple(plan), tops
    out = []
    for key, name, xy in plan:
        spec = surf.get(key)
        if not spec:
            out.append((key, name, xy))
            continue
        name = spec.get("name", name)
        xy = tuple(spec.get("xy", xy))
        if spec.get("top") is not None:
            tops[name] = float(spec["top"])
        out.append((key, name, xy))
    return tuple(out), tops


ROBOT_START_XY = (-1.5, 0.0)
ROBOT_PELVIS_Z = 0.793
ON_SURFACE_A = ["orange", "fork", "napkin"]
ON_SURFACE_B = ["plate"]

# --- fitting the generated room to the layout -------------------------------
# The empty room reconstruction is 3.91 x 4.46 m with a 2.77 m ceiling, and a
# panorama's depth is only defined up to scale, so both the orientation and the
# size are ours to choose. Rotating +90 about z puts its long axis along world
# x, which is the axis the 4.0 m layout runs along; 1.15x then leaves ~0.4 m of
# clearance at each end (ceiling 3.19 m, still a plausible room).
#
#   +90 x  puts the y-up mesh z-up;  +90 z  then maps mesh (x, y, z) -> (z, x, y)
BG_ROTATE = (90.0, 0.0, 90.0)
BG_SCALE = 1.15
BG_SHIFT_XY = (-0.05, 0.25)  # centre the footprint on the layout, not the origin;
# x was 0.65: the back wall then sat at x -1.85, inside
# the crate pilots' 1.0-1.3 m walk-in spawn (pilot 21
# renders showed the robot walking out of the wall)

# Room lighting. Placed over the two work areas so both surfaces are lit; the
# ceiling is at ~3.19 m after BG_SCALE, so 2.7 m keeps the lamps clear of it.
CEILING_LIGHT_XY = ((-0.3, 0.0), (2.2, 0.0))
CEILING_LIGHT_Z = 2.70
CEILING_LIGHT_INTENSITY = 40000.0

# Floor-flattening parameters. FLOOR_CELL is the grid the bowl is measured on —
# coarse enough that each cell holds thousands of vertices, fine enough to
# follow a half-metre rise over two metres.
FLOOR_CELL = 0.25
FLOOR_MIN_PTS = 40
FLOOR_BLEND_TOP = 1.20  # height over which the correction fades to zero


def _rotate_xyz_matrix(deg) -> "np.ndarray":
    """R = Rz @ Ry @ Rx — the order UsdGeom's RotateXYZ op composes in."""
    rx, ry, rz = np.deg2rad(deg)
    cx, sx, cy, sy, cz, sz = (np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


def asset_usd_path(out_dir: str, name: str) -> str:
    """One directory per asset — the converter writes textures next to the USD
    under a fixed `textures/` subdirectory, and every EmbodiedGen asset names
    its texture `material_0.png`. Converting them all into one directory means
    the last conversion silently overwrites the rest, and the dining table ends
    up wearing the orange's skin."""

    return os.path.join(out_dir, "usd", name, f"{name}.usd")


def convert_obj_to_usd(obj_path: str, usd_path: str) -> bool:
    """omni.kit.asset_converter is async; drive it to completion synchronously."""
    os.makedirs(os.path.dirname(usd_path), exist_ok=True)
    if os.path.exists(usd_path):
        return True

    async def run():
        # The context is a class, not a factory function
        # (omni.kit.asset_converter exposes AssetConverterContext / get_instance).
        ctx = asset_converter.AssetConverterContext()
        ctx.ignore_materials = False
        ctx.ignore_animations = True
        task = asset_converter.get_instance().create_converter_task(obj_path, usd_path, None, ctx)
        return await task.wait_until_finished()

    ok = asyncio.get_event_loop().run_until_complete(run())
    if not ok:
        print(f"  convert FAILED: {obj_path}")

    return ok


def floor_height_field(v, cell: float = FLOOR_CELL):
    """Estimate the reconstructed floor's height on a coarse xy grid.

    A panorama's depth is least reliable straight down, so the reconstructed
    floor is a shallow bowl rather than a plane: measured on this room it sits
    at 0.01 m in the middle and rises to 0.46 m at x = -1.5 and 0.40 m at
    x = +2.5. That is not a tilt a plane fit can remove, and it is not
    cosmetic — the robot's feet and the counter's base end up buried in it,
    which is exactly the vision/physics mismatch an empty room was meant to
    avoid.

    Per cell, the 8th percentile of z is the floor: low enough to ignore
    anything standing on it, high enough to ignore stray reconstruction noise.
    Cells with too few points inherit from their neighbours by nearest lookup.
    """
    lo, hi = v.min(0)[:2], v.max(0)[:2]
    nx, ny = np.ceil((hi - lo) / cell).astype(int) + 1
    ix = np.clip(((v[:, 0] - lo[0]) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((v[:, 1] - lo[1]) / cell).astype(int), 0, ny - 1)
    flat = ix * ny + iy

    field = np.full(nx * ny, np.nan)
    order = np.argsort(flat, kind="stable")
    z_sorted, cells = v[order, 2], flat[order]
    starts = np.searchsorted(cells, np.arange(nx * ny), side="left")
    ends = np.searchsorted(cells, np.arange(nx * ny), side="right")
    for c, (a, b) in enumerate(zip(starts, ends)):
        if b - a >= FLOOR_MIN_PTS:
            field[c] = np.percentile(z_sorted[a:b], 8)

    field = field.reshape(nx, ny)
    known = ~np.isnan(field)
    if not known.any():
        return np.zeros(len(v))
    # Fill gaps from the nearest known cell so the lookup is defined everywhere.
    ky, kx = np.meshgrid(np.arange(ny), np.arange(nx))
    kpts = np.stack([kx[known], ky[known]], 1)
    kval = field[known]
    upts = np.stack([kx[~known], ky[~known]], 1)
    if len(upts):
        d = ((upts[:, None, :] - kpts[None, :, :]) ** 2).sum(-1)
        field[~known] = kval[d.argmin(1)]

    return field[ix, iy]


FLOOR_SNAP = 0.08  # after the warp, |z| below this is floor: snap to the plane


def flatten_floor(v):
    """Warp the bowl flat, keeping the mesh connected to its walls.

    Cutting the floor out and dropping in a plane would leave a gap all the way
    round where the dish used to meet the walls — up to half a metre of it.
    Warping instead moves each vertex down by the floor height under it, with
    the correction fading out by FLOOR_BLEND_TOP so the walls above are
    untouched and the junction just stretches slightly. Connectivity, colours
    and the wall geometry all survive.
    """
    f = floor_height_field(v)
    # 1 at floor level, 0 well up the wall, smooth in between.
    t = np.clip((FLOOR_BLEND_TOP - (v[:, 2] - f)) / FLOOR_BLEND_TOP, 0.0, 1.0)
    out = v.copy()
    out[:, 2] -= f * (t * t * (3 - 2 * t))  # smoothstep
    # The 8th-percentile field leaves the floor's own scatter above z=0: on
    # the crate scene the median under the robot's path was +2..+3.5 cm, p95
    # +7.5 cm, so the G1's feet (soles at z=0 in MuJoCo) rendered buried. The
    # real floor is a plane: press everything within FLOOR_SNAP of it onto it.
    floor = np.abs(out[:, 2]) < FLOOR_SNAP
    out[floor, 2] = 0.0
    print(f"  floor snap: {int(floor.sum())} of {len(out)} vertices within {FLOOR_SNAP:.2f} m pressed to z=0")
    return out


def add_vertex_coloured_mesh(stage, prim_path: str, ply_path: str, translate, rotate_xyz, scale=1.0, flatten=True):
    """The generated room carries colour on vertices, so it becomes displayColor.

    The rotate/scale/shift is baked into the point array rather than left to
    xform ops, because the floor warp below is defined in world coordinates and
    has to be applied to world-space vertices.
    """
    mesh = trimesh.load(ply_path, force="mesh")
    world = (np.asarray(mesh.vertices) * scale) @ _rotate_xyz_matrix(rotate_xyz).T
    world += np.asarray(translate, dtype=float)
    if flatten:
        before = world[:, 2].copy()
        world = flatten_floor(world)
        print(f"  floor warp: dropped up to {float((before - world[:, 2]).max()):.3f} m")

    prim = UsdGeom.Mesh.Define(stage, prim_path)
    # Vt.*.FromNumpy is the only sane path at this size: the room has ~2.2 M
    # vertices, and Gf.Vec3f rejects numpy scalars anyway, so building Python
    # objects per vertex would be both wrong and unusably slow.
    prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(world.astype(np.float32)))
    prim.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(mesh.faces), 3, dtype=np.int32)))
    prim.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(mesh.faces.astype(np.int32).ravel()))

    colours = getattr(mesh.visual, "vertex_colors", None)
    if colours is not None and len(colours):
        rgb = (np.asarray(colours)[:, :3] / 255.0).astype(np.float32)
        pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
        )
        pv.Set(Vt.Vec3fArray.FromNumpy(rgb))
        bind_unlit_vertex_colour_material(stage, prim, f"{prim_path}_Mat")
        print(f"  background: {len(mesh.vertices)} verts, unlit vertex colours")
    else:
        print(f"  background: {len(mesh.vertices)} verts, NO vertex colours")

    # No xform ops: the transform is already baked into the points above.
    return world


def bind_unlit_vertex_colour_material(stage, mesh_prim, mat_path: str) -> None:
    """Render the reconstructed room with its baked-in lighting, not relit.

    A panorama reconstruction already has the original room lighting baked into
    its vertex colours. Lighting it again with a dome light double-counts the
    illumination, which is why the first renders came out dark and muddy. Wiring
    displayColor into `emissiveColor` (with a black diffuse) makes the mesh
    display exactly the colours that were captured.
    """
    material = UsdShade.Material.Define(stage, mat_path)

    reader = UsdShade.Shader.Define(stage, f"{mat_path}/displayColorReader")
    reader.CreateIdAttr("UsdPrimvarReader_float3")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)

    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(reader.ConnectableAPI(), "result")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)

    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh_prim.GetPrim())
    UsdShade.MaterialBindingAPI(mesh_prim.GetPrim()).Bind(material)


def add_reference(stage, prim_path: str, usd_path: str, translate, rotate_xyz=(0.0, 0.0, 0.0), scale=1.0):
    """Place a referenced USD under a clean wrapper Xform.

    The converted assets already carry [xformOp:translate, orient, scale], so
    adding ops directly to the referencing prim raises "xformOp already exists".
    Referencing on a child and transforming the parent avoids the clash
    entirely.
    """
    wrapper = UsdGeom.Xform.Define(stage, prim_path)
    ops = (wrapper.AddTranslateOp(), wrapper.AddRotateXYZOp(), wrapper.AddScaleOp())
    ops[0].Set(Gf.Vec3d(*translate))
    ops[1].Set(Gf.Vec3f(*rotate_xyz))
    ops[2].Set(Gf.Vec3f(scale, scale, scale))

    child = stage.DefinePrim(f"{prim_path}/asset", "Xform")
    # A reference with no target prim path resolves to the layer's defaultPrim,
    # and a layer without one composes to *nothing* — silently, with no error
    # and no geometry. Naming the root prim explicitly is the only reliable
    # form; the G1 USD is one of the layers that has no defaultPrim.
    layer = Sdf.Layer.FindOrOpen(usd_path)
    target = Sdf.Path()
    if layer is not None and not layer.defaultPrim:
        roots = [p for p in layer.rootPrims]
        if roots:
            target = Sdf.Path(f"/{roots[0].name}")
            print(f"    {os.path.basename(usd_path)}: no defaultPrim, using {target}")
    child.GetReferences().AddReference(usd_path, target)

    return wrapper.GetPrim(), ops


def rotated_bounds(mesh_bounds, rotate_xyz):
    """World AABB of a mesh AABB after the wrapper's rotation."""
    lo, hi = np.asarray(mesh_bounds[0]), np.asarray(mesh_bounds[1])
    corners = np.array(np.meshgrid(*zip(lo, hi))).reshape(3, -1).T
    w = corners @ _rotate_xyz_matrix(rotate_xyz).T

    return w.min(0), w.max(0)


def place_asset(
    stage,
    prim_path: str,
    usd_path: str,
    mesh_bounds,
    xy,
    bottom_z: float,
    rotate_xyz=(90.0, 0.0, 0.0),
    target_height: float | None = None,
):
    """Reference an asset, then size and seat it from its *source mesh* bounds.

    The size comes from the OBJ rather than the stage because the OBJ is also
    what MuJoCo measures, so using it is what keeps the two engines agreeing.

    Op order is translate, rotate, scale, so USD applies scale first: a local
    bound B becomes s * B, and the translate that lands the asset centred on
    (x, y) and resting on bottom_z follows directly.

    The returned `actual` is the world bound USD itself reports afterwards, with
    `useExtentsHint=False` so it reflects real geometry rather than the hint the
    converter wrote. When it disagrees with the intended box, the reference did
    not compose the way the source mesh says it should.
    """
    lo, hi = rotated_bounds(mesh_bounds, rotate_xyz)
    size = hi - lo
    s = 1.0 if not target_height or size[2] <= 0 else target_height / size[2]

    prim, ops = add_reference(stage, prim_path, usd_path, (0.0, 0.0, 0.0), rotate_xyz, s)
    ops[0].Set(
        Gf.Vec3d(
            float(xy[0] - s * 0.5 * (lo[0] + hi[0])),
            float(xy[1] - s * 0.5 * (lo[1] + hi[1])),
            float(bottom_z - s * lo[2]),
        )
    )

    make_double_sided(stage, prim)
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=False
    )
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    actual = ([round(float(v), 3) for v in rng.GetMin()], [round(float(v), 3) for v in rng.GetMax()])

    return prim, [round(float(v) * s, 3) for v in size], actual


def make_double_sided(stage, root_prim) -> int:
    """Render generated meshes from both sides.

    Meshes reconstructed from images routinely have inconsistent face winding,
    and Isaac culls backfaces by default while MuJoCo does not — so the same
    asset can look complete in one engine and partly missing in the other. The
    counter is the case in point: its legs and lower cabinet are inward-facing,
    so it rendered as a box hovering 45 cm above the floor while its bounds,
    its physics and the MuJoCo render all said it reached the ground.
    """
    n = 0
    for prim in Usd.PrimRange(root_prim):
        gprim = UsdGeom.Gprim(prim)
        if gprim:
            gprim.CreateDoubleSidedAttr(True)
            n += 1

    return n


def obj_and_height(scene_root: str, name: str):
    """Return (obj_path, vertical size in m, mesh AABB) for a generated asset."""
    for sub in ("layout", "surface_b"):
        base = f"{scene_root}/{sub}/asset3d/{name}/result"
        obj = f"{base}/mesh/{name}.obj"
        if os.path.exists(obj):
            m = trimesh.load(obj, force="mesh")
            # meshes are exported y-up; we rotate +90 about x when placing
            return obj, float(m.extents[1]), m.bounds

    return None, 0.0, None


def _mjcf_texture_for(mesh_name: str, xml: str):
    """<geom mesh=..> -> material -> texture -> file, as MuJoCo resolves it."""
    import re

    g = re.search(r'<geom[^>]*material="([^"]+)"[^>]*mesh="' + re.escape(mesh_name) + r'"', xml) or re.search(
        r'<geom[^>]*mesh="' + re.escape(mesh_name) + r'"[^>]*material="([^"]+)"', xml
    )
    if not g:
        return None
    m = re.search(r'<material[^>]*name="' + re.escape(g.group(1)) + r'"[^>]*texture="([^"]+)"', xml)
    if not m:
        return None
    t = re.search(r'<texture[^>]*name="' + re.escape(m.group(1)) + r'"[^>]*file="([^"]+)"', xml)
    return t.group(1) if t else None


def _bind_mjcf_textures(top_usd: str, name: str, parts, xml: str) -> None:
    """Override each referenced part's material with the MJCF's own texture."""
    from pxr import Sdf, UsdShade

    st = Usd.Stage.Open(top_usd)
    if st is None:
        log(f"  object {name}: cannot reopen {top_usd} to bind textures")
        return
    for i, (mesh_name, _obj, _sc) in enumerate(parts):
        tex = _mjcf_texture_for(mesh_name + "_vis", xml)
        prim = st.GetPrimAtPath(f"/{name}/part_{i}")
        if tex is None or not os.path.exists(tex) or not prim.IsValid():
            log(f"  object {name} part {i}: no MJCF texture ({tex})")
            continue
        mp = Sdf.Path(f"/{name}/Looks/part_{i}_mat")
        mat = UsdShade.Material.Define(st, mp)
        sh = UsdShade.Shader.Define(st, mp.AppendChild("shader"))
        sh.CreateIdAttr("UsdPreviewSurface")
        uv = UsdShade.Shader.Define(st, mp.AppendChild("uv"))
        uv.CreateIdAttr("UsdPrimvarReader_float2")
        uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tx = UsdShade.Shader.Define(st, mp.AppendChild("tex"))
        tx.CreateIdAttr("UsdUVTexture")
        tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(os.path.abspath(tex))
        # MuJoCo repeats textures; USD's default wrap ("useMetadata") falls
        # back to BLACK outside [0, 1] when the image carries no wrap metadata.
        # The crate's UVs run to 1.57, so it rendered black (2026-09-07).
        tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), "result")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
        log(f"  object {name} part {i}: bound {os.path.basename(tex)}")
    st.GetRootLayer().Save()


def convert_target_objects(meta: dict, scene_xml: str, out_dir: str) -> None:
    """Convert the MANIPULABLE objects (the ones MuJoCo moves) to USD.

    They are not placed here: the renderer references them and sets a world
    pose per frame from the sidecar. All this does is make the asset exist --
    which it previously did not. Everything rendered before 2026-09-05 carried
    a "WARNING no bottle usd" line and simply had no target object in frame.

    The mesh scale is read from the scene XML rather than assumed: the MJCF
    applies it (0.16 for bottled_water_1) and the raw OBJ is unscaled, so the
    renderer has to reapply it. It is written next to the USD so the render
    side does not have to hard-code a number that is per-asset.
    """
    import json as _json
    import re

    xml = open(scene_xml).read()
    meshes = re.findall(
        r'<mesh name="([^"]+)_vis"[^>]*file="([^"]+)"'
        r'[^>]*scale="([^"]+)"',
        xml,
    )
    for name in meta.get("objects") or {}:
        parts = [(n, f, sc) for n, f, sc in meshes if n.startswith(name + "_")]
        if not parts:
            log(f"  object {name}: no _vis mesh in the scene XML, skipped")
            continue
        dst_dir = os.path.join(out_dir, "usd", name)
        os.makedirs(dst_dir, exist_ok=True)
        top = os.path.join(dst_dir, f"{name}.usd")
        refs = []
        for i, (_, obj_path, _sc) in enumerate(parts):
            if not os.path.exists(obj_path):
                log(f"  object {name}: missing OBJ {obj_path}")
                continue
            part = os.path.join(dst_dir, f"part_{i}.usd")
            if convert_obj_to_usd(obj_path, part):
                refs.append(part)
        if not refs:
            log(f"  object {name}: nothing converted")
            continue
        if os.path.exists(top):
            os.remove(top)
        st = Usd.Stage.CreateNew(top)
        UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(st, 1.0)
        root = st.DefinePrim(f"/{name}", "Xform")
        st.SetDefaultPrim(root)
        for i, r in enumerate(refs):
            pp = st.DefinePrim(f"/{name}/part_{i}", "Xform")
            pp.GetReferences().AddReference("./" + os.path.basename(r))
        st.GetRootLayer().Save()
        # Per-part texture from the MJCF, NOT from the OBJ's material.mtl.
        # bottled_water_1 ships one MTL shared by both OBJs pointing at
        # material_0.png, which is byte-identical to image0.png (the cap). The
        # MJCF instead binds the body to image1.png and the cap to image0.png.
        # Trusting the MTL renders the whole bottle in the cap's flat green --
        # geometry aligned, appearance wrong, which for a VLA dataset is the
        # part that matters. Caught 2026-09-06 by the ticket-05 side-by-side.
        _bind_mjcf_textures(top, name, parts, xml)
        # Full scale vector (the crate is 0.667 in x/y and 1.0 in z) and the
        # body->geom offset the MJCF applies to the visual mesh (the crate's
        # mesh sits at pos "0 0 -0.085" under its body). The renderer composes
        # them under the object's world pose. A scalar "scale" is still
        # accepted downstream for the older bottle json.
        scale = [float(v) for v in parts[0][2].split()]
        if len(scale) == 1:
            scale = scale * 3
        offset_pos, offset_quat = [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        mg = re.search(r'<geom([^>]*)mesh="' + re.escape(parts[0][0]) + r'_vis"', xml)
        if mg is not None:
            attrs = mg.group(1)
            mp_ = re.search(r'pos="([^"]+)"', attrs)
            mq_ = re.search(r'quat="([^"]+)"', attrs)
            if mp_:
                offset_pos = [float(v) for v in mp_.group(1).split()]
            if mq_:
                offset_quat = [float(v) for v in mq_.group(1).split()]
        _json.dump(
            {"scale": scale, "offset_pos": offset_pos, "offset_quat": offset_quat, "parts": len(refs)},
            open(os.path.join(dst_dir, f"{name}.json"), "w"),
        )
        log(f"  object {name}: {len(refs)} part(s), scale {scale}, offset {offset_pos} -> {top}")


def main() -> None:
    import json

    layout = json.load(open(f"{SCENE_ROOT}/layout/layout.json"))
    usda = f"{OUT_DIR}/scene.usda"
    if os.path.exists(usda):
        os.remove(usda)  # CreateNew refuses to overwrite
    stage = Usd.Stage.CreateNew(usda)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.DefinePrim("/World", "Xform")

    # Lights go *inside* the room, not around it. A DomeLight is an
    # infinite-distance environment light, so a closed panorama shell blocks it
    # completely: with only a dome, the background rendered correctly (it is
    # emissive) while every real object came out black. Sphere lights near the
    # ceiling are what actually illuminates the scene; the dome is a weak fill
    # that leaks through the reconstructed windows.
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr(300.0)
    for i, (lx, ly) in enumerate(CEILING_LIGHT_XY):
        lamp = UsdLux.SphereLight.Define(stage, f"/World/CeilingLight_{i}")
        lamp.CreateRadiusAttr(0.25)
        lamp.CreateIntensityAttr(CEILING_LIGHT_INTENSITY)
        lamp.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
        UsdGeom.Xformable(lamp).AddTranslateOp().Set(Gf.Vec3d(float(lx), float(ly), CEILING_LIGHT_Z))

    # --- background ---------------------------------------------------------
    # Prefer the purpose-generated empty room. The room that came with the
    # layout is a fully furnished kitchen: its own counters and cabinets exist
    # only in the render, not in MuJoCo, so the robot would walk straight
    # through furniture it can see. An empty shell keeps vision and physics
    # consistent, with every object present in both.
    ply = f"{SCENE_ROOT}/bg_empty/scene_0000/mesh_model.ply"
    if not os.path.exists(ply):
        ply = f"{SCENE_ROOT}/layout/background/mesh_model.ply"
    print(f"  background source: {ply}")
    if os.path.exists(ply):
        # The whole transform has to be known before the points are written,
        # because the floor warp operates on world-space vertices. Derive the
        # z-shift that puts the mesh's lowest point on 0 from the raw bounds,
        # using the actual rotation matrix rather than a hand-written axis
        # mapping, so changing BG_ROTATE cannot silently invalidate it.
        raw = trimesh.load(ply, force="mesh").bounds
        corners = np.array(np.meshgrid(*zip(raw[0], raw[1]))).reshape(3, -1).T
        rotated = (corners * BG_SCALE) @ _rotate_xyz_matrix(BG_ROTATE).T
        translate = (BG_SHIFT_XY[0], BG_SHIFT_XY[1], -float(rotated.min(0)[2]))

        room = add_vertex_coloured_mesh(
            stage, "/World/Background", ply, translate=translate, rotate_xyz=BG_ROTATE, scale=BG_SCALE
        )

        # Measured from the warped vertices, not predicted from the raw bounds:
        # flattening the floor moves geometry, so anything derived beforehand
        # would describe a room that is no longer there.
        wlo, whi = room.min(0), room.max(0)

        # Cameras must sit inside this box — the room is a single-sided shell
        # and any viewpoint outside it sees straight through the backfaces.
        room_box = dict(
            x=(float(wlo[0]), float(whi[0])), y=(float(wlo[1]), float(whi[1])), z=(float(wlo[2]), float(whi[2]))
        )
        print(f"  room world box: x={room_box['x']} y={room_box['y']} z={room_box['z']}")

        span_needed = (ROBOT_START_XY[0] - 0.4, SURFACE_B_XY[0] + 0.4)
        if wlo[0] > span_needed[0] or whi[0] < span_needed[1]:
            print(
                f"  WARNING: room x {room_box['x']} does not cover the layout "
                f"span {span_needed} — raise BG_SCALE or retune BG_SHIFT_XY"
            )
        globals()["ROOM_BOX"] = room_box

    # --- surfaces -----------------------------------------------------------
    surfaces = {}
    meta = load_scene_meta(emit=log)
    plan, tops = surface_plan(meta)
    for key, name, xy in plan:
        obj, h, bounds = obj_and_height(SCENE_ROOT, name)
        if obj is None:
            print(f"  MISSING surface {key}: {name}")
            continue
        usd = asset_usd_path(OUT_DIR, name)
        if not convert_obj_to_usd(obj, usd):
            continue
        target = tops[name]
        _, size, actual = place_asset(stage, f"/World/{name}", usd, bounds, xy, 0.0, target_height=target)
        surfaces[key] = dict(xy=xy, top=target)
        log(f"  surface {key}: {name:13s} top={target:.3f} planned size={size} usd bbox={actual}")

    # --- objects ------------------------------------------------------------
    for obj_name, pose in layout["position"].items():
        name = obj_name.replace(" ", "_")
        if name in ("dining_table", "kitchen", "franka"):
            continue
        key = "A" if obj_name in ON_SURFACE_A else "B" if obj_name in ON_SURFACE_B else None
        if key is None or key not in surfaces:
            continue
        obj, h, bounds = obj_and_height(SCENE_ROOT, name)
        if obj is None:
            continue
        usd = asset_usd_path(OUT_DIR, name)
        if not convert_obj_to_usd(obj, usd):
            continue
        surf = surfaces[key]
        x, y = (pose[0] + surf["xy"][0], pose[1] + surf["xy"][1]) if key == "A" else surf["xy"]
        # target_height keeps the render honest against MuJoCo, where the same
        # asset is scaled to the same measured extent.
        _, size, actual = place_asset(stage, f"/World/{name}", usd, bounds, (x, y), surf["top"], target_height=h)
        print(
            f"  {obj_name:12s} -> {key} at ({x:.3f}, {y:.3f}) on {surf['top']:.3f} "
            f"planned size={size} usd bbox={actual}"
        )

    # --- manipulable objects (converted, not placed) -------------------------
    sx = os.environ.get("SCENE_XML") or paths.SCENE_XML
    if meta and os.path.exists(sx):
        convert_target_objects(meta, sx, OUT_DIR)
    else:
        log(f"  target objects: skipped (meta={bool(meta)} scene_xml={sx})")

    # --- robot --------------------------------------------------------------
    if os.path.exists(G1_USD):
        # The G1 USD is authored in metres at the right scale already, so it is
        # positioned rather than fitted: pelvis at the episode start height.
        g1, _ = add_reference(
            stage,
            "/World/G1",
            G1_USD,
            (ROBOT_START_XY[0], ROBOT_START_XY[1], ROBOT_PELVIS_Z),
            rotate_xyz=(0.0, 0.0, 0.0),
        )
        make_double_sided(stage, g1)
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=False
        )
        rng = cache.ComputeWorldBound(g1).ComputeAlignedRange()
        # An empty range means the reference composed to nothing — the usual
        # cause is a layer whose defaultPrim is missing or is not the geometry
        # root. Report it rather than rendering a scene with no robot in it.
        if rng.IsEmpty():
            print(f"  WARNING: G1 reference composed to nothing: {G1_USD}")
        else:
            print(
                f"  G1 at {ROBOT_START_XY} z={ROBOT_PELVIS_Z} bbox="
                f"{[round(float(v), 3) for v in rng.GetMin()]}.."
                f"{[round(float(v), 3) for v in rng.GetMax()]}"
            )
    else:
        print(f"  WARNING: G1 USD not found: {G1_USD}")

    stage.GetRootLayer().Save()
    print(f"  stage saved: {OUT_DIR}/scene.usda")

    # --- render -------------------------------------------------------------
    import omni.usd

    omni.usd.get_context().open_stage(f"{OUT_DIR}/scene.usda")
    for _ in range(60):
        app.update()

    # Auto-exposure makes renders incomparable between runs, and here it is
    # actively wrong: the emissive background dominates the histogram, so the
    # lit objects get crushed to black. Fix the exposure instead.
    import carb

    carb.settings.get_settings().set("/rtx/post/histogram/enabled", False)

    # The room is a panorama reconstruction: a single-sided shell. A camera
    # outside it sees straight through the backfaces, so every viewpoint has to
    # sit inside.
    box = globals().get("ROOM_BOX")
    if box:
        cy = 0.5 * (box["y"][0] + box["y"][1])
        y_near = box["y"][0] + 0.35  # a clear margin off the near wall
        x_near = box["x"][0] + 0.25
    else:
        cy, y_near, x_near = 0.0, -2.0, -1.7

    views = {
        # The whole 4 m corridor: robot on the left, both surfaces to the right.
        "overview": ((x_near, y_near, 1.95), (1.0, cy, 0.70)),
        # Roughly what the robot sees at the episode start.
        "from_robot": ((ROBOT_START_XY[0], cy, 1.35), (SURFACE_B_XY[0], cy, 0.90)),
        # Close on the pick-and-place, to check object seating on the tops.
        "surfaces": ((0.6, y_near + 0.5, 1.80), (1.4, cy, 0.82)),
    }
    for name, (pos, look) in views.items():
        cam = rep.create.camera(position=pos, look_at=look)
        rp = rep.create.render_product(cam, (1280, 960))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=os.path.join(OUT_DIR, name), rgb=True)
        writer.attach([rp])
        rep.orchestrator.step()
        for _ in range(15):
            app.update()
        writer.detach()
        print(f"  rendered {name} from {pos}")


try:
    log("=== build start ===")
    main()
    log("=== build OK ===")
except BaseException:
    import traceback

    log("=== build FAILED ===")
    log(traceback.format_exc())
finally:
    app.close()
