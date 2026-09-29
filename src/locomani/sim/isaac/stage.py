"""The Isaac half: build the mirror stage, push a frame into it, grab a frame.

Every isaacsim / pxr / carb import lives inside a function on purpose. A
module-level ``import isaacsim`` boots Kit, which takes minutes, and would make
this module unimportable from anything that only wants to read the signatures.

The offline re-renderer (``isaac_render``) and the future live mirror
(``isaac_obs_server``, ticket 08) both compose exactly these functions, so the
two can never drift into rendering different pixels from the same state.

Nothing here is a physics world. Gravity is zeroed and each frame is posed and
then stepped once purely so fabric syncs the new transforms into the render
pipeline -- ``render()`` alone never pulls from the physics view, which is what
three identical frames taught us.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from locomani import paths


@dataclass
class StageCfg:
    """Everything the stage assembly needs that is not per-frame state.

    The asset layout is the one scripts/build_isaac_assets.sh produces:

        $ISAAC_ASSETS/robot/g1_dex1.usda          one build, scene-independent
        $ISAAC_ASSETS/scene/<label>/scene.usda    one build per scene_meta hash
        $ISAAC_ASSETS/scene/<label>/usd/<obj>/    the objects that scene refs

    ``scene_dir`` picks which scene; the label is normally the batch's seed, so
    re-rendering an episode means pointing this at that batch's build.
    """

    assets: str = paths.ISAAC_ASSETS
    scene_dir: str = ""  # default: <assets>/scene/work
    robot_usd: str = ""  # default: <assets>/robot/g1_dex1.usda
    robot_mjcf: str = ""  # colours come from here; default paths.SCENE_XML
    width: int = 640
    height: int = 480
    torso_prim: str = "/World/scene/torso_link"
    # A cam-traj puts the ego camera at stage top level with an ABSOLUTE world
    # pose per frame. Parented under torso_link, set_world_pose back-computes a
    # local offset against the parent's pre-step (stale) transform and the view
    # ends up ~1 m off the head.
    absolute_camera: bool = False
    # Visual prims (under /visuals) to hide before rendering. The head camera
    # sits INSIDE head_link -- 1.5 cm behind its front face, inside its bbox
    # on every axis -- so the only thing that mesh can contribute to the ego
    # image is its own interior. MuJoCo never shows it because its renderer
    # culls back faces; RTX does not cull (the /rtx/hydra/faceCulling flag is
    # ignored, tried before and after stage load), and from inside a closed
    # mesh every face is a back face, so hiding the mesh is exactly MuJoCo's
    # result. Left visible it covered the lower right quarter of the frame as
    # a grey dome (2026-09-07, once the reference split let it compose at all).
    hidden_visuals: tuple = ("torso_link/head_link_visual",)
    # Cameras render at this multiple of their resolution and `downsample` box-averages back. RTX realtime
    # anti-aliases with FXAA only (1 spp): small print blurs and shelf grids shimmer. 2x against the Blender
    # stills: structural similarity 0.89-0.94 -> 0.93-0.96 (docs/ledger/blender_scenes.md section 3). Per scene,
    # `isaac.supersample` in configs/scenes.yaml.
    supersample: int = 1


def downsample(image: np.ndarray, factor: int) -> np.ndarray:
    """Box-average an HxWxC uint8 image rendered at `factor` times its size back down."""
    if factor == 1:
        return image
    h, w, c = image.shape
    if h % factor or w % factor:
        raise ValueError(f"{w}x{h} is not a multiple of the supersample factor {factor}")
    blocks = image.reshape(h // factor, factor, w // factor, factor, c).astype(np.float32)
    return np.rint(blocks.mean(axis=(1, 3))).astype(np.uint8)


def start_app(cfg: StageCfg, headless: bool = True):
    """Boot Kit. Minutes on a cold TensorRT/shader cache."""
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp

    return SimulationApp({"headless": headless, "width": cfg.width, "height": cfg.height})


# abspath is not cosmetic, and this is the second time it has bitten this repo
# (isaac_scene.py carries the same warning). USD resolves a reference against
# the LAYER's directory, not the process cwd, so a relative --scene-dir turns
# `.../usd/water_bottle/water_bottle.usd` into a path under the scene layer's
# own folder, finds nothing, and composes an EMPTY prim. No error, no warning:
# the object simply is not in the picture. Caught 2026-09-06 after a full
# episode rendered with no bottle while the log said "bottle referenced".
def scene_dir(cfg: StageCfg) -> str:
    return os.path.abspath(cfg.scene_dir or os.path.join(cfg.assets, "scene", "work"))


def robot_usd(cfg: StageCfg) -> str:
    return os.path.abspath(cfg.robot_usd or os.path.join(cfg.assets, "robot", "g1_dex1.usda"))


def _mjcf_multi_mesh_bodies(mjcf_path: str) -> set:
    """Bodies whose visual geometry is more than one mesh.

    The importer composes all of a body's visual geoms onto ONE prim and writes
    a single `material:binding` there. That local opinion beats anything the
    referenced prototypes carry, so those links can only ever show one colour.
    Read from the MJCF rather than from USD metadata: the robot layer is
    sublayered into the scene stage and the composed prims do not report their
    references at runtime.
    """
    import re

    out, cur = {}, None
    for m in re.finditer(r'<body name="([^"]+)"|<geom([^>]*)>', xml_text(mjcf_path)):
        if m.group(1):
            cur = m.group(1)
            out.setdefault(cur, 0)
        elif cur and 'group="1"' in (m.group(2) or "") and 'mesh="' in (m.group(2) or ""):
            out[cur] = out.get(cur, 0) + 1
    return {k for k, v in out.items() if v > 1}


_XML_CACHE: dict = {}


def xml_text(path: str) -> str:
    if path not in _XML_CACHE:
        try:
            _XML_CACHE[path] = open(path).read()
        except Exception:  # noqa: BLE001
            _XML_CACHE[path] = ""
    return _XML_CACHE[path]


def _mjcf_mesh_colours(mjcf_path: str, log=print) -> dict:
    """{mesh asset name: (r, g, b)} from the MJCF's own geom rgba, plus
    {body name: (r, g, b)} for bodies whose visual meshes share one colour.

    Visual geoms win over collision ones when a mesh appears in both, and a
    fully transparent rgba is ignored -- those are the alpha-0 collision
    proxies, and taking their colour would paint a visible mesh with an
    invisible geom's value.

    The body keys exist because the importer names a link's visual prim after
    the BODY: /visuals/left_dex1_finger_1_link references the mesh asset
    Dex1_finger_link_1, so a lookup by mesh name alone missed all four Dex1
    fingers and they rendered in the near-white fallback (2026-09-29).
    """
    import re

    out: dict = {}
    body_rgb: dict = {}
    try:
        xml = open(mjcf_path).read()
    except Exception as e:  # noqa: BLE001
        log(f"WARNING no MJCF colours ({mjcf_path}): {e}")
        return out
    body = None
    for g in re.finditer(r'<body name="([^"]+)"|<geom[^>]*>', xml):
        if g.group(1):
            body = g.group(1)
            continue
        a = g.group(0)
        mesh = re.search(r'mesh="([^"]+)"', a)
        rgba = re.search(r'rgba="([^"]+)"', a)
        if not (mesh and rgba):
            continue
        vals = [float(v) for v in rgba.group(1).split()]
        if len(vals) >= 4 and vals[3] <= 0.0:
            continue
        visual = 'group="1"' in a
        name = mesh.group(1)
        if name not in out or (visual and not out[name][1]):
            out[name] = (tuple(vals[:3]), visual)
        if visual and body:
            body_rgb.setdefault(body, set()).add(tuple(vals[:3]))
    colours = {k: v[0] for k, v in out.items()}
    for b, rgbs in body_rgb.items():
        if b not in colours and len(rgbs) == 1:
            colours[b] = next(iter(rgbs))
    return colours


# The G1's two materials, copied from Unitree's own Isaac Lab asset for the G1
# with Dex1 (huggingface.co/datasets/unitreerobotics/unitree_sim_isaaclab_usds,
# assets/robots/g1-29dof_wholebody_dex1/g1_29dof_with_dex1_rev_1_0.usd:
# DefaultMaterial on the shells, material_CAD1EE on the dark parts). Its dark
# set -- head, pelvis, hip pitch, ankle roll, Dex1 base and fingers -- is the
# MJCF's rgba < 0.4 set, so the MJCF colour picks the material.
# The MJCF rgba itself is not usable as a colour here: it is a display value
# (MuJoCo applies no gamma), which RTX reads as linear, so its 0.2 came out
# as 0.48 and the head, joints and gripper rendered mid grey (2026-09-29).
# OmniPBR multiplies diffuse_color_constant by diffuse_tint: the shells are
# 0.68 linear brushed metal and the dark parts 0.011, near black. The user
# chose these over the same albedo as non-metallic paint (2026-09-29,
# outputs/isaac_g1_colour_20260929/g1_looks_compare.jpg); being metal, the
# shells take on the room, so they read darker against the warehouse's dark
# floor and shelves. Unreal uses the same values
# (scripts/experiments/unreal_robot_materials_20260928.py, preset "unitree").
G1_LOOKS = {
    "shell": {
        "diffuse_color_constant": (0.75, 0.75, 0.76),
        "diffuse_tint": (0.9, 0.9, 0.91),
        "metallic_constant": 1.0,
        "reflection_roughness_constant": 0.75,
        "specular_level": 0.25,
    },
    "dark": {
        "diffuse_color_constant": (0.05796, 0.0587, 0.06178),
        "diffuse_tint": (0.18533, 0.16887, 0.16887),
        "metallic_constant": 0.42,
    },
}


def assemble_stage(cfg: StageCfg, log=print):
    """Open the base stage and compose room + robot + bottle onto it."""
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

    base = os.path.join(scene_dir(cfg), "scene.usda")
    rob = robot_usd(cfg)
    for path, what in ((base, "scene"), (rob, "robot")):
        if not os.path.exists(path):
            raise SystemExit(
                f"missing {what} USD: {path}\n"
                f"  build it: bash scripts/build_isaac_assets.sh "
                f"[--snapshot <batch scene dir>]"
            )
    ctx = omni.usd.get_context()
    ctx.open_stage(base)
    stage = ctx.get_stage()
    # open_stage returns nothing on failure and the next call dies with a bare
    # AttributeError on None; say which file instead.
    if stage is None:
        raise SystemExit(f"omni.usd could not open {base}")
    log(f"scene stage: {base}")

    # Hide the orange-layout movables baked into the base stage; the tables
    # and the room stay.
    for name in ("plate", "fork", "napkin", "orange"):
        prim = stage.GetPrimAtPath(f"/World/{name}")
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
            log(f"hid /World/{name}")
    # The scene builder's /World/G1 placeholder (the SONIC G1 USD at the start
    # pose, for the overview snapshots) is a live, undriven PhysX articulation.
    # Out of the ego camera's view it went unnoticed; a tracking shot showed a
    # second robot which the floor contact then launched (gravity is off).
    # Deactivate: gone from the render AND from physics.
    g1 = stage.GetPrimAtPath("/World/G1")
    if g1.IsValid():
        g1.SetActive(False)
        log("deactivated /World/G1 placeholder")

    # Robot (MJCF-imported articulation).  Sublayer, not reference: the
    # importer puts the mesh payloads in ROOT-level /meshes /visuals
    # /collisions scopes and the body prims reference them by root path, so
    # referencing only /World/scene silently drops every mesh (first light
    # rendered a scene with an invisible robot this way).  Sublayering merges
    # the whole layer; the robot lands at its authored /World/scene path.
    stage.GetRootLayer().subLayerPaths.append(rob)
    log(f"robot layer: {rob}")

    # The MJCF importer leaves the STL visuals with no usable material -- they
    # render near-black -- so the renderer has to supply one. It used to supply
    # ONE flat neutral PBR for every mesh on the robot, which threw away the
    # MJCF's per-geom rgba: wrist 0.7 grey, the dex1 housing 0.25, the fingers
    # 0.3, the pelvis 0.2. Everything came out the same near-white, and the
    # dark housing that visually joins the forearm to the jaws vanished against
    # a bright room -- reported as "the connection between the arm and the
    # gripper disappeared", which is what it looks like. The MJCF's rgba now
    # decides which of Unitree's two materials a mesh gets (G1_LOOKS).
    colours = _mjcf_mesh_colours(cfg.robot_mjcf or paths.SCENE_XML, log)
    mats, n_bound, n_default = {}, 0, 0

    def material_for(rgb):
        key = "shell" if sum(rgb) / 3 >= 0.6 else "dark"
        if key not in mats:
            mp = Sdf.Path(f"/World/Looks/g1_{key}")
            mt = UsdShade.Material.Define(stage, mp)
            sh = UsdShade.Shader.Define(stage, mp.AppendChild("shader"))
            sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
            sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
            for name, value in G1_LOOKS[key].items():
                vtype = Sdf.ValueTypeNames.Color3f if isinstance(value, tuple) else Sdf.ValueTypeNames.Float
                sh.CreateInput(name, vtype).Set(Gf.Vec3f(*value) if isinstance(value, tuple) else value)
            mt.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
            mats[key] = mt
        return mats[key]

    # Bind on the referenced PROTOTYPES first. A link with several visual
    # meshes is one prim under /visuals with one child per mesh (the build's
    # split_visual_refs step), each child referencing its /meshes prototype;
    # a binding on the prototype inherits through the reference, so each mesh
    # keeps its own colour -- the wrist 0.7, the dex1 housing 0.25.
    # The importer writes a direct `material:binding` on the link prim itself.
    # That is a LOCAL opinion and beats anything inherited through a reference,
    # so it has to be cleared on the multi-mesh links or every child would show
    # the link's single (unresolvable /Tmp/Looks) binding.
    multi = _mjcf_multi_mesh_bodies(cfg.robot_mjcf or paths.SCENE_XML)
    merged, n_unbound = set(), 0
    for prim in stage.Traverse():
        if prim.GetName() not in multi:
            continue
        merged.add(str(prim.GetPath()))
        # Unconditionally: the importer's binding points at /Tmp/Looks/... which
        # does not resolve in the composed stage, so GetMaterial() comes back
        # invalid while the relationship is still there blocking inheritance.
        try:
            if prim.HasRelationship("material:binding"):
                UsdShade.MaterialBindingAPI(prim).UnbindDirectBinding()
                n_unbound += 1
        except Exception as e:  # noqa: BLE001
            log(f"  could not unbind {prim.GetPath()}: {e}")
    log(f"multi-mesh bodies {sorted(multi)}; cleared {n_unbound} link bindings")

    n_proto = 0
    for scope in ("/meshes", "/visuals"):
        root = stage.GetPrimAtPath(scope)
        if not root.IsValid():
            continue
        for proto in root.GetChildren():
            rgb = colours.get(proto.GetName())
            if rgb is None:
                continue
            UsdShade.MaterialBindingAPI.Apply(proto).Bind(material_for(rgb), UsdShade.Tokens.strongerThanDescendants)
            n_proto += 1
    log(f"bound {n_proto} mesh prototypes from the MJCF's geom rgba")

    DEFAULT_RGB = (0.82, 0.82, 0.85)
    for p in stage.Traverse():
        if p.GetTypeName() != "Mesh" or not str(p.GetPath()).startswith(("/World/scene", "/meshes", "/visuals")):
            continue
        # Leave the merged links to their prototype bindings: binding here would
        # paint both of their meshes with the link's single colour again.
        if any(str(p.GetPath()).startswith(mp + "/") for mp in merged):
            continue
        # The importer names the wrapping Xform after the MJCF mesh asset and
        # the leaf just "mesh", so walk up until a name is recognised.
        rgb, q = None, p
        while q and q.GetPath() != q.GetPath().GetParentPath():
            rgb = colours.get(q.GetName())
            if rgb is not None:
                break
            q = q.GetParent()
        if rgb is None:
            rgb, _n = DEFAULT_RGB, n_default
            n_default += 1
        UsdShade.MaterialBindingAPI.Apply(p).Bind(material_for(rgb))
        UsdGeom.Mesh(p).GetDoubleSidedAttr().Set(True)
        n_bound += 1
    log(f"bound {len(mats)} MJCF colours to {n_bound} robot meshes ({n_default} fell back to neutral)")
    if not any(p.GetTypeName() in ("DomeLight", "DistantLight") for p in stage.Traverse()):
        dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/dome_light"))
        dome.CreateIntensityAttr(1000.0)
        sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/sun_light"))
        sun.CreateIntensityAttr(2500.0)
        UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-40.0, 20.0, 0.0))
        log("added dome + distant lights")

    # Meshes the ego camera must not see; see StageCfg.hidden_visuals.
    for rel in cfg.hidden_visuals:
        prim = stage.GetPrimAtPath(f"/visuals/{rel}")
        if not prim.IsValid():
            raise SystemExit(
                f"hidden_visuals: /visuals/{rel} does not exist -- the robot "
                f"USD predates the reference split? rebuild it: "
                f"bash scripts/build_isaac_assets.sh --robot-only --force"
            )
        UsdGeom.Imageable(prim).MakeInvisible()
        log(f"hid /visuals/{rel}")

    configure_renderer(log)

    # Manipulable objects: every converted USD under the scene's usd/ dir is
    # referenced at /World/<name>/geom, with the MJCF's mesh scale and the
    # body->geom offset from the build's sidecar json, so that sync_replay can
    # set /World/<name> to the BODY pose straight from the replay. (Was a
    # bottle-only block with a scalar scale; generalised 2026-09-07 for the
    # crate, whose mesh is scaled 0.667/0.667/1.0 and offset -0.085 in z.)
    import json as _json

    usd_root = os.path.join(scene_dir(cfg), "usd")
    names = sorted(os.listdir(usd_root)) if os.path.isdir(usd_root) else []
    # usd/ also holds the STATIC props the scene layer already places (table,
    # fork, plate...); only the manipulable objects named by scene_meta.json
    # (next to the scene xml) get a pose-driven /World/<name> prim.
    meta_path = os.path.join(os.path.dirname(os.path.abspath(cfg.robot_mjcf)), "scene_meta.json")
    if os.path.exists(meta_path):
        movable = set(_json.load(open(meta_path)).get("objects") or [])
        names = [n for n in names if n in movable]
    referenced = 0
    for name in names:
        top = os.path.join(usd_root, name, f"{name}.usd")
        if not os.path.exists(top):
            continue
        meta = {}
        meta_json = os.path.join(usd_root, name, f"{name}.json")
        if os.path.exists(meta_json):
            meta = _json.load(open(meta_json))
        scale = meta.get("scale", 1.0)
        scale = [float(scale)] * 3 if not isinstance(scale, list) else [float(v) for v in scale]
        opos = meta.get("offset_pos", [0.0, 0.0, 0.0])
        oq = meta.get("offset_quat", [1.0, 0.0, 0.0, 0.0])
        stage.DefinePrim(f"/World/{name}", "Xform")
        g = stage.DefinePrim(f"/World/{name}/geom", "Xform")
        g.GetReferences().AddReference(top)
        # The converted asset already authors [translate, orient, scale] on its
        # default prim, and the reference brings that op order with it, so
        # AddTranslateOp() raises "already exists": set the existing ops.
        xf = UsdGeom.Xformable(g)
        ops = {op.GetOpType(): op for op in xf.GetOrderedXformOps()}
        (ops.get(UsdGeom.XformOp.TypeTranslate) or xf.AddTranslateOp()).Set(Gf.Vec3d(*opos))
        oop = ops.get(UsdGeom.XformOp.TypeOrient) or xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        if oop.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
            oop.Set(Gf.Quatd(oq[0], Gf.Vec3d(*oq[1:])))
        else:
            oop.Set(Gf.Quatf(oq[0], Gf.Vec3f(*oq[1:])))
        (ops.get(UsdGeom.XformOp.TypeScale) or xf.AddScaleOp()).Set(Gf.Vec3f(*scale))
        bb = UsdGeom.Imageable(g).ComputeWorldBound(0, UsdGeom.Tokens.default_).ComputeAlignedBox()
        if bb.IsEmpty():
            raise SystemExit(
                f"object {name} composed EMPTY from {top}. The reference "
                f"resolved to nothing -- almost always a relative path, which "
                f"USD resolves against the layer's directory, not the cwd."
            )
        referenced += 1
        log(
            f"object {name} referenced (scale {scale}, offset {opos}, bbox "
            f"{[round(v, 3) for v in bb.GetMin()]}..{[round(v, 3) for v in bb.GetMax()]})"
        )
    if not referenced:
        log(f"WARNING no object usd under {usd_root} -- rendering without objects")
    return stage


def configure_renderer(log=print) -> None:
    """Match SIMPLE's renderer configuration (engines/isaacsim.py:_config_isaac).

    Our first pass set only `/rtx/post/aa/op = 3` (DLSS) and left everything
    else default. That produced renders that were blurry, stippled, and a frame
    behind the state -- while SIMPLE's own Isaac output is sharp. Three of these
    settings matter, and none of them is cosmetic:

    * sampledLighting/autoEnable off -- stochastic direct lighting is what makes
      the image look dithered at one sample per pixel.
    * asyncRendering off + hydraEngine/waitIdle on -- otherwise `get_rgba()` can
      return a frame composed before the transforms of THIS step landed, so the
      picture lags the state. That is a correctness bug, not a quality one: it
      shows the gripper and the object at the previous frame's poses.
    * set_render_rtx_realtime() -- pick the renderer explicitly instead of
      inheriting whatever the app defaulted to.

    DLSS is deliberately NOT enabled: it upscales from a lower internal
    resolution, which is the opposite of what a dataset at 640x480 wants.
    """
    import carb.settings
    import omni.kit.commands
    import omni.replicator.core as rep

    ok, res = omni.kit.commands.execute(
        "ChangeSetting", path="/rtx/directLighting/sampledLighting/autoEnable", value=False
    )
    if not ok:
        log(f"WARNING could not disable sampled lighting: {res}")
    st = carb.settings.get_settings()
    st.set("/omni/replicator/captureOnPlay", False)
    st.set("/omni/replicator/asyncRendering", False)
    st.set("/app/asyncRendering", False)
    st.set("/app/hydraEngine/waitIdle", True)
    st.set("/app/updateOrder/checkForHydraRenderComplete", 1000)
    st.set("/exts/isaacsim.core.throttling/enable_async", False)
    rep.settings.set_render_rtx_realtime()
    log("renderer configured (rtx realtime, sync, waitIdle, no sampled lighting)")


def apply_rtx_settings(items, log=print) -> None:
    """Apply `PATH=VALUE` carb settings after the stage is assembled; bool/int/float parsed, else string.

    For per-scene look settings the renderer configuration does not cover -- the warehouse
    turns auto-exposure off (/rtx/post/histogram/enabled=false, /rtx/post/tonemap/filmIso=80,
    scripts/render_isaac_videos.sh). Every renderer of a scene must apply the same list, or
    the offline dataset and the live mirror stop being the same picture.
    """
    if not items:
        return
    import carb.settings

    st = carb.settings.get_settings()
    for item in items:
        path, _, raw = item.partition("=")
        low = raw.strip().lower()
        value: object
        if low in ("true", "false"):
            value = low == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        st.set(path.strip(), value)
        log(f"rtx setting {path.strip()} = {value!r}")


ROBOT_ROOT = "/World/scene"


def make_world():
    """A kinematic mirror, not a physics world: gravity off."""
    from isaacsim.core.api import World

    world = World(stage_units_in_meters=1.0)
    world.reset()
    world.get_physics_context().set_gravity(0.0)
    return world


def root_body(robot) -> str:
    """The link PhysX treats as the articulation root -- what set_world_pose
    actually moves.

    It is NOT the MJCF's floating base. For our G1 the MJCF base is `pelvis`
    but PhysX roots the articulation at `torso_link`, and it lists the root
    first in body_names. Commanding the pelvis pose therefore pinned the TORSO
    to the pelvis's place and let the waist joints swing the rest of the body
    the wrong way round: measured 2026-09-06, driving a waist joint left the
    torso completely unrotated, the angular error equalling the commanded angle
    exactly (0.25 rad -> 14.324 deg, 0.50 rad -> 28.648 deg). Arms and legs
    looked perfect because they hang off the torso; only the waist exposed it.
    Feeding the root body's own pose instead takes every link to 0.00 mm.
    """
    try:
        return str(robot._articulation_view.body_names[0])
    except Exception:  # noqa: BLE001
        return "torso_link"


def bind_robot(joint_map, log=print):
    """Resolve the 43-vector source indices onto the USD articulation's DOFs."""
    from isaacsim.core.prims import SingleArticulation

    robot = SingleArticulation(ROBOT_ROOT)
    robot.initialize()
    log(f"articulation root link: {root_body(robot)}")
    dof_names = list(robot.dof_names)
    log(f"robot dofs ({len(dof_names)}): {dof_names}")
    jidx, jsrc = [], []
    for name, src in joint_map:
        if name not in dof_names:
            log(f"WARNING joint missing in USD: {name}")
            continue
        jidx.append(dof_names.index(name))
        jsrc.append(src)
    return robot, jidx, jsrc


def mount_camera(stage, cfg: StageCfg, cam_spec, log=print):
    """Place the ego camera from the MJCF spec and wrap it in a SensorCamera."""
    from isaacsim.sensors.camera import Camera as SensorCamera
    from pxr import Gf, Sdf, UsdGeom

    cam_pos, cam_quat, cam_fovy = cam_spec
    torso = stage.GetPrimAtPath(cfg.torso_prim)
    if not torso.IsValid():
        for p in stage.Traverse():
            if p.GetName() == "torso_link" and ROBOT_ROOT in str(p.GetPath()):
                torso = p
                break
    log(f"torso prim: {torso.GetPath() if torso.IsValid() else 'NOT FOUND'}")
    cam_path = "/World/ego_cam" if cfg.absolute_camera else str(torso.GetPath()) + "/ego_cam"
    cam_prim = UsdGeom.Camera.Define(stage, Sdf.Path(cam_path))
    xf = UsdGeom.Xformable(cam_prim.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*cam_pos))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(cam_quat[0], Gf.Vec3d(*cam_quat[1:])))
    # fovy -> USD vertical aperture at an arbitrary focal length.
    focal = 18.0
    # USD default clippingRange starts at 1 m -- the robot's own arms
    # (0.3-0.6 m from the ego camera) were silently near-clipped for seven
    # iterations.  MuJoCo's znear is model-scaled and tiny; match that.
    cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.02, 1000.0))
    cam_prim.GetFocalLengthAttr().Set(focal)
    ap_v = 2 * focal * np.tan(np.deg2rad(cam_fovy) / 2)
    cam_prim.GetVerticalApertureAttr().Set(ap_v)
    cam_prim.GetHorizontalApertureAttr().Set(ap_v * cfg.width / cfg.height)

    ss = cfg.supersample
    sensor = SensorCamera(prim_path=cam_path, resolution=(cfg.width * ss, cfg.height * ss))
    sensor.initialize()
    return sensor, cam_path


def mount_inset_cameras(stage, cfg: StageCfg, specs: dict, log=print) -> dict:
    """Place picture-in-picture cameras (ego, left/right wrist, or any
    combination) as absolute world-space prims, keyed by their MuJoCo camera
    name.

    Always absolute, never parented to a body: the ego entry here is a
    SEPARATE camera from `mount_camera`'s main one, because the main one
    often (via a caller's --cam-traj) follows a third-person tracking shot
    rather than the ego view -- so this ego inset would not be redundant even
    when the main camera happens to be the ego view too. Every entry is
    re-posed from its own named replay FK every frame, independent of
    `cfg.absolute_camera` and of whatever the main camera is doing. Sized to
    the caller's own resolution -- normally a small inset, not the full
    frame, since RTX render cost scales with viewport pixel count.

    `specs` is `{camera_name: (pos, quat, fovy, width, height)}` (wxyz quat,
    degrees fovy); returns `{camera_name: (sensor, cam_path)}`. The sensors render
    at `cfg.supersample` times that size; `downsample` brings them back.
    """
    from isaacsim.sensors.camera import Camera as SensorCamera
    from pxr import Gf, Sdf, UsdGeom

    ss = cfg.supersample
    out = {}
    for cam_name, (cam_pos, cam_quat, cam_fovy, width, height) in specs.items():
        cam_path = f"/World/inset_{cam_name}"
        cam_prim = UsdGeom.Camera.Define(stage, Sdf.Path(cam_path))
        xf = UsdGeom.Xformable(cam_prim.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*cam_pos))
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(cam_quat[0], Gf.Vec3d(*cam_quat[1:])))
        focal = 18.0
        cam_prim.GetClippingRangeAttr().Set(Gf.Vec2f(0.02, 1000.0))
        cam_prim.GetFocalLengthAttr().Set(focal)
        ap_v = 2 * focal * np.tan(np.deg2rad(cam_fovy) / 2)
        cam_prim.GetVerticalApertureAttr().Set(ap_v)
        cam_prim.GetHorizontalApertureAttr().Set(ap_v * width / height)
        sensor = SensorCamera(prim_path=cam_path, resolution=(width * ss, height * ss))
        sensor.initialize()
        out[cam_name] = (sensor, cam_path)
        log(f"inset camera [{cam_name}]: {cam_path} res=({width}x{height}) x{ss} fovy={cam_fovy}")
    return out


def _freeze_root(robot) -> None:
    """Zero the floating base's velocity. Nothing else does.

    `sync_dofs` zeroes JOINT velocities and `set_world_pose` does not touch the
    root's, so the base kept whatever linear velocity it picked up at world
    reset and integrated it on every physics step. Measured 2026-09-06: root
    linear velocity -0.327 m/s in z, dt 1/60, two steps per frame -> the robot
    rendered 10.9 mm below the pose it was commanded to, every frame. Gravity
    was already off and collisions made no difference; it was pure momentum on
    a base nobody had told to stop. Zeroing it takes the drift to 0.00 mm.
    """
    import numpy as _np

    try:
        robot.set_linear_velocity(_np.zeros(3))
        robot.set_angular_velocity(_np.zeros(3))
    except Exception:  # noqa: BLE001
        pass


def sync_dofs(robot, dof_idx, dof_values, _zeros=[None]):  # noqa: B006  (intentional per-function cache)
    """Push explicit per-DOF positions -- the replay path, where the values come
    straight out of a recorded qpos by joint name and there is no 43-vector."""
    robot.set_joint_positions(np.asarray(dof_values), joint_indices=dof_idx)
    if _zeros[0] is None or len(_zeros[0]) != robot.num_dof:
        _zeros[0] = np.zeros(robot.num_dof, dtype=np.float32)
    robot.set_joint_velocities(_zeros[0])


def sync_replay(stage, robot, dof_idx, dof_values, root_pose, objects_t, sensor=None, cam_pose=None):
    """One recorded MuJoCo state -> the mirror stage. SIMPLE's isaac.step().

    `root_pose` must be the pose of the ARTICULATION ROOT LINK (see
    :func:`root_body`), not of the MJCF's floating base.
    """
    from isaacsim.core.prims import SingleXFormPrim

    if cam_pose is not None and sensor is not None:
        sensor.set_world_pose(position=cam_pose[0], orientation=cam_pose[1], camera_axes="usd")
    robot.set_world_pose(position=root_pose[:3], orientation=root_pose[3:])
    _freeze_root(robot)
    sync_dofs(robot, dof_idx, dof_values)
    for key, pose in objects_t.items():
        prim_path = f"/World/{key}"
        if stage.GetPrimAtPath(prim_path).IsValid():
            SingleXFormPrim(prim_path).set_world_pose(position=pose[:3], orientation=pose[3:])


def sync_frame(stage, robot, jidx, jsrc, joints_t, pelvis_t, objects_t, sensor=None, cam_pose=None, _zeros=[None]):  # noqa: B006, E501
    """Push one kinematic state into the stage. The live mirror calls exactly
    this, which is what keeps online and offline pixels the same function of
    the same state.

    joints_t   : the 43-vector for this frame
    pelvis_t   : (7,) world pos + wxyz quat
    objects_t  : {name: (7,)} world poses
    cam_pose   : (pos, wxyz quat) to override the mount, or None
    """
    from isaacsim.core.prims import SingleXFormPrim

    if cam_pose is not None and sensor is not None:
        sensor.set_world_pose(position=cam_pose[0], orientation=cam_pose[1], camera_axes="usd")
    robot.set_world_pose(position=pelvis_t[:3], orientation=pelvis_t[3:])
    _freeze_root(robot)
    robot.set_joint_positions(joints_t[jsrc], joint_indices=jidx)
    if _zeros[0] is None or len(_zeros[0]) != robot.num_dof:
        _zeros[0] = np.zeros(robot.num_dof, dtype=np.float32)
    robot.set_joint_velocities(_zeros[0])
    for key, pose in objects_t.items():
        prim_path = f"/World/{key}"
        if stage.GetPrimAtPath(prim_path).IsValid():
            SingleXFormPrim(prim_path).set_world_pose(position=pose[:3], orientation=pose[3:])


def warm_up(
    world, sensor, stage, robot, jidx, jsrc, joints_t, pelvis_t, objects_t, cam_pose=None, max_steps: int = 8, log=print
) -> int:
    """Step until the sensor actually returns pixels, discarding the output.

    The first one or two `get_rgba()` calls after a stage is built come back
    empty while the render product spins up. Left alone that silently drops the
    leading frames: a 1209-frame episode wrote 1207, and an image stream two
    frames short of its action stream is misaligned for the whole episode.
    Doing it here, on the FIRST state, means the discarded frames are duplicates
    of frame 0 rather than missing frames.
    """
    for i in range(max_steps):
        sync_frame(stage, robot, jidx, jsrc, joints_t, pelvis_t, objects_t, sensor=sensor, cam_pose=cam_pose)
        if step_and_grab(world, sensor) is not None:
            if i:
                log(f"sensor warm-up: {i} discarded step(s)")
            return i
    log(f"WARNING sensor still empty after {max_steps} warm-up steps")
    return max_steps


# A camera world pose needs TWO steps to reach the renderer, not one. Measured
# 2026-09-06: after a 50-frame state jump, one step renders the table 67 px off
# with an IoU of 0.43 against MuJoCo; two steps land at 1 px and 0.76, and
# three or five change nothing. One step means every frame is drawn with the
# PREVIOUS frame's camera. Between adjacent frames that is a small, invisible,
# systematic lag -- which is worse than an obvious one, because it ships.
# SIMPLE does not hit this: it parents the camera to the robot, so the pose
# rides the same fabric update as the articulation. Ours is an absolute world
# pose per frame, and pays for it here.
SETTLE_STEPS = 2


def step_and_grab(world, sensor, steps: int = SETTLE_STEPS):
    """Advance until the new transforms are in the render, then read. None if
    the sensor is still warming up and returns nothing."""
    rgba = None
    for _ in range(max(1, steps)):
        world.step(render=True)
        got = sensor.get_rgba()
        if got is not None and got.size:
            rgba = got
    return rgba


def step_and_grab_all(world, sensors: dict, steps: int = SETTLE_STEPS) -> dict:
    """Read fresh, synchronized camera buffers from the final settled render step.

    Isaac's current-frame buffer pairs RGBA with its rendering-frame ID. Missing
    or stale cameras return None; a previous step's image must never fill the gap.
    """
    rgba = {name: None for name in sensors}
    for _ in range(max(1, steps)):
        previous = {name: sensor.get_current_frame()["rendering_frame"] for name, sensor in sensors.items()}
        world.step(render=True)
        frames = {name: dict(sensor.get_current_frame()) for name, sensor in sensors.items()}
        latest = max((f["rendering_frame"] for f in frames.values()), default=0)
        rgba = {}
        for name, frame in frames.items():
            got, stamp = frame.get("rgba"), frame["rendering_frame"]
            fresh = stamp > previous[name] and stamp == latest
            rgba[name] = got.copy() if fresh and got is not None and got.size else None
    return rgba
