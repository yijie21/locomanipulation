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

from .. import paths


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
    scene_dir: str = ""          # default: <assets>/scene/work
    robot_usd: str = ""          # default: <assets>/robot/g1_dex1.usda
    robot_mjcf: str = ""         # colours come from here; default paths.SCENE_XML
    width: int = 640
    height: int = 480
    torso_prim: str = "/World/scene/torso_link"
    # A cam-traj puts the ego camera at stage top level with an ABSOLUTE world
    # pose per frame. Parented under torso_link, set_world_pose back-computes a
    # local offset against the parent's pre-step (stale) transform and the view
    # ends up ~1 m off the head.
    absolute_camera: bool = False


def start_app(cfg: StageCfg, headless: bool = True):
    """Boot Kit. Minutes on a cold TensorRT/shader cache."""
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp
    return SimulationApp({"headless": headless,
                          "width": cfg.width, "height": cfg.height})


# abspath is not cosmetic, and this is the second time it has bitten this repo
# (isaac_scene.py carries the same warning). USD resolves a reference against
# the LAYER's directory, not the process cwd, so a relative --scene-dir turns
# `.../usd/water_bottle/water_bottle.usd` into a path under the scene layer's
# own folder, finds nothing, and composes an EMPTY prim. No error, no warning:
# the object simply is not in the picture. Caught 2026-09-06 after a full
# episode rendered with no bottle while the log said "bottle referenced".
def scene_dir(cfg: StageCfg) -> str:
    return os.path.abspath(cfg.scene_dir
                           or os.path.join(cfg.assets, "scene", "work"))


def robot_usd(cfg: StageCfg) -> str:
    return os.path.abspath(cfg.robot_usd
                           or os.path.join(cfg.assets, "robot", "g1_dex1.usda"))


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
        except Exception:                                      # noqa: BLE001
            _XML_CACHE[path] = ""
    return _XML_CACHE[path]


def _mjcf_mesh_colours(mjcf_path: str, log=print) -> dict:
    """{mesh asset name: (r, g, b)} from the MJCF's own geom rgba.

    Visual geoms win over collision ones when a mesh appears in both, and a
    fully transparent rgba is ignored -- those are the alpha-0 collision
    proxies, and taking their colour would paint a visible mesh with an
    invisible geom's value.
    """
    import re

    out: dict = {}
    try:
        xml = open(mjcf_path).read()
    except Exception as e:                                     # noqa: BLE001
        log(f"WARNING no MJCF colours ({mjcf_path}): {e}")
        return out
    for g in re.finditer(r"<geom[^>]*>", xml):
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
    return {k: v[0] for k, v in out.items()}


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
                f"[--snapshot <batch scene dir>]")
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
    # gripper disappeared", which is what it looks like. Colours now come from
    # the MJCF, one material per distinct rgba.
    colours = _mjcf_mesh_colours(cfg.robot_mjcf or paths.SCENE_XML, log)
    mats, n_bound, n_default = {}, 0, 0

    def material_for(rgb):
        key = tuple(round(c, 4) for c in rgb)
        if key not in mats:
            mp = Sdf.Path("/World/Looks/g1_%s" % "_".join(
                f"{int(c * 255):03d}" for c in key))
            mt = UsdShade.Material.Define(stage, mp)
            sh = UsdShade.Shader.Define(stage, mp.AppendChild("shader"))
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*key))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.1)
            mt.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(),
                                                     "surface")
            mats[key] = mt
        return mats[key]

    # Bind on the referenced PROTOTYPES first. The importer composes several
    # meshes onto one link prim with one binding, so a link carrying both the
    # wrist and the 0.25-grey dex1 housing can only have one colour there. The
    # prototypes under /meshes are distinct prims, and a binding on them
    # inherits through the reference -- per-mesh colour with no structural
    # change. Splitting the references into children also works visually and
    # MOVES THE GEOMETRY (silhouette IoU 0.49 against the unsplit robot), so it
    # must not be done.
    # The importer writes a direct `material:binding` on each link prim. That
    # is a LOCAL opinion and beats anything inherited through a reference, so a
    # link that composes two meshes forces both to one colour and the prototype
    # bindings below never show. Clearing it on the multi-reference links is
    # what lets the dex1 housing keep its own 0.25 grey while the wrist stays
    # 0.7 -- without splitting the references, which moves geometry.
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
        except Exception as e:                                 # noqa: BLE001
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
            UsdShade.MaterialBindingAPI.Apply(proto).Bind(
                material_for(rgb), UsdShade.Tokens.strongerThanDescendants)
            n_proto += 1
    log(f"bound {n_proto} mesh prototypes from the MJCF's geom rgba")

    DEFAULT_RGB = (0.82, 0.82, 0.85)
    for p in stage.Traverse():
        if p.GetTypeName() != "Mesh" or not str(p.GetPath()).startswith(
                ("/World/scene", "/meshes", "/visuals")):
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
    log(f"bound {len(mats)} MJCF colours to {n_bound} robot meshes "
        f"({n_default} fell back to neutral)")
    if not any(p.GetTypeName() in ("DomeLight", "DistantLight")
               for p in stage.Traverse()):
        dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/dome_light"))
        dome.CreateIntensityAttr(1000.0)
        sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/sun_light"))
        sun.CreateIntensityAttr(2500.0)
        UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(
            Gf.Vec3f(-40.0, 20.0, 0.0))
        log("added dome + distant lights")

    configure_renderer(log)

    # Bottle: reference the converted USD.
    bottle_usd = os.path.join(scene_dir(cfg), "usd", "water_bottle",
                              "water_bottle.usd")
    if os.path.exists(bottle_usd):
        b = stage.DefinePrim("/World/water_bottle", "Xform")
        b.GetReferences().AddReference(bottle_usd)
        # The MJCF applies a mesh scale the raw OBJ does not carry, so it has to
        # be reapplied here. It is per-asset (0.16 for bottled_water_1), so it
        # comes from the build's sidecar json rather than a literal.
        scale = 0.16
        meta_json = os.path.join(os.path.dirname(bottle_usd), "water_bottle.json")
        if os.path.exists(meta_json):
            import json as _json
            scale = float(_json.load(open(meta_json)).get("scale", scale))
        UsdGeom.Xformable(b).AddScaleOp().Set(Gf.Vec3f(scale))
        bb = UsdGeom.Imageable(b).ComputeWorldBound(
            0, UsdGeom.Tokens.default_).ComputeAlignedBox()
        if bb.IsEmpty():
            raise SystemExit(
                f"bottle prim composed EMPTY from {bottle_usd}. The reference "
                f"resolved to nothing -- almost always a relative path, which "
                f"USD resolves against the layer's directory, not the cwd.")
        log(f"bottle referenced (scale {scale}, bbox "
            f"{[round(v, 3) for v in bb.GetMin()]}..{[round(v, 3) for v in bb.GetMax()]})")
    else:
        log(f"WARNING no bottle usd at {bottle_usd} -- rendering without it")
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
        "ChangeSetting",
        path="/rtx/directLighting/sampledLighting/autoEnable", value=False)
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
    except Exception:                                          # noqa: BLE001
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
    cam_path = ("/World/ego_cam" if cfg.absolute_camera
                else str(torso.GetPath()) + "/ego_cam")
    cam_prim = UsdGeom.Camera.Define(stage, Sdf.Path(cam_path))
    xf = UsdGeom.Xformable(cam_prim.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*cam_pos))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(cam_quat[0], Gf.Vec3d(*cam_quat[1:])))
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

    sensor = SensorCamera(prim_path=cam_path,
                          resolution=(cfg.width, cfg.height))
    sensor.initialize()
    return sensor, cam_path


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
    except Exception:                                          # noqa: BLE001
        pass


def sync_dofs(robot, dof_idx, dof_values, _zeros=[None]):
    """Push explicit per-DOF positions -- the replay path, where the values come
    straight out of a recorded qpos by joint name and there is no 43-vector."""
    robot.set_joint_positions(np.asarray(dof_values), joint_indices=dof_idx)
    if _zeros[0] is None or len(_zeros[0]) != robot.num_dof:
        _zeros[0] = np.zeros(robot.num_dof, dtype=np.float32)
    robot.set_joint_velocities(_zeros[0])


def sync_replay(stage, robot, dof_idx, dof_values, root_pose, objects_t,
                sensor=None, cam_pose=None):
    """One recorded MuJoCo state -> the mirror stage. SIMPLE's isaac.step().

    `root_pose` must be the pose of the ARTICULATION ROOT LINK (see
    :func:`root_body`), not of the MJCF's floating base.
    """
    from isaacsim.core.prims import SingleXFormPrim

    if cam_pose is not None and sensor is not None:
        sensor.set_world_pose(position=cam_pose[0], orientation=cam_pose[1],
                              camera_axes="usd")
    robot.set_world_pose(position=root_pose[:3], orientation=root_pose[3:])
    _freeze_root(robot)
    sync_dofs(robot, dof_idx, dof_values)
    for key, pose in objects_t.items():
        prim_path = f"/World/{key}"
        if stage.GetPrimAtPath(prim_path).IsValid():
            SingleXFormPrim(prim_path).set_world_pose(
                position=pose[:3], orientation=pose[3:])


def sync_frame(stage, robot, jidx, jsrc, joints_t, pelvis_t, objects_t,
               sensor=None, cam_pose=None, _zeros=[None]):
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
        sensor.set_world_pose(position=cam_pose[0], orientation=cam_pose[1],
                              camera_axes="usd")
    robot.set_world_pose(position=pelvis_t[:3], orientation=pelvis_t[3:])
    _freeze_root(robot)
    robot.set_joint_positions(joints_t[jsrc], joint_indices=jidx)
    if _zeros[0] is None or len(_zeros[0]) != robot.num_dof:
        _zeros[0] = np.zeros(robot.num_dof, dtype=np.float32)
    robot.set_joint_velocities(_zeros[0])
    for key, pose in objects_t.items():
        prim_path = f"/World/{key}"
        if stage.GetPrimAtPath(prim_path).IsValid():
            SingleXFormPrim(prim_path).set_world_pose(
                position=pose[:3], orientation=pose[3:])


def warm_up(world, sensor, stage, robot, jidx, jsrc, joints_t, pelvis_t,
            objects_t, cam_pose=None, max_steps: int = 8, log=print) -> int:
    """Step until the sensor actually returns pixels, discarding the output.

    The first one or two `get_rgba()` calls after a stage is built come back
    empty while the render product spins up. Left alone that silently drops the
    leading frames: a 1209-frame episode wrote 1207, and an image stream two
    frames short of its action stream is misaligned for the whole episode.
    Doing it here, on the FIRST state, means the discarded frames are duplicates
    of frame 0 rather than missing frames.
    """
    for i in range(max_steps):
        sync_frame(stage, robot, jidx, jsrc, joints_t, pelvis_t, objects_t,
                   sensor=sensor, cam_pose=cam_pose)
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
