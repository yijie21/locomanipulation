"""Convert the built MuJoCo scene to USD via Isaac's MJCF importer.

Why the MJCF importer instead of SIMPLE's separate robot-USD + object-USD
assembly: our scene XML is the single source of truth the physics sim runs
(robot + dex1 gripper + furniture + objects + head_camera, one file), and the
importer preserves joint names/ordering 1:1 -- so the state sync in
:mod:`stage` is a dumb qpos copy with no name-mapping table to maintain.

Runs in the RENDER venv (teleop_free.paths.RENDER_PY):

    "$RENDER_PY" -m teleop_free.isaac.isaac_convert [scene_xml] [out_usd]

Prints -- to the log file, since Isaac swallows stdout -- the imported joint
and camera prims, so the state sync can be written against verified names.
"""

from __future__ import annotations

import os
import sys

from .. import paths

SCENE_XML = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else paths.SCENE_XML)
OUT_USD = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else
                          os.path.join(paths.ISAAC_ASSETS, "scene_locomani.usda"))
os.makedirs(os.path.dirname(OUT_USD), exist_ok=True)
LOG = os.path.join(os.path.dirname(OUT_USD), "convert.log")
open(LOG, "w").close()


def log(msg: str) -> None:
    with open(LOG, "a") as f:
        f.write(f"{msg}\n")
        f.flush()


log(f"scene_xml = {SCENE_XML}")
log(f"out_usd   = {OUT_USD}")

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

try:
    import omni.kit.commands
    import omni.usd
    from pxr import Sdf
    from omni.isaac.core.utils.extensions import enable_extension

    enable_extension("isaacsim.asset.importer.mjcf")
    app.update()

    from isaacsim.asset.importer.mjcf import _mjcf  # noqa: E402

    cfg = _mjcf.ImportConfig()
    # Render mirror, not a second physics world: a fixed articulation that the
    # state sync poses kinematically. visualize_collision_geoms off keeps the
    # stage to visual meshes only.
    cfg.set_fix_base(False)
    cfg.set_import_sites(False)
    cfg.set_visualize_collision_geoms(False)
    cfg.set_make_default_prim(True)
    cfg.set_self_collision(False)

    ok = omni.kit.commands.execute(
        "MJCFCreateAsset",
        mjcf_path=SCENE_XML,
        import_config=cfg,
        prim_path="/World/scene",
    )
    log(f"MJCFCreateAsset -> {ok}")
    app.update()

    stage = omni.usd.get_context().get_stage()

    # The importer composes a body's several visual meshes onto ONE prim as a
    # reference list, and every prototype names its geometry `mesh`, so the
    # composed prim keeps exactly one of them (pelvis shell, wrist cylinders,
    # three of the four torso meshes: gone). This script leaves that as
    # imported; scripts/build_isaac_assets.sh runs split_visual_refs.py on the
    # exported layer afterwards, which gives each reference its own child
    # prim. Doing it on the text of the layer is deliberate: the same edit
    # through a live stage (ClearReferences + re-home) was what displaced the
    # robot on 2026-09-06 -- with the articulation root still taken from the
    # wrong prim at the time -- and the Sdf.BatchNamespaceEdit rename never
    # applied. The layer edit moves nothing: link transforms identical to the
    # unsplit build to the printed digit, pixel diff confined to the meshes
    # that appeared.

    n_prims = 0
    for prim in stage.Traverse():
        t = prim.GetTypeName()
        if t in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint", "Camera"):
            log(f"{t:24s} {prim.GetPath()}")
        n_prims += 1
    log(f"total prims: {n_prims}")

    stage.GetRootLayer().Export(OUT_USD)
    log(f"exported {OUT_USD}")
except Exception:  # noqa: BLE001
    import traceback
    log("FAILED:\n" + traceback.format_exc())
    raise
finally:
    app.close()
