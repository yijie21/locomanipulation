# Logistics room with NOR-CAL cardboard box — v1

This directory is the versioned, Git LFS backed copy of the logistics-room scene
used by the 2026-09-20 box-rim planning and live execution work. The entry points
are `mujoco/scene_locomani.xml` for physics and
`isaac/logistics_room_box.usdc` for offline Isaac rendering. The runtime sources
under `outputs/scene/` remain preserved and are identified in `manifest.json`.

The packaged MJCF differs from its source XML only in file paths: scene meshes
and textures are package-relative, and G1/Dex1 mesh references are relative to
the standard `third_party/GR00T-WholeBodyControl` installation. Geometry,
materials, robot defaults, poses, actuators, collisions, cameras and lights are
unchanged. The USD bytes and all 49 referenced texture files are unchanged. The
465-byte dome EXR emitted by Blender was omitted because the repaired USD clears
that unsupported texture reference and uses a flat dome colour.

## Measured geometry

All values are metres in a Z-up frame.

| Item | Size or pose |
|---|---|
| HP platform | 1.200 × 0.900 × 0.371; centre `(0, -0.105, 0.1855)`; top `z=0.371` |
| Platform Y edges | near `-0.555`, far `+0.345` |
| Box visual mesh | nine OBJs, one per material (six are photographs of the real box); union 0.450 × 0.320 × 0.330; local bounds `(-0.225, -0.160, 0)` to `(0.225, 0.160, 0.330)` |
| Box body | origin `(0, -0.390, 0.371)`, identity rotation; rim `z=0.701` |
| Box physics | same 0.450 × 0.320 × 0.330 outer dimensions; 0.010 wall/floor thickness; five active box colliders |
| Box inertial | mass 1.2 kg; local COM `(0, 0, 0.165)` |
| Ground plane | collision plane at `z=0.001`, hidden from rendering |
| Robot start | pelvis `(0, -1.205, 0.793)`, +90° yaw |

The 0.010 m physical wall thickness is twice the `0.005` half-thickness passed
to MuJoCo's `box.size`; it must not be reported as a 5 mm wall. Blender/USD and
the MuJoCo visual OBJ both measure 0.450 × 0.320 × 0.330. The physical box is a
hollow approximation with a floor and four walls, while the visual mesh retains
the actual reconstructed cardboard geometry.

## Install and validate

Pull Git LFS content and install the repository's SONIC dependency first (steps 1-2 of
`docs/setup.md`: the upstream clone and its `.venv_sim`); the scene deliberately does not
duplicate upstream robot meshes. `scripts/setup_sonic.sh` is setup step 6 (Isaac Sim), optional
and not needed here.

```bash
git lfs pull
. scripts/paths.sh
"$VENV_SIM_PY" assets/scene/logistics_room_box_v1/check_assets.py \
  --load-mujoco --load-usd
```

The checker verifies both payload tree hashes, rejects absolute or missing
references, measures the box OBJ, compiles the MJCF on CPU, and opens the USDC
through Blender's bundled USD API (`--load-usd` needs Blender 5.x on the PATH, see
`docs/blender_scenes.md`; drop the flag to skip the USD check). It also checks the USD box's world bounds:
`(-0.225, -0.550, 0.371)` to `(0.225, -0.230, 0.701)`.

The source MJCF is directly loadable for inspection. Box execution additionally
needs `scene_meta.json` and a named surface body. Create an isolated runtime copy
without changing the packaged source:

```bash
. scripts/paths.sh
"$VENV_SIM_PY" scripts/prepare_logistics_scene.py \
  outputs/scene/logistics_room_box_v1_runtime
```

The preparer copies the complete MuJoCo payload, rewrites external references
for the destination, wraps the unchanged `platform_col` in an identity
`logistics_platform` body, compiles the result, and derives truthful body IDs,
surface dimensions, box metadata and camera selection from that compiled model.
It refuses to overwrite an existing destination.

## Regenerate from Blender

The exact tracked sources are `assets/blender/logistics_room.blend` and
`assets/blender/NOR_CAL_Box_Refined_V2.blend`; their hashes are in the manifest.
The room blend retains several image paths from its author's machine. The
preparation helper remaps missing paths to this package's Isaac texture closure,
then appends the tracked box at the measured pose. The CJK font used by the room
is packed in the blend, so no external font file is needed.

```bash
rebuild_dir=outputs/scene/logistics_room_box_rebuild
mkdir -p "$rebuild_dir"
blender --background assets/blender/logistics_room.blend \
  --python assets/scene/logistics_room_box_v1/rebuild_blender_scene.py \
  --python scripts/blender_to_usd.py -- "$rebuild_dir"
mv "$rebuild_dir/logistics_room.usdc" \
  "$rebuild_dir/logistics_room_box.usdc"
blender --background \
  --python assets/scene/logistics_room_box_v1/check_usd.py -- \
  "$rebuild_dir/logistics_room_box.usdc"
```

Regenerating MuJoCo meshes also requires the exact calibrated base MJCF named in
the manifest (`sha256:42b090c…639da`). That archived runtime input is not tracked,
so the full MuJoCo rebuild command is not supported by a fresh clone. The ready
package itself remains loadable after the standard SONIC install. The builder
inherits the archived input's robot tree and controller defaults instead of
reconstructing them. Reuse the packaged PNG texture closure, because MuJoCo does
not read the room's source JPG materials.

```bash
rebuild_dir=outputs/scene/logistics_mujoco_rebuild
mkdir -p "$rebuild_dir"
cp -a assets/scene/logistics_room_box_v1/mujoco/textures "$rebuild_dir/"
cp assets/scene/logistics_room_box_v1/mujoco/texture_map.json "$rebuild_dir/"
blender --background assets/blender/logistics_room.blend \
  --python scripts/blender_to_mujoco.py -- "$rebuild_dir"
. scripts/paths.sh
"$VENV_SIM_PY" scripts/build_mujoco_scene.py "$rebuild_dir" \
  --base outputs/scene/crate_table30_depth04/scene_locomani.xml
```

The exporter writes the carton as one OBJ per material (`mesh/nor_cal_00..08.obj`,
listed in `carton_parts.json`) because six of its nine materials are photographs
of the real box and MuJoCo binds one texture per geom. Those photographs are the
Isaac payload's `Figure1..6.jpeg`; the packaged `Figure1..6.png` are the same
images resized to 2048 px wide, made with Pillow from `.venv_sim`:

```bash
"$VENV_SIM_PY" -c '
from PIL import Image
for i in range(1, 7):
    im = Image.open(f"assets/scene/logistics_room_box_v1/isaac/textures/Figure{i}.jpeg").convert("RGB")
    im.resize((2048, round(im.height * 2048 / im.width)), Image.LANCZOS).save(
        f"assets/scene/logistics_room_box_v1/mujoco/textures/Figure{i}.png", optimize=True)'
```

Blender 5.2.0 LTS produced the room payload; the carton parts and their PNG
textures were regenerated with Blender 5.2.2 LTS on 2026-09-22 (before that the
carton was a single untextured OBJ with a flat kraft colour). The Isaac USD
keeps light gain at 1.0; callers apply renderer-specific gain at render time.
`room_parts.json`, `carton_parts.json` and `texture_map.json` record the MuJoCo
material split and JPG-to-PNG mapping.
