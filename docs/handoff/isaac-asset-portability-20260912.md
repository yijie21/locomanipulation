# Isaac fresh-install asset repair — 2026-09-12

Request: another developer's GitHub checkout renders the dual-arm task without
background or tables. Baseline `6aa0b2f`; implementation `3f0f4cc` and `c9be6e0`.
Current instructions: [rendering assets](../rendering.md#资产构建).

## Reproduced cause

The tracked `assets/scene_locomani_v1.tar.gz` contains MuJoCo meshes/textures and
`layout/layout.json`, but no `layout/asset3d`, `surface_b/asset3d` or room PLY.
The old Isaac source resolver returned `None` for dining_table on that installed
archive. Both room candidates were absent. These paths fall under ignored
`outputs/`; source availability differed between developer machines. The builder
logged missing surfaces, continued and stamped the incomplete stage successful.

## Repair

- Track the existing complete empty-room mesh in `assets/scene/`, with checksum,
  counts and coordinate provenance. Compressed size: 42,918,604 bytes. No Git LFS.
- Use installed MuJoCo table/counter meshes when original generated sources are
  unavailable. Restore Y-up coordinates, normals and the XML texture binding for
  the existing OBJ importer; retain original mesh geometry and UVs.
- Validate required source assets before Kit startup or cache acceptance. Include
  sources, textures, builder, XML/meta, checkout and destination paths in the scene
  cache key. Preserve replaced caches and require a successful scene build marker.
- Match snapshot labels to the directory basename used by the rendering caller.
  Converted USD remains ignored and is regenerated in each checkout.

## Checks and observations

`make check` and five targeted tests pass. Tests install relevant assets from the
tracked archive, verify coordinate round trips and texture bytes, check the room
checksum/header, exercise missing sources through the launcher, and invalidate
cache keys on texture changes and destination relocation. Independent Standards
and Spec reviews pass; the cache destination omission found in review was fixed.

Artifacts: `outputs/isaac_asset_portability_20260912/`.
`fresh_checkout/outputs/scene` was installed anew by `scripts/install_scene.sh`
with `SCENE_PRESET=crate_bimanual`; code, tracked assets and third-party runtime
were shared by symlink for this run. No local EmbodiedGen-generated source folders
were copied. Isaac Sim 4.5 built and rendered that installation successfully.

```bash
LOCOMANI_ROOT=<fresh_checkout> ISAAC_ASSETS=<new_usd_directory> \
CUDA_VISIBLE_DEVICES=0 bash scripts/build_isaac_assets.sh \
  --snapshot <fresh_checkout>/outputs/scene/sim
```

`build_final_launcher.log`, `isaac/scene/sim/build.log` and the overview,
from_robot and surfaces PNGs retain the render evidence. The room and textured
table are visible. The dining table bounds are approximately
[-0.503, -0.503, 0] to [0.503, 0.503, 0.620] m; counter bounds match the source
layout. A second identical build reports both robot and scene cached without Kit
startup (`cache_hit.log`). The original incomplete/previous caches are preserved.

The old room ends at X≈2.514 m while the counter reaches X≈2.685 m, so that wall
intersection remains; it is an inherited scene-design limitation. This repair
restores asset availability and does not validate every camera, Isaac version,
layout, or VLA input modality. No robot dynamics or collection controller changed
in the asset commits. The separate walking-control changes have their own
[GPU evidence](crate-walk-stability-20260912.md).
