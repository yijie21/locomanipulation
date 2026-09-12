# Shared Isaac room source

`empty_room_world.ply.gz` is a curated source asset tracked in Git. A fresh clone
uses it directly; no EmbodiedGen generation, Git LFS, or manual asset copy is needed.
Tables and their textures come from the tracked `../scene_locomani_v1.tar.gz`,
installed by `bash scripts/install_scene.sh`.

The PLY contains the complete existing empty-room mesh and vertex colours, with
metres, Z up, and world coordinates. Its scale, rotation, translation, floor warp
and floor snap are already baked. The builder must not apply them again. The
asset is visual background; it does not add collisions or change robot control.

Provenance: extracted on 2026-09-12 from `/World/Background` in the existing
`assets/isaac/scene/crate_bimanual/scene.usda`. That scene's build log names the
original source `outputs/scene/bg_empty/scene_0000/mesh_model.ply`; the original
source is no longer present in this checkout. The retained USD, rather than a
new room generated from a prompt, supplied the points, triangle indices and
vertex colours. Positions remain float32; colours are losslessly recovered as
8-bit RGB from the original `n / 255` values. No vertices/faces were decimated.

`empty_room_world.json` records the compressed asset's checksum, size, element
counts, source scene checksum and coordinate frame. Curated source files belong
here; derived USD caches remain ignored under `assets/isaac/`. Do not commit
runtime `outputs/` or `third_party/` directories to distribute assets.

Build/rebuild instructions and ownership: [rendering](../../docs/rendering.md#资产构建).
