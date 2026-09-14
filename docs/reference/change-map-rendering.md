# Scenes, cameras and rendering

Use the matching row; expand along actual callers only as the change requires.
Back to [domain navigation](change-map.md). Test names are relative to `tests/`.

| Change | Entry / owner | Configuration or example | Tests |
|---|---|---|---|
| Scene / camera / rendering | [build_sim_scene.py](../../src/locomani/sim/build_sim_scene.py), [scene_lib.py](../../src/locomani/sim/scene_lib.py), [convex_boxes.py](../../src/locomani/sim/convex_boxes.py) | [rendering tutorial](../rendering.md); [crate support collider](../crate_visual_collection.md#run) | `test_head_camera_spec.py`, `test_seat_on_floor.py`, `test_convex_boxes.py` |
| Isaac ego/wrist picture-in-picture | [isaac_render.py](../../src/locomani/sim/isaac/isaac_render.py) → [stage.py](../../src/locomani/sim/isaac/stage.py), [frames.py](../../src/locomani/sim/isaac/frames.py) | [render-wall-pilot insets](../crate_collection.md#渲染并排视频) | `test_head_camera_spec.py`, `test_isaac_dataset_insets.py`, `test_isaac_capture.py`, `test_isaac_inset_render.py` |
| Render launch / scoped shutdown | [render_job.py](../../src/locomani/pipeline/render_job.py) ← [render_wall_pilot.sh](../../scripts/render_wall_pilot.sh), [stop_render.sh](../../scripts/stack/stop_render.sh) | [Job ownership](../crate_collection.md#中途停止渲染--清理) | `test_render_job.py` |
| Isaac source assets / cache | [scene_assets.py](../../src/locomani/sim/isaac/scene_assets.py), [build_isaac_assets.sh](../../scripts/build_isaac_assets.sh) | [rendering](../rendering.md#资产构建), [shared room source](../../assets/scene/README.md) | `test_isaac_scene_assets.py` |
