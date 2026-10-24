# Patch: URLab "fresh_depth" camera mode (depth of the colour camera's view)

On top of [urlab_fresh_capture](urlab_fresh_capture.md) (plugin branch `loco-fresh-capture`, `923b689`). Written for
vision RSI L3 ([design](../docs/handoff/vision-rsi-design-20261023.md) section 16): the masks SAM 3 draws on Unreal's
RGB cover the objects' visual shapes, so the depth they are back-projected with must be Unreal's too. MuJoCo's depth
of the physics model shows the collision shapes: the mug's five boxes have no handle, and the handle's pixels took the
depth of whatever lay behind it (1.9-4.0 cm of fit error at L3, under 1 cm with exact masks).

**Problem at 923b689.** `UMjCamera` has a Depth capture mode (`SCS_SceneDepth` into an R32f target), but a camera is
either colour or depth, and the "fresh" path skips depth cameras (`RenderNow` refuses them, the dispatcher filters
them out, `ReadNow` reads BGRA only). Depth twins of the robot's cameras in the level would need the level rebuilt
and two cameras kept in step.

**Patch.** One new include mode; "sync", "latest", "fresh" and "stream" are unchanged.
- `include_cameras: {cam: "fresh_depth"}` is "fresh" (a camera cut plus `urlab.FreshCaptureRepeats` colour captures,
  after the pose reached the scene) plus the depth of the same view. The reply block keeps `data` (bgra8) and adds
  `depth`: float32, width x height, row-major, Unreal's scene depth (centimetres along the view axis).
- `UMjCamera::RenderDepthNow` renders the depth with a second `USceneCaptureComponent2D`, made on first use
  and attached to the colour capture with no offset. At each call it takes the colour capture's FOV, projection, near
  clip, view distance, show flags and hidden / show-only lists. It captures once, with the TemporalAA and
  AntiAliasing show flags off (TSR's sub-pixel jitter would shift the depth) and no persistent view state, so the
  colour capture's TSR and Lumen history are untouched. Real-mode cameras only.
- `UMjCamera::ReadDepthNow` reads it back synchronously (`ReadLinearColorPixels`, raw floats, R channel). The capture
  task renders every camera's colour and depth before the first readback, as "fresh" does.

Two engine behaviours the first build ran into, both fixed in the patch:
- **Vulkan readback.** `ReadLinearColorPixels` on an R32f target asserts and then crashes the editor ("Unsupported
  format [100] for conversion to FLinearColor", `ConvertRawDataToFLinearColor` in `VulkanRenderTarget.cpp`): Vulkan
  converts RGBA32f, not R32f. The depth target is RGBA32f with the depth in R (4.9 MB per 640 x 480 camera). The
  plugin's own Depth mode reads its R32f target the same way and would hit the same assert under Vulkan; the patch
  leaves it alone.
- **A hidden capture does not capture.** `CaptureScene()` returns without rendering when `IsVisible()` is false, and
  `bHiddenInGame` makes it false (`USceneComponent::IsVisible`). The depth came back all zeros until the depth capture
  was unhidden, as `SetStreamingEnabled` unhides the colour one. A capture component draws nothing itself.

Measured 2026-10-24 (`scripts/experiments/unreal_depth_check_20261024.py`, apartment level, the robot at a sealed-pool
start, 640 x 480, `outputs/vision_rsi_20261023/depth_check/`): see "Check" below.

Apply and rebuild (the vision line builds it in its own copy of the host project, so the main line's plugin and
binaries stay at `loco-fresh-capture`):

```bash
git -C <Project>/Plugins/UnrealRoboticsLab apply patches/urlab_fresh_depth.patch
. /workspace/ue_57/ue_env.sh
bash "$UE_ROOT/Engine/Build/BatchFiles/Linux/Build.sh" URLabSmokeEditor Linux Development "-Project=$UE_PROJECT" -WaitMutex
```

Client: `locomani.sim.unreal.view.UnrealView.grab_depth` (metres), `locomani.sim.unreal.rgb_server` with
`{"depth": True}`, `locomani.tasks.scene.perceive.RenderConfig(depth="unreal")`.

## Check

Against MuJoCo's depth of the same state from the full scene model, visual geoms only (the model the level was built
from):

| camera | pixels within 1 cm | within 5 cm | median difference | 90th percentile | ratio centre / corner |
|---|---|---|---|---|---|
| head (ego_view) | 98.8% | 99.7% | 0.07 cm | 0.22 cm | 1.0009 / 1.0005 |
| left wrist | 100% | 100% | 0.01 cm | 0.03 cm | 1.0002 / 1.0002 |
| right wrist | 100% | 100% | 0.04 cm | 0.14 cm | 1.0009 / 1.0005 |

- Unreal's scene depth is along the view axis, as MuJoCo's is: the ratio at the corners stays 1.0005, where a distance
  along the ray would give the corner rays' 1.27-1.35.
- The head camera's 1.2% beyond 1 cm is almost all one ornament on the coffee table (a small house), farther in
  Unreal than in MuJoCo; none of the sealed pool's objects.
- Two "fresh_depth" grabs of one state give bit-identical depth. A request costs what "fresh" does (0.18-0.19 s for
  three cameras at 8 colour captures each); the colour images differ from plain "fresh" ones by 0.60-0.76 mean abs
  (0-255), the same as two "fresh_depth" grabs differ from each other (0.67-0.99).
