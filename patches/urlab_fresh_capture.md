# Patch: URLab "fresh" camera capture for puppet mode

URLab ([github.com/URLab-Sim/UnrealRoboticsLab](https://github.com/URLab-Sim/UnrealRoboticsLab), MuJoCo inside
Unreal Engine 5.7) is the candidate replacement for the Isaac mirror: our MuJoCo sim pushes each state as a puppet
`qpos`, and URLab renders the dataset cameras. The plugin is not tracked here. It lives at
`/workspace/ue_57/projects/URLabSmoke/Plugins/UnrealRoboticsLab` (branch `loco-fresh-capture` on top of `efbc5e2`).

**Problem at efbc5e2.** In puppet mode the camera ZMQ publishers are paused. Images come only through the step RPC's
`include_cameras: {cam: "sync"}`, and "sync" returns whatever GPU readback the camera tick already has in flight.
That readback can predate the pose the request just wrote. Measured 2026-09-28 by alternating two poses: about half
the images show the previous pose when the wait after the push is 100 ms or less. With 200 ms or more none are stale.

**Patch.** Everything below is behind URLab's existing include mode. Two new modes cover Real and segmentation
cameras; Depth gets no frame. The reply block carries `"fresh": true` for both.
- `include_cameras: {cam: "fresh"}` is for a pose that jumped (the closed-loop mirror). The dispatcher pushes a render
  snapshot from the current `mjData` and queues the capture onto `AAMjManager::Tick` (`EnqueueTickTask`). Tick N runs
  `ApplyRenderStateNow()` (the meshes move); tick N+1 captures, so the robot's previous transform equals its current
  one and the repeats see no motion. The capture is a camera cut plus `urlab.FreshCaptureRepeats` captures of the
  frozen world, which let TSR and Lumen converge.
- `include_cameras: {cam: "stream"}` is for the next pose of a continuous sequence (a state trace played at its own
  rate). It captures in the same tick as the apply, once, without a camera cut: the motion vectors carry the step, and
  TSR / Lumen reproject their history along it, as in a game frame.
- Every requested camera renders (`UMjCamera::RenderNow`) before the first synchronous readback (`ReadNow`).
  A readback flushes the GPU, so reading each camera right after its own capture serialises them.

Found while bringing up the warehouse level, 2026-09-28:
- **Deadlock.** The capture is queued onto the manager Tick rather than `AsyncTask(GameThread)`. An AsyncTask can be
  pumped inside the engine's own render-thread waits, and the capture's flush then deadlocks.
- **GPU hang.** The first `CaptureNow` switches the camera to on-demand. A per-frame `bCaptureEveryFrame` capture plus
  an explicit one in the same frame hung the GPU at 100% in the 545k-triangle level. The host project also sets
  `r.Shadow.Virtual.Enable=0`.
- **FOV.** MJCF `fovy` is vertical, but `USceneCaptureComponent2D::FOVAngle` is horizontal. URLab passed it straight
  through, so every 640x480 camera was 73.7 deg wide instead of 90. `HorizontalFovDeg` converts it by aspect.
- **Near clip.** New CVar `urlab.CameraNearClipCm` (default 0.1, as upstream). MuJoCo and our Isaac renderer clip at
  2 cm; at 1 mm the head camera shows the inside of the robot's head as a white band.
- **Lumen accumulation.** New CVar `urlab.FreshCaptureRepeats` sets captures per request, and `bAlwaysPersistRenderingState`
  is on. Lumen only accumulates across captures with a view state. It also needs the level's post-process volume to
  override `DynamicGlobalIlluminationMethod = Lumen`: scene captures ignore the project default and otherwise render no GI.
- **TSR.** Scene captures turn TemporalAA off by default, which drops the project's TSR to FXAA (aliased edges and
  text). The first on-demand capture turns the `TemporalAA` and `AntiAliasing` show flags on.

Files: `RpcDispatcher.{h,cpp}`, `AMjManager.{h,cpp}`, `MjCamera.{h,cpp}`.

**Measured with the patch.**
- Robot-only level, 3 cameras, 1 capture per request: 0/50 stale at 0 ms wait, p50 54-67 ms. The Isaac mirror takes 95 ms.
- Warehouse level, 3 dataset cameras (`scripts/experiments/unreal_capture_bench_20260928.py`, 2026-09-28): "fresh"
  8 captures p50 634 ms; "fresh" 1 capture 154 ms (194 ms before the batched readback); "stream" 129 ms.
  A 25 fps trace video takes 139 ms per frame for capture (was 642 ms). Against 8-capture frames, "stream" frames differ by
  2.1-2.2 mean abs (0-255); two 8-capture renders of one pose differ by 0.8-1.1, and a single cut capture by 3.5.
- Where a capture's ~23 ms goes: ProfileGPU of an editor frame after a pose change shows ShadowDepths 14 ms of 18. Each
  of the warehouse's six RectLights re-renders a 1024^2 Nanite cube shadow map (6 faces x ~0.36 ms). With the lights'
  shadows off a capture costs 4.2 ms, but the image loses every contact shadow. The robot is not the cost: with its
  meshes non-Nanite (`scripts/experiments/unreal_no_nanite_20260928.py` with SCOPE "robot") a capture still costs 23-25 ms, and the
  robot's share of the shadow pass is 0.3 ms. The rest (13.3 ms) is the Nanite cube pass. For every face of every
  shadowed light, UE runs the whole Nanite shadow pipeline (`ShadowDepthRendering.cpp`, "Nanite Cubemap"). A cache
  mode only hides static geometry; it never skips the pass. That costs ~0.37 ms per face, 36 faces, whatever is
  drawn. A normal view caches the static map (profile: `WholeScene MovablePrimitives`, 0.04 ms per light) and still
  pays the 13.3 ms. Scene captures don't cache at all: `ShadowSetup.cpp` skips `ComputeWholeSceneShadowCacheModes`
  when `bAnyViewIsSceneCapture`. The headless editor also renders its own
  viewport each tick (24 ms tick idle, 8 ms with `ShowFlag.Rendering 0`); that is ~35 ms of a request's fixed wait.
  Switching it off changes the captures (`scripts/experiments/unreal_pose_set_20260928.py`). ON vs OFF differs
  by 0.74-1.30 mean abs; repeats with the same setting differ by 0.26-0.36. Indirectly lit areas come out ~2/255 darker
  with it off, and sharpness is unchanged. So whichever setting is chosen must be the same for every Unreal render.
  It is not a quality loss: at 8 repeats each setting is equally far from its own 64-repeat reference (1.25 ON,
  1.30 OFF). The two references differ by 0.86, a different converged state that is smaller than the 8-repeat residual.
- No Nanite in shadows, measured 2026-09-28: every mesh non-Nanite (`unreal_no_nanite_20260928.py`) plus `r.Shadow.Nanite 0`. Shadow pass 14.3 -> 0.29 ms, capture 23.9 -> 3.9 ms, 3 cameras fresh x8 640 ->
  138 ms, stream 117 -> 35 ms. Images differ from Nanite by 1.4 (38 dB), on edges only. `r.Shadow.Nanite 0` alone drops
  every Nanite mesh's shadow. With the shadows cheap, the viewport costs ~nothing (8.6 ms idle tick with it on).
- Virtual shadow maps (`r.Shadow.Virtual.Enable=1`) hang the GPU at the first on-demand capture (reproduced
  2026-09-28 with the patch); changing `r.ShadowQuality` at runtime crashes the editor (assert in ShadowRendering.cpp).

The host project needs these in `Config/`:
- `DefaultEditorSettings.ini`: `bThrottleCPUWhenNotForeground=False`. A headless editor is never foreground and drops to about 9 fps.
- `DefaultEngine.ini` `[ConsoleVariables]`: `urlab.FreshCaptureRepeats`, `urlab.CameraNearClipCm=2.0` and
  `r.Shadow.Nanite=0` (render default since 2026-09-28, together with non-Nanite meshes: run
  `scripts/experiments/unreal_no_nanite_20260928.py` after every URLab import).

Apply and rebuild:

```bash
git -C <Project>/Plugins/UnrealRoboticsLab apply patches/urlab_fresh_capture.patch
. /workspace/ue_57/ue_env.sh
bash "$UE_ROOT/Engine/Build/BatchFiles/Linux/Build.sh" URLabSmokeEditor Linux Development "-Project=$UE_PROJECT" -WaitMutex
```

Test: `scripts/experiments/unreal_smoke_20260928.py alternate --delays-ms 0` (defaults to `--camera-mode fresh`).
Timing: `scripts/experiments/unreal_capture_bench_20260928.py` (modes, cameras and repeats per request).
