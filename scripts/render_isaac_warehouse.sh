#!/usr/bin/env bash
# Isaac re-render of a warehouse run's episodes, side by side with the MuJoCo tracking shot of the SAME trace.
#
#   bash scripts/render_isaac_warehouse.sh <run_dir> [--scene-xml <scene_locomani.xml>] [--gpu N] [--stride 4] [labels...]
#
# Per episode (default: every trace under <run_dir>/trace): mujoco_render writes the tracking shot + cam_traj.npz,
# isaac_render renders the same camera path on the Isaac scene, ffmpeg hstacks them, and a <=500 KB preview lands in
# <run_dir>/previews_isaac/. The Isaac scene is $ISAAC_ASSETS/scene/warehouse_textured (scripts/build_isaac_warehouse_scene.py);
# the robot USD is the cached assets/isaac/robot/g1_dex1.usda (scripts/build_isaac_assets.sh --robot-only).
# The tracking camera is front-right and in the room (azimuth 225, 2.2 m): the recorder's default 2.5 m / 135 shot sits
# behind the south partition and both renderers see a wall (2026-09-24).
# Exposure: auto-exposure OFF and film ISO 30 (RTX_SETTINGS below). The Blender room is lit by five 24828-intensity ceiling
# RectLights plus emissive diffusers, so dome/sun intensities barely matter; with the default auto-exposure + ACES the
# robot's 0.2 greys rendered 108/255 and its 0.7 greys 212 (washed out, head/torso 0.51 vs MuJoCo 0.23). ISO 30 puts the
# torso at 139 vs MuJoCo's 136 and head/torso at 0.36; the rest is the white room's indirect light (2026-09-24).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
. scripts/paths.sh
RUN=""; XML="$PWD/outputs/scene/warehouse_textured/scene_locomani.xml"; GPU=0; STRIDE=4; LABELS=()
RTX_SETTINGS=(--rtx-setting /rtx/post/histogram/enabled=false --rtx-setting /rtx/post/tonemap/filmIso=30)
while [ $# -gt 0 ]; do
  case "$1" in
    --scene-xml) XML="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --stride) STRIDE="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) if [ -z "$RUN" ]; then RUN="$1"; else LABELS+=("$1"); fi; shift ;;
  esac
done
[ -n "$RUN" ] && [ -d "$RUN/trace" ] || { echo "usage: $0 <run_dir with trace/> ..." >&2; exit 2; }
[ -x "$RENDER_PY" ] || { echo "RENDER_PY not executable: $RENDER_PY (scripts/setup_render_runtime.sh)" >&2; exit 1; }
SCENE="$ISAAC_ASSETS/scene/warehouse_textured"
[ -f "$SCENE/scene.usda" ] || { echo "no $SCENE/scene.usda -- run scripts/build_isaac_warehouse_scene.py" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$GPU" OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$LOCOMANI_ROOT/src"
if [ ${#LABELS[@]} = 0 ]; then
  for t in "$RUN"/trace/*_state_trace.npz; do LABELS+=("$(basename "$t" _state_trace.npz)"); done
fi
mkdir -p "$RUN/isaac" "$RUN/previews_isaac"
for L in "${LABELS[@]}"; do
  OUT="$RUN/isaac/$L"; mkdir -p "$OUT"
  GHOST=(); [ -f "$RUN/trace/${L}_ghost.npz" ] && GHOST=(--ghost-trace "$RUN/trace/${L}_ghost.npz")  # ghost-hands rule: third-person views only
  echo "=== $L mujoco_render $(date -u +%H:%M:%S)"
  "$RENDER_PY" -m locomani.sim.isaac.mujoco_render --state-trace "$RUN/trace/${L}_state_trace.npz" --scene-xml "$XML" \
    --out "$OUT" --stride "$STRIDE" --azimuth 225 --distance 2.2 "${GHOST[@]}" > "$OUT/mujoco_render.log" 2>&1; echo "mujoco_render rc=$?"
  echo "=== $L isaac_render $(date -u +%H:%M:%S)"
  "$RENDER_PY" -m locomani.sim.isaac.isaac_render --session "" --state-trace "$RUN/trace/${L}_state_trace.npz" --scene-xml "$XML" \
    --scene-dir "$SCENE" --cam-traj "$OUT/cam_traj.npz" --stride "$STRIDE" --out "$OUT" "${RTX_SETTINGS[@]}" "${GHOST[@]}" > "$OUT/isaac_render.stdout" 2>&1; echo "isaac_render rc=$?"
  if [ -f "$OUT/ego_isaac.mp4" ] && [ -f "$OUT/tracking_mujoco.mp4" ]; then
    ffmpeg -loglevel error -y -i "$OUT/tracking_mujoco.mp4" -i "$OUT/ego_isaac.mp4" -filter_complex "[0:v][1:v]hstack=inputs=2" \
      -c:v libx264 -crf 23 -an "$OUT/side_by_side.mp4"
    ffmpeg -loglevel error -y -i "$OUT/side_by_side.mp4" -vf "scale=640:-2" -r 12 -c:v libx264 -crf 34 -preset veryfast -an \
      "$RUN/previews_isaac/${L}_side_by_side.mp4"
    echo "side_by_side rc=$? preview $(stat -c %s "$RUN/previews_isaac/${L}_side_by_side.mp4") bytes"
  fi
done
echo "=== DONE $(date -u +%H:%M:%S)"
