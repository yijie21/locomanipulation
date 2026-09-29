#!/usr/bin/env bash
# Isaac re-render of a run's episodes, side by side with the MuJoCo tracking shot of the SAME trace, for any scene with an
# Isaac stage (an `isaac` entry in configs/scenes.yaml: warehouse_textured, box_platform).
#
#   bash scripts/render_isaac_videos.sh <run_dir> [--scene-xml <scene_locomani.xml>] [--gpu N] [--stride 4]
#       [--size 1280x960] [--path-trace] [--traces <dir>] [--out <dir>] [--insets] [--all] [labels...]
#
# Which rounds: by default only those that succeeded, as the manifest that judged each one says (the run's
# manifest_NN.jsonl for <run_dir>/trace, closed_loop.manifest.jsonl beside an eval's state_trace), because a round
# costs minutes of Isaac and a failed collection round never reaches the dataset. --all renders every trace; labels on
# the command line render exactly those.
# Picture: both renderers draw --size (default 1280x960; at 640x480 the carton print and the robot's edges alias), the
# Isaac side in RTX Real-Time (the renderer the dataset and the eval mirror use; about 110 ms a frame at 1280x960,
# about 1.7x that at the scenes' 2x supersample).
# --path-trace switches the Isaac side to PathTracing, 64 samples per pixel + OptiX denoiser: indirect light and
# soft contact shadows, about 1.5 s a frame at 1280x960 (0.43 s at 640x480), and a mean luma about 10/255 higher than
# the Real-Time ISO the scene was tuned under (A/B 2026-09-28, docs/ledger/blender_scenes.md). For showing, not data.
# Per round: mujoco_render writes the tracking shot + cam_traj.npz,
# isaac_render renders the same camera path on the Isaac scene, ffmpeg hstacks them, and a preview lands in
# <out>/previews_isaac/ (<out> defaults to <run_dir>). Previews are shareable as written: locomani.pipeline.video_preview
# keeps each file <= 500,000 bytes at 1280 px wide / 10 fps, raising CRF and then splitting into consecutive
# <label>_side_by_side_partKofN.mp4 (a 66 s round at the default size: three parts). The
# Isaac-only render is <out>/isaac/<label>/ego_isaac.mp4. A closed-loop evaluation: --traces <run>/eval/state_trace
# --out <run>/eval --insets; --insets adds the ego + wrist cameras (a separate pass, about twice the time).
# The stage and its look (`isaac.rtx_settings`, `isaac.supersample`) come from the run's scene through
# run_context.isaac_scene, as the eval mirror and the dataset re-render read them (--path-trace drops the supersample);
# the scene xml defaults to the run's own scene copy. The robot USD is the cached assets/isaac/robot/g1_dex1.usda (scripts/build_isaac_assets.sh --robot-only).
# The tracking camera is front-right (azimuth 225, 2.2 m): in the warehouse the recorder's default 2.5 m / 135 shot
# sits behind the south partition and both renderers see a wall (2026-09-24). Each scene's exposure is explained in
# configs/scenes.yaml (warehouse ISO 80, matched to the Blender still, box_platform ISO 20 under the default dome and sun).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
. scripts/paths.sh
RUN=""; XML=""; GPU=0; STRIDE=4; LABELS=(); TRACES=""; DEST=""; INSETS=(); ALL=0; SIZE=1280x960; PT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --scene-xml) XML="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --stride) STRIDE="$2"; shift 2 ;;
    --size) SIZE="$2"; shift 2 ;;
    --path-trace) PT=1; shift ;;
    --traces) TRACES="$2"; shift 2 ;;
    --out) DEST="$2"; shift 2 ;;
    --insets) INSETS=(--insets); shift ;;
    --all) ALL=1; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) if [ -z "$RUN" ]; then RUN="$1"; else LABELS+=("$1"); fi; shift ;;
  esac
done
[ -n "$RUN" ] || { echo "usage: $0 <run_dir> [--traces <dir>] ..." >&2; exit 2; }
[[ "$SIZE" =~ ^[0-9]+x[0-9]+$ ]] || { echo "--size wants WIDTHxHEIGHT, got $SIZE" >&2; exit 2; }
WH=(--width "${SIZE%x*}" --height "${SIZE#*x}")
TRACES="${TRACES:-$RUN/trace}"; DEST="${DEST:-$RUN}"
[ -d "$TRACES" ] || { echo "no state traces in $TRACES" >&2; exit 2; }
[ -x "$RENDER_PY" ] || { echo "RENDER_PY not executable: $RENDER_PY (scripts/setup_render_runtime.sh)" >&2; exit 1; }
# The run's Isaac stage, its scene copy and its look: line 1 the stage dir, line 2 the scene xml, line 3 the
# supersample factor, then the RTX settings.
ISAAC_ENTRY=$(PYTHONPATH="$LOCOMANI_ROOT/src" "$VENV_SIM_PY" -c '
import sys
from pathlib import Path
from locomani.pipeline import run_context
from locomani.pipeline.lines import line_of
run = Path(sys.argv[1]).resolve(); cfg = run_context.load_run_config(run); isaac = run_context.isaac_scene(cfg)
print(isaac["dir"]); print(line_of(cfg).scene_name(cfg, run) / "scene_locomani.xml"); print(isaac["supersample"])
print(*isaac["rtx_settings"], sep="\n")' "$RUN") \
  || { echo "$RUN: its scene has no usable Isaac stage (see above)" >&2; exit 1; }
SCENE=$(sed -n 1p <<<"$ISAAC_ENTRY"); XML="${XML:-$(sed -n 2p <<<"$ISAAC_ENTRY")}"
SUPERSAMPLE=$(sed -n 3p <<<"$ISAAC_ENTRY")
# The path tracer's 64 samples a pixel already anti-alias; the scene's 2x on top would quadruple its 1.5 s a frame.
[ "$PT" = 1 ] && SUPERSAMPLE=1
LOOK=(--supersample "$SUPERSAMPLE")
while IFS= read -r s; do [ -n "$s" ] && LOOK+=(--rtx-setting "$s"); done < <(tail -n +4 <<<"$ISAAC_ENTRY")
[ "$PT" = 1 ] && for s in /rtx/rendermode=PathTracing /rtx/pathtracing/spp=64 /rtx/pathtracing/totalSpp=64 \
  /rtx/pathtracing/clampSpp=0 /rtx/pathtracing/optixDenoiser/enabled=true; do LOOK+=(--rtx-setting "$s"); done
[ -f "$XML" ] || { echo "no scene xml $XML (pass --scene-xml)" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$GPU" OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$LOCOMANI_ROOT/src"
if [ ${#LABELS[@]} = 0 ]; then
  ALL_TRACES=("$TRACES"/*_state_trace.npz)
  if [ "$ALL" = 1 ]; then
    for t in "${ALL_TRACES[@]}"; do LABELS+=("$(basename "$t" _state_trace.npz)"); done
  else
    OK_LABELS=$("$VENV_SIM_PY" -c '
import sys
from locomani.pipeline import run_context
print(*run_context.successful_traces(sys.argv[1], sys.argv[2]), sep="\n")' "$RUN" "$TRACES") || exit 1
    [ -n "$OK_LABELS" ] && mapfile -t LABELS <<<"$OK_LABELS"
  fi
  echo "rendering ${#LABELS[@]} of ${#ALL_TRACES[@]} traces in $TRACES ($([ "$ALL" = 1 ] && echo every round || echo successful rounds only, --all for every one))"
fi
mkdir -p "$DEST/isaac" "$DEST/previews_isaac"
for L in "${LABELS[@]}"; do
  OUT="$DEST/isaac/$L"; mkdir -p "$OUT"
  GHOST=(); [ -f "$TRACES/${L}_ghost.npz" ] && GHOST=(--ghost-trace "$TRACES/${L}_ghost.npz")  # ghost-hands rule: third-person views only
  echo "=== $L mujoco_render $(date -u +%H:%M:%S)"
  "$RENDER_PY" -m locomani.sim.isaac.mujoco_render --state-trace "$TRACES/${L}_state_trace.npz" --scene-xml "$XML" \
    --out "$OUT" --stride "$STRIDE" --azimuth 225 --distance 2.2 "${WH[@]}" "${GHOST[@]}" "${INSETS[@]}" > "$OUT/mujoco_render.log" 2>&1; echo "mujoco_render rc=$?"
  echo "=== $L isaac_render $(date -u +%H:%M:%S)"
  "$RENDER_PY" -m locomani.sim.isaac.isaac_render --session "" --state-trace "$TRACES/${L}_state_trace.npz" --scene-xml "$XML" \
    --scene-dir "$SCENE" --cam-traj "$OUT/cam_traj.npz" --stride "$STRIDE" --out "$OUT" "${WH[@]}" "${LOOK[@]}" "${GHOST[@]}" "${INSETS[@]}" > "$OUT/isaac_render.stdout" 2>&1; echo "isaac_render rc=$?"
  if [ -f "$OUT/ego_isaac.mp4" ] && [ -f "$OUT/tracking_mujoco.mp4" ]; then
    ffmpeg -loglevel error -y -i "$OUT/tracking_mujoco.mp4" -i "$OUT/ego_isaac.mp4" -filter_complex "[0:v][1:v]hstack=inputs=2" \
      -c:v libx264 -crf 23 -an "$OUT/side_by_side.mp4"
    echo "side_by_side rc=$?"
    "$VENV_SIM_PY" -m locomani.pipeline.video_preview "$OUT/side_by_side.mp4" "$DEST/previews_isaac/${L}_side_by_side"
    echo "preview rc=$?"
  fi
done
echo "=== DONE $(date -u +%H:%M:%S)"
