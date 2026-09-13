#!/bin/bash
# Side-by-side MuJoCo | Isaac videos of a wall-pilot run.
#
#   bash scripts/render_wall_pilot.sh <pilot_dir> <out_dir> [rounds=10] [label=wall_s0]
#
# Needs the pilot's state traces (<pilot_dir>/trace/<label>_NN_state_trace.npz,
# recorded with SIM_STATE_TRACE_DIR) and the Isaac scene assets for the same
# snapshot (bash scripts/build_isaac_assets.sh --snapshot outputs/scene/crate_bimanual).
# MUJOCO_RENDER_ARGS="--insets" / ISAAC_RENDER_ARGS="--insets" independently turn
# on each side's own picture-in-picture (ego + left/right wrist thumbnails); set
# either, both, or neither. Isaac's additionally needs --state-trace, which this
# script always passes, and roughly doubles that side's render time (see its own
# --insets --help for why: a fully separate second pass, not a cheap overlay).
# Per round: MuJoCo re-render with the tracking camera (+ cam_traj.npz), the Isaac
# replay on that camera path, then an ffmpeg hstack -> <out_dir>/sbs_roundNN.mp4.
set -uo pipefail
. "$(dirname "$0")/paths.sh"
P="${1:?pilot dir}"; O="${2:?out dir}"; N="${3:-10}"; LABEL="${4:-wall_s0}"
SCENE="${LOCOMANI_SIM_DIR:-$LOCOMANI_ROOT/outputs/scene/crate_bimanual}"
ISAAC_SCENE="$LOCOMANI_ROOT/assets/isaac/scene/$(basename "$SCENE")"
mkdir -p "$O"; cd "$LOCOMANI_ROOT"
for i in $(seq -f %02g 0 $((N - 1))); do
  d="$O/r$i"; mkdir -p "$d"
  T="$P/trace/${LABEL}_${i}_state_trace.npz"
  [ -f "$T" ] || { echo "r$i: no trace $T"; continue; }
  # Ghost hands (the IK's aimed wrist poses) in both renders, whenever the
  # expert wrote them. True by default only since 2026-09-11: before that the
  # runner passed --ghost-trace only if the CALLER had exported
  # SIM_STATE_TRACE_DIR, which it never set itself, so this comment was wrong.
  # run_crate_pilot.py writes <out>/trace unless --no-state-trace.
  GH=(); [ -f "$P/trace/${LABEL}_${i}_ghost.npz" ] && GH=(--ghost-trace "$P/trace/${LABEL}_${i}_ghost.npz")
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-1}" PYTHONPATH=src "$RENDER_PY" -m locomani.sim.isaac.mujoco_render \
    --state-trace "$T" --scene-xml "$SCENE/scene_locomani.xml" --out "$d" --stride 4 "${GH[@]}" ${MUJOCO_RENDER_ARGS:-} 2>&1 | grep -v Warning | tail -1
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src timeout 1500 "$RENDER_PY" -m locomani.sim.isaac.isaac_render \
    --session "" --episode 0 --state-trace "$T" --scene-xml "$SCENE/scene_locomani.xml" \
    --scene-dir "$ISAAC_SCENE" --cam-traj "$d/cam_traj.npz" --stride 4 "${GH[@]}" --out "$d" \
    ${ISAAC_RENDER_ARGS:-} > "$d/isaac_stdout.log" 2>&1
  echo "r$i: isaac exit $?  $(grep -E 'wrote|FAILED' "$d/render.log" | tail -1)"
  ffmpeg -y -loglevel error -i "$d/tracking_mujoco.mp4" -i "$d/ego_isaac.mp4" -filter_complex "[0:v][1:v]hstack=inputs=2" \
    -c:v libx264 -crf 23 -preset fast -an "$O/sbs_round$i.mp4" && echo "r$i: sbs ok"
done
echo RENDER_ALL_DONE
