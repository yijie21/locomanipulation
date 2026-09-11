#!/bin/bash
# Pilot: a few scripted two-handed crate picks on their OWN stack instance,
# with the third-person tracking video per episode.
#
#   bash scripts/run_wall_pilot.sh [rounds] [seed] [-- extra args to bimanual_wall.py]
#
# Any args after [rounds] [seed] are forwarded verbatim to bimanual_wall.py,
# e.g. `bash scripts/run_wall_pilot.sh 10 0 --set wall_walk=true` or
# `--config configs/tasks/crate_wall.json` (the canonical wall-pilot config:
# wall_feature=rail, wall_walk=true -- the GPU gate's crate baseline was
# recorded with WALL_WALK=1 WALL_FEATURE=rail, from the days those were env
# vars; CrateTaskConfig.wall_walk now defaults to False, so reproducing that
# baseline needs this config or an explicit --set).
#
# Instance 1 (ports 5566/5567/5590/5591/5592, DDS domain 1), the crate scene
# from bimanual_scene.py via LOCOMANI_SIM_DIR, no exporter (start_stack skips
# it for INSTANCE != 0), and everything on GPU 1 so a collection or eval on
# instance 0 / GPU 0 is left alone. Stops ONLY its own processes at the end:
# stop_stack.sh's pkill is unscoped and would take down instance 0 too.
set -uo pipefail
# INSTANCE must be fixed BEFORE paths.sh is sourced: paths.sh derives every
# port from it and defaults it to 0, and 0 is the collection/eval stack. The
# first run of this script sourced paths.sh first, inherited instance 0, and
# tried to start a second stack on the live one (2026-09-07 06:00).
export INSTANCE="${INSTANCE:-1}"
[ "$INSTANCE" != "0" ] || { echo "refusing INSTANCE=0: that is the collection/eval stack"; exit 2; }
. "$(dirname "$0")/paths.sh"
# WALL_WALK / WALL_FEATURE stopped being read on 2026-09-09 (refactor step 3:
# CrateTaskConfig); a run that still sets them silently gets the ready-stand
# default (2026-09-10: 10 rendered episodes without a walk). Refuse, don't warn.
for dead in WALL_WALK WALL_FEATURE; do
    [ -z "${!dead:-}" ] || { echo "refusing: $dead is no longer read; pass --config configs/tasks/crate_wall.json or --set wall_walk=true / wall_feature=..."; exit 2; }
done
ROUNDS="${1:-3}"
SEED="${2:-0}"
shift $(($# < 2 ? $# : 2))  # drop [rounds] [seed]; "$@" now holds only the extra args to forward
export LOCOMANI_SIM_DIR="${LOCOMANI_SIM_DIR:-$LOCOMANI_ROOT/outputs/scene/crate_bimanual}"
export SIM_VIDEO_DIR="${SIM_VIDEO_DIR:-$LOCOMANI_ROOT/outputs/videos/wall_pilot}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-1}"
export SCENE=locomani
# A clean third-person video: the default overlay draws the target's pose box
# as a point cloud, which for a 60 cm crate hides the whole robot.
export SIM_VIS="${SIM_VIS:-none}"
MANIFEST="$SIM_VIDEO_DIR/manifest.jsonl"
[ -f "$LOCOMANI_SIM_DIR/scene_meta.json" ] || {
    echo "no crate scene at $LOCOMANI_SIM_DIR; build it:"
    echo "  $SONIC_ROOT/.venv_sim/bin/python src/locomani/tasks/crate/bimanual_scene.py $LOCOMANI_SIM_DIR"; exit 1; }
mkdir -p "$SIM_VIDEO_DIR"
LOGS=$(locomani_session_root "$INSTANCE")
SIM_PORT=$((5556 + 10 * INSTANCE))

stop_mine() {
    # Only the processes of THIS instance, matched on the exact instance
    # number / port, and never instance 0 whatever the variables say.
    [ "$INSTANCE" != "0" ] || return
    # The sim forks an image-publisher child that outlives its parent and
    # keeps the camera port (found the hard way: the second pilot start died
    # on 'Address already in use' 5575). Kill children first, then parents.
    for pid in $(pgrep -f "run_sim_locomani.py --instance $INSTANCE( |$)") \
               $(pgrep -f "zmq-port $SIM_PORT( |$)"); do
        for kid in $(pgrep -P "$pid"); do kill -9 "$kid" 2>/dev/null && echo "[pilot] killed child $kid"; done
        kill -9 "$pid" 2>/dev/null && echo "[pilot] killed $pid"
    done
    for pid in $(ss -ltnp 2>/dev/null | grep -E ":($((5555 + 10 * INSTANCE))|$((5556 + 10 * INSTANCE))|$((5557 + 10 * INSTANCE))|$((5580 + 10 * INSTANCE))|$((5581 + 10 * INSTANCE))|$((5582 + 10 * INSTANCE))) " | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u); do
        kill -9 "$pid" 2>/dev/null && echo "[pilot] killed port holder $pid"
    done
    echo "[pilot] instance $INSTANCE stopped"
}
trap stop_mine EXIT

echo "[pilot] instance $INSTANCE  scene $LOCOMANI_SIM_DIR  video -> $SIM_VIDEO_DIR  logs $LOGS"
bash "$LOCOMANI_STACK/start_stack.sh" || { echo "[pilot] stack failed to start; see $LOGS"; exit 1; }

cd "$LOCOMANI_ROOT"
# The aimed hand poses go next to the state trace (<label>_NN_ghost.npz) so the
# side-by-side render can draw the ghost hands; GHOST_TRACE=0 turns it off.
GHOST_ARGS=()
[ -n "${SIM_STATE_TRACE_DIR:-}" ] && [ "${GHOST_TRACE:-1}" != "0" ] && GHOST_ARGS=(--ghost-trace "$SIM_STATE_TRACE_DIR")
"$SONIC_ROOT/.venv_sim/bin/python" src/locomani/tasks/crate/bimanual_wall.py \
    --rounds "$ROUNDS" --seed "$SEED" --manifest "$MANIFEST" --label "wall_s${SEED}" "${GHOST_ARGS[@]}" "$@" \
    2>&1 | tee "$SIM_VIDEO_DIR/pilot_s${SEED}.log" | grep -v Warning
echo "[pilot] manifest: $MANIFEST"; ls -l "$SIM_VIDEO_DIR"/*.mp4 2>/dev/null | tail -n +1
