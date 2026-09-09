#!/bin/bash
# Pilot: a few scripted two-handed crate picks on their OWN stack instance,
# with the third-person tracking video per episode.
#
#   bash scripts/run_wall_pilot.sh [rounds] [seed]
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
. /workspace/code/locomanipulation/scripts/paths.sh
ROUNDS="${1:-3}"
SEED="${2:-0}"
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
"$SONIC_ROOT/.venv_sim/bin/python" src/locomani/tasks/crate/bimanual_wall.py \
    --rounds "$ROUNDS" --seed "$SEED" --manifest "$MANIFEST" --label "wall_s${SEED}" \
    2>&1 | tee "$SIM_VIDEO_DIR/pilot_s${SEED}.log" | grep -v Warning
echo "[pilot] manifest: $MANIFEST"; ls -l "$SIM_VIDEO_DIR"/*.mp4 2>/dev/null | tail -n +1
