#!/bin/bash
# Start the Unreal editor that renders `images.source: unreal` runs, and wait until it takes commands.
#
#   bash scripts/start_unreal.sh --gpu 1         # [--cpus 28-31]; prints "unreal editor ready" and returns
#   bash scripts/stop_unreal.sh
#
# One editor per machine (its command server and URLab's bridge have fixed ports), long-lived: start it once,
# then build levels (scripts/build_scene.sh <scene> --unreal), fill datasets (render_dataset_cameras_unreal.sh)
# and serve evaluations (run_unreal_mirror.sh) against it. It renders on --gpu; keep it off the stack's card, as
# the Isaac mirror is. It opens the warehouse room level; a render opens its scene's level itself. Already
# running: says so and returns. Runs under scripts/unreal/editor_guard.sh (pinned cores, saturation / GPU-hang
# kill); logs in $UNREAL_ROOT/logs/. First start compiles shaders for minutes; later ones take ~15 s.
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
GPU="" CPUS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu) GPU="${2:?--gpu needs a value}"; shift 2 ;;
        --cpus) CPUS="${2:?--cpus needs a value}"; shift 2 ;;
        *) echo "usage: $(basename "$0") --gpu N [--cpus A-B]" >&2; exit 1 ;;
    esac
done
[ -n "$GPU" ] || { echo "usage: $(basename "$0") --gpu N [--cpus A-B]" >&2; exit 1; }
if pgrep -f "^$UE_ENGINE/Engine/Binaries/Linux/UnrealEditor" > /dev/null; then
    echo "an Unreal editor is already running ($(pgrep -f "^$UE_ENGINE/Engine/Binaries/Linux/UnrealEditor" | head -1)); bash scripts/stop_unreal.sh to restart it"
    exit 0
fi
LOG="$UNREAL_ROOT/logs/editor_run.log"
mkdir -p "$UNREAL_ROOT/logs"
rm -f "$LOG"
[ -n "$CPUS" ] && export UE_CPUS="$CPUS"
nohup setsid bash "$LOCOMANI_ROOT/scripts/unreal/editor_guard.sh" "$GPU" > "$UNREAL_ROOT/logs/guard.log" 2>&1 &
for _ in $(seq 1 900); do
    grep -q 'uecmd\] listening' "$LOG" 2>/dev/null && { echo "unreal editor ready on GPU $GPU (log $LOG)"; exit 0; }
    grep -q "editor exited\|KILLING UE\|^no " "$UNREAL_ROOT/logs/guard.log" 2>/dev/null && break
    sleep 2
done
echo "the Unreal editor did not come up; see $UNREAL_ROOT/logs/guard.log and $LOG" >&2
tail -5 "$UNREAL_ROOT/logs/guard.log" >&2
exit 1
