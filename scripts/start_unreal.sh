#!/bin/bash
# Start the Unreal editor that renders `images.source: unreal` runs, and wait until it takes commands.
#
#   bash scripts/start_unreal.sh --gpu 1         # [--cpus 28-31]; prints "unreal editor ready" and returns
#   bash scripts/stop_unreal.sh
#
# One editor per machine (its command server and URLab's bridge have fixed ports), long-lived: start it once,
# then build levels (scripts/build_scene.sh <scene> --unreal), fill datasets (render_dataset_cameras_unreal.sh)
# and serve evaluations (run_unreal_mirror.sh) against it. It renders on --gpu; keep it off the stack's card, as
# the Isaac mirror is. It opens the warehouse room level; a render opens its scene's level itself. Runs under
# scripts/unreal/editor_guard.sh (pinned cores, saturation / GPU-hang kill); logs in $UNREAL_ROOT/logs/. First
# start compiles shaders for minutes; later ones take ~15 s. Its ports (URLab's PortBase, 5800-5812, and the
# command server's 18801) stay clear of every MuJoCo stack instance.
# --gpu is a Vulkan device index; the editor's context is then looked up in nvidia-smi and must be on nvidia-smi
# GPU N (they agreed on this machine, 2026-09-29, but the orders need not). An editor already running on another
# card is refused, not reused: bash scripts/stop_unreal.sh first.
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
# the nvidia-smi index of the GPU where the editor holds the most memory (its rendering context)
editor_gpu() {
    local uuid
    uuid=$(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits |
        awk -F', ' -v p="$1" '$1 == p {print $3, $2}' | sort -rn | head -1 | cut -d' ' -f2)
    [ -n "$uuid" ] && nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v u="$uuid" '$2 == u {print $1}'
}
RUNNING=$(pgrep -f "^$UE_ENGINE/Engine/Binaries/Linux/UnrealEditor" | head -1)
if [ -n "$RUNNING" ]; then
    ON=$(editor_gpu "$RUNNING")
    if [ "$ON" != "$GPU" ]; then
        echo "an Unreal editor is already running (pid $RUNNING) on GPU ${ON:-?}, not $GPU: bash scripts/stop_unreal.sh, then start it on $GPU" >&2
        exit 1
    fi
    echo "an Unreal editor is already running (pid $RUNNING) on GPU $ON; using it"
    exit 0
fi
LOG="$UNREAL_ROOT/logs/editor_run.log"
mkdir -p "$UNREAL_ROOT/logs"
rm -f "$LOG"
[ -n "$CPUS" ] && export UE_CPUS="$CPUS"
nohup setsid bash "$LOCOMANI_ROOT/scripts/unreal/editor_guard.sh" "$GPU" > "$UNREAL_ROOT/logs/guard.log" 2>&1 &
for _ in $(seq 1 900); do
    if grep -q 'uecmd\] listening' "$LOG" 2>/dev/null; then
        PID=$(pgrep -f "^$UE_ENGINE/Engine/Binaries/Linux/UnrealEditor" | head -1)
        ON=$(editor_gpu "$PID")
        if [ "$ON" != "$GPU" ]; then
            echo "Vulkan device $GPU is nvidia-smi GPU ${ON:-?} here; stopping the editor: pass the Vulkan index of the card you want" >&2
            bash "$LOCOMANI_ROOT/scripts/stop_unreal.sh" > /dev/null
            exit 1
        fi
        echo "unreal editor ready on GPU $GPU (log $LOG)"
        exit 0
    fi
    grep -q "editor exited\|KILLING UE\|^no " "$UNREAL_ROOT/logs/guard.log" 2>/dev/null && break
    sleep 2
done
echo "the Unreal editor did not come up; see $UNREAL_ROOT/logs/guard.log and $LOG" >&2
tail -5 "$UNREAL_ROOT/logs/guard.log" >&2
exit 1
