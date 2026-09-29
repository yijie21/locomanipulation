#!/bin/bash
# The Unreal editor under a guard: run it headless on one GPU, pinned to a few cores, and kill every UE process if
# the container is truly saturated or the GPU hangs. scripts/start_unreal.sh starts this in the background; one
# event per line on stdout ($UNREAL_ROOT/logs/guard.log), the editor's own log in $UNREAL_ROOT/logs/editor_run.log.
#
#   bash scripts/unreal/editor_guard.sh <gpu>          # env: UE_CPUS (default the last 4 cores), UE_EXTRA_ARGS
#
# Pinned (taskset, nice 19) so it cannot starve SONIC stacks: the container's CPU quota is shared, and throttling
# the cgroup throttles every process in it; its first shader compile otherwise fills 15 cores within 30 s
# (2026-09-28). Saturated = average use above 90 % of the quota over 5 s, twice in a row; throttled time alone is
# no signal, the stacks' bursty thread pools are throttled 1-2.4 s per 5 s with no UE running. GPU hang = the GPU
# pegged at >= 95 % while the editor log is silent for 90 s (a virtual-shadow-map hang, 2026-09-28).
# The editor runs in_editor/cmd_server.py (Python over TCP 18801, the editor viewport off) and loads nothing else.
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/../paths.sh"
GPU=${1:?usage: editor_guard.sh <gpu>}
N=$(nproc)
UE_CPUS=${UE_CPUS:-$((N > 4 ? N - 4 : 0))-$((N - 1))}
LOGS="$UNREAL_ROOT/logs"; LOG="$LOGS/editor_run.log"
mkdir -p "$LOGS" "$UNREAL_ROOT/home" "$UNREAL_ROOT/tmp" "$UNREAL_ROOT/ddc"
# Everything UE writes under $HOME (~/.config/Epic, the Zen DDC, NVIDIA shader caches) goes under UNREAL_ROOT:
# a container root is often a few GB.
export HOME="$UNREAL_ROOT/home" TMPDIR="$UNREAL_ROOT/tmp"
export XDG_CACHE_HOME="$HOME/.cache" XDG_CONFIG_HOME="$HOME/.config" XDG_DATA_HOME="$HOME/.local/share"
EDITOR_BIN="$UE_ENGINE/Engine/Binaries/Linux/UnrealEditor"
[ -x "$EDITOR_BIN" ] || { echo "no Unreal editor at $EDITOR_BIN (UE_ENGINE; docs/setup.md, the Unreal step)"; exit 1; }
[ -f "$UE_PROJECT" ] || { echo "no host project $UE_PROJECT (UE_PROJECT; scripts/setup_unreal.sh)"; exit 1; }
CMD_SERVER="$LOCOMANI_ROOT/src/locomani/sim/unreal/in_editor/cmd_server.py"
kill_ue() { pkill -KILL -f "^$UE_ENGINE/Engine/Binaries/Linux/"; }

# shellcheck disable=SC2086  # UE_EXTRA_ARGS is a word list
taskset -c "$UE_CPUS" nice -n 19 "$EDITOR_BIN" "$UE_PROJECT" \
    -RenderOffscreen -vulkan -graphicsadapter="$GPU" \
    -unattended -nosplash -nosound -stdout -FullStdOutLogOutput -log \
    -LocalDataCachePath="$UNREAL_ROOT/ddc" ${UE_EXTRA_ARGS:-} "-ExecCmds=py $CMD_SERVER" > "$LOG" 2>&1 &
ED=$!
echo "editor started pid $ED on GPU $GPU, cores $UE_CPUS, at $(date -u +%T)"
read -r quota period < /sys/fs/cgroup/cpu.max 2>/dev/null || quota=max
limit=$([ "$quota" = max ] && echo 0 || awk -v q="$quota" -v p="$period" 'BEGIN{printf "%.2f", 0.9 * q / p}')
bad=0
u_prev=$(awk '/^usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat 2>/dev/null || echo 0)
while kill -0 $ED 2>/dev/null; do
  sleep 5
  u=$(awk '/^usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat 2>/dev/null || echo 0)
  used=$(awk -v d=$((u - u_prev)) 'BEGIN{printf "%.2f", d / 5e6}'); u_prev=$u
  if awk -v x="$used" -v l="$limit" 'BEGIN{exit !(l > 0 && x > l)}'; then bad=$((bad+1)); echo "saturated: ${used} cores ($bad)"; else bad=0; fi
  if [ $bad -ge 2 ]; then echo "KILLING UE: container saturated"; kill_ue; break; fi
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' ')
  age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
  if [ "${util:-0}" -ge 95 ] && [ $age -ge 90 ]; then echo "KILLING UE: GPU $GPU at ${util}% with log silent ${age}s (hang)"; kill_ue; break; fi
done
wait $ED 2>/dev/null; echo "editor exited at $(date -u +%T)"
