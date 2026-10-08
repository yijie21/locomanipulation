#!/bin/bash
# The Unreal editor under a guard: run it headless on one GPU, pinned to a few cores, and kill every UE process if
# the container is truly saturated or the GPU hangs. scripts/start_unreal.sh starts this in the background; one
# event per line on stdout ($UNREAL_ROOT/logs/guard.log), the editor's own log in $UNREAL_ROOT/logs/editor_run.log.
#
#   bash scripts/unreal/editor_guard.sh <gpu>          # env: UE_CPUS (default the last 4 cores), UE_EXTRA_ARGS
#
# Pinned (taskset, nice 19) so it cannot starve SONIC stacks: the container's CPU quota is shared, and throttling
# the cgroup throttles every process in it; its first shader compile otherwise fills 15 cores within 30 s
# (2026-09-28). Saturated = average use above 90 % of the quota over 5 s, twice in a row, with the editor's own
# processes using at least one core of it; throttled time alone is
# no signal, the stacks' bursty thread pools are throttled 1-2.4 s per 5 s with no UE running. GPU hang = the
# editor's own processes at >= 95 % of the GPU (nvidia-smi pmon; the whole card where pmon is missing) while the
# editor log is silent for 90 s (a virtual-shadow-map hang, 2026-09-28).
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

# -nocef: no embedded Chromium (CEF). It is only the editor's web views, which nothing here opens, and on a desktop
# host it aborted the editor at startup: a Chromium CHECK ends in int3, so the process dies of SIGTRAP right after
# "LogCEFBrowser: CEF GPU acceleration enabled", outside UE's crash handler (SRA0889, 2026-09-29).
# shellcheck disable=SC2086  # UE_EXTRA_ARGS is a word list
taskset -c "$UE_CPUS" nice -n 19 "$EDITOR_BIN" "$UE_PROJECT" \
    -RenderOffscreen -vulkan -graphicsadapter="$GPU" -nocef \
    -unattended -nosplash -nosound -stdout -FullStdOutLogOutput -log \
    -LocalDataCachePath="$UNREAL_ROOT/ddc" ${UE_EXTRA_ARGS:-} "-ExecCmds=py $CMD_SERVER" > "$LOG" 2>&1 &
ED=$!
echo "editor started pid $ED on GPU $GPU, cores $UE_CPUS, at $(date -u +%T)"
# no cpu.max = no quota (a bare host, not a container); the redirect is silenced before it is opened
read -r quota period 2>/dev/null < /sys/fs/cgroup/cpu.max || quota=max
limit=$([ "$quota" = max ] && echo 0 || awk -v q="$quota" -v p="$period" 'BEGIN{printf "%.2f", 0.9 * q / p}')
bad=0
# CPU seconds (in clock ticks) of every UE process: utime + stime, fields 14 and 15 of /proc/<pid>/stat
ue_ticks() {
  local p t=0
  for p in $(pgrep -f "^$UE_ENGINE/Engine/Binaries/Linux/"); do
    t=$((t + $(awk '{print $14 + $15}' "/proc/$p/stat" 2>/dev/null || echo 0)))
  done
  echo $t
}
HZ=$(getconf CLK_TCK)
u_prev=$(awk '/^usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat 2>/dev/null || echo 0)
t_prev=$(ue_ticks)
while kill -0 $ED 2>/dev/null; do
  sleep 5
  u=$(awk '/^usage_usec/{print $2}' /sys/fs/cgroup/cpu.stat 2>/dev/null || echo 0)
  used=$(awk -v d=$((u - u_prev)) 'BEGIN{printf "%.2f", d / 5e6}'); u_prev=$u
  t=$(ue_ticks)
  mine=$(awk -v d=$((t - t_prev)) -v hz="$HZ" 'BEGIN{printf "%.2f", d / hz / 5}'); t_prev=$t
  # saturated and the editor is part of it (>= 1 core): an idle editor frees nothing when killed, and the
  # container's other users (a convert, another project) saturated it under an idle editor (killed 2026-10-03 17:32)
  if awk -v x="$used" -v l="$limit" -v m="$mine" 'BEGIN{exit !(l > 0 && x > l && m >= 1)}'; then
    bad=$((bad+1)); echo "saturated: ${used} cores, the editor ${mine} ($bad)"
  else bad=0; fi
  if [ $bad -ge 2 ]; then echo "KILLING UE: container saturated"; kill_ue; break; fi
  age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
  if [ $age -ge 90 ]; then
    # the editor's own share of the card (nvidia-smi pmon, its processes' SM %): the card is shared, and another
    # project pegging it while the idle editor's log is quiet read as a hang (killed 2026-10-03 16:51 mid-smoke).
    # Without pmon, the whole card's utilisation, as before.
    ue_pids=" $(pgrep -d ' ' -f "^$UE_ENGINE/Engine/Binaries/Linux/") "
    # a process with no sample in pmon's window shows "-": it used nothing, 0 %
    util=$(nvidia-smi pmon -c 1 -s u -i "$GPU" 2>/dev/null |
      awk -v p="$ue_pids" '/^# gpu/ {ok = 1} $1 !~ /^#/ && index(p, " " $2 " ") && $4 ~ /^[0-9]+$/ {s += $4} END {if (ok) print s + 0}')
    what="the editor's processes"
    [ -n "$util" ] || { util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | tr -d ' '); what="the card"; }
    if [ "${util:-0}" -ge 95 ]; then echo "KILLING UE: GPU $GPU at ${util}% ($what) with log silent ${age}s (hang)"; kill_ue; break; fi
  fi
done
wait $ED 2>/dev/null; echo "editor exited at $(date -u +%T)"
