#!/bin/bash
# Restart ONLY the data exporter, leaving the sim and the deploy running.
#
#   NEW_SESSION=$(bash scripts/restart_exporter.sh)
#
# Why this exists: the exporter leaks about 93 threads per episode, and this
# container's cgroup caps the whole instance at pids.max = 3840. Around the
# 33rd episode of a session the quota runs out and everything wedges -- the
# exporter stops logging, and the gym blocks in a futex forever (measured:
# 3061 exporter threads, pids.current 3707/3840, both the gym and the
# exporter frozen at the same instant while sim and deploy kept running).
#
# A session is created when the exporter starts, so restarting it both frees
# the threads and rolls to a fresh session -- which is exactly the unit the
# converter pairs a manifest with. Restarting the exporter alone takes ~10 s
# against ~40 s for the whole stack, and leaves the sim's compiled scene and
# the deploy's TensorRT engines untouched.
#
# Prints the new session directory on stdout.
set -u
# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
REPO="$SONIC_ROOT"
LOGS=$(locomani_session_root "${INSTANCE:-0}")
TASK_PROMPT="${1:-${TASK_PROMPT:-pick up the object}}"

before=$(ls -dt "$LOGS"/outputs/*/ 2>/dev/null | head -1)

# Kill by PID, never by a pkill pattern on this command line.
for pid in $(pgrep -f "run_data_""exporter"); do kill -9 "$pid" 2>/dev/null; done
sleep 2

WRIST_ARG=(--record-wrist-cameras)
if [ "${WRIST_CAMS:-1}" = "0" ]; then WRIST_ARG=(); fi
cd "$REPO" || exit 1
nohup "$REPO/.venv_data_collection/bin/python" gear_sonic/scripts/run_data_exporter.py \
    --task-prompt "$TASK_PROMPT" \
    --root-output-dir "$LOGS/outputs" \
    "${WRIST_ARG[@]}" \
    --no-text-to-speech \
    >> "$LOGS/exporter.log" 2>&1 &

# Wait for the NEW session directory rather than sleeping: "still starting"
# and "died on launch" look the same to a fixed sleep.
for _ in $(seq 1 30); do
    now=$(ls -dt "$LOGS"/outputs/*/ 2>/dev/null | head -1)
    if [ -n "$now" ] && [ "$now" != "$before" ]; then echo "$now"; exit 0; fi
    sleep 1
done
echo "restart_exporter: no new session appeared" >&2
exit 1
