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

# Which objects get a privileged pelvis-frame pose column. This MUST be
# repeated here and not merely set by start_stack.sh: this script replaces the
# exporter process, and the fresh one inherits only what is exported into its
# environment -- so a chunked collection, which restarts the exporter between
# every chunk, gets the exporter's hardcoded "can,plate" fallback from chunk 2
# onwards. That is not a warning, it is 9 sessions of observation.can_pose =
# 0.0 with a real water_bottle standing next to it, and it cost the privileged
# channel of a 200-episode set before it was noticed. Same default source as
# start_stack.sh: the scene the sim actually booted.
if [ -z "${TF_POSE_OBJECTS:-}" ] && [ -f "$LOCOMANI_ROOT/outputs/scene/sim/scene_meta.json" ]; then
    TF_POSE_OBJECTS="$("$REPO/.venv_sim/bin/python" -c "
import json, sys
print(','.join(json.load(open(sys.argv[1])).get('objects', {})))
" "$LOCOMANI_ROOT/outputs/scene/sim/scene_meta.json" 2>/dev/null)"
fi
export TF_POSE_OBJECTS
echo "  pose columns: ${TF_POSE_OBJECTS:-<none>}" >&2

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
