#!/usr/bin/env bash
# 10-round bottle pilot on the VR 3-point channel (TeleopGraspPolicy), on the
# SAME fixed case bank ledger bottle.md 33 scored release 4/10 / v1.1 0/30 on
# with the coupled (IK + predicted squat) policy. Usage:
#   bash scripts/experiments/teleop_pilot_20260912.sh [policy/sonic_v1_1] [configs/tasks/bottle_visible_teleop.json] [tag]
# Instance 0 (the exporter lives there; --record produces a LeRobot-convertible
# session), GPU from CUDA_VISIBLE_DEVICES (default 0). Stop early: kill the runner PID
# (its INT/TERM trap stops the instance); never pkill -f a pattern from a shell whose
# own command line contains it.
set -uo pipefail
cd /workspace/code/locomanipulation
POLICY="${1:-policy/sonic_v1_1}"
TASKCFG="${2:-configs/tasks/bottle_visible_teleop.json}"
TAG="${3:-$(basename "$POLICY")}"
O="$PWD/outputs/experiments/teleop_pilot_20260912/$TAG"
# ledger 33's bank (release 4/10, v1.1 0/30 on the coupled policy). NOT
# v11_bottle_pilot_20260910/cases10_visible.json: that is the 09-10 "badspawn"
# bank, spawns 1.1-3.0 m, 6 of 10 die at the ego-view gate for any arm.
CASES="${CASES:-$PWD/outputs/experiments/retreat_seed10c_20260912/cases10_az374.json}"
[ -f "$CASES" ] || { echo "missing $CASES"; exit 1; }
export INSTANCE=0 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES:-0}"
export SCENE_RANDOM=0 SIM_VIS=bbox,readout
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
. scripts/paths.sh
export PYTHONPATH="$PWD/src"
PY="$VENV_SIM_PY"
PROMPT="pick up the water bottle"
mkdir -p "$O/videos"
clean() { bash scripts/stack/stop_instance.sh 0 >/dev/null 2>&1; sleep 3; }
alive() { ps -eo comm,args --no-headers | awk -v p="$1" '$1 ~ /^python/ && $0 ~ p {print}'; }
for pat in "run_sim_""locomani[.]py" "run_data_""exporter[.]py"; do
    hit=$(alive "$pat")
    if [ -n "$hit" ]; then echo "REFUSING: '$pat' already running"; echo "$hit" | cut -c1-140; exit 1; fi
done
trap 'clean; echo "ABORTED"; exit 130' INT TERM
echo "===== $TAG policy=$POLICY task=$TASKCFG $(date -u +%FT%TZ)"
clean
SIM_VIDEO_DIR="$O/videos" SONIC_POLICY="$POLICY" bash scripts/stack/start_stack.sh "$PROMPT" > "$O/stack.log" 2>&1
if ! grep -q "stack up" "$O/stack.log"; then echo "  stack failed, see $O/stack.log"; clean; exit 1; fi
echo "  stack up $(date -u +%H:%M:%S), running 10 fixed cases"
SONIC_POLICY="$POLICY" timeout 3600 "$PY" -u src/locomani/tasks/bottle/grasp_gym.py \
    --task-config "$TASKCFG" --cases "$CASES" --rounds 10 --record \
    --manifest "$O/manifest.jsonl" > "$O/gym.log" 2>&1
echo "  gym exit $? $(date -u +%H:%M:%S)"
grep -E "^\[gym\] [0-9]+/|^EPISODE " "$O/gym.log" | tail -12
cp "$(locomani_session_root 0)/deploy.log" "$O/deploy.log" 2>/dev/null
clean
printf '%-12s ' "$TAG"; grep -h -oE "^EPISODE (OK|FAILED \[[a-z_]+\])" "$O/gym.log" | sort | uniq -c | tr '\n' ' '
echo "  videos: $(ls "$O/videos"/*.mp4 2>/dev/null | wc -l)"
touch "$O/done"; echo ALLDONE
