#!/usr/bin/env bash
# The reset-seating fix (ledger bottle.md 24): _seat_on_floor used the geom
# bounding-SPHERE radius as the foot's downward extent, so every reset in every
# task line seated the robot 10.8 cm too high and dropped it.
#
# The ACCEPTANCE is the landing transient, not the success rate (which 10 rounds
# cannot measure, ledger 23.3): displacement in the first second after the reset
# while the command is IDLE with a zero movement vector. Pooled baseline over 42
# recorded rounds -- release median 0.071 / max 0.211, v1.1 median 0.089-0.103
# with 4 rounds >= 0.33 and one fall. Nothing downstream can touch that
# statistic, so the 0.35 rad tolerance still in the tree cannot confound it.
set -uo pipefail
cd /workspace/code/locomanipulation
AB="$PWD/outputs/experiments/v11_bottle_ab20_20260910"
O="$PWD/outputs/experiments/staged_seat_20260911"
mkdir -p "$O/videos"
ROUNDS=10

export INSTANCE=0 CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0
export SCENE_RANDOM=0 SIM_VIS=bbox,readout
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
# The point of the run: the default checkpoint. Stated anyway so the log records it.
export SONIC_POLICY="${SONIC_POLICY:-policy/sonic_v1_1}"
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
PROMPT="pick up the water bottle"

clean() { bash scripts/stack/stop_instance.sh 0 >/dev/null 2>&1; sleep 3; }
trap 'clean; echo "ABORTED"; exit 130' INT TERM

alive() { ps -eo comm,args --no-headers | awk -v p="$1" '$1 ~ /^python/ && $0 ~ p {print}'; }
for pat in "run_sim_""locomani[.]py" "run_data_""exporter[.]py"; do
    hit=$(alive "$pat")
    [ -n "$hit" ] && { echo "REFUSING: '$pat' already running"; echo "$hit" | cut -c1-140; exit 1; }
done

[ -f "$O/done" ] && { echo "already done"; exit 0; }
clean
SIM_VIDEO_DIR="$O/videos" bash scripts/stack/start_stack.sh "$PROMPT" > "$O/stack.log" 2>&1
rc=$?
[ "$rc" = 3 ] && { echo "instance-0 lock held"; tail -3 "$O/stack.log"; exit 3; }
grep -q "stack up" "$O/stack.log" || { echo "stack failed, see $O/stack.log"; clean; exit 1; }
grep -m1 "policy:" "$O/stack.log"
echo "stack up $(date -u +%H:%M:%S), $ROUNDS fixed cases"
timeout 3600 "$PY" -u src/locomani/tasks/bottle/grasp_gym.py \
    --reach-config "$AB/reach_visible.json" \
    --task-config configs/tasks/bottle_visible.json \
    --targets water_bottle \
    --spawn-edge 1.1 1.3 --face-object --spawn-yaw-err 20.0 \
    --spawn-lateral 0.2 --edge-jitter 0.2 \
    --stand-retries 0 --record --video-include-init \
    --cases "$AB/cases20.json" --rounds "$ROUNDS" \
    --manifest "$O/manifest.jsonl" > "$O/gym.log" 2>&1
echo "gym exit $? $(date -u +%H:%M:%S)"
grep -E "^\[gym\] [0-9]+/" "$O/gym.log" | tail -2
clean
touch "$O/done"
echo "===== done $(date -u +%FT%TZ)"
