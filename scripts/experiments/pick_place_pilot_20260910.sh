#!/bin/bash
# Pick-and-place pilot on GPU1 / instance 0 (user-authorised 2026-09-10): configs/tasks/pick_place.json,
# scene outputs/scene/pick_place, learn_200 pilot config with through=collect.
#   bash scripts/experiments/pick_place_pilot_20260910.sh <out_dir> <successes> <batch_rounds> <max_batches> <seed0>
# Waits for CPU headroom (>= 6 free cores, no throttling) before starting; GPU1 must be free (checked here too).
# Run from a scratch COPY of this file (memory never-edit-a-running-bash-script).
set -u
cd /workspace/code/locomanipulation
E=${1:?out dir}; N=${2:-10}; BR=${3:-10}; MB=${4:-3}; SEED=${5:-32000}
mkdir -p "$E"; LOG=$E/runner.log
echo "$(date -u +%FT%TZ) runner start ($N successes, $BR rounds/batch, $MB batches, seed $SEED)" >> "$LOG"
for i in $(seq 1 60); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | awk '{print ($1 > 500) ? 1 : 0}')
  head=$(bash scripts/cpu_headroom.sh 5)
  echo "$(date -u +%T) wait $i: gpu1 busy=$busy | $head" >> "$LOG"
  if [ "$busy" = "0" ] && echo "$head" | grep -q "OK for a stack"; then break; fi
  sleep 60
done
echo "$(date -u +%FT%TZ) launching" >> "$LOG"
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -u src/locomani/pipeline/run_learn_pilot.py \
  --config configs/pilot/learn_200.json --task configs/tasks/pick_place.json \
  --out "$E" --scene outputs/scene/pick_place \
  --set through=collect --set successes="$N" --set batch_rounds="$BR" --set max_batches="$MB" --set seed0="$SEED" \
  --set stack_env=CUDA_VISIBLE_DEVICES=1,MUJOCO_EGL_DEVICE_ID=1 >> "$E/pilot.log" 2>&1
echo "$(date -u +%FT%TZ) pilot exit=$?" >> "$LOG"
bash scripts/stack/stop_instance.sh 0 >> "$LOG" 2>&1
echo "$(date -u +%FT%TZ) stack stopped" >> "$LOG"
