#!/bin/bash
# Full 200-success visible-expert collection + LeRobot conversion + check_dataset audit
# (configs/tasks/bottle_visible.json + configs/pilot/learn_200.json, through=convert, seed0 30000,
# first 200-episode run with restart_deploy=false). Waits for GPU0 and CPU headroom, instance 0 / GPU0.
# Source of outputs/experiments/visible200_20260910/launch.sh.
set -u
cd /workspace/code/locomanipulation
E=outputs/experiments/visible200_20260910
LOG=$E/runner.log
echo "$(date -u +%FT%TZ) runner start" >> "$LOG"
for i in $(seq 1 120); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | awk '{print ($1 > 500) ? 1 : 0}')
  head=$(bash scripts/cpu_headroom.sh 5)
  echo "$(date -u +%T) wait $i: gpu0 busy=$busy | $head" >> "$LOG"
  if [ "$busy" = "0" ] && echo "$head" | grep -q "OK for a stack"; then break; fi
  sleep 60
done
echo "$(date -u +%FT%TZ) launching" >> "$LOG"
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -u src/locomani/pipeline/run_learn_pilot.py \
  --config configs/pilot/learn_200.json --task configs/tasks/bottle_visible.json \
  --out "$E" --scene outputs/scene/sim \
  --set through=convert \
  --set stack_env=CUDA_VISIBLE_DEVICES=0,MUJOCO_EGL_DEVICE_ID=0 >> "$E/pilot.log" 2>&1
echo "$(date -u +%FT%TZ) pilot exit=$?" >> "$LOG"
bash scripts/stack/stop_instance.sh 0 >> "$LOG" 2>&1
echo "$(date -u +%FT%TZ) stack stopped" >> "$LOG"
