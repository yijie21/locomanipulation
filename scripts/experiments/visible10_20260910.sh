#!/bin/bash
# 10-episode look at the visible-expert collection protocol (configs/tasks/bottle_visible.json,
# configs/pilot/learn_200.json, through=collect). Waits for GPU0 / CPU headroom, then runs
# run_learn_pilot on instance 0 / GPU0. Source of outputs/experiments/visible10_20260910/launch.sh.
set -u
cd /workspace/code/locomanipulation
E=outputs/experiments/visible10_20260910
LOG=$E/runner.log
echo "$(date -u +%FT%TZ) runner start" >> "$LOG"
# wait for the other session's Isaac render on GPU0 and for CPU headroom (>= 6 free cores, no throttle)
for i in $(seq 1 120); do
  # GPU0 memory, not pgrep: stale shell wrappers carry the process names in their argv (pkill self-match trap)
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | awk '{print ($1 > 500) ? 1 : 0}')
  head=$(bash scripts/cpu_headroom.sh 5)
  echo "$(date -u +%T) wait $i: gpu0 procs=$busy | $head" >> "$LOG"
  if [ "$busy" = "0" ] && echo "$head" | grep -q "OK for a stack"; then break; fi
  sleep 60
done
echo "$(date -u +%FT%TZ) launching" >> "$LOG"
third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -u src/locomani/pipeline/run_learn_pilot.py \
  --config configs/pilot/learn_200.json --task configs/tasks/bottle_visible.json \
  --out "$E" --scene outputs/scene/sim \
  --set through=collect --set successes=10 --set batch_rounds=10 --set max_batches=3 --set seed0=31000 \
  --set stack_env=CUDA_VISIBLE_DEVICES=0,MUJOCO_EGL_DEVICE_ID=0 >> "$E/pilot.log" 2>&1
echo "$(date -u +%FT%TZ) pilot exit=$?" >> "$LOG"
bash scripts/stack/stop_instance.sh 0 >> "$LOG" 2>&1
echo "$(date -u +%FT%TZ) stack stopped" >> "$LOG"
