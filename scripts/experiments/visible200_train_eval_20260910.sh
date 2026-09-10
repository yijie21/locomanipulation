#!/bin/bash
# visible200_20260910: train GR00T 10k steps on GPU0, then the 100-round closed-loop eval (seed 6000, clean protocol,
# configs/eval/closed_loop_100.json) on the same GPU. Two driver invocations so the eval stage can wait for CPU headroom
# (the stack needs >= 6 free cores; training does not). Source of outputs/experiments/visible200_20260910/launch_train_eval.sh.
set -u
cd /workspace/code/locomanipulation
E=outputs/experiments/visible200_20260910
LOG=$E/runner_train_eval.log
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
COMMON="--config configs/pilot/learn_200.json --task configs/tasks/bottle_visible.json --out $E --scene outputs/scene/sim --set steps=10000 --set save_steps=10000 --set train_gpu=0 --set stack_env=CUDA_VISIBLE_DEVICES=0,MUJOCO_EGL_DEVICE_ID=0"
echo "$(date -u +%FT%TZ) train start" >> "$LOG"
$PY -u src/locomani/pipeline/run_learn_pilot.py $COMMON --set through=train >> "$E/pilot_train.log" 2>&1
echo "$(date -u +%FT%TZ) train exit=$?" >> "$LOG"
[ -f $E/train.done ] || { echo "$(date -u +%FT%TZ) no train.done, stopping" >> "$LOG"; exit 1; }
for i in $(seq 1 180); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | awk '{print ($1 > 500) ? 1 : 0}')
  head=$(bash scripts/cpu_headroom.sh 5)
  echo "$(date -u +%T) eval wait $i: gpu0 busy=$busy | $head" >> "$LOG"
  if [ "$busy" = "0" ] && echo "$head" | grep -q "OK for a stack"; then break; fi
  sleep 60
done
echo "$(date -u +%FT%TZ) eval start" >> "$LOG"
$PY -u src/locomani/pipeline/run_learn_pilot.py $COMMON --set through=eval >> "$E/pilot_eval.log" 2>&1
echo "$(date -u +%FT%TZ) eval exit=$?" >> "$LOG"
bash scripts/stack/stop_instance.sh 0 >> "$LOG" 2>&1
echo "$(date -u +%FT%TZ) stack stopped" >> "$LOG"
