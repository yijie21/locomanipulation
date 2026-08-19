#!/bin/bash
# Three sim stacks in parallel (INSTANCE=0,1,2), EPN episodes each.
# Iteration batches: --no-record (no exporter dependency on instances 1/2).
trap '' TERM HUP
cd /workspace/code/locomanipulation
PY=/workspace/code/locomanipulation/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
OUT=outputs/logs/parbatch
EPN="${EPN:-5}"
# GYM=1 runs the teleport grasp gym (grasp_gym.py, one round per invocation)
# instead of full navigation episodes — ~2x the grasp samples per wall-clock.
if [ "${GYM:-0}" = "1" ]; then EPCMD="grasp_gym.py --rounds 1"; else EPCMD="collect_episode.py --no-record"; fi
mkdir -p $OUT
rm -f $OUT/ep_*.log $OUT/result.txt outputs/logs/parbatch_done.flag

kill_instance() {
  local i=$1
  for port in $((5555 + 10*i)) $((5581 + 10*i)) $((5557 + 10*i)); do
    pid=$(ss -tlnp 2>/dev/null | grep ":$port " | grep -oP 'pid=\K[0-9]+' | head -1)
    [ -n "${pid:-}" ] && kill -9 "$pid" 2>/dev/null
  done
}

run_instance() {
  local i=$1
  local envs="INSTANCE=$i TF_SONIC_PORT=$((5556+10*i)) TF_KEYBOARD_PORT=$((5580+10*i)) TF_PRIVILEGED_PORT=$((5581+10*i)) TF_EPISODE_PORT=$((5582+10*i))"
  kill_instance $i
  sleep 2
  env INSTANCE=$i SIM_VIDEO_DIR=$PWD/outputs/videos/stream_i$i \
    nohup bash src/teleop_free/start_stack.sh > $OUT/stack_$i.log 2>&1 &
  for _ in $(seq 1 90); do
    grep -q "stack up" $OUT/stack_$i.log 2>/dev/null && break
    sleep 5
  done
  if ! grep -q "stack up" $OUT/stack_$i.log 2>/dev/null; then
    echo "EPISODE FAILED [stack] instance $i did not boot" > $OUT/ep_${i}_0.log
    return
  fi
  sleep 3
  for e in $(seq 1 "$EPN"); do
    ( cd src/teleop_free && eval env $envs OMP_NUM_THREADS=1 \
        $PY $EPCMD ) > $OUT/ep_${i}_$e.log 2>&1
    if grep -qE "fell|did not stabilise|ZMQError|Address already in use" $OUT/ep_${i}_$e.log; then
      kill_instance $i
      sleep 2
      env INSTANCE=$i SIM_VIDEO_DIR=$PWD/outputs/videos/stream_i$i \
        nohup bash src/teleop_free/start_stack.sh > $OUT/stack_$i.log 2>&1 &
      for _ in $(seq 1 90); do
        grep -q "stack up" $OUT/stack_$i.log 2>/dev/null && break
        sleep 5
      done
      sleep 3
    fi
  done
}

for i in 0 1 2; do run_instance $i & done
wait
grep -h EPISODE $OUT/ep_*.log | sed 's/ in .*//' | sort | uniq -c > $OUT/result.txt
echo DONE > outputs/logs/parbatch_done.flag
