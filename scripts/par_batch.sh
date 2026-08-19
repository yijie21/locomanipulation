#!/bin/bash
# Three sim stacks in parallel (INSTANCE=0,1,2), EPN episodes each.
# Iteration batches: --no-record (no exporter dependency on instances 1/2).
# GYM=1 runs the teleport grasp gym instead of full navigation episodes.
trap '' TERM HUP
cd /workspace/code/locomanipulation
PY=/workspace/code/locomanipulation/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
OUT=outputs/logs/parbatch
EPN="${EPN:-5}"
GYM="${GYM:-0}"
mkdir -p $OUT
rm -f $OUT/ep_*.log $OUT/result.txt outputs/logs/parbatch_done.flag

kill_instance() {
  local i=$1
  for port in $((5555 + 10*i)) $((5581 + 10*i)) $((5557 + 10*i)); do
    pid=$(ss -tlnp 2>/dev/null | grep ":$port " | grep -oP 'pid=\K[0-9]+' | head -1)
    [ -n "${pid:-}" ] && kill -9 "$pid" 2>/dev/null
  done
  # The exporter LISTENS on nothing (it subscribes), so the port sweep above
  # never kills it — 35 of them piled up across restarts and exhausted the
  # thread quota (zmq 'Resource temporarily unavailable'). Only instance 0
  # ever runs one; killing them all on any restart is safe.
  for pid in $(pgrep -f run_data_exporter); do kill -9 "$pid" 2>/dev/null; done
}

boot_stack() {
  local i=$1
  kill_instance $i
  sleep 2
  env INSTANCE=$i SIM_VIDEO_DIR=$PWD/outputs/videos/stream_i$i \
    nohup bash src/teleop_free/start_stack.sh > $OUT/stack_$i.log 2>&1 &
  for _ in $(seq 1 90); do
    grep -q "stack up" $OUT/stack_$i.log 2>/dev/null && return 0
    sleep 5
  done
  return 1
}

run_instance() {
  local i=$1
  local envs="INSTANCE=$i TF_SONIC_PORT=$((5556+10*i)) TF_KEYBOARD_PORT=$((5580+10*i)) TF_PRIVILEGED_PORT=$((5581+10*i)) TF_EPISODE_PORT=$((5582+10*i))"
  if ! boot_stack $i; then
    echo "EPISODE FAILED [stack] instance $i did not boot" > $OUT/ep_${i}_0.log
    return
  fi
  sleep 3
  if [ "$GYM" = "1" ]; then
    # One process for ALL rounds: the per-round cold start (imports + the
    # nominal-spawn calibration reset) costs ~25 s, and the gym resets the
    # scene itself between rounds anyway. A stack restart is only worth
    # 90-120 s when the STACK is actually broken (zmq/port errors, or the
    # gym died mid-run) — a fallen robot just resets and continues.
    local remaining=$EPN chunk=1
    while [ "$remaining" -gt 0 ] && [ "$chunk" -le 4 ]; do
      ( cd src/teleop_free && eval env $envs OMP_NUM_THREADS=1 \
          $PY grasp_gym.py --rounds "$remaining" ) > $OUT/ep_${i}_c$chunk.log 2>&1
      local done_n
      done_n=$(grep -c "^EPISODE" $OUT/ep_${i}_c$chunk.log 2>/dev/null || true)
      remaining=$((remaining - done_n))
      chunk=$((chunk + 1))
      if [ "$remaining" -gt 0 ]; then
        boot_stack $i || break
        sleep 3
      fi
    done
  else
    for e in $(seq 1 "$EPN"); do
      ( cd src/teleop_free && eval env $envs OMP_NUM_THREADS=1 \
          $PY collect_episode.py --no-record ) > $OUT/ep_${i}_$e.log 2>&1
      if grep -qE "fell|did not stabilise|ZMQError|Address already in use" $OUT/ep_${i}_$e.log; then
        boot_stack $i
        sleep 3
      fi
    done
  fi
}

for i in 0 1 2; do run_instance $i & done
wait
grep -h EPISODE $OUT/ep_*.log | sed 's/ in .*//' | sort | uniq -c > $OUT/result.txt
echo DONE > outputs/logs/parbatch_done.flag
