#!/usr/bin/env bash
# Hang guard for the 10-13 overnight queues: when the active SONIC log has not grown for 12 minutes while a stack is up,
# take the instance-0 stack down (the round in progress is lost, the queue goes on) and note it in the log. Stops when
# after_sonic_queue.sh writes AFTER DONE.
set -u
cd /workspace/code/locomanipulation || exit 1
O=outputs/controller_compare_20261013
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
while ! grep -q "AFTER DONE" $O/after.log 2>/dev/null; do
  sleep 60
  pgrep -f "g1_deploy_onnx_ref" >/dev/null || continue
  newest=$(ls -t $O/sonic_l2_p25.log $O/after.log $O/complex/sonic/*.log $O/p25/clips/sonic_*/*.log $O/p25/brake/sonic_*/*.log outputs/runs/controller_compare_*_2026101[3]/collect_00.log 2>/dev/null | head -1)
  [ -n "$newest" ] || continue
  age=$(( $(date +%s) - $(stat -c %Y "$newest") ))
  if [ "$age" -gt 720 ]; then
    echo "=== watchdog $(date -u +%T): $newest silent ${age}s with the stack up -> stack down" | tee -a $O/sonic_l2_p25.log >> $O/after.log
    PYTHONPATH=src timeout 120 $PY -m locomani.stack.cli down --instance 0 >> $O/after.log 2>&1
    sleep 30
    for p in $(pgrep -f "locomani.tasks.bottle.collect|locomani.tasks.crate.collect|scripts/plan_episode.py|sonic_clip.py|diag_brake"); do
      kill -TERM "$p" 2>/dev/null
    done
  fi
done
