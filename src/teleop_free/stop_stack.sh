#!/bin/bash
# Stop every process of the SONIC data-collection stack.
#
# Lives in a file on purpose: putting these patterns directly on a shell
# command line makes pkill match the very shell that is running it (the
# pattern appears in that shell's own /proc/<pid>/cmdline), so the killer
# kills itself and the real targets survive.

pkill -9 -x "g1_deploy_onnx_" 2>/dev/null      # comm is truncated to 15 chars
pkill -9 -f "run_sim_loop.py" 2>/dev/null
pkill -9 -f "run_sim_locomani.py" 2>/dev/null
pkill -9 -f "run_data_exporter.py" 2>/dev/null
pkill -9 -f "zmq_driver.py" 2>/dev/null
pkill -9 -f "collect_episode.py" 2>/dev/null
pkill -9 -f "diag_walk.py" 2>/dev/null
# grasp_gym.py binds the same :5556/:5580 pair zmq_driver.py does (it builds
# a Primitives -> AgentCommander like every diag_*.py script does too, see
# run_batch.py's docstring) -- left running, it holds those ports and the
# NEXT start_stack.sh fails with "Address already in use" even though this
# script reports everything stopped.
pkill -9 -f "grasp_gym.py" 2>/dev/null
pkill -9 -f "spawn_main" 2>/dev/null            # MuJoCo image-publish subprocess

# The sim binds :5555 and :5581; a survivor makes the next start fail with
# "Address already in use" *after* it has already printed its startup banner,
# which reads like a working stack until the deploy times out waiting for
# LowState. So wait for the ports, do not just wait a fixed time.
for _ in $(seq 1 15); do
    ss -lnt 2>/dev/null | grep -qE ":5555 |:5556 |:5557 |:5580 |:5581 " || break
    sleep 1
done

echo "remaining:"
pgrep -a -f "g1_deploy_onnx_ref|run_sim_loop|run_sim_locomani|run_data_exporter|zmq_driver|collect_episode|grasp_gym" \
  | grep -v stop_stack || echo "  (none)"

echo "ports:"
ss -lnt 2>/dev/null | grep -E ":5555 |:5556 |:5557 |:5580 |:5581 " || echo "  (all free)"
