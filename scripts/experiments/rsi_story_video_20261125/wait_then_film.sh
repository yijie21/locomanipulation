#!/bin/bash
# wait_then_film.sh: waits for the sealed run (pid 1545866) to end (polls every 3 min), does the pre-start checks (GPU memory, other sim
# processes listed, SAM 3 alive else started from the worktree's sam3_server and noted), then runs run_all.sh.
# Launch: nohup setsid bash wait_then_film.sh > /workspace/tmp/storyvid_art/wait_then_film.log 2>&1 < /dev/null &
HERE=$(cd "$(dirname "$0")" && pwd)
echo "$(date +%T) waiting for 1545866"
while kill -0 1545866 2>/dev/null; do sleep 180; done
echo "$(date +%T) sealed run ended"
for i in $(seq 1 24); do   # GPU 0 needs room for an Isaac server (~8 GB), GPU 1 for the depth service
  f0=$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits); f1=$(nvidia-smi --id=1 --query-gpu=memory.free --format=csv,noheader,nounits)
  echo "$(date +%T) gpu free MiB: gpu0 $f0 gpu1 $f1"
  [ "$f0" -gt 14000 ] && [ "$f1" -gt 8000 ] && break
  sleep 300
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
echo "--- other sim-ish processes (listing only):"; ps -eo pid,pgid,etime,cmd | grep -E "official_server|bench/simple/server|stereo_server|bench.simple|rsi/(sealed|loop)|run_q" | grep -v grep | cut -c1-200
if ! (echo > /dev/tcp/127.0.0.1/18832) 2>/dev/null; then
  echo "$(date +%T) SAM 3 DOWN: starting it from the worktree (noted for the report)"
  ( cd /workspace/code/locomanipulation-bench-loop && PYTHONPATH=src CUDA_VISIBLE_DEVICES=1 nohup setsid /workspace/miniconda3/envs/sam3d/bin/python -s -m locomani.perception.sam3_server --port 18832 > /workspace/tmp/storyvid_art/sam3_started_by_recorder.log 2>&1 < /dev/null & )
  for i in $(seq 1 60); do (echo > /dev/tcp/127.0.0.1/18832) 2>/dev/null && break; sleep 5; done
else echo "SAM 3 alive"; fi
bash $HERE/run_all.sh
echo "$(date +%T) run_all rc=$?"
