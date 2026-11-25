#!/bin/bash
# run_all.sh [stories...]: Phase 2. Films every (task, version) of stories.json one session at a time (v0 then v1 per task),
# with taskset -c 24-31. Start it with:  nohup setsid bash run_all.sh > <log> 2>&1 < /dev/null &
# Refuses to start while the sealed run (pgid 1545866) is alive. Never touches SAM 3 (checks the port only).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/workspace/code/locomanipulation/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
WT=/workspace/code/locomanipulation-bench-loop
if kill -0 1545866 2>/dev/null; then echo "sealed run 1545866 still alive: not starting"; exit 4; fi
(echo > /dev/tcp/127.0.0.1/18832) 2>/dev/null || { echo "SAM 3 (18832) is down: not starting"; exit 5; }
unset EXAM_OPEN
export PYTHONPATH=$WT/src PYTHONDONTWRITEBYTECODE=1 RSV_TRIES=${RSV_TRIES:-3} LOCOMANI_TMP=/workspace/tmp
cd $WT   # cwd of the loop's rounds; nothing is written there
for task in OpenOvenTeleop CloseDoorTeleop OpenFaucetTeleop OpenTrashCanTeleop; do
  labels=$($PY -c "import json;print(','.join(s['label'] for s in json.load(open('$HERE/stories.json'))['stories'] if s['task']=='$task'))")
  for v in 0 1; do
    echo "=== $(date +%T) $task v$v $labels"
    taskset -c 24-31 $PY $HERE/record_story.py $v $task $labels
    echo "=== $(date +%T) $task v$v rc=$?"
  done
done
echo finished
