#!/bin/bash
# Re-run ONLY the closed-loop eval stage of visible200_20260910, after fixing eval_closed_loop.py's
# src bootstrap. collect.done / convert.done / train.done all exist, so run_learn_pilot skips
# straight to eval and cannot retrain. Also captures the tree the eval runs against, at start and
# end, because config_snapshot.json describes the 07:02 collect stage.
#
# Note on `$?`: it is captured into a variable BEFORE any other command runs. The earlier runner
# wrote `echo "$(date ...) eval exit=$?"`, and the command substitution resets $?, so a failed eval
# was logged as "eval exit=0" -- silence is not success, and neither is a 0 you computed wrong.
set -u
cd /workspace/code/locomanipulation
E=outputs/experiments/visible200_20260910
LOG=$E/runner_eval2.log
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
snap() {
    local d="$E/eval_provenance/$1"; mkdir -p "$d"
    date -u +%FT%TZ > "$d/captured_utc"
    git rev-parse HEAD > "$d/git_sha"
    git status --porcelain > "$d/worktree_status"
    git diff HEAD > "$d/worktree.patch"
    PYTHONPATH=src python3 -c "
import json
from locomani.pipeline.provenance import source_digests
json.dump(source_digests(), open('$d/source_digests.json', 'w'), indent=1)
" 2>/dev/null || echo failed > "$d/source_digests.error"
}
for i in $(seq 1 120); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | awk '{print ($1 > 500) ? 1 : 0}')
    # comm filter, not a bare pgrep -f: this script's own subshell carries the pattern in its
    # command line and matches itself (memory pkill-self-match-in-shell-chains). awk's comm is
    # "awk", so the checker cannot count itself.
    live=$(ps -eo comm,args --no-headers | awk '($1 ~ /^python/ || $1 ~ /^g1_deploy/) && /run_sim_locomani\.py|g1_deploy_onnx_ref|run_data_exporter/ {n++} END {print n+0}')
    head=$(bash scripts/cpu_headroom.sh 5)
    echo "$(date -u +%T) wait $i: gpu0 busy=$busy instance procs=$live | $head" >> "$LOG"
    if [ "$busy" = "0" ] && [ "$live" = "0" ] && echo "$head" | grep -q "OK for a stack"; then break; fi
    sleep 60
done
echo "$(date -u +%FT%TZ) eval start" >> "$LOG"
snap at_eval_start2
$PY -u src/locomani/pipeline/run_learn_pilot.py \
    --config configs/pilot/learn_200.json --task configs/tasks/bottle_visible.json \
    --out "$E" --scene outputs/scene/sim \
    --set steps=10000 --set save_steps=10000 --set train_gpu=0 --set through=eval \
    --set stack_env=CUDA_VISIBLE_DEVICES=0,MUJOCO_EGL_DEVICE_ID=0 >> "$E/pilot_eval2.log" 2>&1
rc=$?
echo "$(date -u +%FT%TZ) eval exit=$rc" >> "$LOG"
snap at_eval_end2
bash scripts/stack/stop_instance.sh 0 >> "$LOG" 2>&1
echo "$(date -u +%FT%TZ) stack stopped" >> "$LOG"
