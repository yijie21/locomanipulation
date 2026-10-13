#!/bin/bash
# TODO 24.4 / 24.6: bfm_grasp.py over the 65 objects in N processes (objects interleaved k::N so the load balances),
# each OMP_NUM_THREADS=2 (torch's sync spin otherwise takes 4-5 cores per process; the container has 15.36).
#   bash bfm_batch.sh [procs 4] [gpu 1] <out dir> [bfm_grasp.py flags]
#   bash bfm_batch.sh 4 1 outputs/wbc_compare_20261012/bfm_grasp/paired2 --select paired --pair-runs _wbc2 --video \
#       --rounds 2 --yaws 2 --bias outputs/wbc_compare_20261012/probe/bias.json
# Logs: <out dir>/run_<k>.log; every process rewrites <out dir>/summary.json with its own rounds (the per-round
# record.json under obj_XXX/round_RR is what compare_summary.py reads). Never beside a SONIC batch (DECISIONS 10-12).
set -u
cd "$(dirname "$0")/../../.." || exit 1
PROCS=${1:-4}; GPU=${2:-1}; OUT=$3
EXTRA=("${@:4}")
export TMPDIR=/workspace/tmp
OBJECTS=$(python3 -c "import json;print(' '.join(str(o['id']) for o in json.load(open('outputs/graspnet_bottle_20261008/objects.json'))['objects']))")
mkdir -p "$OUT"
for k in $(seq 0 $((PROCS - 1))); do
    ids=$(echo "$OBJECTS" | tr ' ' '\n' | awk -v k="$k" -v n="$PROCS" 'NR % n == (k + 1) % n' | paste -sd,)
    echo "process $k: objects $ids $(date +%T)"
    CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=osmesa PYTHONPATH=src OMP_NUM_THREADS=2 \
        third_party/Isaac-GR00T/.venv/bin/python scripts/experiments/wbc_compare_20261012/bfm_grasp.py \
        --objects "$ids" --out "$OUT" "${EXTRA[@]}" > "$OUT/run_$k.log" 2>&1 &
    sleep 2
done
wait
echo "batch finished $(date +%T)"
