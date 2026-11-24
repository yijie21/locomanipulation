#!/usr/bin/env bash
# s_run.sh <bench dir> <scalebfm|sonic> <model dir> <S dir> <rounds> [s_check.py check options]: the S arm's check and
# repairs (selfmodel_s_20261124, TODO 38.10.4) on the J arm's plans of a bench dir that has run bench_via.sh's steps A,
# plan and JR (the state at the command). Round 0 checks J's own plan; a failed check writes the next round's planner
# inputs (goal moved, voxels moved), plan2.py plans them, the next round checks the new plan; s_check.py choose picks the
# first passed candidate per episode -> <S dir>/via, for bench_via.sh step S (S_DIR=<S dir>).
# Env as bench_via.sh: BENCH_REPO, SIMPLE_ROOT, BENCH_GPU (the planner's card), SIMPLE_EVAL_OUT, SIMPLE_LEVELS, BENCH_SET.
set -uo pipefail
B=$1 CTRL=$2 MODEL=$3 SD=$4 ROUNDS=$5; shift 5
W=${BENCH_REPO:-/workspace/code/locomanipulation-selfwm}; cd $W
S=${SIMPLE_ROOT:-$W/third_party/SIMPLE}; SP=$S/.venv/bin/python; PY=$W/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
E=scripts/experiments/simple_eval_20261112; C=scripts/experiments/curobo_avoid_20261120
O=${SIMPLE_EVAL_OUT:-$W/outputs/simple_eval_20261112}
export PYTHONPATH=src REACH_HAND_OFF=0.0415,0,0 OMP_NUM_THREADS=1 MUJOCO_GL=egl SIMPLE_ROOT=$S
export CUDA_VISIBLE_DEVICES=${BENCH_GPU:-1} MUJOCO_EGL_DEVICE_ID=${BENCH_GPU:-1}
SETS="--bench-set controller=$CTRL"; for kv in ${BENCH_SET:-}; do SETS="$SETS --bench-set $kv"; done
BED=$($PY -m locomani.bench.simple $SETS) || exit 2; eval "$BED"; CL=$O/$BED_CLOUD
mkdir -p $SD/asrun; cp $E/s_check.py $E/s_run.sh $C/plan2.py $SD/asrun/
log() { echo "[$(date -u +%T)] $*" >> $SD/s_pipeline.log; }
log "start: $B $CTRL model $MODEL rounds $ROUNDS check options: $*"
$PY -I $E/s_check.py state $B $SD > $SD/logs_state.txt 2>&1 || { log "FAIL state"; exit 2; }
log "state: $(wc -l < $SD/state.jsonl) episodes with J's plan"
for r in $(seq 0 $ROUNDS); do
  $PY -I $E/s_check.py check $SD $MODEL $B $r "$@" > $SD/logs_check$r.txt 2>&1 || { log "FAIL check $r"; exit 2; }
  log "round $r: $(grep -c '"plan": true' $SD/check_round$r.jsonl) checked, $(grep -c '"passed": true' $SD/check_round$r.jsonl) passed"
  [ "$r" = "$ROUNDS" ] && break
  n=$((r + 1))
  python3 -c "
import json
for l in open('$SD/check_round$r.jsonl'):
    x = json.loads(l)
    if x.get('plan') and not x['passed']:
        print(x['task'], x['ep'], x['act'])
" | xargs -P 4 -L1 sh -c "cd $S && $SP -I $W/$C/plan2.py $S $SD/round$n/step1 $SD/round$n/step2 \$1 --task \$0 --cloud-dir $CL --act \$2 2>/dev/null | grep '^{'" > $SD/plan_round$n.jsonl
  log "replan $n: $(wc -l < $SD/plan_round$n.jsonl) rows, planned $(grep -c '"ok": true' $SD/plan_round$n.jsonl)"
done
$PY -I $E/s_check.py choose $SD $B $ROUNDS > $SD/logs_choose.txt 2>&1 || { log "FAIL choose"; exit 2; }
log "chosen: $(ls $SD/via/via_*.npz 2>/dev/null | wc -l) via files, given up $(grep -c '"given_up": true' $SD/choice.jsonl)"
