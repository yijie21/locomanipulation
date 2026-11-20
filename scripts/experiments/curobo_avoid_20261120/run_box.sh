#!/usr/bin/env bash
# curobo_avoid_20261120, the three box tasks (4090): step 1 (obstacles_box.py), step 2 (prep2.py, plan2.py at the
# pre-registered 3 cm and a 2 cm retry for the failures, via2.py), step 3 (the bench: direct and model with --avoid, model
# straight; --trace). direct straight with --film --trace is in box/rows already (90 / 90 as rows_eyes).
# Progress: box/logs/pipeline.log; "FAIL <step>" on a failed step, "DONE" at the end.
set -uo pipefail
export C=/wuji-vefps-D/wuji-il/yijie/loco_outputs/simple_bench_20261119/curobo
export R=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
export S=/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE
export PY=$R/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python SP=$S/.venv/bin/python
export CL=$R/outputs/simple_eval_20261112/cloud_eyes_final LOOK=$R/outputs/simple_eval_20261112/look_dump_eyes
export D=$C/box
TASKS="XMovePickTeleop XMoveBendPickTeleop BendPickMP"
mkdir -p $D/logs $D/step1 $D/step2 $D/try_act02 $D/via $D/step3/rows $D/step3/logs
log() { echo "[$(date +%T)] $*" >> $D/logs/pipeline.log; }
log "start, commit $(git -C $R log -1 --format=%h) + uncommitted motor.py via_rot, simple_bench.py --avoid"
cd $R
export SIMPLE_ROOT=$R/third_party/SIMPLE MUJOCO_GL=egl

# step 1
for T in $TASKS; do for e in $(seq 0 29); do echo "$T $e"; done; done |
  xargs -P 15 -L1 sh -c '$PY -I $C/obstacles_box.py $LOOK $D/films $CL $D/step1 $0 $1 2>/dev/null | grep "^{"' > $D/step1_rows.jsonl
log "step 1: $(wc -l < $D/step1_rows.jsonl) rows, $(ls $D/step1/obst_*.npz | wc -l) obstacle files"
[ "$(ls $D/step1/obst_*.npz | wc -l)" -ge 85 ] || log "FAIL step 1"

# step 2a
for T in $TASKS; do
  seq 0 29 | xargs -P 15 -I{} sh -c "TASK=$T TRACE_FMT='direct_straight_{task}_{e}.jsonl' \$PY -I \$C/prep2.py \$D/films \$D/rows \$D/step1 \$D/step2 {} 2>&1 | grep -v Warn"
done > $D/logs/prep.txt
log "step 2a: $(ls $D/step2/prep_*.npz | wc -l) prep files"

# step 2b: the pre-registered setting, then 2 cm for the failures
for T in $TASKS; do
  seq 0 29 | xargs -P 8 -I@ sh -c "CUDA_VISIBLE_DEVICES=\$((@ % 4)) \$SP -I \$C/plan2.py \$S \$D/step1 \$D/step2 @ --task $T --cloud-dir \$CL --diag 2>/dev/null | grep '^{'"
done > $D/plan_rows.jsonl
log "step 2b: $(wc -l < $D/plan_rows.jsonl) rows, ok $(grep -c '"ok": true' $D/plan_rows.jsonl)"
ln -sf $D/step2/prep_*.npz $D/try_act02/
python3 -c "
import json
for l in open('$D/plan_rows.jsonl'):
    r = json.loads(l)
    if 'skip' not in r and not r['ok']:
        print(r['name'].rsplit('_L', 1)[0], r['ep'])
" | xargs -P 8 -L1 sh -c 'CUDA_VISIBLE_DEVICES=$(($1 % 4)) $SP -I $C/plan2.py $S $D/step1 $D/try_act02 $1 --task $0 --cloud-dir $CL --act 0.02 2>/dev/null | grep "^{"' > $D/plan_rows_act02.jsonl
log "step 2b retry at 2 cm: $(wc -l < $D/plan_rows_act02.jsonl) rows, ok $(grep -c '"ok": true' $D/plan_rows_act02.jsonl)"
$PY -I $C/via2.py $D/via 0.01 $D/step2 $D/try_act02 > $D/logs/via.txt 2>&1 || log "FAIL via2"
log "step 2c: $(ls $D/via/via_*.npz | wc -l) via files"

# step 3
export REACH_HAND_OFF=0.0415,0,0 BFM_ROOT=odo REACHWM_GOAL_REF=1 PYTHONPATH=src OMP_NUM_THREADS=1 TMPDIR=/workspace/tmp
E=$R/scripts/experiments/simple_eval_20261112
WM="--wm outputs/reachwm_20261030/invert/wrist_wrist_odo/wm_ref.pt"
ARGS="--column perception --cloud cloud_eyes_final --trace"
for T in $TASKS; do
  pids=()
  for e in $(seq 0 29); do
    REACH_BFM_DEVICE=cpu CUDA_VISIBLE_DEVICES= $PY $E/simple_bench.py $T --method direct --episodes $e $ARGS --avoid $D/via \
      --out $D/step3/rows/direct_avoid_${T}_$e.jsonl > $D/step3/logs/direct_avoid_${T}_$e.log 2>&1 & pids+=($!)
    CUDA_VISIBLE_DEVICES=$((e % 4)) $PY $E/simple_bench.py $T --method model --episodes $e $ARGS $WM --avoid $D/via \
      --out $D/step3/rows/model_avoid_${T}_$e.jsonl > $D/step3/logs/model_avoid_${T}_$e.log 2>&1 & pids+=($!)
    CUDA_VISIBLE_DEVICES=$((e % 4)) $PY $E/simple_bench.py $T --method model --episodes $e $ARGS $WM \
      --out $D/step3/rows/model_straight_${T}_$e.jsonl > $D/step3/logs/model_straight_${T}_$e.log 2>&1 & pids+=($!)
  done
  fails=0
  for p in "${pids[@]}"; do wait $p || fails=$((fails + 1)); done
  [ $fails = 0 ] || log "FAIL step 3 $T: $fails processes"
  log "step 3 $T: $(ls $D/step3/rows/*_${T}_*.jsonl | wc -l) rows"
done
for T in $TASKS; do
  ln -sf $D/rows/direct_straight_${T}_*.jsonl $D/step3/rows/
done
log "DONE"
