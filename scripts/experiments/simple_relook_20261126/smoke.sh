#!/usr/bin/env bash
# notebook: skip relook dev -- simple_relook_20261126 smoke: the whole pipeline with the second look on development
# episodes (the dev tree, TabletopGrasp): A --relook, the plan from A's films, JR --relook, sfull_run.sh SF and kin_rr
# --relook, then relook_eval.py against the dev tree's bench (its A and JR rows; no S there, so S = J's rows).
#   smoke.sh <tag> <scalebfm|sonic> <GPU> <first ep> <last ep> "<relook options>"   -> $X/smoke/<tag>/
set -uo pipefail
X=/wuji-vefps-D/wuji-il/yijie/loco_outputs/simple_relook_20261126
TAG=$1 C=$2 G=$3 E0=$4 E1=$5 RL=$6
D=$X/smoke/$TAG; mkdir -p $D
R=${SMOKE_REPO:-$X/repo}  # SMOKE_REPO: another copy (asrun/repo_b on 4090b, with LOCOMANI_WORKSPACE and SONIC_POLICY set)
export BENCH_REPO=$R SIMPLE_ROOT=/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE BENCH_GPU=$G BENCH_SET=sensor=rgbd
export SIMPLE_EVAL_OUT=$R/outputs/simple_dev_20261124 SIMPLE_LEVELS=0,1,2,3,4 BENCH_TASKS=TabletopGraspMP
export REACH_SONIC_NETS=${SONIC_NETS:-gpu}
E=$R/scripts/experiments/simple_eval_20261112; PC=$SIMPLE_EVAL_OUT/precision_const
log() { echo "[$(date -u +%FT%TZ)] smoke $TAG: $*" >> $X/logs/dev_phase.log; }
log "start $C GPU $G eps $E0-$E1 relook '$RL' nets $REACH_SONIC_NETS md5 $(md5sum $E/simple_bench.py $E/relook.py | cut -c1-8 | tr '\n' ' ')"
BENCH_EXTRA="$RL" bash $E/bench_via.sh $D/bench $C none A $E0 $E1 || { log "FAIL A"; exit 2; }
bash $E/bench_via.sh $D/bench $C none plan $E0 $E1 || { log "FAIL plan"; exit 2; }
BENCH_EXTRA="$RL" bash $E/bench_via.sh $D/bench $C none JR $E0 $E1 || { log "FAIL JR"; exit 2; }
log "bench: $(tail -4 $D/bench/pipeline.log | tr '\n' ' ')"
SF_ARMS="SF kin_rr" SF_BENCH_EXTRA="$RL" ARM_PAR=1 PLAN_PAR=1 bash $E/sfull_run.sh $D/sfull $C $D/bench \
  $PC/const_$C.json $PC/ctrack_$C.json $PC/ctrack_rr_$C.json "$G"
log "sfull: $(tail -2 $D/sfull/run.log | tr '\n' ' ')"
cd $R && third_party/GR00T-WholeBodyControl/.venv_sim/bin/python -I scripts/experiments/simple_relook_20261126/relook_eval.py \
  $C $D/bench $D/sfull $SIMPLE_EVAL_OUT/bench/$C $D/none JR > $D/eval.txt 2>&1
log "done; eval $D/eval.txt"
