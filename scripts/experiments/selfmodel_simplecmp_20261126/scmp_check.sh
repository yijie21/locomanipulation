#!/usr/bin/env bash
# notebook: skip selfmodel_simplecmp_20261126's bit-identity checks and smoke runs (before its registration; no result is scored)
#   scmp_check.sh <code dir> <check dir> <gpu> <phase: off|dry|smoke|all>
# On the 4090, from the private code copy (scmp_copy.sh). ScaleBFM, perception column (sensor=rgbd), simple_bench's A / J
# options as bench_via.sh runs them (no --film: the disk is full; --film writes videos only). Two processes at a time on
# <gpu> (both EGL variables set), nice 10, one thread each.
#   off    the switch off: simple_bench_before.py (the Mac's simple_bench before --correct) and simple_bench.py, each on
#          the test episodes below (A; J on two TabletopGrasp episodes, one with a planned path) -> rows to compare with
#          each other and with the main metric's rows (bench_rgbd_20261124/scalebfm); the development episodes 30, 36
#          (A) against the development tree's A rows
#   dry    --correct-rule dryR (R's inversion on D planned and recorded, A's / J's command sent) on the same test episodes:
#          the rows must equal A's / J's; D's prediction of the sent command and the landing are recorded
#   smoke  --correct-rule R and S on the development episodes TabletopGraspMP 30-39 (SIMPLE_LEVELS 0-4, the dev tree):
#          the mechanics (candidates, shrink, records, online = offline features); outcomes are not scored
# Test episodes only go through the switch-off and dryR runs, which send A's / J's own commands.
set -uo pipefail
R=$1 OUT=$2 G=$3 PHASE=$4
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126
SD=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpledata_20261126
MAIN=/wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_eval_20261112/bench_rgbd_20261124/scalebfm
DEVB=/wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_dev_20261124/bench/scalebfm
MODEL=$SD/train/models_D_s0/E3_ensemble_f1 TEMPLATE=$SD/real/task/data/all.npz
cd $R
PY=$R/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
E=scripts/experiments/simple_eval_20261112
export REACH_HAND_OFF=0.0415,0,0 REACHWM_GOAL_REF=1 PYTHONPATH=src OMP_NUM_THREADS=1 MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=$G CUDA_VISIBLE_DEVICES=$G SIMPLE_ROOT=/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE
SETS="--bench-set controller=scalebfm --bench-set sensor=rgbd"
mkdir -p $OUT/rows $OUT/logs
log() { echo "[$(date -u +%T)] $*" >> $OUT/check.log; }
DFLAGS="REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1"
# run <arm> <script> <tree: eval|dev> <task> <episodes> [extra...]
run() {
  local arm=$1 sb=$2 tree=$3 T=$4 eps=$5; shift 5
  local ev=$O/evalout lv=0,1,2
  [ $tree = dev ] && ev=$O/devout lv=0,1,2,3,4
  SIMPLE_EVAL_OUT=$ev SIMPLE_LEVELS=$lv nice -n 10 $PY $E/$sb $T --method direct $SETS --trace --episodes $eps "$@" \
    --out $OUT/rows/${arm}_$T.jsonl >> $OUT/logs/${arm}_$T.log 2>&1
  log "$arm $T $eps: exit $?, $(wc -l < $OUT/rows/${arm}_$T.jsonl 2>/dev/null || echo 0) rows"
}
corr() {  # corr <arm> <rule> <arm id>: the --correct options
  echo "--correct $MODEL --correct-rule $2 --correct-template $TEMPLATE --correct-out $OUT/corr_$1 --correct-arm-id $3"
}
tests() {  # tests <arm> <script> [extra...]: the test episodes of the A options
  local arm=$1 sb=$2; shift 2
  run $arm $sb eval XMovePickTeleop 0,5 "$@"; run $arm $sb eval XMoveBendPickTeleop 0,6 "$@"
  run $arm $sb eval BendPickMP 0,5 "$@"; run $arm $sb eval TabletopGraspMP 0,3 "$@"
}
if [ $PHASE = off ] || [ $PHASE = all ]; then
  log "phase off: start"
  ( tests Ab simple_bench_before.py; run Jb simple_bench_before.py eval TabletopGraspMP 0,3 --avoid $MAIN/via --joint ) &
  ( tests Ao simple_bench.py; run Jo simple_bench.py eval TabletopGraspMP 0,3 --avoid $MAIN/via --joint
    run Ado simple_bench.py dev TabletopGraspMP 30,36 ) &
  wait
  log "phase off: done"
fi
if [ $PHASE = dry ] || [ $PHASE = all ]; then
  log "phase dry: start"
  ( export $DFLAGS; tests AL simple_bench.py $(corr AL dryR 1) ) &
  ( export $DFLAGS; run JL simple_bench.py eval TabletopGraspMP 0,3 --avoid $MAIN/via --joint $(corr JL dryR 2) ) &
  wait
  log "phase dry: done"
fi
if [ $PHASE = smoke ] || [ $PHASE = all ]; then
  log "phase smoke: start"
  ( export $DFLAGS; run Rs simple_bench.py dev TabletopGraspMP 30-39 $(corr Rs R 3) ) &
  ( export $DFLAGS; run Ss simple_bench.py dev TabletopGraspMP 30-39 $(corr Ss S 4) ) &
  wait
  log "phase smoke: done"
fi
log "DONE $PHASE"
