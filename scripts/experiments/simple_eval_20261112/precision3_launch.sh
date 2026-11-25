#!/usr/bin/env bash
# notebook: selfmodel_precision3_20261125 -- the arms kin_rr, kinc_rr (precision_run.sh NEW_ARMS=3) on the repeats
# selfmodel_precision_20261124 and selfmodel_precision2_20261125 ran, on the 4090: precision3_launch.sh <dev|test|trial> [K]
#   dev    outputs/simple_dev_20261124/precision_20261124/<ctrl>/r0..r<K-1>, c_rr without the episode's own position (by
#          name, CONST_LOO)
#   test   outputs/simple_eval_20261112/precision_20261124/<ctrl>/r0..r<K-1>, c_rr from all development episodes
#   trial  outputs/simple_dev_20261124/precision2_trial/<ctrl>/r0 (precision2's trial copy of two dev positions)
# ScaleBFM on GPU_B (default 1), SONIC on GPUS_S (default "2 3"); PAR_B / PAR_S repeats at a time (default 4 / 2), two arms
# each. c_rr: outputs/simple_dev_20261124/precision_const/ctrack_rr_<ctrl>.json (precision.py ctrack_rr, made before any
# run). Check nvidia-smi first. Markers <tree>/started3, done3; progress in <ctrl>/run3.log, tables <ctrl>/table3.txt.
# Nothing of the earlier runs is rewritten (precision_run.sh refuses a repeat that has shifts3.jsonl or the arms' dirs).
set -uo pipefail
KIND=$1 K=${2:-4}
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco; cd $W
E=scripts/experiments/simple_eval_20261112
DEV=$W/outputs/simple_dev_20261124
export BENCH_REPO=$W SIMPLE_ROOT=/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE BENCH_TASKS=TabletopGraspMP \
  BENCH_SET=sensor=rgbd REACH_SONIC_NETS=gpu NEW_ARMS=3
case $KIND in
  dev) export SIMPLE_EVAL_OUT=$DEV SIMPLE_LEVELS=0,1,2,3,4 CONST_LOO=1; E0=30 E1=49; T=$DEV/precision_20261124 ;;
  test) export SIMPLE_EVAL_OUT=$W/outputs/simple_eval_20261112; unset SIMPLE_LEVELS CONST_LOO; E0=0 E1=29
        T=$W/outputs/simple_eval_20261112/precision_20261124 ;;
  trial) export SIMPLE_EVAL_OUT=$DEV SIMPLE_LEVELS=0,1,2,3,4 CONST_LOO=1; E0=30 E1=49; T=$DEV/precision2_trial; K=1 ;;
  *) echo "dev, test or trial"; exit 2 ;;
esac
CONSTS=$DEV/precision_const
REPS=$(seq 0 $((K - 1)) | tr "\n" " ")
[ -e $T/started3 ] && { echo "$T/started3 exists: this run was made"; exit 2; }
for c in scalebfm sonic; do [ -s $CONSTS/ctrack_rr_$c.json ] || { echo "no ctrack_rr_$c.json"; exit 2; }; done
date -u +%FT%TZ > $T/started3
PAR=${PAR_B:-4} ARM_PAR=2 bash $E/precision_run.sh $T/scalebfm scalebfm $CONSTS/ctrack_rr_scalebfm.json "$REPS" $E0 $E1 "${GPU_B:-1}" &
PAR=${PAR_S:-2} ARM_PAR=2 bash $E/precision_run.sh $T/sonic sonic $CONSTS/ctrack_rr_sonic.json "$REPS" $E0 $E1 "${GPUS_S:-2 3}" &
wait
date -u +%FT%TZ > $T/done3
