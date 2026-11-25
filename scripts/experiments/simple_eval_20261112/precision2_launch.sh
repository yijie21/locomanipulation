#!/usr/bin/env bash
# notebook: selfmodel_precision2_20261125 -- the arms kin, kinc, modelf (precision_run.sh NEW_ARMS=1) on the repeats
# selfmodel_precision_20261124 ran, on the 4090: precision2_launch.sh <dev|test|trial> [K]
#   dev    outputs/simple_dev_20261124/precision_20261124/<ctrl>/r0..r<K-1> (the second development run), c_t left out per
#          layout (CONST_LOO)
#   test   outputs/simple_eval_20261112/precision_20261124/<ctrl>/r0..r<K-1>, c_t from all development episodes
#   trial  outputs/simple_dev_20261124/precision2_trial/<ctrl>/r0 (a copy of two dev positions of r0, made beforehand)
# ScaleBFM on GPU_B (default 1), SONIC on GPUS_S (default "2 3"); PAR_B / PAR_S repeats at a time (default 3 / 2), ARM_PAR_S
# SONIC arms at a time (default 3). The tracking constant: outputs/simple_dev_20261124/precision_const/ctrack_<ctrl>.json
# (precision.py ctrack, made before any run). Check nvidia-smi first. Markers <tree>/started2, done2; progress in
# <ctrl>/run2.log, tables <ctrl>/table2.txt. Nothing of the earlier run is rewritten.
set -uo pipefail
KIND=$1 K=${2:-4}
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco; cd $W
E=scripts/experiments/simple_eval_20261112
DEV=$W/outputs/simple_dev_20261124
export BENCH_REPO=$W SIMPLE_ROOT=/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE BENCH_TASKS=TabletopGraspMP \
  BENCH_SET=sensor=rgbd REACH_SONIC_NETS=gpu NEW_ARMS=1
case $KIND in
  dev) export SIMPLE_EVAL_OUT=$DEV SIMPLE_LEVELS=0,1,2,3,4 CONST_LOO=1; E0=30 E1=49; T=$DEV/precision_20261124 ;;
  test) export SIMPLE_EVAL_OUT=$W/outputs/simple_eval_20261112; unset SIMPLE_LEVELS CONST_LOO; E0=0 E1=29
        T=$W/outputs/simple_eval_20261112/precision_20261124 ;;
  trial) export SIMPLE_EVAL_OUT=$DEV SIMPLE_LEVELS=0,1,2,3,4 CONST_LOO=1; E0=30 E1=49; T=$DEV/precision2_trial; K=1 ;;
  *) echo "dev, test or trial"; exit 2 ;;
esac
CONSTS=$DEV/precision_const
REPS=$(seq 0 $((K - 1)) | tr "\n" " ")
[ -e $T/done2 ] && { echo "$T/done2 exists: this run was made"; exit 2; }
date -u +%FT%TZ > $T/started2
PAR=${PAR_B:-3} ARM_PAR=3 bash $E/precision_run.sh $T/scalebfm scalebfm $CONSTS/ctrack_scalebfm.json "$REPS" $E0 $E1 "${GPU_B:-1}" &
PAR=${PAR_S:-2} ARM_PAR=${ARM_PAR_S:-3} bash $E/precision_run.sh $T/sonic sonic $CONSTS/ctrack_sonic.json "$REPS" $E0 $E1 "${GPUS_S:-2 3}" &
wait
date -u +%FT%TZ > $T/done2
