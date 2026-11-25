#!/usr/bin/env bash
# notebook: selfmodel_precision_20261124 -- the registered runs on the 4090: precision_launch.sh <dev|test> <K>
#   dev   TabletopGrasp levels 3-4 (positions 30-49) of outputs/simple_dev_20261124, the constant left out per layout
#   test  TabletopGrasp levels 0-2 (positions 0-29), the official episodes, the constant from all development episodes
# K jitter repeats a position, both controllers side by side: ScaleBFM on GPU_B (default 1, PAR_B repeats at a time),
# SONIC on GPUS_S (default "2 3", PAR_S repeats (default 2), ARM_PAR_S arms at a time (default 3); a SONIC process took
# ~1.8 GB in the trial, the planners ~0.5 GB each, BiGym's planning servers share GPUs 1-3). Check nvidia-smi first.
# Output: <tree>/precision_20261124/<ctrl>/r<k>/, table.txt per controller; progress in <ctrl>/run.log, done marker
# <tree>/precision_20261124/done.
set -uo pipefail
KIND=$1 K=$2
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco; cd $W
E=scripts/experiments/simple_eval_20261112
DEV=$W/outputs/simple_dev_20261124
export BENCH_REPO=$W SIMPLE_ROOT=/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE BENCH_TASKS=TabletopGraspMP \
  BENCH_SET=sensor=rgbd REACH_SONIC_NETS=gpu
case $KIND in
  dev) export SIMPLE_EVAL_OUT=$DEV SIMPLE_LEVELS=0,1,2,3,4 CONST_LOO=1; E0=30 E1=49; T=$DEV/precision_20261124 ;;
  test) export SIMPLE_EVAL_OUT=$W/outputs/simple_eval_20261112; unset SIMPLE_LEVELS CONST_LOO; E0=0 E1=29
        T=$W/outputs/simple_eval_20261112/precision_20261124 ;;
  *) echo "dev or test"; exit 2 ;;
esac
CONSTS=$DEV/precision_const  # const_<ctrl>.json, made once by precision.py const from the development tree's JR rows
REPS=$(seq 0 $((K - 1)) | tr "\n" " ")
mkdir -p $T; rm -f $T/done
date -u +%FT%TZ > $T/started
PAR=${PAR_B:-3} ARM_PAR=5 bash $E/precision_run.sh $T/scalebfm scalebfm $CONSTS/const_scalebfm.json "$REPS" $E0 $E1 "${GPU_B:-1}" &
PAR=${PAR_S:-2} ARM_PAR=${ARM_PAR_S:-3} bash $E/precision_run.sh $T/sonic sonic $CONSTS/const_sonic.json "$REPS" $E0 $E1 "${GPUS_S:-2 3}" &
wait
date -u +%FT%TZ > $T/done
