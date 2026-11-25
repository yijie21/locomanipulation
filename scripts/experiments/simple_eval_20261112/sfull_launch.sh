#!/usr/bin/env bash
# notebook: selfmodel_sfull_20261125 -- S-full (SF), const, kinc, kin_rr and kinc_rr (sfull_run.sh) on the 4090:
#   sfull_launch.sh <test|trial>
#   test   SIMPLE's official episodes, outputs/simple_eval_20261112/bench_rgbd_20261124/<ctrl> (J, JR, the old S) ->
#          outputs/simple_eval_20261112/sfull_20261125/<ctrl>; const and c_t from all development episodes
#   trial  two routed development positions (30, 48) of outputs/simple_dev_20261124/bench/<ctrl> (JR) ->
#          outputs/simple_dev_20261124/sfull_trial_rr/<ctrl>; const and c_t left out per layout, c_rr per position (CONST_LOO)
# ScaleBFM on GPU_B (default 1, ARM_PAR_B arms at a time, default 5), SONIC on GPUS_S (default "2 3", ARM_PAR_S, default 3).
# The frozen files: outputs/simple_dev_20261124/precision_const/const_<ctrl>.json, ctrack_<ctrl>.json and (the revision of
# 11-25) ctrack_rr_<ctrl>.json. Check nvidia-smi first. Markers <tree>/started, done.
set -uo pipefail
KIND=$1
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco; cd $W
E=scripts/experiments/simple_eval_20261112
DEV=$W/outputs/simple_dev_20261124 K=$W/outputs/simple_dev_20261124/precision_const
export BENCH_REPO=$W SIMPLE_ROOT=/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE BENCH_SET=sensor=rgbd REACH_SONIC_NETS=gpu
case $KIND in
  test) export SIMPLE_EVAL_OUT=$W/outputs/simple_eval_20261112; unset SIMPLE_LEVELS CONST_LOO ONLY; export J_ARM=J
        SRC=$W/outputs/simple_eval_20261112/bench_rgbd_20261124; T=$W/outputs/simple_eval_20261112/sfull_20261125 ;;
  trial) export SIMPLE_EVAL_OUT=$DEV SIMPLE_LEVELS=0,1,2,3,4 CONST_LOO=1 ONLY=30,48 J_ARM=JR
         SRC=$DEV/bench; T=$DEV/sfull_trial_rr ;;
  *) echo "test or trial"; exit 2 ;;
esac
[ -e $T ] && { echo "$T exists: this run was made"; exit 2; }
mkdir -p $T; date -u +%FT%TZ > $T/started
go() {  # go <ctrl> <GPUs> <arms at a time>
  local c=$1
  [ "$KIND" = test ] && export S_CHOICE=$SRC/$c/S/choice.jsonl
  ARM_PAR=$3 bash $E/sfull_run.sh $T/$c $c $SRC/$c $K/const_$c.json $K/ctrack_$c.json $K/ctrack_rr_$c.json "$2"
}
go scalebfm "${GPU_B:-1}" "${ARM_PAR_B:-5}" &
go sonic "${GPUS_S:-2 3}" "${ARM_PAR_S:-3}" &
wait
date -u +%FT%TZ > $T/done
