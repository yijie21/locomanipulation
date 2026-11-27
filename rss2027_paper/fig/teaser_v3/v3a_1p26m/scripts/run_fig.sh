#!/usr/bin/env bash
# notebook: skip figure render, not an experiment
# run_fig.sh <A|R|servo> <episodes a-b> <gpu> <out dir>: Figure 1 v3's episodes on the shelf scene (fig_run.py) from the
# private copy of selfmodel_simplecmp_20261126's frozen code (fig1_teaser_v3_20261127/code, CODE_MD5 checked), with
# run_arm.sh's ScaleBFM R_D environment (REACH_HAND_OFF palm, REACHWM_GOAL_REF/APPROACH/QFIX/FDVEL = 1, BFM_ROOT odo via
# the switches), the policy on GPU <gpu>, one CPU thread, nice 10, taskset $TASKSET.
set -uo pipefail
ARM=$1 EPS=$2 G=$3 OUT=$4
Y=/wuji-vefps-D/wuji-il/yijie F=$Y/loco_outputs/fig1_teaser_v3_20261127 C=$F/code
mkdir -p $OUT/logs
LOG=$OUT/logs/${ARM}_${EPS}.log
cd $C || exit 2
md5sum -c --quiet CODE_MD5 > /dev/null 2>&1 || { echo "FAIL: $C differs from its CODE_MD5" >> $LOG; echo 2 > $OUT/logs/${ARM}_${EPS}.exit; exit 2; }
TREE=$C/outputs/simple_eval_20261112
[ ! -L $TREE ] && [ ! -L $TREE/scene ] || { echo "FAIL: $TREE or its scene is a link" >> $LOG; exit 2; }
PY=$C/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$G CUDA_VISIBLE_DEVICES=$G SIMPLE_ROOT=$Y/tools/simple_isaac/SIMPLE
export SIMPLE_EVAL_OUT=$TREE SIMPLE_LEVELS=0,1,2 LOCOMANI_WORKSPACE=$Y/ws
export REACH_HAND_OFF=0.0415,0,0 REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1
echo "$(date -u +%FT%TZ) start $ARM $EPS GPU $G" >> $LOG
taskset -c ${TASKSET:-0,2,5,6,8,10,12,15,16,19,21,22} nice -n 10 $PY $F/scripts/fig_run.py $C $ARM $EPS $OUT >> $LOG 2>&1
rc=$?
echo "$(date -u +%FT%TZ) end exit $rc" >> $LOG
echo $rc > $OUT/logs/${ARM}_${EPS}.exit
exit $rc
