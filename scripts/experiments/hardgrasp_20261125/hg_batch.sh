#!/usr/bin/env bash
# selfmodel_hardgrasp_20261125 LocoManiLab batch on 4090b: hg_batch.sh <look|run> <eps a:b> <shards per task> <task...>
# runs from $CODE (the asrun copy for the formal run), outputs under $HG_OUT; one process per (task, shard), all in background
PHASE=$1; EPS=$2; NS=$3; shift 3
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
CODE=${CODE:-$W/scripts/experiments/hardgrasp_20261125}
export HG_OUT=${HG_OUT:-$W/outputs/hardgrasp_20261125/locomanilab}
PY=/wuji-vefps-D/wuji-il/yijie/envs/locomanilab_py311/bin/python
G=/wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend
cd $G
export PYTHONPATH=$G:$W/src SIM2REAL_ORT_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [ "$PHASE" = look ]; then
  export LD_LIBRARY_PATH=$HG_OUT/look_dev/egllib:$LD_LIBRARY_PATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
fi
mkdir -p $HG_OUT/logs
for T in "$@"; do
  for ((k=0; k<NS; k++)); do
    nohup $PY $CODE/hg_run.py $PHASE --task $T --eps=$EPS --shard $k --shards $NS > $HG_OUT/logs/${PHASE}_${EPS/:/-}_${T}_$k.log 2>&1 &
  done
done
echo "launched $PHASE $EPS x$NS: $#  tasks"
