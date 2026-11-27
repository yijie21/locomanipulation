#!/usr/bin/env bash
# notebook: selfmodel_lmlreal_20261125
# lr_batch.sh <backend> <eps a:b> <shards per task> <task...>: one lr_run.py run per (task, shard) in the background
# (nohup, single-threaded), logs in $LR_OUT/<backend>/logs/, pids appended to $LR_OUT/<backend>/pids.txt; waits for its
# own processes (by pid), then prints a line. Code: the repo copy ($W/scripts/experiments; see lr_freeze.sh).
BE=$1; EPS=$2; NS=$3; shift 3
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
CODE=$W/scripts/experiments/selfmodel_lmlreal_20261125
export LR_OUT=${LR_OUT:-/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlreal_20261125}
PY=/wuji-vefps-D/wuji-il/yijie/envs/locomanilab_py311/bin/python
G=/wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend
cd $G || exit 1
export PYTHONPATH=$G:$W/src SIM2REAL_ORT_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
L=$LR_OUT/$BE/logs
mkdir -p $L
PIDS=()
for T in "$@"; do
  for ((k=0; k<NS; k++)); do
    nohup nice -n ${LR_NICE:-0} $PY $CODE/lr_run.py run --backend $BE --task $T --eps=$EPS --shard $k --shards $NS \
      > $L/run_${T}_$k.log 2>&1 &
    PIDS+=($!)
    echo "$! $BE $T $k" >> $LR_OUT/$BE/pids.txt
  done
done
echo "$(date -u +%FT%TZ) launched $BE $EPS: $# tasks x $NS shards = ${#PIDS[@]} processes"
for p in "${PIDS[@]}"; do wait $p; done
echo "$(date -u +%FT%TZ) $BE runs finished"
