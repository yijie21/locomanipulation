#!/usr/bin/env bash
# notebook: selfmodel_lmlcmp2_20261126
# lc2_batch.sh <backend> <eps a:b | e1,e2,...> <shards per task> <task...>: lmlcmp's lc_batch.sh on lc2_run.py -- one run
# per (task, shard) in the background (nohup, nice 19, single-threaded, no GPU: CUDA_VISIBLE_DEVICES empty), the four
# groups in every shard (LC_METHODS overrides), logs in $LC_OUT/<backend>/logs/, pids appended to
# $LC_OUT/<backend>/pids.txt; waits for its own processes (by pid). Code: the repo copy ($W/scripts/experiments).
BE=$1; EPS=$2; NS=$3; shift 3
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
CODE=$W/scripts/experiments/selfmodel_lmlcmp2_20261126
export LC_OUT=${LC_OUT:-/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp2_20261126}
METHODS=${LC_METHODS:-alone,selfmodel,guard,reachable}
PY=/wuji-vefps-D/wuji-il/yijie/envs/locomanilab_py311/bin/python
G=/wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend
cd $G || exit 1
export PYTHONPATH=$G:$W/src SIM2REAL_ORT_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
case $EPS in *:*) EA="--eps=$EPS" ;; *) EA="--ep-list=$EPS" ;; esac
L=$LC_OUT/$BE/logs
mkdir -p $L
PIDS=()
for T in "$@"; do
  for ((k=0; k<NS; k++)); do
    nohup nice -n ${LC_NICE:-19} $PY $CODE/lc2_run.py run --backend $BE --task $T $EA --methods $METHODS --shard $k --shards $NS \
      > $L/run_${T}_$k.log 2>&1 &
    PIDS+=($!)
    echo "$! $BE $T $k" >> $LC_OUT/$BE/pids.txt
  done
done
echo "$(date -u +%FT%TZ) launched $BE $EPS: $# tasks x $NS shards = ${#PIDS[@]} processes"
for p in "${PIDS[@]}"; do wait $p; done
echo "$(date -u +%FT%TZ) $BE runs finished"
