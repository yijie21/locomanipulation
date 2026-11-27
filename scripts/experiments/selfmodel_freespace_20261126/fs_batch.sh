#!/usr/bin/env bash
# notebook: selfmodel_freespace_20261126
# fs_batch.sh <backend> <eps a:b> <cells a:b | c1,c2,...> <processes>: fs_run.py run in <processes> shards in the background
# (nohup, nice $FS_NICE (default 19), single-threaded, no GPU), the controller's groups (fs_run.GROUPS), logs in $FS_OUT/<backend>/logs/, pids
# appended to $FS_OUT/<backend>/pids.txt (the python processes); waits for its own processes (by pid). Code: $LC_CODE_ROOT when set (the round's
# private copy: scripts and PYTHONPATH from there, lc3_run's code check on every process), else the repo tree.
BE=$1; EPS=$2; CELLS=$3; NP=$4
W=${LC_CODE_ROOT:-/wuji-vefps-D/wuji-il/yijie/ws/code/loco}
CODE=$W/scripts/experiments/selfmodel_freespace_20261126
export FS_OUT=${FS_OUT:?FS_OUT required}
PY=/wuji-vefps-D/wuji-il/yijie/envs/locomanilab_py311/bin/python
G=/wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend
cd $G || exit 1
export PYTHONPATH=$G:$W/src SIM2REAL_ORT_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=
L=$FS_OUT/$BE/logs
mkdir -p $L
PIDS=()
for ((k=0; k<NP; k++)); do
  # the policy's per-step timing warnings (SONIC: one every few steps) are dropped from the log; $! stays the python pid
  nohup nice -n ${FS_NICE:-19} $PY $CODE/fs_run.py run --backend $BE --eps $EPS --cells $CELLS --shard $k --shards $NP \
    > >(grep -a --line-buffered -v "RL step took\|Control mode set to policy\|motion layout updated" > $L/run_n${NP}_$k.log) 2>&1 &
  PIDS+=($!)
  echo "$! $BE $k/$NP $(date -u +%FT%TZ)" >> $FS_OUT/$BE/pids.txt
done
echo "$(date -u +%FT%TZ) launched $BE eps $EPS cells $CELLS: ${#PIDS[@]} processes"
for p in "${PIDS[@]}"; do wait $p; done
echo "$(date -u +%FT%TZ) $BE runs finished"
