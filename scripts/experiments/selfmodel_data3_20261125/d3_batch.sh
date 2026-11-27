#!/usr/bin/env bash
# selfmodel_data3_20261125: d3_batch.sh <backend> <out dir> <resets> <shards> <first shard> <last shard> [drive args...]
# One single-threaded d3_collect.py drive per shard, in the background (nohup); logs in <out>/logs/drive_<shard>.log,
# pids appended to <out>/pids_<hostname>.txt. Code from $CODE (default: the repo copy; the formal run sets the asrun copy).
# Waits for its own shards (by pid) unless D3_NOWAIT=1, then prints a line.
BE=$1; OUT=$2; RESETS=$3; NS=$4; A=$5; B=$6; shift 6
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
CODE=${CODE:-$W/scripts/experiments/selfmodel_data3_20261125}
PY=/wuji-vefps-D/wuji-il/yijie/envs/locomanilab_py311/bin/python
G=/wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend
cd $G || exit 1
export PYTHONPATH=$G:$W/src SIM2REAL_ORT_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p $OUT/logs
PIDS=()
for ((k=A; k<=B; k++)); do
  # SONIC on the GPU (D3_SONIC_INFER=onnx-gpu): shards dealt round-robin to D3_NGPU cards
  [ "$BE" = sonic ] && [ "${D3_SONIC_INFER:-}" = onnx-gpu ] && export CUDA_VISIBLE_DEVICES=$((k % ${D3_NGPU:-2}))
  nohup $PY $CODE/d3_collect.py drive --backend $BE --out $OUT --resets $RESETS --shards $NS --shard $k "$@" \
    > $OUT/logs/drive_$k.log 2>&1 &
  PIDS+=($!)
  echo "$! $BE $k" >> $OUT/pids_$(hostname).txt
done
echo "$(date -u +%FT%TZ) launched $BE shards $A-$B of $NS ($RESETS resets) on $(hostname): ${#PIDS[@]} processes"
[ "${D3_NOWAIT:-0}" = 1 ] && exit 0
for p in "${PIDS[@]}"; do wait $p; done
echo "$(date -u +%FT%TZ) $BE shards $A-$B finished"
