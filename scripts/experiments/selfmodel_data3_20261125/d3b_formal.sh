#!/usr/bin/env bash
# selfmodel_data3b_20261125 formal supplement of one controller on this machine (tmux d3b_<backend>):
#   d3b_formal.sh <backend> <processes> [drive args...]
# runs the frozen copy in $O/asrun (D3_NB=selfmodel_data3b_20261125 d3_freeze.sh), 15000 resets in --b mode, every
# shard here, then merge, check and summary (--b: reset ids from 30000000).
BE=$1; NS=$2; shift 2
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3b_20261125/$BE
export CODE=$O/asrun/selfmodel_data3_20261125 D3_REACHWM=$W/scripts/experiments/reachwm_20261030
[ "$BE" = sonic ] && export D3_SONIC_INFER=${D3_SONIC_INFER:-onnx-gpu} D3_NGPU=${D3_NGPU:-2}
PY=/wuji-vefps-D/wuji-il/yijie/envs/locomanilab_py311/bin/python
G=/wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend
echo "$(date -u +%FT%TZ) formal data3b $BE start: $NS processes, 15000 resets, code $CODE (md5 $(md5sum $CODE/d3_collect.py | cut -c1-32))" | tee -a $O/formal.log
bash $CODE/d3_batch.sh $BE $O 15000 $NS 0 $((NS - 1)) --b "$@" | tee -a $O/formal.log
cd $G
export PYTHONPATH=$G:$W/src OMP_NUM_THREADS=4
$PY $CODE/d3_collect.py merge --b --out $O > $O/merge.log 2>&1
$PY $CODE/d3_collect.py check --b --out $O > $O/check.log 2>&1
$PY $CODE/d3_collect.py summary --b --out $O > $O/summary.txt 2>&1
echo "$(date -u +%FT%TZ) formal data3b $BE merged: $(tail -1 $O/check.log)" | tee -a $O/formal.log
