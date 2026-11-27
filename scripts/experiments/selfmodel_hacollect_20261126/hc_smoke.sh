#!/usr/bin/env bash
# notebook: skip HA recollection dev
# Development smoke of selfmodel_hacollect_20261126 (not a formal run): one controller, P drive processes on one card,
# then sx_collect's merge / check and hc_start's comparison with LocoManiLab's data3 (and data3b for --b).
#   hc_smoke.sh <ctrl> <gpu> <d3|d3b> <resets> <procs> <out>
set -uo pipefail
CTRL=$1; GPU=$2; MODE=$3; N=$4; P=$5; X=$6
C=$(cd "$(dirname "$0")" && pwd)
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs
VPY=$W/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
mkdir -p $X/logs
if [ "$MODE" = d3b ]; then
  EX="--b --b-config $O/selfmodel_data3b_20261125/$CTRL/b_config.json --hc-seed 89500000 --hc-id0 90500000"; ID0=90500000
else
  EX="--hc-seed 89000000 --hc-id0 90000000"; ID0=90000000
fi
[ "$CTRL" = sonic ] && EX="$EX --no-camera"
EX="$EX ${HC_EXTRA:-}"  # dev switches (e.g. --settle-max-s 6 --vtrace)
echo "$(date -u +%FT%TZ) start $CTRL $MODE resets $N procs $P gpu $GPU" >> $X/run.log
pids=()
for k in $(seq 0 $((P - 1))); do
  $C/hc_ha.sh $GPU --phase drive --controller $CTRL --out $X --resets $N --shards $P --shard $k $EX > $X/logs/w$k.log 2>&1 &
  pids+=($!)
  [ $k -eq 0 ] && sleep 20  # shard 0 writes run.json first
done
echo "pids ${pids[*]}" >> $X/run.log
rc=0
for p in "${pids[@]}"; do wait $p || rc=1; done
echo "$(date -u +%FT%TZ) drive done rc $rc" >> $X/run.log
cd $W
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
$VPY scripts/experiments/selfmodel_xfer_20261125/sx_collect.py merge --out $X 2>&1 | grep -v Warn > $X/merge.log
$VPY scripts/experiments/selfmodel_xfer_20261125/sx_collect.py check --out $X --resets $N --id0 $ID0 2>&1 | grep -v Warn > $X/check.log
LB=""; [ "$MODE" = d3b ] && LB="--lml-b $O/selfmodel_data3b_20261125/$CTRL"
SX_REPO=$W REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1 nice -n 10 $VPY $C/hc_start.py --ha $X \
  --lml $O/selfmodel_data3_20261125/$CTRL $LB 2>&1 | grep -v Warn > $X/start.txt
echo "$(date -u +%FT%TZ) post done" >> $X/run.log
touch $X/DONE
