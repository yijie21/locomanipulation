#!/usr/bin/env bash
# selfmodel_hardgrasp_20261125 LocoManiLab formal run (4090b, tmux hg_lml): looks, then both groups, then the stats.
# Runs the frozen copy in $O/asrun. Episodes -1 (the official unjittered start) and 0..29 (our jitter).
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
O=$W/outputs/hardgrasp_20261125/locomanilab
export CODE=$O/asrun/hardgrasp_20261125 HG_OUT=$O
TASKS="$*"
NT=$(echo $TASKS | wc -w)
echo $((NT * 31 * 2)) > $O/FORMAL_TOTAL
echo "formal start $(date -u +%FT%TZ) tasks $NT" | tee -a $O/formal.log
bash $CODE/hg_batch.sh look -1:30 3 $TASKS
while pgrep -f "[h]g_run.py look" > /dev/null; do sleep 5; done
echo "looks done $(date -u +%FT%TZ)" | tee -a $O/formal.log
bash $CODE/hg_batch.sh run -1:30 $((90 / NT)) $TASKS
while pgrep -f "[h]g_run.py run" > /dev/null; do sleep 10; done
echo "runs done $(date -u +%FT%TZ)" | tee -a $O/formal.log
cd /wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend
PYTHONPATH=.:$W/src HG_OUT=$O /wuji-vefps-D/wuji-il/yijie/envs/locomanilab_py311/bin/python $CODE/hg_run.py stats --eps=-1:30 > $O/stats_formal.txt 2>&1
echo "stats done $(date -u +%FT%TZ)" | tee -a $O/formal.log
