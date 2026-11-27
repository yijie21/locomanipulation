#!/usr/bin/env bash
# notebook: selfmodel_hardgrasp_20261125
# BiGym part, formal run (4090, tmux hg_bigym): the looks, GraspGen-X, both groups, the analysis. Runs the frozen copy in
# $O/asrun (the repo's layout: $A/scripts/experiments/{hardgrasp_20261125,bigym_20261124,selfmodel_via_20261120,
# reachwm_20261030}, M2's numpy export in $A/m2). dishwasher_unload_cups, seeds 620000-620029 (30), groups D and S paired by seed.
# GPU 0 is not ours: GraspGen-X on GPU 3, the planners hg_plan1 / hg_plan2 (GPUs 1 / 2) already running.
set -u
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
O=$W/outputs/hardgrasp_20261125/bigym
F=$O/formal
A=$O/asrun
H=$A/scripts/experiments/hardgrasp_20261125
M2A="--m2-weights $A/m2/m2_weights.npz --m2-band $A/m2/band.npz"
G=/wuji-vefps-D/wuji-il/yijie/tools/perception/GraspGenX
TASK=dishwasher_unload_cups
source /wuji-vefps-D/wuji-il/yijie/tools/bigym2/env.sh
export OMP_NUM_THREADS=1
cd $W
mkdir -p $F/look $F/rows $F/logs
date -u +%FT%TZ > $F/started
(cd $A && md5sum -c md5.txt) > $F/asrun_md5_check.txt 2>&1 || { echo "FAIL asrun md5" > $F/failed; exit 1; }
# 1 the looks: 6 processes of 5 seeds
for k in 0 1 2 3 4 5; do
  a=$((620000 + 5 * k))
  python $H/bigym_hg.py look --task $TASK --seeds $a:$((a + 5)) --dir $F/look --allow-eval-seeds > $F/logs/look_$k.log 2>&1 &
done
wait
# 2 GraspGen-X on every cloud (one pass)
CUDA_VISIBLE_DEVICES=3 $G/.venv/bin/python $H/graspgen_dex1.py $F/look > $F/logs/graspgen.log 2>&1
date -u +%FT%TZ > $F/perceived
# 3 both groups, one process per seed, the two planners alternating
for s in $(seq 620000 620029); do
  sock=$O/plan$((1 + s % 2)).sock
  timeout 900 python $H/bigym_hg.py run --task $TASK --seeds $s --groups D,S --dir $F/look --out $F/rows/$s.jsonl \
    --plan-socket $sock $M2A --allow-eval-seeds > $F/logs/run_$s.log 2>&1 &
done
wait
# 4 the analysis
python $H/bigym_hg.py analyze "$F/rows/*.jsonl" > $F/analysis.txt 2>&1
date -u +%FT%TZ > $F/done
