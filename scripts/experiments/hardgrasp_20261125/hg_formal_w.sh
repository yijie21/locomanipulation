#!/usr/bin/env bash
# selfmodel_hardgrasp_20261125 LocoManiLab formal run, launch 2 (4090b, tmux hg_lml). Same steps as hg_formal.sh.
# Launch 1 (06:56Z) ran the asrun copy: its looks worked, but every run worker died at start, because selfwm env.py
# takes REPO from the file's own location (parents[2]), so under asrun it pointed at outputs/hardgrasp_20261125 and
# reachwm data.py could not find assets/scalebfm. This launch runs the same files from $W/scripts/experiments (the
# layout the design ran in), checked against asrun/MD5SUMS before the start and again after the runs.
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
O=$W/outputs/hardgrasp_20261125/locomanilab
E=$W/scripts/experiments
export CODE=$E/hardgrasp_20261125 HG_OUT=$O
md5ok() { (cd $E && grep -v '\.\./' $O/asrun/MD5SUMS | md5sum -c --quiet) && (cd $O/asrun && grep '\.\./' MD5SUMS | md5sum -c --quiet); }
if ! md5ok; then echo "launch 2 aborted: code differs from asrun/MD5SUMS $(date -u +%FT%TZ)" | tee -a $O/formal.log; exit 1; fi
TASKS="$*"
NT=$(echo $TASKS | wc -w)
echo $((NT * 31 * 2)) > $O/FORMAL_TOTAL
echo "formal launch 2 start $(date -u +%FT%TZ) tasks $NT, code $E (md5 = asrun)" | tee -a $O/formal.log
bash $CODE/hg_batch.sh look -1:30 3 $TASKS
while pgrep -f "[h]g_run.py look" > /dev/null; do sleep 5; done
echo "looks done $(date -u +%FT%TZ)" | tee -a $O/formal.log
bash $CODE/hg_batch.sh run -1:30 $((90 / NT)) $TASKS
while pgrep -f "[h]g_run.py run" > /dev/null; do sleep 10; done
echo "runs done $(date -u +%FT%TZ)" | tee -a $O/formal.log
if md5ok; then echo "md5 after the runs: same as asrun" | tee -a $O/formal.log; else echo "md5 after the runs: CHANGED" | tee -a $O/formal.log; fi
cd /wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend
PYTHONPATH=.:$W/src HG_OUT=$O /wuji-vefps-D/wuji-il/yijie/envs/locomanilab_py311/bin/python $CODE/hg_run.py stats --eps=-1:30 > $O/stats_formal.txt 2>&1
echo "stats done $(date -u +%FT%TZ)" | tee -a $O/formal.log
