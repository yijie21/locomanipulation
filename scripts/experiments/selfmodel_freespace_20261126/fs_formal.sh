#!/usr/bin/env bash
# notebook: selfmodel_freespace_20261126
# E2a's formal run on 4090 (tmux fs_formal): fs_formal.sh <processes> <backend>:<episode ids per cell> ...
#   e.g. fs_formal.sh 8 scalebfm:16 heft:35 sonic:16
# Every process at nice 15 (the coordinator, 2026-11-26: the failure diagnosis and the cuRobo route come first).
# Per controller, in the order given: the private copy (fs_freeze.sh) and the inputs checked against asrun/, the runs
# (fs_batch.sh from the copy: all 216 cells, episode ids 0..n-1, the controller's groups), fs_check, fs_eval, the copy
# checked again. Everything runs from the copy C0 (LC_CODE_ROOT: lc3_run's code check on every run process).
# Restartable: the runs skip every job whose record exists (a different process count is fine).
NP=$1; shift
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_freespace_20261126
C0=$O/code
A=$O/asrun
CODE=$C0/scripts/experiments/selfmodel_freespace_20261126
PY=/wuji-vefps-D/wuji-il/yijie/envs/locomanilab_py311/bin/python
G=/wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend
echo "$@" > $O/FORMAL_PLAN  # for the progress poll
md5ok() { (cd $C0 && md5sum -c --quiet $A/CODE_MD5); }
inok() { md5sum -c --quiet $A/MD5; }
if ! inok; then echo "aborted: inputs differ from asrun/MD5 $(date -u +%FT%TZ)" | tee -a $O/formal.log; exit 1; fi
for item in "$@"; do
  BE=${item%%:*}; N=${item##*:}
  mkdir -p $O/$BE
  if ! md5ok; then echo "$BE aborted: code differs from asrun/CODE_MD5 $(date -u +%FT%TZ)" | tee -a $O/formal.log; exit 1; fi
  echo "$BE formal start $(date -u +%FT%TZ): 216 cells x episode ids 0..$((N - 1)), $NP processes, code $C0 (md5 = asrun)" | tee -a $O/formal.log
  FS_NICE=15 LC_CODE_ROOT=$C0 FS_OUT=$O FS_GRID=$O/grid/grid.json bash $CODE/fs_batch.sh $BE 0:$N 0:216 $NP >> $O/formal.log 2>&1
  n=$(cat $O/$BE/runs/*.jsonl 2>/dev/null | wc -l)
  nst=$(cat $O/$BE/logs/run_n*.log | grep -c "^lc code check start: env.py $C0/")
  nf=$(cat $O/$BE/logs/run_n*.log | grep -c "^lc code check .*FAIL")
  echo "$BE runs done $(date -u +%FT%TZ): $n records; code check: $nst process starts with env.py from the copy, failures $nf" | tee -a $O/formal.log
  cd $G
  export PYTHONPATH=$G:$C0/src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 FS_OUT=$O FS_GRID=$O/grid/grid.json CUDA_VISIBLE_DEVICES= \
    REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1
  nice -n 15 $PY $CODE/fs_check.py --backend $BE 2>&1 | grep -v "Warn\|INFO" > $O/$BE/check.txt
  echo "$BE checked $(date -u +%FT%TZ): $(tail -1 $O/$BE/check.txt)" | tee -a $O/formal.log
  nice -n 15 $PY $CODE/fs_eval.py --backend $BE --json $O/$BE/eval.json 2>&1 | grep -v "Warn\|INFO" > $O/$BE/eval.txt
  echo "$BE evaluated $(date -u +%FT%TZ): $(grep VERDICT $O/$BE/eval.txt)" | tee -a $O/formal.log
  if md5ok; then echo "$BE md5 after: same as asrun" | tee -a $O/formal.log; else echo "$BE md5 after: CHANGED" | tee -a $O/formal.log; fi
done
echo "formal done $(date -u +%FT%TZ)" | tee -a $O/formal.log
