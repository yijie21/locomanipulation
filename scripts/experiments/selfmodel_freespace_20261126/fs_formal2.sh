#!/usr/bin/env bash
# notebook: selfmodel_freespace_20261126
# Revision 2's driver for the rest of E2a (tmux fs_formal2), taking over from fs_formal.sh (stopped by pid with SIGSTOP
# while ScaleBFM's runs go on; that driver would have run HEFT at 35 ids everywhere):
#   fs_formal2.sh <old fs_formal.sh pid>
# 1. waits until ScaleBFM's run processes (scalebfm/pids.txt, the 08:17:37Z launch) have all ended, then kills the
#    stopped old driver (SIGKILL by pid, after checking its command line) and runs ScaleBFM's records count, fs_check,
#    fs_eval and md5-after exactly as fs_formal.sh does;
# 2. HEFT in two steps (coordinator, 2026-11-26, to save CPU, decided after the smoke): every cell with ids 0..15; then
#    fs_heft2.py picks the cells with >= 1 valid pair in those 16 and only those run ids 16..34; then check / eval;
# 3. SONIC with ids 0..21 in every cell, then check / eval.
# Everything from the private copy C0, nice 19 (coordinator: E2a gives way to the failure diagnosis), 8 processes.
# The copy's original files are checked against asrun/CODE_MD5, revision 2's added files against asrun/CODE_MD5_REV2.
OLD=$1
NP=8
NI=19
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_freespace_20261126
C0=$O/code
A=$O/asrun
CODE=$C0/scripts/experiments/selfmodel_freespace_20261126
PY=/wuji-vefps-D/wuji-il/yijie/envs/locomanilab_py311/bin/python
G=/wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend
LOG=$O/formal.log
md5ok() { (cd $C0 && md5sum -c --quiet $A/CODE_MD5 && md5sum -c --quiet $A/CODE_MD5_REV2); }
log() { echo "$*" | tee -a $LOG; }
alive() { [ -r /proc/$1/cmdline ] && tr '\0' ' ' < /proc/$1/cmdline | grep -q "$2"; }
totals() {  # for the progress poll: <backend> <episodes planned>
  printf "scalebfm %d\nheft %d\nsonic %d\n" $((216 * 16 * 3)) $1 $((216 * 22 * 2)) > $O/FORMAL_TOTALS
}
post() {  # fs_formal.sh's steps after a controller's runs
  local BE=$1
  n=$(cat $O/$BE/runs/*.jsonl 2>/dev/null | wc -l)
  nst=$(cat $O/$BE/logs/run_n*.log | grep -c "^lc code check start: env.py $C0/")
  nf=$(cat $O/$BE/logs/run_n*.log | grep -c "^lc code check .*FAIL")
  log "$BE runs done $(date -u +%FT%TZ): $n records; code check: $nst process starts with env.py from the copy, failures $nf"
  (cd $G && PYTHONPATH=$G:$C0/src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 FS_OUT=$O FS_GRID=$O/grid/grid.json CUDA_VISIBLE_DEVICES= \
    REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 nice -n $NI $PY $CODE/fs_check.py --backend $BE 2>&1 | grep -v "Warn\|INFO" > $O/$BE/check.txt)
  log "$BE checked $(date -u +%FT%TZ): $(tail -1 $O/$BE/check.txt)"
  (cd $G && PYTHONPATH=$G:$C0/src OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 FS_OUT=$O FS_GRID=$O/grid/grid.json CUDA_VISIBLE_DEVICES= \
    REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 nice -n $NI $PY $CODE/fs_eval.py --backend $BE --json $O/$BE/eval.json 2>&1 | grep -v "Warn\|INFO" > $O/$BE/eval.txt)
  log "$BE evaluated $(date -u +%FT%TZ): $(grep VERDICT $O/$BE/eval.txt)"
  if md5ok; then log "$BE md5 after: same as asrun"; else log "$BE md5 after: CHANGED"; fi
}
batch() {  # <backend> <ids a:b> <cells>
  if ! md5ok; then log "$1 aborted: code differs from asrun/CODE_MD5(_REV2) $(date -u +%FT%TZ)"; exit 1; fi
  FS_NICE=$NI LC_CODE_ROOT=$C0 FS_OUT=$O FS_GRID=$O/grid/grid.json bash $CODE/fs_batch.sh $1 $2 "$3" $NP >> $LOG 2>&1
}
totals $((216 * 35 * 2))
log "fs_formal2 start $(date -u +%FT%TZ): waiting for ScaleBFM's runs (old driver pid $OLD, stopped)"
for p in $(grep "scalebfm .*/8 2026-11-26T08:17:37Z" $O/scalebfm/pids.txt | awk '{print $1}'); do
  while alive $p "fs_run.py run --backend scalebfm"; do sleep 60; done
done
for i in $(seq 30); do pgrep -f "[f]s_batch.sh scalebfm 0:16" > /dev/null || break; sleep 10; done  # its "runs finished" line
if alive $OLD "fs_formal.sh 8 scalebfm:16 heft:35 sonic:22"; then kill -9 $OLD && log "old driver $OLD killed $(date -u +%FT%TZ)"; fi
post scalebfm
# HEFT, step 1
log "heft formal start $(date -u +%FT%TZ): step 1, 216 cells x ids 0..15, $NP processes, nice $NI, code $C0"
batch heft 0:16 0:216
(cd $G && PYTHONPATH=$G:$C0/src OMP_NUM_THREADS=1 FS_OUT=$O FS_GRID=$O/grid/grid.json CUDA_VISIBLE_DEVICES= \
  nice -n $NI $PY $CODE/fs_heft2.py --eps 0:16 --out $O/heft/stage2 2>&1 | grep -v "Warn\|INFO" | tail -3 >> $LOG)
if [ ! -s $O/heft/stage2_cells.txt ]; then log "heft aborted: step 1 incomplete or no selection file $(date -u +%FT%TZ)"; exit 1; fi
CELLS=$(tr -d '\n' < $O/heft/stage2_cells.txt)
NSEL=$(echo "$CELLS" | tr ',' '\n' | grep -c .)
totals $((216 * 16 * 2 + NSEL * 19 * 2))
log "heft step 2 $(date -u +%FT%TZ): $NSEL cells x ids 16..34"
[ "$NSEL" -gt 0 ] && batch heft 16:35 "$CELLS"
post heft
# SONIC
log "sonic formal start $(date -u +%FT%TZ): 216 cells x ids 0..21, $NP processes, nice $NI, code $C0"
batch sonic 0:22 0:216
post sonic
log "formal done $(date -u +%FT%TZ)"
