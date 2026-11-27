#!/usr/bin/env bash
# notebook: selfmodel_freespace_20261126
# Revision 3's 4090 driver (tmux fs_formal3a on 4090), in place of fs_formal2.sh (killed by pid while it only waited):
#   fs_formal3a.sh <old fs_formal.sh pid>
# H20 takes over HEFT and SONIC (fs_formal3b.sh); 4090 only finishes ScaleBFM. This is fs_formal2.sh's step 1, unchanged:
# waits until ScaleBFM's run processes (scalebfm/pids.txt, the 08:17:37Z launch) have all ended, kills the stopped old
# driver (SIGKILL by pid, after checking its command line), then ScaleBFM's records count, fs_check, fs_eval and
# md5-after as fs_formal2.sh's post (nice 19). Then writes $O/SCALEBFM_POST_DONE (fs_formal3b.sh's final tally waits for
# it). Starts no run process: no HEFT, no SONIC on 4090.
# Copy checks: the original files against asrun/CODE_MD5, revision 2's against CODE_MD5_REV2, revision 3's against
# CODE_MD5_REV3. The code-check count compares env.py's path with the copy's realpath (lc3's code check prints resolved
# paths).
OLD=$1
NI=19
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_freespace_20261126
C0=$O/code
C0R=$(realpath $C0)
A=$O/asrun
CODE=$C0/scripts/experiments/selfmodel_freespace_20261126
PY=/wuji-vefps-D/wuji-il/yijie/envs/locomanilab_py311/bin/python
G=/wuji-vefps-D/wuji-il/yijie/code/locomanilab_20261117/external/g1_backend
LOG=$O/formal.log
md5ok() { (cd $C0 && md5sum -c --quiet $A/CODE_MD5 && md5sum -c --quiet $A/CODE_MD5_REV2 && md5sum -c --quiet $A/CODE_MD5_REV3); }
log() { echo "$*" | tee -a $LOG; }
alive() { [ -r /proc/$1/cmdline ] && tr '\0' ' ' < /proc/$1/cmdline | grep -q "$2"; }
post() {  # fs_formal2.sh's post, unchanged except the copy's realpath in the code-check count
  local BE=$1
  n=$(cat $O/$BE/runs/*.jsonl 2>/dev/null | wc -l)
  nst=$(cat $O/$BE/logs/run_n*.log | grep -c "^lc code check start: env.py $C0R/")
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
if pgrep -f "[f]s_formal2.sh" > /dev/null; then log "fs_formal3a refused $(date -u +%FT%TZ): fs_formal2.sh still runs"; exit 1; fi
log "fs_formal3a start $(date -u +%FT%TZ) (4090): waiting for ScaleBFM's runs (old driver pid $OLD, stopped); HEFT and SONIC run on H20 (fs_formal3b.sh)"
for p in $(grep "scalebfm .*/8 2026-11-26T08:17:37Z" $O/scalebfm/pids.txt | awk '{print $1}'); do
  while alive $p "fs_run.py run --backend scalebfm"; do sleep 60; done
done
for i in $(seq 30); do pgrep -f "[f]s_batch.sh scalebfm 0:16" > /dev/null || break; sleep 10; done  # its "runs finished" line
if alive $OLD "fs_formal.sh 8 scalebfm:16 heft:35 sonic:22"; then kill -9 $OLD && log "old driver $OLD killed $(date -u +%FT%TZ)"; fi
post scalebfm
log "fs_formal3a done $(date -u +%FT%TZ) (4090): ScaleBFM finished and checked; 4090 starts nothing else"
date -u +%FT%TZ > $O/SCALEBFM_POST_DONE  # last: fs_formal3b.sh writes to formal.log only after this
