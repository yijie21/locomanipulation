#!/usr/bin/env bash
# notebook: selfmodel_freespace_20261126
# Revision 3's H20 driver (tmux fs_formal3 on H20): HEFT (revision 2's two steps) and SONIC, which fs_formal2.sh would
# have run on 4090 after ScaleBFM, from the same frozen copy C0 (CPU only, single-threaded, nice 19), plus H20's
# LOCOMANI_WORKSPACE. 4090 finishes only ScaleBFM (fs_formal3a.sh).
#   fs_formal3b.sh
# 1. HEFT step 1 (216 cells x ids 0..15, NH1 processes) and SONIC (216 x ids 0..21, NS processes) start together.
# 2. When step 1 has ended: fs_heft2.py picks the cells with >= 1 valid pair (revision 2's rule, unchanged), FORMAL_TOTALS
#    gets HEFT's real total, step 2 runs those cells x ids 16..34 (NH2 processes; NH2 != NH1, so fs_batch.sh's
#    run_n<NP>_<k>.log of step 1 are not overwritten); then HEFT's post.
# 3. When SONIC has ended: SONIC's post.
# 4. When 4090 has written $O/SCALEBFM_POST_DONE: fs_tally.py over the three controllers (formal_h20.log and formal.log),
#    then "formal done" in formal.log (what fs_formal2.sh wrote last).
# Process counts are odd: with two groups per (cell, ep) and the job list sharded k::NP, an even NP would give every
# shard only A or only R. An episode depends only on (cell, ep, group): the shard count changes nothing in the records.
# post = fs_formal2.sh's (records count, code-check count, fs_check, fs_eval, md5 after), the code-check count against
# the copy's realpath (on H20 /wuji-vefps-D/wuji-il is a link to /wuji-vefps-D and lc3's check prints resolved paths).
# Log: $O/formal_h20.log; formal.log (4090's) gets only the final tally and "formal done", after 4090's last line.
NH1=${NH1:-23}
NS=${NS:-33}
NH2=${NH2:-21}
NI=19
Y=/wuji-vefps-D/wuji-il/yijie
O=$Y/loco_outputs/selfmodel_freespace_20261126
C0=$O/code
C0R=$(realpath $C0)
A=$O/asrun
CODE=$C0/scripts/experiments/selfmodel_freespace_20261126
PY=$Y/envs/locomanilab_py311/bin/python
G=$Y/code/locomanilab_20261117/external/g1_backend
LOG=$O/formal_h20.log
export LOCOMANI_WORKSPACE=$Y/ws CUDA_VISIBLE_DEVICES=
unset D3_SONIC_INFER
md5ok() { (cd $C0 && md5sum -c --quiet $A/CODE_MD5 && md5sum -c --quiet $A/CODE_MD5_REV2 && md5sum -c --quiet $A/CODE_MD5_REV3); }
inok() { md5sum -c --quiet $A/MD5; }
log() { echo "$*" | tee -a $LOG; }
totals() {  # fs_formal2.sh's: <HEFT episodes planned>
  printf "scalebfm %d\nheft %d\nsonic %d\n" $((216 * 16 * 3)) $1 $((216 * 22 * 2)) > $O/FORMAL_TOTALS
}
post() {
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
batch() {  # <backend> <ids a:b> <cells> <processes>
  FS_NICE=$NI LC_CODE_ROOT=$C0 FS_OUT=$O FS_GRID=$O/grid/grid.json bash $CODE/fs_batch.sh $1 $2 "$3" $4 >> $LOG 2>&1
}
[ -f $O/H20_OWNS_HEFT_SONIC ] || { log "fs_formal3b refused $(date -u +%FT%TZ): no $O/H20_OWNS_HEFT_SONIC"; exit 1; }
if ! inok; then log "fs_formal3b aborted: inputs differ from asrun/MD5 $(date -u +%FT%TZ)"; exit 1; fi
if ! md5ok; then log "fs_formal3b aborted: code differs from asrun/CODE_MD5(_REV2/_REV3) $(date -u +%FT%TZ)"; exit 1; fi
log "fs_formal3b start $(date -u +%FT%TZ) on $(hostname): HEFT step 1 (216 cells x ids 0..15, $NH1 processes) and SONIC (216 x ids 0..21, $NS processes) together, nice $NI, CPU only, code $C0 (realpath $C0R; md5 = asrun); existing records: heft $(cat $O/heft/runs/*.jsonl 2>/dev/null | wc -l), sonic $(cat $O/sonic/runs/*.jsonl 2>/dev/null | wc -l)"
batch heft 0:16 0:216 $NH1 &
PH=$!
batch sonic 0:22 0:216 $NS &
PS=$!
wait $PH
log "heft step 1 runs ended $(date -u +%FT%TZ)"
(cd $G && PYTHONPATH=$G:$C0/src OMP_NUM_THREADS=1 FS_OUT=$O FS_GRID=$O/grid/grid.json CUDA_VISIBLE_DEVICES= \
  nice -n $NI $PY $CODE/fs_heft2.py --eps 0:16 --out $O/heft/stage2 2>&1 | grep -v "Warn\|INFO" | tail -3 >> $LOG)
if [ ! -s $O/heft/stage2_cells.txt ]; then
  log "heft aborted: step 1 incomplete or no selection file $(date -u +%FT%TZ)"
else
  CELLS=$(tr -d '\n' < $O/heft/stage2_cells.txt)
  NSEL=$(echo "$CELLS" | tr ',' '\n' | grep -c .)
  totals $((216 * 16 * 2 + NSEL * 19 * 2))
  log "heft step 2 $(date -u +%FT%TZ): $NSEL cells x ids 16..34, $NH2 processes"
  if [ "$NSEL" -gt 0 ]; then
    if md5ok; then batch heft 16:35 "$CELLS" $NH2; else log "heft aborted: code differs from asrun/CODE_MD5(_REV2/_REV3) $(date -u +%FT%TZ)"; fi
  fi
  post heft
fi
wait $PS
log "sonic runs ended $(date -u +%FT%TZ)"
post sonic
log "waiting for 4090's ScaleBFM ($O/SCALEBFM_POST_DONE) $(date -u +%FT%TZ)"
while [ ! -f $O/SCALEBFM_POST_DONE ]; do sleep 60; done
T=$(cd $G && PYTHONPATH=$G:$C0/src FS_OUT=$O nice -n $NI $PY $CODE/fs_tally.py 2>&1 | grep -v "Warn\|INFO")
echo "$T" >> $LOG
echo "$T" | tail -1 >> $O/formal.log
log "formal done $(date -u +%FT%TZ)"
echo "formal done $(date -u +%FT%TZ) (fs_formal3b on H20, after 4090's ScaleBFM post; tally in formal_h20.log)" >> $O/formal.log
