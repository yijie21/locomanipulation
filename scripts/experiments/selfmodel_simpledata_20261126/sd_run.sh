#!/usr/bin/env bash
# notebook: selfmodel_simpledata_20261126
# selfmodel_simpledata_20261126 (revision 2): from the formal collection to evaluations 1, 2 and 4 (3 once the real
# episodes are merged), from the asrun copy:
#   sd_run.sh <GPU for D frac 1.0>,<GPU for D frac 0.3>
# SD_MACHINE=4090b (revision 2 addendum: the training goes to 4090b's two cards): asrun/repo_b with the 4090b launcher
# (py4090b.py, LOCOMANI_WORKSPACE, SONIC_POLICY; SD_PY for sd_mid_check's child); default 4090 (asrun/repo, its venv).
# This experiment's scripts are always asrun/repo's. Before each GPU phase (each training, eval 1, eval 4) the card
# must have at least 8 GB free (nvidia-smi), else the run waits (other jobs share the cards).
# Phases (each in $OUT/run.log and $OUT/STATUS; any failure stops the run, STATUS "FAIL ..."):
#   code     asrun MD5.check and MD5_r2.check (revision 2's files); train3b's frozen t3b_train.py / t3b_mid.py at their md5s (c7ecf581 / 6a39671c)
#   data     $O/logs/finished (asrun/tools/finish.sh); data3 and data3b: "CHECK PASSED" and "SIMPLE CHECK PASSED";
#            the task-walk twist criterion (check item 8, "task walk end a/b = x %" >= 80 % in both); then $O/TRAIN_GO
#            (written by hand once revision 2 is committed and pushed)
#   mid      the frozen t3b_mid.py (-> $OUT/mid_t3b, for the check only) and sd_mid.py (-> $OUT/data), REACHWM_FDVEL off
#   checks   sd_mid_check.py (FDVEL off byte equality, FDVEL on differences) and sd_split_check.py (M / D)
#   train    t3b_train.py E3_ensemble, D's spec (data3, data3b, the two mid files), seed 0, frac 1.0 and 0.3 side by
#            side, REACHWM_GOAL_REF / APPROACH / QFIX / FDVEL = 1, nice 10, 4 CPU threads each; nvidia-smi logged first
#   eval     1: sd_eval.py (every split) -> $OUT/eval1.{txt,json}
#            2: xfer's sx_eval.py on selfmodel_xfer_20261125/scalebfm_simple, --lml $O/data3 (SIMPLE val), keep
#               settled and ended -> $OUT/eval2_<keep>.{txt,json}
#            3: waits for $REAL/<label>/merge.txt "REAL MERGE <label> PASS" (sd_real.sh), then sx_eval.py on
#               $REAL/<label> (main, task) with both keeps -> $OUT/eval3_<label>_<keep>.{txt,json}
#            4: sd_range.py -> $OUT/eval4.{txt,json}
# Test only: SD_TEST=1 (no collection / TRAIN_GO wait, no criterion, no md5 check) with SD_A (a repo copy's parent), SD_OUT,
# SD_D3, SD_D3B, SD_REAL, SD_ITERS, SD_XF.
set -uo pipefail
IFS=, read -ra GPUS <<< "$1"
L=/wuji-vefps-D/wuji-il/yijie/loco_outputs
O=$L/selfmodel_simpledata_20261126
A=${SD_A:-$O/asrun}
MACH=${SD_MACHINE:-4090}
E=$A/repo/scripts/experiments/selfmodel_simpledata_20261126
if [ "$MACH" = 4090b ]; then
  R=$A/repo_b
  WS=/wuji-vefps-D/wuji-il/yijie/ws
  export LOCOMANI_WORKSPACE=$WS SONIC_POLICY=$WS/sonic_models/sonic_v1_1
  PY="/usr/bin/python3.10 -S $R/scripts/experiments/selfmodel_simplecollect_20261126/py4090b.py"
  export SD_PY="$PY"
else
  R=$A/repo
  PY=$R/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
fi
T3B=$L/selfmodel_train3b_20261125/asrun/selfmodel_train3b_20261125
SX=$R/scripts/experiments/selfmodel_xfer_20261125/sx_eval.py
OUT=${SD_OUT:-$O/train}
D3=${SD_D3:-$O/data3/data/all.npz}
D3B=${SD_D3B:-$O/data3b/data/all.npz}
D3DIR=$(dirname $(dirname $D3))
REAL=${SD_REAL:-$O/real}
XF=${SD_XF:-$L/selfmodel_xfer_20261125/scalebfm_simple}
IT=${SD_ITERS:+--iters $SD_ITERS}
cd $R || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 LOCO_REPO=$R SX_REPO=$R
unset REACHWM_FDVEL REACHWM_X0_REF
mkdir -p $OUT/data $OUT/logs
phase() { echo "$1" > $OUT/STATUS; echo "$(date -u +%FT%TZ) $1" | tee -a $OUT/run.log; }
die() { phase "FAIL $1"; exit 1; }
gpu_free() {  # wait until card $1 has >= 8 GB free
  local f
  while true; do
    f=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i $1 | awk -F, '{print $1 - $2}')
    [ "$f" -ge 8000 ] && break
    echo "$(date -u +%FT%TZ) GPU $1: $f MiB free, waiting" >> $OUT/run.log; sleep 120
  done
}
[ -n "${SD_TEST:-}" ] || (cd $A && md5sum -c --quiet MD5.check && md5sum -c --quiet MD5_r2.check) || die "asrun md5"
[ "$(md5sum < $T3B/t3b_train.py | cut -c1-8)" = c7ecf581 ] && [ "$(md5sum < $T3B/t3b_mid.py | cut -c1-8)" = 6a39671c ] || die "train3b md5"
phase "code: asrun md5 ok (or SD_TEST), train3b's t3b_train / t3b_mid frozen; $MACH ($R), GPUs ${GPUS[*]}; out $OUT"
if [ -z "${SD_TEST:-}" ]; then
  phase "wait for the collection"
  until [ -f $O/logs/finished ]; do sleep 60; done
  for m in data3 data3b; do
    grep -q "^CHECK PASSED" $O/$m/check.txt && grep -q "^SIMPLE CHECK PASSED" $O/$m/check.txt || die "$m check: $(grep -c FAIL $O/$m/check.txt) FAIL lines"
    tw=$(grep -o "task walk end [0-9]*/[0-9]* = [0-9.]* %" $O/$m/check.txt | head -1)
    pc=$(echo "$tw" | awk '{print $(NF-1)}')
    echo "$(date -u +%FT%TZ) $m: checks passed; $tw" | tee -a $OUT/run.log
    awk -v p="$pc" 'BEGIN {exit !(p >= 80)}' || die "$m: task-walk twist <= 15 deg in $pc % < 80 % (criterion)"
  done
  phase "wait for TRAIN_GO"
  until [ -f $O/TRAIN_GO ]; do sleep 30; done
  echo "$(date -u +%FT%TZ) TRAIN_GO: $(cat $O/TRAIN_GO)" | tee -a $OUT/run.log
fi
phase "mid"
nice -n 10 $PY $T3B/t3b_mid.py --backend scalebfm --data3 $D3 --data3b $D3B --out $OUT/mid_t3b > $OUT/logs/mid_t3b.log 2>&1 || die "t3b_mid"
nice -n 10 $PY $E/sd_mid.py --backend scalebfm --data3 $D3 --data3b $D3B --out $OUT/data > $OUT/logs/mid.log 2>&1 || die "sd_mid"
grep -h "MID CHECKS" $OUT/logs/mid_t3b.log $OUT/logs/mid.log | tee -a $OUT/run.log
phase "mid check"
nice -n 10 $PY $E/sd_mid_check.py --old $OUT/mid_t3b --new $OUT/data --data3 $D3 --data3b $D3B > $OUT/logs/mid_check.log 2>&1 || die "mid check"
grep -v Warn $OUT/logs/mid_check.log | tee -a $OUT/run.log
MID=$OUT/data/mid_data3.npz,$OUT/data/mid_data3b.npz
phase "split check"
nice -n 10 $PY $E/sd_split_check.py --data3 $D3 --data3b $D3B --mid $MID > $OUT/logs/split_check.log 2>&1 || die "split check"
tail -n 1 $OUT/logs/split_check.log | tee -a $OUT/run.log
DS=$D3,$D3B,$MID
export REACHWM_FDVEL=1
phase "train"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> $OUT/run.log
free -g | head -2 >> $OUT/run.log
PIDS=()
for j in 0 1; do
  f=$([ $j = 0 ] && echo 1 || echo 0.3); g=${GPUS[$j]}
  log=$OUT/logs/train_D_s0_f$f.log
  gpu_free $g
  CUDA_VISIBLE_DEVICES=$g nice -n 10 $PY $T3B/t3b_train.py E3_ensemble --seed 0 --frac $f --data $DS $IT --out-root $OUT/models_D_s0 > $log 2>&1 &
  p=$!; PIDS+=($p)
  echo "$(date -u +%FT%TZ) train D seed 0 frac $f on GPU $g (FDVEL 1): pid $p" >> $OUT/run.log
  until grep -q " segments, splits" $log || ! kill -0 $p 2>/dev/null; do sleep 5; done
  echo "$(date -u +%FT%TZ)   loaded: $(grep " segments, splits" $log | cut -c1-90); free $(free -g | awk '/Mem/ {print $7}') GB" >> $OUT/run.log
done
for p in "${PIDS[@]}"; do wait $p || die "training pid $p"; done
grep -h "best val" $OUT/logs/train_*.log | tee -a $OUT/run.log
phase "eval 1"
gpu_free ${GPUS[0]}
CUDA_VISIBLE_DEVICES=${GPUS[0]} nice -n 10 $PY $E/sd_eval.py --models $OUT/models_D_s0 --data3 $D3 --data3b $D3B --json $OUT/eval1.json \
  > $OUT/eval1.raw 2>&1 || die "eval 1"
grep -v Warn $OUT/eval1.raw > $OUT/eval1.txt
grep "SIMPLE val" $OUT/eval1.txt | tee -a $OUT/run.log
phase "eval 2"
for k in settled ended; do
  CUDA_VISIBLE_DEVICES= nice -n 10 $PY $SX --lml $D3DIR --xfer $XF --models $OUT/models_D_s0 --keep $k --json $OUT/eval2_$k.json \
    > $OUT/eval2_$k.raw 2>&1 || die "eval 2 $k"
  grep -v Warn $OUT/eval2_$k.raw > $OUT/eval2_$k.txt
  grep -m1 "    all " $OUT/eval2_$k.txt | sed "s/^/eval 2 $k: /" | tee -a $OUT/run.log
done
phase "eval 4 (and 3) wait for the real episodes"
for lab in main task; do
  until grep -q "REAL MERGE $lab PASS" $REAL/$lab/merge.txt 2>/dev/null; do
    grep -q "REAL MERGE $lab FAIL" $REAL/$lab/merge.txt 2>/dev/null && die "real $lab merge failed"
    sleep 60
  done
done
phase "eval 3"
for lab in main task; do
  for k in settled ended; do
    CUDA_VISIBLE_DEVICES= nice -n 10 $PY $SX --lml $D3DIR --xfer $REAL/$lab --models $OUT/models_D_s0 --keep $k --json $OUT/eval3_${lab}_$k.json \
      > $OUT/eval3_${lab}_$k.raw 2>&1 || die "eval 3 $lab $k"
    grep -v Warn $OUT/eval3_${lab}_$k.raw > $OUT/eval3_${lab}_$k.txt
    grep -m1 "    all " $OUT/eval3_${lab}_$k.txt | sed "s/^/eval 3 $lab $k: /" | tee -a $OUT/run.log
  done
done
phase "eval 4"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> $OUT/run.log
gpu_free ${GPUS[0]}
CUDA_VISIBLE_DEVICES=${GPUS[0]} nice -n 10 $PY $E/sd_range.py --train $DS --data3 $D3 --models $OUT/models_D_s0 \
  --real main=$REAL/main/data/all.npz,task=$REAL/task/data/all.npz --json $OUT/eval4.json > $OUT/eval4.raw 2>&1 || die "eval 4"
grep -v Warn $OUT/eval4.raw > $OUT/eval4.txt
phase "done"
