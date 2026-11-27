#!/usr/bin/env bash
# notebook: selfmodel_simplesonicdata_20261126 -- REVISION 1 (2026-11-27, before any training): the frozen ssd_train.sh
# (b432976c) with one change: data3b's check.txt is accepted when it is the one revision 1 documents (md5 706111ce...:
# CHECK PASSED, SIMPLE CHECK FAILED (1), the one hand point of item (2) in data_both, which neither training nor the gate
# reads); data3 still needs both PASSED lines. Run from $O/asrun/rev1/ by rev1/ssd_chain.sh.
# From the finished collection to the model D_S and its held-out errors, on
# H20b from the frozen copy (selfmodel_simpledata_20261126's sd_run.sh with SONIC: the same phases, scripts and recipe):
#   ssd_train.sh <GPU for frac 1.0>,<GPU for frac 0.3>
# Phases ($OUT/run.log, $OUT/STATUS; a failure stops the run, STATUS "FAIL ..."):
#   code    $O/asrun/code CODE_MD5; train3b's frozen t3b_train.py / t3b_mid.py at their md5s (c7ecf581 / 6a39671c)
#   data    waits for $O/logs/finished (ssd_collect.sh formal's finisher); data3 and data3b: "CHECK PASSED" and "SIMPLE
#           CHECK PASSED" (no walk: simpledata's task-walk criterion does not apply); ssd_datacheck.py (>= 15 000 usable
#           one-hand segments, fell + lost <= 40 % of the resets) -> $OUT/datacheck.{txt,json}
#   mid     the frozen t3b_mid.py (-> $OUT/mid_t3b, for the check) and sd_mid.py (-> $OUT/data), --backend sonic, FDVEL off
#   checks  sd_mid_check.py, sd_split_check.py
#   train   t3b_train.py E3_ensemble, simpledata's D recipe (data3, data3b, the two mid files; 30 000 steps, batch 1024,
#           5 members), seed 0, frac 1.0 and 0.3 side by side, REACHWM_GOAL_REF / APPROACH / QFIX / FDVEL = 1, nice 10, 4
#           threads each; each card must have >= 8 GB free first (nvidia-smi logged)
#   eval 1  sd_eval.py (every split) -> $OUT/eval1.{txt,json}
#   eval 2  xfer's sx_eval.py on selfmodel_xfer_20261125/sonic_simple (xfer's SONIC SIMPLE record, 400 resets), keep
#           settled and ended -> $OUT/eval2_<keep>.{txt,json}
#   done    $OUT/DONE
set -uo pipefail
IFS=, read -ra GPUS <<< "$1"
Y=/wuji-vefps-D/wuji-il/yijie; L=$Y/loco_outputs; WS=$Y/ws
O=$L/selfmodel_simplesonicdata_20261126
R=${SSD_CODE:-$O/asrun/code}
OUT=${SSD_OUT:-$O/train}
D3=${SSD_D3:-$O/data3/data/all.npz} D3B=${SSD_D3B:-$O/data3b/data/all.npz}
D3DIR=$(dirname $(dirname $D3))
T3B=$L/selfmodel_train3b_20261125/asrun/selfmodel_train3b_20261125
E=$R/scripts/experiments/selfmodel_simpledata_20261126
SX=$R/scripts/experiments/selfmodel_xfer_20261125/sx_eval.py
XF=$L/selfmodel_xfer_20261125/sonic_simple
IT=${SSD_ITERS:+--iters $SSD_ITERS}
export LOCOMANI_WORKSPACE=$WS SONIC_POLICY=$WS/sonic_models/sonic_v1_1
PY="/usr/bin/python3.10 -S $R/scripts/experiments/selfmodel_simplecollect_20261126/py4090b.py"
export SD_PY="$PY"
cd $R || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 LOCO_REPO=$R SX_REPO=$R PYTHONDONTWRITEBYTECODE=1
unset REACHWM_FDVEL REACHWM_X0_REF
mkdir -p $OUT/data $OUT/logs
phase() { echo "$1" > $OUT/STATUS; echo "$(date -u +%FT%TZ) $1" | tee -a $OUT/run.log; }
die() { phase "FAIL $1"; exit 1; }
gpu_free() {
  local f
  while true; do
    f=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i $1 | awk -F, '{print $1 - $2}')
    [ "$f" -ge 8000 ] && break
    echo "$(date -u +%FT%TZ) GPU $1: $f MiB free, waiting" >> $OUT/run.log; sleep 120
  done
}
[ -n "${SSD_TEST:-}" ] || md5sum -c --quiet CODE_MD5 || die "asrun CODE_MD5"
[ "$(md5sum < $T3B/t3b_train.py | cut -c1-8)" = c7ecf581 ] && [ "$(md5sum < $T3B/t3b_mid.py | cut -c1-8)" = 6a39671c ] || die "train3b md5"
phase "code: $R (CODE_MD5 ok or SSD_TEST), train3b's t3b_train / t3b_mid frozen; GPUs ${GPUS[*]}; out $OUT"
if [ -z "${SSD_TEST:-}" ]; then
  phase "wait for the collection"
  until [ -f $O/logs/finished ]; do [ -f $O/logs/failed ] && die "collection failed"; sleep 60; done
  m=data3
  grep -q "^CHECK PASSED" $O/$m/check.txt && grep -q "^SIMPLE CHECK PASSED" $O/$m/check.txt || die "$m check: $(grep -c FAIL $O/$m/check.txt) FAIL lines"
  echo "$(date -u +%FT%TZ) $m: checks passed" | tee -a $OUT/run.log
  m=data3b  # revision 1: the documented check.txt only (a deviation, the check unchanged)
  grep -q "^CHECK PASSED" $O/$m/check.txt && [ "$(md5sum < $O/$m/check.txt | cut -d' ' -f1)" = 706111ce14577b7f9339cb892f7b58fa ] \
    || die "$m check: not the check.txt revision 1 documents"
  echo "$(date -u +%FT%TZ) $m: CHECK PASSED; SIMPLE CHECK FAILED (1) accepted as revision 1's deviation (check.txt md5 706111ce)" | tee -a $OUT/run.log
fi
nice -n 10 $PY $R/scripts/experiments/selfmodel_simplesonicdata_20261126/ssd_datacheck.py $D3DIR $(dirname $(dirname $D3B)) --json $OUT/datacheck.json \
  > $OUT/datacheck.txt 2>&1; rc=$?
grep -v Warn $OUT/datacheck.txt | tee -a $OUT/run.log
[ -n "${SSD_TEST:-}" ] || [ $rc = 0 ] || die "the data criteria (datacheck.txt)"
phase "mid"
nice -n 10 $PY $T3B/t3b_mid.py --backend sonic --data3 $D3 --data3b $D3B --out $OUT/mid_t3b > $OUT/logs/mid_t3b.log 2>&1 || die "t3b_mid"
nice -n 10 $PY $E/sd_mid.py --backend sonic --data3 $D3 --data3b $D3B --out $OUT/data > $OUT/logs/mid.log 2>&1 || die "sd_mid"
grep -h "MID CHECKS" $OUT/logs/mid_t3b.log $OUT/logs/mid.log | tee -a $OUT/run.log
phase "mid check"
nice -n 10 $PY $E/sd_mid_check.py --old $OUT/mid_t3b --new $OUT/data --data3 $D3 --data3b $D3B > $OUT/logs/mid_check.log 2>&1 || die "mid check"
grep -v Warn $OUT/logs/mid_check.log | tee -a $OUT/run.log
MID=$OUT/data/mid_data3.npz,$OUT/data/mid_data3b.npz
phase "split check"
nice -n 10 $PY $E/sd_split_check.py --data3 $D3 --data3b $D3B --mid $MID > $OUT/logs/split_check.log 2>&1 || die "split check"
tail -n 1 $OUT/logs/split_check.log | tee -a $OUT/run.log
DS=$D3,$D3B,$MID
echo "$DS" > $OUT/DSPEC
export REACHWM_FDVEL=1
phase "train"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> $OUT/run.log
PIDS=()
for j in 0 1; do
  f=$([ $j = 0 ] && echo 1 || echo 0.3); g=${GPUS[$j]}
  log=$OUT/logs/train_D_s0_f$f.log
  gpu_free $g
  CUDA_VISIBLE_DEVICES=$g nice -n 10 $PY $T3B/t3b_train.py E3_ensemble --seed 0 --frac $f --data $DS $IT --out-root $OUT/models_D_s0 > $log 2>&1 &
  p=$!; PIDS+=($p)
  echo "$(date -u +%FT%TZ) train D_S seed 0 frac $f on GPU $g (FDVEL 1): pid $p" >> $OUT/run.log
  until grep -q " segments, splits" $log || ! kill -0 $p 2>/dev/null; do sleep 5; done
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
date -u +%FT%TZ > $OUT/DONE
phase "done"
