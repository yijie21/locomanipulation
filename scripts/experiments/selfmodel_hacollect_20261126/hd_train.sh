#!/usr/bin/env bash
# notebook: selfmodel_hadata_20261126
# selfmodel_hadata_20261126 revision 2b: one controller's HA training on H20b, from the frozen copy asrun_train:
#   hd_train.sh <collection root> <controller> <GPUs, comma-separated> [parallel runs, default 4]
# (<collection root>: selfmodel_hadata_20261126 for scalebfm / heft, selfmodel_hadata_sonic_20261126 for sonic.)
# Reader and launcher: simpledata's frozen asrun/repo_b (reachwm data.py 174071cb = fdvel's FDVEL reader, train.py,
# models.py) through its py4090b.py (H20b has no /workspace); train3b's frozen t3b_train.py / t3b_mid.py; simpledata's
# frozen sd_mid.py / sd_mid_check.py / sd_split_check.py (revision 2). Phases (each in $O/run.log and $O/STATUS; any
# failure stops the run, STATUS "FAIL ..."):
#   code     asrun_train's MD5.check (its own scripts and every pinned file above)
#   data     <ctrl>/d3 and d3b: FINISHED, finish_status all 0, "CHECK PASSED" and "XFER CHECK PASSED" in check.txt;
#            then $ROOT/TRAIN_GO_2b (written by hand once revision 2b is committed and pushed)
#   touch    hd_touch.py on d3 and d3b -> $O/data/d3.npz, d3b.npz (revision 1's contact rule; FDVEL reader)
#   mid      t3b_mid.py (-> $O/mid_t3b, for the check) and sd_mid.py (-> $O/data/mid_data3.npz, mid_data3b.npz) on the
#            filtered files, REACHWM_FDVEL off
#   checks   sd_mid_check.py, sd_split_check.py
#   two-hand hd_touch.py on d3's and d3b's data_both -> $O/data/both_d3.npz, both_d3b.npz (evaluation only: gate (b)'s
#            two-hand row), hd_both_check.py (their val / test_iid resets are not in the training)
#   train    hd_train.py (t3b_train's training, val every 1000 iterations; T: F's weights, HA's normalisation), D's spec
#            (d3, d3b, the two mid files), REACHWM_GOAL_REF / APPROACH / QFIX / FDVEL = 1, 30000 iterations, nice 10, 4 CPU
#            threads each, at most <parallel> at once, round-robin over the GPUs, in this order:
#              S f1 s0, T f1 s0, S f1 s1, T f1 s1, S f1 s2, T f1 s2, S f0.6 s0, T f0.6 s0, S f0.3 s0, T f0.3 s0
#            (S from scratch, T fine-tuned from fdvel's F of the controller, seed 0) -> $O/models_<S|T>_s<seed>/E3_ensemble_f<frac>
#   done     every run's best val in run.log
# Test only: HT_TEST=1 (no data / TRAIN_GO wait, no md5 check) with HT_OUT (output dir), HT_ITERS (iterations), HT_RUNS (names).
set -uo pipefail
ROOT=$1 C=$2 MAXP=${4:-4}
IFS=, read -ra GPUS <<< "$3"
Y=/wuji-vefps-D/wuji-il/yijie; L=$Y/loco_outputs; WS=$Y/ws
OH=$L/selfmodel_hadata_20261126
A=${HT_ASRUN:-$OH/asrun_train}
R=$L/selfmodel_simpledata_20261126/asrun/repo_b
SDE=$L/selfmodel_simpledata_20261126/asrun/repo/scripts/experiments/selfmodel_simpledata_20261126
T3B=$L/selfmodel_train3b_20261125/asrun/selfmodel_train3b_20261125
F=$L/selfmodel_fdvel_20261126/$C/models_s0/E3_ensemble_f1/model.pt
E=$A/selfmodel_hacollect_20261126
X=$ROOT/$C
O=${HT_OUT:-$ROOT/train/$C}
IT=${HT_ITERS:+--iters $HT_ITERS}
RUNS=${HT_RUNS:-S_f1_s0 T_f1_s0 S_f1_s1 T_f1_s1 S_f1_s2 T_f1_s2 S_f0.6_s0 T_f0.6_s0 S_f0.3_s0 T_f0.3_s0}
PY="/usr/bin/python3.10 -S $R/scripts/experiments/selfmodel_simplecollect_20261126/py4090b.py"
export LOCOMANI_WORKSPACE=$WS SONIC_POLICY=$WS/sonic_models/sonic_v1_1 LOCO_REPO=$R SX_REPO=$R SD_PY="$PY"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1
unset REACHWM_FDVEL REACHWM_X0_REF REACHWM_SUBSET REACHWM_TRAIN_HELD REACHWM_ROOT
mkdir -p $O/data $O/logs
phase() { echo "$1" > $O/STATUS; echo "$(date -u +%FT%TZ) $1" | tee -a $O/run.log; }
die() { phase "FAIL $1"; exit 1; }
cd $R || die "no $R"
[ -n "${HT_TEST:-}" ] || (cd $A && md5sum -c --quiet MD5.check) || die "asrun_train md5"
phase "code: asrun_train $A ($(cat $A/FROZEN_UTC 2>/dev/null)); $C on $(hostname), GPUs ${GPUS[*]}, $MAXP at once; out $O"
if [ -z "${HT_TEST:-}" ]; then
  phase "wait for the collections and TRAIN_GO_2b"
  for m in d3 d3b; do
    until [ -f $X/$m/FINISHED ]; do sleep 60; done
    [ "$(awk '$2 != 0' $X/$m/finish_status | wc -l)" = 0 ] || die "$m finish_status: $(tr '\n' ' ' < $X/$m/finish_status)"
    grep -q "^CHECK PASSED" $X/$m/check.txt && grep -q "^XFER CHECK PASSED" $X/$m/check.txt || die "$m check: $(grep -c FAIL $X/$m/check.txt) FAIL lines"
    echo "$(date -u +%FT%TZ) $m: finished $(cat $X/$m/FINISHED), checks passed" | tee -a $O/run.log
  done
  until [ -f $ROOT/TRAIN_GO_2b ]; do sleep 30; done
  echo "$(date -u +%FT%TZ) TRAIN_GO_2b: $(cat $ROOT/TRAIN_GO_2b)" | tee -a $O/run.log
fi
export REACHWM_FDVEL=1
phase "touch"
for m in d3 d3b; do
  nice -n 10 $PY $E/hd_touch.py $X/$m $O/data/$m.npz > $O/logs/touch_$m.log 2>&1 || die "touch $m"
  grep -v Warn $O/logs/touch_$m.log | grep -v "^skip" | tee -a $O/run.log
done
unset REACHWM_FDVEL
phase "mid"
nice -n 10 $PY $T3B/t3b_mid.py --backend $C --data3 $O/data/d3.npz --data3b $O/data/d3b.npz --out $O/mid_t3b > $O/logs/mid_t3b.log 2>&1 || die "t3b_mid"
nice -n 10 $PY $SDE/sd_mid.py --backend $C --data3 $O/data/d3.npz --data3b $O/data/d3b.npz --out $O/data > $O/logs/mid.log 2>&1 || die "sd_mid"
grep -h "MID CHECKS" $O/logs/mid_t3b.log $O/logs/mid.log | tee -a $O/run.log
phase "mid check"
nice -n 10 $PY $SDE/sd_mid_check.py --old $O/mid_t3b --new $O/data --data3 $O/data/d3.npz --data3b $O/data/d3b.npz > $O/logs/mid_check.log 2>&1 || die "mid check"
grep -v Warn $O/logs/mid_check.log | tail -n 3 | tee -a $O/run.log
MID=$O/data/mid_data3.npz,$O/data/mid_data3b.npz
phase "split check"
nice -n 10 $PY $SDE/sd_split_check.py --data3 $O/data/d3.npz --data3b $O/data/d3b.npz --mid $MID > $O/logs/split_check.log 2>&1 || die "split check"
tail -n 1 $O/logs/split_check.log | tee -a $O/run.log
DS=$O/data/d3.npz,$O/data/d3b.npz,$MID
export REACHWM_FDVEL=1
phase "two-hand"
for m in d3 d3b; do
  nice -n 10 $PY $E/hd_touch.py $X/$m $O/data/both_$m.npz data_both > $O/logs/touch_both_$m.log 2>&1 || die "touch both $m"
  grep -v Warn $O/logs/touch_both_$m.log | grep -v "^skip" | tee -a $O/run.log
done
nice -n 10 $PY $E/hd_both_check.py --train $DS --both $O/data/both_d3.npz,$O/data/both_d3b.npz > $O/logs/both_check.log 2>&1 || die "both check"
grep -v Warn $O/logs/both_check.log | tee -a $O/run.log
phase "train"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> $O/run.log
free -g | head -2 >> $O/run.log
md5sum $F >> $O/run.log
declare -A PID
i=0
for r in $RUNS; do
  IFS=_ read -r arm fr sd <<< "$r"
  f=${fr#f} s=${sd#s}
  g=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i + 1))
  while [ "$(jobs -rp | wc -l)" -ge "$MAXP" ]; do sleep 10; done
  init=""; [ "$arm" = T ] && init="--init $F"
  log=$O/logs/train_$r.log
  CUDA_VISIBLE_DEVICES=$g nice -n 10 $PY $E/hd_train.py --t3b $T3B/t3b_train.py --frac $f --seed $s --data $DS $IT \
    --out-root $O/models_${arm}_s$s $init > $log 2>&1 &
  p=$!; PID[$r]=$p
  echo "$(date -u +%FT%TZ) train $r ($arm frac $f seed $s) on GPU $g: pid $p" >> $O/run.log
  until grep -q " segments, splits" $log || ! kill -0 $p 2>/dev/null; do sleep 5; done
done
for r in $RUNS; do wait ${PID[$r]} || die "training $r (pid ${PID[$r]})"; done
for r in $RUNS; do echo "$r: $(grep -h "best val" $O/logs/train_$r.log)"; done | tee -a $O/run.log
phase "done"
