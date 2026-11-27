#!/usr/bin/env bash
# notebook: selfmodel_train3b_20261125
# selfmodel_train3b_20261125 (revision 0), one controller on this machine (tmux t3b_<backend>):
#   t3b_run.sh <backend> <GPUs, comma-separated, assigned to the 11 trainings in turn>
# Waits for selfmodel_data3b_20261125's formal collection of <backend> to merge with CHECK PASSED, then
#   mid starts (t3b_mid.py) -> split check M / D (t3b_split_check.py) -> C-row check (t3b_cref.py; MX is skipped if it
#   fails) -> 11 trainings side by side (t3b_train.py E3_ensemble, train3's recipe: M seeds 0, 1, 2 at frac 1, seed 0
#   at frac 0.3 / 0.6; MX = M with REACHWM_X0_REF=1, seeds 0, 1, 2; D seeds 0, 1, 2; nice 10, 4 CPU threads each; each
#   started once the previous one has loaded its data, so one load's transient memory at a time)
#   MX waits for $T/MX_DECISION (first word "train" or "skip"; "train <spec>" trains MX on <spec> instead of M's data),
#   written by hand after the C-row question is settled; M and D train meanwhile
#   -> t3b_eval.py on val, test_iid and edge (T3 = train3's frac 1.0 seed 0 model; MX's errors from a REACHWM_X0_REF=1
#   --dump run, merged into the main run).
# Code: the frozen copy in $A (t3b_freeze.sh); the repo's reachwm / selfwm files it imports must equal $A's copies.
# Every phase in $O/run.log and $O/STATUS; any failure stops the run (STATUS "FAIL ...").
# Test only: T3B_B (data3b output dir), T3B_OUT (this run's output dir), T3B_ITERS (iterations; default train.py's).
set -uo pipefail
BE=$1; IFS=, read -ra GPUS <<< "$2"; NG=0
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
R=/wuji-vefps-D/wuji-il/yijie/loco_outputs
T=$R/selfmodel_train3b_20261125
O=${T3B_OUT:-$T/$BE}
A=$T/asrun
D3=$R/selfmodel_data3_20261125/$BE/data/all.npz
B=${T3B_B:-$R/selfmodel_data3b_20261125/$BE}
IT=${T3B_ITERS:+--iters $T3B_ITERS}
D3B=$B/data/all.npz
cd $W || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 LOCO_REPO=$W
VPY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
mkdir -p $O/data $O/logs
phase() { echo "$1" > $O/STATUS; echo "$(date -u +%FT%TZ) $BE $1" | tee -a $O/run.log; }
die() { phase "FAIL $1"; exit 1; }
(cd $A && md5sum -c --quiet MD5) || die "asrun md5"
for f in reachwm_20261030/{data,train,models}.py selfwm_20261029/{env,wm,fk}.py; do
  cmp -s scripts/experiments/$f $A/$f || die "repo $f differs from asrun"
done
phase "code: asrun md5 ok, repo reachwm / selfwm files = asrun's; GPUs ${GPUS[*]} in turn on $(hostname)"
phase "wait for data3b $BE"
until grep -q merged $B/formal.log 2>/dev/null; do sleep 60; done
grep -q "CHECK PASSED" $B/check.log || die "data3b $BE check did not pass: $(tail -1 $B/check.log)"
phase "data3b: $(grep merged $B/formal.log | tail -1)"
phase "mid"
$VPY $A/selfmodel_train3b_20261125/t3b_mid.py --backend $BE --data3 $D3 --data3b $D3B --out $O/data > $O/logs/mid.log 2>&1 || die "mid"
phase "split check"
MID=$O/data/mid_data3.npz,$O/data/mid_data3b.npz
$VPY $A/selfmodel_train3b_20261125/t3b_split_check.py --data3 $D3 --data3b $D3B --mid $MID > $O/logs/split_check.log 2>&1 \
  || die "split check"
tail -n 1 $O/logs/split_check.log | tee -a $O/run.log
phase "C rows"
MX=1
$VPY $A/selfmodel_train3b_20261125/t3b_cref.py $D3 $D3B > $O/logs/cref.log 2>&1 || MX=0
tail -n 1 $O/logs/cref.log | tee -a $O/run.log
[ $MX = 1 ] || echo "$(date -u +%FT%TZ) $BE C rows failed the check: MX not trained" | tee -a $O/run.log
MS=$D3,$D3B
DS=$MS,$MID
PIDS=()
run() {  # name seed frac spec [env]
  local nm=$1 s=$2 f=$3 spec=$4 xe=${5:-REACHWM_X0_REF=0} g=${GPUS[$((NG % ${#GPUS[@]}))]}
  NG=$((NG + 1))
  [ -f $O/models_${nm}_s$s/E3_ensemble_f$f/model.pt ] && return 0
  local log=$O/logs/train_${nm}_s${s}_f$f.log
  env $xe CUDA_VISIBLE_DEVICES=$g nice -n 10 $VPY $A/selfmodel_train3b_20261125/t3b_train.py E3_ensemble --seed $s --frac $f --data $spec $IT \
    --out-root $O/models_${nm}_s$s > $log 2>&1 &
  local p=$!
  PIDS+=($p)
  echo "$(date -u +%FT%TZ) $BE train $nm seed $s frac $f on GPU $g ($xe): pid $p" >> $O/run.log
  until grep -q " segments, splits" $log || ! kill -0 $p 2>/dev/null; do sleep 5; done
  echo "$(date -u +%FT%TZ) $BE   loaded: $(grep " segments, splits" $log | cut -c1-60); free $(free -g | awk '/Mem/ {print $7}') GB" >> $O/run.log
}
phase "train"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> $O/run.log
free -g | head -2 >> $O/run.log
run M 0 1 $MS; run M 0 0.3 $MS; run M 0 0.6 $MS; run M 1 1 $MS; run M 2 1 $MS
run D 0 1 $DS; run D 1 1 $DS; run D 2 1 $DS
if [ $MX = 1 ]; then
  phase "train (MX waits for $T/MX_DECISION)"
  until [ -f $T/MX_DECISION ]; do sleep 30; done
  read -r dec xs < $T/MX_DECISION
  echo "$(date -u +%FT%TZ) $BE MX decision: $dec ${xs:-}" | tee -a $O/run.log
  if [ "$dec" = train ]; then
    XS=${xs:-$MS}
    run MX 0 1 $XS REACHWM_X0_REF=1; run MX 1 1 $XS REACHWM_X0_REF=1; run MX 2 1 $XS REACHWM_X0_REF=1
  else
    MX=0
  fi
  phase "train"
fi
for p in "${PIDS[@]}"; do wait $p || die "training pid $p"; done
grep -h "best val" $O/logs/train_*.log | tee -a $O/run.log
phase "eval"
EV="$VPY $A/selfmodel_train3b_20261125/t3b_eval.py --backend $BE --root $O --data3b $D3B"
for s in test_iid val edge; do
  MXA=""
  if [ $MX = 1 ]; then
    env REACHWM_X0_REF=1 CUDA_VISIBLE_DEVICES=${GPUS[0]} $EV --split $s --dump $O/eval_${s}_mx.npz > $O/logs/eval_${s}_mx.txt \
      2> $O/logs/eval_${s}_mx.err || die "eval $s MX dump"
    MXA="--mx $O/eval_${s}_mx.npz"
  fi
  CUDA_VISIBLE_DEVICES=${GPUS[0]} $EV --split $s $MXA --json $O/eval_$s.json > $O/eval_$s.txt 2> $O/logs/eval_$s.err || die "eval $s"
done
phase "done"
