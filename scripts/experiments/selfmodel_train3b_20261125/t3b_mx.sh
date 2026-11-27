#!/usr/bin/env bash
# notebook: selfmodel_train3b_20261125
# selfmodel_train3b_20261125 revisions 2-3: MX and MXa of one controller, next to t3b_run.sh (tmux t3b_mx_<backend>):
#   t3b_mx.sh <backend> <GPUs for MX seed 0, 1, 2 and MXa, comma-separated>
# Waits for selfmodel_data3b_20261125's formal collection of <backend> (merged, CHECK PASSED), then
#   MX's data (t3b_mxdata.py: C[k0, :3] = this reach's reference start; every check must pass, else no MX / MXa)
#   -> MX seeds 0, 1, 2 on MX's data and MXa seed 0 on M's data (the stored C[k0]), both REACHWM_X0_REF=1, train3's
#   recipe (t3b_train.py E3_ensemble, nice 10, 4 CPU threads, one load at a time)
#   -> waits for t3b_run.sh's M / D trainings (their model.pt) -> the final table (t3b_eval.py from this freeze) on
#   test_iid, val and edge: T3, M, M_f0.3, M_f0.6, D, MX, MXa (MX / MXa from X0_REF=1 dumps).
# Code: the frozen copy in $A (asrun_mx, t3b_freeze_mx.sh). Phases in $O/mx_run.log and $O/MX_STATUS.
set -uo pipefail
BE=$1; IFS=, read -ra GPUS <<< "$2"; NG=0
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
R=/wuji-vefps-D/wuji-il/yijie/loco_outputs
T=$R/selfmodel_train3b_20261125
O=${T3B_OUT:-$T/$BE}
A=$T/asrun_mx
D3=$R/selfmodel_data3_20261125/$BE/data/all.npz
B=${T3B_B:-$R/selfmodel_data3b_20261125/$BE}
IT=${T3B_ITERS:+--iters $T3B_ITERS}
D3B=$B/data/all.npz
cd $W || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 LOCO_REPO=$W
VPY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
C=$A/selfmodel_train3b_20261125
mkdir -p $O/mxdata $O/logs
phase() { echo "$1" > $O/MX_STATUS; echo "$(date -u +%FT%TZ) $BE $1" | tee -a $O/mx_run.log; }
die() { phase "FAIL $1"; exit 1; }
(cd $A && md5sum -c --quiet MD5) || die "asrun_mx md5"
for f in reachwm_20261030/{data,train,models}.py selfwm_20261029/{env,wm,fk}.py; do
  cmp -s scripts/experiments/$f $A/$f || die "repo $f differs from asrun_mx"
done
phase "code: asrun_mx md5 ok; GPUs ${GPUS[*]} in turn on $(hostname)"
phase "wait for data3b $BE"
until grep -q merged $B/formal.log 2>/dev/null; do sleep 60; done
grep -q "CHECK PASSED" $B/check.log || die "data3b $BE check did not pass"
phase "MX data"
env REACHWM_X0_REF=1 $VPY $C/t3b_mxdata.py --backend $BE --data3 $(dirname $(dirname $D3)) --data3b $B --out $O/mxdata \
  > $O/logs/mxdata.log 2>&1 || die "MX data checks (logs/mxdata.log): no MX / MXa"
tail -n 1 $O/logs/mxdata.log | tee -a $O/mx_run.log
MXS=$O/mxdata/mx_data3.npz,$O/mxdata/mx_data3b.npz
MS=$D3,$D3B
PIDS=()
run() {  # name seed spec
  local nm=$1 s=$2 spec=$3 g=${GPUS[$((NG % ${#GPUS[@]}))]}
  NG=$((NG + 1))
  [ -f $O/models_${nm}_s$s/E3_ensemble_f1/model.pt ] && return 0
  local log=$O/logs/train_${nm}_s${s}_f1.log
  env REACHWM_X0_REF=1 CUDA_VISIBLE_DEVICES=$g nice -n 10 $VPY $C/t3b_train.py E3_ensemble --seed $s --frac 1 --data $spec $IT \
    --out-root $O/models_${nm}_s$s > $log 2>&1 &
  local p=$!
  PIDS+=($p)
  echo "$(date -u +%FT%TZ) $BE train $nm seed $s on GPU $g (REACHWM_X0_REF=1): pid $p" >> $O/mx_run.log
  until grep -q " segments, splits" $log || ! kill -0 $p 2>/dev/null; do sleep 5; done
  echo "$(date -u +%FT%TZ) $BE   loaded: $(grep " segments, splits" $log | cut -c1-60); free $(free -g | awk '/Mem/ {print $7}') GB" >> $O/mx_run.log
}
phase "train"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> $O/mx_run.log
run MX 0 $MXS; run MX 1 $MXS; run MX 2 $MXS; run MXa 0 $MS
for p in "${PIDS[@]}"; do wait $p || die "training pid $p"; done
grep -h "best val" $O/logs/train_MX*.log | tee -a $O/mx_run.log
phase "wait for M / D"
for m in M_s0/E3_ensemble_f1 M_s0/E3_ensemble_f0.3 M_s0/E3_ensemble_f0.6 M_s1/E3_ensemble_f1 M_s2/E3_ensemble_f1 \
         D_s0/E3_ensemble_f1 D_s1/E3_ensemble_f1 D_s2/E3_ensemble_f1; do
  until [ -f $O/models_$m/model.pt ]; do grep -q FAIL $O/STATUS 2>/dev/null && die "t3b_run.sh failed"; sleep 30; done
done
phase "eval"
EV="$VPY $C/t3b_eval.py --backend $BE --root $O"
for s in test_iid val edge; do
  env REACHWM_X0_REF=1 CUDA_VISIBLE_DEVICES=${GPUS[0]} $EV --split $s --data3 $O/mxdata/mx_data3.npz --data3b $O/mxdata/mx_data3b.npz \
    --dump $O/final_${s}_mx.npz --arm MX > $O/logs/final_${s}_mx.txt 2>&1 || die "eval $s MX dump"
  env REACHWM_X0_REF=1 CUDA_VISIBLE_DEVICES=${GPUS[0]} $EV --split $s --data3b $D3B --dump $O/final_${s}_mxa.npz --arm MXa \
    > $O/logs/final_${s}_mxa.txt 2>&1 || die "eval $s MXa dump"
  CUDA_VISIBLE_DEVICES=${GPUS[0]} $EV --split $s --data3b $D3B --mx $O/final_${s}_mx.npz,$O/final_${s}_mxa.npz \
    --json $O/final_$s.json > $O/final_$s.txt 2> $O/logs/final_$s.err || die "eval $s"
done
phase "done"
