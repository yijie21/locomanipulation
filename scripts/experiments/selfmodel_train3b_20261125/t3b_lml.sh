#!/usr/bin/env bash
# notebook: selfmodel_train3b_20261125
# selfmodel_train3b_20261125, lmlreal revision 1: the train3b models on selfmodel_lmlreal_20261125's episodes of one
# controller (tmux t3b_lml_<backend>):   t3b_lml.sh <backend> <GPU>
# Waits for lmlreal's merge of <backend> ("merged and checked ... CHECK PASSED" in its formal.log), builds MX's input
# (t3b_lmlmx.py: chained reaches only; every check must pass, else no MX column), waits for this controller's 12 train3b
# models, then t3b_lmleval.py: MX dump (X0_REF=1, NaN on chain 0), MXa dump (X0_REF=1), the main table.
# Code: the frozen copy in $A (asrun_lml, t3b_freeze_lml.sh). Phases in $O/lml_run.log and $O/LML_STATUS.
set -uo pipefail
BE=$1; G=$2
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
R=/wuji-vefps-D/wuji-il/yijie/loco_outputs
T=$R/selfmodel_train3b_20261125
O=${T3B_OUT:-$T/$BE}
A=$T/asrun_lml
LR=$R/selfmodel_lmlreal_20261125
REAL=${T3B_REAL:-$LR/$BE/data/all.npz}
cd $W || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 LOCO_REPO=$W CUDA_VISIBLE_DEVICES=$G
VPY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
C=$A/selfmodel_train3b_20261125
mkdir -p $O/lmlmx $O/logs
phase() { echo "$1" > $O/LML_STATUS; echo "$(date -u +%FT%TZ) $BE $1" | tee -a $O/lml_run.log; }
die() { phase "FAIL $1"; exit 1; }
(cd $A && md5sum -c --quiet MD5) || die "asrun_lml md5"
for f in reachwm_20261030/{data,train,models}.py selfwm_20261029/{env,wm,fk}.py; do
  cmp -s scripts/experiments/$f $A/$f || die "repo $f differs from asrun_lml"
done
phase "code: asrun_lml md5 ok; GPU $G on $(hostname); real $REAL"
phase "wait for lmlreal $BE"
if [ -z "${T3B_REAL:-}" ]; then
  until grep -q "^$BE merged and checked" $LR/formal.log 2>/dev/null; do sleep 60; done
  grep "^$BE merged and checked" $LR/formal.log | tail -1 | grep -q "CHECK PASSED" || die "lmlreal $BE check did not pass"
fi
phase "LML MX data"
MXOK=1
env REACHWM_X0_REF=1 nice -n 10 $VPY $C/t3b_lmlmx.py --backend $BE --real $REAL --out $O/lmlmx > $O/logs/lmlmx.log 2>&1 || MXOK=0
tail -n 1 $O/logs/lmlmx.log | tee -a $O/lml_run.log
[ $MXOK = 1 ] || echo "$(date -u +%FT%TZ) $BE MX column left out (logs/lmlmx.log)" | tee -a $O/lml_run.log
phase "wait for models"
for m in M_s0/E3_ensemble_f1 M_s0/E3_ensemble_f0.3 M_s0/E3_ensemble_f0.6 M_s1/E3_ensemble_f1 M_s2/E3_ensemble_f1 \
         D_s0/E3_ensemble_f1 D_s1/E3_ensemble_f1 D_s2/E3_ensemble_f1 MX_s0/E3_ensemble_f1 MX_s1/E3_ensemble_f1 \
         MX_s2/E3_ensemble_f1 MXa_s0/E3_ensemble_f1; do
  until [ -f $O/models_$m/model.pt ]; do
    grep -q FAIL $O/STATUS 2>/dev/null && die "t3b_run.sh failed"
    case "$m" in MX*) grep -q FAIL $O/MX_STATUS 2>/dev/null && die "t3b_mx.sh failed";; esac
    sleep 30
  done
done
phase "eval"
EV="nice -n 10 $VPY $C/t3b_lmleval.py --backend $BE --root $O"
MXS=""
if [ $MXOK = 1 ]; then
  env REACHWM_X0_REF=1 $EV --real $O/lmlmx/mx_lmlreal.npz --dump $O/lml_mx.npz --arm MX > $O/logs/lml_mx.txt 2>&1 || die "eval MX dump"
  MXS=$O/lml_mx.npz,
fi
env REACHWM_X0_REF=1 $EV --real $REAL --dump $O/lml_mxa.npz --arm MXa > $O/logs/lml_mxa.txt 2>&1 || die "eval MXa dump"
$EV --real $REAL --mx ${MXS}$O/lml_mxa.npz --json $O/lml_real.json > $O/lml_real.txt 2> $O/logs/lml_real.err || die "eval"
phase "done"
