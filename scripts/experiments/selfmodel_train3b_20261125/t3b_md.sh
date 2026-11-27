#!/usr/bin/env bash
# notebook: selfmodel_train3b_20261125
# selfmodel_train3b_20261125: revision 0's chained ratio (model / direct-execution position medians) with paired
# intervals, one controller (tmux t3b_md_<backend>):   t3b_md.sh <backend> <GPU>
# Waits until SONIC's 12 trainings are done (all three controllers run together after that) and this controller's final
# table is written (MX_STATUS done), then t3b_mdirect.py on test_iid and val. Code: the frozen copy in $A (asrun_md,
# t3b_freeze_md.sh). Phases in $O/md_run.log and $O/MD_STATUS.
set -uo pipefail
BE=$1; G=$2
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
R=/wuji-vefps-D/wuji-il/yijie/loco_outputs
T=$R/selfmodel_train3b_20261125
O=${T3B_OUT:-$T/$BE}
A=$T/asrun_md
D3=$R/selfmodel_data3_20261125/$BE/data/all.npz
D3B=${T3B_B:-$R/selfmodel_data3b_20261125/$BE}/data/all.npz
cd $W || exit 1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 LOCO_REPO=$W CUDA_VISIBLE_DEVICES=$G
VPY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
phase() { echo "$1" > $O/MD_STATUS; echo "$(date -u +%FT%TZ) $BE $1" | tee -a $O/md_run.log; }
die() { phase "FAIL $1"; exit 1; }
(cd $A && md5sum -c --quiet MD5) || die "asrun_md md5"
for f in reachwm_20261030/{data,train,models}.py selfwm_20261029/{env,wm,fk}.py; do
  cmp -s scripts/experiments/$f $T/asrun_mx/$f || die "repo $f differs from asrun_mx"
done
phase "code: asrun_md md5 ok; GPU $G on $(hostname)"
if [ -z "${T3B_NO_SONIC_WAIT:-}" ]; then
  phase "wait for SONIC's trainings"
  S=$T/sonic
  for m in M_s0/E3_ensemble_f1 M_s0/E3_ensemble_f0.3 M_s0/E3_ensemble_f0.6 M_s1/E3_ensemble_f1 M_s2/E3_ensemble_f1 \
           D_s0/E3_ensemble_f1 D_s1/E3_ensemble_f1 D_s2/E3_ensemble_f1 MX_s0/E3_ensemble_f1 MX_s1/E3_ensemble_f1 \
           MX_s2/E3_ensemble_f1 MXa_s0/E3_ensemble_f1; do
    until [ -f $S/models_$m/model.pt ]; do
      grep -q FAIL $S/STATUS $S/MX_STATUS 2>/dev/null && die "SONIC's training failed"
      sleep 30
    done
  done
fi
phase "wait for the final table"
until [ "$(cat $O/MX_STATUS 2>/dev/null)" = done ]; do grep -q FAIL $O/MX_STATUS 2>/dev/null && die "t3b_mx.sh failed"; sleep 30; done
phase "run"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> $O/md_run.log
for s in test_iid val; do
  nice -n 10 $VPY $A/selfmodel_train3b_20261125/t3b_mdirect.py --backend $BE --split $s --root $O --data3 $D3 --data3b $D3B \
    --final $O/final_$s.json --json $O/md_$s.json > $O/md_$s.txt 2> $O/logs/md_$s.err || die "t3b_mdirect $s (md_$s.txt)"
  tail -n 1 $O/md_$s.txt | tee -a $O/md_run.log
done
phase "done"
