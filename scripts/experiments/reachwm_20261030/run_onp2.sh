#!/bin/bash
# Round 2 of on-policy data, at the stances the wrist-aware stance search picks (2026-11-03): collect, retrain on the
# turn model's 7 datasets + this one, then the 345 core goals eyes closed, staying (wrist_invert.py) and with the stance
# search (stance_wrist.py), one after the other on 6 processes. Logs in outputs/reachwm_20261030/logs/*_onp2.log.
#   nohup setsid bash scripts/experiments/reachwm_20261030/run_onp2.sh > outputs/reachwm_20261030/logs/run_onp2.log 2>&1 < /dev/null &
set -euo pipefail
cd /workspace/code/locomanipulation-selfwm
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
X=scripts/experiments/reachwm_20261030
O=$PWD/outputs/reachwm_20261030
export REACHWM_GOAL_REF=1 OMP_NUM_THREADS=1
N=${SONP_N:-480}

echo "== collect $N goals $(date +%T)"
$PY $X/stance_onpolicy.py h --procs 6 --n "$N" > $O/logs/collect_onp2.log 2>&1
$PY $X/stance_onpolicy.py merge

echo "== train $(date +%T)"
D=ref,side,wrist,wrist2,onp1,approach_reach,approach0_reach,onp2
SPEC=$(echo $D | tr , '\n' | sed "s|.*|$O/data_&/all.npz|" | paste -sd,)
CUDA_VISIBLE_DEVICES=1 $PY $X/train.py E3_ensemble --data "$SPEC" --out-root $O/models_goalref_onp2 > $O/logs/train_onp2.log 2>&1
tail -1 $O/logs/train_onp2.log
WRIST_OUT=$O/invert/wrist_onp2 WRIST_MODEL_ROOT=models_goalref_onp2 WRIST_DATA=$D $PY $X/wrist_invert.py x

echo "== evaluate $(date +%T)"
WRIST_OUT=$O/invert/wrist_onp2 WRIST_SERVOS=open WRIST_METHODS=pos,pose3g,w20 \
  $PY $X/wrist_invert.py h --procs 6 > $O/logs/wrist_onp2.log 2>&1
STANCE_WM=$O/invert/wrist_onp2/wm_ref.pt STANCE_OUT=$O/invert/stance_wrist_onp2 \
  $PY $X/stance_wrist.py h --procs 6 > $O/logs/stance_wrist_onp2.log 2>&1
cat $O/logs/wrist_onp2.log $O/logs/stance_wrist_onp2.log
$PY $X/stance_report.py > /dev/null
echo "== done $(date +%T)"
