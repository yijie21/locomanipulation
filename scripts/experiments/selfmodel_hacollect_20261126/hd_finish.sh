#!/usr/bin/env bash
# notebook: selfmodel_hadata_20261126
# One finished collection: sx_collect's merge / check (d3's checks and XFER CHECK) / summary and hc_rates, from the frozen
# copy, CPU. The reader the check loads the data with is the training one: fdvel's frozen FDVEL data.py (174071cb, pinned
# in MD5.check) with the training flags. Writes <out>/<ctrl>/<mode>/{merge,check,summary,rates}.txt, each step's exit
# code to finish_status, and FINISHED (FINISHING while it runs).
#   hd_finish.sh <asrun> <out root> <controller> <d3|d3b> <resets> <id0>
set -uo pipefail
A=$1 OUT=$2 C=$3 MODE=$4 N=$5 I0=$6
X=$OUT/$C/$MODE
VPY=$A/repo/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
H=$A/selfmodel_xfer_20261125
FD=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_fdvel_20261126/asrun/repo/scripts/experiments/reachwm_20261030
(cd $A && md5sum -c --quiet MD5.check) || { echo "the frozen copy or a pinned install changed"; exit 2; }
unset SX_D3
export SX_REPO=$A/repo D3_REACHWM=$FD OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
export REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_FDVEL=1
cd $A/repo
: > $X/finish_status
step() {  # name, command...
  local nm=$1; shift
  nice -n 10 "$@" > $X/$nm.txt 2>&1
  echo "$nm $?" >> $X/finish_status
}
step merge $VPY $H/sx_collect.py merge --out $X
step check $VPY $H/sx_collect.py check --out $X --resets $N --id0 $I0
step summary $VPY $H/sx_collect.py summary --out $X
step rates $VPY $A/selfmodel_hacollect_20261126/hc_rates.py $X
date -u +%FT%TZ > $X/FINISHED
rm -f $X/FINISHING
