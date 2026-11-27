#!/usr/bin/env bash
# selfmodel_fdvel_20261126: one run of xfer's frozen sx_eval.py (fd_run.sh's xfer phase; cwd $A/repo).
#   fd_sx1.sh <record> <T3|F_s0|F_s1|F_s2> <settled|ended>
# T3: train3's model (selfmodel_data3_20261125/<backend>/models_s0), x0 as recorded (REACHWM_FDVEL=0); F_s<k>: this
# experiment's <out>/<backend>/models_s<k>, REACHWM_FDVEL=1 (LocoManiLab val and the xfer record both differenced).
# Out: <out>/xfer/<record>/<tag>_<keep>.{txt,json} (.raw: unfiltered output).
set -uo pipefail
REC=$1 TAG=$2 KEEP=$3 BE=${1%%_*}
: "${FD_O:?} ${FD_A:?} ${FD_D3:?} ${FD_XF:?}"
if [ "$TAG" = T3 ]; then MD=$FD_D3/$BE/models_s0; FD=0; else MD=$FD_O/$BE/models_${TAG#F_}; FD=1; fi
mkdir -p $FD_O/xfer/$REC
OUT=$FD_O/xfer/$REC/${TAG}_$KEEP
env REACHWM_FDVEL=$FD SX_REPO=$FD_A/repo CUDA_VISIBLE_DEVICES= nice -n 10 $FD_A/repo/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
  scripts/experiments/selfmodel_xfer_20261125/sx_eval.py --lml $FD_D3/$BE --xfer $FD_XF/$REC --models $MD --keep $KEEP \
  --json $OUT.json > $OUT.raw 2>&1
rc=$?
grep -v Warn $OUT.raw > $OUT.txt
echo "$(date -u +%FT%TZ) sx_eval $REC $TAG $KEEP rc $rc" >> $FD_O/xfer/done.log
exit $rc
