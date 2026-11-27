#!/usr/bin/env bash
# selfmodel_train3_20261125: train the first-generation end model (reachwm_20261030 train.py E3_ensemble, seed 0, its
# recipe unchanged) on one controller's selfmodel_data3 one-hand segments (data/all.npz; the two-hand ones in
# data_both/ are not trained), REACHWM_* as hostdata.
#   d3_train.sh <backend> <gpu> <frac>...     (fractions as train.py names them: 0.1 0.3 1)
# The fractions run in parallel on that card; once the three models exist, the card that finished last runs this
# directory's d3_eval.py on val and test_iid. Run the asrun copy (d3_train_freeze below), from any directory.
# Models: $O/models_s0/E3_ensemble_f<frac>; logs: $O/logs/train_f<frac>.log; reports: $O/eval_<split>.{txt,json}.
#   d3_train.sh freeze <backend>...            copy d3_train.sh, d3_eval.py and the reachwm / selfwm code it runs into
#                                              $O/asrun/train/ and write $O/asrun/train/MD5
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
E=$W/scripts/experiments
if [ "$1" = freeze ]; then
  shift
  for BE in "$@"; do
    A=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/$BE/asrun/train
    mkdir -p $A
    cp $E/selfmodel_data3_20261125/{d3_train.sh,d3_eval.py} $A/
    cp $E/reachwm_20261030/{train.py,data.py,models.py} $E/selfwm_20261029/env.py $A/
    (cd $A && md5sum *.py *.sh) | tee $A/MD5
    (cd $W && git rev-parse HEAD && git status --short scripts/experiments/reachwm_20261030 scripts/experiments/selfwm_20261029) >> $A/MD5
  done
  exit 0
fi
BE=$1; GPU=$2; shift 2
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_data3_20261125/$BE
HERE=$(cd "$(dirname "$0")" && pwd)
cd $W || exit 1
VPY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
export OMP_NUM_THREADS=4 REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 REACHWM_QFIX=1 REACHWM_ROOT=$O CUDA_VISIBLE_DEVICES=$GPU
grep -q "CHECK PASSED" $O/check.log || { echo "$BE: check has not passed, not training"; exit 1; }
# the code that runs must be the frozen one
for f in train.py data.py models.py; do
  cmp -s $HERE/$f $E/reachwm_20261030/$f || { echo "$BE: $f differs from asrun"; exit 1; }
done
cmp -s $HERE/env.py $E/selfwm_20261029/env.py || { echo "$BE: env.py differs from asrun"; exit 1; }
mkdir -p $O/logs $O/models_s0
PIDS=()
for f in "$@"; do
  [ -f $O/models_s0/E3_ensemble_f$f/model.pt ] && continue
  $VPY $E/reachwm_20261030/train.py E3_ensemble --seed 0 --frac $f --data $O/data/all.npz \
    --out-root $O/models_s0 > $O/logs/train_f$f.log 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait $p; done
echo "$(date -u +%FT%TZ) $BE trained $* on GPU $GPU" >> $O/train.log
n=$(ls $O/models_s0/E3_ensemble_f*/model.pt 2>/dev/null | wc -l)
if [ "$n" -ge 3 ] && mkdir $O/.eval_lock 2>/dev/null; then  # the last card to finish writes the reports
  for s in val test_iid; do
    $VPY $HERE/d3_eval.py $O $O/models_s0 --split $s --json $O/eval_$s.json 2>&1 | grep -v Warn > $O/eval_$s.txt
  done
  echo "$(date -u +%FT%TZ) $BE evaluated" >> $O/train.log
fi
