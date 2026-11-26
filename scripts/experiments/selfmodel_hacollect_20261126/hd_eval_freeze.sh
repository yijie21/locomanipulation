#!/usr/bin/env bash
# notebook: selfmodel_hadata_20261126
# selfmodel_hadata_20261126 revision 2b, the evaluation scripts' freeze (before any evaluation runs):
#   hd_eval_freeze.sh <dir holding the four scripts>
# -> $OH/eval_code/selfmodel_hacollect_20261126/{hd_eval.py, hd_xfer_pair.py, hd_real.py, hd_eval.sh}, read-only, with
# FROZEN_UTC and MD5 (md5sum -c from $OH/eval_code): the four scripts, and by absolute path every file they import, call
# or read as a fixed input: repo_b's reader / trainer / models / py4090b / d3_collect / reach table, the frozen sx_eval,
# sd_range, hr_smoke_stats and hd_touch, the LML denominators' jsons (train3b, fdvel), F's model.pt per controller and
# xfer's HA records (data/all.npz, run.json) per controller. hd_eval.sh checks this MD5 before every run.
set -euo pipefail
SRC=$1
L=/wuji-vefps-D/wuji-il/yijie/loco_outputs
OH=$L/selfmodel_hadata_20261126
EC=$OH/eval_code
R=$L/selfmodel_simpledata_20261126/asrun/repo_b
[ -e $EC ] && { echo "$EC exists: not overwritten"; exit 1; }
mkdir -p $EC/selfmodel_hacollect_20261126
cp $SRC/{hd_eval.py,hd_xfer_pair.py,hd_real.py,hd_eval.sh,hd_eval_freeze.sh} $EC/selfmodel_hacollect_20261126/
EXT=(
  $R/scripts/experiments/reachwm_20261030/data.py $R/scripts/experiments/reachwm_20261030/train.py
  $R/scripts/experiments/reachwm_20261030/models.py $R/scripts/experiments/selfmodel_simplecollect_20261126/py4090b.py
  $R/scripts/experiments/selfmodel_data3_20261125/d3_collect.py $R/scripts/experiments/selfmodel_data3_20261125/d3_reach_table.json
  $L/selfmodel_xfer_20261125/asrun_ha/selfmodel_xfer_20261125/sx_eval.py
  $L/selfmodel_simpledata_20261126/asrun/repo/scripts/experiments/selfmodel_simpledata_20261126/sd_range.py
  $OH/asrun_real/selfmodel_hacollect_20261126/hr_smoke_stats.py
  $OH/asrun_train/selfmodel_hacollect_20261126/hd_touch.py
)
for c in scalebfm heft sonic; do
  EXT+=($L/selfmodel_train3b_20261125/$c/final_val.json $L/selfmodel_fdvel_20261126/$c/eval_lml_val.json
        $L/selfmodel_fdvel_20261126/$c/models_s0/E3_ensemble_f1/model.pt
        $L/selfmodel_xfer_20261125/${c}_ha/data/all.npz $L/selfmodel_xfer_20261125/${c}_ha/run.json)
done
cd $EC
md5sum selfmodel_hacollect_20261126/* "${EXT[@]}" > MD5
date -u +%FT%TZ > FROZEN_UTC
chmod -R a-w $EC
md5sum -c --quiet MD5 && echo "eval_code frozen $(cat FROZEN_UTC): $(wc -l < MD5) files, MD5 $(md5sum MD5 | cut -c1-8)"
