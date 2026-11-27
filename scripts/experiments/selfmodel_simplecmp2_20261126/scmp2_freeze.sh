#!/usr/bin/env bash
# notebook: skip the freeze of selfmodel_simplecmp2_20261126's code (no run) -- on the Mac, after the registration commit:
#   scmp2_freeze.sh
# scmp2_copy.sh asrun/code, then $O/asrun/FROZEN_UTC, $O/asrun/KEY_MD5 (this entry's files, simple_bench.py and the files
# they run on), V1's duration files ($O/asrun/jdur_J0S.json: J_D's plans and recorded durations, jdur_JDF.json: J0's;
# scmp2_jdur_make.py) and $O/asrun/INPUTS_MD5 (model D, the template, the main metric's A rows, selfmodel_simplecmp's
# RD / J0 / JD rows and plans, the duration files). Refuses when asrun/code exists.
set -euo pipefail
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp2_20261126
O1=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126
H="ssh -p 27984 root@14.103.60.188"
$H "[ ! -e $O/asrun/code ]" || { echo "$O/asrun/code exists: frozen already"; exit 2; }
bash "$(dirname "$0")/scmp2_copy.sh" asrun/code
$H bash -s <<EOF
set -euo pipefail
cd $O/asrun/code
date -u +%FT%TZ > $O/asrun/FROZEN_UTC
md5sum scripts/experiments/selfmodel_simplecmp2_20261126/* scripts/experiments/selfmodel_simplecmp_20261126/* \
  scripts/experiments/simple_eval_20261112/simple_bench.py scripts/experiments/simple_eval_20261112/episodes.py \
  scripts/experiments/selfmodel_simpledata_20261126/sd_real.py scripts/experiments/selfmodel_simpledata_20261126/sd_range.py \
  scripts/experiments/selfmodel_data3_20261125/d3_collect.py scripts/experiments/reachwm_20261030/data.py \
  scripts/experiments/reachwm_20261030/train.py scripts/experiments/reachwm_20261030/models.py \
  src/locomani/backends/scalebfm/motor.py src/locomani/backends/scalebfm/runtime.py > $O/asrun/KEY_MD5
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
\$PY scripts/experiments/selfmodel_simplecmp2_20261126/scmp2_jdur_make.py $O1/run/P/JD/via $O1/run/rows/JD_TabletopGraspMP.jsonl $O/asrun/jdur_J0S.json
\$PY scripts/experiments/selfmodel_simplecmp2_20261126/scmp2_jdur_make.py $O1/run/P/J0/via $O1/run/rows/J0_TabletopGraspMP.jsonl $O/asrun/jdur_JDF.json
SD=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpledata_20261126
MAIN=/wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_eval_20261112/bench_rgbd_20261124/scalebfm
( md5sum \$SD/train/models_D_s0/E3_ensemble_f1/* \$SD/real/task/data/all.npz \$MAIN/rows/A_*.jsonl $O1/run/rows/RD_*.jsonl \
    $O1/run/rows/J0_*.jsonl $O1/run/rows/JD_*.jsonl $O/asrun/jdur_*.json
  find $O1/run/P/J0 $O1/run/P/JD -type f | sort | xargs md5sum ) > $O/asrun/INPUTS_MD5
echo "frozen \$(cat $O/asrun/FROZEN_UTC): \$(wc -l < CODE_MD5) files, \$(wc -l < $O/asrun/INPUTS_MD5) inputs"
EOF
