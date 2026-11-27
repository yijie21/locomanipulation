#!/usr/bin/env bash
# notebook: skip the freeze of selfmodel_simplecmp_20261126's code (no run) -- on the Mac, before the formal run:
#   scmp_freeze.sh
# scmp_copy.sh asrun/code (the Mac's working tree, others' uncommitted edits as they are, -> $O/asrun/code with its
# CODE_MD5), then $O/asrun/FROZEN_UTC and $O/asrun/KEY_MD5 (this entry's files, simple_bench.py and the files it runs on)
# and $O/asrun/INPUTS_MD5 (model D, the template, the main metric's A / J rows, via files and plans). Refuses when
# asrun/code exists (a freeze is made once; a new one is a revision of the entry).
set -euo pipefail
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126
H="ssh -p 27984 root@14.103.60.188"
$H "[ ! -e $O/asrun/code ]" || { echo "$O/asrun/code exists: frozen already"; exit 2; }
$H "mkdir -p $O/asrun"
bash "$(dirname "$0")/scmp_copy.sh" asrun/code
$H bash -s <<EOF
set -euo pipefail
cd $O/asrun/code
date -u +%FT%TZ > $O/asrun/FROZEN_UTC
md5sum scripts/experiments/selfmodel_simplecmp_20261126/* scripts/experiments/simple_eval_20261112/simple_bench.py \
  scripts/experiments/simple_eval_20261112/episodes.py scripts/experiments/selfmodel_simpledata_20261126/sd_real.py \
  scripts/experiments/selfmodel_simpledata_20261126/sd_range.py scripts/experiments/selfmodel_simplecollect_20261126/sc_collect.py \
  scripts/experiments/selfmodel_xfer_20261125/sx_collect.py scripts/experiments/selfmodel_data3_20261125/d3_collect.py \
  scripts/experiments/selfmodel_data3_20261125/d3_reach_table.json scripts/experiments/reachwm_20261030/data.py \
  scripts/experiments/reachwm_20261030/train.py scripts/experiments/reachwm_20261030/models.py \
  scripts/experiments/locomanilab_20261117/lml_selfmodel.py scripts/experiments/curobo_avoid_20261120/plan2.py \
  scripts/experiments/selfwm_20261029/env.py src/locomani/backends/scalebfm/motor.py src/locomani/backends/scalebfm/runtime.py \
  > $O/asrun/KEY_MD5
SD=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpledata_20261126
MAIN=/wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_eval_20261112/bench_rgbd_20261124/scalebfm
( md5sum \$SD/train/models_D_s0/E3_ensemble_f1/* \$SD/real/task/data/all.npz \$MAIN/rows/A_*.jsonl \$MAIN/rows/J_*.jsonl \$MAIN/via/*.npz
  for f in \$MAIN/via/*.npz; do n=\$(basename \$f .npz); n=\${n#via_}; for d in \$MAIN/step2 \$MAIN/try_act02; do [ -e \$d/plan_\$n.npz ] && md5sum \$d/plan_\$n.npz \$d/prep_\$n.npz; done; md5sum \$MAIN/step1/obst_\$n.npz; done ) > $O/asrun/INPUTS_MD5
echo "frozen \$(cat $O/asrun/FROZEN_UTC): \$(wc -l < CODE_MD5) files, \$(wc -l < $O/asrun/INPUTS_MD5) inputs"
EOF
