#!/usr/bin/env bash
# notebook: skip the freeze of selfmodel_simpleheft_20261126's code (no run) -- on the Mac, after the registration commit:
#   sh_freeze.sh
# sh_copy.sh asrun/code, then $O/asrun/FROZEN_UTC, $O/asrun/GIT (the commit and the working tree's status the copy was
# made from), $O/asrun/KEY_MD5 (this entry's files, simple_bench.py and the files they run on, the HEFT backend) and
# $O/asrun/INPUTS_MD5 (D_heft, F_heft, the template, D_heft's and F_heft's training data, the HEFT policy files, the main
# metric's ScaleBFM A rows the look is checked against). Refuses when asrun/code exists.
set -euo pipefail
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpleheft_20261126
H="ssh -p 27984 root@14.103.60.188"
$H "[ ! -e $O/asrun/code ]" || { echo "$O/asrun/code exists: frozen already"; exit 2; }
cd /Users/zed/code/loco
GITS="$(git log -1 --format='%H %s' | cut -c1-200)"
git status --short | $H "mkdir -p $O/asrun && cat > $O/asrun/GIT_STATUS"
bash "$(dirname "$0")/sh_copy.sh" asrun/code
$H bash -s <<EOF
set -euo pipefail
cd $O/asrun/code
date -u +%FT%TZ > $O/asrun/FROZEN_UTC
echo "$GITS" > $O/asrun/GIT
X=scripts/experiments
md5sum \$X/selfmodel_simpleheft_20261126/* \$X/selfmodel_simplecmp_20261126/scmp_correct.py \$X/selfmodel_simplecmp_20261126/scmp_eval.py \
  \$X/selfmodel_simplesonic_20261126/ss_f1pre.py \$X/simple_eval_20261112/simple_bench.py \$X/simple_eval_20261112/episodes.py \
  \$X/selfmodel_simpledata_20261126/sd_real.py \$X/selfmodel_data3_20261125/d3_collect.py \$X/selfmodel_simplecollect_20261126/sc_collect.py \
  \$X/selfmodel_xfer_20261125/sx_collect.py \$X/reachwm_20261030/data.py \$X/reachwm_20261030/train.py \$X/reachwm_20261030/models.py \
  \$X/selfwm_20261029/env.py src/locomani/backends/heft/*.py src/locomani/backends/scalebfm/motor.py \
  src/locomani/backends/scalebfm/runtime.py src/locomani/bench/simple/__init__.py configs/bench/simple.yaml > $O/asrun/KEY_MD5
L=/wuji-vefps-D/wuji-il/yijie/loco_outputs
MAIN=/wuji-vefps-D/wuji-il/yijie/ws/code/loco/outputs/simple_eval_20261112/bench_rgbd_20261124/scalebfm
md5sum \$L/selfmodel_train3b_20261125/heft/models_D_s0/E3_ensemble_f1/* \$L/selfmodel_fdvel_20261126/heft/models_s0/E3_ensemble_f1/* \
  \$L/selfmodel_simpledata_20261126/real/task/data/all.npz \$L/selfmodel_data3_20261125/heft/data/all.npz \
  \$L/selfmodel_data3b_20261125/heft/data/all.npz \$L/selfmodel_train3b_20261125/heft/data/mid_data3.npz \
  \$L/selfmodel_train3b_20261125/heft/data/mid_data3b.npz /workspace/models/heft/pmg/* \$MAIN/rows/A_*.jsonl > $O/asrun/INPUTS_MD5
echo "frozen \$(cat $O/asrun/FROZEN_UTC): \$(wc -l < CODE_MD5) files, \$(wc -l < $O/asrun/KEY_MD5) key files, \$(wc -l < $O/asrun/INPUTS_MD5) inputs"
EOF
