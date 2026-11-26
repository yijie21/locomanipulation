#!/usr/bin/env bash
# notebook: skip the freeze of selfmodel_simplesonicdata_20261126's code (no run) -- on the Mac, after the registration
# commit:  ssd_freeze.sh
# ssd_copy.sh asrun/code, then $O/asrun/FROZEN_UTC, $O/asrun/GIT (the commit and the working tree's status the copy was
# made from), $O/asrun/KEY_MD5 (this entry's files and the files they run on) and $O/asrun/INPUTS_MD5 (the template, xfer's
# SONIC SIMPLE record, the SONIC policy, the main metric's SONIC A rows, train3b's frozen training scripts). Refuses when
# asrun/code exists.
set -euo pipefail
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplesonicdata_20261126
H="ssh -o ConnectTimeout=15 -p 16093 root@14.103.60.188"
$H "[ ! -e $O/asrun/code ]" || { echo "$O/asrun/code exists: frozen already"; exit 2; }
cd /Users/zed/code/loco
GITS="$(git log -1 --format='%H %s' | cut -c1-200)"
git status --short | $H "mkdir -p $O/asrun && cat > $O/asrun/GIT_STATUS"
bash "$(dirname "$0")/ssd_copy.sh" asrun/code
$H bash -s <<EOF
set -euo pipefail
cd $O/asrun/code
date -u +%FT%TZ > $O/asrun/FROZEN_UTC
echo "$GITS" > $O/asrun/GIT
X=scripts/experiments
md5sum \$X/selfmodel_simplesonicdata_20261126/* \$X/selfmodel_simplecollect_20261126/*.py \$X/selfmodel_simplecmp_20261126/scmp_correct.py \
  \$X/selfmodel_simplecmp_20261126/scmp_eval.py \$X/selfmodel_simplecmp_20261126/scmp_range.py \$X/selfmodel_simpleheft_20261126/sh_err.py \
  \$X/selfmodel_simplesonic_20261126/ss_f1pre.py \$X/simple_eval_20261112/simple_bench.py \$X/simple_eval_20261112/episodes.py \
  \$X/selfmodel_simpledata_20261126/*.py \$X/selfmodel_data3_20261125/d3_collect.py \$X/selfmodel_xfer_20261125/sx_collect.py \
  \$X/selfmodel_xfer_20261125/sx_eval.py \$X/reachwm_20261030/data.py \$X/reachwm_20261030/train.py \$X/reachwm_20261030/models.py \
  \$X/selfwm_20261029/env.py \$X/simple_relook_20261126/py4090b_rl.py src/locomani/backends/sonic_onnx/*.py \
  src/locomani/backends/scalebfm/motor.py src/locomani/backends/scalebfm/runtime.py src/locomani/bench/simple/__init__.py \
  configs/bench/simple.yaml > $O/asrun/KEY_MD5
L=/wuji-vefps-D/wuji-il/yijie/loco_outputs
WS=/wuji-vefps-D/wuji-il/yijie/ws
T3B=\$L/selfmodel_train3b_20261125/asrun/selfmodel_train3b_20261125
md5sum \$L/selfmodel_simpledata_20261126/real/task/data/all.npz \$L/selfmodel_xfer_20261125/sonic_simple/data/all.npz \
  \$(find \$WS/sonic_models/sonic_v1_1 -type f | sort) \$WS/code/loco/outputs/simple_eval_20261112/bench_rgbd_20261124/sonic/rows/A_*.jsonl \
  \$T3B/t3b_train.py \$T3B/t3b_mid.py > $O/asrun/INPUTS_MD5
echo "frozen \$(cat $O/asrun/FROZEN_UTC): \$(wc -l < CODE_MD5) files, \$(wc -l < $O/asrun/KEY_MD5) key files, \$(wc -l < $O/asrun/INPUTS_MD5) inputs"
EOF
