#!/usr/bin/env bash
# notebook: skip code copy (no run) -- selfmodel_simplecmp_20261126's private code tree on the 4090, from the Mac:
#   scmp_copy.sh [code dir name, default code]
# Run on the Mac. Copies the Mac checkout's src/, scripts/, configs/, pyproject.toml (the working tree: others'
# uncommitted edits included, as they are) to $O/<name>/ on the 4090 and links what the code only reads from the shared
# tree (third_party, assets, grasp_kit, rsi_kit; outputs/* one by one). Private SIMPLE_EVAL_OUT trees $O/evalout (the test
# episodes) and $O/devout (the development episodes): every entry of the shared tree linked, but scene/ (EP.build writes
# the episode scenes there) a real directory of our own. Writes $O/<name>/CODE_MD5 (every copied file). Nothing in the
# shared trees is written. The pre-edit simple_bench.py (the Mac's before this entry's --correct) goes beside it as
# simple_bench_before.py when $BEFORE names it.
set -euo pipefail
NAME=${1:-code}
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
H="ssh -p 27984 root@14.103.60.188"
cd /Users/zed/code/loco
$H "mkdir -p $O/$NAME"
rsync -a --delete -e "ssh -p 27984" --exclude __pycache__ --exclude .DS_Store --exclude '._*' \
  src scripts configs pyproject.toml root@14.103.60.188:$O/$NAME/
if [ -n "${BEFORE:-}" ]; then
  rsync -a -e "ssh -p 27984" "$BEFORE" root@14.103.60.188:$O/$NAME/scripts/experiments/simple_eval_20261112/simple_bench_before.py
fi
$H bash -s <<EOF
set -euo pipefail
cd $O/$NAME
for x in third_party assets grasp_kit rsi_kit; do ln -sfn $W/\$x \$x; done
mkdir -p outputs
for x in $W/outputs/*; do ln -sfn \$x outputs/\$(basename \$x); done
for pair in evalout:$W/outputs/simple_eval_20261112 devout:$W/outputs/simple_dev_20261124; do
  d=$O/\${pair%%:*} s=\${pair#*:}
  mkdir -p \$d/scene
  for x in \$s/*; do b=\$(basename \$x); [ "\$b" = scene ] || ln -sfn \$x \$d/\$b; done
done
find src scripts configs pyproject.toml -type f | sort | xargs md5sum > CODE_MD5
echo "copied: \$(wc -l < CODE_MD5) files; simple_bench \$(md5sum < scripts/experiments/simple_eval_20261112/simple_bench.py | cut -c1-8)"
EOF
