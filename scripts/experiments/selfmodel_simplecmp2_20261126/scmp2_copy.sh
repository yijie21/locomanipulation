#!/usr/bin/env bash
# notebook: skip code copy (no run) -- selfmodel_simplecmp2_20261126's private code tree on the 4090:
#   scmp2_copy.sh <code dir name under $O, e.g. code_dev or asrun/code>
# Run on the Mac. The tree is selfmodel_simplecmp_20261126's frozen copy ($O1/asrun/code, the code its rows came from)
# with two things from the Mac on top: simple_bench.py (its --joint-dur, off by default) and this entry's directory. Nothing
# else of the Mac's working tree is taken (other experiments' files there changed after that freeze; none is on this
# entry's path). Private SIMPLE_EVAL_OUT trees $O/evalout (test episodes) and $O/devout (development episodes): every entry
# of the shared tree linked, scene/ our own. Writes <name>/CODE_MD5 and prints the files that differ from $O1's copy.
set -euo pipefail
NAME=$1
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp2_20261126
O1=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplecmp_20261126
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
H="ssh -p 27984 root@14.103.60.188"
cd /Users/zed/code/loco
$H "set -e; [ ! -e $O/$NAME ] || { echo '$O/$NAME exists'; exit 2; }; mkdir -p \$(dirname $O/$NAME); cp -a $O1/asrun/code $O/$NAME; rm -f $O/$NAME/CODE_MD5"
rsync -a -e "ssh -p 27984" scripts/experiments/simple_eval_20261112/simple_bench.py root@14.103.60.188:$O/$NAME/scripts/experiments/simple_eval_20261112/simple_bench.py
rsync -a -e "ssh -p 27984" --exclude __pycache__ --exclude .DS_Store --exclude '._*' scripts/experiments/selfmodel_simplecmp2_20261126/ \
  root@14.103.60.188:$O/$NAME/scripts/experiments/selfmodel_simplecmp2_20261126/
$H bash -s <<EOF
set -euo pipefail
for pair in evalout:$W/outputs/simple_eval_20261112 devout:$W/outputs/simple_dev_20261124; do
  d=$O/\${pair%%:*} s=\${pair#*:}
  [ -d \$d ] && continue
  mkdir -p \$d/scene
  for x in \$s/*; do b=\$(basename \$x); [ "\$b" = scene ] || ln -sfn \$x \$d/\$b; done
done
cd $O/$NAME
find src scripts configs pyproject.toml -type f -not -path '*/__pycache__/*' | sort | xargs md5sum > CODE_MD5
echo "copied: \$(wc -l < CODE_MD5) files; simple_bench \$(md5sum < scripts/experiments/simple_eval_20261112/simple_bench.py | cut -c1-8)"
echo "differs from simplecmp's frozen copy:"
diff <(sort -k2 $O1/asrun/code/CODE_MD5) <(sort -k2 CODE_MD5) | grep '^[<>]' || true
EOF
