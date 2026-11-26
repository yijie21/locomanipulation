#!/usr/bin/env bash
# notebook: skip code copy (no run) -- selfmodel_simpleheft_20261126's private code tree on the 4090, from the Mac:
#   sh_copy.sh [code dir name, default code]
# Run on the Mac. selfmodel_simplecmp_20261126's scmp_copy.sh with this entry's output dir, the episode trees inside the
# copy and without $BEFORE: copies the Mac checkout's src/, scripts/, configs/, pyproject.toml (the working tree: others'
# uncommitted edits included, as they are; __pycache__, .DS_Store and ._* left out) to $O/<name>/ on the 4090 and links
# what the code only reads from the shared tree (third_party, assets, grasp_kit, rsi_kit; outputs/* one by one) -- but
# outputs/simple_eval_20261112 (the test episodes) and outputs/simple_dev_20261124 (the development episodes), the
# SIMPLE_EVAL_OUT trees sh_run.sh uses, are real directories of the copy (coordinator 11-26: never a link to the shared
# tree): every entry of the shared tree linked into them (the cached looks, the episode lists: read only), scene/ (EP.build
# writes the episode's scene there, every episode, so nothing needs copying in) a real directory of the copy's own. Writes
# $O/<name>/CODE_MD5 (every copied file) and $O/<name>/LINKS (find -type l, each link with its realpath). Nothing in the
# shared trees is written.
set -euo pipefail
NAME=${1:-code}
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpleheft_20261126
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
H="ssh -p 27984 root@14.103.60.188"
cd /Users/zed/code/loco
$H "mkdir -p $O/$NAME"
rsync -a --delete -e "ssh -p 27984" --exclude __pycache__ --exclude .DS_Store --exclude '._*' \
  src scripts configs pyproject.toml root@14.103.60.188:$O/$NAME/
$H bash -s <<EOF
set -euo pipefail
cd $O/$NAME
for x in third_party assets grasp_kit rsi_kit; do ln -sfn $W/\$x \$x; done
mkdir -p outputs
for x in $W/outputs/*; do
  b=\$(basename \$x)
  case \$b in simple_eval_20261112 | simple_dev_20261124) continue ;; esac
  ln -sfn \$x outputs/\$b
done
for b in simple_eval_20261112 simple_dev_20261124; do
  d=outputs/\$b s=$W/outputs/\$b
  if [ -L \$d ]; then rm \$d; fi
  mkdir -p \$d/scene
  for x in \$s/*; do c=\$(basename \$x); [ "\$c" = scene ] || ln -sfn \$x \$d/\$c; done
  [ -d \$d/scene ] && [ ! -L \$d/scene ] && [ ! -L \$d ] || { echo "\$d/scene is not a real directory of the copy"; exit 2; }
done
find src scripts configs pyproject.toml -type f | sort | xargs md5sum > CODE_MD5
find . -type l | sort | while read -r l; do echo "\$l -> \$(realpath \$l)"; done > LINKS
echo "copied: \$(wc -l < CODE_MD5) files, \$(wc -l < LINKS) links; simple_bench \$(md5sum < scripts/experiments/simple_eval_20261112/simple_bench.py | cut -c1-8)"
EOF
