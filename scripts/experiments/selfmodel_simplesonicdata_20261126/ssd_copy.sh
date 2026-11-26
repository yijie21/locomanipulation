#!/usr/bin/env bash
# notebook: skip code copy (no run) -- selfmodel_simplesonicdata_20261126's private code tree on H20b, from the Mac:
#   ssd_copy.sh [code dir name, default code]
# Run on the Mac. Copies the Mac checkout's src/, scripts/, configs/, pyproject.toml (the working tree: others' uncommitted
# edits included, as they are; __pycache__, .DS_Store and ._* left out) to $O/<name>/ on H20b and builds the rest as
# simple_relook_20261126's mkrepo_b.sh builds a tree for a machine without /workspace: assets and outputs/* linked to the
# shared ws tree (read only); outputs/scene's entries linked to selfmodel_simpledata_20261126's repo_b copies where those
# are real directories (simple_g1_sonic, walk_mars: the vePFS paths), else to the ws tree's; third_party/SIMPLE linked;
# third_party/GR00T-WholeBodyControl a directory of links whose .venv_sim/bin/python wraps the copy's own
# simple_relook_20261126/py4090b_rl.py. outputs/simple_eval_20261112 and outputs/simple_dev_20261124 (SIMPLE_EVAL_OUT) are
# real directories of the copy (coordinator 11-26: never a link to the shared tree): every entry of the ws tree's linked
# into them (the cached looks, the episode lists: read only) but scene/, a real directory of the copy's own (EP.build
# writes each episode's scene there). Writes $O/<name>/CODE_MD5 (every copied file) and $O/<name>/LINKS (find -type l,
# each link with its realpath). Nothing in the shared trees is written.
set -euo pipefail
NAME=${1:-code}
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simplesonicdata_20261126
WS=/wuji-vefps-D/wuji-il/yijie/ws
SB=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpledata_20261126/asrun/repo_b/outputs/scene
P="-o ConnectTimeout=15 -p 16093"
H="ssh $P root@14.103.60.188"
cd /Users/zed/code/loco
$H "[ ! -e $O/$NAME ] || { echo '$O/$NAME exists'; exit 2; }; mkdir -p $O/$NAME"
rsync -a -e "ssh $P" --exclude __pycache__ --exclude .DS_Store --exclude '._*' \
  src scripts configs pyproject.toml root@14.103.60.188:$O/$NAME/
$H bash -s <<EOF
set -euo pipefail
cd $O/$NAME
W=$WS/code/loco
ln -s \$(readlink -f \$W/assets) assets
mkdir -p outputs/scene third_party/GR00T-WholeBodyControl/.venv_sim/bin
for x in \$W/outputs/*; do
  b=\$(basename \$x)
  case \$b in scene | simple_eval_20261112 | simple_dev_20261124) continue ;; esac
  ln -s \$(readlink -f \$x) outputs/\$b
done
for d in \$(readlink -f \$W/outputs/scene)/*; do
  n=\$(basename \$d)
  if [ -d $SB/\$n ] && [ ! -L $SB/\$n ]; then ln -s $SB/\$n outputs/scene/\$n; else ln -s \$(readlink -f \$d) outputs/scene/\$n; fi
done
for b in simple_eval_20261112 simple_dev_20261124; do
  d=outputs/\$b s=\$W/outputs/\$b
  mkdir -p \$d/scene
  for x in \$s/*; do c=\$(basename \$x); [ "\$c" = scene ] || ln -s \$(readlink -f \$x) \$d/\$c; done
  [ -d \$d/scene ] && [ ! -L \$d/scene ] && [ ! -L \$d ] || { echo "\$d/scene is not a real directory of the copy"; exit 2; }
done
ln -s $WS/code/loco/third_party/SIMPLE third_party/SIMPLE
G=third_party/GR00T-WholeBodyControl
ln -s $WS/code/loco/third_party/GR00T-WholeBodyControl/gear_sonic_deploy \$G/gear_sonic_deploy
for p in decoupled_wbc gear_sonic; do ln -s $WS/code/locomanipulation/third_party/GR00T-WholeBodyControl/\$p \$G/\$p; done
printf '#!/bin/bash\nexec /usr/bin/python3.10 -S %s "\$@"\n' $O/$NAME/scripts/experiments/simple_relook_20261126/py4090b_rl.py > \$G/.venv_sim/bin/python
chmod +x \$G/.venv_sim/bin/python
find src scripts configs pyproject.toml -type f | sort | xargs md5sum > CODE_MD5
find . -type l | sort | while read -r l; do echo "\$l -> \$(realpath \$l)"; done > LINKS
echo "copied: \$(wc -l < CODE_MD5) files, \$(wc -l < LINKS) links; simple_bench \$(md5sum < scripts/experiments/simple_eval_20261112/simple_bench.py | cut -c1-8), sc_collect \$(md5sum < scripts/experiments/selfmodel_simplecollect_20261126/sc_collect.py | cut -c1-8), scmp_correct \$(md5sum < scripts/experiments/selfmodel_simplecmp_20261126/scmp_correct.py | cut -c1-8)"
EOF
