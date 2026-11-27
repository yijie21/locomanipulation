#!/usr/bin/env bash
# mkrepo_b.sh <source repo copy> <new repo_b>: simple_relook_20261126's tree for 4090b (no /workspace there). src, scripts,
# configs and pyproject copied from the source; assets and outputs/* linked as in the source, except outputs/scene's
# simple_g1_sonic and walk_mars, linked to selfmodel_simpledata_20261126's copies with the vePFS paths; third_party/SIMPLE
# linked; third_party/GR00T-WholeBodyControl a directory of links (decoupled_wbc and gear_sonic spelled on vePFS) whose
# .venv_sim/bin/python is a wrapper around py4090b_rl.py, so bench_via.sh / sfull_run.sh / dev.sh run unchanged.
set -euo pipefail
SRC=$(realpath $1) DST=$2
WS=/wuji-vefps-D/wuji-il/yijie/ws
SB=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_simpledata_20261126/asrun/repo_b/outputs/scene
[ -e $DST ] && { echo "$DST exists"; exit 2; }
mkdir -p $DST/outputs/scene $DST/third_party/GR00T-WholeBodyControl/.venv_sim/bin
cp -a $SRC/src $SRC/scripts $SRC/configs $SRC/pyproject.toml $DST/
ln -s $(readlink -f $SRC/assets) $DST/assets
for o in $SRC/outputs/*; do n=$(basename $o); [ $n = scene ] || ln -s $(readlink -f $o) $DST/outputs/$n; done
for d in $(readlink -f $SRC/outputs/scene)/*; do
  n=$(basename $d)
  if [ -d $SB/$n ] && [ ! -L $SB/$n ]; then ln -s $SB/$n $DST/outputs/scene/$n; else ln -s $(readlink -f $d) $DST/outputs/scene/$n; fi
done
ln -s $WS/code/loco/third_party/SIMPLE $DST/third_party/SIMPLE
G=$DST/third_party/GR00T-WholeBodyControl
ln -s $WS/code/loco/third_party/GR00T-WholeBodyControl/gear_sonic_deploy $G/gear_sonic_deploy
for p in decoupled_wbc gear_sonic; do ln -s $WS/code/locomanipulation/third_party/GR00T-WholeBodyControl/$p $G/$p; done
printf '#!/bin/bash\nexec /usr/bin/python3.10 -S %s "$@"\n' $(realpath $DST)/scripts/experiments/simple_relook_20261126/py4090b_rl.py > $G/.venv_sim/bin/python
chmod +x $G/.venv_sim/bin/python
echo "made $DST"
