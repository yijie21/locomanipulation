# notebook: skip figure render, not an experiment
# mk_copy.sh <frozen code dir> <new copy dir>: a private copy of a frozen SIMPLE code tree for H20b (no /workspace), as
# simple_relook_20261126's mkrepo_b.sh builds repo_b, except: outputs/simple_eval_20261112 is a real directory with a real
# scene/ (simple_bench rewrites scene/<episode>/ every episode; the other entries are links into the shared tree, read
# only), outputs/scene's robot scenes are copied in (not linked), and simple_dev_20261124 is left out. The frozen copy's
# CODE_MD5 is checked on the source and on the copy. Writes <copy>/LINKS (every link with its realpath).
set -euo pipefail
SRC=$(realpath $1) DST=$2
Y=/wuji-vefps-D/wuji-il/yijie WS=$Y/ws W=$Y/ws/code/loco
SB=$Y/loco_outputs/selfmodel_simpledata_20261126/asrun/repo_b/outputs/scene
[ -e $DST ] && { echo "$DST exists"; exit 2; }
(cd $SRC && md5sum -c --quiet CODE_MD5) || { echo "FAIL: $SRC differs from its CODE_MD5"; exit 2; }
mkdir -p $DST
cp -a $SRC/src $SRC/scripts $SRC/configs $SRC/pyproject.toml $SRC/CODE_MD5 $DST/
(cd $DST && md5sum -c --quiet CODE_MD5) || { echo "FAIL: the copy differs from CODE_MD5"; exit 2; }
for x in assets grasp_kit rsi_kit; do ln -s $(readlink -f $SRC/$x) $DST/$x; done
mkdir -p $DST/outputs/scene $DST/outputs/simple_eval_20261112/scene
for o in $SRC/outputs/*; do
  n=$(basename $o)
  case $n in scene | simple_eval_20261112 | simple_dev_20261124) continue ;; esac
  ln -s $(readlink -f $o) $DST/outputs/$n
done
for d in $W/outputs/scene/*; do
  n=$(basename $d)
  if [ -d $SB/$n ] && [ ! -L $SB/$n ]; then cp -a $SB/$n $DST/outputs/scene/$n; else cp -a $(readlink -f $d) $DST/outputs/scene/$n; fi
done
for x in $W/outputs/simple_eval_20261112/*; do c=$(basename $x); [ "$c" = scene ] || ln -s $x $DST/outputs/simple_eval_20261112/$c; done
G=$DST/third_party/GR00T-WholeBodyControl
mkdir -p $G/.venv_sim/bin
ln -s $W/third_party/SIMPLE $DST/third_party/SIMPLE
ln -s $W/third_party/GR00T-WholeBodyControl/gear_sonic_deploy $G/gear_sonic_deploy
for p in decoupled_wbc gear_sonic; do ln -s $WS/code/locomanipulation/third_party/GR00T-WholeBodyControl/$p $G/$p; done
printf '#!/bin/bash\nexec /usr/bin/python3.10 -S %s "$@"\n' $(realpath $DST)/scripts/experiments/simple_relook_20261126/py4090b_rl.py > $G/.venv_sim/bin/python
chmod +x $G/.venv_sim/bin/python
T=$DST/outputs/simple_eval_20261112
[ ! -L $T ] && [ ! -L $T/scene ] || { echo "FAIL: $T or its scene/ is a link"; exit 2; }
case "$(realpath $T/scene)" in $(realpath $DST)/*) ;; *) echo "FAIL: $T/scene resolves to $(realpath $T/scene)"; exit 2 ;; esac
find $DST -type l | sort | while read -r l; do echo "$l -> $(realpath $l)"; done > $DST/LINKS
echo "made $DST: $(wc -l < $DST/CODE_MD5) files (CODE_MD5 ok), $(wc -l < $DST/LINKS) links; tree $(realpath $T/scene)"
echo "simple_bench $(md5sum < $DST/scripts/experiments/simple_eval_20261112/simple_bench.py | cut -c1-8), scmp_correct $(md5sum < $DST/scripts/experiments/selfmodel_simplecmp_20261126/scmp_correct.py | cut -c1-8), py4090b_rl $(md5sum < $DST/scripts/experiments/simple_relook_20261126/py4090b_rl.py | cut -c1-8)"
