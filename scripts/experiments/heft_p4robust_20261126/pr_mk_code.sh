#!/usr/bin/env bash
# notebook: skip heft_p4robust private copy
# heft_p4robust_20261126's private copy: heft_reanchor2x2's copy (code, its asrun/CODE_MD5 verified), symlinks listed first
# and kept as symlinks (read-only runtime paths: outputs, third_party, assets, rsi_kit, grasp_kit -- nothing is written
# under them); then this round's hw_run.py and pr_sum.py from $STAGE; md5s in asrun/CODE_MD5.
set -euo pipefail
S=/wuji-vefps-D/wuji-il/yijie/loco_outputs/heft_reanchor2x2_20261126
P=/wuji-vefps-D/wuji-il/yijie/loco_outputs/heft_p4robust_20261126
C=$P/code STAGE=$P/stage
[ -e $C ] && { echo "exists: $C"; exit 1; }
(cd $S/code && md5sum -c --quiet $S/asrun/CODE_MD5) && echo "source copy verifies against heft_reanchor2x2 CODE_MD5"
echo "symlinks in the source copy (kept, read-only):"; find $S/code -type l -printf "  %p -> %l\n" | grep -v __pycache__
rsync -a --exclude __pycache__ --exclude '*.pyc' $S/code/ $C/
for f in $(find $C -type l); do echo "$f" | grep -q "^$C/[a-z_]*$" || { echo "nested symlink: $f"; exit 1; }; done
X=$C/scripts/experiments
RC=$(realpath $C)  # H20: /wuji-vefps-D/wuji-il -> /wuji-vefps-D
for p in $X/heft_walkstop_20261126 $X/heft_p4robust_20261126; do mkdir -p $p; case "$(realpath $p)" in $RC/*) ;; *) echo "outside: $p"; exit 1;; esac; done
cp $STAGE/hw_run.py $X/heft_walkstop_20261126/hw_run.py
cp $STAGE/pr_sum.py $X/heft_p4robust_20261126/pr_sum.py
mkdir -p $P/asrun
cd $C && find scripts src ha_ctrl -type f \( -name "*.py" -o -name "*.sh" -o -name "*.xml" \) | grep -v __pycache__ | sort | xargs md5sum > $P/asrun/CODE_MD5
wc -l $P/asrun/CODE_MD5; md5sum $P/asrun/CODE_MD5
join -1 2 -2 2 <(sort -k2 $S/asrun/CODE_MD5) <(sort -k2 $P/asrun/CODE_MD5) | awk '$2 != $3 {print "changed vs heft_reanchor2x2:", $1}'
comm -13 <(cut -c35- $S/asrun/CODE_MD5 | sort) <(cut -c35- $P/asrun/CODE_MD5 | sort) | sed 's/^/added: /'
echo BUILD OK
