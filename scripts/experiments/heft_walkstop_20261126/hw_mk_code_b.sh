#!/usr/bin/env bash
# notebook: skip 4090b copy for heft_walkstop revision 2 (HA)
# heft_walkstop_20261126 revision 2 on 4090b: code_b = the round's copy (code, CODE_MD5_REV2 verified) EXCEPT outputs:
# code/outputs is a symlink to the shared tree, so code_b gets its own real outputs/ holding only hadata's frozen
# asrun_real/repo/outputs (the scenes HA reads: sx_ha.SCENE = outputs/scene/simple_g1_sonic/scene_locomani.xml), with
# "/workspace/ spelled "/wuji-vefps-D/wuji-il/yijie/ws/ (4090b has no /workspace; hadata's mk_repo_b.sh rule). Every
# rewritten file must sit inside code_b (realpath) and give asrun_real's md5 when the substitution is reversed.
set -euo pipefail
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/heft_walkstop_20261126
C=$O/code B=$O/code_b WS=/wuji-vefps-D/wuji-il/yijie/ws
A=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hadata_20261126/asrun_real
[ -e $B ] && { echo "exists: $B"; exit 1; }
(cd $C && md5sum -c --quiet $O/asrun/CODE_MD5_REV2) && echo "code verifies against CODE_MD5_REV2"
(cd $A && md5sum -c --quiet MD5.check) && echo "hadata asrun_real verifies"
rsync -a --exclude __pycache__ --exclude '*.pyc' --exclude /outputs $C/ $B/
[ -e $B/outputs ] && { echo "outputs came along"; exit 1; }
mkdir $B/outputs
rsync -a $A/repo/outputs/ $B/outputs/
find $B/outputs -type l | grep . && { echo "symlink inside code_b/outputs"; exit 1; } || true
CH=$(grep -rl "\"/workspace/" $B/outputs || true)
for f in $CH; do
  case "$(realpath $f)" in $B/*) ;; *) echo "outside code_b: $f -> $(realpath $f)"; exit 1;; esac
  grep -q "$WS/" $f && { echo "$f already holds $WS/"; exit 1; }
  sed -i "s#\"/workspace/#\"$WS/#g" $f
  grep -q "\"/workspace" $f && { echo "$f: /workspace left"; exit 1; }
  { grep -oE "(file|meshdir)=\"$WS/[^\"]*\"" $f || true; } | sed -E 's/^[a-z]+="//; s/"$//' | while read -r x; do [ -e "$x" ] || echo "missing: $x"; done
  p=${f#$B/}
  r=$(sed "s#\"$WS/#\"/workspace/#g" $f | md5sum | cut -c1-32); m=$(md5sum < $A/repo/$p | cut -c1-32)
  [ "$r" = "$m" ] && echo "rewritten, reverse = asrun_real ($m): $p" || { echo "reverse mismatch: $p"; exit 1; }
done
diff -rq --no-dereference $C $B | grep -v "^Only in $C: outputs$\|__pycache__" | grep . && { echo "code_b differs from code beyond outputs"; exit 1; } || true
cd $B && { find scripts src ha_ctrl -type f \( -name "*.py" -o -name "*.sh" -o -name "*.xml" \) | grep -v __pycache__; find outputs -type f; } | sort | xargs md5sum > $O/asrun/CODE_MD5_B
wc -l $O/asrun/CODE_MD5_B; md5sum $O/asrun/CODE_MD5_REV2 $O/asrun/CODE_MD5_B
echo "symlinks in code_b (read-only use):"; find $B -maxdepth 1 -type l -printf "  %p -> %l\n"
echo BUILD OK
