#!/usr/bin/env bash
# notebook: selfmodel_lmlroute_20261126
# Freeze selfmodel_lmlroute_20261126's private copy C0 = $O/code (made from lmlcmp3's frozen code/, md5 = its
# asrun/CODE_MD5, then this round's files synced in: scripts/experiments/selfmodel_lmlroute_20261126/ and hardgrasp's
# hg_look.py with the mask switch). Writes $A = $O/asrun: CODE_MD5 (every file of the copy but __pycache__, relative to
# C0), VS_LMLCMP3 (the copy against lmlcmp3's CODE_MD5: changed / added / missing files), MD5 (the inputs read: lmlcmp3's
# input lines but the SAM 3 service script, plus this round's copy of the joint band and the cuRobo robot files lrt_plan
# reads), LOOKS_MD5 (the round's looks),
# FROZEN_UTC. One shot: refuses an existing asrun/.
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlroute_20261126
L3=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126
S=/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE
A=$O/asrun
C0=$O/code
[ -e $A ] && { echo "$A exists: refusing"; exit 1; }
mkdir -p $A
date -u +%FT%TZ > $A/FROZEN_UTC
(cd $C0 && find src configs scripts -type f -not -path "*/__pycache__/*" | sort | xargs md5sum) > $A/CODE_MD5
{
  join -j 2 <(awk '{print $1, $2}' $L3/asrun/CODE_MD5 | sort -k2) <(awk '{print $1, $2}' $A/CODE_MD5 | sort -k2) -a 1 -a 2 -e MISSING -o 0,1.1,2.1 |
    awk '$2 != $3 {print ($2 == "MISSING" ? "added  " : $3 == "MISSING" ? "missing" : "changed"), $1, substr($2, 1, 8), "->", substr($3, 1, 8)}'
} > $A/VS_LMLCMP3
grep -v sam3_server $L3/asrun/MD5 > $A/MD5
md5sum $O/hg/train_q_range.npz $S/data/robots/g1/curobo/g1_29dof_with_dex3.yml $S/data/robots/g1/curobo/spheres/g1_29dof.yml \
  $(grep -o 'urdf_path: *[^ ]*' $S/data/robots/g1/curobo/g1_29dof_with_dex3.yml | awk -v d=$S/data/robots/g1/ '{print d $2}') >> $A/MD5
(cd $O/hg/looks && md5sum $(ls -d */ | tr -d / | sed 's|$|/*.npz|') | sed "s|^\([0-9a-f]*\)  |\1  $O/hg/looks/|") > $A/LOOKS_MD5
cat $A/FROZEN_UTC; echo "copy: $(wc -l < $A/CODE_MD5) files; against lmlcmp3's copy:"; cat $A/VS_LMLCMP3
echo "inputs: $(wc -l < $A/MD5) lines; looks: $(wc -l < $A/LOOKS_MD5) files"
