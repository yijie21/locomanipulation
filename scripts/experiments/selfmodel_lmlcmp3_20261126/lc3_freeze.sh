#!/usr/bin/env bash
# notebook: selfmodel_lmlcmp3_20261126
# Freeze selfmodel_lmlcmp3_20261126's code as the round's PRIVATE COPY C0 = $O/code (revision 1 addendum: the shared
# repo tree on 4090 gets synced into by other rounds, so the formal run imports only from this copy):
#   copied (rsync -a, no __pycache__): src/ (locomani, gr00t, real_robot), scripts/rsi (env.py puts it on the path),
#     configs/, and the scripts/experiments directories of the chain: lmlcmp3, lmlcmp2 (lc2_run.py, the diff base),
#     lmlcmp (heft_gait), lmlreal, hardgrasp, data3, locomanilab, reachwm, selfwm;
#   linked to the shared tree (data and third-party trees the code reaches through repo-relative paths; lc3_run's code
#     check stops any run process that imports a module through them): assets, outputs, third_party, grasp_kit, rsi_kit.
# Writes $A = $O/asrun: CODE_MD5 (every file of the copy, relative to C0), MD5 (the inputs read: hardgrasp's joint
# band and this round's copy of it, the SAM 3 service script the looks used, the T3 / M / D model.pt of every controller,
# lmlreal's and data3's data), LOOKS3_MD5 (the new seeds' looks), LOOKS_MD5 (hardgrasp's looks: the smoke's design
# episodes), VS_PREV (the copy against the 02:22:49Z freeze of commit 73cdf7a4) and FROZEN_UTC. One shot: refuses an
# existing copy.
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
O=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126
A=$O/asrun
C0=$O/code
R=/wuji-vefps-D/wuji-il/yijie/loco_outputs
HGO=$W/outputs/hardgrasp_20261125/locomanilab
PREV=$O/aborted_formal_0225/asrun_0222  # the 02:22:49Z freeze (commit 73cdf7a4)
DIRS="selfmodel_lmlcmp3_20261126 selfmodel_lmlcmp2_20261126 selfmodel_lmlcmp_20261126 selfmodel_lmlreal_20261125 hardgrasp_20261125 selfmodel_data3_20261125 locomanilab_20261117 reachwm_20261030 selfwm_20261029"
[ -e $C0 ] && { echo "$C0 exists: refusing"; exit 1; }
[ -e $A ] && { echo "$A exists: refusing"; exit 1; }
mkdir -p $C0/scripts/experiments $A
rsync -a --exclude __pycache__ $W/src $W/configs $C0/
rsync -a --exclude __pycache__ $W/scripts/rsi $C0/scripts/
for d in $DIRS; do rsync -a --exclude __pycache__ $W/scripts/experiments/$d $C0/scripts/experiments/; done
for d in assets outputs third_party grasp_kit rsi_kit; do ln -s $W/$d $C0/$d; done
date -u +%FT%TZ > $A/FROZEN_UTC
(cd $C0 && find src configs scripts -type f -not -path "*/__pycache__/*" | sort | xargs md5sum) > $A/CODE_MD5
md5sum $HGO/train_q_range.npz $O/hg/train_q_range.npz $W/scripts/rsi/sam3_server.py > $A/MD5
for BE in scalebfm heft sonic; do
  md5sum $R/selfmodel_data3_20261125/$BE/models_s0/E3_ensemble_f1/model.pt $R/selfmodel_train3b_20261125/$BE/models_M_s0/E3_ensemble_f1/model.pt \
    $R/selfmodel_train3b_20261125/$BE/models_D_s0/E3_ensemble_f1/model.pt $R/selfmodel_lmlreal_20261125/$BE/data/all.npz \
    $R/selfmodel_data3_20261125/$BE/data/all.npz >> $A/MD5
done
(cd $O/hg/looks && md5sum $(ls -d */ | tr -d / | sed 's|$|/*.npz|') | sed "s|^\([0-9a-f]*\)  |\1  $O/hg/looks/|") > $A/LOOKS3_MD5
(cd $HGO/looks && md5sum $(ls -d */ | tr -d / | sed 's|$|/*.npz|') | sed "s|^\([0-9a-f]*\)  |\1  $HGO/looks/|") > $A/LOOKS_MD5
{
  echo "== the 02:22:49Z freeze's code list (relative to scripts/experiments) against the copy"
  grep -v '^[0-9a-f]*  /' $PREV/MD5 | while read -r h f; do
    n=$(md5sum < $C0/scripts/experiments/$f | cut -c1-32)
    [ "$h" = "$n" ] && echo "same     $f" || echo "DIFFERS  $f ${h:0:8} -> ${n:0:8}"
  done
  echo "== the 02:22:49Z freeze's input lines against this freeze's"
  diff <(grep '^[0-9a-f]*  /' $PREV/MD5 | sort) <(sort $A/MD5) && echo "inputs: same lines (this freeze adds none)"
  cmp -s $PREV/LOOKS3_MD5 $A/LOOKS3_MD5 && echo "LOOKS3_MD5 same" || echo "LOOKS3_MD5 DIFFERS"
  cmp -s $PREV/LOOKS_MD5 $A/LOOKS_MD5 && echo "LOOKS_MD5 same" || echo "LOOKS_MD5 DIFFERS"
  echo "== copied files outside that list modified after $(cat $PREV/FROZEN_UTC) (rsync keeps the tree's mtimes)"
  (cd $C0 && find src configs scripts -type f -not -path "*/__pycache__/*" -newermt "$(cat $PREV/FROZEN_UTC)" | sort | while read -r f; do
    grep -q "  ${f#scripts/experiments/}$" $PREV/MD5 || echo "  $(stat -c %y $f | cut -c1-19) $f"
  done)
} > $A/VS_PREV
cat $A/FROZEN_UTC; echo "copy: $(wc -l < $A/CODE_MD5) files, $(du -sh --exclude=__pycache__ $C0/src $C0/configs $C0/scripts | awk '{print $1}' | tr '\n' ' ')"
echo "inputs: $(wc -l < $A/MD5) lines; new looks: $(wc -l < $A/LOOKS3_MD5) files; hardgrasp looks: $(wc -l < $A/LOOKS_MD5) files"
cat $A/VS_PREV
