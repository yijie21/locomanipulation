#!/usr/bin/env bash
# notebook: selfmodel_hadata_20261126
# Revision 2a's bit-identity check with replayed looks (coordinator 2026-11-26, option B): Isaac's colour image differs
# run to run (depth is bit-identical), so SAM 3's masks and the episode drift; each look's colour image is taken from
# real_smoke6's saved look instead (--replay-looks; depth, pose and SAM 3 itself live). After hr_chain.sh's CHAIN DONE,
# in slot 1b (hold_1b stays): installs the staged hr_real.py.next / hr_real_old.py.next (the switch added to both, off by
# default), runs ScaleBFM PPBox 296 lift with each, then hr_same.py new vs old (the decision) and each vs real_smoke6,
# and SAM 3's masks per look against real_smoke6's. Writes dev/real_check8/.
set -uo pipefail
E=/wuji-vefps-D/wuji-il/yijie/ws/code/loco/scripts/experiments/selfmodel_hacollect_20261126
DV=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hacollect_20261126/dev
L=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hadata_20261126/logs
PY=/wuji-vefps-D/wuji-il/yijie/ws/code/loco/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
R=$DV/real_check8 S6=$DV/real_smoke6/lift/scalebfm_ppbox N=ppbox_s296_r0
mkdir -p $R
until grep -q "CHAIN DONE" $DV/hr_chain.log 2>/dev/null; do pgrep -f "[h]r_chain.sh" > /dev/null || { echo "hr_chain.sh gone without CHAIN DONE"; exit 2; }; sleep 30; done
[ -e $L/hold_1b ] || { echo "no hold_1b"; exit 2; }
cd $E
cp -p hr_real.py hr_real.py.12d43be5; cp -p hr_real_old.py hr_real_old.py.pre_replay
mv hr_real.py.next hr_real.py; mv hr_real_old.py.next hr_real_old.py
echo "$(date -u +%FT%TZ) slot 1b; hr_real $(md5sum < hr_real.py | cut -c1-8), hr_real_old $(md5sum < hr_real_old.py | cut -c1-8)"
run() {  # script out log
  SX_DEPTH=1 SX_HA_TASK=ppbox HC_SCRIPT=$1 bash hc_ha.sh 1 --controller scalebfm --out $2 --seeds 296 --repeats 0 --via 1 --via-rule lift \
    --replay-looks $S6/looks > $3 2>&1
  echo "$(date -u +%FT%TZ) $1 replay: exit $?; $(grep -c "replay-looks: rgb from" $3) looks replayed, $(grep -c "missing; live rgb" $3) missing"
}
run hr_real.py $R/new/scalebfm_ppbox $R/new.log
run hr_real_old.py $R/old/scalebfm_ppbox $R/old.log
for pr in "new old" "new smoke6" "old smoke6"; do
  set -- $pr; a=$R/$1/scalebfm_ppbox; [ $1 = smoke6 ] && a=$S6; b=$R/$2/scalebfm_ppbox; [ $2 = smoke6 ] && b=$S6
  echo "== $1 vs $2"; (cd /tmp && $PY -I $E/hr_same.py $a $b $N) | tee $R/same_${1}_$2.txt | grep -E "^(IDENTICAL|DIFFERENT)|arrays compared"
done
(cd /tmp && $PY -I - $R $S6 $N <<'PYEOF'
import glob, os, sys
import numpy as np
R, S6, N = sys.argv[1:4]
ref = sorted(glob.glob(f"{S6}/looks/{N}_*.npz"))
for run in ("new", "old"):
    got = sorted(glob.glob(f"{R}/{run}/scalebfm_ppbox/looks/{N}_*.npz"))
    out = []
    for a, b in zip(got, ref):
        za, zb = np.load(a), np.load(b)
        same_rgb = za["rgb"].tobytes() == zb["rgb"].tobytes()
        same_d = za["depth"].tobytes() == zb["depth"].tobytes()
        same_m = za["masks"].shape == zb["masks"].shape and za["masks"].tobytes() == zb["masks"].tobytes()
        out.append(f"{os.path.basename(a)[len(N) + 1:-4]}: rgb {'=' if same_rgb else '!='}, depth {'=' if same_d else '!='}, masks {'=' if same_m else '!='}")
    print(f"looks {run} vs smoke6 ({len(got)} vs {len(ref)}):", "; ".join(out))
PYEOF
) | tee $R/looks.txt
echo "$(date -u +%FT%TZ) REPLAY DONE"
