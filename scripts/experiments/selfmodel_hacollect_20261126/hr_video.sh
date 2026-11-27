#!/usr/bin/env bash
# notebook: selfmodel_hadata_20261126
# The HEFT diagnosis videos (for viewing only; no statistic, no criterion; coordinator 2026-11-26, top priority): on
# 4090b card 1 in slot b1b, freed by stopping scalebfm_d3_w14 (the shard started last, 06:46:53Z; handed back to the
# queue, hc_ha.py skips its chunks already written). HEFT P&PBox seeds 296 and 297, the lift rule, hr_real.py --hrvid
# (dev copy, as the 4090's smokes: the dev humanoidarena / xfer dirs; hd_job_b.sh's repo_b, ha_ctrl and workspace for
# 4090b). The slot goes back to the collection (hold_b1b removed) once both videos exist; a failed run keeps the hold
# (logged) for a retry by hand.
set -uo pipefail
E=/wuji-vefps-D/wuji-il/yijie/ws/code/loco/scripts/experiments/selfmodel_hacollect_20261126
A=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hadata_20261126/asrun
D=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hacollect_20261126/dev/b4090b
L=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hadata_20261126/logs
V=/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_hacollect_20261126/dev/heft_video
mkdir -p $V
echo "$(date -u +%FT%TZ) start on $(hostname) card 1; hr_real $(md5sum < $E/hr_real.py | cut -c1-8), hr_video $(md5sum < $E/hr_video.py | cut -c1-8)"
unset SX_D3 SX_NORENDER
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco/scripts/experiments
export SX_REPO=$D/repo_b SX_HA_EXP=$W/humanoidarena_20261124 SX_HA_CTRL=$A/ha_ctrl HC_XFER=$W/selfmodel_xfer_20261125
export LOCOMANI_WORKSPACE=/wuji-vefps-D/wuji-il/yijie/ws
cd $E
SX_DEPTH=1 SX_HA_TASK=ppbox HC_SCRIPT=hr_real.py bash hc_ha.sh 1 --controller heft --out $V/run --seeds 296-297 --repeats 0 \
  --via 1 --via-rule lift --hrvid $V > $V/run.log 2>&1
rc=$?
if [ $(ls $V/video/*.mp4 2>/dev/null | wc -l) -ge 2 ]; then
  rm -f $L/hold_b1b
  echo "$(date -u +%FT%TZ) slot b1b back to the collection (hold_b1b removed; video run exit $rc)" >> $L/formal_4090b.log
  echo "$(date -u +%FT%TZ) end exit $rc; hold_b1b removed"
else
  echo "$(date -u +%FT%TZ) video run failed (exit $rc); hold_b1b kept for a retry by hand" >> $L/formal_4090b.log
  echo "$(date -u +%FT%TZ) end exit $rc; FAILED, hold_b1b kept"
fi
ls -la $V/video $V/steps
echo "VIDEO DONE"
