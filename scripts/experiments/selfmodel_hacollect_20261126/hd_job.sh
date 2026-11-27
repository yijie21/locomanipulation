#!/usr/bin/env bash
# notebook: selfmodel_hadata_20261126
# One formal shard: one hc_ha.py drive process on one card, from a frozen asrun copy (hd_freeze.sh), after verifying it.
#   hd_job.sh <asrun> <out root> <gpu> <controller> <d3|d3b> <shard> <shards> <resets> <seed0> <id0>
# Writes <out root>/<controller>/<d3|d3b>/ (data, data_both, data_detour for d3b, run.json from shard 0).
set -uo pipefail
A=$1 OUT=$2 GPU=$3 C=$4 MODE=$5 K=$6 P=$7 N=$8 S0=$9 I0=${10}
[ -e $A/MD5 ] || { echo "not frozen: $A"; exit 2; }
(cd $A && md5sum -c --quiet MD5.check) || { echo "the frozen copy or a pinned install changed"; exit 2; }
unset SX_D3 SX_NORENDER HC_SCRIPT
export SX_REPO=$A/repo SX_HA_EXP=$A/humanoidarena_20261124 SX_HA_CTRL=$A/ha_ctrl HC_XFER=$A/selfmodel_xfer_20261125
EX=(--hc-seed $S0 --hc-id0 $I0)
[ "$MODE" = d3b ] && EX+=(--b --b-config $A/b_config/$C.json)
[ "$C" = sonic ] && EX+=(--no-camera)
mkdir -p $OUT/$C/$MODE
exec bash $A/selfmodel_hacollect_20261126/hc_ha.sh $GPU --phase drive --controller $C --out $OUT/$C/$MODE \
  --resets $N --shards $P --shard $K "${EX[@]}"
