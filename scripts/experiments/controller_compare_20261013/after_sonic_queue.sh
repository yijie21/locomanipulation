#!/usr/bin/env bash
# Chain for the night of 10-13: wait for sonic_l2_p25_queue.sh, top up any L2 bottle batch that ended short of 40 rounds
# (the nominal one hung at round 30), redo the loaded C4 clips on SONIC (the first batch streamed yaw-only pelvis
# attitudes), then everything ScaleBFM-side and the tables. Log: outputs/controller_compare_20261013/after.log
set -uo pipefail
cd /workspace/code/locomanipulation || exit 1
. scripts/paths.sh
export TMPDIR=/workspace/tmp NUMPY_MADVISE_HUGEPAGE=0
D=scripts/experiments/controller_compare_20261013
E=configs/experiments
L=outputs/controller_compare_20261013/after.log
echo "=== waiting $(date -u)" >> $L
until grep -q "SONIC L2 P25 DONE" outputs/controller_compare_20261013/sonic_l2_p25.log 2>/dev/null; do sleep 30; done
echo "=== sonic queue done $(date -u)" >> $L
for tier in "" _combo; do
  n=$(cat outputs/runs/controller_compare_l2_bottle_sonic${tier}_20261013/collect/manifest_*.jsonl 2>/dev/null | wc -l)
  if [ "$n" -lt 40 ]; then
    C=$E/controller_compare_l2_bottle_sonic${tier}_b_20261013.yaml
    echo "=== make-up $C ($n rounds) $(date -u)" >> $L
    bash scripts/start_stack.sh collect $C --gpu 0 >> $L 2>&1 && bash scripts/single_arm/run_agent_server.sh $C --gpu 0 >> $L 2>&1
    echo "=== make-up done rc=$? $(date -u)" >> $L
  fi
done
bash $D/sonic_clip.sh walk_mars_load12 0 0 --repeats 10 >> $L 2>&1
echo "=== load12 clips done $(date -u)" >> $L
bash $D/p25_l2_post.sh convert >> $L 2>&1
echo "=== convert done $(date -u)" >> $L
bash $D/bfm_l2_batch.sh >> $L 2>&1
bash $D/bfm_p25_batch.sh >> $L 2>&1
bash $D/p25_l2_post.sh tables >> $L 2>&1
echo "=== AFTER DONE $(date -u)" >> $L
