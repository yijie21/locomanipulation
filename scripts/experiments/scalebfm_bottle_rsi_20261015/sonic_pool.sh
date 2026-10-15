#!/usr/bin/env bash
# TODO 26.3.2 (user 2026-10-15): SONIC runs a bottle RSI start-state pool on instance 0 / GPU 0 with nothing else on the
# machine, unrecorded (recording pulled the real-time factor to 0.835-0.864 and voided half the rounds; the sim's state
# trace is written either way and trace_states.py converts it). Batches run until the pool's runs hold TARGET finished
# rounds: each batch is its own run (name ..._k<N>_..., seeds BASE + 100 N); a watchdog stops the stack and kills the
# expert client when the collect log has been silent for 12 min (a hung round), and the next batch tops up.
# Usage: bash sonic_pool.sh accept|sealed TARGET [first batch index] [log]
set -uo pipefail
cd /workspace/code/locomanipulation || exit 1
. scripts/paths.sh
export TMPDIR=/workspace/tmp NUMPY_MADVISE_HUGEPAGE=0
POOL=${1:?accept or sealed}
TARGET=${2:?finished rounds wanted}
K=${3:-1}
O=outputs/scalebfm_bottle_rsi_20261015_hidden
L=${4:-$O/sonic_${POOL}.log}
BASE_CFG=configs/experiments/scalebfm_bottle_rsi_${POOL}_sonic_20261015.yaml
GEN=$O/configs
mkdir -p "$O" "$GEN"
rounds_done() {  # finished rounds over every batch of this pool
  cat outputs/runs/scalebfm_bottle_rsi_${POOL}_k*_sonic_20261015/collect/manifest_*.jsonl 2>/dev/null | wc -l
}
while :; do
  have=$(rounds_done)
  [ "$have" -ge "$TARGET" ] && break
  n=$(( TARGET - have + 3 ))
  NAME=scalebfm_bottle_rsi_${POOL}_k${K}_sonic_20261015
  CFG=$GEN/$NAME.yaml
  python3 - "$BASE_CFG" "$CFG" "$NAME" "$K" "$n" <<'EOF'
import re, sys
src, dst, name, k, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
s = open(src).read()
s = re.sub(r"\nname: \S+", f"\nname: {name}", s, count=1)
seed = int(re.search(r"\n  seed0: (\d+)", s).group(1)) + 100 * k
s = re.sub(r"\n  seed0: \d+", f"\n  seed0: {seed}", s, count=1)
s = re.sub(r"\n  successes: \d+", f"\n  successes: {n}", s, count=1)
s = re.sub(r"\n  batch_rounds: \d+", f"\n  batch_rounds: {n}", s, count=1)
s, k_ = re.subn(r"\n  record: true", "\n  record: false", s, count=1)
assert k_ == 1, "record flag not found"
open(dst, "w").write(s)
print(f"{name}: seeds {seed}.., {n} rounds, unrecorded")
EOF
  RUN=outputs/runs/$NAME
  echo "=== collect $NAME ($n rounds; $have of $TARGET done) $(date -u)" >> "$L"
  ( while sleep 60; do
      f=$RUN/collect_00.log; [ -f "$f" ] || continue
      age=$(( $(date +%s) - $(stat -c %Y "$f") ))
      if [ "$age" -gt 720 ]; then
        echo "=== watchdog: $f silent ${age}s, stopping the stack and the expert client $(date -u)" >> "$L"
        bash scripts/stop_stack.sh --instance 0 >> "$L" 2>&1
        for p in $(pgrep -f "[l]ocomani.tasks.bottle.collect $CFG"); do kill "$p"; done
        break
      fi
    done ) &
  WD=$!
  bash scripts/start_stack.sh collect "$CFG" --gpu 0 >> "$L" 2>&1 && bash scripts/single_arm/run_agent_server.sh "$CFG" --gpu 0 >> "$L" 2>&1
  echo "=== collect $NAME done rc=$? $(date -u)" >> "$L"
  kill $WD 2>/dev/null
  bash scripts/stop_stack.sh --instance 0 >> "$L" 2>&1
  K=$(( K + 1 ))
done
echo "=== SONIC POOL $POOL DONE: $(rounds_done) rounds $(date -u)" >> "$L"
