#!/usr/bin/env bash
# notebook: bigym_selfmodel_20261124 -- the self model's data on BiGym 2.0's own controller (bigym_collect.py), the cuRobo
# planners kept up (curobo/plan_server.py). docs/handoff/notebook-20261120/bigym_selfmodel_20261124.md
# collect.sh <out dir> <stances a task> [<processes a task>]
#   reach stances are ids 0..N-1, the drawer's DRAWER_BASE..DRAWER_BASE+N-1 (seed 800000 + id; N <= DRAWER_BASE).
#   GPUS (default "1 2 3"), SRV_PER_GPU (default 5): the planners. A GPU is used only if nvidia-smi shows it under
#   BUSY_MB MiB used (default 4000) and BUSY_UTIL % busy (default 20) at the start; skipped ones are logged.
#   The sim processes (default 15 a task) run on the CPU (no rendering) and take the planners in turn.
#   Progress: <out>/collect.log; rows in <out>/rows, films in <out>/films, the scripts as run in <out>/asrun (md5s).
set -uo pipefail
D=$1 N=$2 P=${3:-15}
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco; cd $W
SIMPLE=/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE
E=scripts/experiments/bigym_20261124
DRAWER_BASE=${DRAWER_BASE:-2500}
[ "$N" -le "$DRAWER_BASE" ] || { echo "N $N > DRAWER_BASE $DRAWER_BASE"; exit 1; }
read -ra GS <<< "${GPUS:-1 2 3}"; NS=${SRV_PER_GPU:-5}
export BIGYM_EGL_GPU=0 OMP_NUM_THREADS=1
source /wuji-vefps-D/wuji-il/yijie/tools/bigym2/env.sh
mkdir -p $D/rows $D/logs $D/films $D/srv $D/asrun
cp $E/bigym_collect.py $E/bigym_bench.py $E/spheres24.py $E/bigym_data.py $E/collect.sh $E/curobo/plan_server.py $D/asrun/
(cd $D/asrun && md5sum * > md5.txt)
log() { echo "[$(date -u +%T)] $*" >> $D/collect.log; }
log "start: $N stances a task, $P processes a task, GPUs ${GS[*]} x $NS planners, commit $(git log -1 --format=%h) + uncommitted (asrun/)"

ok_gpus=()
for g in "${GS[@]}"; do
  read -r mem util <<< "$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i $g | tr -d ',')"
  if [ "$mem" -lt "${BUSY_MB:-4000}" ] && [ "$util" -lt "${BUSY_UTIL:-20}" ]; then ok_gpus+=($g)
  else log "GPU $g skipped: ${mem} MiB used, ${util} % busy (someone else's)"; fi
done
[ ${#ok_gpus[@]} -gt 0 ] || { log "FAIL no free GPU among ${GS[*]}"; exit 1; }

socks=() spids=() SK=/tmp/bigym_sm_$$; mkdir -p $SK  # AF_UNIX paths stop at 108 characters: not under $D
for g in "${ok_gpus[@]}"; do for j in $(seq 1 $NS); do
  sk=$SK/g${g}_$j.sock; rm -f $sk; socks+=($sk)
  CUDA_VISIBLE_DEVICES=$g $SIMPLE/.venv/bin/python -I $E/curobo/plan_server.py $sk > $D/srv/g${g}_$j.log 2>&1 & spids+=($!)
done; done
for sk in "${socks[@]}"; do
  for i in $(seq 1 90); do [ -S $sk ] && break; sleep 2; done
  [ -S $sk ] || { log "FAIL planner $sk did not come up"; kill "${spids[@]}"; exit 1; }
done
log "${#socks[@]} planners up on GPUs ${ok_gpus[*]} (pids ${spids[*]})"

per=$(((N + P - 1) / P)); pids=() w=0
for T in reach_target_multi_modal drawer_top_open; do
  base=0; [ $T = drawer_top_open ] && base=$DRAWER_BASE
  for c in $(seq 0 $((P - 1))); do
    a=$((base + c * per)); b=$((a + per)); [ $b -gt $((base + N)) ] && b=$((base + N)); [ $a -ge $b ] && continue
    sk=${socks[$((w % ${#socks[@]}))]}; w=$((w + 1))
    nice -n 10 python $E/bigym_collect.py --task $T --sids $a:$b --out $D/rows/${T}_$a-$b.jsonl --film $D/films \
      --plan-socket $sk > $D/logs/${T}_$a-$b.log 2>&1 & pids+=($!)
  done
done
log "${#pids[@]} sim processes (pids ${pids[*]})"
fails=0
for p in "${pids[@]}"; do wait $p || fails=$((fails + 1)); done
kill "${spids[@]}" 2>/dev/null
rm -rf $SK
log "DONE: $(cat $D/rows/*.jsonl | wc -l) stances, $fails failed processes"
