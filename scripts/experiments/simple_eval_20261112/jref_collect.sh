#!/usr/bin/env bash
# notebook: selfmodel_jrefwm_20261124 -- TODO 38.10.3's data on the 4090: cuRobo right-arm plans from SIMPLE stances, run by the
# frozen controller (simple_bench.py --collect, jref_collect.py), the planners kept up (jref_plan_server.py).
# jref_collect.sh <out dir> <scalebfm|sonic> <reps a-b> <chunks a task> [k a stance]
#   GPUS (default "1 2 3"), SRV_PER_GPU (default 5): the planners; the bench processes go to the same GPUs in turn.
#   Layouts with crc32 % 3 == 0 are left out (simple_bench.py --holdout 3). Progress: <out>/collect.log, rows in <out>/rows.
set -uo pipefail
D=$1 CTRL=$2 REPS=$3 CHUNKS=$4 K=${5:-2}
W=/wuji-vefps-D/wuji-il/yijie/ws/code/loco; cd $W
S=/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE; SP=$S/.venv/bin/python; PY=$W/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
E=scripts/experiments/simple_eval_20261112
TASKS="XMovePickTeleop XMoveBendPickTeleop BendPickMP TabletopGraspMP"
read -ra GS <<< "${GPUS:-1 2 3}"; NS=${SRV_PER_GPU:-5}
export REACH_HAND_OFF=0.0415,0,0 BFM_ROOT=odo REACHWM_GOAL_REF=1 PYTHONPATH=src OMP_NUM_THREADS=1 MUJOCO_GL=egl SIMPLE_ROOT=$S
[ "$CTRL" = sonic ] && export REACH_CONTROLLER=sonic REACH_SONIC_NETS=${REACH_SONIC_NETS:-gpu}
mkdir -p $D/rows $D/logs $D/films $D/srv $D/asrun
cp $E/simple_bench.py $E/jref_collect.py $E/jref_plan_server.py $E/jref_data.py $E/jref_collect.sh src/locomani/backends/scalebfm/motor.py $D/asrun/
log() { echo "[$(date -u +%T)] $*" >> $D/collect.log; }
log "start: $CTRL, reps $REPS, $CHUNKS chunks a task, k $K, GPUs ${GS[*]} x $NS planners, commit $(git log -1 --format=%h) + uncommitted (asrun/)"

socks=() spids=() SK=/tmp/jrefwm_$CTRL; mkdir -p $SK
for g in "${GS[@]}"; do for j in $(seq 1 $NS); do
  sk=$SK/g${g}_$j.sock; rm -f $sk; socks+=($sk)  # AF_UNIX paths stop at 108 characters: not under $D
  (cd $S && CUDA_VISIBLE_DEVICES=$g exec $SP -I $W/$E/jref_plan_server.py $S $sk > $D/srv/g${g}_$j.log 2>&1) & spids+=($!)
done; done
for sk in "${socks[@]}"; do
  for i in $(seq 1 60); do [ -S $sk ] && break; sleep 2; done
  [ -S $sk ] || { log "FAIL planner $sk did not come up"; kill "${spids[@]}"; exit 1; }
done
log "${#socks[@]} planners up (pids ${spids[*]})"

r0=${REPS%-*} r1=${REPS#*-}; n=$((r1 - r0 + 1)); per=$(((n + CHUNKS - 1) / CHUNKS))
pids=() w=0
for T in $TASKS; do for c in $(seq 0 $((CHUNKS - 1))); do
  a=$((r0 + c * per)); b=$((a + per - 1)); [ $b -gt $r1 ] && b=$r1; [ $a -gt $r1 ] && continue
  g=${GS[$((w % ${#GS[@]}))]}; sk=${socks[$((w % ${#socks[@]}))]}; w=$((w + 1))
  MUJOCO_EGL_DEVICE_ID=$g CUDA_VISIBLE_DEVICES=$g nice -n 10 $PY $E/simple_bench.py $T --method direct --column perception \
    --cloud cloud_eyes_final --episodes 0-29 --collect $sk --reps $a-$b --collect-k $K --film $D/films \
    --out $D/rows/${T}_r$a-$b.jsonl > $D/logs/${T}_r$a-$b.log 2>&1 & pids+=($!)
done; done
log "${#pids[@]} bench processes (pids ${pids[*]})"
fails=0
for p in "${pids[@]}"; do wait $p || fails=$((fails + 1)); done
kill "${spids[@]}" 2>/dev/null
log "DONE: $(cat $D/rows/*.jsonl | wc -l) stances, $(cat $D/rows/*.jsonl | grep -o '"stage": "jref"' | wc -l) samples tried, $fails failed processes"
