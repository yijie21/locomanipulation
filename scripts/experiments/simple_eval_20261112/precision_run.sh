#!/usr/bin/env bash
# precision_run.sh <out dir> <scalebfm|sonic> <const json> "<repeats>" <first episode> <last episode> "<GPUs>":
# selfmodel_precision_20261124, one controller. Each repeat k gets its own bench dir <out>/r<k>, with the stance jittered by
# simple_bench.py --stance-jitter $JIT --jitter-rep k (the same draw for every arm), and runs:
#   bench_via.sh A, plan, JR (J filmed: the state at the command); s_check.py state; precision.py shifts (the five arms'
#   goals); plan2.py for each arm and routed episode (the replan arm with another IK seed); precision.py pick (J's own plan
#   where an arm's replan failed); bench_via.sh S for each arm on the routed episodes only.
# Repeats run side by side, $PAR at a time, repeat k on GPU k mod the list; within one, $ARM_PAR arms at a time (SONIC
# takes ~1.8 GB a process on a short run); the five arms' replans $PLAN_PAR at a time. Env as bench_via.sh: BENCH_REPO, SIMPLE_ROOT, SIMPLE_EVAL_OUT, SIMPLE_LEVELS,
# BENCH_SET, BENCH_TASKS, REACH_SONIC_NETS; JIT (m, default 0.06); PAR (default 3); MODEL (default the jrefwm model);
# CONST_LOO=1: the constant arm leaves out the const file's episodes of the episode's own layout (the development runs).
# Progress: <out>/run.log.
# NEW_ARMS=1 (selfmodel_precision2_20261125): the arms kin, kinc, modelf on repeats this script already ran -- A, plan, JR,
# s_check.py state and the five arms are not rerun, nothing of theirs is rewritten. <const json> is then the tracking
# constant (precision.py ctrack); precision.py shifts2 gives the goals, the same replans and scale retries, pick to
# pick2.jsonl, bench_via.sh S for each new arm. Code copies in <out>/asrun2/, progress in <out>/run2.log, <out>/table2.txt.
# NEW_ARMS=3 (selfmodel_precision3_20261125): the arms kin_rr, kinc_rr on repeats NEW_ARMS=1 already ran, the same way;
# <const json> is then c_rr (precision.py ctrack_rr), shifts2 --arms kin_rr,kinc_rr to shifts3.jsonl, pick to pick3.jsonl;
# CONST_LOO=1 leaves c_rr's episodes of the episode's own position out (by name). <out>/asrun3/, run3.log, table3.txt.
set -uo pipefail
OUT=$1 CTRL=$2 CONST=$3 REPS=$4 E0=$5 E1=$6
read -ra GPUS <<< "$7"
W=${BENCH_REPO:?}; cd $W
S=${SIMPLE_ROOT:?}; SP=$S/.venv/bin/python; PY=$W/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
E=scripts/experiments/simple_eval_20261112; C=scripts/experiments/curobo_avoid_20261120
O=${SIMPLE_EVAL_OUT:-$W/outputs/simple_eval_20261112}
JIT=${JIT:-0.06} PAR=${PAR:-3} ARM_PAR=${ARM_PAR:-5} PLAN_PAR=${PLAN_PAR:-6}
M=${MODEL:-$W/outputs/simple_eval_20261112/jrefwm_20261124/model_$CTRL}
export PYTHONPATH=src OMP_NUM_THREADS=1 MUJOCO_GL=egl REACH_HAND_OFF=0.0415,0,0 REACHWM_GOAL_REF=1 SIMPLE_ROOT=$S
SETS="--bench-set controller=$CTRL"; for kv in ${BENCH_SET:-}; do SETS="$SETS --bench-set $kv"; done
BED=$($PY -m locomani.bench.simple $SETS) || exit 2; eval "$BED"; CL=$O/$BED_CLOUD
if [ "${NEW_ARMS:-0}" = 1 ]; then
  ARMS_RUN="kin kinc modelf" SHIFTED_RUN="kin kinc modelf" SHF=shifts2.jsonl PKF=pick2.jsonl ASR=asrun2 TBL=table2 RT=retry2 L=$OUT/run2.log
elif [ "${NEW_ARMS:-0}" = 3 ]; then
  ARMS_RUN="kin_rr kinc_rr" SHIFTED_RUN="kin_rr kinc_rr" SHF=shifts3.jsonl PKF=pick3.jsonl ASR=asrun3 TBL=table3 RT=retry3 L=$OUT/run3.log
else
  ARMS_RUN="replan const model sham oracle" SHIFTED_RUN="const model sham oracle" SHF=shifts.jsonl PKF=pick.jsonl ASR=asrun TBL=table RT=retry L=$OUT/run.log
fi
mkdir -p $OUT/$ASR; cp $E/precision.py $E/precision_run.sh $E/s_check.py $E/bench_via.sh $E/simple_bench.py $C/plan2.py $CONST $OUT/$ASR/
log() { echo "[$(date -u +%T)] $*" >> $L; }
log "start $CTRL arms '$ARMS_RUN' reps '$REPS' episodes $E0-$E1 GPUs ${GPUS[*]} par $PAR arm_par $ARM_PAR jitter $JIT model $M const $(cat $CONST | head -c 200); commit $(git log -1 --format=%h) + uncommitted (asrun/)"

one() {  # one <repeat>
  local k=$1 B=$OUT/r$1 P=$OUT/r$1/P GPU=${GPUS[$(($1 % ${#GPUS[@]}))]}
  mkdir -p $P
  export BENCH_EXTRA="--stance-jitter $JIT --jitter-rep $k"  # every arm of this repeat, the five below too
  if [ "${NEW_ARMS:-0}" = 1 ]; then
    [ -s $P/shifts.jsonl ] && [ -s $P/pick.jsonl ] && [ ! -e $P/shifts2.jsonl ] || { log "r$k FAIL: no earlier run here, or shifts2 exists"; return 2; }
    CUDA_VISIBLE_DEVICES=$GPU $PY -I $E/precision.py shifts2 $P $M $B --ctrack $CONST --rep $k ${CONST_LOO:+--loo} > $P/logs_shifts2.txt 2>&1 || { log "r$k FAIL shifts2"; return 2; }
  elif [ "${NEW_ARMS:-0}" = 3 ]; then
    [ -s $P/shifts.jsonl ] && [ -s $P/pick.jsonl ] && [ -s $P/shifts2.jsonl ] && [ -s $P/pick2.jsonl ] && [ ! -e $P/shifts3.jsonl ] && [ ! -e $P/kin_rr ] && [ ! -e $P/kinc_rr ] \
      || { log "r$k FAIL: no earlier runs here, or shifts3 / the arms' dirs exist"; return 2; }
    CUDA_VISIBLE_DEVICES=$GPU $PY -I $E/precision.py shifts2 $P $M $B --arms kin_rr,kinc_rr --ctrack-rr $CONST --out shifts3.jsonl --rep $k ${CONST_LOO:+--loo} > $P/logs_shifts3.txt 2>&1 || { log "r$k FAIL shifts3"; return 2; }
  else
    for s in A plan JR; do BENCH_GPU=$GPU BENCH_GPUS=$GPU bash $E/bench_via.sh $B $CTRL none $s $E0 $E1; done
    log "r$k: A, plan, JR: $(grep -c . $B/rows/JR_*.jsonl | paste -sd' ') rows; $(tail -1 $B/pipeline.log)"
    CUDA_VISIBLE_DEVICES=$GPU $PY -I $E/s_check.py state $B $P > $P/logs_state.txt 2>&1 || { log "r$k FAIL state"; return 2; }
    CUDA_VISIBLE_DEVICES=$GPU $PY -I $E/precision.py shifts $P $M $B --const $CONST --rep $k ${CONST_LOO:+--loo} > $P/logs_shifts.txt 2>&1 || { log "r$k FAIL shifts"; return 2; }
  fi
  log "r$k: $(wc -l < $P/$SHF) routed episodes"
  # every arm's replans at once, $PLAN_PAR planners (~0.5 GB each); one row file an arm, scale and episode (no shared
  # appends). A shifted goal that does not plan is tried again at 2/3, then 1/3 of the shift (precision.py SCALES): only the
  # episodes that did not plan go to the next scale; replan has no shift and plans once
  plan_scale() {  # plan_scale <scale dir> < lines "arm task ep act seed"
    local d=$1
    CUDA_VISIBLE_DEVICES=$GPU xargs -r -P $PLAN_PAR -L1 sh -c "cd $S && $SP -I $W/$C/plan2.py $S $P/\$0/step1 $P/\$0/$d \$2 --task \$1 --cloud-dir $CL --act \$3 --ik-seed \$4 2>/dev/null | grep '^{' > $P/\$0/$d/plan_row_\$2.json"
  }
  not_planned() {  # not_planned <scale dir>: the shifted arms' episodes with no plan at that scale
    python3 - "$P" "$1" "$SHF" "$SHIFTED_RUN" <<'PY'
import json, os, sys
P, d = sys.argv[1], sys.argv[2]
for l in open(os.path.join(P, sys.argv[3])):
    x = json.loads(l)
    for arm in sys.argv[4].split():
        f = os.path.join(P, arm, d, "plan_row_%d.json" % x["ep"])
        ok = os.path.exists(f) and os.path.getsize(f) > 0 and bool(json.loads(open(f).read().splitlines()[0]).get("ok"))
        if not ok:
            print(arm, x["task"], x["ep"], x["act"], 1531)
PY
  }
  python3 -c "
import json
for arm in '$ARMS_RUN'.split():
    for l in open('$P/$SHF'):
        x = json.loads(l)
        print(arm, x['task'], x['ep'], x['act'], 7 if arm == 'replan' else 1531)
" | plan_scale step2
  local R2=$P/${RT}_s2of3.txt R1=$P/${RT}_s1of3.txt
  not_planned step2 > $R2
  plan_scale step2_s2of3 < $R2
  not_planned step2_s2of3 | grep -Fxf $R2 > $R1
  plan_scale step2_s1of3 < $R1
  for arm in $ARMS_RUN; do
    cat $P/$arm/step2*/plan_row_*.json > $P/plan_$arm.jsonl 2>/dev/null
    log "r$k $arm: planned $(cat $P/$arm/step2/plan_row_*.json 2>/dev/null | grep -c '"ok": true') of $(cat $P/$arm/step2/plan_row_*.json 2>/dev/null | wc -l) at the full shift; at 2/3 $(cat $P/$arm/step2_s2of3/plan_row_*.json 2>/dev/null | grep -c '"ok": true') of $(grep -c "^$arm " $R2); at 1/3 $(cat $P/$arm/step2_s1of3/plan_row_*.json 2>/dev/null | grep -c '"ok": true') of $(grep -c "^$arm " $R1)"
  done
  CUDA_VISIBLE_DEVICES=$GPU $PY -I $E/precision.py pick $P $B --arms ${ARMS_RUN// /,} --shifts $SHF --out $PKF > $P/logs_${PKF%.jsonl}.txt 2>&1 || { log "r$k FAIL pick"; return 2; }
  local eps=$(python3 -c "import json; print(','.join(str(json.loads(l)['ep']) for l in open('$P/$SHF')))")
  [ -n "$eps" ] || { log "r$k: no routed episodes"; return 0; }
  local n=0
  for arm in $ARMS_RUN; do
    # --film-ref only records (the motor's reference in the film; JR equals J on 240 / 240): the check that each arm sent its own plan
    BENCH_EXTRA="$BENCH_EXTRA --film-ref" S_DIR=$P/$arm S_ARM=$arm BENCH_EPISODES=$eps BENCH_GPU=$GPU BENCH_GPUS=$GPU bash $E/bench_via.sh $B $CTRL none S $E0 $E1 &
    n=$((n + 1)); if [ $n -ge $ARM_PAR ]; then wait -n; n=$((n - 1)); fi
  done
  wait
  log "r$k: arms done; $(for arm in $ARMS_RUN; do echo -n "$arm $(cat $B/rows/${arm}_[A-Z]*.jsonl 2>/dev/null | wc -l) "; done)"
}

running=0
for k in $REPS; do
  one $k &
  running=$((running + 1))
  if [ $running -ge $PAR ]; then wait -n; running=$((running - 1)); fi
done
wait
$PY -I $E/precision.py $TBL $OUT > $OUT/$TBL.txt 2>&1
log "all done; $(head -3 $OUT/$TBL.txt | tail -1)"
