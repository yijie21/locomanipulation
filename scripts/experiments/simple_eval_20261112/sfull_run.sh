#!/usr/bin/env bash
# sfull_run.sh <out dir> <scalebfm|sonic> <source bench dir> <const json> <ctrack json> <ctrack_rr json> "<GPUs>":
# selfmodel_sfull_20261125, one controller. S-full (SF: the pregrasp goal moved by -(g + p), precision2's modelf, one round,
# no gate), const, kinc and (the revision of 11-25, selfmodel_precision3_20261125's) kin_rr (-er) and kinc_rr (-(er + c_rr))
# on the episodes J ran a plan on in <source bench dir> (its via/, films_JR/, rows/JR_*), without the stance jitter: J's own
# code path (simple_bench.py's --stance-jitter left at its default 0), --film-ref only records. Steps:
#   s_check.py state from JR's films; precision.py shifts2 --arms SF,const,kinc,kin_rr,kinc_rr; plan2.py per arm and routed episode, the
#   failed ones again at 2/3 and 1/3 of the shift; precision.py pick (J's own plan where none planned); bench_via.sh S per
#   arm and task on the routed episodes; jref_data.py build of JR (the model's error, metric 6) and range; table_sfull.
# <out dir> is new: the source bench's step1, step2, try_act02, via and films_JR are linked (read only), JR's rows copied
# (ONLY="30,48": those episode numbers only, the trial); nothing is written into the source. Env as bench_via.sh
# (BENCH_REPO, SIMPLE_ROOT, SIMPLE_EVAL_OUT, SIMPLE_LEVELS, BENCH_SET, REACH_SONIC_NETS); SF_BENCH_EXTRA (2026-11-26,
# simple_relook_20261126): more simple_bench.py options for the arms' runs (e.g. --relook); CONST_LOO=1 leaves the
# const and c_t files' episodes of the episode's own layout out (development episodes; by goal distance, as before) and
# c_rr's of the episode's own position (by name); J_ARM (default J) and S_CHOICE
# (the old S's choice.jsonl) for the table; ARM_PAR (default 3), PLAN_PAR (default 6). Progress: <out>/run.log.
set -uo pipefail
D=$1 CTRL=$2 SRC=$3 CONST=$4 CTRACK=$5 CTRACK_RR=$6
read -ra GPUS <<< "$7"
W=${BENCH_REPO:?}; cd $W
S=${SIMPLE_ROOT:?}; SP=$S/.venv/bin/python; PY=$W/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
E=scripts/experiments/simple_eval_20261112; C=scripts/experiments/curobo_avoid_20261120
O=${SIMPLE_EVAL_OUT:-$W/outputs/simple_eval_20261112}
ARM_PAR=${ARM_PAR:-3} PLAN_PAR=${PLAN_PAR:-6} GPU=${GPUS[0]}
M=${MODEL:-$W/outputs/simple_eval_20261112/jrefwm_20261124/model_$CTRL}
DATA=${DATA:-$W/outputs/simple_eval_20261112/jrefwm_20261124/data_$CTRL}
export PYTHONPATH=src OMP_NUM_THREADS=1 MUJOCO_GL=egl REACH_HAND_OFF=0.0415,0,0 REACHWM_GOAL_REF=1 SIMPLE_ROOT=$S
SETS="--bench-set controller=$CTRL"; for kv in ${BENCH_SET:-}; do SETS="$SETS --bench-set $kv"; done
BED=$($PY -m locomani.bench.simple $SETS) || exit 2; eval "$BED"; CL=$O/$BED_CLOUD
ARMS=${SF_ARMS:-"SF const kinc kin_rr kinc_rr"}  # SF_ARMS (2026-11-26, simple_relook_20261126): fewer arms
[ -e $D ] && { echo "$D exists: a run was made there"; exit 2; }
for f in $CONST $CTRACK $CTRACK_RR; do [ -s $f ] || { echo "no $f"; exit 2; }; done
SRC=$(realpath $SRC); P=$D/P
mkdir -p $D/rows $P $D/asrun_sf
for x in step1 step2 try_act02 via films_JR; do [ -e $SRC/$x ] && ln -s $SRC/$x $D/$x; done
python3 - "$SRC" "$D" "${ONLY:-}" <<'PY'
import glob, json, os, sys
src, d, only = sys.argv[1], sys.argv[2], {int(x) for x in sys.argv[3].split(",") if x}
for f in sorted(glob.glob(src + "/rows/JR_*.jsonl")):
    rows = [l for l in open(f) if not only or json.loads(l)["ep"] in only]
    open(d + "/rows/" + os.path.basename(f), "w").writelines(rows)
PY
cp $E/precision.py $E/sfull_run.sh $E/s_check.py $E/jref_data.py $E/bench_via.sh $E/simple_bench.py $C/plan2.py $CONST $CTRACK $CTRACK_RR $D/asrun_sf/
L=$D/run.log
log() { echo "[$(date -u +%T)] $*" >> $L; }
log "start $CTRL arms '$ARMS' source $SRC (JR rows $(cat $D/rows/JR_*.jsonl | wc -l)${ONLY:+, only $ONLY}) GPUs ${GPUS[*]} model $M const $CONST ctrack $CTRACK ctrack_rr $CTRACK_RR loo ${CONST_LOO:-0}; commit $(git log -1 --format=%h) + uncommitted (asrun_sf/)"

CUDA_VISIBLE_DEVICES=$GPU $PY -I $E/s_check.py state $D $P > $P/logs_state.txt 2>&1 || { log "FAIL state"; exit 2; }
CUDA_VISIBLE_DEVICES=$GPU $PY -I $E/precision.py shifts2 $P $M $D --arms ${ARMS// /,} --const $CONST --ctrack $CTRACK \
  --ctrack-rr $CTRACK_RR --out shifts_sf.jsonl ${CONST_LOO:+--loo} > $P/logs_shifts_sf.txt 2>&1 || { log "FAIL shifts"; exit 2; }
log "$(wc -l < $P/shifts_sf.jsonl) routed episodes; the command state rebuilt against the film, max cm: $(python3 -c "import json; print(max(json.loads(l)['rebuilt_vs_film_cm_max'] for l in open('$P/state.jsonl')))")"

# the replans, $PLAN_PAR at a time; one row file an arm, scale, task and episode (episode numbers repeat across tasks);
# a shift that does not plan is tried again at 2/3, then 1/3 (precision.py SCALES), only the episodes that did not plan
plan_scale() {  # plan_scale <scale dir> < lines "arm task ep act seed"
  local d=$1
  CUDA_VISIBLE_DEVICES=$GPU xargs -r -P $PLAN_PAR -L1 sh -c "cd $S && $SP -I $W/$C/plan2.py $S $P/\$0/step1 $P/\$0/$d \$2 --task \$1 --cloud-dir $CL --act \$3 --ik-seed \$4 2>/dev/null | grep '^{' > $P/\$0/$d/plan_row_\$1_\$2.json"
}
not_planned() {  # not_planned <scale dir>: the arms' episodes with no plan at that scale
  python3 - "$P" "$1" "$ARMS" <<'PY'
import json, os, sys
P, d = sys.argv[1], sys.argv[2]
for l in open(os.path.join(P, "shifts_sf.jsonl")):
    x = json.loads(l)
    for arm in sys.argv[3].split():
        f = os.path.join(P, arm, d, "plan_row_%s_%d.json" % (x["task"], x["ep"]))
        ok = os.path.exists(f) and os.path.getsize(f) > 0 and bool(json.loads(open(f).read().splitlines()[0]).get("ok"))
        if not ok:
            print(arm, x["task"], x["ep"], x["act"], 1531)
PY
}
python3 -c "
import json
for arm in '$ARMS'.split():
    for l in open('$P/shifts_sf.jsonl'):
        x = json.loads(l)
        print(arm, x['task'], x['ep'], x['act'], 1531)
" | plan_scale step2
not_planned step2 > $P/retry_s2of3.txt
plan_scale step2_s2of3 < $P/retry_s2of3.txt
not_planned step2_s2of3 | grep -Fxf $P/retry_s2of3.txt > $P/retry_s1of3.txt
plan_scale step2_s1of3 < $P/retry_s1of3.txt
for arm in $ARMS; do
  cat $P/$arm/step2*/plan_row_*.json > $P/plan_$arm.jsonl 2>/dev/null
  log "$arm: planned $(cat $P/$arm/step2/plan_row_*.json 2>/dev/null | grep -c '"ok": true') of $(cat $P/$arm/step2/plan_row_*.json 2>/dev/null | wc -l) at the full shift; at 2/3 $(cat $P/$arm/step2_s2of3/plan_row_*.json 2>/dev/null | grep -c '"ok": true') of $(grep -c "^$arm " $P/retry_s2of3.txt); at 1/3 $(cat $P/$arm/step2_s1of3/plan_row_*.json 2>/dev/null | grep -c '"ok": true') of $(grep -c "^$arm " $P/retry_s1of3.txt)"
done
CUDA_VISIBLE_DEVICES=$GPU $PY -I $E/precision.py pick $P $D --arms ${ARMS// /,} --shifts shifts_sf.jsonl --out pick_sf.jsonl \
  --row-fmt 'plan_row_{task}_{ep}.json' > $P/logs_pick_sf.txt 2>&1 || { log "FAIL pick"; exit 2; }

# each arm's routed episodes, task by task (bench_via.sh runs one process a task); arms $ARM_PAR at a time over the GPUs
TASKS_EPS=$(python3 -c "
import json, collections
t = collections.defaultdict(list)
for l in open('$P/shifts_sf.jsonl'):
    x = json.loads(l); t[x['task']].append(str(x['ep']))
print(' '.join(k + ':' + ','.join(v) for k, v in t.items()))")
log "running $TASKS_EPS"
run_arm() {  # run_arm <arm> <GPU>
  local arm=$1 g=$2 te
  for te in $TASKS_EPS; do
    BENCH_EXTRA="--film-ref ${SF_BENCH_EXTRA:-}" BENCH_TASKS=${te%%:*} BENCH_EPISODES=${te#*:} S_DIR=$P/$arm S_ARM=$arm BENCH_GPU=$g BENCH_GPUS=$g \
      bash $E/bench_via.sh $D $CTRL none S 0 29
  done
}
i=0 n=0
for arm in $ARMS; do
  run_arm $arm ${GPUS[$((i % ${#GPUS[@]}))]} & i=$((i + 1)) n=$((n + 1))
  if [ $n -ge $ARM_PAR ]; then wait -n; n=$((n - 1)); fi
done
wait
log "arms done; $(for arm in $ARMS; do echo -n "$arm $(cat $D/rows/${arm}_[A-Z]*.jsonl 2>/dev/null | wc -l) "; done)"

CUDA_VISIBLE_DEVICES=$GPU $PY -I $E/jref_data.py build $D/jref_check outputs/scene/simple_g1_sonic $D/films_JR $D/rows/JR_*.jsonl > $D/logs_jref_check.txt 2>&1 || log "FAIL jref_data build"
$PY -I $E/jref_data.py range $D/jref_check $DATA/*/ > $D/range.txt 2>&1 || log "FAIL jref_data range"
$PY -I $E/precision.py table_sfull $D --bench $SRC --j-arm ${J_ARM:-J} ${S_CHOICE:+--s-choice $S_CHOICE} --check $D/jref_check \
  --range $D/range.txt > $D/table_sfull.txt 2>&1 || log "FAIL table_sfull"
log "all done; $(grep -m1 '^  all' $D/table_sfull.txt | cut -c1-150)"
