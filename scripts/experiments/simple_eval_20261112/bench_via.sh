#!/usr/bin/env bash
# notebook: selfmodel_bench_20261123 -- SIMPLE's four pick tasks, perception column (the Isaac look), on vastgpu, three ways:
#   A  a straight reach (the bench as it was); --film --trace, which also gives each reach's start for the planner
#   B  the pregrasp reach along a path planned around the look's points (curobo_avoid_20261120's steps), sent as it is
#   C  the same path, its via points and goal moved by the controller's multi-waypoint self model (simple_bench.py --comp)
#   D  (control, 2026-11-24) the same path, every via point and the goal 6 cm back and 2 cm up, no model (--shift)
#   E  (control) the same path moved by the model file's constant correction (REACH_VIA_COMP_MODE=const)
#   M  (2026-11-24, TODO 38.10.2) the fixed template on the episodes that have a path: 8 cm back along the heading, up to
#      the pregrasp height, over to it (curobo_avoid_20261120/vias_e1.py's group E, from B's via files and the preps)
#   J  (2026-11-24, TODO 38.10.2) the planned right-arm joint trajectory itself as the reference (simple_bench.py --joint,
#      Motor.reach_joints), the rest of the body held: no IK of ours on the way
#   JR (2026-11-24, TODO 38.10.3) J again with --film-ref: the films carry the motor's reference and the rows the self
#      model's inputs at the command (jref_data.py turns them into the check's dataset); the rows should equal J's
# bench_via.sh <out dir> <scalebfm|sonic> <model dir for C> <step: A|plan|B|C|all|D|E|M|J|JR|S> [first episode] [last episode]
# One process a task (4 at a time); the planner 4 at a time on one GPU. Progress: <out>/pipeline.log.
# Another machine: BENCH_REPO (the checkout, default vastgpu's), SIMPLE_ROOT (default <repo>/third_party/SIMPLE), BENCH_GPU
# (default 1). On the 4090: BENCH_REPO=/wuji-vefps-D/wuji-il/yijie/ws/code/loco
# SIMPLE_ROOT=/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE; SONIC's networks run on the GPU there, so take
# REACH_SONIC_NETS from the environment (default allcpu, as vastgpu's onnxruntime has no CUDA provider).
# Run switches (2026-11-24, TODO 38.11; locomani.bench): <scalebfm|sonic> is --bench-set controller=; BENCH_SET adds
# more, space-separated (e.g. "sensor=rgbd" or "target=truth"); the look and the planner's cloud follow the sensor
# (python -m locomani.bench.simple). Default: configs/bench/simple.yaml, since 2026-11-24 the RGB-D look (the replica's
# MuJoCo head camera, cloud_blind and look_frame.py's look_dump_blind); the runs above were on the stereo look, so
# BENCH_SET=sensor=stereo repeats them.
set -uo pipefail
D=$1 CTRL=$2 MODEL=$3 STEP=$4 E0=${5:-0} E1=${6:-29}
W=${BENCH_REPO:-/workspace/code/locomanipulation-selfwm}; cd $W
S=${SIMPLE_ROOT:-$W/third_party/SIMPLE}; SP=$S/.venv/bin/python; PY=$W/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
G=${BENCH_GPU:-1}
E=scripts/experiments/simple_eval_20261112; C=scripts/experiments/curobo_avoid_20261120
# SIMPLE_EVAL_OUT (2026-11-24): another tree of episodes and cached looks (the development episodes of selfmodel_s_20261124);
# BENCH_TASKS: fewer tasks. Defaults: the test tree, the four pick tasks
O=${SIMPLE_EVAL_OUT:-$W/outputs/simple_eval_20261112}
TASKS=${BENCH_TASKS:-"XMovePickTeleop XMoveBendPickTeleop BendPickMP TabletopGraspMP"}
export REACH_HAND_OFF=0.0415,0,0 REACHWM_GOAL_REF=1 PYTHONPATH=src OMP_NUM_THREADS=1 MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=$G CUDA_VISIBLE_DEVICES=$G SIMPLE_ROOT=$S
[ "$CTRL" = sonic ] && export REACH_SONIC_NETS=${REACH_SONIC_NETS:-allcpu}
SETS="--bench-set controller=$CTRL"; for kv in ${BENCH_SET:-}; do SETS="$SETS --bench-set $kv"; done
BED=$($PY -m locomani.bench.simple $SETS) || exit 2; eval "$BED"
CL=$O/$BED_CLOUD; LOOK=$O/$BED_LOOK
mkdir -p $D/rows $D/rows_ep $D/logs $D/step1 $D/step2 $D/try_act02 $D/via $D/asrun
log() { echo "[$(date -u +%T)] $*" >> $D/pipeline.log; }
ARGS="$SETS --trace --episodes $E0-$E1"

# BENCH_GPUS (2026-11-24, the 4090): the GPUs the four tasks' processes go to in turn, e.g. "2 3" -- SONIC with its
# networks on the GPU takes ~7.4 GB a process, so four do not fit one 24 GB card. Default: all four on BENCH_GPU.
read -ra GS <<< "${BENCH_GPUS:-$G}"
bench() {  # bench <arm> <extra args...>: the four tasks side by side
  local arm=$1; shift
  local pids=() i=0 g
  for T in $TASKS; do
    g=${GS[$((i % ${#GS[@]}))]}; i=$((i + 1))
    rm -f $D/rows/${arm}_$T.jsonl
    MUJOCO_EGL_DEVICE_ID=$g CUDA_VISIBLE_DEVICES=$g nice -n 10 $PY $E/simple_bench.py $T --method direct $ARGS \
      --film $D/films_$arm "$@" --out $D/rows/${arm}_$T.jsonl > $D/logs/${arm}_$T.log 2>&1 & pids+=($!)
  done
  local fails=0
  for p in "${pids[@]}"; do wait $p || fails=$((fails + 1)); done
  log "arm $arm (GPUs ${GS[*]}): $(cat $D/rows/${arm}_*.jsonl | wc -l) rows, $fails failed processes"
}

if [ "$STEP" = A ] || [ "$STEP" = all ]; then
  cp $E/simple_bench.py $E/bench_via.sh $C/obstacles_box.py $C/prep2.py $C/plan2.py $C/via2.py \
     scripts/experiments/selfmodel_via_20261120/via_comp.py $D/asrun/
  log "start: $BED_SIGNATURE, episodes $E0-$E1, model $MODEL, commit $(git log -1 --format=%h) + uncommitted (copies in asrun/)"
  bench A
  $PY - <<PYEOF
import json, glob
for f in glob.glob("$D/rows/A_*.jsonl"):
    task = f.split("/A_")[1][:-6]
    for l in open(f):
        r = json.loads(l)
        open(f"$D/rows_ep/A_{task}_{r['ep']}.jsonl", "w").write(l)
PYEOF
fi

if [ "$STEP" = plan ] || [ "$STEP" = all ]; then
  [ -d $LOOK ] || { log "FAIL plan: no look dump $LOOK for $BED_SIGNATURE (stereo: look_all.sh makes it; rgbd: look_frame.py)"; exit 2; }
  for T in $TASKS; do for e in $(seq $E0 $E1); do echo "$T $e"; done; done |
    xargs -P 4 -L1 sh -c "$PY -I $C/obstacles_box.py $LOOK $D/films_A $CL $D/step1 \$0 \$1 2>/dev/null | grep '^{'" > $D/step1_rows.jsonl
  log "step 1: $(wc -l < $D/step1_rows.jsonl) rows, $(ls $D/step1/obst_*.npz 2>/dev/null | wc -l) obstacle files"
  for T in $TASKS; do
    seq $E0 $E1 | xargs -P 4 -I{} sh -c "TASK=$T TRACE_FMT='A_{task}_{e}.jsonl' $PY -I $C/prep2.py $D/films_A $D/rows_ep $D/step1 $D/step2 {} 2>&1 | grep -v Warn"
  done > $D/logs/prep.txt
  log "step 2a: $(ls $D/step2/prep_*.npz 2>/dev/null | wc -l) prep files"
  for T in $TASKS; do
    seq $E0 $E1 | xargs -P 4 -I@ sh -c "cd $S && $SP -I $W/$C/plan2.py $S $D/step1 $D/step2 @ --task $T --cloud-dir $CL --diag 2>/dev/null | grep '^{'"
  done > $D/plan_rows.jsonl
  log "step 2b: $(wc -l < $D/plan_rows.jsonl) rows, planned $(grep -c '"ok": true' $D/plan_rows.jsonl)"
  ln -sf $D/step2/prep_*.npz $D/try_act02/ 2>/dev/null
  python3 -c "
import json
for l in open('$D/plan_rows.jsonl'):
    r = json.loads(l)
    if 'skip' not in r and not r['ok']:
        print(r['name'].rsplit('_L', 1)[0], r['ep'])
" | xargs -P 4 -L1 sh -c "cd $S && $SP -I $W/$C/plan2.py $S $D/step1 $D/try_act02 \$1 --task \$0 --cloud-dir $CL --act 0.02 2>/dev/null | grep '^{'" > $D/plan_rows_act02.jsonl
  log "step 2b retry at 2 cm: $(wc -l < $D/plan_rows_act02.jsonl) rows, planned $(grep -c '"ok": true' $D/plan_rows_act02.jsonl)"
  $PY -I $C/via2.py $D/via 0.01 $D/step2 $D/try_act02 > $D/logs/via.txt 2>&1 || log "FAIL via2"
  log "step 2c: $(ls $D/via/via_*.npz 2>/dev/null | wc -l) via files"
fi

if [ "$STEP" = B ] || [ "$STEP" = all ]; then bench B --avoid $D/via; fi
if [ "$STEP" = C ] || [ "$STEP" = all ]; then bench C --avoid $D/via --comp $MODEL; fi
if [ "$STEP" = D ]; then bench D --avoid $D/via --shift=-6,0,2; fi
if [ "$STEP" = E ]; then export REACH_VIA_COMP_MODE=const; bench E --avoid $D/via --comp $MODEL; unset REACH_VIA_COMP_MODE; fi
if [ "$STEP" = M ] || [ "$STEP" = J ] || [ "$STEP" = JR ] || [ "$STEP" = S ]; then
  mkdir -p $D/asrun_$STEP
  cp $E/simple_bench.py $E/bench_via.sh $C/vias_e1.py src/locomani/backends/scalebfm/motor.py $D/asrun_$STEP/
fi
if [ "$STEP" = M ]; then
  $PY -I $C/vias_e1.py $D/via $D/step2 $D/tmpl > $D/logs/tmpl.txt 2>&1 || log "FAIL vias_e1"
  log "template: $(ls $D/tmpl/via_E/via_*.npz 2>/dev/null | wc -l) via files"
  bench M --avoid $D/tmpl/via_E
fi
if [ "$STEP" = J ]; then bench J --avoid $D/via --joint; fi
if [ "$STEP" = JR ]; then bench JR --avoid $D/via --joint --film-ref; fi
# S (2026-11-24, selfmodel_s_20261124, TODO 38.10.4): J's paths as the self model's check chose them (s_run.sh, s_check.py
# choose -> $S_DIR/via); an episode it gave up on has no via file (scored a failure, s_check.py table); S_ARM names the rows
if [ "$STEP" = S ]; then bench ${S_ARM:-S} --avoid $S_DIR/via --joint; fi
log "DONE $STEP"
