#!/bin/bash
# One closed-loop evaluation arm on the CLEAN deploy protocol (docs/runbook.md).
#
#   bash scripts/run_eval_arm.sh NAME ROUNDS INSTANCE GPU PORT CKPT ["extra client args"]
#
# e.g. the current baseline:
#   bash scripts/run_eval_arm.sh flow200_base 100 0 0 6590 \
#        outputs/train/flow200/checkpoint-10000
#
# Everything the protocol depends on is set here rather than remembered:
#
#   * the eval SCENE is rebuilt from EVAL_SCENE_SEED (922, table top 0.6214 m)
#     before the stack starts. Collection rebuilds the scene per batch, so a
#     stack started without this inherits whatever table height the last
#     collection left on disk -- three arms were compared across 0.62/0.63/0.65
#     tables before this was caught (docs/evaluation.md 2026-09-04).
#   * CLEAN_DEPLOY=1: the deploy is engaged by SONIC_AUTO_START and never
#     receives a planner command (the async client always installs the clean
#     shim -- docs/lockstep.md '校准诊断'). One planner command anywhere in the
#     process's life leaves it executing walking tokens at ~1/3 amplitude.
#   * NO_ENCODER=1 is correct HERE (the VLA streams external tokens) and wrong
#     for collection, where the planner-mode upper body needs the encoder.
#   * seed 6000 + --spawn-edge 1.1 1.3: same fresh spawns as every other arm,
#     so arms are paired round-by-round (McNemar).
#
# MODE (env, default wall) = wall | lock. `wall` is the token protocol on the
# clean deploy; `lock` is the simulated clock, for diagnosis.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

NAME=$1; N=$2; export INSTANCE=$3; GPU=$4; PORT=$5; CKPT=$6; EXTRA=${7:-}
MODE="${MODE:-wall}"
VENV_SIM_PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
GR00T_PY=third_party/Isaac-GR00T/.venv/bin/python
NVDEF="${SONIC_DEPLOY_NVDEF:-/workspace/deps/gear_sonic_deploy_ours_lockstep_nvdef}"
LAT="$PWD/outputs/eval/latency_flow200_wall.json"
SERVE_EXTRA=""
OUT=outputs/eval/$NAME
[ -d "$OUT" ] && { echo "refusing to overwrite $OUT -- move it or pick a new NAME"; exit 1; }
mkdir -p "$OUT" outputs/videos/cl_$NAME outputs/logs

unset LOCKSTEP CLEAN_DEPLOY SONIC_DEPLOY_BIN LOCKSTEP_DEPLOY_ROOT
case $MODE in
  wall) export CLEAN_DEPLOY=1; CL="--client async" ;;
  lock) export LOCKSTEP=1 LOCKSTEP_DEPLOY_ROOT=$NVDEF; CL="--client lockstep --latency-dist $LAT" ;;
  *)    echo "MODE must be wall|lock"; exit 1 ;;
esac

"$VENV_SIM_PY" src/locomani/sim/build_sim_scene.py outputs/scene outputs/scene/sim \
  --preset bottle_only --seed "${EVAL_SCENE_SEED:-922}" \
  | grep -E "^\[scene\]"

env NO_ENCODER=1 SIM_VIS=readout SIM_VIDEO_DIR="$PWD/outputs/videos/cl_$NAME" \
  bash scripts/stack/start_stack.sh "pick up the water bottle" \
  > outputs/logs/stack_$NAME.log 2>&1
grep -qi "stack up" outputs/logs/stack_$NAME.log \
  || { echo "STACK_FAILED $NAME"; tail -8 outputs/logs/stack_$NAME.log; exit 1; }
echo "STACK_UP $NAME"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU} nohup "$GR00T_PY" \
  src/locomani/pipeline/serve_policy.py --model-path "$CKPT" --port "$PORT" ${SERVE_EXTRA:-} \
  > /workspace/tmp/serve_$NAME.log 2>&1 &
for _ in $(seq 1 90); do
  grep -q "listening" /workspace/tmp/serve_$NAME.log 2>/dev/null && break; sleep 5
done
grep -q "listening" /workspace/tmp/serve_$NAME.log || { echo "SERVE_FAILED $NAME"; exit 1; }
echo "SERVE_UP $NAME"

( . scripts/paths.sh; cd src/locomani/pipeline \
  && TF_DATASET_FPS=25 "$VENV_SIM_PY" -u eval_closed_loop.py \
       --fresh "$N" --seed 6000 --spawn-edge 1.1 1.3 $CL $EXTRA \
       --policy-port "$PORT" \
       --out "$PWD/../../../$OUT/closed_loop.json" \
       --trace-dir "$PWD/../../../$OUT/traces" ) > "$OUT/eval.log" 2>&1

ok=$(grep -c '"ok": true' "$OUT/closed_loop.manifest.jsonl" 2>/dev/null || echo 0)
tr=$(ls "$OUT"/traces/round_*.json 2>/dev/null | wc -l)
echo "ARM_DONE $NAME ok=$ok rounds=$tr"
bash scripts/stack/stop_instance.sh "$INSTANCE" "$PORT" > /dev/null 2>&1
