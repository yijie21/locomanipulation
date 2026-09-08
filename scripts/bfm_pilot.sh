#!/bin/bash
# ScaleBFM overfitting pilot: collect -> stats -> train -> in-process closed loop.
#
#   GRIP_KD=0.5 bash scripts/bfm_pilot.sh <out_dir> [successes=200] [holdout=10] [steps=3000] [eval_rounds=60]
#
# Fixed conditions (run_bottle.py v3 defaults, no appearance/scene randomization), only
# successful rounds kept. Collection + eval on GPU1 (in-process, no stack); training on GPU0.
# Stage markers (*.done) make a rerun resume at the first incomplete stage.
set -eu
cd "$(dirname "$0")/.."
OUT=${1:?out dir}; N=${2:-200}; HOLD=${3:-10}; STEPS=${4:-3000}; ROUNDS=${5:-60}
SIM_PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
GR_PY=third_party/Isaac-GR00T/.venv/bin/python
export HF_HOME=/workspace/huggingface_cache MUJOCO_GL=egl PYTHONUNBUFFERED=1
mkdir -p "$OUT"
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/pilot.log"; }

if [ ! -f "$OUT/collect.done" ]; then
  log "collect: $N successes (+$HOLD holdout), GRIP_KD=${GRIP_KD:-1.0}, seed ${SEED:-1}"
  rm -rf "$OUT/train_dataset" "$OUT/train_dataset_val"
  CUDA_VISIBLE_DEVICES=1 GRIP_KD=${GRIP_KD:-1.0} $SIM_PY -u src/teleop_free/scalebfm/record_bottle.py \
    --successes $((N + HOLD)) --holdout "$HOLD" --seed "${SEED:-1}" --out "$OUT/train_dataset" > "$OUT/collect.log" 2>&1
  for d in train_dataset train_dataset_val; do
    third_party/GR00T-WholeBodyControl/.venv_data_collection/bin/python src/teleop_free/tools/check_dataset.py "$OUT/$d" > "$OUT/audit_$d.log" 2>&1 || { log "audit FAILED on $d"; exit 1; }
    $GR_PY -m gr00t.data.stats --dataset-path "$OUT/$d" --embodiment-tag UNITREE_G1_SONIC --modality-config-path configs/gr00t_bfm_2cam.py > "$OUT/stats_$d.log" 2>&1
  done
  tail -1 "$OUT/collect.log" | tee -a "$OUT/pilot.log"; touch "$OUT/collect.done"
fi

if [ ! -f "$OUT/train.done" ]; then
  log "train: $STEPS steps on GPU0"
  rm -rf "$OUT/model"
  MODALITY=$PWD/configs/gr00t_bfm_2cam.py DATA=$PWD/$OUT/train_dataset RUN=model OUT_ROOT=$PWD/$OUT \
    MAX_STEPS=$STEPS SAVE_STEPS=$STEPS SAVE_LIMIT=1 WANDB=0 WORKERS=2 GPUS=1 SINGLE_GPU=1 CUDA_VISIBLE_DEVICES=0 SQUAT_MAX=4096 \
    bash scripts/train_bottle.sh > "$OUT/train.log" 2>&1
  [ -d "$OUT/model/checkpoint-$STEPS" ] || { log "no checkpoint"; exit 1; }
  touch "$OUT/train.done"
fi

if [ ! -f "$OUT/openloop.done" ]; then
  log "open-loop on val"
  CUDA_VISIBLE_DEVICES=1 $GR_PY -u src/teleop_free/tools/eval_open_loop.py --model-path "$OUT/model/checkpoint-$STEPS" \
    --dataset "$OUT/train_dataset_val" --modality-config configs/gr00t_bfm_2cam.py --horizons 8 16 --out "$OUT/open_loop_val.json" > "$OUT/open_loop_val.log" 2>&1 || log "open-loop failed (non-fatal)"
  touch "$OUT/openloop.done"
fi

if [ ! -f "$OUT/eval.done" ]; then
  log "eval: $ROUNDS rounds in-process on GPU1"
  rm -rf "$OUT/eval"
  CUDA_VISIBLE_DEVICES=1 GRIP_KD=${GRIP_KD:-1.0} $GR_PY -u src/teleop_free/scalebfm/eval_bfm_bottle.py --model-path "$OUT/model/checkpoint-$STEPS" \
    --rounds "$ROUNDS" --seed 6000 --policy-seed 1 --video --out "$OUT/eval" > "$OUT/eval.log" 2>&1
  tail -1 "$OUT/eval.log" | tee -a "$OUT/pilot.log"; touch "$OUT/eval.done"
fi
log "pilot complete"
