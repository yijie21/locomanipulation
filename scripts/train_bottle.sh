#!/bin/bash
# Finetune GR00T-N1.7-3B on the bottle dataset, slot 11 (unitree_g1_sonic).
#
#   bash scripts/train_bottle.sh [extra launch_finetune args...]
#
# Every choice below is a decision with a reason; the reasons live here rather
# than in a design doc that drifts.

set -eu

# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
# The venv lives with the upstream clone (an environment is not code); the
# gr00t PACKAGE it imports is the vendored copy in src/ (src/gr00t/UPSTREAM.md,
# wired by scripts/link_gr00t_src.sh). The check below is not optional
# paranoia: the failure mode is a stale third_party copy imported under the
# same name and version, which trains to completion on the wrong code.
GR00T="$LOCOMANI_ROOT/third_party/Isaac-GR00T"
PY="$GR00T/.venv/bin/python"
GR00T_SRC=$("$PY" -c "import gr00t; print(gr00t.__file__)")
case "$GR00T_SRC" in
    "$LOCOMANI_ROOT/src/gr00t/"*) : ;;
    *) echo "gr00t resolves to $GR00T_SRC, not src/gr00t --"
       echo "run: bash scripts/link_gr00t_src.sh"; exit 1 ;;
esac
export HF_HOME="${HF_HOME:-$LOCOMANI_WORKSPACE/huggingface_cache}"

DATA="${DATA:-$LOCOMANI_ROOT/datasets/g1_locomani_bottle200_train}"
# RUN names the run in BOTH places at once, which is not cosmetic: experiment.py
# rebuilds output_dir as <output-dir>/<experiment-name> whenever
# --experiment-name is given (experiment.py:213), so passing a leaf directory
# AND a name would bury the checkpoints one level deeper than the caller asked
# for. Passing the PARENT as --output-dir and the leaf as --experiment-name
# makes the two agree: checkpoints in outputs/train/$RUN, wandb run "$RUN".
RUN="${RUN:-bottle200}"
OUT_ROOT="${OUT_ROOT:-$LOCOMANI_OUTPUTS/train}"
OUT="$OUT_ROOT/$RUN"
GPUS="${GPUS:-2}"

# --- metrics ------------------------------------------------------------------
# On by default: this run is 10k steps over several hours and the whole point of
# Q10's "pick the checkpoint from the curve, not in advance" is that the curve
# has to be visible while it happens. WANDB=0 turns it off.
# logging_steps is 10 (training_config.py:67) and is not a CLI flag, so the
# train-loss trace comes at 10-step resolution without asking.
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-locomani-bottle}"

# --- the batch, and why it is split this way ---------------------------------
# GLOBAL_BATCH is the OPTIMIZATION batch and must not move: it is what the lr
# was chosen against. GRAD_ACCUM is the MEMORY knob. Adding the second camera
# lengthens the visual token sequence, so if 64 in one go does not fit, raise
# GRAD_ACCUM and lower GLOBAL_BATCH by the same factor -- the product is what
# the optimizer sees. Dropping GLOBAL_BATCH on its own silently changes the
# training run into a different experiment.
GLOBAL_BATCH="${GLOBAL_BATCH:-64}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"

# --- SINGLE_GPU=1: the whole finetune on one 32 GB card ----------------------
# The two-GPU requirement was never compute -- it is Adam: fp32 m+v for 1.62B
# trainable params is 13 GB, and 6.5 (master weights) + 13 + grads + frozen VLM
# + activations ≈ 33 GB > 31.4. ZeRO-2 solved it by SHARDING the optimizer;
# this solves it by QUANTISING it (adamw_bnb_8bit, 13 -> 3.2 GB, total ~22 GB).
# The per-device batch halves and grad accumulation doubles, so the OPTIMIZER
# still sees the same effective 64 -- lr stays valid, the experiment stays the
# same experiment, only slower (~2x: half the FLOPs/step throughput, plus no
# fused optimizer).
OPTIM="${OPTIM:-adamw_torch}"
if [ "${SINGLE_GPU:-0}" = "1" ]; then
    GPUS=1
    OPTIM="adamw_bnb_8bit"
    GLOBAL_BATCH=32
    GRAD_ACCUM=2
fi

# --- how long -----------------------------------------------------------------
# 180 episodes x ~630 frames @30Hz is ~113k samples, so 10k steps at an
# effective batch of 64 is ~5.6 epochs. The number is a budget, not the answer:
# GR00T asserts eval_strategy == "no" (DatasetFactory.build), so there is no
# in-training validation loss and the stopping point has to be chosen AFTER the
# fact by scoring every checkpoint on the held-out split. That is what the two
# save_* settings below are for.
MAX_STEPS="${MAX_STEPS:-10000}"
# save_total_limit defaults to 5, which with save_steps=1000 keeps only steps
# 6000..10000 -- and overfitting on a dataset this narrow shows up in the FIRST
# half. Keeping all ten is 65 GB at ~6.5 GB each; the same ten with optimizer
# state would be 340 GB, and /workspace has ~154 GB free. Hence
# --save-only-model, which is a hard requirement here, not a tidiness choice.
# The cost is that these checkpoints cannot resume training.
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_LIMIT="${SAVE_LIMIT:-10}"

WANDB_ARGS=()
if [ "$WANDB" = "1" ]; then
    # wandb.init() prompts on stdin when it finds no credentials. Under
    # torchrun that prompt has nowhere to go and the job hangs instead of
    # failing, so check here rather than discovering it after the model has
    # loaded. WANDB_MODE=offline is honoured by wandb itself and needs no key.
    if [ "${WANDB_MODE:-}" != "offline" ] \
       && [ -z "${WANDB_API_KEY:-}" ] \
       && ! grep -q "api.wandb.ai" "${HOME}/.netrc" 2>/dev/null; then
        echo "wandb: no credentials (no WANDB_API_KEY, nothing for api.wandb.ai"
        echo "       in ~/.netrc). Run 'wandb login', or set WANDB_MODE=offline,"
        echo "       or WANDB=0 to train without it."
        exit 1
    fi
    WANDB_ARGS=(--use-wandb --wandb-project "$WANDB_PROJECT" --experiment-name "$RUN")
else
    WANDB_ARGS=(--no-use-wandb --experiment-name "$RUN")
fi

[ -d "$DATA" ] || { echo "no dataset at $DATA (set DATA=)"; exit 1; }
[ -f "$DATA/meta/stats.json" ] || { echo "no meta/stats.json in $DATA -- run:
  $PY -m gr00t.data.stats --dataset-path $DATA \\
      --embodiment-tag UNITREE_G1_SONIC \\
      --modality-config-path $LOCOMANI_ROOT/configs/gr00t_sonic_2cam.py"; exit 1; }

echo "dataset      $DATA ($(ls "$DATA/data/chunk-000" | wc -l) episodes)"
echo "output       $OUT"
if [ "$WANDB" = "1" ]; then
    echo "wandb        $WANDB_PROJECT / $RUN${WANDB_MODE:+ (mode $WANDB_MODE)}"
else
    echo "wandb        off"
fi
echo "gpus         $GPUS"
echo "batch        $GLOBAL_BATCH x $GRAD_ACCUM accum = $((GLOBAL_BATCH * GRAD_ACCUM)) effective"
echo "optim        $OPTIM${SINGLE_GPU:+ (SINGLE_GPU=$SINGLE_GPU)}"
echo "steps        $MAX_STEPS, checkpoint every $SAVE_STEPS (keep $SAVE_LIMIT)"
echo

cd "$LOCOMANI_ROOT"
# DRY=1 prints the assembled command instead of running it. Worth having: most
# of what can go wrong here (a stale flag name, a path that resolved to the
# wrong tree, an env override that did not take) is visible in the command line
# and invisible after 20 minutes of model loading.
RUNNER=(exec)
if [ "${DRY:-0}" = "1" ]; then RUNNER=(echo); fi
"${RUNNER[@]}" "$GR00T/.venv/bin/torchrun" --nproc_per_node="$GPUS" --master_port="${PORT:-29500}" \
    -m gr00t.experiment.launch_finetune \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path "$DATA" \
    --embodiment-tag UNITREE_G1_SONIC \
    --modality-config-path "$LOCOMANI_ROOT/configs/gr00t_sonic_2cam.py" \
    --num-gpus "$GPUS" \
    --output-dir "$OUT_ROOT" \
    --max-steps "$MAX_STEPS" \
    --global-batch-size "$GLOBAL_BATCH" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --optim "$OPTIM" \
    --save-steps "$SAVE_STEPS" \
    --save-total-limit "$SAVE_LIMIT" \
    --save-only-model \
    --dataloader-num-workers "${WORKERS:-4}" \
    "${WANDB_ARGS[@]}" \
    `# Slot 11 is EMPTY in nvidia/GR00T-N1.7-3B -- only 24/25/26/27 carry` \
    `# trained projector weights -- so tune-projector is not optional: without` \
    `# it the state/action embedding stays at its 0.02*randn init.` \
    --tune-projector \
    --tune-diffusion-model \
    `# The vision and language towers stay frozen. 200 episodes of ONE scene` \
    `# with a fixed appearance seed is exactly the data that would replace a` \
    `# general visual encoder with a memory of this orange table.` \
    --no-tune-visual \
    --no-tune-llm \
    "$@"
