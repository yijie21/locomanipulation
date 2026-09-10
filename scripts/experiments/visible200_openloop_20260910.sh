#!/bin/bash
# Open-loop eval of visible200's checkpoint-10000 on its own val split, to separate
# "the model did not learn the mapping" from "the closed-loop stop point lands off the
# demonstrated band". The metric that matters here is close_frame_err_abs (memory
# creep-hidden-stop-target: it was the ONLY offline metric that caught cover400's absorbing state).
set -u
cd /workspace/code/locomanipulation
E=outputs/experiments/visible200_20260910
CUDA_VISIBLE_DEVICES=0 third_party/Isaac-GR00T/.venv/bin/python -u src/locomani/pipeline/eval_open_loop.py \
    --model-path "$E/model/checkpoint-10000" \
    --dataset "$E/val_dataset" \
    --out "$E/open_loop" > "$E/open_loop.log" 2>&1
echo "open_loop exit=$?" >> "$E/open_loop.log"
