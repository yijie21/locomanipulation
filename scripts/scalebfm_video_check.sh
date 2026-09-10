#!/usr/bin/env bash
# 固定功能：检查采集视频。CPU检查当前随机化，导出10集侧前方全身跟随视频。
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$task_root"
export CUDA_VISIBLE_DEVICES=''
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1
export PYTHONPATH="$task_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$task_root/third_party/Isaac-GR00T/.venv/bin/python" \
  -m locomani.tasks.bottle_visual.pipeline.video_check "$@"
