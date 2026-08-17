#!/bin/bash
# Environment for gear_sonic_deploy (C++ stack).
# Usage:  source /workspace/code/locomanipulation/env_deploy.sh
#
# Replaces gear_sonic_deploy/scripts/setup_env.sh, which is not used here:
# that script's companion install_deps.sh has `apt install cuda` fallbacks,
# and the `cuda` metapackage pulls cuda-drivers, which would overwrite the
# host-injected NVIDIA driver on this Vast instance.

export TensorRT_ROOT=/workspace/deps/TensorRT-10.13.3.9
export onnxruntime_ROOT=/workspace/deps/onnxruntime
export CUDA_HOME=/usr/local/cuda

export LD_LIBRARY_PATH="${TensorRT_ROOT}/lib:${onnxruntime_ROOT}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export SONIC_DEPLOY_ROOT=/workspace/code/locomanipulation/GR00T-WholeBodyControl/gear_sonic_deploy
export PATH="${SONIC_DEPLOY_ROOT}/target/release:${PATH}"

# Rebuild:
#   cmake -S "$SONIC_DEPLOY_ROOT" -B "$SONIC_DEPLOY_ROOT/build" \
#         -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
#   cmake --build "$SONIC_DEPLOY_ROOT/build" -j$(nproc)
