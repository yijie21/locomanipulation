#!/bin/bash
# Environment for gear_sonic_deploy (C++ stack).
# Usage:  source /workspace/code/locomanipulation/scripts/env_deploy.sh
#
# Replaces gear_sonic_deploy/scripts/setup_env.sh, which is not used here:
# that script's companion install_deps.sh has `apt install cuda` fallbacks,
# and the `cuda` metapackage pulls cuda-drivers, which would overwrite the
# host-injected NVIDIA driver on this Vast instance.

export TensorRT_ROOT=/workspace/deps/TensorRT-10.13.3.9
export onnxruntime_ROOT=/workspace/deps/onnxruntime
export CUDA_HOME=/usr/local/cuda

export SONIC_DEPLOY_ROOT=/workspace/code/locomanipulation/third_party/GR00T-WholeBodyControl/gear_sonic_deploy

# unitree_sdk2's libddsc is also on the binary's baked-in RUNPATH, but that
# RUNPATH holds the build-time absolute path — LD_LIBRARY_PATH keeps the
# binary working after the repo moves without a patchelf/rebuild.
export LD_LIBRARY_PATH="${TensorRT_ROOT}/lib:${onnxruntime_ROOT}/lib:${CUDA_HOME}/lib64:${SONIC_DEPLOY_ROOT}/thirdparty/unitree_sdk2/thirdparty/lib/x86_64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PATH="${SONIC_DEPLOY_ROOT}/target/release:${PATH}"

# Rebuild:
#   cmake -S "$SONIC_DEPLOY_ROOT" -B "$SONIC_DEPLOY_ROOT/build" \
#         -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
#   cmake --build "$SONIC_DEPLOY_ROOT/build" -j$(nproc)
