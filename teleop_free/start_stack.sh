#!/bin/bash
# Bring up the headless SONIC data-collection stack, in dependency order:
#
#   MuJoCo sim  --(camera :5555)-->  data exporter
#        ^                                ^
#        | DDS (lo)                       | :5556 planner/pose, :5557 g1_debug
#   C++ deploy (g1_deploy_onnx_ref, --input-type zmq)
#
# The scripted commander (zmq_driver.py) is started separately; it binds
# :5556 and :5580, so it must come last.
#
# Usage:  bash teleop_free/start_stack.sh [task-prompt]

set -u

REPO=/workspace/code/locomanipulation/GR00T-WholeBodyControl
# INSTANCE selects a port block and DDS domain so several stacks can share
# the host: instance i uses ports base+10*i and DDS domain i. Instance 0 is
# byte-identical to the historical single-stack layout.
INSTANCE="${INSTANCE:-0}"
export SONIC_DDS_DOMAIN=$INSTANCE
export TF_CAMERA_PORT=$((5555 + 10 * INSTANCE))
export TF_SONIC_PORT=$((5556 + 10 * INSTANCE))
TF_OUT_PORT=$((5557 + 10 * INSTANCE))
export TF_KEYBOARD_PORT=$((5580 + 10 * INSTANCE))
export TF_PRIVILEGED_PORT=$((5581 + 10 * INSTANCE))
export TF_EPISODE_PORT=$((5582 + 10 * INSTANCE))
if [ "$INSTANCE" = "0" ]; then LOGS=/workspace/tmp/p03; else LOGS=/workspace/tmp/p03_i$INSTANCE; fi
TASK_PROMPT="${1:-walk forward}"

mkdir -p "$LOGS"
cd "$REPO" || exit 1

# shellcheck disable=SC1091
source /workspace/code/locomanipulation/env_deploy.sh
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

wait_for_port() {  # $1=port $2=timeout_s
    local end=$((SECONDS + $2))
    while [ $SECONDS -lt $end ]; do
        ss -lnt 2>/dev/null | grep -q ":$1 " && return 0
        sleep 1
    done
    return 1
}

wait_for_text() {  # $1=file $2=pattern $3=timeout_s
    local end=$((SECONDS + $3))
    while [ $SECONDS -lt $end ]; do
        [ -f "$1" ] && grep -qi "$2" "$1" && return 0
        sleep 2
    done
    return 1
}

# SCENE=locomani (default) runs our two-surface scene and publishes privileged
# state on :5581; SCENE=default runs the stock empty G1 scene, which is what
# the P0.3 walk-only proof used.
SCENE="${SCENE:-locomani}"
if [ "$SCENE" = "locomani" ]; then
    SIM_SCRIPT=/workspace/code/locomanipulation/teleop_free/run_sim_locomani.py
else
    SIM_SCRIPT=gear_sonic/scripts/run_sim_loop.py
fi

echo "[1/3] MuJoCo sim (headless, EGL, camera -> :5555, scene=$SCENE)"
nohup "$REPO/.venv_sim/bin/python" "$SIM_SCRIPT" \
    --interface lo --no-enable-onscreen --enable-offscreen --enable-image-publish \
    > "$LOGS/sim.log" 2>&1 &
wait_for_port "$TF_CAMERA_PORT" 90 || { echo "  FAILED: camera port never opened"; tail -20 "$LOGS/sim.log"; exit 1; }
echo "  ok"

# Input types, from src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp:2451-2502:
#   zmq          subscribes to 'pose' only          -> raw SMPL stream, ignores our commands
#   manager      keyboard/gamepad/zmq multiplexer   -> starts in KEYBOARD, needs Shift+4 on stdin
#   zmq_manager  'command' + 'planner' + 'pose'     -> what pico_manager_thread_server talks to
echo "[2/3] C++ deploy (--input-type zmq_manager)"
cd "$SONIC_DEPLOY_ROOT" || exit 1
# SONIC_POLICY picks the checkpoint. v1.1 is "robot-heading-normalized,
# trained with wrist-pose augmentation for heading-stable 3-point teleoperation"
# — i.e. trained against the two things this pipeline fights hardest: wrist pose
# accuracy and heading drift. Its observation space differs from the release
# model (heading-normalized anchor orientations, no motion_root_z_position), so
# the obs-config has to travel with it; the deploy binary already knows the
# names. First run rebuilds the TensorRT engines, which takes ~10 min.
#
# TESTED AND REVERTED. v1.1 loads cleanly — every observation dimension matches
# (encoder 1751, decoder 994) and the same planner_sonic.onnx serves both — but
# the robot cannot use it through this interface: 4 of 8 episodes failed to
# stand at all and the other 4 could not walk ("no progress for 8s", 1.0-1.3 m
# of field left). Dimensions matching while behaviour collapses points at the
# heading normalization: v1.1 expects reference orientations relative to the
# robot's heading, and every `facing` this pipeline sends is a WORLD-frame
# heading vector. Adopting v1.1 means re-deriving that convention, not swapping
# a file. Set SONIC_POLICY=policy/sonic_v1_1 to pick the experiment back up.
SONIC_POLICY="${SONIC_POLICY:-policy/release}"
echo "  policy: $SONIC_POLICY"
nohup ./target/release/g1_deploy_onnx_ref lo "$SONIC_POLICY/model_decoder.onnx" reference/example/ \
    --obs-config "$SONIC_POLICY/observation_config.yaml" \
    --encoder-file "$SONIC_POLICY/model_encoder.onnx" \
    --planner-file planner/target_vel/V2/planner_sonic.onnx \
    --input-type zmq_manager --output-type all --zmq-host localhost --zmq-port "$TF_SONIC_PORT" --zmq-out-port "$TF_OUT_PORT" --disable-crc-check \
    > "$LOGS/deploy.log" 2>&1 &
# First run compiles TensorRT engines (~10 min); later runs load the cached .trt.
wait_for_text "$LOGS/deploy.log" "Init Done" 1200 || {
    echo "  FAILED: deploy never reached 'Init Done'"
    grep -viE "^INFO|^WARNING" "$LOGS/deploy.log" | tail -20; exit 1; }
echo "  ok"

echo "[3/3] data exporter"
if [ "$INSTANCE" != "0" ]; then
    echo "  skipped (instance $INSTANCE records no episodes)"
else
cd "$REPO" || exit 1
nohup "$REPO/.venv_data_collection/bin/python" gear_sonic/scripts/run_data_exporter.py \
    --task-prompt "$TASK_PROMPT" \
    --root-output-dir /workspace/tmp/p03/outputs \
    --no-text-to-speech \
    > "$LOGS/exporter.log" 2>&1 &
sleep 8
echo "  ok"
fi

echo
echo "stack up. logs in $LOGS/{sim,deploy,exporter}.log"
echo "next:  $REPO/.venv_sim/bin/python /workspace/code/locomanipulation/teleop_free/zmq_driver.py --record"
