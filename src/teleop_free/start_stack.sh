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
# Usage:  bash src/teleop_free/start_stack.sh [task-prompt]

set -u

# shellcheck disable=SC1091
. "$(dirname "$0")/../../scripts/paths.sh"
REPO="$SONIC_ROOT"
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
LOGS=$(locomani_session_root "$INSTANCE")
TASK_PROMPT="${1:-walk forward}"

mkdir -p "$LOGS"
cd "$REPO" || exit 1

# shellcheck disable=SC1091
source "$LOCOMANI_ROOT/scripts/env_deploy.sh"
# EGL renders straight to the offscreen framebuffer the ego camera reads,
# skipping the display entirely -- exactly what a headless box needs. But
# MUJOCO_GL is a single process-wide backend choice, not per-renderer: with
# SIM_ONSCREEN's window open (mujoco.viewer.launch_passive, GLFW/GLX), forcing
# EGL hands the viewer's render calls a context that was never bound to its
# GLX drawable ("Failed to make the EGL context current", then GLXBadDrawable
# on swap -- window opens and dies immediately). So SIM_ONSCREEN skips EGL and
# lets MuJoCo default to GLX, which renders the offscreen camera feed fine too
# as long as a real $DISPLAY is up.
if [ "${SIM_ONSCREEN:-0}" != "1" ]; then
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
fi

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
    SIM_SCRIPT="$LOCOMANI_SRC/run_sim_locomani.py"
else
    SIM_SCRIPT=gear_sonic/scripts/run_sim_loop.py
fi

# SIM_ONSCREEN=1 opens a live MuJoCo viewer window (mujoco.viewer.launch_passive,
# base_sim.py) tracking the pelvis. Needs a real $DISPLAY (see the MUJOCO_GL
# switch above); off by default because it costs render time and does nothing
# on a headless box.
ONSCREEN_ARG=--no-enable-onscreen
if [ "${SIM_ONSCREEN:-0}" = "1" ]; then ONSCREEN_ARG=--enable-onscreen; fi
echo "[1/3] MuJoCo sim (headless, EGL, camera -> :5555, scene=$SCENE, onscreen=${SIM_ONSCREEN:-0})"
nohup "$REPO/.venv_sim/bin/python" "$SIM_SCRIPT" \
    --interface lo "$ONSCREEN_ARG" --enable-offscreen --enable-image-publish \
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
# RETESTED 2026-08-21 and now WORKING. The earlier note here blamed the
# facing convention ("v1.1 expects reference orientations relative to the
# robot's heading, every `facing` we send is world-frame") — wrong on the
# mechanism: the heading-normalized observations are computed inside the
# deploy from the robot's own pelvis quaternion
# (g1_deploy_onnx_ref.cpp:1720, mode=1), so the wire format needs nothing.
# The real blocker was the heading ANCHOR: the deploy captures it at
# start_control, i.e. before the gym teleports the robot to the round's
# yaw. The release model tolerates a stale anchor (absolute reference
# orientations); v1.1 normalizes by the robot heading, so a stale anchor
# is a standing heading error it keeps turning to close — 0.134 m of
# drift and up to 60 deg of yaw during a 5 s IDLE hold, which failed
# every round at "teleport did not land". grasp_gym now re-anchors before
# the settle and the drift falls to 0.004 m, better than release's
# 0.03-0.12 m (diag_idle_drift.py).
#
# What v1.1 changes once it runs: the stand stage gets better (six rounds,
# zero stand corrections, yaw_err within 4 deg) and the stride gets much
# shorter (speed 0.30 delivers 4.3 cm/step against release's 17), so the
# walk timings want re-measuring with diag_shuffle.py before adopting it.
# Grasp success was 2/6 against release's 4/6 on the same seed — the same
# close-phase topple tail, and both samples far too small to separate.
# Set SONIC_POLICY=policy/sonic_v1_1 to run it.
SONIC_POLICY="${SONIC_POLICY:-policy/release}"
echo "  policy: $SONIC_POLICY"
# NO_ENCODER=1 launches the deploy WITHOUT the encoder, which is the
# token-only (VLA) configuration. It is not just a saving: with an encoder
# loaded, initial_encoder_mode_ is 0, and the deploy's token path then keeps
# flipping back ("if encoder mode is not -1, we need to turn on the encoder",
# g1_deploy_onnx_ref.cpp:3096) so an external token stream never holds. Drop
# the file and the mode becomes -1, "Encoder config found but no encoder file
# provided - tokens can be set externally".
ENC_ARG=(--encoder-file "$SONIC_POLICY/model_encoder.onnx")
if [ "${NO_ENCODER:-0}" = "1" ]; then
    ENC_ARG=()
    echo "  encoder: OFF (token-only / VLA deployment mode)"
fi
nohup ./target/release/g1_deploy_onnx_ref lo "$SONIC_POLICY/model_decoder.onnx" reference/example/ \
    --obs-config "$SONIC_POLICY/observation_config.yaml" \
    "${ENC_ARG[@]}" \
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
# --record-wrist-cameras adds observation.images.{left,right}_wrist to the
# dataset. It pairs with the sim's CAMERA_CONFIGS (run_sim_locomani.py): the
# exporter resolves each image feature by the last segment of its name, so the
# stream names line the two up with no further configuration -- and it raises
# rather than recording blanks if a stream is missing. WRIST_CAMS=0 turns both
# halves off together.
WRIST_ARG=(--record-wrist-cameras)
if [ "${WRIST_CAMS:-1}" = "0" ]; then WRIST_ARG=(); fi
# Which objects get a privileged pelvis-frame pose column, defaulted from the
# scene the sim actually booted rather than from a hardcoded "can,plate". The
# exporter asks the privileged stream for each name and writes zeros for one
# it does not find -- so a scene whose target is called anything else recorded
# 1008 rows of can_pose = 0.0 with nothing logged, and every downstream check
# read that as "the object never moved". It warns now, but not needing the
# warning is better.
if [ -z "${TF_POSE_OBJECTS:-}" ] && [ -f "$LOCOMANI_ROOT/outputs/scene/sim/scene_meta.json" ]; then
    TF_POSE_OBJECTS="$("$REPO/.venv_sim/bin/python" -c "
import json, sys
m = json.load(open(sys.argv[1]))
print(','.join(m.get('objects', {})))
" "$LOCOMANI_ROOT/outputs/scene/sim/scene_meta.json" 2>/dev/null)"
    export TF_POSE_OBJECTS
    echo "  pose columns: ${TF_POSE_OBJECTS:-<none found>}"
fi
nohup "$REPO/.venv_data_collection/bin/python" gear_sonic/scripts/run_data_exporter.py \
    --task-prompt "$TASK_PROMPT" \
    --root-output-dir "$LOGS/outputs" \
    "${WRIST_ARG[@]}" \
    --no-text-to-speech \
    > "$LOGS/exporter.log" 2>&1 &
sleep 8
echo "  ok"
fi

echo
# The sim's image publisher is a SUBPROCESS. If its port is still held -- by
# an orphaned publisher from a previous stack, which outlives stop_stack --
# it dies with "Address already in use (tcp://*:5555)" while the sim's main
# process carries on, and this script would print "stack up" over a stack that
# never emits a single frame. Every VLA round then fails with "camera/state
# streams never became ready", which names the symptom and not the cause.
# (Cost three rounds of an eval on 2026-08-27 before anyone read sim.log.)
if [ "${SKIP_IMAGE_CHECK:-0}" != "1" ]; then
    _img=0
    for _ in $(seq 1 20); do
        if grep -q "Image publish frequency" "$LOGS/sim.log" 2>/dev/null; then
            _img=1; break
        fi
        sleep 2
    done
    if [ "$_img" != "1" ]; then
        echo
        echo "STACK IS NOT USABLE: the sim has not published a single image."
        if grep -q "Address already in use" "$LOGS/sim.log" 2>/dev/null; then
            grep -m1 "Address already in use" "$LOGS/sim.log" | sed 's/^/  /'
            echo "  An orphaned image publisher still holds the port. Kill the"
            echo "  parentless multiprocessing children and start again:"
            echo "    pgrep -f 'venv_sim/bin/python -c from multi''processing'"
        else
            echo "  Last lines of $LOGS/sim.log:"
            tail -5 "$LOGS/sim.log" | sed 's/^/  /'
        fi
        echo "  (SKIP_IMAGE_CHECK=1 bypasses this if you know it is fine.)"
        exit 1
    fi
fi

echo "stack up. logs in $LOGS/{sim,deploy,exporter}.log"
echo "next:  $VENV_SIM_PY $LOCOMANI_SRC/grasp_gym.py --rounds 12 --record --manifest outputs/logs/manifest.jsonl"
