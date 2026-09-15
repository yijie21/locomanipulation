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
# Usage:  bash scripts/stack/start_stack.sh [task-prompt]

set -u
_requested_dds="${SONIC_DDS_DOMAIN:-}"

# shellcheck disable=SC1091
. "$(dirname "$0")/../../scripts/paths.sh"
# The readiness checks below read the sim's stdout (sim.log) for "Image publish
# frequency", printed by the image-publisher SUBPROCESS with plain print(). With
# stdout redirected to a file Python block-buffers 8 KB, so on a host without a
# global PYTHONUNBUFFERED the first line lands ~5 min later than the 40 s
# window and the stack is declared unusable although it streams fine
# (2026-09-11, SRA0889). Unbuffer every Python this script launches.
export PYTHONUNBUFFERED=1
REPO="$SONIC_ROOT"
# INSTANCE selects a port block and DDS domain so several stacks can share
# the host: instance i uses ports base+10*i and DDS domain i. Instance 0 is
# byte-identical to the historical single-stack layout.
INSTANCE="${INSTANCE:-0}"
export SONIC_DDS_DOMAIN="${SONIC_DDS_DOMAIN:-$INSTANCE}"
export TF_CAMERA_PORT=$((5555 + 10 * INSTANCE))
export TF_SONIC_PORT=$((5556 + 10 * INSTANCE))
TF_OUT_PORT=$((5557 + 10 * INSTANCE))
export TF_KEYBOARD_PORT=$((5580 + 10 * INSTANCE))
export TF_PRIVILEGED_PORT=$((5581 + 10 * INSTANCE))
export TF_EPISODE_PORT=$((5582 + 10 * INSTANCE))
LOGS=$(locomani_session_root "$INSTANCE")
# Per-instance admission lock. 2026-09-10: two runners raced for instance 0's port
# block in the same second (both had checked GPU memory, one had just freed the
# ports), both printed "stack up", one gym died with "Address already in use" and
# the other's every round was "teleport landed 1.85 m off". A port check is a
# time-of-check/time-of-use race; a lock is not. fd 9 is inherited by the
# nohup'd sim / deploy / exporter below, so the lock is held exactly as long as
# any of them lives and releases itself when they die -- no liveness check, no
# stale-lock cleanup; stop_instance.sh releases it by killing them.
# Under $LOCOMANI_TMP (paths.sh), not a hard-coded /workspace: on a machine
# without /workspace the mkdir failed, `exec 9>>` failed, and flock's "bad
# file descriptor" was reported as "instance held by a live stack" (2026-09-11).
LOCK_DIR="$LOCOMANI_TMP/locomani_locks"
mkdir -p "$LOCK_DIR" || { echo "cannot create lock dir $LOCK_DIR (set LOCOMANI_WORKSPACE / LOCOMANI_TMP, see scripts/paths.sh)"; exit 3; }
LOCK="$LOCK_DIR/instance_$INSTANCE.lock"
exec 9>>"$LOCK" || { echo "cannot open lock file $LOCK"; exit 3; }
if ! flock -n 9; then
    echo "REFUSED: instance $INSTANCE is held by a live stack -- $(tail -1 "$LOCK" 2>/dev/null)"
    echo "  (a second stack on the same port block silently replaces the first; pick another INSTANCE"
    echo "   or stop that one with scripts/stack/stop_instance.sh $INSTANCE)"
    exit 3
fi
: > "$LOCK"
echo "start_stack pid $$ at $(date -u +%FT%TZ) SIM_VIDEO_DIR=${SIM_VIDEO_DIR:-} LOCOMANI_SIM_DIR=${LOCOMANI_SIM_DIR:-}" >> "$LOCK"
# Only this shell's live background jobs are eligible for failure cleanup.
# Install after acquiring fd 9: an admission refusal must not touch its owner.
kill_owned_tree() {
    local owned_pid="$1" descendant
    for descendant in $(pgrep -P "$owned_pid" || true); do
        kill_owned_tree "$descendant"
    done
    kill -KILL "$owned_pid" 2>/dev/null || true
}
cleanup_failed_start() {
    local status=$? owned_pid
    if [ "$status" -ne 0 ]; then
        for owned_pid in $(jobs -pr); do kill_owned_tree "$owned_pid"; done
        wait 2>/dev/null || true
    fi
}
trap cleanup_failed_start EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
TASK_PROMPT="${1:-walk forward}"

mkdir -p "$LOGS"
cd "$REPO" || exit 1

# shellcheck disable=SC1091
source "$LOCOMANI_ROOT/scripts/env_deploy.sh"
export SONIC_DDS_DOMAIN="${_requested_dds:-$INSTANCE}"
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
    SIM_SCRIPT="$LOCOMANI_SRC/sim/run_sim_locomani.py"
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
# --instance is stripped by run_sim_locomani.py before tyro; it only puts the
# instance on the cmdline so instance_watchdog.py can tell two sims apart.
# The stock run_sim_loop.py would reject it, so only pass it to our wrapper.
INSTANCE_ARG=""
[ "$SCENE" = "locomani" ] && INSTANCE_ARG="--instance $INSTANCE"
# shellcheck disable=SC2086
nohup "$REPO/.venv_sim/bin/python" "$SIM_SCRIPT" $INSTANCE_ARG \
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
# SONIC_POLICY picks the checkpoint, and since 2026-09-11 the DEFAULT IS v1.1
# (user ruling; the default itself lives in scripts/paths.sh, this is only the
# fallback for a shell that never sourced it). v1.1 is "robot-heading-normalized,
# trained with wrist-pose augmentation for heading-stable 3-point teleoperation"
# -- i.e. trained against the two things this pipeline fights hardest: wrist pose
# accuracy and heading drift. Its observation space differs from the release
# model (heading-normalized anchor orientations, no motion_root_z_position, so
# 1751 encoder inputs against release's 1762), so the obs-config has to travel
# with it; the deploy binary already knows the names.
#
# The heading ANCHOR was the blocker that made v1.1 look broken: the deploy
# captures it at start_control, i.e. before the gym teleports the robot to the
# round's yaw. The release model tolerates a stale anchor (absolute reference
# orientations); v1.1 normalizes by the robot heading, so a stale anchor is a
# standing heading error it keeps turning to close -- 0.134 m of drift and up to
# 60 deg of yaw during a 5 s IDLE hold, which failed every round at "teleport did
# not land". grasp_gym re-anchors before the settle, which is what makes v1.1
# runnable at all. Do NOT repeat the old "0.004 m, better than release" line
# from 2026-08-21: it was measured on the pre-reanchor path, and on the current
# one v1.1's SETTLE drift is worse, 0.179 m against release's 0.100 (ledger §17).
#
# WHAT IS MEASURED AND NOT YET RE-CALIBRATED (read before trusting a number):
#   * reference replay: v1.1 tracks the waist far better -- waist RMSE 3.9 deg
#     against release's 12.9, torso 5.7 vs 12.4 (ledger bottle.md 16, STATE).
#   * the bottle grasp line was CALIBRATED ON RELEASE and v1.1 scored 0/20
#     against release's 9/20 on one fixed case bank (McNemar p=0.0039, ledger
#     bottle.md 17): v1.1 parks 12 cm further out and 11 cm further sideways,
#     and its approach ends "stalled" 10/20. The cause is arrival geometry, not
#     grasping -- the stride is much shorter (speed 0.30 delivers 4.3 cm/step
#     against release's 17), so the approach timings, the brake band and
#     primitives_tuning.TRACK_ROT_BOOST all want re-measuring on v1.1 before any
#     grasp number from this default is comparable to a release-era one.
#   * a token is a CHECKPOINT-PRIVATE latent. Every dataset and GR00T checkpoint
#     collected before this switch holds release-encoder tokens; replaying them
#     through a v1.1 deploy is silently meaningless. docs/reference/
#     silent_failures.md has the row.
# SONIC_POLICY=policy/release runs the old checkpoint for a comparison arm.
SONIC_POLICY="${SONIC_POLICY:-policy/sonic_v1_1}"
echo "  policy: $SONIC_POLICY"
# NO_ENCODER=1 launches the deploy WITHOUT the encoder: initial_encoder_mode_
# becomes -1, "Encoder config found but no encoder file provided - tokens can
# be set externally", and the token path is the only source.
#
# CORRECTED 2026-08-28 (docs/official_parity_audit.md C5). This comment used to
# claim an encoder-loaded deploy "keeps flipping back so an external token
# stream never holds". The source does not support that, and NVIDIA's own VLA
# flow runs WITH the encoder (deploy.sh always passes --encoder-file):
#   * cpp:3022  the first external token sets is_using_encoder_ = false, for good
#   * cpp:3054  the flip-back needs has_ext_token_data FALSE, and that comes
#               from a LATCHING DataBuffer (input_interface.hpp:149,
#               utils.hpp:59). has_external_token_state_ is cleared only on a
#               protocol error, an interface-manager safety reset, or a return
#               to pre-loaded motion (zmq_endpoint_interface.hpp:253/291/341)
#               -- never by an ordinary gap between publishes.
# So NO_ENCODER is NOT required for the token stream to hold. It is kept because
# it is free here and cheaper: the gym restarts the deploy EVERY round, and not
# loading a second TensorRT engine keeps that restart (already 20-40 s) shorter.
# Running with the encoder is untested by us; it would be one deviation fewer
# from the official deployment form if real-robot transfer ever needs it.
ENC_ARG=(--encoder-file "$SONIC_POLICY/model_encoder.onnx")
if [ "${NO_ENCODER:-0}" = "1" ]; then
    ENC_ARG=()
    echo "  encoder: OFF (token-only / VLA deployment mode)"
fi
# LOCKSTEP=1: SIMPLE's simulation-clock patch (--sim-lockstep), built from a
# copy of their tree plus our SONIC_DDS_DOMAIN patch (/workspace/deps/
# gear_sonic_deploy_lockstep). Same protocol v4, same $SONIC_POLICY files
# (md5-identical decoder); the binary only ticks when the sim hands it an
# envelope-stamped LowState. See lockstep_sim.py.
# SONIC_DEPLOY_BIN: run a different build of the deploy WITHOUT lockstep
# (e.g. the patched SIMPLE build in wall-clock mode, so a lockstep-vs-
# wall-clock calibration compares the same binary). A foreign tree needs the
# policy directory too -- the guard below says so rather than letting the deploy
# die on a missing file.
DEPLOY_BIN="${SONIC_DEPLOY_BIN:-./target/release/g1_deploy_onnx_ref}"
if [ -n "${SONIC_DEPLOY_BIN:-}" ]; then
    # SONIC_DEPLOY_LIBS=simple: SIMPLE's TRT 10.9 libs (its own tree only);
    # our builds under /workspace/deps carry their engine caches for the
    # toolchain they were built with, and the wrong libs fail with
    # "Serialization assertion stdVersionRead == kSERIALIZATION_VERSION".
    [ "${SONIC_DEPLOY_LIBS:-}" = "simple" ] && \
    export LD_LIBRARY_PATH="$HOME/tools/zmq-install/lib:$HOME/tools/onnxruntime-linux-x64-gpu-1.22.0/lib:$HOME/tools/tensorrt-root/lib:${LD_LIBRARY_PATH:-}"
    echo "  deploy binary: $DEPLOY_BIN"
fi
LOCKSTEP_ARG=()
if [ "${CLEAN_DEPLOY:-0}" = "1" ] && [ "${LOCKSTEP:-0}" != "1" ]; then
    # Wall-clock "clean deploy" (docs/lockstep.md 校准诊断). NOT lockstep:
    # the _nvdef tree is only used for its auto-start support and NVIDIA
    # default_angles; no --sim-lockstep here, the deploy free-runs. The lockstep-port
    # build with NVIDIA's default_angles, engaged by SONIC_AUTO_START, never
    # sent a planner command. One planner hold before enter_streamed leaves
    # the deploy executing walking tokens at ~1/3 amplitude.
    CD_ROOT="${CLEAN_DEPLOY_ROOT:-$DEPS_DIR/gear_sonic_deploy_ours_lockstep_nvdef}"
    DEPLOY_BIN="$CD_ROOT/target/release/g1_deploy_onnx_ref"
    LOCKSTEP_ARG=(--default-motion neutral_kick_R_001__A543)
    export SONIC_AUTO_START=1
    echo "  CLEAN_DEPLOY: $DEPLOY_BIN (auto-start, no planner)"
fi
if [ "${LOCKSTEP:-0}" = "1" ]; then
    # Default: OUR deploy source + SIMPLE's lockstep commits (branch
    # lockstep-port in third_party/GR00T-WholeBodyControl), built with our
    # TensorRT/onnxruntime under /workspace/deps -- so the lockstep binary is
    # behaviourally the production deploy. LOCKSTEP_DEPLOY_ROOT=/workspace/
    # deps/gear_sonic_deploy_lockstep (+ LOCKSTEP_DEPLOY_LIBS=simple) selects
    # the copy of SIMPLE's own tree instead.
    LS_ROOT="${LOCKSTEP_DEPLOY_ROOT:-$DEPS_DIR/gear_sonic_deploy_ours_lockstep}"
    DEPLOY_BIN="$LS_ROOT/target/release/g1_deploy_onnx_ref"
    # --default-motion: under lockstep there is no planner thread, so the
    # reference-motion features in the policy observation come from
    # current_motion_ frame 0 -- which is the FIRST clip in reference/example
    # (a dance) unless a default is named. SIMPLE's choice; its frame 0 is a
    # neutral stance, the lockstep counterpart of the planner's idle pose.
    LOCKSTEP_ARG=(--sim-lockstep --default-motion neutral_kick_R_001__A543)
    # No planner thread under lockstep, and the zmq 'start' command only
    # takes effect inside the planner-mode handler -- so the policy is
    # engaged the way SIMPLE does it: auto-start on the first
    # WAIT_FOR_CONTROL tick after the 3 s init ramp (one-shot; session
    # resets keep CONTROL state).
    export SONIC_AUTO_START=1
    if [ "${LOCKSTEP_DEPLOY_LIBS:-}" = "simple" ]; then
        export LD_LIBRARY_PATH="$HOME/tools/zmq-install/lib:$HOME/tools/onnxruntime-linux-x64-gpu-1.22.0/lib:$HOME/tools/tensorrt-root/lib:${LD_LIBRARY_PATH:-}"
    fi
    echo "  LOCKSTEP: $DEPLOY_BIN --sim-lockstep"
fi
# A foreign build runs from ITS tree: the TensorRT engine caches next to the
# policy ($SONIC_POLICY/*.trt) are serialized by the TensorRT the binary was
# linked against (10.9 there, 10.13 here) and do not cross over; planner and
# reference clips resolve relative to cwd as well. The policy files that
# matter (observation_config.yaml, model_decoder.onnx) are md5-identical.
case "$DEPLOY_BIN" in
    /*) cd "$(dirname "$DEPLOY_BIN")/../.." || exit 1
        echo "  deploy cwd: $PWD" ;;
esac
# $SONIC_POLICY is resolved relative to the deploy's cwd, which the case above
# may have just moved into a foreign tree -- and those trees were populated
# before v1.1 became the default, so SIMPLE's own copy
# (/workspace/deps/gear_sonic_deploy_lockstep) has policy/release only. Without
# this check the deploy exits inside ONNX Runtime with a path error 200 lines
# into deploy.log and the stack looks like a timeout.
if [ ! -f "$SONIC_POLICY/model_decoder.onnx" ]; then
    echo "  FAILED: $PWD/$SONIC_POLICY/model_decoder.onnx does not exist."
    echo "          This deploy tree has: $(ls -d policy/*/ 2>/dev/null | tr -d / | tr '\n' ' ')"
    echo "          Copy the checkpoint in (model_{encoder,decoder}.onnx +"
    echo "          observation_config.yaml; the .trt engines rebuild on first run,"
    echo "          ~10 min) or set SONIC_POLICY to one this tree has."
    exit 1
fi
# Record the final cwd and files, including foreign clean/lockstep builds.
PYTHONPATH="$LOCOMANI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m locomani.pipeline.provenance \
    --deploy-receipt "${LOCOMANI_DEPLOY_RECEIPTS:-$LOGS/deploy_receipts.jsonl}" \
    --binary "$DEPLOY_BIN" --policy "$SONIC_POLICY" || exit 1
nohup "$DEPLOY_BIN" lo "$SONIC_POLICY/model_decoder.onnx" reference/example/ \
    --obs-config "$SONIC_POLICY/observation_config.yaml" \
    "${ENC_ARG[@]}" \
    --planner-file planner/target_vel/V2/planner_sonic.onnx \
    --input-type zmq_manager --output-type all --zmq-host localhost --zmq-port "$TF_SONIC_PORT" --zmq-out-port "$TF_OUT_PORT" --disable-crc-check "${LOCKSTEP_ARG[@]}" ${DEPLOY_EXTRA_ARGS:-} \
    > "$LOGS/deploy.log" 2>&1 &
# First run compiles TensorRT engines (~10 min); later runs load the cached .trt.
# Under lockstep the deploy reaches "Init Done" only once the sim ticks it,
# and nothing ticks before the agent connects: its ready marker is the
# lockstep control thread instead.
READY_PAT="Init Done"; [ "${LOCKSTEP:-0}" = "1" ] && READY_PAT="simulation-clock control thread active"
wait_for_text "$LOGS/deploy.log" "$READY_PAT" 1200 || {
    echo "  FAILED: deploy never reached 'Init Done'"
    grep -viE "^INFO|^WARNING" "$LOGS/deploy.log" | tail -20; exit 1; }
if [ "${CLEAN_DEPLOY:-0}" = "1" ] && [ "${LOCKSTEP:-0}" != "1" ]; then
    wait_for_text "$LOGS/deploy.log" "transitioning to CONTROL state" 120 || {
        echo "  FAILED: SONIC_AUTO_START never engaged the policy"; tail -20 "$LOGS/deploy.log"; exit 1; }
fi
echo "  ok"

echo "[3/3] data exporter"
# Instance 0 always records. Another instance records only when the caller
# opts in with RECORD_INSTANCE=1 (run_crate_pilot.py --record does): the
# exporter is ~500 threads against the container's pid cap, so it is not free,
# but its ports and its session root are already per-instance
# (locomani.sim.ports, paths.session_root), and stop_instance.sh kills only
# the exporter serving this instance's camera port.
if [ "$INSTANCE" != "0" ] && [ "${RECORD_INSTANCE:-0}" != "1" ]; then
    echo "  skipped (instance $INSTANCE records no episodes; RECORD_INSTANCE=1 to record here)"
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
STACK_SCENE_META="${LOCOMANI_SIM_DIR:-$LOCOMANI_ROOT/outputs/scene/sim}/scene_meta.json"
if [ -z "${TF_POSE_OBJECTS:-}" ] && [ -f "$STACK_SCENE_META" ]; then
    TF_POSE_OBJECTS="$("$REPO/.venv_sim/bin/python" -c "
import json, sys
m = json.load(open(sys.argv[1]))
print(','.join(m.get('objects', {})))
" "$STACK_SCENE_META" 2>/dev/null)"
    export TF_POSE_OBJECTS
    echo "  pose columns: ${TF_POSE_OBJECTS:-<none found>}"
fi
# Thread caps: the exporter's three processes idle at ~800 threads each
# (OpenMP/BLAS pools), ~2500 pids per stack against the container's 3840
# limit -- a second stack then hangs at thread creation (2026-09-03).
nohup env OMP_NUM_THREADS="${EXPORTER_THREADS:-4}" MKL_NUM_THREADS="${EXPORTER_THREADS:-4}" \
    OPENBLAS_NUM_THREADS="${EXPORTER_THREADS:-4}" NUMEXPR_NUM_THREADS="${EXPORTER_THREADS:-4}" \
    "$REPO/.venv_data_collection/bin/python" "${TF_EXPORTER_SCRIPT:-$LOCOMANI_SRC/pipeline/run_instance_exporter.py}" \
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
echo "next:  $VENV_SIM_PY $LOCOMANI_SRC/tasks/bottle/grasp_gym.py --rounds 12 --record --manifest outputs/logs/manifest.jsonl"
