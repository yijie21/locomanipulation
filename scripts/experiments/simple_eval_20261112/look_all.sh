#!/usr/bin/env bash
# look_all.sh TASK[:only,only] ...: every episode's first look through SIMPLE's Isaac head camera (blind.py look --eyes;
# the replica is the physics, eyes_server.py draws its state in SIMPLE's scene), one Isaac process at a time on one GPU.
# Then GraspGen-X on the new clouds (skipped with NO_GRASPS=1).
#   MAIN          main's checkout, the owner of third_party/ (default /workspace/code/locomanipulation)
#   GPU           the card for Isaac, the depth model and GraspGen-X (default 1: card 0 is other jobs' on vastgpu)
#   TAG           prefix of this run's logs under outputs/simple_eval_20261112/look_logs/ (default run)
#   LOOK_ARGS     more arguments for blind.py look, e.g. "--unnamed depth --cloud cloud_eyes_depth"
#   CLOUD_NAME    the cloud folder GraspGen-X reads; the same name as LOOK_ARGS' --cloud (default cloud_eyes)
#   LOOK_DUMP_DIR every look's intermediates, one npz per episode (default outputs/.../look_dump_eyes)
#   NICE          10 yields the CPU to other sessions' jobs;  LOOK_TIMEOUT  s per task (default 3000)
#   SIMPLE_OLD_TASK  XMoveBendPickTeleop's task file from SIMPLE eef474f^; fetched from SIMPLE's git history if absent
# One line per finished step in look_logs/<TAG>_done.txt.
set -uo pipefail
E=$(cd "$(dirname "$0")" && pwd)
W=$(cd "$E/../../.." && pwd)
M=${MAIN:-/workspace/code/locomanipulation}
O=$W/outputs/simple_eval_20261112
L=$O/look_logs
GPU=${GPU:-1}
PY=$W/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
SOCK=${SOCK_DIR:-/workspace/tmp/simple_sock}; DUMP=${LOOK_DUMP_DIR:-$O/look_dump_eyes}
OLD=${SIMPLE_OLD_TASK:-$O/film_20261119/scripts/oldtask/g1_wholebody_xmove_bend_pick_teleop.py}
TAG=${TAG:-run}
if [ ! -f "$OLD" ]; then  # SIMPLE dropped XMoveBendPickTeleop in eef474f; its episodes are from the version before
    mkdir -p "$(dirname "$OLD")"
    git -C $M/third_party/SIMPLE show eef474f^:src/simple/tasks/g1_wholebody_xmove_bend_pick_teleop.py > "$OLD"
fi
mkdir -p "$L" "$DUMP" "$SOCK"
rm -f $SOCK/look_depth.sock
(cd $M && PYTHONPATH=third_party/lingbot-depth HF_HOME=${HF_HOME:-/workspace/huggingface_cache} HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$GPU \
    exec ${DEPTH_PY:-/workspace/miniconda3/envs/sam3d/bin/python} -s $E/depth_server.py $SOCK/look_depth.sock > $L/${TAG}_depth.log 2>&1) &
DPID=$!
for spec in "$@"; do
    task=${spec%%:*}; only=""
    [[ "$spec" == *:* ]] && only="--only ${spec#*:}" && only=${only//,/ }
    rm -f $SOCK/look_env.sock
    (cd $M/third_party/SIMPLE && SIMPLE_OLD_TASK=$OLD MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TMPDIR=/workspace/tmp \
        exec nice -n ${NICE:-0} .venv/bin/python $E/eyes_server.py $SOCK/look_env.sock $O/x > $L/${TAG}_env_$task.log 2>&1) &
    EPID=$!
    until grep -q listening $L/${TAG}_env_$task.log 2>/dev/null && grep -q listening $L/${TAG}_depth.log 2>/dev/null; do
        kill -0 $EPID 2>/dev/null || break
        sleep 5
    done
    (cd $W && LOOK_DUMP=$DUMP BFM_ROOT=odo OMP_NUM_THREADS=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src timeout ${LOOK_TIMEOUT:-3000} \
        nice -n ${NICE:-0} $PY $E/blind.py look $task $only ${LOOK_ARGS:-} --eyes $SOCK/look_env.sock --depth $SOCK/look_depth.sock > $L/${TAG}_look_$task.log 2>&1)
    echo "rc=$? look $task $(date +%T)" >> $L/${TAG}_done.txt
    kill $EPID 2>/dev/null
    sleep 8
    kill -9 $EPID 2>/dev/null
done
kill $DPID 2>/dev/null
if [ -z "${NO_GRASPS:-}" ]; then
    G=$M/third_party/GraspGenX
    (cd $W && CUDA_VISIBLE_DEVICES=$GPU $G/.venv/bin/python $E/grasps_graspgen.py outputs/simple_eval_20261112/${CLOUD_NAME:-cloud_eyes} \
        --assets-dir $G/ext/gripper_descriptions/gripper_descriptions/assets > $L/${TAG}_grasps.log 2>&1)
    echo "rc=$? grasps $(date +%T)" >> $L/${TAG}_done.txt
fi
echo "all done $(date +%T)" >> $L/${TAG}_done.txt
