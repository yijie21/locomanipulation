#!/usr/bin/env bash
# notebook: selfmodel_via_20261120 -- sixth step (the plan entry's "第六步"): ScaleBFM replays of an obst_routes.py route set
# on vastgpu, one reset per (slot, pass), 4 passes; settings as the fourth test's replays, the scene with the two mocap boxes.
# obst_replay.sh <out dir> <obst routes dir> <seed> <ID0> <procs> <obst 0|1> [<model dir> <point|goal|const>]
#   obst 1: the path's real boxes stand in the simulation during the reach (REACH_GAP_OBST); 0: the open floor (boxes parked).
# Prepares <out dir>/launch.sh (replay, merge, arm labels, table) and runs it in the foreground.
set -eu
OUT=$1 RT=$2 SEED=$3 ID0=$4 PROCS=$5 OBST=$6 MD=${7:-} MODE=${8:-}
W=/workspace/code/locomanipulation-selfwm; I=$W/outputs/reachwm_20261030; cd $W
[ -e "$OUT" ] && { echo "$OUT exists"; exit 1; }
SC=$I/scene/simple_g1_obst; V=scripts/experiments/selfmodel_via_20261120
N=$(python3 -c "import json; print(4 * len(json.load(open('$RT/slots.json'))['names']))")
mkdir -p $OUT/asrun
cp scripts/experiments/reachwm_20261030/gen_reach.py $V/via_comp.py $V/via_gen.py $V/via_model.py $V/via_point.py \
   $V/arm_spheres.py $V/arm_fk.py "$0" $RT/slots.json $OUT/asrun/
cp src/locomani/backends/scalebfm/motor.py $OUT/asrun/motor_scalebfm.py
COMP=""; [ -n "$MD" ] && COMP="REACH_VIA_COMP=$MD REACH_VIA_COMP_MODE=$MODE"
GAP=""; [ "$OBST" = 1 ] && GAP="REACH_GAP_OBST=1"
{ echo "obstacle replay of $RT ($N resets = 4 passes x slots, seed $SEED, ID0 $ID0), mode ${MODE:-direct}${MD:+, model $MD}, obstacles $OBST; ScaleBFM; scene $SC"
  echo "commit $(git rev-parse HEAD), uncommitted: $(git status --short | tr '\n' ' ')"; } > $OUT/run_info.txt
cat > $OUT/launch.sh <<EOS
#!/usr/bin/env bash
cd $W
export BFM_ROOT=odo OMP_NUM_THREADS=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1
PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
echo "started \$(date -u +%FT%TZ) on \$(hostname), $PROCS processes, nice 10" >> $OUT/run_info.txt
env REACH_HAND_OFF=0.0415,0,0 REACH_SCENE_DIR=$SC REACH_GRASP=1 REACH_GRASP_ROOM=1 REACH_GRASP_SPEED=0.08 REACH_GRASP_CONT=0.5 \\
    REACH_GRASP_IKREF=1 REACH_WRIST=1 REACH_VIA_REPLAY=$RT/via:$RT/prep REACH_VIA_SLOTS=$RT/slots.json REACH_ID0=$ID0 \\
    REACH_OUT=$OUT $COMP $GAP nice -n 10 \$PY scripts/experiments/reachwm_20261030/gen_reach.py run --procs $PROCS --resets $N \\
    --seed $SEED --gpus 1 > $OUT/run.log 2>&1
REACH_OUT=$OUT \$PY scripts/experiments/reachwm_20261030/gen_reach.py merge > $OUT/merge.txt 2>&1
\$PY -I $V/arm_spheres.py $OUT $SC > $OUT/arm.txt 2>&1
OMP_NUM_THREADS=2 \$PY $V/via_model.py build $OUT > $OUT/build.txt 2>&1
echo "finished \$(date -u +%FT%TZ) \$(tail -1 $OUT/merge.txt)" >> $OUT/run_info.txt
EOS
chmod +x $OUT/launch.sh
$OUT/launch.sh > $OUT/launch.log 2>&1
tail -1 $OUT/run_info.txt
