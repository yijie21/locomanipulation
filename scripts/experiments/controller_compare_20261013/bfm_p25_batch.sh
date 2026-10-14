#!/usr/bin/env bash
# TODO 25.5 perturbation cells + the estimated-pose arm, ScaleBFM side (plan §4.5 / §4.6). Run only when no SONIC stack is
# up (it is CPU-heavy). The perturbation specs are perturb_configs.py's (the SONIC cells use the same).
#   bash scripts/experiments/controller_compare_20261013/bfm_p25_batch.sh [stage...]   stages: experts carry brake odometry clips
# Experts: the 25.7 ports on the L1 seeds (bottle mode 4; warehouse mode 2, visual.tracking_gain 0.5), nominal + 7 cells.
set -uo pipefail
cd /workspace/code/locomanipulation || exit 1
export TMPDIR=/workspace/tmp NUMPY_MADVISE_HUGEPAGE=0 PYTHONPATH=src OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=${GPU:-1}
D=scripts/experiments/controller_compare_20261013
O=outputs/controller_compare_20261013/p25
SIM_PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
BFM_PY=third_party/Isaac-GR00T/.venv/bin/python
L1B=outputs/controller_compare_20261013/bottle/sonic/states
L1W=outputs/controller_compare_20261013/warehouse/sonic/states
CB=configs/experiments/controller_compare_l2_bottle_sonic_20261013.yaml
CW=configs/experiments/controller_compare_l2_warehouse_sonic_20261013.yaml
spec() { python3 -c "import sys; sys.path.insert(0, '$D'); import perturb_configs as P, json; print(json.dumps(P.CELLS['$1'], separators=(',', ':')))"; }
STAGES=${*:-experts carry brake odometry clips}
mkdir -p $O
for stage in $STAGES; do
  case $stage in
  experts)
    for cell in nominal b20 b40 c d04 f e050 e150; do
      args=(); case $cell in nominal) ;; e050|e150) ;; *) args=(--perturb "$(spec $cell)") ;; esac
      sb=(); sw=()
      case $cell in e050|e150) sb=(--scene outputs/scene/sim_lowcom_m${cell#e}); sw=(--scene outputs/scene/warehouse_box_45x32x33_mars_m${cell#e}) ;; esac
      $SIM_PY $D/bfm_expert_bottle.py $L1B --config $CB --mode 4 --tag _$cell "${args[@]}" "${sb[@]}" --out $O/bottle/bfm_$cell > $O/bottle_$cell.log 2>&1 &
      $SIM_PY $D/bfm_expert_warehouse.py $L1W --config $CW --mode 2 --set visual.tracking_gain=0.5 --tag _$cell "${args[@]}" "${sw[@]}" \
        --out $O/warehouse/bfm_$cell > $O/warehouse_$cell.log 2>&1 &
      wait
      echo "experts $cell done $(date -u +%T)"
    done
    ;;
  carry)
    for cell in nominal c g e050 e150; do
      args=(); case $cell in nominal|e050|e150) ;; *) args=(--perturb "$(spec $cell)") ;; esac
      kb=(); kw=()
      case $cell in e050) kb=(--bottle-mass 0.176); kw=(--box-mass 0.6) ;; e150) kb=(--bottle-mass 0.529); kw=(--box-mass 1.8) ;; esac
      $BFM_PY $D/bfm_bottle.py --program replay --mode 6 --carry back2 --tag _$cell "${args[@]}" "${kb[@]}" --sonic $L1B --out $O/bottle/bfm_carry_$cell \
        > $O/bottle_carry_$cell.log 2>&1 &
      $BFM_PY $D/bfm_warehouse.py --program replay --servo pad --mode 6 --carry turn180 --tag _$cell "${args[@]}" "${kw[@]}" --sonic $L1W \
        --out $O/warehouse/bfm_carry_$cell > $O/warehouse_carry_$cell.log 2>&1 &
      wait
      echo "carry $cell done $(date -u +%T)"
    done
    ;;
  brake)
    for cell in b20 b40 d04; do
      $BFM_PY scripts/experiments/wbc_compare_20261012/bfm_brake.py --program plan --ff speed --seed 2 --perturb "$(spec $cell)" \
        --out $O/brake/bfm_$cell > $O/brake_$cell.log 2>&1 &
    done
    wait
    echo "brake done $(date -u +%T)"
    ;;
  odometry)
    # plan §4.5: the estimated pose (leg kinematics + IMU, encoder / gyro noise c on) instead of the truth
    C="$(spec c)"
    $BFM_PY scripts/experiments/wbc_compare_20261012/bfm_brake.py --program plan --ff speed --seed 2 --odometry --perturb "$C" \
      --out $O/odometry/brake_odo_c > $O/odo_brake.log 2>&1 &
    $BFM_PY scripts/experiments/wbc_compare_20261012/bfm_brake.py --program plan --ff speed --seed 2 --perturb "$C" \
      --out $O/odometry/brake_truth_c > $O/odo_brake_truth.log 2>&1 &
    $SIM_PY $D/bfm_expert_bottle.py $L1B --config $CB --mode 4 --tag _odo_c --odometry --perturb "$C" --out $O/odometry/bottle_odo_c > $O/odo_bottle.log 2>&1 &
    $SIM_PY $D/bfm_expert_warehouse.py $L1W --config $CW --mode 2 --set visual.tracking_gain=0.5 --tag _odo_c --odometry --perturb "$C" \
      --out $O/odometry/warehouse_odo_c > $O/odo_warehouse.log 2>&1 &
    wait
    echo "odometry done $(date -u +%T)"
    ;;
  clips)
    $BFM_PY $D/bfm_clip.py --names C2_high_reach --modes 2,7 --repeats 10 --perturb "$(spec f)" --out $O/clips/bfm_f > $O/clips_f.log 2>&1 &
    $BFM_PY $D/bfm_clip.py --names C4_side_left,C4_side_right,C4_backward,C4_turn --modes 2,4,6 --repeats 10 --perturb "$(spec d04)" \
      --out $O/clips/bfm_d04 > $O/clips_d04.log 2>&1 &
    $BFM_PY $D/bfm_clip.py --names S0_stand --modes 2,4 --repeats 10 --perturb "$(spec g)" --out $O/clips/bfm_g > $O/clips_g.log 2>&1 &
    wait
    echo "clips done $(date -u +%T)"
    ;;
  esac
done
echo "BFM P25 DONE $(date -u)"
