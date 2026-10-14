#!/usr/bin/env bash
# TODO 25.8 L2 + 25.5 perturbation cells, SONIC side: every batch in sequence on instance 0 / GPU 0, nothing else on
# the machine (SONIC is wall-clock). Usage: bash sonic_l2_p25_queue.sh [log] [stage...]
#   stages: clips0 (25.4's nominal clips on walk_mars, resuming), l2 (the four L2 batches), p25 (14 production cells),
#   carry (B7 / W6 under c, e, g), brake (b20, b40, d04), clips (C2 under f, C4 under d04, S0 under g).
#   Default: all, in that order.
set -uo pipefail
cd /workspace/code/locomanipulation || exit 1
. scripts/paths.sh
export TMPDIR=/workspace/tmp NUMPY_MADVISE_HUGEPAGE=0
L=${1:-outputs/controller_compare_20261013/sonic_l2_p25.log}
shift || true
STAGES=${*:-clips0 l2 p25 carry brake clips}
D=scripts/experiments/controller_compare_20261013
E=configs/experiments
spec() { python3 -c "import sys; sys.path.insert(0, '$D'); import perturb_configs as P, json; print(json.dumps(P.CELLS['$1'], separators=(',', ':')))"; }
collect() {  # config, kind (single_arm | bimanual)
  echo "=== collect $1 $(date -u)" >> "$L"
  local lanes=""; [ "$2" = bimanual ] && lanes="--lanes 1"
  bash scripts/start_stack.sh collect "$1" --gpu 0 $lanes >> "$L" 2>&1 && bash scripts/$2/run_agent_server.sh "$1" --gpu 0 >> "$L" 2>&1
  echo "=== collect $1 done rc=$? $(date -u)" >> "$L"
}
echo "=== start $(date -u) stages: $STAGES" >> "$L"
for stage in $STAGES; do
  case $stage in
  clips0)
    echo "=== clips0 walk_mars $(date -u)" >> "$L"
    bash $D/sonic_clip.sh walk_mars 0 0 --repeats 10 >> "$L" 2>&1
    echo "=== clips0 done rc=$? $(date -u)" >> "$L"
    ;;
  l2)
    collect $E/controller_compare_l2_bottle_sonic_20261013.yaml single_arm
    collect $E/controller_compare_l2_warehouse_sonic_20261013.yaml bimanual
    collect $E/controller_compare_l2_bottle_sonic_combo_20261013.yaml single_arm
    collect $E/controller_compare_l2_warehouse_sonic_combo_20261013.yaml bimanual
    ;;
  p25)
    for cell in b20 b40 c d04 f e050 e150; do collect $E/controller_compare_p25_bottle_${cell}_20261013.yaml single_arm; done
    for cell in b20 b40 c d04 f e050 e150; do collect $E/controller_compare_p25_warehouse_${cell}_20261013.yaml bimanual; done
    ;;
  carry)
    for cell in c g e050 e150; do
      for task in bottle warehouse; do
        if [ "$task" = bottle ]; then cfg=$E/bottle_simple_200.yaml; plan=configs/plans/bottle_carry_back2.json; seed=40000
        else cfg=$E/warehouse_mars_200.yaml; plan=configs/plans/warehouse_mars_carry_back1.json; seed=7000; fi
        case $cell in e*) cfg=$E/controller_compare_p25_${task}_${cell}_20261013.yaml; unset LOCOMANI_PERTURB ;;
                      *) export LOCOMANI_PERTURB="$(spec $cell)" ;; esac
        name=controller_compare_p25_${task}_carry_${cell}_20261013
        export SIM_STATE_TRACE_DIR=$PWD/outputs/runs/$name/trace
        echo "=== plan $name ${LOCOMANI_PERTURB:-} $(date -u)" >> "$L"
        "$VENV_SIM_PY" -u scripts/plan_episode.py "$cfg" --gpu 0 --instance 0 --rounds 10 --seed $seed --plan "$plan" --name $name >> "$L" 2>&1
        echo "=== plan $name done rc=$? $(date -u)" >> "$L"
        unset LOCOMANI_PERTURB SIM_STATE_TRACE_DIR
      done
    done
    ;;
  brake)
    for cell in b20 b40 d04; do
      export LOCOMANI_PERTURB="$(spec $cell)"
      echo "=== brake $cell $LOCOMANI_PERTURB $(date -u)" >> "$L"
      BRAKE_OUT=$PWD/outputs/controller_compare_20261013/p25/brake/sonic_$cell bash scripts/experiments/brake_marker_20260923.sh \
        --guard --extra 0.20 --margin 0.03 --speeds 0.55,0.35,0.25 --repeats 10 >> "$L" 2>&1
      echo "=== brake $cell done rc=$? $(date -u)" >> "$L"
      unset LOCOMANI_PERTURB
    done
    ;;
  clips)
    for job in "f C2_high_reach" "d04 C4_side_left,C4_side_right,C4_backward,C4_turn" "g S0_stand"; do
      set -- $job
      export LOCOMANI_PERTURB="$(spec $1)" CLIP_OUT=outputs/controller_compare_20261013/p25/clips/sonic_$1
      echo "=== clips $1 $2 $(date -u)" >> "$L"
      bash $D/sonic_clip.sh walk_mars 0 0 --names "$2" --repeats 10 >> "$L" 2>&1
      echo "=== clips $1 done rc=$? $(date -u)" >> "$L"
      unset LOCOMANI_PERTURB CLIP_OUT
    done
    ;;
  esac
done
echo "=== SONIC L2 P25 DONE $(date -u)" >> "$L"
