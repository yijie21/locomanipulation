#!/bin/bash
# Phase 0 of the deployment-side parameter study: OFFLINE screen of the two
# knobs that need no closed-loop time.
#
#   bash scripts/phase0_denoise_sweep.sh
#
# WHAT IT DECIDES
#   --denoising-steps  swept 4/8/16/32 (serve_policy's default is 4 and has
#                      never been varied in this repo)
#   FSQ snap           free in the same run: eval_open_loop reports both
#                      nmse_token (snapped) and nmse_token_unsnapped
#
# THE JUDGE is std_ratio_token / std_ratio_gripper, added to eval_open_loop.py
# for this study. nMSE is symmetric in the sign of the residual and therefore
# blind to MAGNITUDE SHRINKAGE, which is the specific thing 4-step flow
# matching does. The closed-loop signature it is meant to explain, measured on
# walk200's 111 traces:
#
#   every `reach` failure parked the commanded gripper at +0.0168, inside a
#   band the training data occupies 3.3% of the time (open >0.020 is 53.3%,
#   shut <0.005 is 43.5%), and the arm never rose above 0.95 m in 0/33 and
#   0/58 of them, while 33/33 and 16/16 successes did.
#
# Read the result this way:
#   ratio < 1 at 4 steps, climbing toward 1  -> the standing attractor is a
#       sampler artefact; --denoising-steps is the lever and Phase 1 should
#       run it first.
#   ratio ~ 1 already at 4 steps             -> it is not; drop the knob and
#       spend Phase 1's budget on --ensemble / --replan-interval instead.
set -uo pipefail
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

RUN="${RUN:-walk200}"
CKPT="${CKPT:-$(ls -d "$LOCOMANI_OUTPUTS/train/$RUN"/checkpoint-* 2>/dev/null \
        | sort -t- -k2 -n | tail -1)}"
VAL="${VAL:-$LOCOMANI_ROOT/datasets/g1_locomani_${RUN}_val}"
OUT="${OUT:-$LOCOMANI_OUTPUTS/eval/phase0}"
STEPS="${STEPS:-4 8 16 32}"
HORIZON="${HORIZON:-16}"
GPU="${GPU:-1}"
PY="$LOCOMANI_ROOT/third_party/Isaac-GR00T/.venv/bin/python"

[ -d "$CKPT" ] || { echo "no checkpoint in outputs/train/$RUN"; exit 1; }
[ -d "$VAL" ]  || { echo "no val dataset at $VAL"; exit 1; }
mkdir -p "$OUT"
echo "phase0: $(basename "$CKPT")  val $(basename "$VAL")  horizon $HORIZON  steps [$STEPS]"

for n in $STEPS; do
    echo
    echo "=========== denoising $n  ($(date '+%F %T')) ==========="
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u \
        "$LOCOMANI_SRC/pipeline/eval_open_loop.py" \
        --model-path "$CKPT" --dataset "$VAL" \
        --horizons "$HORIZON" --denoising-steps "$n" \
        --out "$OUT/dn${n}_h${HORIZON}.json" \
        > "$OUT/dn${n}_h${HORIZON}.log" 2>&1 \
        && tail -20 "$OUT/dn${n}_h${HORIZON}.log" \
        || { echo "FAILED at n=$n"; tail -15 "$OUT/dn${n}_h${HORIZON}.log"; }
done

echo
echo "=========== summary ==========="
"$PY" - "$OUT" "$HORIZON" <<'PYEOF'
import json, sys, glob, os, re
out, h = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(glob.glob(f"{out}/dn*_h{h}.json"),
                key=lambda p: int(re.search(r"dn(\d+)_", p).group(1))):
    d = json.load(open(f))
    m = d["horizons"][h]["model"]
    rows.append((d["denoising_steps"], m))
if not rows:
    sys.exit("no results")
cols = ["std_ratio_token", "std_ratio_gripper", "grip_deadband_pred",
        "nmse_token", "nmse_token_unsnapped", "deg_mean", "hit_rate_per_dim"]
print(f"{'steps':>6} " + " ".join(f"{c:>20}" for c in cols))
for n, m in rows:
    print(f"{n:>6} " + " ".join(f"{m.get(c, float('nan')):>20.4f}" for c in cols))
gt = rows[0][1].get("grip_deadband_gt")
if gt is not None:
    print(f"\nground-truth gripper dead-band occupancy: {gt:.4f} "
          f"(the number a healthy prediction should match)")
print("snap check: nmse_token is SNAPPED, nmse_token_unsnapped is RAW; "
      "raw < snapped means the FSQ snap is costing accuracy at that setting.")
PYEOF
