#!/usr/bin/env bash
# ScaleBFM visual-expert arm on the 2026-09-10 no-retract 200-episode dataset.
#
#   stage 1  GR00T N1.7 finetune, seed 17, 3000 steps, GPU1
#   stage 2  registered closed loop, 100 ordinary-reset cases, seeds 31001:31101, GPU1
#   stage 3  stratified summary -> RESULTS.json
#
# Comparable arm: ledger scalebfm.md section 9 (yplus200, seed 17, 3000 steps, same 100-case
# pool) = 25/100. The only intended difference is the dataset (no post-lift retraction,
# success protocol sonic_lift6cm_hold2s_v1).
#
# This line runs a standalone MuJoCo sim in-process: no SONIC deploy, no stack, no ports,
# so it never touches instance 0. GPU1 only.
#
# Source of record: scripts/experiments/scalebfm200_train_eval_20260910.sh
# The EXECUTED copy is <OUT>/launch.sh. Never edit a running copy: bash reads a script by
# byte offset (memory never-edit-a-running-bash-script).
# Stop it by creating <OUT>/STOP; the runner kills the current child within 30 s.
set -euo pipefail

ROOT=/workspace/code/locomanipulation
NAME=scalebfm200_noretract_s17_20260910
OUT=$ROOT/outputs/experiments/$NAME
COLLECTION=$ROOT/outputs/experiments/scalebfm200_noretract_sonic_20260910
DATASET=$COLLECTION/dataset
ASSETS=$COLLECTION/runtime_assets.json
PY=$ROOT/third_party/Isaac-GR00T/.venv/bin/python
GPU=1
SEED=17
MODEL=$OUT/model/checkpoint-3000

mkdir -p "$OUT"
cd "$ROOT"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/runner.log"; }

status() {  # status <stage> <state>
  python3 - "$OUT/progress.json" "$1" "$2" "$$" <<'PY'
import json, sys, time, pathlib
p, stage, state, pid = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
d = json.loads(p.read_text()) if p.exists() else {}
d.update(stage=stage, state=state, runner_pid=pid, updated=time.time(),
         updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
p.write_text(json.dumps(d, indent=2) + "\n")
PY
}

gate() {  # refuse to start a stage on a machine that is not actually free
  [ -e "$OUT/STOP" ] && { log "STOP file present; not starting $1"; status "$1" stopped; exit 3; }
  local free
  free=$(nvidia-smi --id=$GPU --query-gpu=memory.free --format=csv,noheader,nounits)
  if [ "$free" -lt 28000 ]; then
    log "GPU$GPU has only ${free} MiB free; refusing to start $1"; status "$1" blocked_gpu; exit 4
  fi
  local headroom
  headroom=$(bash "$ROOT/scripts/cpu_headroom.sh" 3 | sed -n 's/.*free \([0-9.]*\).*/\1/p')
  if [ "$(python3 -c "print(1 if float('${headroom:-0}') >= 4.0 else 0)")" != "1" ]; then
    log "CPU headroom ${headroom} cores; refusing to start $1"; status "$1" blocked_cpu; exit 5
  fi
  local avail
  avail=$(df -BG --output=avail /workspace | tail -1 | tr -dc '0-9')
  if [ "$avail" -lt 80 ]; then
    log "/workspace has only ${avail}G free; refusing to start $1"; status "$1" blocked_disk; exit 6
  fi
  log "gate ok for $1: GPU$GPU ${free} MiB free, ${headroom} cores, ${avail}G disk"
}

run_child() {  # run_child <stage> <logfile> -- <command...>
  local stage=$1 logfile=$2; shift 3
  "$@" >> "$logfile" 2>&1 &
  local child=$!
  python3 - "$OUT/progress.json" "$stage" "$child" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1]); d = json.loads(p.read_text())
d["child_pid"] = int(sys.argv[3]); d["child_stage"] = sys.argv[2]
p.write_text(json.dumps(d, indent=2) + "\n")
PY
  while kill -0 "$child" 2>/dev/null; do
    if [ -e "$OUT/STOP" ]; then
      log "STOP file appeared; killing $stage child $child"
      kill "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true
      status "$stage" stopped; exit 3
    fi
    sleep 30
  done
  wait "$child"
}

log "=== $NAME start (runner pid $$) ==="

# ---------------------------------------------------------------- stage 1: train
if [ -d "$MODEL" ]; then
  log "stage 1 skipped: $MODEL already exists"
else
  gate train
  status train running
  RESUME=()
  LAST=$(ls -d "$OUT"/model/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1 || true)
  if [ -n "$LAST" ]; then
    log "resuming from checkpoint-$LAST"
    RESUME=(--resume "$OUT/model/checkpoint-$LAST")
  fi
  log "stage 1: finetune seed $SEED, 3000 steps, GPU$GPU -> $OUT/model"
  run_child train "$OUT/train.log" -- env \
    CUDA_VISIBLE_DEVICES=$GPU \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    "$PY" -u -m locomani.tasks.bottle_visual.pipeline.train \
      --dataset "$DATASET" --out "$OUT/model" --seed $SEED "${RESUME[@]}"
  [ -d "$MODEL" ] || { log "training finished but $MODEL is missing"; status train failed; exit 7; }
  log "stage 1 done"
fi
status train complete

# ---------------------------------------------------------------- stage 2: closed loop
if [ -f "$OUT/eval100/summary.json" ]; then
  log "stage 2 skipped: $OUT/eval100/summary.json already exists"
else
  gate eval
  status eval running
  log "stage 2: 100-case closed loop, seeds 31001:31101, yplus_center_v1, no recovery perturbation"
  run_child eval "$OUT/eval.log" -- env \
    CUDA_VISIBLE_DEVICES=$GPU \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=$GPU \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    "$PY" -u -m locomani.tasks.bottle_visual.pipeline.eval \
      --model "$MODEL" \
      --out "$OUT/eval100" \
      --seeds 31001:31101 \
      --recovery-episodes 0 \
      --scene-profile yplus_center_v1 \
      --runtime-assets "$ASSETS"
  log "stage 2 done"
fi
status eval complete

# ---------------------------------------------------------------- stage 3: summary
log "stage 3: stratified summary"
python3 - "$OUT" <<'PY' | tee -a "$OUT/runner.log"
import collections, json, pathlib, sys
out = pathlib.Path(sys.argv[1])
summary = json.loads((out / "eval100/summary.json").read_text())
cases = summary["cases"]
by = collections.defaultdict(lambda: [0, 0])
reasons = collections.Counter()
for c in cases:
    s = by[c.get("stratum", "?")]
    s[1] += 1
    s[0] += bool(c["ok"])
    if not c["ok"]:
        reasons[c.get("why") or "unknown"] += 1
result = dict(
    experiment=out.name,
    dataset="scalebfm200_noretract_sonic_20260910",
    train=dict(seed=17, steps=3000, gpu=1),
    eval=dict(cases=summary["count"], seeds="31001:31101", scene_profile="yplus_center_v1",
              recovery_episodes=0),
    successes=summary["successes"],
    by_stratum={k: f"{v[0]}/{v[1]}" for k, v in sorted(by.items())},
    failure_reasons=dict(reasons.most_common()),
    reference="ledger scalebfm.md section 9: yplus200 seed17/3000 = 25/100 (10/34 high, 10/33 low_near, 5/33 low_deep)",
)
(out / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY
status summary complete
log "=== $NAME complete ==="
