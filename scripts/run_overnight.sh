#!/bin/bash
# One unattended chain: gate -> collect 200 -> convert @25 fps -> audit ->
# train -> open-loop (SELECTOR ONLY) -> closed-loop 20 -> resume to 100.
#
#   nohup bash scripts/run_overnight.sh > outputs/logs/overnight.log 2>&1 &
#
# Written for a run nobody is watching, so every stage is judged by its
# ARTEFACT rather than by an exit code, the one stage that can waste hours of
# machine time has a gate in front of it, and the three settings that fail
# SILENTLY are written here instead of remembered:
#
#   SCENE_PRESET=bottle_only  -- without it build_sim_scene makes the default
#                                multi-object table and all 200 episodes are
#                                of the wrong scene.
#   TF_DATASET_FPS=25         -- the checkpoint is trained on 25 fps data now
#                                (convert_to_hf_format's DST_FPS). Evaluating
#                                it at 30 replays every motion 1.2x fast and
#                                raises nothing.
#   --seed 6000               -- the seed round 3 (fast200, 28.0%) and round 4
#                                used. Same seed = same fresh spawns = the
#                                comparison is PAIRED. A different seed makes
#                                the number incomparable to the ladder.
#
# The report is appended stage by stage, so an interrupted run still explains
# how far it got.
set -uo pipefail

# shellcheck disable=SC1091
. "$(dirname "$0")/paths.sh"
cd "$LOCOMANI_ROOT" || exit 1

RUN="${RUN:-ramp200}"
BATCHES="${BATCHES:-10}"
ROUNDS="${ROUNDS:-20}"
SEED0="${SEED0:-1}"
GATE_N="${GATE_N:-12}"
GATE_MIN="${GATE_MIN:-9}"
DISK_MIN_GB="${DISK_MIN_GB:-80}"
EVAL_SEED="${EVAL_SEED:-6000}"

TRAIN_DS="$LOCOMANI_ROOT/datasets/g1_locomani_${RUN}_train"
VAL_DS="$LOCOMANI_ROOT/datasets/g1_locomani_${RUN}_val"
CKPT_DIR="$LOCOMANI_OUTPUTS/train/$RUN"
EVAL_DIR="$LOCOMANI_OUTPUTS/eval/$RUN"
COLLECT_OUT="$LOCOMANI_OUTPUTS/logs/collect_$RUN"
REPORT="$LOCOMANI_OUTPUTS/logs/${RUN}_report.md"
GR00T_PY="$LOCOMANI_ROOT/third_party/Isaac-GR00T/.venv/bin/python"
MODALITY="$LOCOMANI_ROOT/configs/gr00t_sonic_2cam.py"
SESS_ROOT="$(locomani_session_root)/outputs"

export SCENE_PRESET=bottle_only
export TARGETS=water_bottle
export JITTER=0.20
# The spawn distance is the one axis worth stating twice, because it decides
# whether the closed-loop number is on the ladder. 0.7 is where 5.9/13.7/28.0/
# 47.5% were all measured. Anything else has to be evaluated at the SAME
# distance it was trained at, and the report says so.
SPAWN_EDGE="${SPAWN_EDGE:-0.7}"
export SPAWN_ARGS="${SPAWN_ARGS:---spawn-edge $SPAWN_EDGE --face-object --spawn-yaw-err 20 --spawn-lateral 0.20}"

mkdir -p "$EVAL_DIR" "$COLLECT_OUT" "$(dirname "$REPORT")"
say() { echo; echo "=========== $* ($(date '+%F %T')) ==========="; }

# Lifted from run_round4.sh, which is the only orchestration in the repo that
# has actually driven a full closed-loop run.

note() { echo "$*" >> "$REPORT"; }
die() { say "STOPPED: $*"; note ""; note "**停在这里**: $* ($(date '+%F %T'))"; exit 1; }

: > "$REPORT"
note "# $RUN — 无人值守跑（$(date '+%F %T') 开始）"
note ""
if [ "$SPAWN_EDGE" = "0.7" ]; then
    note "对标 **round 3 = fast200 = 28.0%**（200 局纯脚本、零 DAgger，seed 6000）。"
else
    note "出生距桌沿 **$SPAWN_EDGE m**，不是阶梯的 0.7 m —— 训练和评测都在这个距离上，"
    note "所以这个闭环数字 **不在 5.9/13.7/28.0/47.5% 那条阶梯上**，不能和 28.0% 直接比。"
fi
note "round 4 的 47.5% 含 50 局靶向 DAgger，不是同构对象。"
note ""
note "本批相对 fast200 变了四件事，**无法单独归因**：上身接管 ramp、"
note "DST_FPS 30→25、外观随机化关闭、腕相机后移。"
note ""

# --- 0. gate ------------------------------------------------------------------
# 12 rounds before spending 4-6 hours. The repo's rule for any new axis, and
# this batch changes four at once.
say "0/6 gate: $GATE_N rounds, need >= $GATE_MIN"
PYTHONPATH=src "$VENV_SIM_PY" -m locomani.stack.cli down --instance "${INSTANCE:-0}" > /dev/null 2>&1
"$VENV_SIM_PY" src/locomani/sim/build_sim_scene.py \
    outputs/scene outputs/scene/sim --seed "$SEED0" --preset "${SCENE_PRESET:-}" \
    | grep -E '^\[scene\]' \
    || die "gate: scene rebuild failed"
nohup env SIM_VIDEO_DIR="$LOCOMANI_OUTPUTS/sim_video" \
    PYTHONPATH=src "$VENV_SIM_PY" -m locomani.stack.cli up --instance "${INSTANCE:-0}" --gpu "${CUDA_VISIBLE_DEVICES:-0}" --prompt "pick up the water bottle" \
    > "$COLLECT_OUT/stack_gate.log" 2>&1 &
for _ in $(seq 1 120); do
    grep -qiE "stack up|FAILED|Traceback" "$COLLECT_OUT/stack_gate.log" && break
    sleep 5
done
grep -qi "stack up" "$COLLECT_OUT/stack_gate.log" || die "gate: stack did not come up"
# shellcheck disable=SC2086
"$VENV_SIM_PY" -u src/locomani/tasks/bottle/grasp_gym.py --rounds "$GATE_N" \
    --targets "$TARGETS" --edge-jitter "$JITTER" $SPAWN_ARGS \
    > "$COLLECT_OUT/gate.log" 2>&1
GATE_OK=$(grep -c "^EPISODE OK" "$COLLECT_OUT/gate.log" || true)
PYTHONPATH=src "$VENV_SIM_PY" -m locomani.stack.cli down --instance "${INSTANCE:-0}" > /dev/null 2>&1
note "## 0. 12 局闸门"
note ""
note "\`$GATE_OK/$GATE_N\`（阀值 $GATE_MIN）"
if [ "$GATE_OK" -lt "$GATE_MIN" ]; then
    note ""
    note "失败原因分布："
    note '```'
    grep -oE "^EPISODE FAILED \[[a-z]+\]" "$COLLECT_OUT/gate.log" | sort | uniq -c >> "$REPORT"
    note '```'
    die "gate $GATE_OK/$GATE_N < $GATE_MIN -- not spending 4-6 h on this"
fi
say "gate passed: $GATE_OK/$GATE_N"

# --- 1. collect ---------------------------------------------------------------
# Mark the session root first: the converter pairs sessions with manifests
# positionally, and a stale or empty session dir silently shifts that join.
say "1/6 collecting $BATCHES x $ROUNDS"
MARK="$COLLECT_OUT/.mark"; touch "$MARK"
T0=$(date +%s)
OUT="$COLLECT_OUT" bash scripts/collect.sh "$BATCHES" "$ROUNDS" "$SEED0" \
    2>&1 | tee "$COLLECT_OUT/collect.log"
T1=$(date +%s)
SESSIONS=$(find "$SESS_ROOT" -maxdepth 1 -mindepth 1 -type d -newer "$MARK" \
           | sort | tr '\n' ' ')
MANIFESTS=$(find "$COLLECT_OUT" -maxdepth 1 -name 'manifest_*.jsonl' | sort -V | tr '\n' ' ')
NS=$(echo "$SESSIONS" | wc -w); NM=$(echo "$MANIFESTS" | wc -w)
OKN=$(grep -h -c "^EPISODE OK" "$COLLECT_OUT"/gym_*.log 2>/dev/null | paste -sd+ | bc)
note ""
note "## 1. 采集"
note ""
note "成功 **$OKN** 局，耗时 $(( (T1-T0)/60 )) 分钟，$NS 个 session / $NM 个 manifest。"
[ "$NS" -eq "$NM" ] || die "collect: $NS sessions but $NM manifests -- the join would be out of phase"
[ "$NS" -gt 0 ] || die "collect: no sessions produced"

# --- 2. convert @ DST_FPS (25) ------------------------------------------------
say "2/6 converting (holdout 20)"
# shellcheck disable=SC2086
"$VENV_DATA_PY" "$LOCOMANI_SRC/pipeline/convert_to_hf_format.py" \
    --sessions $SESSIONS --manifest $MANIFESTS \
    --out "$TRAIN_DS" --holdout 20 --holdout-out "$VAL_DS" \
    2>&1 | tee "$COLLECT_OUT/convert.log"
[ -f "$TRAIN_DS/meta/info.json" ] || die "convert: no train dataset"
[ -f "$VAL_DS/meta/info.json" ] || die "convert: no val dataset"
FPS=$("$VENV_DATA_PY" -c "import json;print(json.load(open('$TRAIN_DS/meta/info.json'))['fps'])")
NEP=$(ls "$TRAIN_DS/data/chunk-000" | wc -l)
note "转完：**$NEP 局 @ ${FPS} fps**（+ 20 局 holdout）。"
[ "$FPS" = "25" ] || note "> ⚠ fps 不是 25，闭环的 TF_DATASET_FPS 跟着错了。"

# --- 3. stats + audit ---------------------------------------------------------
say "3/6 statistics and audit"
for ds in "$TRAIN_DS" "$VAL_DS"; do
    [ -f "$ds/meta/stats.json" ] && continue
    "$GR00T_PY" -m gr00t.data.stats --dataset-path "$ds" \
        --embodiment-tag UNITREE_G1_SONIC --modality-config-path "$MODALITY" | tail -2
done
AUDIT_OK=1
for ds in "$TRAIN_DS" "$VAL_DS"; do
    "$VENV_DATA_PY" "$LOCOMANI_SRC/pipeline/check_dataset.py" "$ds" || AUDIT_OK=0
done
[ "$AUDIT_OK" = 1 ] || die "audit failed -- not spending GPU hours on it"
note "审计通过。"

# --- 4. train -----------------------------------------------------------------
FREE_GB=$(df -BG --output=avail "$LOCOMANI_WORKSPACE" | tail -1 | tr -dc '0-9')
note ""
note "## 2. 训练"
note ""
note "开始前空闲磁盘 ${FREE_GB} GB。"
[ "$FREE_GB" -ge "$DISK_MIN_GB" ] || die "only ${FREE_GB} GB free, need $DISK_MIN_GB -- 10 checkpoints are ~65 GB"
say "4/6 training"
T2=$(date +%s)
# To a file, not a pipe to tail: a 2 h stage with no visible progress is a
# 2 h stage nobody can tell from a hung one.
RUN="$RUN" DATA="$TRAIN_DS" bash scripts/train_bottle.sh \
    > "$COLLECT_OUT/train.log" 2>&1
tail -40 "$COLLECT_OUT/train.log"
T3=$(date +%s)
ls -d "$CKPT_DIR"/checkpoint-* > /dev/null 2>&1 || die "train: no checkpoints in $CKPT_DIR"
note "训完，耗时 $(( (T3-T2)/60 )) 分钟，$(ls -d "$CKPT_DIR"/checkpoint-* | wc -l) 个 checkpoint。"

# --- 5. pick the checkpoint --------------------------------------------------
# The LONGEST-TRAINED one, by the user's standing decision (2026-08-29).
#
# What that gives up: GR00T asserts eval_strategy == "no", so there is no
# validation loss and nothing stops training early. An open-loop sweep over the
# held-out split used to choose the stopping point after the fact, and it did
# change the answer -- fast200's picked checkpoint-9000 over 10000. What it
# cost was ~2 h, because eval_open_loop's default --horizons is `1 8 16 40`
# while the selector only ever read h16; passing --horizons 16 would have made
# it ~13 min. The decision was taken with that number on the table.
say "5/6 checkpoint"
BEST_CKPT=$(ls -d "$CKPT_DIR"/checkpoint-* | sort -t- -k2 -n | tail -1)
[ -n "$BEST_CKPT" ] || die "no checkpoints in $CKPT_DIR"
echo "$BEST_CKPT" > "$EVAL_DIR/best_checkpoint.txt"
note "评的是 **$(basename "$BEST_CKPT")**（训练最久的那个，不跑开环选择器）。"

# --- 6. closed-loop -----------------------------------------------------------
say "6/6 closed-loop"
RUN="$RUN" CKPT="$BEST_CKPT" EVAL_DIR="$EVAL_DIR" EVAL_SEED="$EVAL_SEED" \
    SPAWN_EDGE="$SPAWN_EDGE" \
    REPORT="$REPORT" DATASET_FPS=25 \
    bash "$LOCOMANI_ROOT/scripts/run_closed_loop.sh"

say "done"
note ""
note "跑完：$(date '+%F %T')"
note ""
note "产物：数据集 \`$TRAIN_DS\` / \`$VAL_DS\`，checkpoint \`$CKPT_DIR\`，评测 \`$EVAL_DIR\`。"
echo "report: $REPORT"
