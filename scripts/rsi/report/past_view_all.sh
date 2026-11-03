#!/bin/bash
# TODO 39.7 (user 2026-11-02): every round of one folder as a scene-graph page (past_view.py), plus an index.html that
# links them with each round's outcome. Defaults: English pages with the MuJoCo frames (EGL device 0, SRA0889's GPU).
#
#   bash scripts/rsi/report/past_view_all.sh <folder> [--out DIR] [--lang en|zh] [--no-frames] [--retired] [--gpu N] [--scene DIR]
#
# <folder> holds <label>.json / .npz / .past.jsonl, e.g. a sandbox's
#   outputs/scene_rsi_20261016/<family>/diag/incumbent_<hash>/
# --out    default outputs/scene_rsi_20261016/past_view/<sandbox>__<folder name>/ (<folder name> alone outside a sandbox)
# --lang   en (default) or zh
# --no-frames  maps and relation lists only: no MuJoCo, seconds per round instead of ~10 s
# --retired    also the `_retired_*` rounds a diagnostic-pool redraw set aside (left out by default)
# --gpu    MUJOCO_EGL_DEVICE_ID for the frames (default 0; on SRA0889 device 1 is the Intel iGPU)
# --scene  the scene directory with scene_meta.json (default outputs/scene_discovery/adt_apartment_mars)
# A round whose page fails is listed in the index as failed and the run goes on; the exit code is 1 when any failed.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PY="${VENV_SIM_PY:-$REPO/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python}"

DIR="" OUT="" LANG_="en" FRAMES=1 RETIRED=0 GPU=0 SCENE="$REPO/outputs/scene_discovery/adt_apartment_mars"
while [ $# -gt 0 ]; do
    case "$1" in
        --out) OUT="${2:?--out needs a directory}"; shift 2 ;;
        --lang) LANG_="${2:?--lang needs en or zh}"; shift 2 ;;
        --no-frames) FRAMES=0; shift ;;
        --retired) RETIRED=1; shift ;;
        --gpu) GPU="${2:?--gpu needs a device index}"; shift 2 ;;
        --scene) SCENE="${2:?--scene needs a directory}"; shift 2 ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        -*) echo "unknown option $1 (see --help)" >&2; exit 1 ;;
        *) [ -z "$DIR" ] && DIR="$1" || { echo "one folder only" >&2; exit 1; }; shift ;;
    esac
done
[ -n "$DIR" ] || { echo "usage: $(basename "$0") <folder> [--out DIR] [--lang en|zh] [--no-frames] [--retired] [--gpu N] [--scene DIR]" >&2; exit 1; }
[ -d "$DIR" ] || { echo "no such folder: $DIR" >&2; exit 1; }
case "$LANG_" in en|zh) ;; *) echo "--lang must be en or zh" >&2; exit 1 ;; esac
DIR="$(cd "$DIR" && pwd)"

if [ -z "$OUT" ]; then
    name="$(basename "$DIR")"
    case "$DIR" in
        */code/rsi-scene/*) sb="${DIR#*/code/rsi-scene/}"; name="${sb%%/*}__$name" ;;
    esac
    OUT="$REPO/outputs/scene_rsi_20261016/past_view/$name"
fi
mkdir -p "$OUT"

shopt -s nullglob
rounds=()
for f in "$DIR"/*.past.jsonl; do
    base="$(basename "$f" .past.jsonl)"
    case "$base" in _retired_*) [ "$RETIRED" = 1 ] || continue ;; esac
    rounds+=("$base")
done
[ ${#rounds[@]} -gt 0 ] || { echo "no .past.jsonl in $DIR" >&2; exit 1; }

frames_arg=()
[ "$FRAMES" = 1 ] && frames_arg=(--frames)
suffix=""; [ "$LANG_" = en ] && suffix=".en"
echo "${#rounds[@]} rounds -> $OUT (lang $LANG_, frames $([ "$FRAMES" = 1 ] && echo on || echo off))"

failed=()
i=0
for base in "${rounds[@]}"; do
    i=$((i + 1))
    page="$OUT/$base$suffix.html"
    if MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$GPU" PYTHONPATH="$REPO/src" "$PY" "$HERE/past_view.py" "$DIR/$base" \
        --lang "$LANG_" --scene "$SCENE" "${frames_arg[@]}" --out "$page" > "$OUT/.log_$base" 2>&1; then
        echo "[$i/${#rounds[@]}] $base"
        rm -f "$OUT/.log_$base"
    else
        echo "[$i/${#rounds[@]}] $base FAILED (log $OUT/.log_$base)" >&2
        failed+=("$base")
    fi
done

# the index: one row per round with its outcome, failure class and step count, linking its page
FAILED_LIST="${failed[*]:-}" "$PY" - "$DIR" "$OUT" "$LANG_" "$suffix" "${rounds[@]}" <<'EOF'
import html, json, os, sys
from pathlib import Path
d, out, lang, suffix, labels = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5:]
failed = set(os.environ.get("FAILED_LIST", "").split())
zh = lang == "zh"
H = (("局", "任务", "结果", "失败类型", "步骤", "页面") if zh else ("round", "task", "outcome", "failure class", "steps", "page"))
rows, n_ok = [], 0
for lab in labels:
    rec = json.loads((d / f"{lab}.json").read_text()) if (d / f"{lab}.json").exists() else {}
    recs = [json.loads(x) for x in (d / f"{lab}.past.jsonl").read_text().splitlines() if x.strip()]
    steps = sum("record_id" in r for r in recs)
    ok = bool(rec.get("ok"))
    n_ok += ok
    page = f"{lab}{suffix}.html"
    link = ("页面生成失败" if zh else "page failed") if lab in failed else f'<a href="{html.escape(page)}">{"打开" if zh else "open"}</a>'
    verdict = ("成功" if zh else "OK") if ok else ("失败" if zh else "FAILED")
    rows.append(f'<tr class="{"ok" if ok else "bad"}"><td>{html.escape(lab)}</td><td>{html.escape(str(rec.get("task", "")))}</td>'
                f'<td>{verdict}</td><td>{html.escape(str(rec.get("failure_class") or ""))}</td><td>{steps}</td><td>{link}</td></tr>')
title = f"{d.name}"
summary = (f"{n_ok} / {len(labels)} 局成功；来源 {d}" if zh else f"{n_ok} / {len(labels)} rounds succeeded; from {d}")
css = (":root{--bg:#fbfbfa;--fg:#1d1f23;--dim:#6b7079;--line:#d9dbe0;--ok:#2f9e44;--bad:#d4372c}"
       "@media (prefers-color-scheme: dark){:root:not([data-theme=light]){--bg:#16181d;--fg:#e7e9ee;--dim:#9aa1ad;--line:#2c3038}}"
       "body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}"
       "table{border-collapse:collapse;width:100%}td,th{padding:4px 8px;border-bottom:1px solid var(--line);text-align:left}"
       "th{color:var(--dim);font-weight:500}tr.ok td:nth-child(3){color:var(--ok)}tr.bad td:nth-child(3){color:var(--bad)}"
       "a{color:inherit}.dim{color:var(--dim)}.tw{overflow-x:auto}")
head = "".join(f"<th>{h}</th>" for h in H)
(out / "index.html").write_text(
    f"<!doctype html><html lang='{lang}'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
    f"<title>{html.escape(title)}</title><style>{css}</style></head><body><h1 style='font-size:18px'>{html.escape(title)}</h1>"
    f"<p class='dim'>{html.escape(summary)}</p><div class='tw'><table><tr>{head}</tr>{''.join(rows)}</table></div></body></html>")
print(f"{out / 'index.html'}  ({n_ok} / {len(labels)} OK)")
EOF

if [ ${#failed[@]} -gt 0 ]; then
    echo "${#failed[@]} page(s) failed: ${failed[*]}" >&2
    exit 1
fi
