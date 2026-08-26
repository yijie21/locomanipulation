#!/bin/bash
# One-glance progress for whatever is running. Usage:
#   bash scripts/progress.sh          # snapshot
#   watch -n 30 bash scripts/progress.sh   # live, refresh every 30 s
cd "$(dirname "$0")/.." || exit 1
echo "== $(date +%H:%M:%S) =="

# tier-1 A/B sweep
for f in outputs/eval/tier1/*.manifest.jsonl; do
    [ -f "$f" ] || continue
    n=$(basename "$f" .manifest.jsonl)
    python3 - "$f" "$n" <<'PY'
import json, sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
ok=sum(1 for r in rows if r['ok'])
bar='#'*len(rows)+'.'*(50-len(rows))
print(f"  {sys.argv[2]:<8} [{bar}] {len(rows)}/50  ok {ok} ({100*ok/max(len(rows),1):.0f}%)")
PY
done
echo "  (基线: 吸附+全预算 前50轮 = 13/50 = 26%)"

# any closed-loop eval elsewhere
p=$(pgrep -f "eval_closed_loop" | head -1)
[ -n "$p" ] && echo "  eval 进程: pid $p ($(ps -o etime= -p "$p" | tr -d ' ') elapsed)" \
            || echo "  eval 进程: (无 — 两组都完成或已停)"
