#!/bin/bash
# Live progress of closed-loop evaluations. Usage:
#   bash scripts/eval_progress.sh clean100 [base100 ...]   # snapshot
#   watch -n 30 bash scripts/eval_progress.sh clean100      # live
cd "$(dirname "$0")/.." || exit 1
echo "== $(date +%H:%M:%S) =="
for name in "${@:-clean100}"; do
    M=outputs/eval/$name/closed_loop.manifest.jsonl
    [ -f "$M" ] || { echo "$name: no manifest yet"; continue; }
    python3 - "$name" "$M" <<'PY'
import json, sys, collections
name, m = sys.argv[1], sys.argv[2]
rs = [json.loads(l) for l in open(m)]
real = [r for r in rs if "[view]" not in r["why"]]
ok = sum(r["ok"] for r in real); n = len(real)
print(f"{name}: {ok}/{n} OK = {ok/max(n,1):.0%}   (attempts {len(rs)}, view-gate skips {len(rs)-n})")
c = collections.Counter(r["why"].split("(")[0].strip()[:48] for r in real if not r["ok"])
for why, k in c.most_common(6):
    print(f"   {k:3d}  {why}")
if real:
    r = real[-1]; print(f"   last: round {r['round']} {'OK' if r['ok'] else 'FAILED'} in {r.get('seconds','?')}s")
PY
done
