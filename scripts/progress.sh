#!/bin/bash
# One-glance progress for whatever is running. Usage:
#   bash scripts/progress.sh              # snapshot
#   watch -n 30 bash scripts/progress.sh  # live, refresh every 30 s
cd "$(dirname "$0")/.." || exit 1
echo "== $(date +%H:%M:%S) =="

# round-4 driver stage
R4LOG=outputs/logs/round4.log
SEQ=/workspace/tmp/seq_swap1.log
[ -f "$SEQ" ] && tail -1 "$SEQ" | sed "s/^/  seq_ab: /"
[ -f outputs/eval/seq_ens_swap1/seq_verdict.json ] && \
    python3 -c "import json;d=json.load(open('outputs/eval/seq_ens_swap1/seq_verdict.json'));print(f\"  上次look: n={d['pairs']}  A {d['a_ok']} vs B {d['b_ok']}  +B{d['only_b']}/+A{d['only_a']}  p={d['p']:.3f} -> {d['verdict']}\")"
[ -f "$R4LOG" ] && grep "^===========" "$R4LOG" | tail -1 | sed 's/^=*/  stage:/;s/=*$//'

# closed-loop runs, live round count from their manifests
for m in outputs/eval/seq_ens_swap1/a.manifest.jsonl \
         outputs/eval/seq_ens_swap1/b.manifest.jsonl \
         outputs/eval/fast250/closed_loop.manifest.jsonl; do
    [ -f "$m" ] || continue
    python3 - "$m" <<'PY'
import json, sys, collections
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
ok=sum(1 for r in rows if r['ok'])
n=len(rows)
bar='#'*(n//2)+'.'*((100-n)//2)
st=collections.Counter()
for r in rows:
    if not r['ok']:
        st[r.get('why','').split(':')[0].replace('[vla] ','')]+=1
name=sys.argv[1].split('/')[-1].replace('.manifest.jsonl','')
print(f"  {name:<14} [{bar}] {n}/100  ok {ok} ({100*ok/max(n,1):.0f}%)  失败 {dict(st)}")
PY
done
echo "  参照: round4(无集成)=48/101=47.5%  round3=28/100"

# training, if any
t=$(grep -oE "[0-9]+/10000 \[[0-9:]+<[0-9:]+" "$R4LOG" 2>/dev/null | tail -1)
[ -n "$t" ] && ! grep -q "7/8" <(grep "^===========" "$R4LOG" | tail -1) && echo "  train: $t"

# processes
p=$(pgrep -f "eval_closed_loop" | head -1)
[ -n "$p" ] && echo "  eval 进程: pid $p ($(ps -o etime= -p "$p" | tr -d ' '))"
d=$(pgrep -f "$(printf 'run_%s.sh' round4)" | head -1)
[ -n "$d" ] && echo "  driver:   pid $d 在跑" || echo "  driver:   已退出 (完成或出错 — 看 $R4LOG 尾部)"
