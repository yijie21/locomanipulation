# notebook: selfmodel_simplesonicdata_20261126 -- read only: this entry's A (H20b, CPU nets) vs the main metric's SONIC A
# (bench_rgbd_20261124/sonic, 4090, GPU nets), per task and time limit, with the episodes that flip
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import scmp_eval as SE  # noqa: E402

run, main = Path(sys.argv[2]), Path(sys.argv[3])
A, S = SE.rows_of(run / "rows", "A"), SE.rows_of(main / "rows", "A")
print(f"this A: {run}/rows ({len(A)} rows); main SONIC A: {main}/rows ({len(S)} rows)")
for lab, col in SE.LIMITS:
    print(f"-- {lab}")
    ta = ts = 0
    for t in SE.TASKS:
        ks = [(t, e) for e in range(30)]
        a = sum(bool(A[k].get(col)) for k in ks)
        s = sum(bool(S[k].get(col)) for k in ks)
        up = [k[1] for k in ks if A[k].get(col) and not S[k].get(col)]
        dn = [k[1] for k in ks if S[k].get(col) and not A[k].get(col)]
        ta, ts = ta + a, ts + s
        print(f"  {SE.SHORT[t]:14s}: this A {a:2d}, main {s:2d} (diff {a - s:+d}); success only here {len(up)} {up}, only in main {len(dn)} {dn}")
    print(f"  all           : this A {ta}, main {ts} (diff {ta - ts:+d})")
