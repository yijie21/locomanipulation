# notebook: selfmodel_simplesonicdata_20261126 -- read only: per task and arm, how many episodes' stage records of run G
# and run T are identical to run C's (the float path's reach), and how many success outcomes (45 s) agree with C's
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import scmp_eval as SE  # noqa: E402

O = Path(sys.argv[2])
C = {arm: SE.rows_of(O / "run" / "rows", arm) for arm in ("A", "RD")}
for tag in ("g", "t"):
    X = {arm: SE.rows_of(O / f"run_{tag}" / "rows", arm) for arm in ("A", "RD")}
    for arm in ("A", "RD"):
        cells = []
        for t in SE.TASKS + ("all",):
            ks = [k for k in X[arm] if t in ("all", k[0])]
            same = sum(SE.clean(X[arm][k])["stages"] == SE.clean(C[arm][k])["stages"] for k in ks)
            agree = sum(bool(X[arm][k].get("success_45")) == bool(C[arm][k].get("success_45")) for k in ks)
            cells.append(f"{SE.SHORT.get(t, t)} {same}/{len(ks)} ({agree})")
        print(f"run {tag.upper()} {arm:2s} vs run C: stages identical (45 s outcome agrees): " + ", ".join(cells))
