"""selfmodel_freespace_20261126 revision 3: E2a's final tally over the three controllers (read-only on the records).

    cd $G; PYTHONPATH=$G:$C0/src FS_OUT=$O $PY fs_tally.py

Expected jobs (revision 2's plan): ScaleBFM 216 cells x ids 0..15 x A / S / R; HEFT 216 x 0..15 x A / R plus the
step-2 cells (heft/stage2_cells.txt, fs_heft2.py) x 16..34 x A / R; SONIC 216 x 0..21 x A / R. Per controller: records,
duplicate keys (a job with two records; fs_load.runs), missing and unexpected jobs. Prints TALLY OK when every controller
holds exactly its expected jobs, each once; else TALLY FAIL. A record with a worker error counts as run (fs_run records
it and does not retry)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fs_load as FL  # noqa: E402

O = Path(os.environ["FS_OUT"])
GROUPS = {"scalebfm": ("alone", "selfmodel", "reachable"), "heft": ("alone", "reachable"), "sonic": ("alone", "reachable")}


def expected(be: str) -> set | None:
    cells = range(216)
    if be == "scalebfm":
        jobs = {(c, e) for c in cells for e in range(16)}
    elif be == "sonic":
        jobs = {(c, e) for c in cells for e in range(22)}
    else:
        f = O / "heft" / "stage2_cells.txt"
        if not f.exists():
            return None  # step 2 never selected: the expected list is undefined
        sel = [int(x) for x in f.read_text().strip().split(",") if x]
        jobs = {(c, e) for c in cells for e in range(16)} | {(c, e) for c in sel for e in range(16, 35)}
    return {(c, e, m) for c, e in jobs for m in GROUPS[be]}


ok, tot_n, tot_x = True, 0, 0
for be in ("scalebfm", "heft", "sonic"):
    E, dup = FL.runs(O / be)
    want = expected(be)
    if want is None:
        ok = False
        print(f"{be}: {len(E) + len(dup)} records; no heft/stage2_cells.txt, so the expected list is undefined: FAIL")
        continue
    miss, extra = want - set(E), set(E) - want
    errs = sum(r.get("error") is not None for r in E.values())
    good = not miss and not extra and not dup
    ok &= good
    tot_n += len(E) + len(dup)
    tot_x += len(want)
    print(f"{be}: {len(E) + len(dup)} records, expected {len(want)}; duplicates {len(dup)}, missing {len(miss)}, unexpected {len(extra)}, "
          f"worker errors {errs}: {'ok' if good else 'FAIL'}" + (f"; e.g. missing {sorted(miss)[:3]} unexpected {sorted(extra)[:3]} dup {dup[:3]}" if not good else ""))
print(f"{'TALLY OK' if ok else 'TALLY FAIL'}: {tot_n} records over the three controllers, expected {tot_x}")
sys.exit(0 if ok else 1)
