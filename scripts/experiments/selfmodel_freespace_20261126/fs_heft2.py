"""selfmodel_freespace_20261126 revision 2: HEFT's two-step episode count (read-only on the records).

    cd $G; PYTHONPATH=$G:$C0/src FS_OUT=$O FS_GRID=$O/grid/grid.json $PY fs_heft2.py --eps 0:16 --out $O/heft/stage2

Step 1 ran every cell with episode ids 0..15. A cell goes on to step 2 (ids 16..34) when those 16 episodes hold at least
one valid pair (fs_eval.valid on both A and R: no worker error, no fall / lost, a settled landing). The rule looks only
at whether a valid pair exists, never at which group did better. Cells with none stay at 16 and count as 局数不够.
Refuses (exit 1) when step 1 is incomplete (a job of 216 cells x the ids x A / R without a record).
Writes <out>_cells.txt (the step-2 cells, comma-separated) and <out>.json (per cell: valid pairs in step 1, selected).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fs_eval as FE  # noqa: E402
import fs_grid as FG  # noqa: E402
import fs_load as FL  # noqa: E402
import fs_run as FR  # noqa: E402


def main(eps: range, out: Path, only: list[int] | None = None) -> int:
    backend = "heft"
    E, dup = FL.runs(FR.OUT / backend)
    cells = [c["cid"] for c in FG.cells() if only is None or c["cid"] in only]  # only: a test on the smoke's cells
    groups = FR.GROUPS[backend]
    missing = [(cid, e, m) for cid in cells for e in eps for m in groups if (cid, e, m) not in E]
    if missing:
        print(f"heft step 1 incomplete: {len(missing)} jobs without a record, e.g. {missing[:5]}")
        return 1
    per = {}
    for cid in cells:
        n = sum(FE.valid(E[(cid, e, "alone")]) and FE.valid(E[(cid, e, "reachable")]) for e in eps)
        per[cid] = dict(valid_pairs=int(n), selected=n >= 1)
    sel = [cid for cid in cells if per[cid]["selected"]]
    by_post = {}
    for c in FG.cells():
        if c["cid"] not in per:
            continue
        p = by_post.setdefault(c["posture"], [0, 0])
        p[0] += per[c["cid"]]["selected"]
        p[1] += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{out}_cells.txt").write_text(",".join(map(str, sel)) + "\n")
    Path(f"{out}.json").write_text(json.dumps(dict(eps=[eps.start, eps.stop], duplicates=len(dup), cells=per), indent=1))
    zero = len(cells) - len(sel)
    print(f"heft step 1 (ids {eps.start}..{eps.stop - 1}): {len(E)} records, {len(dup)} duplicates; cells with >= 1 valid pair "
          f"{len(sel)} / {len(cells)} (go on to step 2), with none {zero} (stay at {len(eps)}: 局数不够); per posture "
          f"{ {k: f'{v[0]}/{v[1]}' for k, v in by_post.items()} }")  # fmt: skip
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", required=True, help="step 1's ids a:b")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cells", default=None, help="test only: a comma list instead of all 216")
    a = ap.parse_args()
    x, y = map(int, a.eps.split(":"))
    sys.exit(main(range(x, y), a.out, [int(c) for c in a.cells.split(",")] if a.cells else None))
