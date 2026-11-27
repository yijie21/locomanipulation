"""selfmodel_simplecmp_20261126: are two arms' rows the same episode for episode (the bit-identity checks); read-only.

    python scmp_cmp.py <rows a: file or dir> <rows b: file or dir> [--files A_,B_ (the file prefixes in dirs)]

Rows are matched by (task, ep). Compared: every key of the row and of each stage, except wall_s (the wall clock), the
--correct record (stages' "correct", the row's "correct_reset": what the switch adds) and --film's path fields. Printed per
episode: identical or the keys that differ, with each stage's err_cm / ik_cm side by side (the raw check output).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

SKIP_ROW = {"wall_s", "correct_reset", "film"}
SKIP_STAGE = {"correct"}


def rows_of(p: str, prefix: str) -> dict:
    files = sorted(glob.glob(f"{p}/{prefix}*.jsonl")) if Path(p).is_dir() else [p]
    out = {}
    for f in files:
        for line in open(f):
            r = json.loads(line)
            out[(r["task"], int(r["ep"]))] = r
    return out


def diff(x, y, path="") -> list[str]:
    if isinstance(x, dict) and isinstance(y, dict):
        d = []
        for k in sorted(set(x) | set(y)):
            if k not in x or k not in y:
                d.append(f"{path}{k} (only in {'b' if k not in x else 'a'})")
            else:
                d += diff(x[k], y[k], f"{path}{k}.")
        return d
    if isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            return [f"{path[:-1]} (length {len(x)} / {len(y)})"]
        d = []
        for i, (u, v) in enumerate(zip(x, y)):
            d += diff(u, v, f"{path}{i}.")
        return d
    return [] if x == y else [path[:-1]]


def clean(r: dict) -> dict:
    r = {k: v for k, v in r.items() if k not in SKIP_ROW}
    r["stages"] = [{k: v for k, v in s.items() if k not in SKIP_STAGE} for s in r.get("stages", [])]
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--files", default="", help="prefix_a,prefix_b of the jsonl files when a / b are dirs")
    g = ap.parse_args()
    pa, pb = (g.files.split(",") + [""])[:2] if g.files else ("", "")
    A, B = rows_of(g.a, pa), rows_of(g.b, pb)
    common = sorted(set(A) & set(B))
    print(f"a {g.a} ({len(A)} rows), b {g.b} ({len(B)} rows): {len(common)} episodes in both")
    n_same = 0
    for key in common:
        ra, rb = clean(A[key]), clean(B[key])
        d = diff(ra, rb)
        n_same += not d
        st = " | ".join(f"{sa.get('stage')}: err {sa.get('err_cm')} / {sb.get('err_cm')}, ik {sa.get('ik_cm')} / {sb.get('ik_cm')}"
                        for sa, sb in zip(ra["stages"], rb["stages"]))  # fmt: skip
        print(f"  {key[0]} ep {key[1]}: {'identical' if not d else 'DIFFERS in ' + ', '.join(d[:12]) + (' ...' if len(d) > 12 else '')}"
              f"; success_45 {ra.get('success_45')} / {rb.get('success_45')}; {st}")  # fmt: skip
    print(f"identical {n_same} / {len(common)} (compared: every row and stage key but {sorted(SKIP_ROW)} and stages' {sorted(SKIP_STAGE)})")


if __name__ == "__main__":
    main()
