"""TODO 27.8.2 (plan docs/handoff/scene-rsi-plan-20261016.md §8, D13.3): the code half of an infeasibility claim's
evidence. A Reviewer's claim names a Python file that defines `check(start, meta) -> dict` with at least `holds` (bool;
optional `value`, `limit`, `note`): its hard constraint, computed from the start (the pool row plus `label`) and the
scene file, never typed in. The driver runs it here on the claim's diagnostic starts -- never on a hidden pool -- and
prints {label: result}; an exception counts as "does not hold".

    $PY scripts/rsi/claim_check.py <check.py> <scene_meta.json> <start.json> ...
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def run(check_py: Path, meta: dict, starts: list[Path]) -> dict:
    spec = importlib.util.spec_from_file_location("infeasibility_check", check_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = {}
    for p in starts:
        row = dict(json.loads(p.read_text()), label=p.stem)
        try:
            r = mod.check(row, meta)
            out[p.stem] = dict(holds=bool(r.get("holds")), **{k: r[k] for k in ("value", "limit", "note") if k in r})
        except Exception as e:  # noqa: BLE001 -- a broken check is a check that does not hold
            out[p.stem] = dict(holds=False, error=f"{type(e).__name__}: {e}")
    return out


def main(argv=None) -> int:
    a = sys.argv[1:] if argv is None else argv
    if len(a) < 3:
        raise SystemExit(__doc__)
    print(json.dumps(run(Path(a[0]), json.loads(Path(a[1]).read_text()), [Path(x) for x in a[2:]]), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
