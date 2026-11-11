"""Pair two sets of sealed-pool rounds by start (selfwm item 12): wins each, the starts each wins alone with their
failure classes, and the one-sided McNemar p that the second is worse.

    python3 adt_compare.py <base rounds dir> <other rounds dir> [<more dirs> ...]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def rounds(d: Path) -> dict:
    out = {}
    for f in sorted(d.glob("pick_*.json")):
        r = json.loads(f.read_text())
        steps = {s["step"]: s for s in r.get("steps", [])}
        rel = steps.get("rel_entry") or steps.get("rel_descend") or {}
        out[r["label"]] = dict(ok=bool(r["ok"]), cls=r.get("failure_class"), fell=bool(r.get("fell")),
                               inside=rel.get("inside"), rel_cm=rel.get("rel_cm"))  # fmt: skip
    return out


def p_worse(c: int, b: int) -> float:
    """One-sided exact McNemar: P(at least c of the c + b discordant starts go against the second)."""
    n = b + c
    return 1.0 if n == 0 else sum(math.comb(n, k) for k in range(c, n + 1)) / 2**n


def main(argv) -> None:
    base = rounds(Path(argv[0]))
    print(f"{argv[0]}: {sum(r['ok'] for r in base.values())}/{len(base)}")
    for d in argv[1:]:
        other = rounds(Path(d))
        keys = sorted(set(base) & set(other))
        b = [k for k in keys if other[k]["ok"] and not base[k]["ok"]]
        c = [k for k in keys if base[k]["ok"] and not other[k]["ok"]]
        print(f"{d}: {sum(other[k]['ok'] for k in keys)}/{len(keys)} vs base {sum(base[k]['ok'] for k in keys)}"
              f"  wins alone {len(b)}, losses alone {len(c)}, p(worse) {p_worse(len(c), len(b)):.3f}, "
              f"p(better) {p_worse(len(b), len(c)):.3f}")  # fmt: skip
        for k in b:
            print(f"   + {k[5:]:34s} base {base[k]['cls']}")
        for k in c:
            print(f"   - {k[5:]:34s} now {other[k]['cls']}  inside {other[k]['inside']} rel {other[k]['rel_cm']}")


if __name__ == "__main__":
    main(sys.argv[1:])
