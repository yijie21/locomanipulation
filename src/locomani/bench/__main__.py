"""python -m locomani.bench show [--bench NAME] [--bench-set KEY=VALUE ...]   the resolved switches and the table
python -m locomani.bench table                                             every benchmark's capability table"""

from __future__ import annotations

import argparse
import json
import sys

from locomani.bench import options, registry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m locomani.bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = ap.add_subparsers(dest="cmd", required=True)
    show = sub.add_parser("show", help="resolve a run's switches (planned values allowed) and print them")
    options.add_arguments(show)
    sub.add_parser("table", help="print every benchmark's capability table")
    a = ap.parse_args(argv)
    if a.cmd == "show":
        try:
            r = options.from_args(a, allow_planned=True)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(json.dumps(r.record(), indent=1))
        names = [r.options.bench]
    else:
        names = list(options.BENCHES)
    for name in names:
        spec = registry.spec(name)
        print(f"\n{spec.title} ({name}): {spec.about}")
        for axis, value, status in spec.table():
            print(f"  {axis:16s} {value:10s} {status}{spec.why(axis, value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
