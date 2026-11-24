"""python -m locomani.bench.simple [--bench-set KEY=VALUE ...]: the SIMPLE bed's settings for a run's switches, as shell
assignments (BED_SIGNATURE, BED_COLUMN, BED_CLOUD, BED_LOOK, BED_CONTROLLER, BED_ROOT) for the runners to `eval`.
An impossible or unready combination prints the reason to stderr and exits 2."""

from __future__ import annotations

import argparse
import shlex
import sys

from locomani.bench import options
from locomani.bench.simple import bed_settings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m locomani.bench.simple", description=__doc__)
    options.add_arguments(ap)
    a = ap.parse_args(argv)
    try:
        r = options.resolve(a.bench or "simple", a.bench_set)
        bed = bed_settings(r.options)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"BED_SIGNATURE={shlex.quote(r.options.signature())}")
    for k, v in bed.items():
        print(f"BED_{k.upper()}={shlex.quote(v)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
