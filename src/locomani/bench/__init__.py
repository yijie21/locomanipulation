"""Benchmarks behind one set of run switches (ported from the bench-options branch, its TODO 49; selfwm TODO 38.11).

    options.py    BenchOptions (bench, target, sensor, depth_noise, self_pose, controller_root, controller) and the
                  one place they are resolved and checked
    spec.py       BenchSpec: a benchmark's capability table (which values it supports, which are ready here, which
                  choices fix another on this branch's bed)
    registry.py   benchmark name -> its SPEC, imported lazily
    <name>/       one package per benchmark: simple, locomanilab, bigym2, humanoidarena (on selfwm a table only; the
                  beds themselves are the experiment runners named in each table, e.g. SIMPLE's simple_bench.py)

bench-options' channels.py and perception/ (camera streams, views) are not ported: selfwm's runners read cached looks.

Inspect a resolved run or the tables:  python -m locomani.bench show --bench simple --bench-set target=truth
"""

from locomani.bench.options import BENCHES, BenchOptions, Resolved, add_arguments, from_args, resolve

__all__ = ["BENCHES", "BenchOptions", "Resolved", "add_arguments", "from_args", "resolve"]
