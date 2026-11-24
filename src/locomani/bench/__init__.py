"""Benchmarks behind one interface, and the switches a run is made of (TODO 49, user 2026-11-24).

    options.py    BenchOptions (bench, target, sensor, depth_noise, self_pose, controller_root, controller) and the
                  one place they are resolved and checked
    spec.py       BenchSpec: a benchmark's capability table (which values it supports, which are ready here)
    registry.py   benchmark name -> its SPEC, imported lazily
    channels.py   what a benchmark session hands to perception and to the body (camera streams, the privileged
                  state that only the truth view, the judge and the records may read)
    <name>/       one package per benchmark: simple, locomanilab, bigym2, humanoidarena

Inspect a resolved run or the tables:  python -m locomani.bench show --bench simple --bench-set target=truth
"""

from locomani.bench.options import BENCHES, BenchOptions, Resolved, add_arguments, from_args, resolve

__all__ = ["BENCHES", "BenchOptions", "Resolved", "add_arguments", "from_args", "resolve"]
