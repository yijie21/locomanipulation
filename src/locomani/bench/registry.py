"""Benchmark name -> its capability table, read from the benchmark's own package (`locomani.bench.<name>.SPEC`).

The import is lazy, so resolving options for one benchmark never imports another's adapter.
"""

from __future__ import annotations

import importlib

from locomani.bench.spec import BenchSpec


def spec(name: str) -> BenchSpec:
    try:
        module = importlib.import_module(f"locomani.bench.{name}")
    except ModuleNotFoundError as e:
        raise ValueError(f"no benchmark package locomani.bench.{name}") from e
    s = getattr(module, "SPEC", None)
    if not isinstance(s, BenchSpec) or s.name != name:
        raise ValueError(f"locomani.bench.{name} has no SPEC for {name!r}")
    return s
