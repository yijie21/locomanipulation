"""Which line a run belongs to, and the two things the lines genuinely do differently.

Single-arm and bimanual share the stack, the converter, the finetune and the open-loop eval; they
differ in how a run is frozen before the stack starts (scene name, which configs the expert reads),
in where the collected artifacts sit (`<run>/collect/` versus beside the run), and in where state
traces go (`<run>/collect/state_trace` versus `<run>/trace`) -- the crate line's expert is handed
that directory on its command line, so the two must not drift apart. Everything that
depends on those two answers asks here, so the shared entries carry a lookup rather than an
`if plugin == ...` branch -- the form that hides "declared in the config, never delivered", this
repository's most expensive failure shape.

Adding a line means adding a row here and the four functions above in its task package.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable


@dataclasses.dataclass(frozen=True)
class Line:
    name: str
    prepare: Callable[[dict, Path], Path]  # freeze scene + expert configs, idempotent; returns the scene
    collect_root: Callable[[Path], Path]  # where manifests and videos go
    trace_root: Callable[[Path], Path]  # where the sim writes state traces, and where the expert is told
    scene_name: Callable[[dict, Path], Path]  # what the frozen scene will be called, without freezing it


def _bottle() -> Line:
    from locomani.tasks.bottle import prepare as p

    return Line(
        name="single_arm",
        prepare=p.prepare,
        collect_root=p.collect_root,
        trace_root=p.trace_root,
        scene_name=p.scene_name,
    )


def _crate() -> Line:
    from locomani.tasks.crate import prepare as p

    return Line(
        name="bimanual",
        prepare=p.prepare,
        collect_root=p.collect_root,
        trace_root=p.trace_root,
        scene_name=p.scene_name,
    )


BUILDERS = {  # task plugin -> the line it belongs to
    "bottle": _bottle,
    "pick_place": _bottle,
    "bottle_visual": _bottle,
    "crate": _crate,
}


def line_of(cfg: dict) -> Line:
    plugin = cfg["task"]["plugin"]
    if plugin not in BUILDERS:
        raise SystemExit(f"task.plugin '{plugin}' belongs to no line; known: {', '.join(sorted(BUILDERS))}")
    return BUILDERS[plugin]()
