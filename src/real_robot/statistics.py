"""What a checkpoint's normalization statistics say about its gripper, for the client's unit check.

A GR00T checkpoint carries `statistics.json`: per embodiment, per modality key, the min / max / mean / std / q01 /
q99 of the training data, which the processor normalizes with. The gripper's range there tells which unit system
the checkpoint was trained in -- simulator metres (-0.06 .. 0.0245) or Dex1 motor units (0 .. 5.5) -- which the
field names alone do not: `left_gripper` reads the same in both. The server puts the ranges into `identity`, the
client decides the units from them (`real_robot.gripper.units_of_range`) and refuses a contradiction before the
robot moves (the default `--gripper-units dex1` on a metres checkpoint closed both grippers at start, 2026-10-10).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

STATISTICS_FILES = ("statistics.json", os.path.join("experiment_cfg", "dataset_statistics.json"))
GRIPPER_WORDS = ("gripper", "hand")  # action keys that command a hand, state groups that observe one
QUANTILES = ("q01", "q99")


def gripper_ranges(model_path: str | os.PathLike, embodiment: str) -> dict | None:
    """`{"action": {key: {"q01": [...], "q99": [...]}}, "state": {...}}` for the hand and gripper keys of
    `embodiment` in the checkpoint's statistics, or None when the checkpoint carries none for it."""
    for name in STATISTICS_FILES:
        path = Path(model_path) / name
        if not path.exists():
            continue
        stats = json.loads(path.read_text()).get(embodiment)
        if not stats:
            continue
        out = {}
        for kind in ("action", "state"):
            keys = {k: v for k, v in (stats.get(kind) or {}).items() if any(w in k for w in GRIPPER_WORDS)}
            out[kind] = {k: {q: [float(x) for x in v[q]] for q in QUANTILES if q in v} for k, v in keys.items() if v}
        return out
    return None
