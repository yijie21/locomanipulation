"""The Task plugin seam (decision 3, 12): five pieces bottle and crate both fill in.

`Task` is a thin descriptor -- it holds references to functions and constants
that already exist elsewhere; nothing here re-implements them. The pipeline
skeleton (stack lifecycle, gym loop, recording, convert, train, eval) stays
generic by only ever calling through a `Task`, never importing a task module
directly.

Contract (docs/reference/standards.md), enforced by tests/test_task_contract.py:
  * `config_cls` round-trips through to_dict/from_dict/dump/load
    (locomani/config_base.py gives every config dataclass these for free).
  * `run_round` is the scripted expert's round entry: a function of the
    observation state, the round's scene sample, and the task's config. It
    must not read wall-clock time or keep hidden state across ticks --
    everything a round needs varies through its arguments, not through
    globals or `time.time()`.
  * `classify_failure` must produce a real label (not `None`, not the string
    `"unknown"`) for every prefix in `failure_reasons`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class TaskSchema:
    """The GR00T modality config a task trains against, and its column names.

    `modality_config_path` is a repo-relative path to the `--modality_config_path`
    file (e.g. `configs/gr00t_sonic_2cam.py`); the three key tuples are copied
    from that file's `_cfg["video"/"state"/"action"].modality_keys` rather than
    imported from it, because importing it requires the `gr00t` package (only
    installed in the GR00T/train venv, not the `.venv_sim` these unit tests run
    under) and it has import-time side effects (mutates the global embodiment
    registry, prints a summary line).
    """

    modality_config_path: str
    video_keys: tuple[str, ...]
    state_keys: tuple[str, ...]
    action_keys: tuple[str, ...]


def bracket_label(why: str) -> str:
    """`"[stand] fell during the approach"` -> `"stand"`.

    Every task's acceptance path already writes its `why` strings in this
    `[tag] message` convention (grasp_gym.py, bimanual_carry.py; scripts/gate.py's
    own `_norm_reason` keys off the same tag). Returns `"ok"` for an empty/falsy
    `why` and `"other"` for a `why` with no recognisable tag.
    """
    if not why:
        return "ok"
    m = re.match(r"^\[(\w+)\]", why)
    return m.group(1) if m else "other"


@dataclass(frozen=True)
class Task:
    name: str
    config_cls: type
    sample_scene: Callable
    run_round: Callable
    acceptance: Callable
    schema: TaskSchema
    classify_failure: Callable[[str], str]
    failure_reasons: tuple[str, ...]
