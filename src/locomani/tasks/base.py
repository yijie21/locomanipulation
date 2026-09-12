"""Task entries share typed outcomes and pure acceptance; each owner defines its execution inputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from locomani.pipeline.execution_record import RoundResult
from locomani.schema import TaskSchema


def accept_result(result: RoundResult, *, require_recording: bool = False) -> bool:
    """Inspect a completed task result; artifact audits have their own entry."""
    if not isinstance(result, RoundResult):
        raise TypeError("acceptance takes a completed RoundResult")
    return result.accepted(require_recording=require_recording)


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
    run_round: Callable[..., RoundResult]
    acceptance: Callable[..., bool]
    schema: TaskSchema
    classify_failure: Callable[[str], str]
    failure_reasons: tuple[str, ...]
