"""The primitive-level contract every entry in `skills/catalogue/` satisfies (design doc §4).

`Violation` is a machine-readable precondition failure the agent can act on; `Outcome` is what a
primitive's `run` returns; `Primitive` is the Protocol the executor calls against, so a new primitive
needs no base class, only this shape. Failure reasons follow the `[tag] message` convention of
`tasks/base.bracket_label`, which `scripts/gate.py` and every task's `failure_reasons` already group by.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from locomani.tasks.base import bracket_label

if TYPE_CHECKING:
    from locomani.skills.session import ArmOwner, Session


@dataclass(frozen=True)
class Violation:
    """One precondition that failed: what was checked, what was measured, the allowed band."""

    check: str  # "stance_depth", "arm_owner", "heading", "unimplemented" ...
    measured: float | str | None = None
    bound: tuple = (None, None)  # (low, high); None = unbounded on that side
    note: str = ""

    def to_dict(self) -> dict:
        return {"check": self.check, "measured": _plain(self.measured), "bound": list(self.bound), "note": self.note}


@dataclass
class Outcome:
    """What a primitive's `run` returns: success, a tagged reason, what it recorded and what it produced."""

    ok: bool
    reason: str = ""  # "[tag] message" on failure, "" on success
    stages: list = field(default_factory=list)  # record stages entered during the run
    metrics: dict = field(default_factory=dict)
    duration_s: float = 0.0
    produced: Any = None  # approach -> Stance; grasp -> HeldObject

    @property
    def tag(self) -> str:
        return bracket_label(self.reason)

    def to_dict(self) -> dict:
        produced = self.produced
        if is_dataclass(produced) and not isinstance(produced, type):
            produced = asdict(produced)
        return {
            "ok": self.ok,
            "reason": self.reason,
            "tag": self.tag,
            "stages": _plain(self.stages),
            "metrics": _plain(self.metrics),
            "duration_s": float(self.duration_s),
            "produced": _plain(produced) if produced is None or isinstance(produced, (dict, list)) else repr(produced),
        }


@runtime_checkable
class Primitive(Protocol):
    """The executor calls only this. `requires` are the arm owners a call may start from; `yields` is the
    owner a successful call leaves: one `ArmOwner`, None (unchanged) or a frozenset of owners when the
    same primitive legitimately ends in more than one (approach: the walk hands the arms to the planner
    in guard mode, keeps them readied in depth mode). Keyword arguments are the plan's `args`."""

    name: str
    requires: frozenset
    yields: "ArmOwner | frozenset | None"

    def preconditions(self, session: "Session", **args) -> list[Violation]:
        """Pure: reads the observation and the scene, sends nothing."""
        ...

    def run(self, session: "Session", **args) -> Outcome:
        """Blocking: drives the stack, mutates `session`, returns the result."""
        ...


def tag_of(reason: str) -> str:
    return bracket_label(reason)


def _plain(value: Any) -> Any:
    """JSON-safe copy: numpy, enums, dataclasses, paths and nested containers."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Violation):
        return value.to_dict()
    if isinstance(value, Outcome):
        return value.to_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _plain(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    return str(value)
