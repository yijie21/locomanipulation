"""Per-attempt outcomes and control-stage timestamps, independent of actuation.

Times describe entry into control code, not sensor exposure or physical arrival.
The journal never drives a transition. Sensor/action traces retain their own clocks.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import time
from typing import Callable


@dataclass
class ExecutionRecord:
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    wall_clock: Callable[[], float] = field(default=time.time, repr=False)
    metrics: dict = field(default_factory=dict)
    transition: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    stage: str = 'created'
    first_failure: dict | None = None

    def __post_init__(self):
        self.started_wall = self.wall_clock()
        self._started = self.clock()

    def enter(self, stage: str, **details):
        self.stage = stage
        self.events.append(dict(stage=stage, elapsed_s=self.clock()-self._started,
                                wall_time=self.wall_clock(), **deepcopy(details)))

    def fail(self, reason: str):
        if self.first_failure is None:
            self.first_failure = dict(stage=self.stage, reason=reason,
                                      elapsed_s=self.clock()-self._started)

    def snapshot(self):
        return deepcopy(dict(schema_version=1, clock='monotonic_elapsed+unix_wall',
                             timestamp_semantics='control_stage_entry',
                             started_wall=self.started_wall, stage=self.stage,
                             first_failure=self.first_failure, events=self.events,
                             metrics=self.metrics, transition=self.transition))


@dataclass(frozen=True)
class GraspResult:
    ok: bool
    why: str
    execution: dict

    @property
    def visibility(self):
        return self.execution['metrics'].get('ego_visibility')


@dataclass(frozen=True)
class RoundResult:
    ok: bool
    why: str
    produced: bool
    execution: dict

    def __iter__(self):
        # Compatibility with existing evaluation/diagnostic tuple unpacking.
        return iter((self.ok, self.why, self.produced))
