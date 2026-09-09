"""Task registry (decision 3, 12): one place that knows every Task plugin."""

from __future__ import annotations

from locomani.tasks.base import Task
from locomani.tasks.bottle import TASK as _BOTTLE_TASK
from locomani.tasks.crate import TASK as _CRATE_TASK

TASKS: dict[str, Task] = {
    _BOTTLE_TASK.name: _BOTTLE_TASK,
    _CRATE_TASK.name: _CRATE_TASK,
}


def get_task(name: str) -> Task:
    try:
        return TASKS[name]
    except KeyError:
        raise KeyError(f"unknown task {name!r}; choices are {sorted(TASKS)}") from None
