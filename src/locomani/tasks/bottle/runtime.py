"""Per-invocation bottle execution inputs; callers compose policies without module mutation."""

from dataclasses import dataclass, field
from typing import Callable

from locomani.tasks.bottle.config import BottleTaskConfig


@dataclass(frozen=True)
class GymRuntime:
    task: BottleTaskConfig = field(default_factory=BottleTaskConfig)
    policy: Callable | None = None
    robot_factory: Callable | None = None
    restart: Callable | None = None
