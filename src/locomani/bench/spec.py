"""What one benchmark can be run with: its capability table (ported from bench-options' TODO 49.1; selfwm TODO 38.11).

Each benchmark package (`locomani.bench.<name>`) owns one `BenchSpec`. For every option axis it lists the values the
benchmark supports and, per value, whether this codebase can run it yet:

    ready     implemented behind the bench interfaces and checked
    planned   the benchmark supports it, the adapter does not exist yet or has not been checked

A value missing from an axis is not supported by the benchmark (or not by this branch's bed for it); its `notes` entry,
when there is one, says why. `BenchOptions` resolution (options.py) checks a run's choices against this table before
any resource starts.

`couples` (selfwm's addition): on some beds one choice fixes another, e.g. on selfwm's SIMPLE bed the robot's belief
about its pelvis comes with the target's source (truth targets: the simulator's pelvis; perception: leg odometry), so
`self_pose` cannot be set on its own there. `{"target=truth": ("self_pose", "truth")}` says so; resolution sets the
coupled value when the run left it unset and refuses an explicit value that contradicts it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

READY, PLANNED = "ready", "planned"
STATUSES = (READY, PLANNED)

# the option axes a capability table covers, in the order a signature lists them (options.py owns their choices)
AXES = ("target", "sensor", "depth_noise", "self_pose", "controller_root", "controller")


@dataclass(frozen=True)
class BenchSpec:
    name: str  # the selector value (`bench: <name>`), also the package name under locomani.bench
    title: str  # what people call it
    about: str  # one line: robot, simulator, tasks
    supports: dict[str, dict[str, str]]  # axis -> {value: READY | PLANNED}
    notes: dict[str, str] = field(default_factory=dict)  # "<axis>=<value>" -> why missing / what is planned
    couples: dict[str, tuple[str, str]] = field(default_factory=dict)  # "<axis>=<value>" -> (axis, the value it fixes)

    def __post_init__(self):
        unknown = sorted(set(self.supports) - set(AXES))
        missing = sorted(set(AXES) - set(self.supports))
        if unknown or missing:
            raise ValueError(f"BenchSpec {self.name}: axes unknown {unknown}, missing {missing}")
        for axis, values in self.supports.items():
            bad = {v: s for v, s in values.items() if s not in STATUSES}
            if bad:
                raise ValueError(f"BenchSpec {self.name}.{axis}: statuses {bad} not in {STATUSES}")
        for key, (axis, value) in self.couples.items():
            a, _, v = key.partition("=")
            if a not in self.supports or v not in self.supports[a] or self.status(axis, value) is None:
                raise ValueError(f"BenchSpec {self.name}: coupling {key} -> {axis}={value} names an unsupported value")

    def status(self, axis: str, value: str) -> str | None:
        """READY, PLANNED, or None when the benchmark does not support `value` on `axis`."""
        return self.supports[axis].get(value)

    def why(self, axis: str, value: str) -> str:
        note = self.notes.get(f"{axis}={value}")
        return f" ({note})" if note else ""

    def table(self) -> list[tuple[str, str, str]]:
        """(axis, value, status) rows, for printing what a benchmark can run."""
        return [(a, v, s) for a in AXES for v, s in self.supports[a].items()]
