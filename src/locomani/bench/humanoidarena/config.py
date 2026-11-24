"""HumanoidArena bed knobs (TODO 49.5b): paths, the GPU, the seed pools.

One frozen dataclass, parsed at the entry (`python -m locomani.bench.humanoidarena run ... --set key=value`) and passed on; nothing in
the bed reads `os.environ` for a switch.

Seeds. The benchmark's evaluation episodes are, per task, the 60 seeds sha256("<gym task>|<g>|<r>") (low 31 bits of the last four
bytes) for group g in {0, 1, 2} and repeat r in 0..19 (script/eval_scripts/*/eval_vla_suite.py). An episode is identified here by a
(group, repeat) pair of a pool whose groups never include 0, 1, 2; its seed comes from the same formula, so a pool episode is
a fresh draw of the benchmark's own episode distribution:

    P  practice    groups 1000-1999   (n -> group 1000 + n // 20, repeat n % 20)
    A  acceptance  groups 2000-2999
    S  sealed      groups 3000-3999
    E  the evaluation groups {0, 1, 2}   never run

The refusal is in six places: `validate_scope` (this file), the CLI before a server starts, `episodes.check_id`, `episodes.reset_msg`,
`episodes.collides` (the derived seed against the 60), and the environment server (its own copy of the formula). Tests:
tests/test_humanoidarena_episodes.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from locomani.config_base import apply_overrides, config_dump, config_from_dict, config_load, config_to_dict

SCHEMA_VERSION = 1

_MAIN = "/workspace/code/locomanipulation"  # the main checkout: third_party (symlinked from every worktree) lives there
EVAL_GROUPS = (0, 1, 2)  # the benchmark's group seeds
EVAL_REPEATS = 20
POOLS = {  # letter -> (first group, groups in the pool); 20 episodes per group
    "P": (1000, 1000),
    "A": (2000, 1000),
    "S": (3000, 1000),
}
KINDS = "".join(POOLS)
PER_GROUP = EVAL_REPEATS


def pool_ranges_ok() -> list[str]:
    """What is wrong with the pools' group ranges against the evaluation groups (empty: nothing)."""
    bad = []
    for k, (lo, n) in POOLS.items():
        if any(lo <= g < lo + n for g in EVAL_GROUPS):
            bad.append(f"pool {k} groups [{lo}, {lo + n}) include an evaluation group {EVAL_GROUPS}")
    return bad


@dataclass(frozen=True)
class HumanoidArenaConfig:
    pools: str = "P"  # pools of KINDS this run may open (letters); E, the evaluation groups, is never allowed
    max_steps: int = 0  # control steps (50 Hz) per episode; 0 = the task's own budget (1450 / 2000 / 1800)
    ha_root: str = (
        _MAIN + "/third_party/HumanoidArena"
    )  # the benchmark, cloned at a fixed commit (docs/handoff/bench-options-20261124.md)
    ha_python: str = (
        _MAIN + "/third_party/HumanoidArena/venv/bin/python"
    )  # its own venv: Python 3.11, Isaac Sim 5.0.0, Isaac Lab 2.2.0
    server_file: str = ""  # an alternative server script ("" = the one in this package)
    run_dir: str = ""  # server logs and the lock; "" = the system's temporary folder
    lock_file: str = (
        ""  # one HumanoidArena simulator at a time on this machine; "" = <run_dir or tmp>/humanoidarena.lock
    )
    lock_wait_s: float = 3600.0
    server_ready_s: int = 900  # wait for the environment server's socket (Kit's first start compiles shaders: minutes)
    gpu: int = 1  # the GPU the server may use (CUDA_VISIBLE_DEVICES and Kit's activeGpu index)
    cores: str = "28-31"  # taskset list for the server (the container's quota is 15 cores)
    snap_dir: str = ""  # when set, the third-person camera's picture at every mark of the package is saved there (for a person to look at)
    attach: str = ""  # development: the socket of a server already running (server.py --keep); the bed neither starts nor stops it
    overlay_dir: str = (
        ""  # when set, the first rgbd frame of an episode is saved there as <id>.png with the truth overlaid
    )
    seed_skills: str = ""  # package folder when --skills is not given; "" = the seed package inside this module

    def server_script(self) -> str:
        return self.server_file or str(Path(__file__).resolve().parent / "server.py")

    def lock_path(self) -> str:
        import tempfile

        return self.lock_file or str(Path(self.run_dir or tempfile.gettempdir()) / "humanoidarena.lock")

    def validate_scope(self) -> None:
        """Refuse a configuration that could run the evaluation groups (raises ValueError)."""
        bad = sorted(set(self.pools) - set(KINDS))
        if bad or not self.pools:
            raise ValueError(
                f"pools {self.pools!r}: letters of {KINDS} only; E (the evaluation groups) is never allowed"
            )
        problems = pool_ranges_ok()
        if problems:
            raise ValueError("; ".join(problems))
        if not (isinstance(self.gpu, int) and self.gpu >= 0):
            raise ValueError(f"gpu {self.gpu!r}")

    def to_dict(self) -> dict:
        return config_to_dict(self, SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, data: dict) -> HumanoidArenaConfig:
        return config_from_dict(cls, data, SCHEMA_VERSION)

    @classmethod
    def load(cls, path) -> HumanoidArenaConfig:
        return config_load(cls, path)

    def dump(self, path) -> None:
        config_dump(self, path)

    def with_overrides(self, overrides: list[str]) -> HumanoidArenaConfig:
        return apply_overrides(self, overrides)


SEED_DIR = Path(__file__).resolve().parent / "seed"  # the seed skill package
SCOPE = dict(
    target=("truth",),
    sensor=("none",),
    depth_noise=("none",),
    self_pose=("truth",),
    controller_root=("truth",),
    controller=("native",),
)  # what this bed implements: the rgbd channel is built (Cameras.rgbd) but only perception reads it (planned)


class ScopeError(ValueError):
    """A switch value outside what the HumanoidArena bed implements."""


def check_scope(options) -> None:
    """Refuse the switch values this bed does not implement, naming why."""
    why = {
        (
            "target",
            "perception",
        ): "a detector on the front camera is not wired (planned); the rgbd channel itself exists",
        ("self_pose", "odometry"): "leg odometry needs foot loads (planned)",
        ("controller_root", "odometry"): "the native tracker reads the benchmark's own state",
        ("controller", "scalebfm"): "ScaleBFM runs in MuJoCo; the benchmark's physics is Isaac Lab's",
    }
    bad = []
    for axis, ok in SCOPE.items():
        got = getattr(options, axis)
        if got not in ok and not (axis in ("sensor", "depth_noise") and got == "none"):
            bad.append(f"{axis}={got} -> {why.get((axis, got), 'not implemented')}")
    if bad:
        raise ScopeError(
            "the HumanoidArena bed runs target=truth self_pose=truth controller_root=truth controller=native: "
            + "; ".join(bad)
        )
