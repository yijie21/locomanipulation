"""BiGym 2.0 bed knobs (TODO 49.5): paths, the render size, depth on or off, the seed pools.

One frozen dataclass, parsed at the entry (`python -m locomani.bench.bigym2 run ... --set key=value`) and passed on; nothing in the
bed reads `os.environ` for a switch.

Seed pools. BiGym's evaluation block is seeds [620000, 630000) (`bigym/loco/eval/protocol.py`: episode i of the official 100 is
620000 + i, the block reserved with room for offset blocks). The adapter draws from three pools that never touch it and never touch
the demonstration seeds 1-199 either:

    P  practice    1000 + n      n < 9000      development: tune and look
    A  acceptance  100000 + n    n < 1000      a candidate's gate
    S  sealed      700000 + n    n < 10000     opened once, for the final number
    E  the official block        never run

The refusal is in five places: `validate_scope` (this file), the CLI before a server starts, `episodes.check_id`,
`episodes.reset_msg`, and the environment server (started with the block as arguments). Tests: tests/test_bigym2_episodes.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from locomani.config_base import apply_overrides, config_dump, config_from_dict, config_load, config_to_dict

SCHEMA_VERSION = 1

_MAIN = "/workspace/code/locomanipulation"  # the main checkout: third_party (symlinked from every worktree) lives there
EVAL_BLOCK = (620000, 630000)  # BiGym's reserved evaluation seeds, [lo, hi)
DEMO_SEEDS = (1, 200)  # the 60 demonstrations per task were collected on seeds in [1, 200)
POOLS = {  # letter -> (first seed, size)
    "P": (1000, 9000),
    "A": (100000, 1000),
    "S": (700000, 10000),
}
SEED_IGNORING = (
    "dishwasher_close",
    "wall_cupboard_open",
    "wall_cupboard_close",
)  # their reset does not consume the seed (paper "Seeds")
KINDS = "".join(POOLS)


def pool_ranges_ok() -> list[str]:
    """What is wrong with the pools' ranges against the evaluation block and the demonstration seeds (empty: nothing)."""
    bad = []
    for k, (lo, n) in POOLS.items():
        hi = lo + n
        if lo < EVAL_BLOCK[1] and hi > EVAL_BLOCK[0]:
            bad.append(f"pool {k} [{lo}, {hi}) overlaps the evaluation block {EVAL_BLOCK}")
        if lo < DEMO_SEEDS[1] and hi > DEMO_SEEDS[0]:
            bad.append(f"pool {k} [{lo}, {hi}) overlaps the demonstration seeds {DEMO_SEEDS}")
    return bad


@dataclass(frozen=True)
class BiGym2Config:
    pools: str = "P"  # pools of KINDS this run may open (letters); E, the official block, is never allowed
    depth: bool = (
        False  # render depth: the server applies the depth patch to BiGym's CameraConfig (needs the head camera only)
    )
    render_shape: tuple[int, int] = (
        240,
        320,
    )  # (H, W) of the head camera when depth is on; BiGym's own is 84 x 84 without it
    max_steps: int = 0  # control steps (50 Hz) per episode; 0 = the task's own budget
    bigym_root: str = (
        _MAIN + "/third_party/BiGym2"
    )  # the benchmark, cloned at a fixed commit (docs/handoff/bench-options-20261124.md)
    bigym_python: str = _MAIN + "/third_party/BiGym2/.venv/bin/python"  # its own venv: MuJoCo 3.8.1, Python 3.12
    server_file: str = ""  # an alternative server script ("" = the one in this package)
    run_dir: str = ""  # server logs and the lock; "" = the system's temporary folder
    lock_file: str = ""  # one BiGym simulator at a time on this machine; "" = <run_dir or tmp>/bigym2.lock
    lock_wait_s: float = 3600.0
    server_ready_s: int = 120  # wait for the environment server's socket
    gpu: int = 0  # the server's EGL device
    seed_skills: str = (
        ""  # package folder when --skills is not given; "" = the seed package inside locomani.bench.bigym2
    )

    def server_script(self) -> str:
        return self.server_file or str(Path(__file__).resolve().parent / "server.py")

    def lock_path(self) -> str:
        import tempfile

        return self.lock_file or str(Path(self.run_dir or tempfile.gettempdir()) / "bigym2.lock")

    def validate_scope(self) -> None:
        """Refuse a configuration that could run the evaluation block (raises ValueError)."""
        bad = sorted(set(self.pools) - set(KINDS))
        if bad or not self.pools:
            raise ValueError(f"pools {self.pools!r}: letters of {KINDS} only; E (the official block) is never allowed")
        problems = pool_ranges_ok()
        if problems:
            raise ValueError("; ".join(problems))
        if len(self.render_shape) != 2 or min(self.render_shape) < 16:
            raise ValueError(f"render_shape {self.render_shape!r} is not (H, W)")

    def to_dict(self) -> dict:
        return config_to_dict(self, SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, data: dict) -> BiGym2Config:
        return config_from_dict(cls, data, SCHEMA_VERSION)

    @classmethod
    def load(cls, path) -> BiGym2Config:
        return config_load(cls, path)

    def dump(self, path) -> None:
        config_dump(self, path)

    def with_overrides(self, overrides: list[str]) -> BiGym2Config:
        return apply_overrides(self, overrides)


SEED_DIR = Path(__file__).resolve().parent / "seed"  # the seed skill package
SCOPE = dict(
    target=("truth",),
    sensor=("none",),
    depth_noise=("none",),
    self_pose=("truth",),
    controller_root=("truth",),
    controller=("native",),
)  # what this bed implements: the rgbd channel is built (Cameras.rgbd, --set depth=true) but only perception reads it (planned)


class ScopeError(ValueError):
    """A switch value outside what the BiGym 2.0 bed implements."""


def check_scope(options) -> None:
    """Refuse the switch values this bed does not implement, naming why."""
    why = {
        (
            "target",
            "perception",
        ): "a detector on BiGym's head camera is not built (planned); the rgbd channel itself exists",
        ("self_pose", "odometry"): "leg odometry needs foot loads BiGym does not expose as sensors (planned)",
        ("controller_root", "odometry"): "the native controller reads the benchmark's own state",
        ("controller", "scalebfm"): "decided 2026-11-23: BiGym keeps its own GR00T-WBC controller",
    }
    bad = []
    for axis, ok in SCOPE.items():
        got = getattr(options, axis)
        if got not in ok and not (axis in ("sensor", "depth_noise") and got == "none"):
            bad.append(f"{axis}={got} -> {why.get((axis, got), 'not implemented')}")
    if bad:
        raise ScopeError(
            "the BiGym 2.0 bed runs target=truth self_pose=truth controller_root=truth controller=native: "
            + "; ".join(bad)
        )
