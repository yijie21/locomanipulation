"""LocoManiLab bed knobs (TODO 49.4): every former environment switch and hard-coded path of the escalation LML beds, with today's value.

One frozen dataclass, parsed at the entry (`python -m locomani.bench.locomanilab ... --set key=value`) and passed on; nothing in
the bed reads `os.environ` for a switch. The defaults are what the RSI loops ran with: the practice pools under the esc worktree's
outputs (data only, read through these fields), the slot locks SHARED with the esc runners (so old and new runs together share
the slots), LocoManiLab's own venv for the environment server, SAM 3 on port 18832.

Mapping from the old environment (details in lml_port_notes.md):
LML_LOOP=1 -> kinds (only the listed pools run; the benchmark's own episodes, kind E, never) + seed_min; LML_PRACTICE ->
practice_dir / practice_dir_fridge / practice_dir_drawer (one per task family, families.py); LML_RUN_DIR -> run_dir;
LML_SLOTS -> slots; LML_SERVER -> server_file; LML_SKILLS -> the --skills argument; CAM_CLOUD -> cache_dir (+ a key folder);
TMPDIR -> sock_dir; CUDA_VISIBLE_DEVICES of the worker -> gpu; SAM3_PORT -> sam3_port. CAM_COMPLETE (package mode: "none"),
CAM_PNG / CAM_AZ (record-only pictures) and BFM_ROOT (dead) are gone.
The bed's fixed settings (feed_torque: the loop's `bed_fixed`) stay settings of a run, as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from locomani.config_base import apply_overrides, config_dump, config_from_dict, config_load, config_to_dict

SCHEMA_VERSION = 1

_MAIN = "/workspace/code/locomanipulation"  # the main checkout: third_party (symlinked from every worktree) lives there
_ESC = "/workspace/code/locomanipulation-esc"  # the esc worktree: data under outputs/ is read only
_DATA = _ESC + "/outputs/escalation_20261116"
KINDS = "QPW"  # episode kinds a run can be allowed: Q the loop's pool (practice rows), P development (seed dev_seed), W wide development
PRACTICE_SEED = 1000  # kind P: the benchmark's own randomisation with this jitter seed (it is not an evaluation seed)
WIDE = dict(
    seed=2000, robot=0.30, objects=0.08
)  # kind W: the same with the robot drawn in 0.30 m and the objects in 0.08 m
EVAL_EPISODES = (
    20  # the benchmark's own episodes are jitter seed 0, episodes 0-19: nothing under this number is ever run
)


@dataclass(frozen=True)
class LocoManiLabConfig:
    # --- episodes (data files, read only)
    kinds: str = (
        "Q"  # episode kinds this run may open, as letters of "QPW"; E (the benchmark's own episodes) is never allowed
    )
    seed_min: int = 100000  # the least seed of a practice row (kind Q): the loop's seeds; the benchmark's own are 0-19
    practice_dir: str = _DATA + "/practice_lml"  # <Task>.jsonl, 150 lines each: the 20 original tasks
    practice_dir_fridge: str = _DATA + "/practice_lml_fridge"  # the three fridge-door tasks
    practice_dir_drawer: str = _DATA + "/practice_lml_drawer"  # the three low-drawer tasks
    truth_grasp_dir: str = (
        _DATA + "/a_l/grasp"
    )  # <Task>_LX_e0.npz (the object's points) and grasp_<Task>_LX_e0.npz: the truth track's grasps, read for the record
    specs_file: str = (
        _DATA + "/a_l/bed3/specs.json"
    )  # what each task's success check reads, for the 26 tasks (the record and the judge)
    # --- the run
    cache_dir: str = (
        _DATA + "/a_l/cloud_q"
    )  # look + GraspGen-X cache root; the key folder under it is <sha1 of the package's camera view and the look's parameters>
    run_dir: str = (
        _DATA + "/a_l/run_q"
    )  # scenes, the server's files and logs, timing of this call; a loop gives each unit its own
    slots_dir: str = _DATA + "/a_l/slots"  # slot<k>.lock, cache_<Task>.lock: SHARED with the esc runners
    slots: tuple[int, ...] = (1, 2)  # the slots this call may take (the loops: lml_q 1 2, fridge 3 4, drawer 5 6)
    sock_dir: str = ""  # the environment server's socket folder; "" = the system's temporary folder
    gpu: int = 0  # the driver's GPU (ScaleBFM's policy) and the server's EGL device
    graspgen_gpu: int = 0
    sam3_host: str = "127.0.0.1"  # SAM 3 service: expected to be up, never started or stopped by a run
    sam3_port: int = 18832
    server_ready_s: int = 120  # wait for the environment server's socket
    batch_tries: int = 3
    look_tries: int = 3
    slot_poll_s: float = 10.0
    # --- interpreters and trees
    lml_root: str = _MAIN + "/third_party/LocoManiLab"  # the benchmark: its tasks.yaml, robosuite / robocasa, assets
    lml_python: str = _MAIN + "/third_party/LocoManiLab/.venv/bin/python"  # the environment server
    server_file: str = ""  # an alternative server script ("" = the one in this package)
    sim_python: str = _MAIN + "/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python"  # the bed process
    graspgen_python: str = _MAIN + "/third_party/GraspGenX/.venv/bin/python"
    graspgen_root: str = _MAIN  # GraspGen-X's working directory
    graspgen_assets: str = (
        "third_party/GraspGenX/ext/gripper_descriptions/gripper_descriptions/assets"  # relative to graspgen_root
    )
    seed_skills: str = (
        ""  # package folder when --skills is not given; "" = the seed package inside locomani.bench.locomanilab
    )

    def sam3(self) -> tuple[str, int]:
        return (self.sam3_host, self.sam3_port)

    def seed_floor(self) -> int:
        """The least `jitter seed + episode` the environment server makes a start state of: the loop's practice seeds, or, when a
        development kind is allowed, the development seeds. Always above the benchmark's own episodes (0-19)."""
        dev = [PRACTICE_SEED] * ("P" in self.kinds) + [WIDE["seed"]] * ("W" in self.kinds)
        return min([self.seed_min] * ("Q" in self.kinds) + dev)

    def validate_scope(self) -> None:
        """Refuse a configuration that could run the benchmark's own episodes (raises ValueError)."""
        bad = sorted(set(self.kinds) - set(KINDS))
        if bad or not self.kinds:
            raise ValueError(
                f"kinds {self.kinds!r}: letters of {KINDS} only; E (the benchmark's own episodes) is never allowed"
            )
        if self.seed_min < EVAL_EPISODES:
            raise ValueError(
                f"seed_min {self.seed_min} would admit the benchmark's own episodes (seeds 0-{EVAL_EPISODES - 1})"
            )

    def practice_for(self, field: str) -> str:
        return getattr(self, field)

    def to_dict(self) -> dict:
        return config_to_dict(self, SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, data: dict) -> LocoManiLabConfig:
        return config_from_dict(cls, data, SCHEMA_VERSION)

    @classmethod
    def load(cls, path) -> LocoManiLabConfig:
        return config_load(cls, path)

    def dump(self, path) -> None:
        config_dump(self, path)

    def with_overrides(self, overrides: list[str]) -> LocoManiLabConfig:
        return apply_overrides(self, overrides)


SEED_DIR = Path(__file__).resolve().parent / "seed"  # the seed skill package
SCOPE = dict(
    target=("perception", "truth"),
    sensor=("rgbd", "none"),
    depth_noise=("none", "realsense"),
    self_pose=("odometry", "truth"),
    controller_root=("odometry", "truth"),
    controller=("scalebfm",),
)  # the switch values this bed implements (target=truth: the 20 original tasks; the articulated ones are refused at run time)
DEFAULT_CONTROL = (
    "odometry",
    "odometry",
)  # self_pose, controller_root of the loops: part of the look cache key only when they differ
DEFAULT_VIEW = (
    "perception",
    "rgbd",
    "none",
)  # target, sensor, depth_noise of the loops: the look cache key is unchanged for these


class ScopeError(ValueError):
    """A switch value outside what the LocoManiLab bed implements."""


def check_scope(options) -> None:
    """Refuse the switch values this bed does not implement, naming why."""
    why = {
        ("sensor", "stereo"): "LocoManiLab has one head camera: no second image to build a stereo pair from",
        (
            "controller",
            "native",
        ): "the environment is driven by joint targets; LocoManiLab's own ScaleBFM client is not wired here",
    }
    bad = []
    for axis, ok in SCOPE.items():
        got = getattr(options, axis)
        if got not in ok:
            bad.append(f"{axis}={got} -> {why.get((axis, got), 'not implemented')}")
    if options.target == "truth" and options.self_pose != "truth":
        bad.append(
            "target=truth needs self_pose=truth: the truth view gives positions in the simulator's frame, which is the "
            "belief's frame only when the belief is the simulator's (the old truth track used Belief(odo=False))"
        )
    if bad:
        raise ScopeError(
            "the LocoManiLab bed runs target=perception|truth sensor=rgbd depth_noise=none|realsense self_pose=odometry|truth "
            "controller_root=odometry|truth controller=scalebfm: " + "; ".join(bad)
        )
