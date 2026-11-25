"""SIMPLE bed knobs (TODO 49.2): every former environment switch and hard-coded path of the escalation bed, with today's value.

One frozen dataclass, parsed at the entry (`python -m locomani.bench.simple ... --set key=value`) and passed on; nothing in the
bed reads `os.environ` for a switch. The defaults are what the RSI loops ran with: the practice pools and the look cache under
the esc worktree's outputs (data only, read through these fields), the slot locks SHARED with the esc bed (so old and new runs
together never exceed three Isaac servers), Isaac on GPU 0, the stereo depth service, GraspGen-X and SAM 3 on GPU 1.

The bed's fixed settings (feed_torque, amo_arm_pos: the loop's `bed_fixed`) are not here: they are settings of a run, as before.
Mapping from the old environment (docs in port_notes): SIMPLE_PRACTICE -> practice_dir, SIMPLE_RUN_DIR -> run_dir,
SIMPLE_LAZY_RENDER -> lazy_render, SIMPLE_SLOTS -> slots, SIMPLE_XDIR -> xdir, SIMPLE_ENV_NAMES -> env_names, SAM3_PORT ->
sam3_port, SIMPLE_ROOT -> simple_root, SIMPLE_LOOP -> kinds (only the listed practice pools run; a released id never does),
OFF_ISAAC_SOCK / OFF_DEPTH_SOCK -> slots_dir (+ the slot), SIMPLE_SKILLS -> the --skills argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from locomani.config_base import apply_overrides, config_dump, config_from_dict, config_load, config_to_dict

SCHEMA_VERSION = 1

_MAIN = "/workspace/code/locomanipulation"  # the main checkout: third_party (symlinked from every worktree) lives there
_ESC = "/workspace/code/locomanipulation-esc"  # the esc worktree: data under outputs/ is read only
_DATA = _ESC + "/outputs/escalation_20261116"


@dataclass(frozen=True)
class SimpleConfig:
    # --- episodes (data files, read only)
    practice_dir: str = _DATA + "/practice2"  # the Q pool: <Task>.jsonl, 150 lines each (SIMPLE's own draws)
    dev_dir: str = _DATA + "/practice"  # the P pool: the development set
    kinds: str = "Q"  # practice pools this run may open, as letters ("Q", "PQ"): a released id (L<level>/e<i>) never runs; SIMPLE_LOOP=1 was kinds=Q
    cache_dir: str = (
        _DATA + "/official/cloud_q"
    )  # look + GraspGen-X cache root; the key folder under it is <sha1 of the package's camera view and the look's parameters>
    grasp_lib_dir: str = _DATA + "/grasp_lib"  # the truth track's grasp library (target=truth, not in 49.2's scope)
    visual_dir: str = (
        _ESC + "/outputs/simple_eval_20261112/visual"
    )  # object meshes and textures for the local MuJoCo scene
    run_dir: str = (
        _DATA + "/official/run_q"
    )  # scenes, logs, pid files, timing of this call; a loop gives each unit its own
    # --- the services
    slots_dir: str = _DATA + "/official/slots"  # slot<k>.lock, cache_<Task>.lock, sockets: SHARED with the esc bed
    slots: tuple[int, ...] = (1, 2, 3)  # the slots this call may take (one Isaac server + one depth service each)
    isaac_gpu: int = 0
    depth_gpu: int = 1
    graspgen_gpu: int = 1
    sam3_host: str = "127.0.0.1"  # SAM 3 service: expected to be up, never started or stopped by a run
    sam3_port: int = 18832
    lazy_render: bool = (
        True  # cameras render only on `eyes`, in every round; False = every step (official benchmark / exam reporting)
    )
    lazy_warm: int = (
        48  # lazy render: extra renders of the same state before an `eyes` read (auto-exposure settles; TODO 52)
    )
    lazy_subframes: int = 1  # lazy render: rt_subframes of each `eyes` render
    memo: bool = False  # replay from a saved state: variants of one start share the stepping of their common primitives (memo.py; diagnostic / probe / sweep rounds, not the gate or the sealed set)
    memo_eyes: bool = True  # memo: the head images at a state are stored and reused (renders differ run to run, so keys would never match otherwise)
    memo_check: bool = (
        False  # debug: after every restored state, ask both sides what differs from the state that was stored
    )
    memo_dir: str = ""  # memo: the states are also kept on disk under this folder (memostore.py), so the next process finds them; "" = memory only. The loop sets it to a folder of its scratch area, never inside a sandbox
    memo_sig: str = ""  # memo: the loop's bed signature; with a hash of the code that makes a state it names the store's sub-folder, so a changed bed or switch set reaches no old state
    memo_gb: float = 20.0  # memo: size cap of the disk store in GB (oldest files go first)
    memo_states: int = 400  # saved states kept (client and server); the oldest go first
    env_names: str = _DATA + "/official/env_names.json"  # task -> SIMPLE's gym environment name
    xdir: str = "/nonexistent"  # released-episode folder handed to the server: it is never opened
    server_ready_s: int = 400  # wait for the Isaac server's "listening" line
    depth_ready_s: int = 200  # wait for the depth service's socket
    batch_tries: int = 4
    look_tries: int = 3
    slot_poll_s: float = 10.0
    # --- interpreters and trees
    simple_root: str = (
        _MAIN + "/third_party/SIMPLE"
    )  # SIMPLE's data (robots, assets) and the server's working directory
    simple_python: str = _MAIN + "/third_party/SIMPLE/.venv/bin/python"  # the environment server
    sim_python: str = _MAIN + "/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python"  # the bed process
    sam3d_python: str = "/workspace/miniconda3/envs/sam3d/bin/python"  # depth service and SAM 3
    graspgen_python: str = _MAIN + "/third_party/GraspGenX/.venv/bin/python"
    graspgen_root: str = _MAIN  # GraspGen-X's working directory
    graspgen_assets: str = (
        "third_party/GraspGenX/ext/gripper_descriptions/gripper_descriptions/assets"  # relative to graspgen_root
    )
    lingbot_path: str = _MAIN + "/third_party/lingbot-depth"
    sam3_path: str = _MAIN + "/third_party/sam3"
    hf_home: str = "/workspace/huggingface_cache"
    seed_skills: str = (
        ""  # package folder when --skills is not given; "" = the seed package inside locomani.bench.simple
    )

    def sam3(self) -> tuple[str, int]:
        return (self.sam3_host, self.sam3_port)

    def to_dict(self) -> dict:
        return config_to_dict(self, SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, data: dict) -> SimpleConfig:
        return config_from_dict(cls, data, SCHEMA_VERSION)

    @classmethod
    def load(cls, path) -> SimpleConfig:
        return config_load(cls, path)

    def dump(self, path) -> None:
        config_dump(self, path)

    def with_overrides(self, overrides: list[str]) -> SimpleConfig:
        return apply_overrides(self, overrides)


SEED_DIR = Path(__file__).resolve().parent / "seed"  # the seed skill package
SCOPE = dict(
    target=("perception", "truth"),
    sensor=("stereo", "rgbd", "none"),
    depth_noise=("none", "realsense"),
    self_pose=("odometry", "truth"),
    controller_root=("odometry", "truth"),
    controller=("scalebfm",),
)  # the switch values this bed implements: the loops' (perception, stereo, none, odometry, odometry, scalebfm) and TODO 49.2 step 6's
DEFAULT_CONTROL = (
    "odometry",
    "odometry",
)  # self_pose, controller_root of the loops: part of the look cache key only when they differ
DEFAULT_VIEW = (
    "perception",
    "stereo",
    "none",
)  # target, sensor, depth_noise of the loops: the look cache key is unchanged for these


class ScopeError(ValueError):
    """A switch value outside what the SIMPLE bed implements."""


def check_scope(options) -> None:
    """Refuse the switch values this bed does not implement, naming where they come from."""
    later = {
        ("controller", "native"): "TODO 49.2 step 6 (SIMPLE's own whole-body controller)",
    }
    bad = []
    for axis, ok in SCOPE.items():
        got = getattr(options, axis)
        if got not in ok:
            bad.append(f"{axis}={got} -> {later.get((axis, got), 'not implemented')}")
    if bad:
        raise ScopeError(
            "the SIMPLE bed runs target=perception|truth sensor=stereo|rgbd depth_noise=none|realsense self_pose=odometry|truth "
            "controller_root=odometry|truth controller=scalebfm: " + "; ".join(bad)
        )
