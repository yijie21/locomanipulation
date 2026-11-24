"""SIMPLE (Isaac Sim rendering, MuJoCo physics; Unitree G1 with Dex3 hands): the benchmark adapter (TODO 49.2).

The bed was ported from the escalation bed on branch esc (scripts/experiments/escalation_20261116) in TODO 49.2 and
checked against it (docs/handoff/bench-options-20261124.md); entry `python -m locomani.bench.simple run`.
"""

from locomani.bench.spec import PLANNED, READY, BenchSpec

SPEC = BenchSpec(
    name="simple",
    title="SIMPLE",
    about="G1 + Dex3 in SIMPLE's Isaac/MuJoCo environment: 5 pick/place tasks and 4 articulated tasks (pinned 6d10628)",
    supports=dict(
        target={"truth": READY, "perception": READY},
        sensor={"stereo": READY, "rgbd": READY},
        depth_noise={"none": READY, "realsense": READY},
        self_pose={"truth": PLANNED, "odometry": READY},
        controller_root={"truth": PLANNED, "odometry": READY},
        controller={"scalebfm": READY},
    ),
    notes={
        "target=perception": "the loops so far; the ported bed matches the esc bed step for step with perception frozen (TODO 49.2)",
        "target=truth": "objects from the server's state; smoked on XMovePick, CloseDoor, OpenOven only; art() parts by hinge distance",
        "sensor=stereo": "the head's stereo pair; depth from a stereo model (perception/stereo_server.py; the loops so far)",
        "sensor=rgbd": "the left head camera with Isaac's rendered depth; smoked on XMovePick, CloseDoor, OpenOven only",
        "depth_noise=realsense": "perception/noise.py, shadows from the right camera's rendered depth",
        "self_pose=truth": "the skill's belief from the simulator's pelvis; selectable in TODO 49.3",
        "controller_root=odometry": "the loops so far: OdoRoot is ScaleBFM's root_source, re-anchored at each stand",
        "controller_root=truth": "the policy told the simulator's pelvis (root_source None); selectable in TODO 49.3",
        "controller=native": "the environment is driven by joint targets; SIMPLE's own whole-body controller is not wired here",
    },
)
