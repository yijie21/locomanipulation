"""SIMPLE (Isaac Sim rendering, MuJoCo physics; Unitree G1 with Dex3 hands): the benchmark adapter (TODO 49.2).

Until 49.2 lands, the working bed is the escalation one on branch esc
(scripts/experiments/escalation_20261116: bed_simple_official.py, official_server_esc.py, depth_server_esc.py, the
truth and camera views in seed_pkg/skill_simple.py and view_camera.py); every value here is `planned` until it runs
behind the new interfaces and matches that bed on the development episodes.
"""

from locomani.bench.spec import PLANNED, BenchSpec

SPEC = BenchSpec(
    name="simple",
    title="SIMPLE",
    about="G1 + Dex3 in SIMPLE's Isaac/MuJoCo environment: 5 pick/place tasks and 4 articulated tasks (pinned 6d10628)",
    supports=dict(
        target={"truth": PLANNED, "perception": PLANNED},
        sensor={"stereo": PLANNED, "rgbd": PLANNED},
        depth_noise={"none": PLANNED, "realsense": PLANNED},
        self_pose={"truth": PLANNED, "odometry": PLANNED},
        controller_root={"truth": PLANNED, "odometry": PLANNED},
        controller={"scalebfm": PLANNED},
    ),
    notes={
        "sensor=stereo": "the head's stereo pair; depth from a stereo model served by depth_server_esc.py (the loops so far)",
        "sensor=rgbd": "the head camera's depth rendered by Isaac; new in TODO 49.2",
        "controller_root=odometry": "the loops so far: OdoRoot is ScaleBFM's root_source, re-anchored at each stand",
        "controller_root=truth": "the policy told the simulator's pelvis (root_source None); selectable in TODO 49.3",
        "controller=native": "the environment is driven by joint targets; SIMPLE's own whole-body controller is not wired here",
    },
)
