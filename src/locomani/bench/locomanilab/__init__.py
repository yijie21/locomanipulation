"""LocoManiLab (G1 + Dex3, AI2-THOR / Holodeck scenes): the benchmark adapter (TODO 49.4).

Until 49.4 lands, the working bed is the escalation one on branch esc (escalation_20261116: bed_lml.py, bed_lml_cam.py,
lml_server.py, view_lml.py; the fridge and drawer beds in lml_bed2_20261121 / lml_bed3_20261121). Dropped as the
loops' benchmark on 2026-11-23 (assets unlike their names, light objects, loose checks); kept as an option.
"""

from locomani.bench.spec import PLANNED, BenchSpec

SPEC = BenchSpec(
    name="locomanilab",
    title="LocoManiLab",
    about="G1 + Dex3 in LocoManiLab's iTHOR / Holodeck scenes: 20 practice tasks, fridge and low-drawer tasks",
    supports=dict(
        target={"truth": PLANNED, "perception": PLANNED},
        sensor={"rgbd": PLANNED},
        depth_noise={"none": PLANNED, "realsense": PLANNED},
        self_pose={"truth": PLANNED, "odometry": PLANNED},
        controller_root={"truth": PLANNED, "odometry": PLANNED},
        controller={"scalebfm": PLANNED},
    ),
    notes={
        "sensor=rgbd": "one 320x240 head camera with the simulator's depth image (the loops so far)",
        "sensor=stereo": "one head camera; a second rendered camera would be needed (TODO 49.4 checks)",
        "controller=native": "the bed drives the robot through ScaleBFM; no other controller is wired",
    },
)
