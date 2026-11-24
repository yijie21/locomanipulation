"""LocoManiLab (G1 + Dex3; its own robosuite / robocasa environment on MuJoCo, rooms modelled on iTHOR and Holodeck): the benchmark adapter (TODO 49.4).

The bed was ported from the escalation beds on branch esc (escalation_20261116 / lml_bed2_20261121 / lml_bed3_20261121: three copies of one
bed, one per task family) into this package as ONE bed for all 26 tasks, and checked against them (docs/handoff/bench-options-20261124.md);
entry `python -m locomani.bench.locomanilab run`. Dropped as the loops' benchmark on 2026-11-23 (assets unlike their names, light
objects, loose checks); kept as an option.
"""

from locomani.bench.spec import READY, BenchSpec

SPEC = BenchSpec(
    name="locomanilab",
    title="LocoManiLab",
    about="G1 + Dex3 in LocoManiLab's robosuite / robocasa MuJoCo rooms (iTHOR / Holodeck layouts): 20 practice tasks, 3 fridge-door and 3 low-drawer tasks",
    supports=dict(
        target={"truth": READY, "perception": READY},
        sensor={"rgbd": READY},
        depth_noise={"none": READY, "realsense": READY},
        self_pose={"truth": READY, "odometry": READY},
        controller_root={"truth": READY, "odometry": READY},
        controller={"scalebfm": READY},
    ),
    notes={
        "target=perception": "the loops so far; all 26 tasks; the old beds' episodes are reproduced step for step (TODO 49.4)",
        "target=truth": "the old truth track: the 20 original tasks only (needs self_pose=truth); the fridge and drawer tasks are refused at run time",
        "sensor=rgbd": "one 320x240 head camera with the renderer's exact depth (the loops so far)",
        "sensor=stereo": "one head camera: no second image to build a stereo pair from; a second rendered camera would have to be added to the server (not done)",
        "depth_noise=realsense": "perception/noise.py on the one camera's depth; no right-camera depth, so no occlusion shadows",
        "self_pose=truth": "the skill's belief from the simulator's pelvis (Belief(odo=False))",
        "self_pose=odometry": "leg odometry, re-tied to the true pelvis at the start and after each walk (as on SIMPLE)",
        "controller_root=odometry": "the loops so far: OdoRoot is ScaleBFM's root_source, re-anchored at each stand",
        "controller_root=truth": "the policy told the simulator's pelvis",
        "controller=native": "the environment is driven by joint targets; LocoManiLab's own ScaleBFM client is not wired here",
    },
)
