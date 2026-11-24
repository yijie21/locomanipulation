"""HumanoidArena (arXiv 2606.17833, NeurIPS 2026; Isaac Sim 5.0 + Isaac Lab 2.2, G1 + Dex3 under a TWIST2 motion tracker): the benchmark adapter (TODO 49.5b).

Tasks in scope (user 2026-11-23): pick-and-place box (`pp_box`), double desk (`double_desk`), open door (`open_door`); not the ball kick or
the sofa. Smallest useful adapter: target=truth, controller=native (the benchmark's TWIST2 tracker, driven in process through its own
evaluation worker), the rgbd channel built. Install, versions and what was checked: docs/handoff/bench-options-20261124.md
("HumanoidArena port"). Everything else stays `planned` with its reason.
Entry: `python -m locomani.bench.humanoidarena run <task> --ids F --setting J --out F`.
"""

from locomani.bench.spec import PLANNED, READY, BenchSpec

SPEC = BenchSpec(
    name="humanoidarena",
    title="HumanoidArena",
    about="IsaacLab humanoid tasks with a TWIST2 motion tracker: P&P box, double desk, open door",
    supports=dict(
        target={"truth": READY, "perception": PLANNED},
        sensor={"rgbd": READY},
        depth_noise={"none": READY, "realsense": PLANNED},
        self_pose={"truth": READY},
        controller_root={"truth": READY},
        controller={"native": READY},
    ),
    notes={
        "target=truth": "ready as an interface on pp_box: the truth view reads the box, a spot the benchmark's own check accepts and the layer height, the judge records hand and body contacts; the seed flow ran on 5 practice episodes and succeeded on none (docs/handoff/bench-options-20261124.md); double_desk and open_door run the environment and the judge's record (open_door has none) but have no truth goal",
        "target=perception": "no detector is wired on the front camera; the rgbd frames exist (Cameras.rgbd, 640x480 with the renderer's depth)",
        "sensor=rgbd": "the channel is built and checked: the server returns the front camera's colour and exact depth, K and the camera in the pelvis frame, and the bed saves one overlay png (--set overlay_dir); a run reads it only under target=perception (planned), so a truth run resolves with sensor=none",
        "sensor=stereo": "IsaacLab could render a second camera; not planned yet",
        "depth_noise=none": "the renderer's exact depth, as is (ready with the channel)",
        "depth_noise=realsense": "perception/noise.py on the front depth: not wired (planned)",
        "controller=native": "the benchmark's TWIST2 tracker (the ONNX its repository ships) runs in process on the benchmark's own evaluation worker, checked on all three tasks (reset and 60 steps headless); its reference needs a gait on the legs (env.gait, from the benchmark's example walking clips) before the root moves",
        "controller=scalebfm": "ScaleBFM runs in MuJoCo; the benchmark's physics is IsaacLab's",
        "controller_root=odometry": "the native tracker reads the benchmark's own state",
        "self_pose=truth": "the skill reads the pelvis and the wrists from the simulator (the native tracker does the same)",
        "self_pose=odometry": "not planned until the native controller's interface is known",
    },
)
