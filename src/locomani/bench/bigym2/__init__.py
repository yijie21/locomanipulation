"""BiGym 2.0 (arXiv 2610.07594; G1 + Dex1 under a frozen GR00T whole-body controller, 20 kitchen tasks): the benchmark adapter (TODO 49.5).

Smallest useful adapter: target=truth, controller=native, the rgbd channel available. Install, versions and what was checked:
docs/handoff/bench-options-20261124.md ("BiGym 2.0 port"). Everything else stays `planned` with its reason.
Entry: `python -m locomani.bench.bigym2 run <task> --ids F --setting J --out F`.
"""

from locomani.bench.spec import PLANNED, READY, BenchSpec

SPEC = BenchSpec(
    name="bigym2",
    title="BiGym 2.0",
    about="G1 + Dex1 under its frozen GR00T-WBC, 20 kitchen tasks (MuJoCo)",
    supports=dict(
        target={"truth": READY, "perception": PLANNED},
        sensor={"rgbd": READY},
        depth_noise={"none": READY, "realsense": PLANNED},
        self_pose={"truth": READY, "odometry": PLANNED},
        controller_root={"truth": READY},
        controller={"native": READY},
    ),
    notes={
        "target=truth": "ready on reach_target_single and drawer_top_open (the seed package's two flows); the other tasks run the environment and the judge but have no truth goal",
        "target=perception": "no detector is wired on BiGym's head camera; the rgbd frames exist (Cameras.rgbd), the default view is 84x84 without depth",
        "sensor=rgbd": "the channel is built and checked: --set depth=true makes the server patch BiGym's CameraConfig at run time (their file hard-codes depth=False, bigym/loco/env.py:669) and the head camera renders depth at 240x320; a run reads it only under target=perception (planned), so a truth run resolves with sensor=none",
        "sensor=stereo": "no stereo head camera in the benchmark",
        "depth_noise=none": "the renderer's exact depth, as is (ready with the channel)",
        "depth_noise=realsense": "perception/noise.py on the head depth: not wired (planned)",
        "controller=scalebfm": "decided 2026-11-23: BiGym keeps its own GR00T-WBC controller",
        "controller_root=odometry": "the native controller reads the benchmark's own state",
        "self_pose=truth": "the skill reads the pelvis and the pinch centres from the simulator (the native controller does the same)",
        "self_pose=odometry": "leg odometry needs the G1's foot loads, which BiGym does not expose as sensors (compute from contact forces): planned",
    },
)
