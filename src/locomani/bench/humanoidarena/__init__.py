"""HumanoidArena (arXiv 2606.17833, NeurIPS 2026; IsaacLab, a TWIST2 tracker under a high-level policy): TODO 49.5.

Tasks in scope (user 2026-11-23): pick-and-place box, double desk, open door; not the ball kick or the sofa. Nothing
runs here yet; facts are from the paper, not measured, and 49.5 checks them before a value becomes ready.
"""

from locomani.bench.spec import PLANNED, BenchSpec

SPEC = BenchSpec(
    name="humanoidarena",
    title="HumanoidArena",
    about="IsaacLab humanoid tasks with a TWIST2 motion tracker: P&P box, double desk, open door",
    supports=dict(
        target={"truth": PLANNED, "perception": PLANNED},
        sensor={"rgbd": PLANNED},
        depth_noise={"none": PLANNED, "realsense": PLANNED},
        self_pose={"truth": PLANNED},
        controller_root={"truth": PLANNED},
        controller={"native": PLANNED},
    ),
    notes={
        "sensor=stereo": "IsaacLab could render a second camera; not planned yet",
        "controller=scalebfm": "ScaleBFM runs in MuJoCo; the benchmark's physics is IsaacLab's",
        "controller_root=odometry": "the native tracker reads the benchmark's own state",
        "self_pose=odometry": "not planned until the native controller's interface is known",
    },
)
