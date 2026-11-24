"""BiGym 2.0 (arXiv 2610.07594; G1 + Dex1 under a frozen GR00T whole-body controller, 20 kitchen tasks).

Table ported from bench-options (its TODO 49.5). Nothing of it runs on selfwm either. Facts are from the paper and
project page, not measured; each one (camera resolution and depth, the controller's state input, the evaluation seeds)
is checked before a value becomes ready.
"""

from locomani.bench.spec import PLANNED, BenchSpec

SPEC = BenchSpec(
    name="bigym2",
    title="BiGym 2.0",
    about="G1 + Dex1 under its frozen GR00T-WBC, 20 kitchen tasks (MuJoCo)",
    supports=dict(
        target={"truth": PLANNED, "perception": PLANNED},
        sensor={"rgbd": PLANNED},
        depth_noise={"none": PLANNED, "realsense": PLANNED},
        self_pose={"truth": PLANNED, "odometry": PLANNED},
        controller_root={"truth": PLANNED},
        controller={"native": PLANNED},
    ),
    notes={
        "sensor=rgbd": "the paper's strict views are 84x84; resolution and depth to check in TODO 49.5",
        "sensor=stereo": "no stereo head camera in the benchmark",
        "controller=scalebfm": "decided 2026-11-23: BiGym keeps its own GR00T-WBC controller",
        "controller=sonic": "decided 2026-11-23: BiGym keeps its own GR00T-WBC controller",
        "controller=amo": "decided 2026-11-23: BiGym keeps its own GR00T-WBC controller",
        "controller=gr00t_wbc": "its own GR00T-WBC is controller=native; our Motor layer over it is not wired",
        "controller_root=odometry": "the native controller reads the benchmark's own state",
        "self_pose=odometry": "leg odometry needs the G1's joint and IMU readings and foot loads; to check in 49.5",
    },
)
