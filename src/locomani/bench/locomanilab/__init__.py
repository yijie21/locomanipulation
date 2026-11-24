"""LocoManiLab (G1 + Dex3; its own robosuite / robocasa environment on MuJoCo, rooms modelled on iTHOR and Holodeck).

Table ported from bench-options (its TODO 49.4, where the escalation bed on branch esc is the working one). On selfwm
there is no task bed: scripts/experiments/locomanilab_20261117/lml_selfmodel.py collects self-model data through
LocoManiLab's own G1 controllers (g1_backend: HEFT, ScaleBFM, SONIC v1.1) on its physics model, outside these switches.
"""

from locomani.bench.spec import PLANNED, BenchSpec

SPEC = BenchSpec(
    name="locomanilab",
    title="LocoManiLab",
    about="G1 + Dex3 in LocoManiLab's MuJoCo rooms (iTHOR / Holodeck layouts): 20 practice tasks, fridge and low-drawer tasks",
    supports=dict(
        target={"truth": PLANNED, "perception": PLANNED},
        sensor={"rgbd": PLANNED},
        depth_noise={"none": PLANNED, "realsense": PLANNED},
        self_pose={"truth": PLANNED, "odometry": PLANNED},
        controller_root={"truth": PLANNED, "odometry": PLANNED},
        controller={"scalebfm": PLANNED, "sonic": PLANNED, "native": PLANNED},
    ),
    notes={
        "sensor=rgbd": "one 320x240 head camera with the simulator's depth image (the loops so far)",
        "sensor=stereo": "one head camera; a second rendered camera would be needed (TODO 49.4 checks)",
        "controller=scalebfm": "g1_backend's ScaleBFM (lml_selfmodel.py drives it for self-model data; no task bed)",
        "controller=sonic": "g1_backend's SONIC v1.1 (lml_selfmodel.py; no task bed)",
        "controller=native": "g1_backend's HEFT, LocoManiLab's own tracker (lml_selfmodel.py; no task bed)",
        "controller=amo": "not in g1_backend",
        "controller=gr00t_wbc": "not in g1_backend",
    },
)
