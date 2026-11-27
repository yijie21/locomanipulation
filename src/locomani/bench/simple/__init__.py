"""SIMPLE on selfwm (arXiv 2606.08278; pinned 6d10628): the self-model line's bed for SIMPLE's four pick tasks.

The bed is scripts/experiments/simple_eval_20261112/simple_bench.py: SIMPLE's episodes rebuilt in MuJoCo
(episodes.py, "the replica") with SIMPLE's G1 + Dex3, our Motor layer over the chosen whole-body controller, the
perception looks made beforehand and cached (blind.py look, grasps_graspgen.py). It is not bench-options' bed
(`bench/simple/` there: the RSI loops' port, nine tasks, live looks); records of the two do not compare without a parity
check. simple_bench maps the switches onto its own settings with `bed_settings` below; the shell runners
(bench_via.sh, jref_collect.sh) read the same mapping through `python -m locomani.bench.simple --bench-set ...`.
"""

from locomani.bench.spec import PLANNED, READY, BenchSpec

PICK_TASKS = ("XMovePickTeleop", "XMoveBendPickTeleop", "BendPickMP", "TabletopGraspMP")

# sensor -> the cached look the perception column reads (target points, grasps) and the per-episode look dump the path
# planner's obstacles come from (curobo_avoid_20261120/obstacles_box.py), both under outputs/simple_eval_20261112/.
# The stereo look's dump comes from look_all.sh (blind.py look --eyes with LOOK_DUMP), the RGB-D look's from look_frame.py
# (2026-11-24: blind.py look writes none without --eyes).
LOOKS = {"stereo": ("cloud_eyes_final", "look_dump_eyes"), "rgbd": ("cloud_blind", "look_dump_blind")}

SPEC = BenchSpec(
    name="simple",
    title="SIMPLE",
    about="G1 + Dex3, SIMPLE's four pick tasks on the MuJoCo replica (simple_eval_20261112/simple_bench.py)",
    supports=dict(
        target={"truth": READY, "perception": READY},
        sensor={"stereo": READY, "rgbd": READY},
        depth_noise={"none": READY},
        self_pose={"truth": READY, "odometry": READY},
        controller_root={"odometry": READY},
        controller={"scalebfm": READY, "sonic": READY, "heft": READY, "amo": PLANNED, "gr00t_wbc": PLANNED},
        sonic_vr_frame={"robot": READY, "reference": READY},
    ),
    notes={
        "target=truth": "the target's pose read from the simulator at the start and again at the stance (column truth)",
        "target=perception": "the main metric's column: SAM 3 on the head image, the target's points from depth, "
        "GraspGen-X's grasps, all from the start's look (cached); the table's edge from blind_f1",
        "sensor=stereo": "cloud_eyes_final: SIMPLE's Isaac head stereo pair, depth by matching + LingBot-Depth "
        "(depth_server.py; 2026-11-20); the look of the numbers recorded from 11-20 to 11-24",
        "sensor=rgbd": "cloud_blind: the replica's MuJoCo head camera, its colour and rendered depth (2026-11-19); "
        "MuJoCo's image, not Isaac's; the planner's frame from look_frame.py (look_dump_blind); the default since 11-24",
        "depth_noise=realsense": "bench-options' perception/noise.py is not ported to selfwm",
        "self_pose=truth": "with target=truth only: the Motor gets the goals in the simulator's frame",
        "self_pose=odometry": "with target=perception only: leg odometry after the walk (place.Belief)",
        "controller_root=odometry": "BFM_ROOT=odo, leg odometry (leg_odo.OdoRoot): the self models' data settings",
        "controller_root=truth": "simple_bench refuses it: every self model here was trained with BFM_ROOT=odo",
        "controller=sonic": "SONIC v1.1 in process (backends.sonic_onnx; TODO 38.8, 38.10)",
        "controller=heft": "LocoManiLab's HEFT (g1_backend heft/pmg) from its ONNX file (backends.heft; 2026-11-25): "
        "the Motor's whole-body reference as qpos frames, walks on gait_heft.HEFT (hg_walk's footstep plan, step shapes "
        "tuned toward the LAFAN walks HEFT was trained on; 2026-11-26); the scene's joint dynamics "
        "with HEFT's PD, reset in LocoManiLab's stance (HEFT's own default wedges the Dex3 thumbs in the hips); it reads "
        "no pelvis position, so controller_root only feeds the odometry; wired and smoke-tested, no evaluation yet",
        "controller=amo": "env.py runs AMO's legs and waist with our arm solver; never run on this bed",
        "controller=gr00t_wbc": "env.py runs NVIDIA's GR00T-WBC legs and waist with our arm solver; never run on this bed",
        "controller=native": "the replica is driven by joint targets; SIMPLE's own whole-body controller is not wired",
        "sonic_vr_frame=robot": "SonicMotorConfig.vr_frame \"robot\": the hand targets in the believed pelvis' frame, "
        "the reference's world wrist poses (the default since 2026-11-16, and of every SONIC number here until 11-26)",
        "sonic_vr_frame=reference": "SonicMotorConfig.vr_frame \"reference\": the hand targets in the reference pose's "
        "own pelvis, no loop through the pelvis estimate (simple_sonic_vrref_20261126)",
    },
    couples={"target=truth": ("self_pose", "truth"), "target=perception": ("self_pose", "odometry")},
)


def bed_settings(options) -> dict:
    """simple_bench's own settings for resolved options: the column, the cached look and its dump, the controller,
    BFM_ROOT, SONIC's VR frame. The truth column reads no look; it keeps simple_bench's old default (cloud_blind) so
    rows stay as they were."""
    if options.bench != "simple":
        raise ValueError(f"simple_bench runs bench=simple, not {options.bench}")
    cloud, look = LOOKS.get(options.sensor, LOOKS["rgbd"])
    return dict(
        column=options.target,
        cloud=cloud,
        look=look,
        controller=options.controller,
        root={"odometry": "odo", "truth": "true"}[options.controller_root],
        sonic_vr_frame=options.sonic_vr_frame,
    )
