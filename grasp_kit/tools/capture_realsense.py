"""A look from the G1's head RealSense D435(i), with pyrealsense2 (perception env: uv pip install pyrealsense2).

    python tools/capture_realsense.py --out look.npz --instruction "pick up the red cup" \
        [--waist YAW ROLL PITCH] [--imu-wxyz W X Y Z] [--offset X Y Z] [--size 640x480] [--serial SERIAL]

The colour image, the depth aligned to it (m) and the colour camera's intrinsics come from the camera. The camera's
pose is frames.head_camera: the URDF's mount, the waist joints and the pelvis IMU you pass, read from the robot's low
state at the same moment (leave them out for a locked waist and an upright robot). Check it with tools/check_camera.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1grasp import look as L  # noqa: E402
from g1grasp.frames import head_camera  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--instruction", default="")
    ap.add_argument("--object", default="")
    ap.add_argument("--waist", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--imu-wxyz", type=float, nargs=4, default=None)
    ap.add_argument("--offset", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--size", default="640x480")
    ap.add_argument("--serial", default="")
    ap.add_argument("--warmup", type=int, default=30, help="frames to let the auto exposure settle")
    a = ap.parse_args()
    import pyrealsense2 as rs

    W, H = (int(x) for x in a.size.split("x"))
    cfg = rs.config()
    if a.serial:
        cfg.enable_device(a.serial)
    cfg.enable_stream(rs.stream.color, W, H, rs.format.rgb8, 30)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, 30)
    pipe = rs.pipeline()
    prof = pipe.start(cfg)
    try:
        scale = prof.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        for _ in range(a.warmup):
            pipe.wait_for_frames()
        fs = align.process(pipe.wait_for_frames())
        color, depth = fs.get_color_frame(), fs.get_depth_frame()
        i = color.profile.as_video_stream_profile().intrinsics
        K = np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]])
        rgb = np.asanyarray(color.get_data()).copy()
        d = np.asanyarray(depth.get_data()).astype(np.float32) * scale
    finally:
        pipe.stop()
    T = head_camera(a.waist, a.imu_wxyz, a.offset)
    L.save(a.out, rgb, K, T, depth=d, instruction=a.instruction, obj=a.object)
    print(
        f"{a.out}: {W}x{H}, depth known on {(d > 0).mean():.0%} of the image, fx {i.fx:.1f}; camera at {T[:3, 3].round(3)} in the base frame"
    )


if __name__ == "__main__":
    main()
