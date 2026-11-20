"""A look file from images saved elsewhere (a ROS bag, another capture program).

    python tools/make_look.py --rgb color.png --depth depth.png [--depth-scale 0.001] --K FX FY CX CY \
        (--T T_base_cam.txt | [--waist YAW ROLL PITCH] [--imu-wxyz W X Y Z] [--offset X Y Z]) \
        (--instruction "pick up the red cup" | --object "red cup") --out look.npz

--depth      a 16-bit PNG (times --depth-scale gives metres; 0.001 for millimetres) or a .npy in metres, aligned to the
             colour image (same size, same pixels)
--T          a text file with the 4 x 4 camera pose in the base frame (camera frame: x right, y down, z forward);
             without it the pose is frames.head_camera's
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
    ap.add_argument("--rgb", required=True)
    ap.add_argument("--depth", required=True)
    ap.add_argument("--depth-scale", type=float, default=0.001)
    ap.add_argument("--K", type=float, nargs=4, required=True, metavar=("FX", "FY", "CX", "CY"))
    ap.add_argument("--T", default="")
    ap.add_argument("--waist", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--imu-wxyz", type=float, nargs=4, default=None)
    ap.add_argument("--offset", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--instruction", default="")
    ap.add_argument("--object", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from PIL import Image

    rgb = np.asarray(Image.open(a.rgb).convert("RGB"))
    d = (
        np.load(a.depth).astype(np.float32)
        if a.depth.endswith(".npy")
        else np.asarray(Image.open(a.depth)).astype(np.float32) * a.depth_scale
    )
    assert d.shape == rgb.shape[:2], (
        f"depth {d.shape} and colour {rgb.shape[:2]} differ: align the depth to the colour image first"
    )
    fx, fy, cx, cy = a.K
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    T = np.loadtxt(a.T).reshape(4, 4) if a.T else head_camera(a.waist, a.imu_wxyz, a.offset)
    L.save(a.out, rgb, K, T, depth=d, instruction=a.instruction, obj=a.object)
    print(f"{a.out}: {rgb.shape[1]}x{rgb.shape[0]}, depth known on {(d > 0).mean():.0%}")


if __name__ == "__main__":
    main()
