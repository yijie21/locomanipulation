"""Check the camera's pose (T_base_cam) on a look of level surfaces (the floor, a table) before trusting a grasp.

    python tools/check_camera.py LOOK.npz [--out check_camera.png] [--ply cloud.ply]

Uses the look's depth as it is (no models; either env). Prints each large horizontal surface it finds: its height in
the base frame, how many points, and its tilt. What to expect with a right pose:
  - the floor at about minus the pelvis's height (about -0.75 m standing), tilted under 1 deg;
  - a table at (its height by a tape measure) + (the floor's z), tilted under 1 deg.
Floor and table tilted the same way: the camera's pitch or roll is off (or the IMU's). Heights off by the same amount:
the camera's height is off. Fix T_base_cam (frames.head_camera's offset, or your own calibration) and look again.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g1grasp import look as L  # noqa: E402
from g1grasp.frames import apply, backproject  # noqa: E402


def surfaces(p: np.ndarray, min_frac: float = 0.02):
    """Horizontal surfaces: peaks of the points' heights, each with a plane fitted to the points near it."""
    z = p[:, 2]
    h, e = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.01, 0.01))
    found = []
    for i in np.argsort(-h):
        if h[i] < min_frac * len(z) or len(found) == 4:
            break
        c = 0.5 * (e[i] + e[i + 1])
        if any(abs(c - f["z"]) < 0.04 for f in found):
            continue
        q = p[np.abs(z - c) < 0.015]
        for _ in range(3):  # z = a x + b y + c by least squares, again without the points more than 1 cm off
            A = np.c_[q[:, :2], np.ones(len(q))]
            coef = np.linalg.lstsq(A, q[:, 2], rcond=None)[0]
            q = q[np.abs(A @ coef - q[:, 2]) < 0.01]
        a, b, _ = coef
        found.append(dict(z=float(coef @ [*q[:, :2].mean(axis=0), 1]), n=len(q), tilt=float(np.degrees(np.arctan(np.hypot(a, b)))),
                          down_toward=float(np.degrees(np.arctan2(-b, -a))), x=(float(q[:, 0].min()), float(q[:, 0].max()))))  # fmt: skip
    return sorted(found, key=lambda f: f["z"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("look")
    ap.add_argument("--out", default="check_camera.png")
    ap.add_argument("--ply", default="")
    a = ap.parse_args()
    look = L.load(a.look)
    assert "depth" in look, "check_camera needs a look with a depth"
    d = look["depth"]
    ok = (d > 0.2) & (d < 5)
    p = apply(look["T_base_cam"], backproject(d, look["K"]).reshape(-1, 3))[ok.ravel()]
    col = look["rgb"].reshape(-1, 3)[ok.ravel()]
    print(f"{ok.sum()} points with a depth ({ok.mean():.0%} of the image)")
    print(
        f"{'surface z (m)':>14s} {'points':>8s} {'tilt (deg)':>10s} {'falls toward (deg, 0 = forward)':>32s} {'x from..to (m)':>16s}"
    )
    S = surfaces(p)
    for f in S:
        print(
            f"{f['z']:14.3f} {f['n']:8d} {f['tilt']:10.2f} {f['down_toward']:32.0f} {f['x'][0]:7.2f}..{f['x'][1]:.2f}"
        )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    near = np.flatnonzero(p[:, 0] < 2.0)  # the picture: the 2 m in front of the robot, where it would grasp
    k = np.random.default_rng(0).choice(near, min(len(near), 60000), replace=False)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].imshow(look["rgb"])
    ax[0].set_axis_off()
    for i, (j, lab) in enumerate(((0, "x forward (m)"), (1, "y left (m)"))):
        ax[i + 1].scatter(p[k, j], p[k, 2], s=0.5, c=col[k] / 255)
        for f in S:
            ax[i + 1].axhline(f["z"], color="red", lw=0.8, ls="--")
            ax[i + 1].text(
                p[k, j].min(),
                f["z"],
                f" z {f['z']:+.3f}, tilt {f['tilt']:.1f} deg",
                fontsize=8,
                color="red",
                va="bottom",
            )
        ax[i + 1].set(
            xlabel=lab, ylabel="z up (m)", aspect="equal", title="from the side" if j == 0 else "from the front"
        )
        ax[i + 1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(a.out, dpi=90)
    print("picture:", a.out)
    if a.ply:
        with open(a.ply, "w") as fh:
            fh.write(f"ply\nformat ascii 1.0\nelement vertex {len(k)}\nproperty float x\nproperty float y\nproperty float z\n"
                     "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")  # fmt: skip
            for q, c in zip(p[k], col[k]):
                fh.write(f"{q[0]:.4f} {q[1]:.4f} {q[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
        print("points (base frame):", a.ply)


if __name__ == "__main__":
    main()
