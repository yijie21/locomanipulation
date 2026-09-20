#!/usr/bin/env python3
"""Compare two renders of the same camera and say how far apart they are.

    python scripts/compare_frames.py blender.png isaac.png --out side_by_side.png

Reports two numbers, because they answer different questions:

  raw        straight mean absolute difference. Dominated by exposure, which is
             the one thing that is just a knob (the light gain).
  structural difference after matching mean and standard deviation. This is the
             part a gain cannot fix -- shadows, global illumination, noise.

On the 2026-09-19 logistics room the pair was 20.3% raw / 9.2% structural before
the light gain was tuned, and 5.8% / 5.6% after; the remaining structural gap is
what the dome-light repair and path tracing then closed to 2.9%.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def load(path: Path):
    import numpy as np
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB")).astype("float64")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path, help="the render being matched, usually Blender")
    parser.add_argument("candidate", type=Path, help="the render being judged, usually Isaac")
    parser.add_argument("--out", type=Path, help="also write a side-by-side PNG here")
    args = parser.parse_args(argv)

    import numpy as np

    reference, candidate = load(args.reference), load(args.candidate)
    if reference.shape != candidate.shape:
        print(f"compare: shapes differ, {reference.shape} against {candidate.shape}; render both at one size")
        return 2

    raw = np.abs(reference - candidate).mean()
    rescaled = (candidate - candidate.mean()) / candidate.std() * reference.std() + reference.mean()
    structural = np.abs(reference - np.clip(rescaled, 0, 255)).mean()

    print(f"compare: reference luma {reference.mean():6.1f}  deviation {reference.std():5.1f}  {args.reference}")
    print(f"compare: candidate luma {candidate.mean():6.1f}  deviation {candidate.std():5.1f}  {args.candidate}")
    print(f"compare: raw difference {100 * raw / 255:.1f} percent, mostly exposure")
    print(f"compare: structural difference {100 * structural / 255:.1f} percent, what a light gain cannot fix")
    if candidate.mean() > 0:
        print(f"compare: to match exposure, scale the light gain by about {reference.mean() / candidate.mean():.2f}")

    if args.out:
        command = ["ffmpeg", "-v", "error", "-y", "-i", str(args.reference), "-i", str(args.candidate),
                   "-filter_complex", "hstack", str(args.out)]  # fmt: skip
        if subprocess.run(command, check=False).returncode == 0:
            print(f"compare: wrote {args.out}")
        else:
            print("compare: ffmpeg could not write the side-by-side; the numbers above still stand")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
