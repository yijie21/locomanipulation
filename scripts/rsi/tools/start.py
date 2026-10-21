"""TODO 32.6: a start's geometry -- the robot's pose, each placed object's pose, the surface it rests on (from the scene
file), the object in that surface's frame with its distance to every edge, the surface height and approach edges, and
the robot's distance and bearing to the object. What 40 of 44 Reviewers computed by hand from scene_meta.json.

    PYTHONPATH=src python3 scripts/rsi/tools/start.py <pool dir> <label> [<label> ...] [--scene <scene dir>]
    PYTHONPATH=src python3 scripts/rsi/tools/start.py <pool dir> --task pick_Mug_2
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCENE = REPO / "outputs" / "scene_discovery" / "adt_apartment_mars"


def yaw_of(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def surface_under(meta: dict, xyz) -> tuple | None:
    """The surface whose footprint holds xy and whose top is nearest below the object."""
    best = None
    for k, s in meta.get("surfaces", {}).items():
        c, h, a = s["xy"], s["half_extent"], s.get("yaw", 0.0)
        dx, dy = xyz[0] - c[0], xyz[1] - c[1]
        lx, ly = math.cos(-a) * dx - math.sin(-a) * dy, math.sin(-a) * dx + math.cos(-a) * dy
        dz = xyz[2] - s["top"]
        if abs(lx) <= h[0] + 0.02 and abs(ly) <= h[1] + 0.02 and -0.05 <= dz <= 0.5 and (best is None or dz < best[0]):
            best = (dz, k, s, lx, ly)
    return None if best is None else best


def describe(meta: dict, row: dict) -> list[str]:
    st = row["start"]
    rx, ry = st["robot_xy"]
    out = [f"task {row['task']}  robot ({rx:.3f}, {ry:.3f}) yaw {math.degrees(st['robot_yaw']):.1f} deg"]
    for key, (pos, quat) in st["placements"].items():
        o = meta["objects"].get(key, {})
        d = math.hypot(pos[0] - rx, pos[1] - ry)
        bearing = math.degrees(math.atan2(pos[1] - ry, pos[0] - rx) - st["robot_yaw"])
        bearing = (bearing + 180) % 360 - 180
        box = o.get("box", {}).get("half_extent")
        out.append(f"  {key}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) yaw {math.degrees(yaw_of(quat)):.1f} deg; "
                   f"half extent {box}; {d:.2f} m from the robot, bearing {bearing:+.0f} deg")  # fmt: skip
        s = surface_under(meta, pos)
        if s is None:
            out.append("    on no surface of the scene file")
            continue
        dz, k, surf, lx, ly = s
        h = surf["half_extent"]
        edges = {"+x": h[0] - lx, "-x": h[0] + lx, "+y": h[1] - ly, "-y": h[1] + ly}
        out.append(f"    on {k} ({surf.get('phrase', '')}): top {surf['top']:.3f} m, object centre {dz:+.3f} m above; "
                   f"half extent [{h[0]:.3f}, {h[1]:.3f}], yaw {math.degrees(surf.get('yaw', 0)):.0f} deg; "
                   f"approach edges {surf.get('approach_edges')}")  # fmt: skip
        out.append(f"    in the surface frame ({lx:+.3f}, {ly:+.3f}); to the edges "
                   + ", ".join(f"{e} {v:.3f}" for e, v in edges.items()))  # fmt: skip
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool", type=Path)
    ap.add_argument("labels", nargs="*")
    ap.add_argument("--task", default="")
    ap.add_argument("--scene", type=Path, default=SCENE)
    a = ap.parse_args(argv)
    meta = json.loads((a.scene / "scene_meta.json").read_text())
    labels = a.labels or sorted(p.stem for p in a.pool.glob(f"{a.task}_s*.json"))
    for x in labels:
        row = json.loads((a.pool / f"{x}.json").read_text())
        print(x)
        print("\n".join(describe(meta, row)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
