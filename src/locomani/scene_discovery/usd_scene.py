"""The Isaac `scene.usda` of a scene_discovery scene: the exported room plus the lights it needs (TODO 19.6.5b).

    python -m locomani.scene_discovery.usd_scene <scene dir with scene_meta.json> <isaac scene dir> [--intensity N]

`scripts/adt_to_usd.py` (Blender) writes the heavy `apartment.usdc` and the per-object USDs; this writes the small
text layer that references them, so lighting is tuned without re-exporting 4.5 GB. A scanned apartment is closed
(walls and ceiling): the dome and distant light `stage.assemble_stage` adds when a scene has none never reach
inside, and the first render came back black (2026-10-03). So every room gets sphere lights on a grid just under
the ceiling. Plain text, no pxr: it runs in any python.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SPACING = 2.5  # m between lights
HEIGHT = 2.3  # m; under the ceiling, above every standing head
RADIUS = 0.15
INTENSITY = 60000.0


def inside(polygon, x: float, y: float) -> bool:
    poly = np.asarray(polygon, dtype=float)
    hit = False
    for (x0, y0), (x1, y1) in zip(poly, np.roll(poly, -1, axis=0)):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            hit = not hit
    return hit


def light_points(rooms: dict, spacing: float = SPACING) -> list[tuple[str, float, float]]:
    """(room, x, y) on a grid inside each room polygon, offset half a step from its bounding box corner."""
    out = []
    for name, r in sorted(rooms.items()):
        poly = np.asarray(r["polygon"], dtype=float)
        lo, hi = poly.min(axis=0), poly.max(axis=0)
        for x in np.arange(lo[0] + spacing / 2, hi[0], spacing):
            for y in np.arange(lo[1] + spacing / 2, hi[1], spacing):
                if inside(poly, x, y):
                    out.append((name, round(float(x), 3), round(float(y), 3)))
    return out


def scene_usda(room_usdc: str, points, intensity: float = INTENSITY, height: float = HEIGHT) -> str:
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        '    upAxis = "Z"',
        "    metersPerUnit = 1",
        '    doc = "scene_discovery scene (scene_discovery.usd_scene); ADT-derived, not in git."',
        ")",
        "",
        'def Xform "World"',
        "{",
        '    def Xform "room" (',
        f"        references = @{room_usdc}@</root>",
        "    )",
        "    {",
        "    }",
        "",
        '    def Xform "interior_lights"',
        "    {",
    ]
    for i, (room, x, y) in enumerate(points):
        lines += [
            f'        def SphereLight "light_{i:02d}"  # {room}',
            "        {",
            f"            float inputs:intensity = {intensity:g}",
            f"            float inputs:radius = {RADIUS:g}",
            "            bool inputs:normalize = 0",
            f"            double3 xformOp:translate = ({x}, {y}, {height})",
            '            uniform token[] xformOpOrder = ["xformOp:translate"]',
            "        }",
        ]
    lines += ["    }", "}", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("scene_dir", type=Path, help="holds scene_meta.json (rooms)")
    ap.add_argument("isaac_dir", type=Path, help="holds apartment.usdc; scene.usda is written here")
    ap.add_argument("--intensity", type=float, default=INTENSITY)
    ap.add_argument("--spacing", type=float, default=SPACING)
    args = ap.parse_args()
    rooms = json.loads((args.scene_dir / "scene_meta.json").read_text())["room"]["rooms"]
    points = light_points(rooms, args.spacing)
    usdc = (args.isaac_dir / "apartment.usdc").resolve()
    (args.isaac_dir / "scene.usda").write_text(scene_usda(str(usdc), points, args.intensity))
    print(f"[usd] {args.isaac_dir / 'scene.usda'}: {len(points)} sphere lights at {args.intensity:g}")


if __name__ == "__main__":
    main()
