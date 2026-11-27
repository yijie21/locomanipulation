#!/usr/bin/env python3
"""Tile the renders of scripts/render_usd_preview.py into one contact sheet (rows: assets, columns: views).

    blender --background --factory-startup --python scripts/usd_contact_sheet.py -- \\
        outputs/samsung_assets_preview outputs/samsung_assets_preview/contact_sheet.png Cable007 Cable008

Reads `<render_dir>/<name>_<view>.png` for every name and view, scales each to one tile and writes the
sheet as PNG; with the default 480x360 tiles and four views it is 1950 px wide, 366 px per asset. It runs
inside Blender, whose Python has numpy and image IO, so it needs no PIL.
"""

from __future__ import annotations

import argparse
import os
import sys

import bpy
import numpy as np


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("render_dir", help="directory holding <name>_<view>.png")
    parser.add_argument("out", help="contact sheet PNG to write")
    parser.add_argument("names", nargs="+", help="asset name prefixes, one row each, top to bottom")
    parser.add_argument("--views", nargs="+", default=["iso", "back", "front", "top"], help="columns, left to right")
    parser.add_argument("--tile", type=int, nargs=2, default=(480, 360), metavar=("W", "H"))
    parser.add_argument("--gap", type=int, default=6, help="white border between tiles, pixels")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    tw, th = args.tile
    gap = args.gap
    missing = [
        f"{n}_{v}.png"
        for n in args.names
        for v in args.views
        if not os.path.isfile(os.path.join(args.render_dir, f"{n}_{v}.png"))
    ]
    if missing:
        print(f"missing renders in {args.render_dir}: {missing}", file=sys.stderr)
        return 1
    width = len(args.views) * tw + (len(args.views) + 1) * gap
    height = len(args.names) * th + (len(args.names) + 1) * gap
    sheet = np.ones((height, width, 4), dtype=np.float32)
    for r, name in enumerate(args.names):
        for c, view in enumerate(args.views):
            img = bpy.data.images.load(os.path.join(args.render_dir, f"{name}_{view}.png"))
            img.scale(tw, th)
            px = np.empty(tw * th * 4, dtype=np.float32)
            img.pixels.foreach_get(px)
            y0 = height - (r + 1) * (th + gap)  # Blender image rows run bottom-up
            x0 = gap + c * (tw + gap)
            sheet[y0 : y0 + th, x0 : x0 + tw] = px.reshape(th, tw, 4)
            bpy.data.images.remove(img)
    out = bpy.data.images.new("sheet", width, height, alpha=False)
    out.pixels.foreach_set(sheet.ravel())
    out.filepath_raw = args.out
    out.file_format = "PNG"
    out.save()
    print("SHEET", args.out, width, height)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []))
