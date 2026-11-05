#!/usr/bin/env python3
"""Write the Google Scanned Objects catalog `assets/mujoco_scanned_objects/catalog.json` (user Q2, 2026-11-03): every
object's size, hull volume, mass, grasp height, dataset category, graspability (`single` / `bimanual` / `none`) and
train / held-out split, from the meshes under `models/`, the dataset's category labels in `categories.json`
(scripts/gso_categories.py) and the thresholds in locomani.sim.gso_objects.Thresholds (each one a flag here, recorded
in the file). A category in `--exclude-categories` (default `Shoe`: user 2026-11-04, no shoes in the logistics room)
is `none` whatever its shape. The scene builder draws from this file, never from the 2 GB of meshes.

    $VENV_SIM_PY scripts/gso_catalog.py [--models assets/mujoco_scanned_objects/models] [--out .../catalog.json]
        [--categories .../categories.json] [--exclude-categories Shoe] [--jaw 0.095] [--bimanual-width 0.20 0.45]
        [--bimanual-min-height 0.10] [--density 300] ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from locomani.sim import gso_objects as G  # noqa: E402

ASSETS = ROOT / "assets/mujoco_scanned_objects"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, default=ASSETS / "models")
    ap.add_argument("--out", type=Path, default=ASSETS / "catalog.json")
    ap.add_argument("--categories", type=Path, default=ASSETS / "categories.json",
                    help="scripts/gso_categories.py's file: the dataset's category label per model")  # fmt: skip
    d = G.Thresholds()
    ap.add_argument("--exclude-categories", nargs="*", default=list(d.excluded_categories), metavar="CATEGORY",
                    help="objects of these dataset categories are `none` (never drawn); give none to exclude nothing")  # fmt: skip
    ap.add_argument(
        "--jaw", type=float, default=d.jaw_m, help="m; the open Dex1 jaw a single-hand object's narrow side fits"
    )
    ap.add_argument("--min-height", type=float, default=d.min_height_m, help="m; flatter objects are `none`")
    ap.add_argument(
        "--surface-top", type=float, default=d.surface_top_m, help="m; the platform top the grasp height counts from"
    )
    ap.add_argument(
        "--min-grasp-height",
        type=float,
        default=d.min_grasp_height_m,
        help="m; a single-hand grasp point under it is `none`",
    )
    ap.add_argument("--bimanual-width", type=float, nargs=2, default=list(d.bimanual_width_m), metavar=("LO", "HI"),
                    help="m; a two-hand object's wide side lies in this range")  # fmt: skip
    ap.add_argument("--bimanual-min-height", type=float, default=d.bimanual_min_height_m, help="m")
    ap.add_argument("--density", type=float, default=d.density_kg_m3, help="kg/m^3 over the hull volume")
    ap.add_argument("--mass-range", type=float, nargs=2, default=list(d.mass_range_kg), metavar=("LO", "HI"), help="kg")
    ap.add_argument(
        "--held-out", type=float, default=d.held_out_fraction, help="fraction of names hashed into the evaluation split"
    )
    a = ap.parse_args(argv)
    th = G.Thresholds(
        jaw_m=a.jaw, min_height_m=a.min_height, surface_top_m=a.surface_top, min_grasp_height_m=a.min_grasp_height,
        bimanual_width_m=tuple(a.bimanual_width), bimanual_min_height_m=a.bimanual_min_height,
        density_kg_m3=a.density, mass_range_kg=tuple(a.mass_range), held_out_fraction=a.held_out,
        excluded_categories=tuple(a.exclude_categories),
    )  # fmt: skip
    if th.excluded_categories and not a.categories.is_file():
        raise SystemExit(
            f"no {a.categories}: python3 scripts/gso_categories.py (or --exclude-categories with no value)"
        )
    categories = G.load_categories(a.categories) if a.categories.is_file() else {}
    version = a.models.parent / "VERSION"
    source = {
        "dataset": "Google Scanned Objects (Downs et al. 2022, CC-BY 4.0), MJCF by kevinzakka/mujoco_scanned_objects (MIT)",
        "models_version": version.read_text().strip() if version.is_file() else None,
        "models_dir": str(a.models.relative_to(ROOT)) if a.models.is_relative_to(ROOT) else str(a.models),
        "categories": (str(a.categories.relative_to(ROOT)) if a.categories.is_relative_to(ROOT) else str(a.categories))
        if a.categories.is_file()
        else None,
    }
    t0 = time.time()
    cat = G.build_catalog(a.models, th, source, categories=categories)
    a.out.write_text(json.dumps(cat, indent=1) + "\n")
    print(json.dumps({"out": str(a.out), "counts": cat["counts"], "thresholds": {f.name: getattr(th, f.name) for f in fields(th)},
                      "seconds": round(time.time() - t0, 1)}, indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
