#!/usr/bin/env python3
"""Write the Google Scanned Objects catalog `assets/mujoco_scanned_objects/catalog.json` (user Q2, 2026-11-03): every
object's size, hull volume, mass, grasp height, dataset category, graspability (`single` / `bimanual` / `none`),
whether it fits the logistics room's carton (`fits_carton`: footprint inside the inner box less a margin, no taller
than the rim; the scenes draw only objects that fit, user 2026-11-05) and train / held-out split, from the meshes under `models/`, the dataset's category labels in `categories.json`
(scripts/gso_categories.py) and the thresholds in locomani.sim.gso_objects.Thresholds (each one a flag here, recorded
in the file). Only objects of the categories in `--use-categories` can be drawn; the others are `none` whatever their
shape. The default is every label but Shoe, Toys, Stuffed Toys, Mouse, Media Cases, Legos, Keyboard, Car Seat, Action
Figures and Hat (user 2026-11-05; gso_objects.USED_CATEGORIES), with the unlabelled models (`(none)`). `all` uses
every category; `--exclude-categories A B` uses every category but those. Names are the labels of categories.json,
matched without case. The hand-kept list `hand_choices.json` (user 2026-11-17; clicked in `scripts/gso_gallery.py
--serve`) excludes or includes an object over its category and sets its shape kind over the shape rules. The scene builder draws from this file, never from the 2 GB of
meshes.

    $VENV_SIM_PY scripts/gso_catalog.py [--models assets/mujoco_scanned_objects/models] [--out .../catalog.json]
        [--categories .../categories.json] [--use-categories "Consumer Goods" Bag ... | all]
        [--exclude-categories Shoe Toys ...] [--hand-choices .../hand_choices.json] [--jaw 0.095] [--bimanual-width 0.20 0.45]
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


def resolve_categories(use: list[str] | None, exclude: list[str] | None, labels: dict[str, str | None]):
    """The used-category tuple (None: every category) from the two flags, every name matched without case against the
    labels of categories.json plus `(none)`; an unknown name stops the build."""
    known = sorted({c or G.UNLABELLED for c in labels.values()} | {G.UNLABELLED})
    by_lower = {k.lower(): k for k in known}

    def real(names):
        bad = [n for n in names if n.lower() not in by_lower]
        if bad:
            raise SystemExit(f"unknown categories {bad}; categories.json has {known}")
        return [by_lower[n.lower()] for n in names]

    if exclude is not None:
        drop = set(real(exclude))
        return tuple(k for k in known if k not in drop)
    if use is not None and [n.lower() for n in use] == ["all"]:
        return None
    if not labels:  # no categories.json: no object is labelled, so no list can select them
        raise SystemExit("no categories.json: python3 scripts/gso_categories.py (or --use-categories all)")
    return G.USED_CATEGORIES if use is None else tuple(real(use))  # the default is checked against the tracked labels


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, default=ASSETS / "models")
    ap.add_argument("--out", type=Path, default=ASSETS / "catalog.json")
    ap.add_argument("--categories", type=Path, default=ASSETS / "categories.json",
                    help="scripts/gso_categories.py's file: the dataset's category label per model")  # fmt: skip
    ap.add_argument("--hand-choices", type=Path, default=ASSETS / G.HAND_CHOICES_FILE,
                    help="the hand-kept list {objects: {name: {excluded, exclude_why, shape, shape_why}}}; missing: none")  # fmt: skip
    d = G.Thresholds()
    pick = ap.add_mutually_exclusive_group()
    pick.add_argument("--use-categories", nargs="+", default=None, metavar="CATEGORY",
                      help="dataset categories whose objects can be drawn, `(none)` for the unlabelled ones, or `all` "
                           "(default: gso_objects.USED_CATEGORIES, every label but shoes, toys, mice, ...)")  # fmt: skip
    pick.add_argument("--exclude-categories", nargs="*", default=None, metavar="CATEGORY",
                      help="use every category but these (no value: every category)")  # fmt: skip
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
    ap.add_argument("--carton-inner", type=float, nargs=2, default=list(d.carton_inner_m), metavar=("LONG", "SHORT"),
                    help="m; the carton's inner box an object must fit (`fits_carton`, user 2026-11-05)")  # fmt: skip
    ap.add_argument(
        "--carton-rim", type=float, default=d.carton_rim_m, help="m; an object taller than this does not fit"
    )
    ap.add_argument("--fit-margin", type=float, default=d.fit_margin_m, help="m; clearance on every side of the carton")
    ap.add_argument(
        "--held-out", type=float, default=d.held_out_fraction, help="fraction of names hashed into the evaluation split"
    )
    a = ap.parse_args(argv)
    categories = G.load_categories(a.categories) if a.categories.is_file() else {}
    th = G.Thresholds(
        jaw_m=a.jaw, min_height_m=a.min_height, surface_top_m=a.surface_top, min_grasp_height_m=a.min_grasp_height,
        bimanual_width_m=tuple(a.bimanual_width), bimanual_min_height_m=a.bimanual_min_height,
        density_kg_m3=a.density, mass_range_kg=tuple(a.mass_range), held_out_fraction=a.held_out,
        used_categories=resolve_categories(a.use_categories, a.exclude_categories, categories),
        carton_inner_m=tuple(a.carton_inner), carton_rim_m=a.carton_rim, fit_margin_m=a.fit_margin,
    )  # fmt: skip
    version = a.models.parent / "VERSION"
    source = {
        "dataset": "Google Scanned Objects (Downs et al. 2022, CC-BY 4.0), MJCF by kevinzakka/mujoco_scanned_objects (MIT)",
        "models_version": version.read_text().strip() if version.is_file() else None,
        "models_dir": str(a.models.relative_to(ROOT)) if a.models.is_relative_to(ROOT) else str(a.models),
        "categories": (str(a.categories.relative_to(ROOT)) if a.categories.is_relative_to(ROOT) else str(a.categories))
        if a.categories.is_file()
        else None,
        "hand_choices": (str(a.hand_choices.relative_to(ROOT)) if a.hand_choices.is_relative_to(ROOT)
                             else str(a.hand_choices)) if a.hand_choices.is_file() else None,
    }  # fmt: skip
    t0 = time.time()
    by_hand = G.load_hand_choices(a.hand_choices)
    cat = G.build_catalog(a.models, th, source, categories=categories, by_hand=by_hand)
    a.out.write_text(json.dumps(cat, indent=1) + "\n")
    print(json.dumps({"out": str(a.out), "counts": cat["counts"], "not_used_categories": cat["not_used_categories"], "thresholds": {f.name: getattr(th, f.name) for f in fields(th)},
                      "seconds": round(time.time() - t0, 1)}, indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
