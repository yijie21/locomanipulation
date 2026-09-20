#!/usr/bin/env python3
"""Build conditional x/y bimanual IK slices from one measured standing state."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from locomani.tasks.crate.reachability_atlas import (
    AtlasConfig,
    ConditionalSlice,
    ReachabilityAtlas,
    attach_original_expert_evidence,
    default_slices,
    load_cache,
    load_standing_state,
)


def _load_config(path: Path | None) -> tuple[AtlasConfig, tuple[ConditionalSlice, ...]]:
    if path is None:
        return AtlasConfig(), default_slices()
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("map config schema_version must be 1")
    unknown = set(raw) - {"schema_version", "atlas", "slices"}
    if unknown:
        raise ValueError(f"unknown map config fields: {', '.join(sorted(unknown))}")
    atlas_raw = raw.get("atlas", {})
    allowed = {field.name for field in fields(AtlasConfig)}
    if not isinstance(atlas_raw, dict) or set(atlas_raw) - allowed:
        raise ValueError("map config atlas contains unknown fields")
    config = AtlasConfig(**atlas_raw)
    slices_raw = raw.get("slices")
    slices = default_slices() if slices_raw is None else tuple(ConditionalSlice(**value) for value in slices_raw)
    if len(slices) != 7 or len({value.slice_id for value in slices}) != 7:
        raise ValueError("first atlas version requires seven uniquely named conditional slices")
    return config, slices


def _map_one(arguments: tuple) -> dict:
    scene_xml, standing_state, config_raw, condition_raw, cached_slice = arguments
    atlas = ReachabilityAtlas(Path(scene_xml), standing_state, AtlasConfig(**config_raw))
    return atlas.map_slice(ConditionalSlice(**condition_raw), cached_slice)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-xml", type=Path, required=True)
    parser.add_argument("--standing-state", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cache", type=Path, help="compatible prior atlas JSON used as a deterministic sample cache")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-bank-output", type=Path)
    parser.add_argument("--dynamic-result", type=Path, action="append", default=[])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 7:
        parser.error("--workers must be between 1 and 7")

    config, conditions = _load_config(args.config.resolve() if args.config else None)
    state = load_standing_state(args.standing_state.resolve())
    atlas = ReachabilityAtlas(args.scene_xml.resolve(), state, config)
    cache = load_cache(args.cache.resolve() if args.cache else None, atlas.identity["sha256"])
    if args.workers == 1:
        mapped = [atlas.map_slice(condition, cache.get(condition.slice_id)) for condition in conditions]
    else:
        tasks = [
            (
                str(args.scene_xml.resolve()),
                state,
                config.to_dict(),
                {
                    "slice_id": condition.slice_id,
                    "height_m": condition.height_m,
                    "separation_m": condition.separation_m,
                    "yaw_deg": condition.yaw_deg,
                },
                cache.get(condition.slice_id),
            )
            for condition in conditions
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            mapped = list(executor.map(_map_one, tasks))
    result = atlas.result(mapped)
    for dynamic_result in args.dynamic_result:
        attach_original_expert_evidence(result, dynamic_result.resolve())
    _atomic_json(args.output.resolve(), result)
    bank_output = args.validation_bank_output or (args.output.parent / "validation_bank.json")
    _atomic_json(bank_output.resolve(), result["representative_bank"])
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "validation_bank": str(bank_output.resolve()),
                "atlas_identity_sha256": atlas.identity["sha256"],
                "slice_count": len(mapped),
                "sample_count": sum(value["summary"]["sample_count"] for value in mapped),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
