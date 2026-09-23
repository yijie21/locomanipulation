#!/usr/bin/env python3
"""Validate the versioned logistics-room scene without starting a renderer."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]
MANIFEST = PACKAGE / "manifest.json"
MJCF = PACKAGE / "mujoco/scene_locomani.xml"
USD = PACKAGE / "isaac/logistics_room_box.usdc"
SONIC_MESHES = (REPO / "third_party/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1/meshes").resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_record(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    total = 0
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(root).as_posix()
        file_hash = _sha256(path)
        digest.update(f"{file_hash}  {relative}\n".encode())
        total += path.stat().st_size
    return digest.hexdigest(), len(files), total


def _floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in value.split())


def _assert_close(actual, expected, tolerance: float = 1e-9) -> None:
    if len(actual) != len(expected) or any(abs(left - right) > tolerance for left, right in zip(actual, expected)):
        raise AssertionError(f"expected {expected}, got {actual}")


def _obj_bounds(path: Path) -> tuple[tuple[float, ...], tuple[float, ...], int]:
    lower = [float("inf")] * 3
    upper = [float("-inf")] * 3
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            point = [float(value) for value in line.split()[1:4]]
            lower = [min(old, value) for old, value in zip(lower, point)]
            upper = [max(old, value) for old, value in zip(upper, point)]
            count += 1
    if not count:
        raise AssertionError(f"no vertices in {path}")
    return tuple(lower), tuple(upper), count


def check_manifest() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    for name in ("mujoco", "isaac"):
        expected = manifest["payloads"][name]
        actual_hash, actual_count, actual_bytes = _tree_record(PACKAGE / name)
        actual = {
            "tree_sha256": actual_hash,
            "file_count": actual_count,
            "bytes": actual_bytes,
        }
        if actual != expected:
            raise AssertionError(f"{name} payload differs: expected {expected}, got {actual}")
    return manifest


def check_mjcf(*, load: bool) -> dict:
    root = ET.parse(MJCF).getroot()
    refs = []
    for element in root.iter():
        value = element.get("file")
        if value is None:
            continue
        if Path(value).is_absolute():
            raise AssertionError(f"absolute MJCF reference: {value}")
        resolved = (MJCF.parent / value).resolve()
        if not resolved.is_file():
            raise AssertionError(f"missing MJCF reference: {value} -> {resolved}")
        try:
            resolved.relative_to(PACKAGE)
        except ValueError:
            try:
                resolved.relative_to(SONIC_MESHES)
            except ValueError as error:
                raise AssertionError(f"undeclared external MJCF dependency: {resolved}") from error
        refs.append(value)

    world = root.find("worldbody")
    assert world is not None
    platform = world.find("geom[@name='platform_col']")
    box = world.find("body[@name='nor_cal_box_object']")
    floor = world.find("geom[@name='floor']")
    pelvis = world.find("body[@name='pelvis']")
    if None in (platform, box, floor, pelvis):
        raise AssertionError("required platform, box, floor, or pelvis is missing")
    _assert_close(_floats(platform.get("pos", "")), (0.0, -0.105, 0.1855))
    _assert_close(_floats(platform.get("size", "")), (0.6, 0.45, 0.1855))
    _assert_close(_floats(box.get("pos", "")), (0.0, -0.39, 0.371))
    _assert_close(_floats(floor.get("pos", "")), (0.0, 0.0, 0.001))
    _assert_close(_floats(pelvis.get("pos", "")), (0.0, -1.205, 0.793))

    colliders = {
        geom.get("name"): geom
        for geom in box.findall("geom")
        if geom.get("name", "").startswith("nor_cal_") and not geom.get("name", "").startswith("nor_cal_vis")
    }
    if set(colliders) != {
        "nor_cal_floor",
        "nor_cal_xp",
        "nor_cal_xm",
        "nor_cal_yp",
        "nor_cal_ym",
    }:
        raise AssertionError(f"unexpected box collision closure: {sorted(colliders)}")
    for geom in colliders.values():
        if geom.get("contype", "1") == "0":
            raise AssertionError(f"inactive box collider: {geom.get('name')}")

    # the carton is one OBJ per material (six are photographs); measure their union
    pieces = sorted((MJCF.parent / "mesh").glob("nor_cal_*.obj"))
    if not pieces:
        raise AssertionError("no carton meshes under mesh/nor_cal_*.obj")
    bounds = [_obj_bounds(piece) for piece in pieces]
    lower = tuple(min(b[0][axis] for b in bounds) for axis in range(3))
    upper = tuple(max(b[1][axis] for b in bounds) for axis in range(3))
    vertices = sum(b[2] for b in bounds)
    _assert_close(lower, (-0.225, -0.16, 0.0), tolerance=1e-6)
    _assert_close(upper, (0.225, 0.16, 0.33), tolerance=1e-6)

    result = {
        "file_references": len(refs),
        "visual_box_parts": len(pieces),
        "visual_box_bounds_m": {"min": lower, "max": upper},
        "visual_box_vertices": vertices,
    }
    if load:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(MJCF))
        expected = {"nbody": 36, "ngeom": 129, "nq": 47, "nv": 45, "nu": 33}
        actual = {name: int(getattr(model, name)) for name in expected}
        if actual != expected:
            raise AssertionError(f"compiled MuJoCo topology differs: {actual}")
        result["compiled"] = actual
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-mujoco", action="store_true", help="compile the MJCF with mujoco")
    parser.add_argument(
        "--load-usd",
        action="store_true",
        help="open the USD with Blender's bundled pxr and check its asset closure",
    )
    args = parser.parse_args()
    manifest = check_manifest()
    result = {
        "asset_version": manifest["asset_version"],
        "manifest": "ok",
        "mujoco": check_mjcf(load=args.load_mujoco),
    }
    if args.load_usd:
        completed = subprocess.run(
            [
                "blender",
                "--background",
                "--python",
                str(PACKAGE / "check_usd.py"),
                "--",
                str(USD),
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        if completed.returncode != 0 or "USD_OK " not in completed.stdout:
            raise RuntimeError("Blender USD validation did not report success")
        result["usd"] = "ok"
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
