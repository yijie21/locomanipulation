"""Open the packaged USD with Blender's pxr and verify its dependency closure."""

from __future__ import annotations

import sys
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom

EXPECTED_BOX_PRIM = Sdf.Path(
    "/root/NOR_CAL_carton___45_x_32_x_33_cm/nor_cal_box/Front___printed__compressed_paperboard"
)


def _close(actual, expected, tolerance: float = 1e-6) -> bool:
    return all(abs(float(left) - right) <= tolerance for left, right in zip(actual, expected))


def main() -> int:
    args = sys.argv[sys.argv.index("--") + 1 :]
    if len(args) != 1:
        raise SystemExit("usage: blender --background --python check_usd.py -- <scene.usdc>")
    scene = Path(args[0]).resolve()
    stage = Usd.Stage.Open(str(scene))
    if stage is None:
        raise AssertionError(f"failed to open {scene}")
    if stage.GetDefaultPrim().GetPath() != Sdf.Path("/root"):
        raise AssertionError(f"unexpected default prim: {stage.GetDefaultPrim().GetPath()}")
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise AssertionError(f"unexpected up axis: {UsdGeom.GetStageUpAxis(stage)}")
    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) > 1e-12:
        raise AssertionError(f"unexpected metres per unit: {UsdGeom.GetStageMetersPerUnit(stage)}")

    references = []
    for prim in stage.Traverse():
        for attribute in prim.GetAttributes():
            if attribute.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = attribute.Get()
            if not value or not value.path:
                continue
            path = Path(value.path)
            if path.is_absolute():
                raise AssertionError(f"absolute USD asset path: {value.path}")
            resolved = (scene.parent / path).resolve()
            if not resolved.is_file():
                raise AssertionError(f"missing USD asset: {prim.GetPath()} {attribute.GetName()} -> {resolved}")
            references.append(value.path)
    if len(references) != 49:
        raise AssertionError(f"expected 49 USD image references, got {len(references)}")
    referenced_files = {(scene.parent / Path(value)).resolve() for value in references}
    packaged_files = {path.resolve() for path in (scene.parent / "textures").iterdir() if path.is_file()}
    if referenced_files != packaged_files:
        raise AssertionError(
            "USD texture closure differs: "
            f"unreferenced={sorted(packaged_files - referenced_files)}, "
            f"unpackaged={sorted(referenced_files - packaged_files)}"
        )

    box = stage.GetPrimAtPath(EXPECTED_BOX_PRIM)
    if not box:
        raise AssertionError(f"missing visual box prim: {EXPECTED_BOX_PRIM}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    aligned = cache.ComputeWorldBound(box).ComputeAlignedRange()
    lower = aligned.GetMin()
    upper = aligned.GetMax()
    if not _close(lower, (-0.225, -0.55, 0.371)):
        raise AssertionError(f"unexpected USD box minimum: {lower}")
    if not _close(upper, (0.225, -0.23, 0.701)):
        raise AssertionError(f"unexpected USD box maximum: {upper}")
    print(f"USD_OK refs={len(references)} box_min={tuple(lower)} box_max={tuple(upper)} scene={scene}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
