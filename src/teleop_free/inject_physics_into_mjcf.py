"""Copy mass and friction from EmbodiedGen URDFs into the exported MJCF files.

EmbodiedGen spends a VLM call per asset estimating physical properties and
writes them into the URDF:

    <inertial><mass value="0.1900"/></inertial>
    <collision><gazebo><mu1>0.60</mu1><mu2>0.50</mu2></gazebo></collision>

but `cvt_embodiedgen_asset_to_anysim(..., AssetType.MJCF, ...)` drops all of it
(asset_converter.py only ever sets `density="0"` on the visual geom). MuJoCo then
falls back to its default density, which produced a 77 kg orange and a 7.3 t
dining table in this scene.

The meshes themselves are already metric — `extra_info/scale` in the URDF is a
record of the scale that was *already applied*, not a factor still to apply.
Multiplying by it again shrinks everything ~10x.

Fix: put `mass` on the collision geom (MuJoCo then derives the inertia tensor
from the mesh with that total mass) and map mu1 onto the geom's slide friction.

Usage:
    python src/teleop_free/inject_physics_into_mjcf.py <layout_dir> <mjcf_root>
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# MuJoCo's default spin/roll friction; only the slide term is estimated for us.
DEFAULT_SPIN_FRICTION = 0.005
DEFAULT_ROLL_FRICTION = 0.0001


def read_urdf_physics(urdf_path: Path) -> dict:
    root = ET.parse(urdf_path).getroot()

    mass_el = root.find(".//inertial/mass")
    mu1_el = root.find(".//collision/gazebo/mu1")

    return {
        "mass": float(mass_el.get("value")) if mass_el is not None else None,
        "mu1": float(mu1_el.text) if mu1_el is not None else None,
    }


def inject(mjcf_path: Path, physics: dict) -> str:
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    collision_geoms = [
        g
        for g in root.iter("geom")
        # visual geoms are the ones excluded from collision
        if g.get("contype") != "0"
    ]
    if not collision_geoms:
        return "no collision geom"

    notes = []
    if physics["mass"] is not None:
        # `mass` overrides density; MuJoCo still takes the inertia *shape* from
        # the mesh, so this only rescales the magnitude.
        share = physics["mass"] / len(collision_geoms)
        for g in collision_geoms:
            g.set("mass", f"{share:.6f}")
            g.attrib.pop("density", None)
        notes.append(f"mass={physics['mass']}")

    if physics["mu1"] is not None:
        friction = f"{physics['mu1']} {DEFAULT_SPIN_FRICTION} {DEFAULT_ROLL_FRICTION}"
        for g in collision_geoms:
            g.set("friction", friction)
        notes.append(f"friction={physics['mu1']}")

    tree.write(mjcf_path, encoding="utf-8", xml_declaration=True)

    return ", ".join(notes) or "nothing to inject"


def main() -> None:
    layout_dir = Path(sys.argv[1])
    mjcf_root = Path(sys.argv[2])

    for urdf_path in sorted(layout_dir.glob("asset3d/*/result/*.urdf")):
        name = urdf_path.stem
        mjcf_path = mjcf_root / name / f"{name}.xml"
        if not mjcf_path.exists():
            print(f"{name:14s} SKIP (no MJCF at {mjcf_path})")
            continue

        physics = read_urdf_physics(urdf_path)
        result = inject(mjcf_path, physics)
        print(f"{name:14s} {result}")


if __name__ == "__main__":
    main()
