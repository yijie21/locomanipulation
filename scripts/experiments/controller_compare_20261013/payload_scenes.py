"""Payload variants for perturbation e (TODO 25.5, plan §4.6: load +-50 %): the scene with the task object's mass and
inertia scaled. Everything but scene_locomani.xml is linked (json copied), as warehouse_kg24_scene.sh does.

    python3 scripts/experiments/controller_compare_20261013/payload_scenes.py outputs/scene/sim_lowcom water_bottle 0.5 1.5
        -> outputs/scene/sim_lowcom_m050, outputs/scene/sim_lowcom_m150
"""

from __future__ import annotations

import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def variant(src: Path, key: str, factor: float) -> Path:
    out = src.parent / f"{src.name}_m{round(100 * factor):03d}"
    out.mkdir(exist_ok=True)
    for f in src.iterdir():
        dst = out / f.name
        if f.name == "scene_locomani.xml":
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink() if not dst.is_dir() or dst.is_symlink() else shutil.rmtree(dst)
        if f.suffix == ".json":
            shutil.copy(f, dst)
        else:
            dst.symlink_to(f.resolve())
    meta = json.loads((src / "scene_meta.json").read_text())
    body_name = meta["objects"][key]["body"]
    tree = ET.parse(src / "scene_locomani.xml")
    body = tree.getroot().find(f".//body[@name='{body_name}']")
    inertial = body.find("inertial")
    mass = float(inertial.get("mass"))
    inertial.set("mass", f"{mass * factor:.6g}")
    for attr in ("diaginertia", "fullinertia"):
        if inertial.get(attr):
            inertial.set(attr, " ".join(f"{float(v) * factor:.6g}" for v in inertial.get(attr).split()))
    tree.write(out / "scene_locomani.xml", encoding="unicode")
    meta.update(
        scene_xml=str((out / "scene_locomani.xml").resolve()),
        payload=dict(object=key, factor=factor, mass=mass * factor),
    )
    (out / "scene_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{out}: {key} {mass} -> {mass * factor:.3f} kg")
    return out


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    src, key, factors = Path(args[0]), args[1], [float(v) for v in args[2:]]
    for f in factors:
        variant(src, key, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
