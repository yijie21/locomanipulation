"""The walk scene with the mars robot (TODO 25.4): MuJoCo's blue floor, nothing on it but the mars G1, for the complex
motions C1-C7 and the clip tracking tests. `--load KG` adds KG / 2 to each wrist yaw link, the Dex1 base (the loaded C4 variant:
a carried box's weight at the hands, without a box to drop).

    third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \\
        scripts/experiments/controller_compare_20261013/walk_mars_scene.py outputs/scene/walk_mars
    ... walk_mars_scene.py outputs/scene/walk_mars_load12 --load 1.2
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from locomani.sim import make_dex1_model  # noqa: E402

SOURCE = REPO / "outputs" / "scene" / "walk"
HAND_BASES = ("left_wrist_yaw_link", "right_wrist_yaw_link")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--load", type=float, default=0.0, help="kg added in total, half at each hand base")
    a = ap.parse_args(argv)
    out = a.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    xml = out / "scene_locomani.xml"
    shutil.copy(SOURCE / "scene_locomani.xml", xml)
    make_dex1_model.write_mars()  # the meshes the scene references are gitignored
    tree = make_dex1_model.apply_mars(ET.parse(xml), make_dex1_model.MARS_MESHES)
    tree.write(xml, encoding="unicode")
    make_dex1_model.check_mars(str(xml))  # against the URDF, so before the load
    added = {}
    if a.load > 0:
        for name in HAND_BASES:
            body = tree.getroot().find(f".//body[@name='{name}']")
            inertial = body.find("inertial")
            m = float(inertial.get("mass"))
            inertial.set("mass", f"{m + a.load / 2:.4f}")
            added[name] = round(a.load / 2, 4)
    tree.write(xml, encoding="unicode")
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml))
    meta = json.loads((SOURCE / "scene_meta.json").read_text())
    meta.update(
        scene_xml=str(xml),
        robot="mars",
        nbody=model.nbody,
        ngeom=model.ngeom,
        nq=model.nq,
        nv=model.nv,
        purpose="floor-only mars G1: complex motions C1-C7 and clip tracking (TODO 25.4)",
        hand_load_kg=added,
    )
    (out / "scene_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[walk_mars] {xml} (load {a.load} kg)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
