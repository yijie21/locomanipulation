"""Build the walk scene: MuJoCo's blue floor with nothing on it but the G1.

    python -m locomani.tasks.walk.scene outputs/scene/walk

It is the reach-envelope floor scene (the bimanual collection robot, table and crate removed, so the
physics is the crate line's) plus one camera that follows the robot. The floor scene's two cameras
are fixed near the origin and lose a robot that walks more than a metre, so `walk_follow` becomes
the scene's video camera.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from locomani.tasks.crate.reach_envelope import floor_scene

# A worldbody camera in trackcom mode keeps the offset it has, in the XML, from the model's centre of
# mass -- with nothing else in the scene, the robot's. So it is placed from the pelvis as the XML has
# it (x=-1.5, the crate scene's start; the reset moves the robot to robot_start afterwards): 3 m to
# the robot's right (yaw 0 faces +x), looking across at it and slightly down.
FOLLOW = {"name": "walk_follow", "mode": "trackcom", "xyaxes": "1 0 0 0 0.15 1", "fovy": "60"}
FOLLOW_OFFSET = (0.0, -3.0, 1.0)


def add_follow_camera(scene_dir: Path) -> None:
    xml_path, meta_path = scene_dir / "scene_locomani.xml", scene_dir / "scene_meta.json"
    tree = ET.parse(xml_path)
    worldbody = tree.getroot().find("worldbody")
    if worldbody.find(f"camera[@name='{FOLLOW['name']}']") is None:
        pelvis = [float(v) for v in worldbody.find("body[@name='pelvis']").get("pos").split()]
        pos = (pelvis[0] + FOLLOW_OFFSET[0], pelvis[1] + FOLLOW_OFFSET[1], FOLLOW_OFFSET[2])
        ET.SubElement(worldbody, "camera", dict(FOLLOW, pos=" ".join(f"{v:g}" for v in pos)))
    tree.write(xml_path, encoding="unicode")
    meta = json.loads(meta_path.read_text())
    meta.update(video_camera=FOLLOW["name"], purpose="floor-only G1 for SONIC locomotion measurements")
    import mujoco

    mujoco.MjModel.from_xml_path(str(xml_path))  # must compile
    meta_path.write_text(json.dumps(meta, indent=2))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)
    out = args.out_dir.resolve()
    floor_scene.main([str(out)])
    add_follow_camera(out)
    print(f"[walk_scene] follow camera {FOLLOW['name']} -> {out / 'scene_locomani.xml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
