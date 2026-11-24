"""A LocoManiLab scene as a scene directory the ScaleBFM runtime loads (escalation plan, stage A-L, 2026-11-17).

LocoManiLab's environment exports its compiled scene with `env.unwrapped.model.get_xml()`: the room, the objects and
the robot (robosuite names: `robot0_pelvis`, `robot0_left_hip_pitch_joint`, ...), the robot's free joint after the
objects', and that joint on a massless root at floor level with the real pelvis 0.793 m above it
(`mobilebase0_support`). The runtime wants the robot first in qpos, Unitree's own names and the free joint at the
pelvis. `mirror` moves the robot's body to the front of the world, strips the `robot0_` prefix and lifts the root to the
pelvis (ROOT_UP is the same offset for states that travel between the two models); nothing else changes (meshes stay
where LocoManiLab keeps them). The robot is the Unitree G1 with Dex3 hands of SIMPLE's model: same link offsets, 36.17 kg.

    python lml_scene.py <exported.xml> <out dir>   -> <out dir>/scene_locomani.xml, scene_meta.json
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PREFIX = "robot0_"


def mirror(xml_in: str, out: Path) -> Path:
    tree = ET.parse(xml_in)
    root = tree.getroot()
    world = root.find("worldbody")
    robot = next(b for b in world.findall("body") if b.get("name") == PREFIX + "pelvis")
    world.remove(robot)
    world.insert(0, robot)
    support = robot.find("body[@name='robot0_mobilebase0_support']") or robot.find("body[@name='mobilebase0_support']")
    up = [float(v) for v in support.get("pos").split()]
    q = [float(v) for v in robot.get("quat", "1 0 0 0").split()]
    assert abs(q[1]) + abs(q[2]) < 1e-6 and up[0] == up[1] == 0, (
        "the root is expected upright, the pelvis straight above it"
    )
    robot.set("pos", " ".join(str(a + b) for a, b in zip(map(float, robot.get("pos").split()), up)))
    support.set("pos", "0 0 0")
    names = {(e.tag, e.get("name")) for e in root.iter() if e.get("name")}
    for e in root.iter():
        for k, v in list(e.attrib.items()):  # names and every reference to them (joint=, mesh=, body1=, ...)
            if k != "file" and v.startswith(PREFIX):
                assert k != "name" or (e.tag, v[len(PREFIX) :]) not in names, f"{v} without its prefix is taken"
                e.set(k, v[len(PREFIX) :])
    out.mkdir(parents=True, exist_ok=True)
    tree.write(out / "scene_locomani.xml")
    (out / "scene_meta.json").write_text(
        json.dumps(dict(surfaces={}, objects={}, lml=dict(source=str(xml_in), root_up=up)))
    )
    return out / "scene_locomani.xml"


if __name__ == "__main__":
    print(mirror(sys.argv[1], Path(sys.argv[2])))
