"""cuRobo's G1 + Dex3 adjusted to HumanoidArena's robot (g1_29dof_with_dex3_rev_1_0_m2.usd).

The kinematics check (kin_isaac.py + kin_compare.py, 200 random right-arm and waist poses, 2026-11-24) found SIMPLE's
g1_29dof_with_dex3.urdf off Isaac Lab's kinematics of HumanoidArena's G1 by 2.2 mm mean, 3.7 mm max at the right wrist
and palm, rotations exact: the waist differs. A least-squares fit of the waist joint origins (and one shift of everything
the torso carries) brings it to 0.0005 mm. HumanoidArena's G1 has the rev-1.0 waist (Unitree's g1_29dof_rev_1_0.urdf):

  waist_roll_joint   origin z 0.035 -> 0.044
  waist_pitch_joint  origin z 0.019 -> 0.0
  every joint on torso_link (shoulders, head, camera, imu, logo, ...)  origin z + 0.010

so the torso link's frame sits 10 mm lower and what it carries 10 mm higher in its frame. Nothing else changes. The torso's
collision spheres (spheres/g1_29dof.yml) are not moved: the planners here check the right arm and hand only.

    $SIMPLE/.venv/bin/python -I make_ha_config.py   (reads SIMPLE's files, writes next to this file)
-> g1_29dof_with_dex3_ha.urdf (mesh paths absolute, SIMPLE's meshes), g1_29dof_with_dex3_ha.yml (urdf_path absolute)
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

S = Path("/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE/data/robots/g1")
HERE = Path(__file__).resolve().parent
DZ_ROLL, DZ_PITCH, DZ_TORSO = 0.009, -0.019, 0.010


def main() -> None:
    tree = ET.parse(S / "g1_29dof_with_dex3.urdf")
    root = tree.getroot()
    changed = []
    for j in root.iter("joint"):
        o = j.find("origin")
        dz = 0.0
        if j.get("name") == "waist_roll_joint":
            dz = DZ_ROLL
        elif j.get("name") == "waist_pitch_joint":
            dz = DZ_PITCH
        elif j.find("parent").get("link") == "torso_link":
            dz = DZ_TORSO
        if dz:
            xyz = [float(v) for v in o.get("xyz").split()]
            xyz[2] = round(xyz[2] + dz, 9)
            o.set("xyz", " ".join(repr(v) for v in xyz))
            changed.append((j.get("name"), dz))
    for m in root.iter("mesh"):
        f = m.get("filename")
        if f and not f.startswith("/") and "://" not in f:
            m.set("filename", str(S / f))
    out = HERE / "g1_29dof_with_dex3_ha.urdf"
    tree.write(out, xml_declaration=True, encoding="utf-8")
    cfg = yaml.safe_load(open(S / "curobo/g1_29dof_with_dex3.yml"))
    cfg["robot_cfg"]["kinematics"]["urdf_path"] = str(out)
    yaml.safe_dump(cfg, open(HERE / "g1_29dof_with_dex3_ha.yml", "w"), sort_keys=False)
    print("changed", changed)


if __name__ == "__main__":
    main()
