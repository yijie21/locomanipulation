"""Generate the Dex1 (parallel gripper) variant of the G1 robot model.

Surgery on gear_sonic's g1_29dof_with_hand.xml (Dex3, 7 joints/hand):
remove the Dex3 palm geoms, finger subtrees, motors and sensors, and graft
the official Unitree Dex1-1 gripper (assets/dex1, from unitree_ros
g1_description g1_29dof_mode_15_with_dex1_1.urdf) onto each wrist.

Design constraints honoured:
  - base_sim maps actuators to joints BY ORDER (ctrl = torques indexed by
    joint id), so the gripper motors/sensors are inserted at the exact
    positions the Dex3 ones occupied, and BOTH fingers are actuated joints
    (a passive follower would shift every joint id after it).
  - Joint names contain "left_hand"/"right_hand" so base_sim counts them
    as hand joints; NUM_HAND_MOTORS/JOINTS becomes 2.
  - The deploy sends hand kp=1.5 (radian-tuned); on a prismatic joint that
    is ~0.07 N, so the motors carry gear=150 -> ~225 N/m effective
    stiffness, ~5 N grip per finger at full close command.

Outputs (into gear_sonic's model_data/g1, which is gitignored — rerun this
script after a fresh clone):
  g1_29dof_with_dex1.xml, scene_31dof.xml, meshes/Dex1_*.STL + dex1_col_*
"""

from __future__ import annotations

import os
import shutil
import xml.etree.ElementTree as ET

import paths

REPO = paths.REPO
SRC_DIR = f"{REPO}/assets/dex1"
G1 = f"{REPO}/third_party/GR00T-WholeBodyControl/gear_sonic/data/robot_model/model_data/g1"

# Dex1-1 numbers from the official URDF (see assets/dex1/*.urdf):
BASE_POS = "0.0415 0 0"  # fixed base joint at the wrist yaw link
JOINT_RANGE = "-0.02 0.0245"  # per-finger prismatic travel
FINGER = {
    1: dict(
        axis="0 -1 0",
        mesh="Dex1_finger_link_1",
        col="dex1_col_1",
        ipos="0.073453 -0.011173 0.0086355",
        fullinertia=("2.5583E-05 1.4549E-05 3.6013E-05 5.1557E-07 -6.3178E-07 7.8542E-08"),
    ),
    2: dict(
        axis="0 1 0",
        mesh="Dex1_finger_link_2",
        col="dex1_col_2",
        ipos="0.073453 0.011173 -0.0086355",
        fullinertia=("2.5583E-05 1.4549E-05 3.6013E-05 -5.1557E-07 6.3178E-07 7.8542E-08"),
    ),
}
MASS = "0.086783"
BASE_INERTIAL = dict(
    pos="0.043853 -0.0001306 1.4003E-05",
    mass="0.19138",
    fullinertia=("7.4842E-05 5.861E-05 3.9364E-05 1.6252E-07 -8.0516E-09 -3.5931E-08"),
)
GEAR = "250"  # 150 gave ~4.7 N/finger: cans slipped out mid-carry (2/6)
CTRL = "-20 20"


def gripper_joint(side: str, i: int) -> str:
    return f"{side}_hand_gripper_{i}_joint"


def build_hand(side: str) -> list[ET.Element]:
    els = []
    for group, extra in (("1", {"contype": "0", "conaffinity": "0", "density": "0"}), (None, {})):
        g = ET.Element("geom", dict(pos=BASE_POS, type="mesh", mesh="Dex1_base_link", rgba="0.25 0.25 0.25 1", **extra))
        if group:
            g.set("group", group)
        els.append(g)
    for i in (1, 2):
        f = FINGER[i]
        b = ET.Element("body", dict(name=f"{side}_dex1_finger_{i}_link", pos=BASE_POS))
        ET.SubElement(b, "inertial", dict(pos=f["ipos"], mass=MASS, fullinertia=f["fullinertia"]))
        ET.SubElement(
            b,
            "joint",
            dict(
                name=gripper_joint(side, i),
                type="slide",
                pos="0 0 0",
                axis=f["axis"],
                range=JOINT_RANGE,
                actuatorfrcrange="-20 20",
            ),
        )
        ET.SubElement(
            b,
            "geom",
            dict(
                type="mesh", contype="0", conaffinity="0", group="1", density="0", rgba="0.3 0.3 0.3 1", mesh=f["mesh"]
            ),
        )
        ET.SubElement(b, "geom", dict(type="mesh", rgba="0.3 0.3 0.3 1", mesh=f["col"], friction="1.2 0.02 0.001"))
        els.append(b)
    return els


def main() -> None:
    for m in (
        "Dex1_base_link.STL",
        "Dex1_finger_link_1.STL",
        "Dex1_finger_link_2.STL",
        "dex1_col_1.stl",
        "dex1_col_2.stl",
    ):
        shutil.copy(f"{SRC_DIR}/meshes/{m}", f"{G1}/meshes/{m}")

    tree = ET.parse(f"{G1}/g1_29dof_with_hand.xml")
    root = tree.getroot()

    asset = root.find("asset")
    for m in ("Dex1_base_link", "Dex1_finger_link_1", "Dex1_finger_link_2", "dex1_col_1", "dex1_col_2"):
        ext = "STL" if m.startswith("Dex1") else "stl"
        ET.SubElement(asset, "mesh", dict(name=m, file=f"{m}.{ext}"))

    # --- worldbody surgery -------------------------------------------------
    n_removed = 0
    for parent in root.iter("body"):
        name = parent.get("name", "")
        if not name.endswith("wrist_yaw_link"):
            continue
        side = "left" if name.startswith("left") else "right"
        for geom in list(parent.findall("geom")):
            if "hand_palm" in (geom.get("mesh") or ""):
                parent.remove(geom)
                n_removed += 1
        for child in list(parent.findall("body")):
            if "_hand_" in (child.get("name") or ""):
                parent.remove(child)
                n_removed += 1
        for el in build_hand(side):
            parent.append(el)
    assert n_removed == 10, f"expected 10 dex3 removals, got {n_removed}"

    # --- actuators: replace 7 dex3 motors per side IN PLACE ---------------
    act = root.find("actuator")
    motors = list(act)
    out = []
    inserted = {"left": False, "right": False}
    for m in motors:
        j = m.get("joint", "")
        if "_hand_" in j:
            side = "left" if j.startswith("left") else "right"
            if not inserted[side]:
                for i in (1, 2):
                    out.append(
                        ET.Element(
                            "motor",
                            dict(
                                name=f"{side}_hand_gripper_{i}", joint=gripper_joint(side, i), gear=GEAR, ctrlrange=CTRL
                            ),
                        )
                    )
                inserted[side] = True
            continue
        out.append(m)
    for m in list(act):
        act.remove(m)
    for m in out:
        act.append(m)
    assert inserted["left"] and inserted["right"]

    # --- sensors: same in-place replacement --------------------------------
    for sensor in root.findall("sensor"):
        kids = list(sensor)
        out, ins = [], {"left": False, "right": False}
        for s in kids:
            j = s.get("joint", "")
            if "_hand_" in j:
                side = "left" if j.startswith("left") else "right"
                if not ins[side]:
                    for i in (1, 2):
                        for tag, suf in (("jointpos", "pos"), ("jointvel", "vel"), ("jointactuatorfrc", "torque")):
                            if any(k.tag == tag for k in kids):
                                out.append(
                                    ET.Element(
                                        tag, dict(name=f"{side}_hand_gripper_{i}_{suf}", joint=gripper_joint(side, i))
                                    )
                                )
                    ins[side] = True
                continue
            out.append(s)
        for s in kids:
            sensor.remove(s)
        for s in out:
            sensor.append(s)

    ET.indent(tree, space="  ")
    out_xml = f"{G1}/g1_29dof_with_dex1.xml"
    tree.write(out_xml)

    scene = open(f"{G1}/scene_43dof.xml").read().replace("g1_29dof_with_hand.xml", "g1_29dof_with_dex1.xml")
    with open(f"{G1}/scene_31dof.xml", "w") as f:
        f.write(scene)
    print(f"wrote {out_xml} and scene_31dof.xml")


if __name__ == "__main__":
    main()
