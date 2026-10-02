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

The same run also writes the mars model: that tree with the lab robot's
own URDF numbers written in (see "mars" below).

Outputs (into gear_sonic's model_data/g1, which is gitignored — rerun this
script after a fresh clone):
  g1_29dof_with_dex1.xml, scene_31dof.xml, meshes/Dex1_*.STL + dex1_col_*
  g1_29dof_with_dex1_mars.xml, scene_31dof_mars.xml, meshes/mars/
"""

from __future__ import annotations

import copy
import math
import os
import re
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from locomani import paths
from locomani.sim import robot_look

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


def build_stock() -> ET.ElementTree:
    """The dex1 tree; main() writes it and copies the meshes it names."""
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
    return tree


# --- mars: the lab's real robot --------------------------------------------
#
# assets/unitree_G1_with_wrist_camera is the URDF of the G1 the lab runs
# (instance "mars", G1-EDU-U2 with Dex1-1 grippers, a pan/tilt head and a D405
# camera on each gripper; henry-zhao, cb3ae2f, 2026-10-01). Its 29 body joints
# have the stock model's names, order, axes and ranges, and its legs the stock
# model's kinematics. What differs, all read from the URDF below rather than
# typed in:
#   - waist rev 1.0: torso 10 mm lower (roll axis 44 mm above waist yaw, stock
#     35; pitch axis on it, stock 19 mm higher); shoulders 10 mm higher
#   - 5010 wrists: wrist yaw 5 mm further out, pitch/yaw effort 13.4 N m
#     (stock 5)
#   - effort limits: ankles and waist roll/pitch 35 N m (confirmed as the
#     firmware limit, 2026-09-30; stock 50), hip roll 139 (stock 88)
#   - masses: 35.02 kg overall (stock 35.87): torso lighter, the head's
#     pan/tilt bracket 1.73 kg, wrist pitch +0.2 kg, each D405 mount 0.1 kg
#   - the D405 mounts collide (a ring round the Dex1 base and the camera's box)
#   - cameras: the ego camera is the D455 on the pan/tilt bracket, the wrist
#     cameras the D405s
# The mars model is the stock dex1 tree with those numbers written in: same
# bodies, joints, actuator order, joint classes (armature, damping,
# frictionloss), sensors, finger geoms and foot boxes, so the SONIC sim bridge
# drives it unchanged. apply_mars() does the writing on any tree holding the
# stock robot, so a packaged scene (prepare_logistics_scene.py) gets the same
# robot as one composed by scene_lib.

MARS_URDF = f"{REPO}/assets/unitree_G1_with_wrist_camera/urdf/g1_comp_new_hip_dex1_1.urdf"
MARS_XML = f"{G1}/g1_29dof_with_dex1_mars.xml"
MARS_SCENE = f"{G1}/scene_31dof_mars.xml"
MARS_MESHES = f"{G1}/meshes/mars"  # the URDF's meshes are copied here
# The head's pan/tilt servo is not driven (URDF effort 0): the sim holds it,
# panned straight ahead and tilted 0.0822 rad (4.7 deg) down, where the D455's
# optical axis is 52 deg below the torso's horizontal -- the stock ego camera's
# pitch, so the ego view keeps the reach it was calibrated on (user,
# 2026-10-02: "about what the old camera saw").
MARS_HELD_JOINTS = {"head_pan_joint": 0.0, "head_tilt_joint": 0.0822}
# The ego camera is the D455 on the pan/tilt bracket (user, 2026-10-02; the
# D435 is fixed in the face, d435_head_link). The frame is used as it is: the
# URDF gives no imager offset for it.
MARS_EGO_LINK = "d455_head_link"
MARS_WRIST_LINKS = {"left": "d405_left_wrist_link", "right": "d405_right_wrist_link"}
# No intrinsics ship with the URDF (its calibration/ has only livox -> D435
# extrinsics), so the fields of view are Intel's datasheet values with the
# 640x480 (4:3) stream keeping the sensor's vertical field: D455 colour 90 x 65
# deg, D405 87 x 58 deg. Replace them with the robot's calibration once it has
# one.
MARS_EGO_FOVY = 65.0
MARS_WRIST_FOVY = 58.0
# The colours the generator writes, so both renderers sort each mesh by its MJCF
# colour into robot_look's shell or dark set (Isaac: mean >= 0.6 is shell;
# Unreal: grey >= 0.6 silver, < 0.4 dark, anything else a third, plain look).
SHELL_RGBA = "0.7 0.7 0.7 1"
DARK_RGBA = "0.2 0.2 0.2 1"
# MuJoCo camera axes in a ROS camera_link frame (x forward, y left, z up): the
# camera looks along -z, x is image right, y is image up.
_CAMERA_IN_LINK = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _sub_once(pattern: str, repl: str, text: str) -> str:
    out, n = re.subn(pattern, repl, text)
    assert n == 1, f"{pattern!r} matched {n} times in {MARS_URDF}"
    return out


def _rpy_of(rot: np.ndarray) -> tuple[float, float, float]:
    """URDF roll-pitch-yaw of a rotation (the inverse of _rpy)."""
    pitch = math.asin(max(-1.0, min(1.0, -rot[2, 0])))
    return math.atan2(rot[2, 1], rot[2, 2]), pitch, math.atan2(rot[1, 0], rot[0, 0])


def _hold(text: str, joint: str, angle: float) -> str:
    """The URDF with `joint` turned into a fixed joint at `angle` (its origin rotated about its axis)."""
    m = re.search(rf'<joint name="{joint}" type="revolute">.*?</joint>', text, re.S)
    assert m, f"{joint} is not a revolute joint of {MARS_URDF}"
    block = m[0]
    origin = re.search(r'<origin xyz="([^"]*)" rpy="([^"]*)"\s*/>', block)
    axis = np.array([float(v) for v in re.search(r'<axis xyz="([^"]*)"', block)[1].split()])
    turn = np.empty(9)
    mujoco.mju_quat2Mat(turn, np.r_[math.cos(angle / 2), math.sin(angle / 2) * axis / np.linalg.norm(axis)])
    rpy = _rpy_of(_rpy(origin[2]) @ turn.reshape(3, 3))
    held = block.replace('type="revolute"', 'type="fixed"', 1).replace(
        origin[0], f'<origin xyz="{origin[1]}" rpy="{_fmt(rpy)}"/>', 1
    )
    return text.replace(block, held, 1)


def mars_urdf_model(fuse: bool) -> mujoco.MjModel:
    """The mars URDF compiled by MuJoCo: pelvis floating, head servo held (MARS_HELD_JOINTS).

    fuse=True folds every fixed link into its moving parent (head and bracket
    into torso_link; the Dex1 base and D405 mount into wrist_yaw_link), which
    is the body set the sim model has; fuse=False keeps every link, for frames
    such as the cameras'.
    """
    text = open(MARS_URDF).read()
    # The ROS footprint frame becomes the world and the pelvis floats; left
    # fixed, the compiler would fuse the pelvis into the world.
    text = _sub_once(r'<link name="base_footprint"\s*/>', '<link name="world"/>', text)
    text = _sub_once(r'<parent link="base_footprint"\s*/>', '<parent link="world"/>', text)
    text = _sub_once(r'(<joint name="base_footprint_to_pelvis") type="fixed"', r'\1 type="floating"', text)
    for joint, angle in MARS_HELD_JOINTS.items():
        text = _hold(text, joint, angle)
    urdf_dir = os.path.dirname(MARS_URDF)
    text = _sub_once(
        r'<compiler meshdir="meshes"', f'<compiler fusestatic="{str(fuse).lower()}" meshdir="{urdf_dir}"', text
    )
    return mujoco.MjModel.from_xml_string(text)


def mars_link(body: str) -> str:
    """URDF link name of a sim model body (the sim names the Dex1 fingers its own way)."""
    m = re.fullmatch(r"(left|right)_dex1_finger_(\d)_link", body)
    return f"{m[1]}_dex1_finger_link_{m[2]}" if m else body


def mars_joint(joint: str) -> str:
    m = re.fullmatch(r"(left|right)_hand_gripper_(\d)_joint", joint)
    return f"{m[1]}_dex1_finger_joint_{m[2]}" if m else joint


def _fmt(values) -> str:
    return " ".join(f"{float(v):.9g}" for v in values)


def _quat(rot: np.ndarray) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(rot).flatten())
    return q


def _rpy(rpy: str) -> np.ndarray:
    r, p, y = (float(v) for v in rpy.split())
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _set_pose(el: ET.Element, pos: np.ndarray, rot: np.ndarray) -> None:
    for key in ("pos", "quat", "euler", "axisangle", "xyaxes", "zaxis"):
        el.attrib.pop(key, None)
    el.set("pos", _fmt(pos))
    if not np.allclose(rot, np.eye(3), atol=1e-12):
        el.set("quat", _fmt(_quat(rot)))


def mesh_rgba(filename: str, urdf_rgba: str) -> str:
    """The colour a mars mesh is drawn in: robot_look's dark parts dark, the rest by the URDF colour's brightness."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    if stem in robot_look.DARK_MESHES:
        return DARK_RGBA
    return SHELL_RGBA if np.mean([float(v) for v in urdf_rgba.split()[:3]]) >= 0.6 else DARK_RGBA


def _mars_geoms(
    urdf: ET.Element, links: list[str], frame, materials: dict[str, str]
) -> tuple[list[ET.Element], list[ET.Element], set[str]]:
    """Visual and collision geoms of `links`, posed in the owning body's frame by `frame(link)`."""
    by_name = {link.get("name"): link for link in urdf.findall("link")}
    visual, collision, meshes = [], [], set()
    for name in links:
        pos, rot = frame(name)
        link = by_name[name]
        link_rgba = None  # the link's first visual colour, which its collision geoms take
        for tag, out in (("visual", visual), ("collision", collision)):
            for el in link.findall(tag):
                origin = el.find("origin")
                o_pos = np.array(
                    [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
                )
                o_rot = _rpy(origin.get("rpy", "0 0 0")) if origin is not None else np.eye(3)
                shape = el.find("geometry")[0]
                g = ET.Element("geom")
                if shape.tag == "mesh":
                    g.set("type", "mesh")
                    g.set("mesh", mars_mesh_name(shape.get("filename")))
                    meshes.add(shape.get("filename"))
                elif shape.tag == "cylinder":
                    g.set("type", "cylinder")
                    g.set("size", _fmt([float(shape.get("radius")), float(shape.get("length")) / 2]))
                elif shape.tag == "sphere":
                    g.set("type", "sphere")
                    g.set("size", _fmt([float(shape.get("radius"))]))
                elif shape.tag == "box":
                    g.set("type", "box")
                    g.set("size", _fmt([float(v) / 2 for v in shape.get("size").split()]))
                else:
                    raise ValueError(f"{name}: unsupported URDF geometry {shape.tag}")
                if tag == "visual":
                    g.attrib.update(contype="0", conaffinity="0", group="1", density="0")
                    mat = el.find("material")
                    color = mat.find("color") if mat is not None else None
                    rgba = color.get("rgba") if color is not None else materials.get(mat.get("name"), SHELL_RGBA)
                    g.set("rgba", mesh_rgba(shape.get("filename") or "", rgba) if shape.tag == "mesh" else rgba)
                    link_rgba = link_rgba or g.get("rgba")
                else:
                    # Collision geoms are group 0, which MuJoCo's cameras draw too (the stock robot's are as well):
                    # coloured like the link's visual, so a MuJoCo picture shows the robot_look colour, not a
                    # light-grey collision hull over a dark part.
                    g.set("rgba", link_rgba or SHELL_RGBA)
                _set_pose(g, pos + rot @ o_pos, rot @ o_rot)
                out.append(g)
    return visual, collision, meshes


def mars_mesh_name(filename: str) -> str:
    """The MuJoCo mesh name of a URDF mesh file.

    A file byte-identical to the stock robot's file of the same name (30 of the
    URDF's 41: legs, shoulders, elbows, the rev 1.0 waist and torso, the Dex1)
    keeps the stock mesh name, so code that picks geoms by mesh name -- the
    bottle guard's grasp envelope (`Dex1_base_link`), the Isaac stage's visual
    split -- reads mars as it reads stock. The rest (pelvis, 5010 wrists, head
    bracket, D405 and its mount) are named mars_<file stem>.
    """
    base = os.path.basename(filename)
    stock = f"{G1}/meshes/{base}"
    src = f"{os.path.dirname(MARS_URDF)}/{filename}"
    same = os.path.isfile(stock) and open(stock, "rb").read() == open(src, "rb").read()
    return os.path.splitext(base)[0] if same else f"mars_{os.path.splitext(base)[0]}"


def mars_mesh_scales() -> dict[str, str]:
    """URDF mesh file -> MuJoCo mesh scale (the D405 and its mount are in millimetres)."""
    out = {}
    for mesh in ET.parse(MARS_URDF).getroot().iter("mesh"):
        out.setdefault(mesh.get("filename"), mesh.get("scale") or "1 1 1")
    return out


def apply_mars(tree: ET.ElementTree, mesh_dir: str) -> ET.ElementTree:
    """`tree` with its robot made mars: the URDF's kinematics, inertia, meshes, effort limits and cameras.

    `tree` holds the stock dex1 robot (bodies pelvis down, its actuators); it
    may hold a scene round it, which is left alone. The URDF's mesh assets are
    written as `mesh_dir`/<file>: "mars" for the robot model (meshdir is
    model_data/g1/meshes), an absolute path in a packaged scene. Cameras: the
    ego camera head_camera on torso_link, and a <side>_wrist camera on each
    wrist_yaw_link, re-posed if the tree has one (a packaged scene) and added if
    not (the robot model, where scene_lib.add_wrist_cameras then keeps it).
    """
    tree = copy.deepcopy(tree)
    root = tree.getroot()
    urdf = ET.parse(MARS_URDF).getroot()
    materials = {m.get("name"): m.find("color").get("rgba") for m in urdf.findall("material")}
    joints = {j.get("name"): j for j in urdf.findall("joint")}

    fused = mars_urdf_model(fuse=True)
    full = mars_urdf_model(fuse=False)
    full_data = mujoco.MjData(full)
    mujoco.mj_forward(full, full_data)

    def world(link: str) -> tuple[np.ndarray, np.ndarray]:
        i = full.body(link).id
        return full_data.xpos[i].copy(), full_data.xmat[i].reshape(3, 3).copy()

    def frame_in(body: str):
        b_pos, b_rot = world(body)

        def frame(other: str) -> tuple[np.ndarray, np.ndarray]:
            p, r = world(other)
            return b_rot.T @ (p - b_pos), b_rot.T @ r

        return frame

    # Fixed descendants of each link (held joints count as fixed), which the
    # sim model carries as geoms of the moving body they hang from.
    children: dict[str, list[str]] = {}
    for j in joints.values():
        if j.get("type") == "fixed" or j.get("name") in MARS_HELD_JOINTS:
            children.setdefault(j.find("parent").get("link"), []).append(j.find("child").get("link"))

    def carried(link: str) -> list[str]:
        out = [link]
        for child in children.get(link, []):
            out += carried(child)
        return out

    pelvis = next(b for b in root.iter("body") if b.get("name") == "pelvis")
    body_els = {b.get("name"): b for b in pelvis.iter("body")}
    parent_of = {child: parent for parent in pelvis.iter("body") for child in parent.findall("body")}
    meshes: set[str] = set()
    for name, body in body_els.items():
        link = mars_link(name)
        i = fused.body(link).id
        want_parent = mujoco.mj_id2name(fused, mujoco.mjtObj.mjOBJ_BODY, fused.body_parentid[i])
        if name != "pelvis":  # the root keeps the scene's standing pose
            assert mars_link(parent_of[body].get("name")) == want_parent, f"{name}: parent differs from the URDF"
            rot = np.empty(9)
            mujoco.mju_quat2Mat(rot, fused.body_quat[i])
            _set_pose(body, fused.body_pos[i], rot.reshape(3, 3))
        inertial = body.find("inertial")
        inertial.attrib.clear()
        inertial.attrib.update(
            pos=_fmt(fused.body_ipos[i]),
            quat=_fmt(fused.body_iquat[i]),
            mass=_fmt([fused.body_mass[i]]),
            diaginertia=_fmt(fused.body_inertia[i]),
        )
        if "_dex1_finger_" in name:
            continue  # the fingers keep their geoms: same meshes, and the pads' friction and hardening are tuned

        visual, collision, used = _mars_geoms(urdf, carried(link), frame_in(link), materials)
        meshes |= used
        old = [g for g in body.findall("geom")]
        if name.endswith("ankle_roll_link"):
            # The URDF's four 5 mm corner spheres span the same sole as the sim's box; SONIC runs on the box.
            collision = [g for g in old if g.get("type") == "box"]
            assert len(collision) == 1, f"{name}: expected the stock foot box"
        if name == "pelvis":
            # The URDF gives the pelvis no collision; the sim's pelvis always collided, so it keeps the stock
            # contour mesh. The visual mesh's hull instead touched the hip roll links in 706 of 3000 random hip
            # poses (stock contour: 120), 2 of them near standing (outputs/mars_model_check_20261002).
            collision = [g for g in old if g.get("group") != "1"]
            assert [g.get("mesh") for g in collision] == ["pelvis_contour_link"], f"{name}: expected the stock contour"
        for g in old:
            body.remove(g)
        first_body = next((k for k, el in enumerate(body) if el.tag in ("body", "camera", "site")), len(body))
        for k, g in enumerate(visual + collision):
            body.insert(first_body + k, g)

    # Joint limits and the actuators' torque limits from the URDF.
    for j in pelvis.iter("joint"):
        if j.get("type") == "free":
            continue
        u = joints[mars_joint(j.get("name"))]
        axis = np.array([float(v) for v in u.find("axis").get("xyz").split()])
        # A scene MjSpec has written leaves MuJoCo's default axis (0 0 1) out.
        assert np.allclose(axis, [float(v) for v in j.get("axis", "0 0 1").split()]), f"{j.get('name')}: axis differs"
        limit = u.find("limit")
        j.set("range", f"{limit.get('lower')} {limit.get('upper')}")
        j.set("actuatorfrcrange", f"-{limit.get('effort')} {limit.get('effort')}")
    for motor in root.find("actuator"):
        if "_hand_" in motor.get("joint"):
            continue  # gripper ctrl is in the deploy's hand units, scaled by gear (see GEAR)
        effort = joints[motor.get("joint")].find("limit").get("effort")
        motor.set("ctrlrange", f"-{effort} {effort}")

    # The cameras at the URDF's camera frames, and the torso IMU site.
    torso = body_els["torso_link"]
    cams = [("torso_link", "head_camera", MARS_EGO_LINK, MARS_EGO_FOVY)] + [
        (f"{side}_wrist_yaw_link", f"{side}_wrist", link, MARS_WRIST_FOVY) for side, link in MARS_WRIST_LINKS.items()
    ]
    for body, cam_name, link, fovy in cams:
        cam = body_els[body].find(f"camera[@name='{cam_name}']")
        if cam is None:
            cam = ET.SubElement(body_els[body], "camera", dict(name=cam_name))
        c_pos, c_rot = frame_in(body)(link)
        _set_pose(cam, c_pos, c_rot @ _CAMERA_IN_LINK)
        cam.set("fovy", _fmt([fovy]))
    imu_pos, _ = frame_in("torso_link")("imu_in_torso")
    torso.find("site[@name='imu']").set("pos", _fmt(imu_pos))

    # Mesh assets: the URDF's files, and none of the stock robot's that nothing references any more.
    asset = root.find("asset")
    scales = mars_mesh_scales()
    have = {m.get("name") for m in asset.findall("mesh")}
    stock_dir = os.path.dirname(mesh_dir)  # model_data/g1/meshes, absolute in a packaged scene, "" in the model
    for filename in sorted(meshes):
        name, base = mars_mesh_name(filename), os.path.basename(filename)
        if name in have:
            continue  # a stock mesh the tree already carries
        if name.startswith("mars_"):
            attrs = dict(name=name, file=f"{mesh_dir}/{base}")
        else:
            attrs = dict(name=name, file=f"{stock_dir}/{base}" if stock_dir else base)
        if scales[filename] != "1 1 1":
            attrs["scale"] = scales[filename]
        ET.SubElement(asset, "mesh", attrs)
        have.add(name)
    used_meshes = {g.get("mesh") for g in root.iter("geom") if g.get("mesh")}
    for m in list(asset.findall("mesh")):
        if m.get("name") not in used_meshes:
            asset.remove(m)
    ET.indent(tree, space="  ")
    return tree


def build_mars(stock: ET.ElementTree) -> ET.ElementTree:
    """The mars robot model: the stock dex1 tree with apply_mars()."""
    tree = apply_mars(stock, os.path.basename(MARS_MESHES))
    tree.getroot().set("model", "g1_29dof_with_dex1_mars")
    return tree


def check_mars(xml_path: str = MARS_XML, seed: int = 0) -> dict:
    """A written mars robot (model or scene) against the URDF: forward kinematics of every body, mass, torque limits.

    Raises on any mismatch; returns the worst errors for the log.
    """
    sim = mujoco.MjModel.from_xml_path(xml_path)
    urdf = mars_urdf_model(fuse=True)
    robot = [b for b in range(1, sim.nbody) if sim.body_rootid[b] == sim.body("pelvis").id]
    rng = np.random.default_rng(seed)
    worst_mm = 0.0
    for trial in range(20):
        ds, du = mujoco.MjData(sim), mujoco.MjData(urdf)
        for j in range(sim.njnt):
            if sim.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            lo, hi = sim.jnt_range[j]
            q = 0.0 if trial == 0 else rng.uniform(lo, hi)
            ds.qpos[sim.jnt_qposadr[j]] = q
            du.qpos[urdf.jnt_qposadr[urdf.joint(mars_joint(sim.joint(j).name)).id]] = q
        mujoco.mj_forward(sim, ds)
        mujoco.mj_forward(urdf, du)
        ps, pu = sim.body("pelvis").id, urdf.body("pelvis").id
        rs, ru = ds.xmat[ps].reshape(3, 3), du.xmat[pu].reshape(3, 3)
        for b in robot:
            u = urdf.body(mars_link(sim.body(b).name)).id
            err = np.linalg.norm(rs.T @ (ds.xpos[b] - ds.xpos[ps]) - ru.T @ (du.xpos[u] - du.xpos[pu]))
            worst_mm = max(worst_mm, err * 1000)
    mass = float(sim.body_subtreemass[sim.body("pelvis").id]), float(urdf.body_subtreemass[0])
    assert worst_mm < 1e-3, f"mars model FK is {worst_mm:.4f} mm off the URDF"
    assert abs(mass[0] - mass[1]) < 1e-6, f"mars mass {mass[0]} != URDF {mass[1]}"
    joints = {j.get("name"): j for j in ET.parse(MARS_URDF).getroot().findall("joint")}
    for a in range(sim.nu):
        name = sim.joint(sim.actuator_trnid[a, 0]).name
        if "_hand_" not in name:
            effort = float(joints[name].find("limit").get("effort"))
            assert np.allclose(sim.actuator_ctrlrange[a], [-effort, effort]), f"{name}: ctrlrange != URDF effort"
    return {"fk_worst_mm": worst_mm, "mass_kg": mass[0]}


def main() -> None:
    for m in (
        "Dex1_base_link.STL",
        "Dex1_finger_link_1.STL",
        "Dex1_finger_link_2.STL",
        "dex1_col_1.stl",
        "dex1_col_2.stl",
    ):
        shutil.copy(f"{SRC_DIR}/meshes/{m}", f"{G1}/meshes/{m}")
    stock = build_stock()
    out_xml = f"{G1}/g1_29dof_with_dex1.xml"
    stock.write(out_xml)
    scene = open(f"{G1}/scene_43dof.xml").read().replace("g1_29dof_with_hand.xml", "g1_29dof_with_dex1.xml")
    with open(f"{G1}/scene_31dof.xml", "w") as f:
        f.write(scene)
    print(f"wrote {out_xml} and scene_31dof.xml")

    os.makedirs(MARS_MESHES, exist_ok=True)
    src = os.path.join(os.path.dirname(MARS_URDF), "meshes")
    for m in os.listdir(src):
        shutil.copy(f"{src}/{m}", f"{MARS_MESHES}/{m}")
    build_mars(stock).write(MARS_XML)
    scene = open(f"{G1}/scene_43dof.xml").read().replace("g1_29dof_with_hand.xml", os.path.basename(MARS_XML))
    with open(MARS_SCENE, "w") as f:
        f.write(scene)
    print(f"wrote {MARS_XML} and {os.path.basename(MARS_SCENE)}: {check_mars()}")


if __name__ == "__main__":
    main()
