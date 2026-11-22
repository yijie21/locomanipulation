"""selfmodel_via_20261120: forward kinematics of the acting arm's 24 collision spheres in a scene's own robot model.

    fk = ArmFK(<scene dir>)                       # run from the repo root
    X, jaw = fk.spheres("right", pelvis_pos, pelvis_quat, joints, "meta", fingers)

arm_spheres.py (labels of a recorded dataset) and via_comp.py (a reference plan's sweep before it is run) share it.
The spheres are cuRobo's for SIMPLE's G1 with Dex3 hands (SIMPLE_ROOT/data/robots/g1/curobo/spheres/g1_29dof.yml): the
right arm's 9 (shoulder pitch, roll, yaw, elbow, wrist pitch), the palm's 3 and the fingers' 12. The left arm uses its own
links with the same spheres mirrored (y -> -y in the link frame): the links' frames are mirror images (0.01 mm at the zero
pose), the file's own left spheres are placed differently. The palm link is not a body of the simulated robot: it sits on
the wrist-yaw link at the offset the URDF gives it. joints: 29 values in ScaleBFM's metadata order ("meta", what a recorded
state S[:29] holds) or in MuJoCo's ("mj", what a reference pose holds); fingers: the acting hand's 7 joints.
"""

import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import yaml

ARM = ("shoulder_pitch_link", "shoulder_roll_link", "shoulder_yaw_link", "elbow_link", "wrist_pitch_link")
HAND = ("hand_palm_link", "hand_thumb_0_link", "hand_thumb_1_link", "hand_thumb_2_link", "hand_middle_0_link",
        "hand_middle_1_link", "hand_index_0_link", "hand_index_1_link")  # fmt: skip
FINGERS = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")


class ArmFK:
    def __init__(self, scene: Path):
        sys.path.insert(0, str(Path.cwd() / "src"))
        from locomani.backends.scalebfm import runtime as RT
        from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

        simple = Path(os.environ.get("SIMPLE_ROOT", "/workspace/code/locomanipulation/third_party/SIMPLE"))
        self.off = np.array([float(v) for v in os.environ.get("REACH_HAND_OFF", "0.0415,0,0").split(",")])  # jaw centre
        sph = yaml.safe_load(open(simple / "data/robots/g1/curobo/spheres/g1_29dof.yml"))
        sph = sph.get("collision_spheres", sph)
        palm = {}
        for j in ET.parse(simple / "data/robots/g1/g1_29dof_with_dex3.urdf").getroot().iter("joint"):
            if j.get("name", "").endswith("_hand_palm_joint"):
                assert j.find("origin").get("rpy", "0 0 0").split() == ["0", "0", "0"], (
                    "the palm link is turned on the wrist"
                )
                palm[j.get("name")[: -len("_hand_palm_joint")]] = np.array(
                    [float(v) for v in j.find("origin").get("xyz").split()]
                )
        self.links = [n for n in ARM + HAND for _ in sph[f"right_{n}"]]
        self.local = np.array([s["center"] for n in ARM + HAND for s in sph[f"right_{n}"]], float)
        self.radius = np.array([s["radius"] for n in ARM + HAND for s in sph[f"right_{n}"]], np.float32)
        self.m = m = mujoco.MjModel.from_xml_path(str(Path(scene) / "scene_locomani.xml"))
        self.d = mujoco.MjData(m)
        assert m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE, "the robot's first joint is its free root"
        adr = lambda names: [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in names]  # noqa: E731
        body = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        self.qadr = dict(meta=adr(json.load(open(RT.META))["joint_names"]), mj=adr(MUJOCO_JOINT_NAMES))
        self.q_hand = {s: adr([f"{s}_hand_{f}_joint" for f in FINGERS]) for s in ("right", "left")}
        self.body, self.mirror, self.shift = {}, {}, {}
        for s in ("right", "left"):
            ids, shift = [], []
            for n in self.links:
                b = body(f"{s}_wrist_yaw_link" if n == "hand_palm_link" else f"{s}_{n}")
                assert b >= 0, f"{s}_{n} is not a body of {scene}"
                ids.append(b)
                shift.append(palm[s] if n == "hand_palm_link" else np.zeros(3))
            self.body[s], self.shift[s] = np.array(ids), np.array(shift)
            self.mirror[s] = np.array([1.0, 1.0 if s == "right" else -1.0, 1.0])
        self.wrist = {s: body(f"{s}_wrist_yaw_link") for s in ("right", "left")}

    def spheres(self, side: str, pos, quat, joints, order: str, fingers):
        """The side's 24 sphere centres and its jaw centre (world) with the pelvis at pos / quat and the joints set."""
        d = self.d
        d.qpos[:] = 0
        d.qpos[:3], d.qpos[3:7] = pos, quat
        d.qpos[self.qadr[order]] = joints
        d.qpos[self.q_hand[side]] = fingers
        mujoco.mj_kinematics(self.m, d)
        b = self.body[side]
        Rm = d.xmat[b].reshape(-1, 3, 3)
        X = d.xpos[b] + np.einsum("kij,kj->ki", Rm, self.shift[side] + self.local * self.mirror[side])
        w = self.wrist[side]
        return X, d.xpos[w] + d.xmat[w].reshape(3, 3) @ self.off
