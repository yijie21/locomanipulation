"""Offline check that the grasp loop keeps the left arm hanging under arm_route curobo (simple_policy.loop_prior).

From the end of the route in a recorded round (the run's state trace and manifest), the loop's SupportedMotion is
stepped for 3 s while the right-hand target slides to the round's pre-grasp and enters to its grip, with the torso lean
the loop applies -- once with the loop's prior as built for arm_route curobo (left arm held by its joints, prior blended
in) and once with the lifted-ready settings (left hand held at its world point). Prints the left elbow and the left
arm's largest joint change at 1, 2 and 3 s, and the right-hand reference error.

    PYTHONPATH=src third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
        scripts/experiments/curobo_route_20261009/left_hang_check.py outputs/runs/<curobo run> [round]
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from locomani.primitives.arm_ik import ArmIK  # noqa: E402
from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES  # noqa: E402
from locomani.primitives.supported_motion import SupportedMotion  # noqa: E402
from locomani.tasks.bottle.ready_grasp_policy import OFFSET  # noqa: E402
from locomani.tasks.bottle.simple_config import SimpleGraspConfig  # noqa: E402
from locomani.tasks.bottle.simple_policy import lean_angle, loop_prior  # noqa: E402
from locomani.tasks.bottle.whole_body_config import WholeBodyConfig  # noqa: E402


def main() -> int:
    run = Path(sys.argv[1])
    rnd = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    xml = str(REPO / "outputs/scene/sim_lowcom/scene_locomani.xml")
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    row = [json.loads(line) for line in (run / "collect/manifest_00.jsonl").read_text().splitlines()][rnd]
    s = row["execution"]["metrics"]["simple"]
    z = np.load(sorted(glob.glob(str(run / "collect/state_trace/*.npz")))[rnd])
    phases = {p["phase"]: p["seq"] for p in s["phases"]}
    k = int(np.searchsorted(z["seq"], phases["route"] + int(s["route"]["seconds"] * 50) + 2))
    d.qpos[:] = z["qpos"][k]
    mujoco.mj_kinematics(m, d)
    joints = d.qpos[[m.jnt_qposadr[m.joint(n).id] for n in MUJOCO_JOINT_NAMES]].copy()
    state = dict(pelvis=dict(pos=d.qpos[0:3].tolist(), quat=d.qpos[3:7].tolist()), joints=joints.tolist())
    palms = [m.body(f"{side}_wrist_yaw_link").id for side in ("left", "right")]
    tools = np.array([d.xpos[b] + d.xmat[b].reshape(3, 3) @ OFFSET for b in palms])
    rotations = np.array([d.xmat[b].reshape(3, 3) for b in palms])
    grip, pregrasp = np.array(s["plan"]["grip"]), np.array(s["plan"]["pregrasp"])
    ik, cfg, simple = ArmIK(xml), WholeBodyConfig(), SimpleGraspConfig()
    for name, hang in (("arm_route curobo", True), ("lifted-ready settings", False)):
        posture, weights, hands = loop_prior(joints, joints.copy(), np.full(29, 0.04), hang)
        motion = SupportedMotion(
            ik, state, cfg, joints, d.qpos[3:7].copy(), tool_offset=OFFSET, hand_weights=hands,
            posture_reference=posture, posture_weights=weights,
        )  # fmt: skip
        if hang:
            motion.blend_posture(1.0)
        heading = motion.body.heading
        targets, out = tools.copy(), []
        for i in range(150):
            if i < 50:
                targets[1] = tools[1] + (i / 50) * (pregrasp - tools[1])
            else:
                targets[1] = pregrasp + min(1.0, (i - 50) / 100) * (grip - pregrasp)
            lean = lean_angle(state["pelvis"]["pos"], grip, simple.lean_max_rad, heading[:, 0])
            torso = heading @ Rotation.from_euler("y", lean).as_matrix()
            pose = motion.step(targets, rotations, tools, 0.02, torso_rotation=torso, measured_rotations=rotations)
            if i in (49, 99, 149):
                right_error = np.linalg.norm(motion.body.tools(motion.x)[1] - targets[1])
                out.append(
                    f"{(i + 1) * 0.02:.0f} s: L elbow {pose.joints[18]:+.2f}, left arm max change "
                    f"{np.abs(pose.joints[15:22] - joints[15:22]).max():.2f} rad, right ref error {100 * right_error:.1f} cm"
                )
        print(f"{name} (left elbow starts {joints[18]:+.2f}): " + "; ".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
