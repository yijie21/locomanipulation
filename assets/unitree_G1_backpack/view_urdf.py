#!/usr/bin/env python3
"""Interactively scrub every joint of a robot URDF in MuJoCo's viewer.

This adds a position actuator for every hinge/slide joint (with its
ctrlrange set to the joint's URDF limits) so each joint shows up as a
slider under the viewer's "Control" tab. Gravity is disabled so the
robot holds still while you drag sliders through their full range,
making it easy to visually check limits, mesh collisions, and gripper
travel.

Usage:
    python view_urdf.py [path/to/robot.urdf]

With no argument, the URDF path is read from this branch's robot.yaml
(the ETHER_ROBOT_URDF export), so the same script works unmodified on
every robot branch in this repo.

In the viewer:
    - Open the right-hand panel and select the "Control" tab.
    - Each slider corresponds to one joint (named after the joint) and
      is bounded by that joint's URDF <limit>.
    - Drag a slider to move the joint through its full range of motion.
"""

import os
import re
import sys

import mujoco
import mujoco.viewer

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def default_urdf_path() -> str:
    """Resolve this branch's robot URDF from robot.yaml's ETHER_ROBOT_URDF."""
    robot_yaml = os.path.join(REPO_ROOT, "robot.yaml")
    with open(robot_yaml, "r") as f:
        text = f.read()
    match = re.search(r"^\s*ETHER_ROBOT_URDF:\s*(\S+)", text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not find ETHER_ROBOT_URDF in {robot_yaml}")
    return os.path.join(REPO_ROOT, match.group(1))


def build_scrubbable_model(urdf_path: str):
    """Load a URDF and attach a position actuator to every real joint."""
    with open(urdf_path, "r") as f:
        text = f.read()

    # The URDF sets <compiler meshdir="meshes"> for the mujoco/mjcf
    # toolchain, but its mesh filenames are already "meshes/xxx.STL"
    # (relative to the URDF's own directory). Keeping meshdir would make
    # MuJoCo look in meshes/meshes/... instead, so drop it and resolve the
    # existing relative paths from the URDF's directory instead.
    text = re.sub(r'meshdir="meshes"\s*', "", text, count=1)

    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    original_cwd = os.getcwd()
    os.chdir(urdf_dir)
    try:
        spec = mujoco.MjSpec.from_string(text)

        for joint in spec.joints:
            if joint.type not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                continue
            # Add damping/armature so light links (e.g. the waist and
            # finger links) don't destabilize the position servo below.
            joint.damping = [2.0, 2.0, 2.0]
            joint.armature = 0.01

            actuator = spec.add_actuator()
            actuator.name = f"{joint.name}_act"
            actuator.target = joint.name
            actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
            actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            # Position servo: gain*(ctrl - qpos) - kv*qvel, via the affine
            # bias term (standard MuJoCo "position" actuator setup).
            kp = 40.0
            kv = 4.0
            actuator.gainprm[0] = kp
            actuator.biasprm[1] = -kp
            actuator.biasprm[2] = -kv
            if joint.limited:
                actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
                actuator.ctrlrange = joint.range
            else:
                actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE

        model = spec.compile()
    finally:
        os.chdir(original_cwd)

    # Disable gravity so the robot holds its pose while you scrub sliders,
    # instead of sagging/falling under the position servos.
    model.opt.gravity[:] = 0.0
    # Implicit integration is more forgiving of stiff position servos.
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.timestep = 0.005

    return model


def main():
    urdf_path = sys.argv[1] if len(sys.argv) > 1 else default_urdf_path()
    model = build_scrubbable_model(urdf_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    # Start actuator targets at the current (zero) pose.
    data.ctrl[:] = data.qpos[: model.nu]

    print(f"Loaded {urdf_path}")
    print(f"  {model.nq} DOFs, {model.nu} actuated joints, {model.nbody} bodies")
    print("Open the 'Control' tab in the viewer to scrub each joint by name.")

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
