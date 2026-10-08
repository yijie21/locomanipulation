import mujoco
import numpy as np
import pytest
from geometry_fixture import DIRECTORY

from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES
from locomani.tasks.crate.bimanual_carry import OFFSET
from locomani.tasks.crate.bimanual_ik import BimanualIK
from locomani.tasks.crate.visual_config import CrateVisualConfig

SCENE = str(DIRECTORY / "scene_locomani.xml")
TABLE_TOP = 0.62  # dining table: box at z 0.6, half height 0.02


def test_collision_guard_setting_is_validated():
    assert CrateVisualConfig().collision_guard == "off"
    assert CrateVisualConfig(collision_guard="mink").collision_guard == "mink"
    with pytest.raises(ValueError, match="collision_guard"):
        CrateVisualConfig(collision_guard="curobo")
    with pytest.raises(ValueError, match="collision_guard_clearance"):
        CrateVisualConfig(collision_guard_clearance=-0.001)


def _guard_and_state():
    pytest.importorskip("mink")
    from geometry_fixture import scene_ik

    from locomani.tasks.crate.collision_guard import HandCollisionGuard

    solver = BimanualIK(scene_ik(), waist_pitch_max=0.52, waist_dofs="all")
    guard = HandCollisionGuard(
        SCENE, tool_offset=OFFSET, joint_limits=(solver.lo, solver.hi), joint_speed=0.7, clearance=0.003
    )
    joints = np.zeros(29)
    joints[[15, 22]], joints[[18, 25]] = -0.35, 0.8  # shoulders forward, elbows bent: the hands over the table
    # Standing 0.75 m from the table centre puts the hands above its near half.
    state = dict(pelvis=dict(pos=[-0.75, 0.0, 0.793], quat=[1.0, 0.0, 0.0, 0.0]), joints=joints, objects={})
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    d.qpos[:3], d.qpos[3:7] = state["pelvis"]["pos"], state["pelvis"]["quat"]
    for name, value in zip(MUJOCO_JOINT_NAMES, joints):
        d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]] = value
    mujoco.mj_kinematics(m, d)
    palms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_wrist_yaw_link") for s in ("left", "right")]
    rotations = np.array([d.xmat[b].reshape(3, 3) for b in palms])
    tools = np.array([d.xpos[b] + R @ OFFSET for b, R in zip(palms, rotations)])
    return guard, state, tools, rotations


def test_guard_holds_the_hands_off_an_obstacle_the_targets_pass_through():
    guard, state, tools, rotations = _guard_and_state()
    assert np.all(tools[:, 2] > TABLE_TOP + 0.05) and np.all(np.abs(tools[:, 0]) < 0.45)  # above the table top
    targets = tools.copy()
    targets[:, 2] = TABLE_TOP - 0.10  # 10 cm into the table
    for _ in range(100):
        grips, attitudes, clearance = guard.filter(targets, rotations, state, 0.02)
    assert clearance >= 0.003 - 0.001
    assert np.all(grips[:, 2] > TABLE_TOP)
    assert np.all(grips[:, 2] < tools[:, 2] - 0.05)  # the hands still came down as far as the table lets them


def test_guard_passes_a_clear_target_through():
    guard, state, tools, rotations = _guard_and_state()
    targets = tools + [0.05, 0.0, -0.05]  # 5 cm forward and down, still well above the table
    for _ in range(100):
        grips, attitudes, _ = guard.filter(targets, rotations, state, 0.02)
    np.testing.assert_allclose(grips, targets, atol=0.005)
    np.testing.assert_allclose(attitudes, rotations, atol=0.05)
    guard.reset()
    assert not guard.started
