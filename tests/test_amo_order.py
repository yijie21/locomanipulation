"""AmoMotor's boundary to a sim in ScaleBFM's joint order (the self-model harness), without a policy or a GPU."""

import json
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from locomani.backends.amo.motor import AmoMotor
from locomani.backends.amo.runtime import in_order
from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

ROOT = Path(__file__).resolve().parents[1]


def scalebfm_names() -> list[str]:
    p = ROOT / "assets" / "scalebfm" / "model_22200_tensorrt_metadata.json"
    text = (
        p.read_text()
        if p.exists()
        else subprocess.check_output(["git", "show", f"HEAD:{p.relative_to(ROOT)}"], cwd=ROOT)
    )
    return json.loads(text)["joint_names"]


class InOrder(unittest.TestCase):
    def test_reorders_every_per_joint_list_by_name(self):
        meta = dict(joint_names=list(MUJOCO_JOINT_NAMES), stiffness=list(range(29)), damping=[2 * i for i in range(29)],
                    torque_limit=[3 * i for i in range(29)], default_dof_pos=[0.1 * i for i in range(29)])  # fmt: skip
        self.assertIs(in_order(meta, "mujoco"), meta)
        out = in_order(meta, "scalebfm")
        names = scalebfm_names()
        self.assertEqual(out["joint_names"], names)
        for i, n in enumerate(names):
            j = MUJOCO_JOINT_NAMES.index(n)
            self.assertEqual(out["stiffness"][i], j)
            self.assertEqual(out["damping"][i], 2 * j)
            self.assertAlmostEqual(out["default_dof_pos"][i], 0.1 * j)


class Boundary(unittest.TestCase):
    def motor(self, names):
        m = AmoMotor.__new__(AmoMotor)
        m.to_mj = np.array([names.index(n) for n in MUJOCO_JOINT_NAMES])
        m.from_mj = np.array([MUJOCO_JOINT_NAMES.index(n) for n in names])
        stepped = {}
        sim_q = np.array([names.index(n) for n in names], float)  # the sim's i-th joint reads i
        m.sim = SimpleNamespace(
            state=lambda: dict(dof_pos=sim_q.copy(), dof_vel=-sim_q.copy(), root_quat=np.array([1.0, 0, 0, 0])),
            step=lambda t: stepped.setdefault("t", np.asarray(t)),
            d=SimpleNamespace(qpos=np.r_[0.1, 0.2, 0.7, 1.0, 0.0, 0.0, 0.0]),
        )
        m.root_est = None
        return m, stepped

    def test_state_and_targets_round_trip_through_mujoco_order(self):
        names = scalebfm_names()
        m, stepped = self.motor(names)
        st = m._mj_state()
        # the j-th MuJoCo joint's value is that joint's index in the sim order
        np.testing.assert_array_equal(st["dof_pos"], [names.index(n) for n in MUJOCO_JOINT_NAMES])
        np.testing.assert_array_equal(st["dof_vel"], [-names.index(n) for n in MUJOCO_JOINT_NAMES])
        targets_mj = np.arange(29, dtype=float)  # target j for MuJoCo joint j
        m._step(targets_mj)
        np.testing.assert_array_equal(stepped["t"], [MUJOCO_JOINT_NAMES.index(n) for n in names])

    def test_mujoco_order_sim_is_the_identity_and_the_estimate_falls_back_to_truth(self):
        m, stepped = self.motor(list(MUJOCO_JOINT_NAMES))
        np.testing.assert_array_equal(m._mj_state()["dof_pos"], np.arange(29))
        np.testing.assert_allclose(m._pelvis_estimate(), [0.1, 0.2, 0.7])
        m.root_est = np.array([1.0, 2.0, 0.6])
        np.testing.assert_allclose(m._pelvis_estimate(), [1.0, 2.0, 0.6])

    def test_servo_measures_jaw_source_when_set(self):
        """The arm servo measures the jaw from `jaw_source` when a caller sets one (as Motor._servo), else the sim."""
        from locomani.backends.scalebfm.motor import MotorConfig

        m, _ = self.motor(list(MUJOCO_JOINT_NAMES))
        m.hand_override, m.servo_on, m.servo_target, m.servo = {}, {"right": True}, {}, {}
        m.cfg, m.amo = MotorConfig(), SimpleNamespace(hand_lead_s=0.0)
        ref_jaw = np.array([0.5, -0.2, 0.9])
        m.kp = lambda pos, quat, joints: {
            "right_wrist_yaw_link": (ref_jaw - np.array([0.1515, 0.0, 0.0]), np.array([1.0, 0, 0, 0]))
        }
        ref = SimpleNamespace(pos=None, quat=None, joints=np.zeros(29), velocity=np.zeros(29))
        m.jaw_centre = lambda side: (ref_jaw.copy(), np.eye(3))  # the sim: the jaw is exactly where it should be
        lead_sim = m._jaw_target("right", ref)[0] - ref_jaw
        m.servo = {}
        m.jaw_source = lambda side: (ref_jaw - np.array([0.0, 0.0, 0.05]), np.eye(3))  # the estimate: 5 cm low
        lead_src = m._jaw_target("right", ref)[0] - ref_jaw
        np.testing.assert_allclose(lead_sim, 0.0, atol=1e-12)
        self.assertGreater(lead_src[2], 0.0)  # it pushes up toward the reference, so it read jaw_source


if __name__ == "__main__":
    unittest.main()


class HeightCommand(unittest.TestCase):
    """The base-height command of a standing tick: reference + integral, capped."""

    def test_without_a_cap_it_is_the_reference_plus_the_integral_as_before(self):
        from locomani.backends.amo.motor import AmoConfig, height_command

        cfg = AmoConfig()
        cmd, h = height_command(0.60, 0.65, 0.02, cfg, 0.02)
        self.assertAlmostEqual(h, 0.02 + cfg.ki_height * 0.02 * (0.60 - 0.65))
        self.assertAlmostEqual(cmd, 0.60 + h)

    def test_the_cap_holds_and_the_integral_does_not_wind_up_under_it(self):
        from dataclasses import replace

        from locomani.backends.amo.motor import AmoConfig, height_command

        cfg, h = replace(AmoConfig(), height_max=0.75), 0.0
        for _ in range(500):  # 10 s standing: asked for 0.79, the robot stays at 0.78
            cmd, h = height_command(0.79, 0.78, h, cfg, 0.02)
            self.assertLessEqual(cmd, 0.75)
        self.assertEqual(h, 0.0)  # nothing accumulated, so a later crouch is not delayed
        cmd, h = height_command(0.60, 0.70, h, cfg, 0.02)  # the reference crouches: the integral works again, downward
        self.assertLess(h, 0.0)
        self.assertAlmostEqual(cmd, 0.60 + h)
