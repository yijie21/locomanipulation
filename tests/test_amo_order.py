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


if __name__ == "__main__":
    unittest.main()
