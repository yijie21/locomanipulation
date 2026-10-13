"""The ScaleBFM runtime's finger PD takes its clamp from the backend config, and the SONIC line's law fits in it.

The SONIC-vs-ScaleBFM comparison (scripts/experiments/wbc_compare_20261012/) runs the SONIC line's finger law on
both controllers: kp 1.5 / kd 0.1 / ctrl clamp 20 (grasp_cache.py KP / KD / FORCE_MAX, transcribed from the deploy's
hand command and base_sim.compute_hand_torques). Until 2026-10-12 `Sim.grip_clip` was a hard-coded 4.0 that callers
poked per round; the law is now `clip(grip_kp (target - q) - grip_kd dq, +-grip_clip)` with all three from the config.
"""

import json
import unittest
from pathlib import Path

import mujoco
import numpy as np
from geometry_fixture import DIRECTORY

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig

SONIC_LAW = dict(grip_kp=1.5, grip_kd=0.1, grip_clip=20.0)
JAW_OPEN, JAW_CLOSED = 0.0245, -0.02  # agent_server._hand_poses (dex1)


def sim_with(**law):
    meta = json.loads(Path(R.META).read_text())
    sim = R.Sim(meta, backend=ScaleBfmBackendConfig(scene_dir=str(DIRECTORY), **law))
    sim.reset([-1.5, 0.0], 0.0, 0.0)
    sim.decim = 1  # one physics substep per step, so the ctrl left behind was computed from the state we read
    return sim


class GripLawTests(unittest.TestCase):
    def test_default_clip_is_the_value_the_runtime_hard_coded(self):
        self.assertEqual(ScaleBfmBackendConfig().grip_clip, 4.0)
        self.assertEqual(sim_with().grip_clip, 4.0)

    def test_step_applies_the_configured_law_to_the_four_finger_actuators(self):
        sim = sim_with(**SONIC_LAW)
        sim.grip_target[:] = JAW_CLOSED
        q, dq = sim.d.qpos[sim.gq].copy(), sim.d.qvel[sim.gv].copy()
        self.assertTrue(np.allclose(q, JAW_OPEN))
        sim.step(sim.default)
        expected = np.clip(1.5 * (JAW_CLOSED - q) - 0.1 * dq, -20.0, 20.0)
        self.assertTrue(np.allclose(sim.d.ctrl[sim.gact], expected), (sim.d.ctrl[sim.gact], expected))
        # full open -> closed error under kp 1.5 is -0.06675 ctrl = -16.7 N through gear 250, inside the +-20 N joint
        # limit (actuatorfrcrange clamps qfrc_actuator; d.actuator_force is the pre-gear scalar)
        self.assertTrue(np.allclose(sim.d.qfrc_actuator[sim.gv], np.clip(250.0 * expected, -20.0, 20.0)))
        self.assertAlmostEqual(float(sim.d.qfrc_actuator[sim.gv][2]), -16.6875, places=3)

    def test_the_config_clamp_is_what_clamps(self):
        sim = sim_with(grip_kp=1.5, grip_kd=0.1, grip_clip=0.01)
        sim.grip_target[:] = JAW_CLOSED
        sim.step(sim.default)
        self.assertTrue(np.allclose(sim.d.ctrl[sim.gact], -0.01))

    def test_sim_runs_the_finger_pads_as_built(self):
        """`R.Sim` alone leaves the XML's pad contact parameters; only bottle_visual.scene.make_sim hardens them."""
        sim = sim_with()
        pads = [
            g
            for g in range(sim.m.ngeom)
            if "dex1_finger" in (mujoco.mj_id2name(sim.m, mujoco.mjtObj.mjOBJ_BODY, sim.m.geom_bodyid[g]) or "")
            and sim.m.geom_contype[g]
        ]
        self.assertEqual(len(pads), 4)
        for g in pads:
            self.assertAlmostEqual(float(sim.m.geom_friction[g, 0]), 1.2)
            self.assertAlmostEqual(float(sim.m.geom_solref[g, 0]), 0.004)
            self.assertEqual(int(sim.m.geom_condim[g]), 4)


if __name__ == "__main__":
    unittest.main()
