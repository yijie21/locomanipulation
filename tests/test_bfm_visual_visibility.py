import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/teleop_free/scalebfm'))
from audit_visual_case import audit_case
from visual_sensor import BottleProxySensor


class VisibilityRegressionTests(unittest.TestCase):
    def test_audit_rejects_initially_hidden_bottle_even_if_close_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            result = dict(seed=1, stratum='high_near', scene_directory='unused', ok=True,
                          physics_warnings={}, metrics=dict(wrist_step_max_cm=0, wrist_step_max_deg=0,
                          torso_roll_max_deg=0, torso_roll_above15_s=0, waist_yaw_max_deg=0))
            commands = [dict(t=t, phase='close', angles=[0]*7) for t in (1.68, 1.70)]
            observations = [dict(t=0., pixels=[0,0], fractions=[0,0]),
                            dict(t=1.6, pixels=[100,0], fractions=[1,0])]
            for name, data in [('result', result), ('commands', commands), ('observations', observations)]:
                (p / f'{name}.json').write_text(json.dumps(data))
            np.savez(p / 'trace.npz', qpos=np.array([[0,0,.76,1,0,0,0]]*2))
            self.assertFalse(audit_case(p)['accepted'])

    def test_recovery_first_frame_must_have_its_own_evidence(self):
        from visual_visibility import initial_visibility
        observations = [dict(t=t, pixels=[100,0], fractions=[1,0]) for t in (0.,1.6)]
        gates = initial_visibility(observations, first_record_time=6.12)
        self.assertFalse(all(gates.values()))
        evidence = dict(t=6.12,pixels=[100,0],fractions=[1,0])
        self.assertTrue(all(initial_visibility(observations, evidence, 6.12).values()))

    def test_geometry_uses_camera_that_passes_visibility_gate(self):
        sensor = BottleProxySensor.__new__(BottleProxySensor)
        sensor.quantization = .005
        sensor.cameras = [0, 1]
        sensor.robot_geoms = np.array([0])
        sensor.options = SimpleNamespace(geomgroup=np.ones(6))
        sensor.approach_edge = None
        sensor.sim = SimpleNamespace(
            m=SimpleNamespace(geom_group=np.zeros(1, dtype=int)),
            d=SimpleNamespace(cam_xmat=np.array([np.eye(3).ravel()]*2),
                              cam_xpos=np.array([[.001,0,0],[0,0,0]]), xmat=np.array([np.eye(3).ravel()])),
            bid={'pelvis':0}, crate=lambda: (np.array([.5,0,.7]), None),
            smeta={'surfaces':{'A':dict(xy=[0,0],flat_half_extent=[.3,.3],top=.62)}})
        sensor._mask = Mock(side_effect=[np.ones((10,10),bool), np.ones((10,100),bool),
                                       np.ones((5,8),bool), np.ones((5,8),bool)])
        observed = sensor.capture(0.)
        self.assertAlmostEqual(observed.bottle_position[0], .5)


if __name__ == '__main__':
    unittest.main()
