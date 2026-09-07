import unittest,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/teleop_free'))
from measured_grasp import bounded_rotation

class RotationHandoverTests(unittest.TestCase):
    def test_large_terminal_rotation_cannot_jump_at_handover(self):
        measured=Rotation.from_euler('zyx',[1.2,.2,-.3]).as_matrix()
        target=measured@Rotation.from_rotvec([.2,-.1,.05]).as_matrix()
        command=bounded_rotation(measured,target,.04)
        self.assertAlmostEqual(Rotation.from_matrix(measured.T@command).magnitude(),.04)
        self.assertLess(Rotation.from_matrix(command.T@target).magnitude(),
                        Rotation.from_matrix(measured.T@target).magnitude())
    def test_nearby_target_is_reached_without_overshoot(self):
        target=Rotation.from_rotvec([0,.01,0]).as_matrix()
        np.testing.assert_allclose(bounded_rotation(np.eye(3),target,.04),target,atol=1e-12)
