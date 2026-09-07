"""Diagnostic initialization must preserve the settled body and object."""
import sys
from pathlib import Path
import unittest
import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/teleop_free'))
from privileged_pub import EpisodeControlServer


class AlignmentTests(unittest.TestCase):
    def test_translation_preserves_posture_velocity_object_and_clock(self):
        m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body name="pelvis" pos="0 0 1"><freejoint/><geom size=".1" mass="1"/>
          <body pos="0 0 .1"><joint/><geom size=".05" mass=".1"/></body></body>
          <body pos="1 1 1"><freejoint/><geom size=".1" mass="1"/></body>
        </worldbody></mujoco>''')
        d=mujoco.MjData(m);d.qpos[7]=.4;d.qvel[:]=.05;d.time=12.
        s=EpisodeControlServer.__new__(EpisodeControlServer)
        s.model,s.data,s._mj=m,d,mujoco
        q=d.qpos.copy();v=d.qvel.copy()
        s._align_robot_xy([2.,3.])
        np.testing.assert_allclose(d.qpos[:2],[2,3])
        np.testing.assert_array_equal(d.qpos[2:],q[2:])
        np.testing.assert_array_equal(d.qvel,v)
        self.assertEqual(d.time,12.)
        before=d.qpos.copy()
        with self.assertRaises(ValueError):s._align_robot_xy([float('nan'),0])
        np.testing.assert_array_equal(d.qpos,before)


if __name__=='__main__':unittest.main()
