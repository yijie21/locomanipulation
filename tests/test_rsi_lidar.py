"""Vision RSI D4: the simulated MID-360 (locomani.tasks.scene.lidar) -- its directions cover the datasheet's field of
view anew each frame, rays end where the geometry is, and the robot's own body is filtered out."""

import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from locomani.tasks.scene import lidar as L  # noqa: E402

# a torso 1.2 m up with an arm-like box beside the sensor (the robot's own body), a floor and a wall 2 m ahead
SCENE = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1"/>
    <geom name="wall" type="box" pos="2.0 0 1.5" size="0.05 5 1.5"/>
    <geom name="hull" type="box" pos="-2.0 0 1.5" size="0.05 5 1.5" group="3"/>
    <body name="pelvis" pos="0 0 0.8">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="torso_link" pos="0 0 0.0">
        <geom name="head" type="sphere" pos="0 0 0.43" size="0.08"/>
        <body name="arm_link">
          <geom name="arm" type="box" pos="0 0.3 0.43" size="0.05 0.05 0.05"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class Directions(unittest.TestCase):
    def test_the_field_of_view_and_a_new_set_each_frame(self):
        cfg = L.Mid360Config(rays_per_frame=5000)
        a, b = L.directions(5000, 0, cfg), L.directions(5000, 5000, cfg)
        np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-12)
        elev = np.degrees(np.arcsin(a[:, 2]))
        self.assertGreaterEqual(elev.min(), -7.0 - 1e-9)
        self.assertLessEqual(elev.max(), 52.0 + 1e-9)
        self.assertEqual(len(np.unique(np.round(np.concatenate([a, b]), 9), axis=0)), 10000)  # nothing repeats
        az = np.degrees(np.arctan2(a[:, 1], a[:, 0]))
        counts = np.histogram(az, bins=12, range=(-180, 180))[0]
        self.assertLess(counts.max() / counts.min(), 1.1)  # even around


class Scan(unittest.TestCase):
    def setUp(self):
        self.m = mujoco.MjModel.from_xml_string(SCENE)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_forward(self.m, self.d)
        self.lidar = L.Mid360(self.m, L.Mid360Config(rays_per_frame=4000, noise_m=0.0))

    def test_returns_lie_on_the_floor_and_the_wall_not_on_the_robot(self):
        f = self.lidar.scan(self.d)
        names = {self.m.geom(int(g)).name for g in f["geoms"]}
        self.assertTrue({"floor", "wall"} <= names)
        self.assertNotIn("arm", names)
        self.assertNotIn("head", names)  # the sensor looks out through the head shell it sits in
        on_floor = f["geoms"] == self.m.geom("floor").id
        np.testing.assert_allclose(f["points"][on_floor, 2], 0.0, atol=1e-9)
        on_wall = f["geoms"] == self.m.geom("wall").id
        np.testing.assert_allclose(f["points"][on_wall, 0], 1.95, atol=1e-9)
        np.testing.assert_allclose(f["origin"], [0.0002835, 0.00003, 0.8 + 0.428434], atol=1e-9)
        self.assertEqual(f["rays"], 4000)
        self.assertLess(len(f["points"]), 4000)  # rays upward and into the arm return nothing

    def test_the_arm_shadows_the_rays_behind_it(self):
        f = self.lidar.scan(self.d)
        rel = f["points"] - f["origin"]
        toward_arm = (
            (np.abs(np.degrees(np.arctan2(rel[:, 1], rel[:, 0])) - 90) < 3) & (rel[:, 2] < 0.05) & (rel[:, 2] > -0.05)
        )
        self.assertEqual(int(toward_arm.sum()), 0)  # behind the arm's box (0.3 m to the side) nothing is seen


class Groups(unittest.TestCase):
    def test_a_geomgroup_mask_hides_the_collision_group(self):
        m = mujoco.MjModel.from_xml_string(SCENE)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        hull = m.geom("hull").id
        every = L.Mid360(m, L.Mid360Config(rays_per_frame=4000, noise_m=0.0)).scan(d)
        visual = L.Mid360(m, L.Mid360Config(rays_per_frame=4000, noise_m=0.0, geomgroup=(1, 1, 1, 0, 0, 0))).scan(d)
        self.assertIn(hull, set(every["geoms"].tolist()))
        self.assertNotIn(hull, set(visual["geoms"].tolist()))


if __name__ == "__main__":
    unittest.main()
