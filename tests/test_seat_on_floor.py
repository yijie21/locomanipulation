"""Seating the robot at a reset must use the geom's real extent, not its bounding sphere.

The bug this pins down (fixed 2026-09-11, ledger bottle.md 24) was silent and
repo-wide: `_seat_on_floor` used `geom_rbound`, the bounding-SPHERE radius, as
each geom's downward extent. For the deciding foot mesh that is 0.1291 m against
a true 0.0213, so every reset in every task line seated the robot 10.8 cm too
high and dropped it. Nothing reported an error -- the robot simply landed, and
the stand-drift budget caught the landing transient as if the policy had walked.
"""

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from locomani import paths

REPO = Path(__file__).resolve().parents[1]

# One body, one geom of each primitive kind plus a flat mesh, all axis-aligned so
# the expected lowest point is analytic.
MJCF = """
<mujoco>
  <asset>
    <mesh name="plate" vertex="-0.1 -0.05 -0.02  0.1 -0.05 -0.02  0.1 0.05 -0.02
                               -0.1 0.05 -0.02  -0.1 -0.05 0.02  0.1 -0.05 0.02
                                0.1 0.05 0.02  -0.1 0.05 0.02"/>
  </asset>
  <worldbody>
    <body name="root" pos="0 0 1">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.05" pos="0 0 0"/>
      <geom name="brick" type="box" size="0.1 0.05 0.02" pos="0 0.3 0"/>
      <geom name="pill" type="capsule" size="0.03 0.1" pos="0 0.6 0"/>
      <geom name="can" type="cylinder" size="0.04 0.06" pos="0 0.9 0"/>
      <geom name="foot" type="mesh" mesh="plate" pos="0 1.2 0"/>
    </body>
  </worldbody>
</mujoco>
"""


def _pub(model, data):
    """A stand-in carrying only what the two seating methods touch."""
    import mujoco

    from locomani.sim.privileged_pub import EpisodeControlServer

    obj = SimpleNamespace(
        _mj=mujoco,
        model=model,
        data=data,
        root_id=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "root"),
        SEAT_CLEARANCE_M=EpisodeControlServer.SEAT_CLEARANCE_M,
    )
    obj._geom_lowest_z = EpisodeControlServer._geom_lowest_z.__get__(obj)
    obj._seat_on_floor = EpisodeControlServer._seat_on_floor.__get__(obj)
    return obj


class GeomExtentTests(unittest.TestCase):
    def setUp(self):
        try:
            import mujoco
        except ImportError:  # pragma: no cover
            self.skipTest("mujoco not installed")
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(MJCF)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.pub = _pub(self.model, self.data)

    def _lowest(self, name):
        g = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_GEOM, name)
        return self.pub._geom_lowest_z(g)

    def test_every_primitive_reports_its_real_lowest_point(self):
        # body at z = 1, every geom centred at its body-frame z = 0
        for name, expected in (
            ("ball", 1.0 - 0.05),
            ("brick", 1.0 - 0.02),
            ("pill", 1.0 - (0.1 + 0.03)),  # half-length + radius
            ("can", 1.0 - 0.06),
            ("foot", 1.0 - 0.02),  # the mesh's own half-thickness
        ):
            self.assertAlmostEqual(self._lowest(name), expected, places=6, msg=name)

    def test_the_flat_mesh_is_not_read_as_its_bounding_sphere(self):
        g = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_GEOM, "foot")
        rbound_estimate = float(self.data.geom_xpos[g][2] - self.model.geom_rbound[g])
        self.assertLess(
            rbound_estimate,
            self._lowest("foot") - 0.05,
            "the bounding sphere must be the pessimistic one -- otherwise this test proves nothing",
        )

    def test_seating_leaves_the_clearance_not_a_drop(self):
        self.pub._seat_on_floor()
        self.mj.mj_forward(self.model, self.data)
        lowest = min(
            self.pub._geom_lowest_z(g)
            for g in range(self.model.ngeom)
            if self.model.body_rootid[self.model.geom_bodyid[g]] == self.pub.root_id
        )
        self.assertAlmostEqual(lowest, self.pub.SEAT_CLEARANCE_M, places=6)


class RealSceneTests(unittest.TestCase):
    """The numbers from the ledger, on the actual composed scene."""

    def setUp(self):
        try:
            import mujoco
        except ImportError:  # pragma: no cover
            self.skipTest("mujoco not installed")
        if not Path(paths.SCENE_XML).is_file():
            self.skipTest(f"{paths.SCENE_XML} not built")
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(paths.SCENE_XML)
        self.data = mujoco.MjData(self.model)

    def test_the_nominal_stance_is_seated_at_standing_height(self):
        angles = np.zeros(29)
        angles[[0, 6]] = -0.1  # hip pitch
        angles[[3, 9]] = 0.3  # knee
        angles[[4, 10]] = -0.2  # ankle pitch
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[7 : 7 + 29] = angles
        pub = _pub(self.model, self.data)
        pub.root_id = self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_BODY, "pelvis")
        pub._seat_on_floor()
        seated = float(self.data.qpos[2])
        # 0.892 was the buggy value, measured identically in all 62 recorded
        # rounds; the stance actually stands at ~0.785.
        self.assertLess(seated, 0.80, f"seated at {seated:.4f} -- the rbound bug is back")
        self.assertGreater(seated, 0.77, f"seated at {seated:.4f} -- too low, the robot would start in contact")


if __name__ == "__main__":
    unittest.main()
