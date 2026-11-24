"""TODO 49.4: the shared camera pieces were generalised (perception/head.py: cam_points(depth, fovy), intrinsics(wh, fovy),
HeadKin(model, qadr, cam); bench/simple/body.py: PlaceBelief.frame() / cam). SIMPLE's results must not move by a bit, and LocoManiLab's
belief gives what the old bed's LmlBelief gave. The references below are the old code's own expressions, written out."""

import types
import unittest

import mujoco
import numpy as np

from locomani.bench.locomanilab.body import LmlBelief
from locomani.bench.simple import body
from locomani.perception import head

XML = """
<mujoco><worldbody>
  <body name="pelvis" pos="0 0 0.8"><freejoint/><geom size="0.1"/>
    <body name="torso" pos="0 0 0.2"><joint name="j1" axis="0 0 1"/><geom size="0.05"/>
      <body name="neck" pos="0.05 0 0.3"><joint name="j2" axis="0 1 0"/><geom size="0.05"/>
        <camera name="{cam}" pos="0.1 0 0.1" xyaxes="0 -1 0 0.5 0 0.87"/>
      </body></body></body>
</worldbody></mujoco>
"""


def rig(cam: str):
    m = mujoco.MjModel.from_xml_string(XML.format(cam=cam))
    d = mujoco.MjData(m)
    d.qpos[:7] = [0.3, -0.2, 0.8, 0.92, 0.0, 0.0, 0.39]
    d.qpos[7:9] = [0.2, -0.3]
    mujoco.mj_forward(m, d)
    return m, d, np.array([7, 8])


def old_scene(bel, rgb, depth, fovy, cam_points):
    """The old PlaceBelief.scene / LmlBelief.scene, expression for expression."""
    kin = head.HeadKin(bel.sim.m, bel.sim.qadr, bel.cam)
    rel_p, rel_R = kin(bel.sim.d.qpos[bel.sim.qadr])
    ps, Rs = bel._pelvis_sim()
    c = cam_points(depth, fovy).reshape(-1, 3)
    return (
        rgb,
        bel.from_sim(ps + (rel_p + c @ rel_R.T) @ Rs.T).reshape(*depth.shape, 3),
        (depth > 0.2) & (depth < 5),
        bel.from_sim(ps + Rs @ rel_p),
    )


def inputs(h, w, seed=1):
    rng = np.random.default_rng(seed)
    depth = rng.uniform(0.1, 6.0, (h, w)).astype(np.float32)
    depth[rng.random((h, w)) < 0.05] = np.nan
    depth[rng.random((h, w)) < 0.02] = np.inf
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8), depth


class Simple(unittest.TestCase):
    def test_defaults_are_simples_calibration(self):
        _, depth = inputs(36, 64)
        a = head.cam_points(depth)
        b = head.cam_points(depth, head.HEAD_CAM_FOVY)
        np.testing.assert_array_equal(a, b)
        H, W = head.HEAD_CAM_WH[1], head.HEAD_CAM_WH[0]
        f = 0.5 * H / np.tan(np.radians(head.HEAD_CAM_FOVY) / 2)
        np.testing.assert_array_equal(
            head.intrinsics(), np.array([[f, 0.0, W / 2 - 0.5], [0.0, f, H / 2 - 0.5], [0.0, 0.0, 1.0]])
        )

    def test_scene_is_the_old_scene_bit_for_bit(self):
        m, d, qadr = rig(head.HEAD_CAM)
        H, W = head.HEAD_CAM_WH[1], head.HEAD_CAM_WH[0]
        rgb, depth = inputs(H, W)
        sim = types.SimpleNamespace(m=m, d=d, qadr=qadr, eyes=lambda: (rgb, depth))
        bel = body.Belief(sim, None, odo=False, root_odo=False)
        bel.start()
        got = bel.scene()
        want = old_scene(bel, rgb, depth, head.HEAD_CAM_FOVY, head.cam_points)
        for g, w in zip(got, want[:3]):
            np.testing.assert_array_equal(g, w)
        np.testing.assert_array_equal(bel.cam_b, want[3])


class LocoManiLab(unittest.TestCase):
    def test_the_belief_reads_its_own_camera_and_the_frames_fov(self):
        m, d, qadr = rig("head_camera")
        rgb, depth = inputs(24, 32)
        for fovy in (70.0, 52.5):
            sim = types.SimpleNamespace(
                m=m, d=d, qadr=qadr, camera_image=lambda f=fovy: dict(rgb=rgb, depth=depth, fovy=f)
            )
            bel = LmlBelief(sim, None, odo=False, root_odo=False)
            bel.start()
            got = bel.scene()
            want = old_scene(bel, rgb, np.asarray(depth, float), fovy, head.cam_points)
            for g, w in zip(got, want[:3]):
                np.testing.assert_array_equal(g, w)
            np.testing.assert_array_equal(bel.cam_b, want[3])

    def test_old_lml_cam_points_agree_where_the_depth_is_finite(self):
        """The old LocoManiLab cam_points did not zero a non-finite depth; the shared one does (its points there are masked by `ok`)."""
        _, depth = inputs(24, 32)
        depth = np.asarray(depth, float)
        H, W = depth.shape
        f = 0.5 * H / np.tan(np.radians(60.0) / 2)
        v, u = np.mgrid[0:H, 0:W]
        old = np.stack([(u - W / 2 + 0.5) * depth / f, -(v - H / 2 + 0.5) * depth / f, -depth], axis=2)
        new = head.cam_points(depth, 60.0)
        fin = np.isfinite(depth)
        np.testing.assert_array_equal(old[fin], new[fin])
        self.assertTrue(np.all(new[~fin] == 0) or np.all(new[~fin][..., :2] == 0))
        self.assertFalse(np.any(((depth > 0.2) & (depth < 5))[~fin]))  # non-finite depth is never an `ok` pixel

    def test_frame_intrinsics_follow_the_frames_fov(self):
        K = head.intrinsics((320, 240), 60.0)
        self.assertAlmostEqual(K[0, 0], 0.5 * 240 / np.tan(np.radians(30.0)), places=9)
        self.assertEqual((K[0, 2], K[1, 2]), (159.5, 119.5))


if __name__ == "__main__":
    unittest.main()
