"""Vision RSI step 2 (L2, docs/handoff/vision-rsi-design-20261023.md): locomani.tasks.scene.perceive -- pixels back to
world points, surface samples of a collision shape, ICP on one visible side, and RenderedView's refresh / sighting /
prior logic over a synthetic camera (no renderer needed)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from locomani.tasks.scene import perceive as P  # noqa: E402
from locomani.tasks.scene.services import pack_mask  # noqa: E402

# an L of two boxes (asymmetric, so its heading is identifiable): 8 x 6 x 10 cm with a 4 x 4 x 4 cm block on one side
SHAPE = [
    dict(type="box", size=np.array([0.04, 0.03, 0.05]), pos=np.zeros(3), quat=np.array([1.0, 0, 0, 0])),
    dict(
        type="box", size=np.array([0.02, 0.02, 0.02]), pos=np.array([0.06, 0.0, -0.03]), quat=np.array([1.0, 0, 0, 0])
    ),
]


def yaw_mat(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def camera_looking_at(target, eye) -> tuple[np.ndarray, np.ndarray]:
    """(pos, mat) of a MuJoCo-convention camera at `eye` looking at `target` (z backwards, y up)."""
    z = np.asarray(eye, float) - np.asarray(target, float)
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 0.0, 1.0], z)
    x /= np.linalg.norm(x)
    return np.asarray(eye, float), np.stack([x, np.cross(z, x), z], axis=1)


def splat(points_world, body_id, pos, mat, K, w=160, h=120) -> P.Frame:
    """A depth / body image of a dense point set (nearest point per pixel), as a renderer would give."""
    depth, body = np.zeros((h, w)), np.full((h, w), -1)
    cam = (np.asarray(points_world) - pos) @ mat
    fx, fy, cx, cy = K
    z = -cam[:, 2]
    u = np.floor(cam[:, 0] / z * fx + cx).astype(int)
    v = np.floor(-cam[:, 1] / z * fy + cy).astype(int)
    ok = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    for ui, vi, zi in sorted(zip(u[ok], v[ok], z[ok]), key=lambda r: -r[2]):  # far first, near overwrites
        depth[vi, ui], body[vi, ui] = zi, body_id
    return P.Frame("cam", depth, body, K, pos, mat)


class Geometry(unittest.TestCase):
    def test_a_pixel_goes_back_to_its_point(self):
        K = P.intrinsics(65.0, 320, 240)
        pos, mat = camera_looking_at([1.0, 0.5, 0.8], [0.0, 0.0, 1.6])
        p = np.array([1.05, 0.48, 0.81])
        cam = (p - pos) @ mat
        u, v = cam[0] / -cam[2] * K[0] + K[2], -cam[1] / -cam[2] * K[1] + K[3]
        depth = np.zeros((240, 320))
        depth[int(v), int(u)] = -cam[2]
        mask = depth > 0
        got = P.back_project(P.Frame("c", depth, mask.astype(int), K, pos, mat), mask)[0]
        pixel = -cam[2] / K[0]  # one pixel's size at that depth
        self.assertLess(float(np.linalg.norm(got - p)), pixel)

    def test_surface_samples_lie_on_the_shape(self):
        pts, nrm = P.surface_points(SHAPE[:1], 4000, np.random.default_rng(0))
        on_face = np.isclose(np.abs(pts) / SHAPE[0]["size"], 1.0, atol=1e-9)
        self.assertTrue(on_face.any(axis=1).all())
        top = on_face[:, 2].mean()  # top and bottom, 2 x 8 x 6 of the box's 376 cm^2
        self.assertAlmostEqual(top, 96 / 376, delta=0.03)
        inside = on_face[:, 2] & ~on_face[:, :2].any(axis=1)  # on a top or bottom face, off its edges
        np.testing.assert_allclose(np.abs(nrm[inside]), np.tile([0.0, 0.0, 1.0], (int(inside.sum()), 1)), atol=1e-9)


class Fit(unittest.TestCase):
    def test_one_visible_side_recovers_the_pose(self):
        rng = np.random.default_rng(1)
        model = P.surface_points(SHAPE, 3000, rng)
        tree = cKDTree(model[0])
        R_true, t_true = yaw_mat(np.radians(37.0)), np.array([1.2, -0.4, 0.82])
        world = model[0] @ R_true.T + t_true
        pos, mat = camera_looking_at(t_true, t_true + [-0.7, -0.2, 0.6])
        K = P.intrinsics(65.0, 320, 240)
        f = splat(P.surface_points(SHAPE, 200000, rng)[0] @ R_true.T + t_true, 7, pos, mat, K, w=320, h=240)
        obs = P.back_project(f, f.body == 7)
        self.assertGreater(len(obs), 100)
        self.assertLess(len(obs), len(world))  # one side only
        R, t, rms = P.fit_from_headings(obs + rng.normal(scale=0.001, size=obs.shape), model, tree, np.eye(3))
        self.assertLess(float(np.linalg.norm(t - t_true)), 0.005)
        err = np.degrees(np.arccos(np.clip((np.trace(R.T @ R_true) - 1) / 2, -1, 1)))
        self.assertLess(err, 3.0)
        self.assertLess(rms, 0.004)


class Rendered(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2)
        self.dense = P.surface_points(SHAPE, 200000, self.rng)[0]
        self.true = {"pos": np.array([1.2, -0.4, 0.82]), "R": yaw_mat(0.4)}
        self.visible = True
        self.motor = SimpleNamespace(t=0.0, k=0, gripping=lambda b: set())
        q = P._mat_quat(self.true["R"])
        world = SimpleNamespace(motor=self.motor, meta={"objects": {"obj": {"body": "obj_b"}}}, free={"obj": 7},
                                occ=None, nav=None, sim=SimpleNamespace(m=None, d=None),
                                object_pose=lambda k: (self.true["pos"].copy(), q.copy()))  # fmt: skip
        self.world = world
        self.cfg = P.RenderConfig(period_s=0.2, min_points=40, prior_bias_m=0.05)

    def frames(self, data):
        if not self.visible:
            return [P.Frame("cam", np.zeros((240, 320)), np.full((240, 320), -1), P.intrinsics(65, 320, 240),
                            np.zeros(3), np.eye(3))]  # fmt: skip
        pos, mat = camera_looking_at(self.true["pos"], self.true["pos"] + [-0.7, -0.2, 0.6])
        pts = self.dense @ self.true["R"].T + self.true["pos"]
        return [splat(pts, 7, pos, mat, P.intrinsics(65.0, 320, 240), w=320, h=240)]

    def view(self):
        with mock.patch.object(P.TruthView, "object_geoms", return_value=SHAPE):
            v = P.RenderedView(self.world, self.cfg, 610001, frames=self.frames)
            v._shape("obj")  # sample the shape while object_geoms is patched
        return v

    def test_never_seen_is_the_prior_and_a_sighting_fits_the_truth(self):
        self.visible = False
        v = self.view()
        p0, _ = v.object_pose("obj")
        off = p0 - self.true["pos"]
        self.assertAlmostEqual(float(np.linalg.norm(off)), 0.05, places=9)
        self.assertEqual(off[2], 0.0)  # the map's memory errs along the floor
        self.visible = True
        self.motor.t, self.motor.k = 0.1, 5
        np.testing.assert_array_equal(v.object_pose("obj")[0], p0)  # younger than period_s: not refreshed
        self.motor.t, self.motor.k = 0.25, 12
        p1, q1 = v.object_pose("obj")
        self.assertLess(float(np.linalg.norm(p1 - self.true["pos"])), 0.005)
        s = v.stats()["obj"]
        self.assertEqual((s["reads"], s["refreshes"], s["seen"], s["ever_seen"]), (3, 2, 1, True))
        self.assertLess(s["err_cm"], 0.5)
        self.assertEqual([r["src"] for r in s["log"]], ["prior", "prior", "seen"])
        self.assertEqual(s["log"][0]["err_cm"], 5.0)  # the prior's 5 cm
        self.assertEqual(s["log"][0]["rot_deg"], 0.0)  # the prior's orientation is the start's
        self.assertLess(s["log"][-1]["rot_deg"], 3.0)
        self.assertIsNone(s["log"][1]["pts"])  # inside period_s: no new picture

    def test_the_maps_memory_is_the_prior_and_a_miss_falls_back_to_the_surface(self):
        self.visible = False
        self.world.meta = {"objects": {"obj": {"body": "obj_b", "surface": "t", "box": {"half_extent": [0.03, 0.03, 0.05]}},
                                       "other": {"body": "o_b", "surface": "t", "box": {"half_extent": [0.03, 0.03, 0.04]}}},
                           "surfaces": {"t": {"xy": [1.0, -0.5], "top": 0.77}}}  # fmt: skip
        self.world.free["other"] = 8
        with tempfile.TemporaryDirectory() as d:
            mem = Path(d) / "memory.json"
            mem.write_text(json.dumps({"610001": {"obj": [1.25, -0.38, 0.83], "other": None}, "610002": {}}))
            self.cfg = P.RenderConfig.of(dict(prior_bias_m=0.05, memory=str(mem)))
            v = self.view()
            np.testing.assert_allclose(v.object_pose("obj")[0], [1.25, -0.38, 0.83])
            np.testing.assert_allclose(v.object_pose("other")[0], [1.0, -0.5, 0.81])  # the surface, the half height up
        self.assertEqual(v.stats()["obj"]["log"][0]["src"], "prior")
        with self.assertRaises(ValueError):
            P.RenderConfig.of(dict(memory="/nonexistent/memory.json"))

    def test_a_perceived_description_is_what_the_plan_reads(self):
        mapped = {"room": {"bounds": {"x": [0.0, 2.0], "y": [0.0, 2.0]}, "walls": [], "obstacles": [[[0.5, 0.5], [1.0, 0.5], [1.0, 1.0], [0.5, 1.0]]]},
                  "surfaces": {}, "objects": {"obj": {"body": "obj_b"}}}  # fmt: skip
        self.world.occ = SimpleNamespace(body_radius=0.35)
        self.motor.pelvis = lambda: (np.array([1.8, 1.8, 0.79]), np.array([1.0, 0, 0, 0]))
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "meta.json"
            f.write_text(json.dumps(mapped))
            self.cfg = P.RenderConfig.of(dict(mapped=str(f)))
            v = self.view()
        self.assertEqual(v.meta["room"]["obstacles"], mapped["room"]["obstacles"])
        self.assertTrue(v.occ.occupied[15, 15])  # (0.75, 0.75): the perceived obstacle
        self.assertIs(v.nav.occ, v.occ)
        self.assertIsNot(v.meta, self.world.meta)  # the judge keeps the world's

    def test_a_hidden_object_keeps_its_last_estimate_and_a_moved_one_is_followed(self):
        v = self.view()
        p1, _ = v.object_pose("obj")
        self.true["pos"] = self.true["pos"] + [0.03, 0.0, 0.0]  # nudged
        self.visible = False
        self.motor.t, self.motor.k = 0.3, 15
        np.testing.assert_allclose(v.object_pose("obj")[0], p1)  # not seen: stale
        self.visible = True
        self.motor.t, self.motor.k = 0.6, 30
        self.assertLess(float(np.linalg.norm(v.object_pose("obj")[0] - self.true["pos"])), 0.005)  # tracked from p1

    def test_a_runaway_fit_is_rejected_and_the_last_estimate_stays(self):
        v = self.view()
        p1, _ = v.object_pose("obj")
        self.motor.t, self.motor.k = 0.3, 15
        with mock.patch.object(P, "fit_pose", return_value=(np.eye(3), np.array([90.0, 40.0, 1.0]), 0.001)):
            np.testing.assert_allclose(v.object_pose("obj")[0], p1)
        s = v.stats()["obj"]
        self.assertEqual((s["rejected"], s["log"][-1]["src"]), (1, "rejected"))

    def test_one_flat_patch_cannot_run_the_fit_off(self):
        model = P.surface_points(SHAPE, 3000, np.random.default_rng(3))
        g = np.random.default_rng(4).uniform(-0.03, 0.03, size=(300, 2))
        patch = np.c_[g, np.full(300, 0.05)] + [1.0, 0.0, 0.8]  # part of the top face only: slides freely in-plane
        R, t, _ = P.fit_pose(patch, model, cKDTree(model[0]), np.eye(3), np.array([1.0, 0.0, 0.8]), iters=40)
        self.assertLess(float(np.linalg.norm(t - [1.0, 0.0, 0.8])), 40 * P.STEP_M + 1e-9)

    def test_sam3_takes_the_instance_near_the_estimate(self):
        self.cfg = P.RenderConfig(masks="sam3", sam3_min_score=0.3, assoc_gate_m=0.25)

        def frames(data):
            (f,) = self.frames(data)
            f.rgb = np.zeros(f.depth.shape + (3,), np.uint8)
            f.depth[10:30, 10:40] = 2.5  # a second object of the same kind, far off
            return [f]

        class Sam3:
            def __init__(self, instances):
                self.instances, self.calls = instances, 0

            def call(self, req):
                self.calls += 1
                return dict(results={req["prompts"][0]: [(s, pack_mask(m)) for s, m in self.instances()]})

        with mock.patch.object(P.TruthView, "object_geoms", return_value=SHAPE):
            v = P.RenderedView(self.world, self.cfg, 610001, frames=frames)
            v._shape("obj")
        f0 = frames(None)[0]
        truth = f0.body == 7
        other = np.zeros_like(truth)
        other[10:30, 10:40] = True
        sam = Sam3(lambda: [(0.89, other), (0.84, truth)])  # the far one scores higher, as with the DVD remote
        v._sam3 = sam
        p, _ = v.object_pose("obj")
        self.assertLess(float(np.linalg.norm(p - self.true["pos"])), 0.005)
        row = v.stats()["obj"]["log"][-1]
        self.assertEqual((row["src"], row["instances"], row["wrong"], row["iou"]), ("seen", 2, 0, 1.0))
        v2 = P.RenderedView(self.world, self.cfg, 610002, frames=frames)
        v2._model = v._model
        v2._sam3 = Sam3(lambda: [(0.89, other)])  # only the other one is found: not seen
        p2, _ = v2.object_pose("obj")
        self.assertEqual(v2.stats()["obj"]["log"][-1]["src"], "prior")
        self.assertAlmostEqual(float(np.linalg.norm(p2 - self.true["pos"])), 0.05, places=9)

    def test_the_config_reads_a_spec(self):
        cfg = P.RenderConfig.of({"cameras": ["head_camera"], "period_s": 0.5})
        self.assertEqual((cfg.cameras, cfg.period_s), (("head_camera",), 0.5))
        with self.assertRaises(TypeError):
            P.RenderConfig.of({"fps": 5})
        with self.assertRaises(ValueError):
            P.RenderConfig.of({"masks": "grounding-dino"})
        with self.assertRaises(ValueError):
            P.RenderConfig.of({"depth": "unreal"})  # Unreal's depth goes with SAM 3's masks on Unreal's frames
        with self.assertRaises(ValueError):
            P.RenderConfig.of({"shape": "visual"})  # needs the scene that carries the visual meshes
        cfg = P.RenderConfig.of({"masks": "sam3", "depth": "unreal", "shape": "visual", "visual_scene": "s.xml"})
        self.assertEqual((cfg.depth, cfg.shape), ("unreal", "visual"))

    def test_unreal_depth_replaces_mujocos_and_keeps_its_masks(self):
        cfg = P.RenderConfig(masks="sam3", depth="unreal", width=4, height=3)
        body = np.zeros((3, 4), int)

        def mujoco_frames(data):
            return [P.Frame("head_camera", np.ones((3, 4)), body, (2.0, 2.0, 2.0, 1.5), np.zeros(3), np.eye(3))]

        class Unreal:
            def call(self, req):
                self.req = req
                return dict(images={"ego_view": np.zeros((3, 4, 3), np.uint8)},
                            depth={"ego_view": np.full((3, 4), 2.5, np.float32)})  # fmt: skip

        unreal = Unreal()
        (f,) = P.ServiceFrames(None, cfg, frames=mujoco_frames, unreal=unreal)(SimpleNamespace(qpos=np.zeros(2)))
        self.assertTrue(unreal.req["depth"])
        np.testing.assert_array_equal(f.depth, np.full((3, 4), 2.5))
        self.assertIs(f.body, body)  # the exact masks stay MuJoCo's: the IoU diagnostic only
        self.assertEqual(f.rgb.shape, (3, 4, 3))
        (g,) = P.ServiceFrames(None, P.RenderConfig(masks="sam3", width=4, height=3), frames=mujoco_frames,
                               unreal=unreal)(SimpleNamespace(qpos=np.zeros(2)))  # fmt: skip
        self.assertFalse(unreal.req["depth"])
        np.testing.assert_array_equal(g.depth, np.ones((3, 4)))

    def test_the_visual_shape_comes_from_the_visual_scene(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "visual.xml"
            path.write_text(VISUAL_SCENE)
            rows = P.visual_geoms(str(path), "obj")
            self.assertEqual([r["type"] for r in rows], ["box"])
            np.testing.assert_allclose(rows[0]["size"], [0.05, 0.03, 0.04])  # not the collision box
            self.cfg = P.RenderConfig(shape="visual", visual_scene=str(path))
            self.world.sim = SimpleNamespace(m=SimpleNamespace(body=lambda b: SimpleNamespace(name="obj")), d=None)
            v = P.RenderedView(self.world, self.cfg, 610001, frames=self.frames)
            pts = v._shape("obj")[0][0]
        self.assertAlmostEqual(float(np.abs(pts[:, 0]).max()), 0.05, places=6)  # samples of the 5 cm visual box


VISUAL_SCENE = """
<mujoco>
  <worldbody>
    <body name="obj" pos="1.0 0.2 0.05">
      <freejoint/>
      <geom name="vis" type="box" size="0.05 0.03 0.04" contype="0" conaffinity="0" group="1"/>
      <geom name="col" type="box" size="0.02 0.02 0.02" group="3"/>
    </body>
  </worldbody>
</mujoco>
"""


BOX_SCENE = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="3 3 0.1"/>
    <camera name="cam" pos="0.4 -0.9 1.3" mode="targetbody" target="obj"/>
    <body name="obj" pos="1.0 0.2 0.05" quat="0.9659 0 0 0.2588">
      <freejoint/>
      <geom name="big" type="box" size="0.04 0.03 0.05" group="3"/>
      <geom name="block" type="box" size="0.02 0.02 0.02" pos="0.06 0 -0.03" group="3"/>
    </body>
  </worldbody>
</mujoco>
"""


class Render(unittest.TestCase):
    """MuJoCoFrames on a box scene: needs a GL context (MUJOCO_GL=egl); skipped without one."""

    @classmethod
    def setUpClass(cls):
        import mujoco

        cls.m = mujoco.MjModel.from_xml_string(BOX_SCENE)
        cls.d = mujoco.MjData(cls.m)
        mujoco.mj_forward(cls.m, cls.d)
        try:
            cls.frames = P.MuJoCoFrames(cls.m, P.RenderConfig(cameras=("cam",), width=320, height=240))
        except Exception as e:  # no display / EGL
            raise unittest.SkipTest(f"no GL context: {type(e).__name__}: {e}")

    def test_masked_pixels_land_on_the_object_and_fit_its_pose(self):
        import mujoco

        (f,) = self.frames(self.d)
        b = self.m.body("obj").id
        self.assertGreater(int((f.body == b).sum()), 200)
        pts = P.back_project(f, f.body == b)
        R, t = self.d.xmat[b].reshape(3, 3), self.d.xpos[b]
        local = (pts - t) @ R
        big = np.max(np.abs(local) - [0.04, 0.03, 0.05], axis=1)
        block = np.max(np.abs(local - [0.06, 0, -0.03]) - 0.02, axis=1)
        self.assertLess(float(np.abs(np.minimum(np.maximum(big, -1), np.maximum(block, -1))).max()), 0.004)
        floor = P.back_project(f, f.body == 0)  # the plane: z = 0
        self.assertLess(float(np.abs(floor[:, 2]).max()), 0.004)
        geoms = [dict(type="box", size=self.m.geom_size[g], pos=self.m.geom_pos[g], quat=self.m.geom_quat[g])
                 for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == b]  # fmt: skip
        model = P.surface_points(geoms, 3000, np.random.default_rng(0))
        R_fit, t_fit, _ = P.fit_from_headings(pts, model, cKDTree(model[0]), R)
        self.assertLess(float(np.linalg.norm(t_fit - t)), 0.004)
        self.assertLess(np.degrees(P._angle(R_fit, R)), 3.0)
        del mujoco


if __name__ == "__main__":
    unittest.main()
