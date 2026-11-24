"""TODO 49.2 step 6: the truth view (TruthView on a Privileged), SimplePrivileged against scripted server replies, the switch scope,
the look cache key, the session without the depth service, and the rgbd path of the client (no simulator, no GPU)."""

import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
from test_simple_session import Fixture

from locomani.bench import options
from locomani.bench.simple import body, client
from locomani.bench.simple import config as cfgmod
from locomani.bench.simple import run as bed
from locomani.bench.simple.config import DEFAULT_VIEW, SEED_DIR, SimpleConfig
from locomani.bench.simple.privileged import SimplePrivileged
from locomani.perception import head
from locomani.perception.truth import TruthView, rot_z


def box(c, h):
    c, h = np.asarray(c, float), np.asarray(h, float)
    return c + h * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


class FakePriv:
    """A box target on a table, a basket, and a hinged door with a leaf and a handle, in the simulator's frame."""

    def __init__(self):
        self.info = dict(
            start=dict(yaw=0.0),
            tables={"table": dict(top=0.7, pos=[1.0, 0.0, 0.35], size=[0.8, 1.2, 0.7], quat=[1, 0, 0, 0])},
            target="box", objects={"container": dict(body="basket")},
        )  # fmt: skip

    def episode(self):
        return dict(instruction="pick up the box", task="XMovePickTeleop", **self.info)

    def body_pose(self, name):
        T = np.eye(4)
        T[:3, 3] = [1.0, 0.1, 0.8]
        return T

    def body_vertices(self, name):
        return (
            box([1.0, 0.1, 0.8], [0.05, 0.05, 0.1]) if name == "target" else box([1.0, -0.4, 0.75], [0.15, 0.2, 0.05])
        )

    def pelvis(self):
        return np.array([0.0, 0.0, 0.8]), 0.0

    def art_geoms(self):
        return dict(
            obj=[box([1.0, 0.0, 1.0], [0.05, 0.5, 1.0]), box([1.0, 0.3, 1.0], [0.05, 0.02, 0.05])],
            moving={"leaf": [box([1.0, 0.0, 1.0], [0.05, 0.5, 1.0]), box([1.0, 0.3, 1.0], [0.05, 0.02, 0.05])]},
        )


def belief(tie=True):
    b = body.Belief.__new__(body.Belief)
    b.pt, b.yt = np.array([0.0, 0.0, 0.8]), 0.0
    b.odo, b.anchored = False, False
    b.sim = types.SimpleNamespace(d=types.SimpleNamespace(qpos=np.array([0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0])))
    b.pb, b.yb = (np.array([0.1, -0.2, 0.8]), 0.3) if tie else (b.pt.copy(), 0.0)
    return b


K = dict(art_pick="score")


class TruthViewMath(unittest.TestCase):
    def view(self, tie=True):
        self.bel = belief(tie)
        return TruthView(FakePriv(), self.bel, lambda *a, **k: ("grasp", a, k), "/lib", K, "XMovePickTeleop")

    def test_target_in_the_robot_frame_through_the_tie(self):
        v = self.view()
        t = v.target()
        want = self.bel.from_sim(np.array([1.0, 0.1, 0.8]))
        self.assertTrue(np.allclose(t["T_wo"][:3, 3], want))
        self.assertTrue(np.allclose(t["T_wo"][:3, :3], rot_z(0.3)))  # yb - yt
        self.assertAlmostEqual(t["top"], 0.7)
        self.assertTrue(np.allclose(t["hi"] - t["lo"][:3], t["hi"] - t["lo"]))
        self.assertAlmostEqual(float(t["hi"][2] - t["lo"][2]), 0.2)

    def test_without_a_tie_it_is_the_simulator_frame(self):
        v = self.view(tie=False)
        t = v.target()
        self.assertTrue(np.allclose(t["T_wo"][:3, 3], [1.0, 0.1, 0.8]))
        self.assertTrue(np.allclose(t["lo"], [0.95, 0.05, 0.7]) and np.allclose(t["hi"], [1.05, 0.15, 0.9]))

    def test_heading_edge_basket(self):
        v = self.view(tie=False)
        self.assertEqual(v.heading(), 0.0)
        self.assertAlmostEqual(v.edge(np.array([1.0, 0.0]), 0.0), 0.6)  # the table's near edge: x = 1.0 - 0.4
        self.assertIsNone(v.edge(np.array([1.0, 0.0]), 0.0, "table2"))
        centre, rim, near = v.basket(0.0)
        self.assertTrue(np.allclose(centre, [1.0, -0.4]))
        self.assertAlmostEqual(rim, 0.8)
        self.assertAlmostEqual(near, 0.85)
        c2, rim2 = v.basket_near(0.0)
        self.assertTrue(np.allclose(c2, centre) and rim2 == rim)
        self.assertFalse(v.search)
        self.assertEqual(v.look(), {})

    def test_heading_and_edge_follow_the_tie(self):
        v = self.view()
        self.assertAlmostEqual(v.heading(), 0.3)
        ahead = np.array([np.cos(0.3), np.sin(0.3)])
        corners = [self.bel.from_sim(np.array([x, y, 0.35])) for x in (0.6, 1.4) for y in (-0.6, 0.6)]
        self.assertAlmostEqual(v.edge(ahead, 0.0), min(c[:2] @ ahead for c in corners))

    def test_grasp_hands_the_skills_chooser_the_library(self):
        v = self.view(tie=False)
        import locomani.perception.truth as T

        T.load_library, old = (lambda d, task: dict(lib=(d, task))), T.load_library
        try:
            tgt = v.target()
            tag, args, kw = v.grasp(tgt, 0.2)
        finally:
            T.load_library = old
        self.assertEqual(tag, "grasp")
        self.assertEqual(args[0], dict(lib=("/lib", "XMovePickTeleop")))
        self.assertEqual(args[2:4], (0.2, tgt["top"]))

    def test_art_has_the_camera_views_structure_with_the_handle_as_the_part(self):
        v = self.view(tie=False)
        out = v.art()
        for k in ("pb", "yaw", "obj", "parts"):
            self.assertIn(k, out)
        self.assertEqual(out["obj"].shape[1], 3)
        self.assertEqual(len(out["parts"]), 1)  # the leaf (1.0 m wide) is no part, the handle (0.1 m) is
        p = out["parts"][0]
        self.assertTrue(set(p) >= {"c", "p", "lo", "hi", "n", "score", "phrase"})
        self.assertTrue(np.allclose(p["p"], [1.0, 0.3, 1.0]))
        self.assertEqual(p["n"], 8)
        self.assertEqual(p["score"], 1.0)

    def test_art_picks_follow_k_and_an_empty_asset_has_no_object(self):
        v = self.view(tie=False)
        v.K = dict(art_pick="high")
        v.priv.art_geoms = lambda: dict(obj=[box([1, 0, 1], [0.05, 0.05, 0.05]), box([1, 0, 1.4], [0.05, 0.05, 0.05])],
                                        moving={"a": [box([1, 0, 1], [0.05, 0.05, 0.05]), box([1, 0, 1.4], [0.05, 0.05, 0.05])]})  # fmt: skip
        self.assertAlmostEqual(float(v.art()["parts"][0]["p"][2]), 1.4)
        v.priv.art_geoms = lambda: dict(obj=[], moving={})
        out = v.art()
        self.assertIsNone(out["obj"])
        self.assertEqual(out["parts"], [])


class DeeperPartFirst(unittest.TestCase):
    def test_a_knob_on_a_hinged_leaf_outranks_the_leaf_geoms(self):
        bel = belief(False)
        priv = FakePriv()
        priv.art_geoms = lambda: dict(
            obj=[box([1, 0, 1], [0.05, 0.05, 0.05])],
            moving={"leaf": [box([1, 0, 1.5], [0.05, 0.05, 0.05])], "knob": [box([1, 0, 1.0], [0.02, 0.02, 0.02])]},
            depth={"leaf": 0, "knob": 1},
        )
        v = TruthView(priv, bel, None, "/lib", dict(art_pick="score"), "CloseDoorTeleop")
        parts = v.art()["parts"]
        self.assertEqual([p["phrase"] for p in parts], ["knob", "leaf"])
        self.assertEqual([p["score"] for p in parts], [2.0, 1.0])


class FakeLink:
    def __init__(self, replies):
        self.replies, self.sent = replies, []

    def send(self, msg):
        self.sent.append(msg)
        self._last = msg

    def recv(self):
        r = self.replies[self._last[0]]
        return r(self._last) if callable(r) else r


class PrivilegedReplies(unittest.TestCase):
    def test_poses_vertices_and_the_ground_offset(self):
        link = FakeLink(dict(
            truth=dict(objects=dict(target=[1.0, 2.0, 0.5, 1, 0, 0, 0])),
            verts=lambda m: {m[1][0]: [box([0, 0, 0], [1, 1, 1]), box([0, 0, 3], [1, 1, 1])]},
            art_verts=dict(obj=[box([0, 0, 0], [1, 1, 1])], moving={"leaf": [box([0, 0, 0], [1, 1, 1])]}, depth={"leaf": 0}),
        ))  # fmt: skip
        d = types.SimpleNamespace(qpos=np.array([0.5, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]))
        sim = types.SimpleNamespace(link=link, zoff=-0.1, d=d)
        info = dict(target="box", objects={"container": {}}, tables={}, start={})
        p = SimplePrivileged(sim, info, "XMovePickTeleop", "pick it")
        T = p.body_pose("target")
        self.assertTrue(np.allclose(T[:3, 3], [1.0, 2.0, 0.4]) and np.allclose(T[:3, :3], np.eye(3)))
        self.assertEqual(p.body_vertices("target").shape, (16, 3))
        self.assertAlmostEqual(p.body_vertices("target")[:, 2].min(), -1.1)
        self.assertEqual(p.pelvis()[1], 0.0)
        self.assertEqual(p.episode()["instruction"], "pick it")
        a = p.art_geoms()
        self.assertEqual(len(a["obj"]), 1)
        self.assertEqual(a["depth"], {"leaf": 0})
        self.assertAlmostEqual(float(a["moving"]["leaf"][0][:, 2].min()), -1.1)
        self.assertEqual([m[0] for m in link.sent], ["truth", "verts", "verts", "art_verts"])


class Scope(unittest.TestCase):
    def opts(self, *kv):
        return options.resolve("simple", list(kv), allow_planned=True).options

    def test_accepted(self):
        for kv in ([], ["target=truth"], ["sensor=rgbd"], ["sensor=rgbd", "depth_noise=realsense"]):
            cfgmod.check_scope(self.opts(*kv))

    def test_both_pose_and_root_switches_take_both_values(self):
        for pose in ("odometry", "truth"):
            for root in ("odometry", "truth"):
                cfgmod.check_scope(self.opts(f"self_pose={pose}", f"controller_root={root}"))

    def test_refused(self):
        import dataclasses

        with self.assertRaises(cfgmod.ScopeError):  # (the capability table refuses native before the bed does)
            cfgmod.check_scope(dataclasses.replace(self.opts(), controller="native"))


class CacheKey(unittest.TestCase):
    def test_pose_and_root_switches_enter_the_key_only_when_not_default(self):
        pkg = bed.load_package(SEED_DIR)
        k = bed.cache_key(pkg, "XMovePickTeleop", {})
        self.assertEqual(k, bed.cache_key(pkg, "XMovePickTeleop", {}, DEFAULT_VIEW, ("odometry", "odometry")))
        others = {
            bed.cache_key(pkg, "XMovePickTeleop", {}, DEFAULT_VIEW, c)
            for c in (("truth", "odometry"), ("odometry", "truth"), ("truth", "truth"))
        }
        self.assertEqual(len(others), 3)
        self.assertNotIn(k, others)

    def test_default_key_is_the_old_recipe_and_every_other_switch_set_differs(self):
        pkg = bed.load_package(SEED_DIR)
        S = pkg.skill
        K = bed.resolved(S, "XMovePickTeleop", {})
        h = hashlib.sha1((Path(pkg.path) / "camera_view.py").read_bytes())
        keys = tuple(bed.LOOK_KEYS) + tuple(S.MOTOR_KEYS)
        h.update(json.dumps({k: K.get(k) for k in sorted(keys)}, sort_keys=True, default=str).encode())
        old = h.hexdigest()[:10]
        self.assertEqual(bed.cache_key(pkg, "XMovePickTeleop", {}), old)
        self.assertEqual(bed.cache_key(pkg, "XMovePickTeleop", {}, DEFAULT_VIEW), old)
        others = {bed.cache_key(pkg, "XMovePickTeleop", {}, s) for s in (
            ("perception", "rgbd", "none"), ("perception", "rgbd", "realsense"), ("truth", "none", "none"))}  # fmt: skip
        self.assertEqual(len(others), 3)
        self.assertNotIn(old, others)

    def test_bed_reports_the_view_and_depth_source(self):
        pkg = bed.load_package(SEED_DIR)
        cfg = SimpleConfig()
        for kv, src in (
            ([], "stereo"),
            (["sensor=rgbd"], "rgbd"),
            (["sensor=rgbd", "depth_noise=realsense"], "rgbd:realsense"),
            (["target=truth"], "none"),
        ):
            o = options.resolve("simple", kv, allow_planned=True).options
            b = bed.Bed(cfg, pkg, 1, {}, o)
            self.assertEqual(b.depth_source, src)
        self.assertEqual(bed.Bed(cfg, pkg).depth_source, "stereo")  # no options: the loops' defaults


class DepthService(Fixture):
    def test_rgbd_never_starts_the_depth_service(self):
        s = self.session(stereo=False)
        self.assertEqual(s.run_round(), 0)
        self.assertFalse((Path(self.cfg.run_dir) / "depth_XMovePickTeleop.pid").exists())
        self.assertFalse((Path(self.cfg.run_dir) / "depth_q_XMovePickTeleop.log").exists())
        self.assertTrue((Path(self.cfg.run_dir) / "server_XMovePickTeleop.pid").exists())
        self.assertEqual(self.calls(), ["cachedir", "look", "graspgen", "batch"])

    def test_truth_has_no_look_pass_and_no_depth(self):
        s = self.session(stereo=False, look=False)
        self.assertEqual(s.run_round(), 0)
        self.assertEqual(self.calls(), ["cachedir", "batch"])
        self.assertFalse((Path(self.cfg.run_dir) / "depth_XMovePickTeleop.pid").exists())

    def test_default_session_still_starts_it(self):
        self.assertEqual(self.session().run_round(), 0)
        self.assertTrue((Path(self.cfg.run_dir) / "depth_XMovePickTeleop.pid").exists())


class NaNDepth(unittest.TestCase):
    def test_cam_points_treat_holes_like_zero_depth(self):
        d = np.full((360, 640), 1.5, np.float32)
        ref = head.cam_points(d)
        d2 = d.copy()
        d2[10, 20], d2[30, 40] = np.nan, np.inf
        pts = head.cam_points(d2)
        self.assertTrue(np.isfinite(pts).all())
        self.assertTrue(np.allclose(pts[0, 0], ref[0, 0]))
        self.assertTrue(np.allclose(pts[10, 20], 0.0) and np.allclose(pts[30, 40], 0.0))
        self.assertTrue(np.array_equal(head.cam_points(np.zeros((4, 4))), head.cam_points(np.full((4, 4), np.nan))))

    def test_valid_mask_ignores_holes(self):
        d = np.array([[np.nan, 0.5, 7.0, 1.0]])
        self.assertEqual(((d > 0.2) & (d < 5)).tolist(), [[False, True, False, True]])

    def test_intrinsics_match_the_projection(self):
        K = head.intrinsics()
        d = np.full((360, 640), 2.0)
        p = head.cam_points(d)[100, 500]
        u, v = K[0, 0] * p[0] / 2.0 + K[0, 2], -K[1, 1] * p[1] / 2.0 + K[1, 2]
        self.assertAlmostEqual(u, 500.0, 4)
        self.assertAlmostEqual(v, 100.0, 4)


class RgbdClient(unittest.TestCase):
    def sim(self):
        rng = np.random.default_rng(1)
        dl = np.where(rng.random((360, 640)) < 0.01, np.inf, 1.2 + 0.3 * rng.random((360, 640))).astype(np.float32)
        dr = dl + 0.01
        rgb = rng.integers(0, 255, (360, 640, 3), dtype=np.uint8)
        link = FakeLink(dict(rgbd_setup=True, eyes_d=dict(left=rgb, right=rgb, depth_left=dl, depth_right=dr)))
        s = client.OffSim.__new__(client.OffSim)
        s.link, s.depth_link, s.steps = link, None, 5
        s.m, s.qadr, s.d = None, [0], types.SimpleNamespace(qpos=np.zeros(3))
        s._kin = lambda joints: (np.zeros(3), np.eye(3))
        return s, rgb, dl

    def test_exact_depth_comes_back_with_nan_holes_and_the_left_image(self):
        s, rgb, dl = self.sim()
        s.use_rgbd("none", 7)
        img, depth = s.eyes()
        self.assertTrue(np.array_equal(img, rgb))
        self.assertTrue(np.isnan(depth[~np.isfinite(dl)]).all())
        self.assertTrue(np.array_equal(depth[np.isfinite(dl)], dl[np.isfinite(dl)]))
        self.assertTrue(np.array_equal(s.last_eyes["lb"], depth, equal_nan=True))
        self.assertEqual([m[0] for m in s.link.sent], ["rgbd_setup", "eyes_d"])

    def test_realsense_is_seeded_by_the_episode_and_has_holes(self):
        a, _, _ = self.sim()
        a.use_rgbd("realsense", 7)
        b, _, _ = self.sim()
        b.use_rgbd("realsense", 7)
        c, _, _ = self.sim()
        c.use_rgbd("realsense", 8)
        da, db, dc = a.eyes()[1], b.eyes()[1], c.eyes()[1]
        self.assertTrue(np.array_equal(da, db, equal_nan=True))
        self.assertFalse(np.array_equal(da, dc, equal_nan=True))
        self.assertGreater(np.isnan(da).mean(), 0.0)
        s, _, dl = self.sim()
        s.use_rgbd("none", 7)
        self.assertFalse(np.array_equal(da, s.eyes()[1], equal_nan=True))

    def test_the_default_sim_sends_no_new_message(self):
        s = client.OffSim.__new__(client.OffSim)
        self.assertIsNone(s.rgbd)  # stereo is the default path: eyes() is unchanged and no rgbd_setup is sent


class ServerMessages(unittest.TestCase):
    def test_new_messages_only_behind_their_switches(self):
        src = (Path(client.__file__).parent / "run.py").read_text()
        self.assertIn('bed.switch("sensor") == "rgbd"', src)
        self.assertIn("use_rgbd", src)
        server = (Path(client.__file__).parent / "server.py").read_text()
        for m in ("rgbd_setup", "eyes_d", "verts", "art_verts"):
            self.assertIn(f'k == "{m}"', server)
        self.assertIn("add_distance_to_image_plane_to_frame", server)


if __name__ == "__main__":
    unittest.main()
