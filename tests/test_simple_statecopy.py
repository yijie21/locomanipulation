"""TODO 52 mechanism 2: the stored-state round trip (capture / restore of live objects), without a simulator."""

import copy
import unittest

import mujoco
import numpy as np

from locomani.bench.simple import statecopy as SC

XML = "<mujoco><worldbody><body><freejoint/><geom size='.1'/></body></worldbody></mujoco>"


class Leaf:
    def __init__(self):
        self.p, self.n = np.zeros(3), 0


class Thing:
    def __init__(self):
        self.leaf, self.xs, self.d, self.k = Leaf(), [1, 2], {"a": np.ones(2)}, 7
        self.fn = lambda: self.k  # a closure over the live object: must stay valid
        self.link = object()  # shared, never copied
        self.skipme = [9]


class RoundTrip(unittest.TestCase):
    def test_restore_puts_every_kind_back_in_place(self):
        t = Thing()
        leaf, xs, d, arr, link = t.leaf, t.xs, t.d, t.d["a"], t.link
        snap = SC.capture(t, skip=("skipme",), owned=(Leaf,))
        fp = SC.fingerprint(snap)
        t.leaf.p += 5
        t.leaf.n = 3
        t.xs.append(4)
        t.d["a"][:] = 9
        t.d["b"] = 1
        t.k = 8
        t.skipme.append(1)
        SC.restore(t, snap)
        self.assertEqual((t.k, t.xs, t.skipme), (7, [1, 2], [9, 1]))
        self.assertEqual(list(t.d), ["a"])
        np.testing.assert_array_equal(t.d["a"], np.ones(2))
        np.testing.assert_array_equal(t.leaf.p, np.zeros(3))
        self.assertEqual(t.leaf.n, 0)
        self.assertTrue(t.leaf is leaf and t.xs is xs and t.d is d and t.d["a"] is arr and t.link is link)
        self.assertEqual(t.fn(), 7)
        self.assertEqual(SC.fingerprint(SC.capture(t, skip=("skipme",), owned=(Leaf,))), fp)

    def test_a_replaced_function_is_put_back_as_the_same_object(self):
        t = Thing()
        t.ref = f0 = lambda x: x + 1
        snap = SC.capture(t, owned=(Leaf,))
        t.ref = lambda x: x + 2
        SC.restore(t, snap)
        self.assertIs(t.ref, f0)
        self.assertNotEqual(
            SC.fingerprint(SC.capture(t)), SC.fingerprint({**SC.capture(t), "ref": SC.pack(lambda: 0, (), [])})
        )

    def test_snapshot_is_detached_and_restorable_twice(self):
        t = Thing()
        snap = SC.capture(t, owned=(Leaf,))
        for _ in range(2):
            t.leaf.p[:] = 1
            t.xs.append(0)
            SC.restore(t, snap)
            np.testing.assert_array_equal(t.leaf.p, np.zeros(3))
            self.assertEqual(t.xs, [1, 2])

    def test_a_replaced_object_is_rebuilt_from_the_snapshot(self):
        t = Thing()
        snap = SC.capture(t, owned=(Leaf,))
        t.leaf = None
        SC.restore(t, snap)
        self.assertIsInstance(t.leaf, Leaf)

    def test_mjdata_all_fields_in_place(self):
        m = mujoco.MjModel.from_xml_string(XML)
        o = type("O", (), {})()
        o.m, o.d = m, mujoco.MjData(m)
        mujoco.mj_forward(m, o.d)
        for _ in range(20):
            mujoco.mj_step(m, o.d)
        d0 = o.d
        snap = SC.capture(o)
        want_x = o.d.xpos.copy()
        want = (o.d.qpos.copy(), o.d.qvel.copy(), o.d.qacc_warmstart.copy(), float(o.d.time))
        for _ in range(50):
            mujoco.mj_step(m, o.d)
        self.assertNotEqual(float(o.d.time), want[3])
        SC.restore(o, snap)
        self.assertIs(o.d, d0)
        np.testing.assert_array_equal(o.d.qpos, want[0])
        np.testing.assert_array_equal(o.d.qvel, want[1])
        np.testing.assert_array_equal(o.d.qacc_warmstart, want[2])
        np.testing.assert_array_equal(o.d.xpos, want_x)
        self.assertEqual(float(o.d.time), want[3])
        # continuing from the restored state gives the same trajectory as continuing the original
        ref = copy.copy(o.d)
        for _ in range(10):
            mujoco.mj_step(m, o.d)
            mujoco.mj_step(m, ref)
        np.testing.assert_array_equal(o.d.qpos, ref.qpos)


if __name__ == "__main__":
    unittest.main()
